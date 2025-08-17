"""
Production-ready model serving infrastructure for Vercel
Maintains full model quality with proper storage and versioning
"""

import os
import sys
import json
import pickle
import base64
import hashlib
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import aiohttp

# Unified storage and monitoring
from ..storage import get_storage, get_sync_storage
from ..monitoring import get_monitor

logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a specific version of a model"""
    
    def __init__(self, model_id: str, version: str, metadata: Dict[str, Any]):
        self.model_id = model_id
        self.version = version
        self.metadata = metadata
        self.created_at = metadata.get('created_at', datetime.now().isoformat())
        self.metrics = metadata.get('metrics', {})
        self.status = metadata.get('status', 'active')
        self.model_data = None
        self.loaded = False
        
    def get_score(self) -> float:
        """Get model quality score for comparison"""
        # Prioritize R² score, then inverse MAPE
        r2 = self.metrics.get('r2', 0)
        mape = self.metrics.get('mape', 100)
        return r2 * 100 - mape / 10


class ModelRegistry:
    """Production model registry with versioning and storage"""
    
    def __init__(self):
        self.storage = get_sync_storage()
        self.monitor = get_monitor("ml-model-server")
        self.local_cache_dir = Path("/tmp/ml_models")
        self.local_cache_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.monitor.log_info("ModelRegistry initialized", {
            "storage_type": type(self.storage).__name__,
            "cache_dir": str(self.local_cache_dir)
        })
        
    async def register_model(self, model_id: str, model_data: bytes, 
                           metadata: Dict[str, Any]) -> ModelVersion:
        """Register a new model version"""
        try:
            # Generate version hash
            version = hashlib.sha256(model_data).hexdigest()[:12]
            
            # Create model version
            model_version = ModelVersion(model_id, version, metadata)
            
            # Store model data in Vercel Blob (or local for development)
            blob_key = f"models/{model_id}/{version}.pkl"
            
            if os.environ.get('VERCEL_ENV') == 'production':
                # Production: Use Vercel Blob
                await self.blob_storage.upload(blob_key, model_data)
            else:
                # Local: Save to disk
                local_path = self.local_cache_dir / f"{model_id}_{version}.pkl"
                with open(local_path, 'wb') as f:
                    f.write(model_data)
            
            # Store metadata in KV
            metadata_key = f"model_metadata:{model_id}:{version}"
            await self.cache.kv.set(metadata_key, json.dumps(metadata), ex=None)  # No expiration
            
            # Update model index
            await self._update_model_index(model_id, version)
            
            logger.info(f"Registered model {model_id} version {version}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    async def get_model(self, model_id: str, version: Optional[str] = None) -> Any:
        """Get a model by ID and optional version"""
        try:
            # If no version specified, get best version
            if not version:
                version = await self._get_best_version(model_id)
                if not version:
                    raise ValueError(f"No model found for {model_id}")
            
            # Check loaded models cache
            cache_key = f"{model_id}:{version}"
            if cache_key in self.loaded_models:
                return self.loaded_models[cache_key]
            
            # Load model data
            model_data = await self._load_model_data(model_id, version)
            if not model_data:
                raise ValueError(f"Model data not found for {model_id}:{version}")
            
            # Deserialize model
            model = pickle.loads(model_data)
            
            # Cache in memory (with size limit)
            if len(self.loaded_models) < 10:  # Keep max 10 models in memory
                self.loaded_models[cache_key] = model
            else:
                # Evict oldest model
                oldest_key = next(iter(self.loaded_models))
                del self.loaded_models[oldest_key]
                self.loaded_models[cache_key] = model
            
            return model
            
        except Exception as e:
            logger.error(f"Error getting model: {e}")
            raise
    
    async def _load_model_data(self, model_id: str, version: str) -> Optional[bytes]:
        """Load model data from storage"""
        try:
            blob_key = f"models/{model_id}/{version}.pkl"
            
            if os.environ.get('VERCEL_ENV') == 'production':
                # Production: Load from Vercel Blob
                return await self.blob_storage.download(blob_key)
            else:
                # Local: Load from disk
                local_path = self.local_cache_dir / f"{model_id}_{version}.pkl"
                if local_path.exists():
                    with open(local_path, 'rb') as f:
                        return f.read()
                        
        except Exception as e:
            logger.error(f"Error loading model data: {e}")
            
        return None
    
    async def _get_best_version(self, model_id: str) -> Optional[str]:
        """Get the best performing version of a model"""
        try:
            # Get model index
            index_key = f"model_index:{model_id}"
            index_data = await self.cache.kv.get(index_key)
            
            if not index_data:
                return None
            
            versions = json.loads(index_data).get('versions', [])
            
            if not versions:
                return None
            
            # Load metadata for each version and score them
            best_version = None
            best_score = -float('inf')
            
            for version in versions:
                metadata_key = f"model_metadata:{model_id}:{version}"
                metadata_str = await self.cache.kv.get(metadata_key)
                
                if metadata_str:
                    metadata = json.loads(metadata_str)
                    model_version = ModelVersion(model_id, version, metadata)
                    
                    # Only consider active models
                    if model_version.status == 'active':
                        score = model_version.get_score()
                        if score > best_score:
                            best_score = score
                            best_version = version
            
            return best_version
            
        except Exception as e:
            logger.error(f"Error getting best version: {e}")
            return None
    
    async def _update_model_index(self, model_id: str, version: str):
        """Update the model version index"""
        try:
            index_key = f"model_index:{model_id}"
            index_data = await self.cache.kv.get(index_key)
            
            if index_data:
                index = json.loads(index_data)
            else:
                index = {'model_id': model_id, 'versions': []}
            
            if version not in index['versions']:
                index['versions'].append(version)
                index['updated_at'] = datetime.now().isoformat()
                
            await self.cache.kv.set(index_key, json.dumps(index), ex=None)
            
        except Exception as e:
            logger.error(f"Error updating model index: {e}")


class ModelServingAPI:
    """Production-grade model serving API"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.storage = get_sync_storage()
        self.monitor = get_monitor("ml-serving-api")
        self.prediction_queue = asyncio.Queue(maxsize=100)
        self.performance_tracker = ModelPerformanceTracker()
        self.is_processing = False
        
    async def predict(self, model_id: str, features: Dict[str, Any], 
                     request_id: Optional[str] = None) -> Dict[str, Any]:
        """Make a prediction with full model quality"""
        try:
            # Generate request ID if not provided
            if not request_id:
                request_id = hashlib.md5(f"{model_id}{datetime.now()}".encode()).hexdigest()[:12]
            
            # Check cache first
            cache_key = f"prediction:{model_id}:{hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()[:8]}"
            cached_result = await self.cache.get_prediction(cache_key)
            
            if cached_result:
                cached_result['cached'] = True
                cached_result['request_id'] = request_id
                return cached_result
            
            # Add to prediction queue
            request = {
                'request_id': request_id,
                'model_id': model_id,
                'features': features,
                'timestamp': datetime.now().isoformat(),
                'future': asyncio.Future()
            }
            
            await self.prediction_queue.put(request)
            
            # Start processing if not already running
            if not self.is_processing:
                asyncio.create_task(self._process_predictions())
            
            # Wait for result (with timeout)
            try:
                result = await asyncio.wait_for(request['future'], timeout=30.0)
                
                # Cache successful result
                if 'error' not in result:
                    await self.cache.set_prediction(cache_key, result, ttl=300)
                
                # Track performance
                await self.performance_tracker.track_prediction(model_id, result)
                
                return result
                
            except asyncio.TimeoutError:
                return {
                    'request_id': request_id,
                    'error': 'Prediction timeout',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'request_id': request_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _process_predictions(self):
        """Process predictions from queue"""
        self.is_processing = True
        
        try:
            while not self.prediction_queue.empty():
                request = await self.prediction_queue.get()
                
                try:
                    # Load model
                    model = await self.registry.get_model(request['model_id'])
                    
                    # Make prediction
                    start_time = datetime.now()
                    
                    # Convert features to DataFrame if needed
                    if hasattr(model, 'predict'):
                        if isinstance(request['features'], dict):
                            # For single prediction
                            features_df = pd.DataFrame([request['features']])
                        else:
                            features_df = pd.DataFrame(request['features'])
                        
                        prediction = model.predict(features_df)
                        
                        # Format result
                        result = {
                            'request_id': request['request_id'],
                            'model_id': request['model_id'],
                            'prediction': prediction,
                            'timestamp': datetime.now().isoformat(),
                            'latency_ms': (datetime.now() - start_time).total_seconds() * 1000,
                            'model_version': getattr(model, 'version', 'unknown')
                        }
                    else:
                        raise ValueError("Model does not have predict method")
                    
                    request['future'].set_result(result)
                    
                except Exception as e:
                    logger.error(f"Error processing prediction: {e}")
                    request['future'].set_result({
                        'request_id': request['request_id'],
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    
        finally:
            self.is_processing = False


class ModelPerformanceTracker:
    """Track model performance in production"""
    
    def __init__(self):
        self.storage = get_sync_storage()
        self.monitor = get_monitor("ml-performance-tracker")
        self.metrics_window = 3600  # 1 hour window
        
    async def track_prediction(self, model_id: str, prediction_result: Dict[str, Any]):
        """Track a prediction result"""
        try:
            # Skip if error
            if 'error' in prediction_result:
                await self._track_error(model_id, prediction_result['error'])
                return
            
            # Track latency
            latency = prediction_result.get('latency_ms', 0)
            await self._track_metric(f"latency:{model_id}", latency)
            
            # Track prediction count
            await self._increment_counter(f"predictions:{model_id}")
            
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
    
    async def track_actual_outcome(self, model_id: str, prediction_id: str, 
                                  predicted_value: float, actual_value: float):
        """Track actual outcome vs prediction for accuracy"""
        try:
            # Calculate error
            error = abs(actual_value - predicted_value)
            error_pct = error / actual_value * 100 if actual_value != 0 else 0
            
            # Track accuracy metrics
            await self._track_metric(f"error:{model_id}", error)
            await self._track_metric(f"error_pct:{model_id}", error_pct)
            
            # Track direction accuracy
            # This would need previous values to determine direction
            
        except Exception as e:
            logger.error(f"Error tracking outcome: {e}")
    
    async def get_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get performance summary for a model"""
        try:
            # Get metrics from cache
            predictions = await self._get_counter(f"predictions:{model_id}")
            errors = await self._get_counter(f"errors:{model_id}")
            avg_latency = await self._get_average(f"latency:{model_id}")
            avg_error_pct = await self._get_average(f"error_pct:{model_id}")
            
            return {
                'model_id': model_id,
                'predictions_total': predictions,
                'errors_total': errors,
                'error_rate': errors / predictions if predictions > 0 else 0,
                'avg_latency_ms': avg_latency,
                'avg_error_pct': avg_error_pct,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def _track_metric(self, key: str, value: float):
        """Track a metric value"""
        timestamp = int(datetime.now().timestamp())
        metric_key = f"metric:{key}:{timestamp // 60}"  # 1-minute buckets
        
        # Get current bucket
        current = await self.cache.kv.get(metric_key)
        if current:
            data = json.loads(current)
            data['values'].append(value)
            data['count'] += 1
        else:
            data = {'values': [value], 'count': 1}
        
        await self.cache.kv.set(metric_key, json.dumps(data), ex=self.metrics_window)
    
    async def _increment_counter(self, key: str):
        """Increment a counter"""
        counter_key = f"counter:{key}"
        current = await self.cache.kv.get(counter_key)
        count = int(current) if current else 0
        await self.cache.kv.set(counter_key, str(count + 1), ex=86400)  # 24 hours
    
    async def _track_error(self, model_id: str, error: str):
        """Track an error"""
        await self._increment_counter(f"errors:{model_id}")
        
        # Log error details
        error_key = f"errors:{model_id}:{datetime.now().strftime('%Y%m%d')}"
        await self.cache.kv.set(error_key, error, ex=86400)
    
    async def _get_counter(self, key: str) -> int:
        """Get counter value"""
        counter_key = f"counter:{key}"
        value = await self.cache.kv.get(counter_key)
        return int(value) if value else 0
    
    async def _get_average(self, key: str) -> float:
        """Get average of recent metrics"""
        # Get recent buckets
        total_sum = 0
        total_count = 0
        
        current_minute = int(datetime.now().timestamp()) // 60
        
        for i in range(60):  # Last 60 minutes
            minute = current_minute - i
            metric_key = f"metric:{key}:{minute}"
            
            data_str = await self.cache.kv.get(metric_key)
            if data_str:
                data = json.loads(data_str)
                values = data['values']
                if values:
                    total_sum += sum(values)
                    total_count += len(values)
        
        return total_sum / total_count if total_count > 0 else 0


# Global instance
model_serving_api = ModelServingAPI()