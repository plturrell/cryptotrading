"""
Enhanced ML Service - Multiple Model Types and Automated Retraining
Supports LSTM, Transformer, XGBoost, and Ensemble models
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from ..infrastructure.monitoring import get_logger, get_business_metrics, trace_context
from ..core.ml.inference import inference_service, PredictionRequest, BatchPredictionRequest

logger = get_logger("services.ml")


class ModelType:
    """Enhanced model types"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"
    PROPHET = "prophet"
    ARIMA = "arima"


class MLModelRegistry:
    """Model registry for version management"""
    
    def __init__(self):
        self.models_path = Path("data/models")
        self.models_path.mkdir(exist_ok=True)
        self.model_metadata = {}
    
    def register_model(self, model_name: str, model_version: str, 
                      model_type: str, performance_metrics: Dict[str, float]):
        """Register a new model version"""
        model_key = f"{model_name}_v{model_version}"
        self.model_metadata[model_key] = {
            "name": model_name,
            "version": model_version,
            "type": model_type,
            "performance": performance_metrics,
            "created_at": datetime.utcnow().isoformat(),
            "path": str(self.models_path / f"{model_key}.pkl")
        }
        logger.info(f"Registered model {model_key} with performance: {performance_metrics}")
    
    def get_best_model(self, model_name: str, metric: str = "accuracy") -> Optional[Dict[str, Any]]:
        """Get the best performing model version"""
        matching_models = [
            (key, meta) for key, meta in self.model_metadata.items()
            if meta["name"] == model_name
        ]
        
        if not matching_models:
            return None
        
        def get_performance_metric(model_tuple):
            return model_tuple[1]["performance"].get(metric, 0)
        
        best_model = max(matching_models, key=get_performance_metric)
        return best_model[1]


class AutoMLRetrainer:
    """Automated model retraining system"""
    
    def __init__(self, model_registry: MLModelRegistry):
        self.registry = model_registry
        self.retraining_schedule = {}
        self.performance_threshold = 0.85
    
    async def schedule_retraining(self, symbol: str, model_type: str, 
                                schedule: str = "daily"):
        """Schedule model retraining"""
        key = f"{symbol}_{model_type}"
        self.retraining_schedule[key] = {
            "symbol": symbol,
            "model_type": model_type,
            "schedule": schedule,
            "last_retrain": datetime.utcnow(),
            "next_retrain": self._calculate_next_retrain(schedule)
        }
        logger.info(f"Scheduled retraining for {key} on {schedule} basis")
    
    def _calculate_next_retrain(self, schedule: str) -> datetime:
        """Calculate next retraining time"""
        now = datetime.utcnow()
        if schedule == "daily":
            return now + timedelta(days=1)
        elif schedule == "weekly":
            return now + timedelta(weeks=1)
        elif schedule == "monthly":
            return now + timedelta(days=30)
        else:
            return now + timedelta(hours=6)  # Default 6 hours
    
    async def check_and_retrain(self) -> List[str]:
        """Check if any models need retraining"""
        retrained_models = []
        
        for key, schedule_info in self.retraining_schedule.items():
            if datetime.utcnow() >= schedule_info["next_retrain"]:
                try:
                    await self._retrain_model(
                        schedule_info["symbol"],
                        schedule_info["model_type"]
                    )
                    retrained_models.append(key)
                    
                    # Update schedule
                    schedule_info["last_retrain"] = datetime.utcnow()
                    schedule_info["next_retrain"] = self._calculate_next_retrain(
                        schedule_info["schedule"]
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to retrain {key}: {e}")
        
        return retrained_models
    
    async def _retrain_model(self, symbol: str, model_type: str):
        """Retrain a specific model"""
        logger.info(f"Retraining {model_type} model for {symbol}")
        
        # Import training components
        from ..core.ml.training import training_pipeline
        
        # Trigger retraining
        await training_pipeline.train_specific_model(symbol, model_type)


class EnhancedMLService:
    """Enhanced ML service with multiple model types"""
    
    def __init__(self):
        self.business_metrics = get_business_metrics()
        self.model_registry = MLModelRegistry()
        self.auto_retrainer = AutoMLRetrainer(self.model_registry)
        self.ensemble_weights = {}
    
    async def get_prediction(self, symbol: str, horizon: str = "24h", 
                           model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get ML prediction with enhanced model selection"""
        start_time = time.time()
        
        with trace_context(f"ml_prediction_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("horizon", horizon)
                span.set_attribute("service", "ml_prediction")
                
                # If no model type specified, use ensemble
                if model_type is None:
                    return await self._get_ensemble_prediction(symbol, horizon)
                
                # Single model prediction
                req = PredictionRequest(
                    symbol=symbol.upper(),
                    horizon=horizon,
                    model_type=model_type,
                    include_confidence=True,
                    include_features=True
                )
                
                result = await inference_service.get_prediction(req)
                
                if result:
                    duration_ms = (time.time() - start_time) * 1000
                    self.business_metrics.track_ai_operation(
                        operation="ml_prediction",
                        model=model_type,
                        symbol=symbol,
                        success=True,
                        duration_ms=duration_ms
                    )
                    
                    span.set_attribute("success", "true")
                    span.set_attribute("model_type", model_type)
                    span.set_attribute("confidence", result.confidence)
                    
                    return {
                        'predicted_price': result.predicted_price,
                        'current_price': result.current_price,
                        'price_change_percent': result.price_change_percent,
                        'confidence': result.confidence,
                        'model_type': result.model_type,
                        'horizon': result.horizon,
                        'features_used': result.features_used,
                        'prediction_timestamp': datetime.utcnow().isoformat()
                    }
                
                raise ValueError(f"No prediction available for {symbol}")
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="ml_prediction",
                    model=model_type or "ensemble",
                    symbol=symbol,
                    success=False,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"ML prediction failed for {symbol}: {e}")
                raise
    
    async def _get_ensemble_prediction(self, symbol: str, horizon: str) -> Dict[str, Any]:
        """Get ensemble prediction from multiple models"""
        model_types = [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.XGBOOST, ModelType.LIGHTGBM]
        
        predictions = []
        confidences = []
        
        for model_type in model_types:
            try:
                req = PredictionRequest(
                    symbol=symbol.upper(),
                    horizon=horizon,
                    model_type=model_type,
                    include_confidence=True
                )
                
                result = await inference_service.get_prediction(req)
                if result:
                    predictions.append(result.predicted_price)
                    confidences.append(result.confidence)
                    
            except Exception as e:
                logger.warning(f"Model {model_type} failed for {symbol}: {e}")
                continue
        
        if not predictions:
            raise ValueError(f"No models available for {symbol}")
        
        # Weighted ensemble based on confidence
        weights = np.array(confidences) / sum(confidences)
        ensemble_price = np.average(predictions, weights=weights)
        ensemble_confidence = np.mean(confidences)
        
        # Calculate price change
        current_price = predictions[0]  # Use first available as reference
        price_change_percent = ((ensemble_price - current_price) / current_price) * 100
        
        return {
            'predicted_price': float(ensemble_price),
            'current_price': float(current_price),
            'price_change_percent': float(price_change_percent),
            'confidence': float(ensemble_confidence),
            'model_type': 'ensemble',
            'horizon': horizon,
            'individual_predictions': [
                {'model': model_types[i], 'price': float(predictions[i]), 'confidence': float(confidences[i])}
                for i in range(len(predictions))
            ],
            'ensemble_weights': weights.tolist(),
            'prediction_timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_batch_predictions(self, symbols: List[str], horizon: str = "24h",
                                  model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get batch predictions for multiple symbols"""
        start_time = time.time()
        
        with trace_context("ml_batch_prediction") as span:
            try:
                span.set_attribute("symbols_count", len(symbols))
                span.set_attribute("horizon", horizon)
                
                if model_type is None:
                    # Use ensemble for all
                    tasks = [
                        self._get_ensemble_prediction(symbol, horizon) 
                        for symbol in symbols
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Use specific model
                    req = BatchPredictionRequest(
                        symbols=[s.upper() for s in symbols],
                        horizon=horizon,
                        model_type=model_type
                    )
                    batch_results = await inference_service.get_batch_predictions(req)
                    results = [r.dict() for r in batch_results]
                
                # Filter out exceptions
                valid_results = [r for r in results if not isinstance(r, Exception)]
                
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="ml_batch_prediction",
                    model=model_type or "ensemble",
                    symbol=",".join(symbols),
                    success=True,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("success", "true")
                span.set_attribute("results_count", len(valid_results))
                
                return valid_results
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="ml_batch_prediction",
                    model=model_type or "ensemble",
                    symbol=",".join(symbols),
                    success=False,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"Batch prediction failed: {e}")
                raise
    
    async def get_model_performance(self, symbol: str, horizon: str = "24h") -> Dict[str, Any]:
        """Get model performance metrics"""
        start_time = time.time()
        
        with trace_context(f"ml_performance_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("horizon", horizon)
                
                result = await inference_service.get_model_performance(symbol.upper(), horizon)
                
                duration_ms = (time.time() - start_time) * 1000
                
                span.set_attribute("success", "true")
                
                return result.dict()
                
            except Exception as e:
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"Performance retrieval failed for {symbol}: {e}")
                raise
    
    async def trigger_training(self, symbols: Optional[List[str]] = None,
                             model_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Trigger model training"""
        start_time = time.time()
        
        with trace_context("ml_training") as span:
            try:
                from ..core.ml.training import training_pipeline
                
                # Start training in background
                import threading
                
                def train_models():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    if symbols and model_types:
                        # Train specific models
                        for symbol in symbols:
                            for model_type in model_types:
                                loop.run_until_complete(
                                    training_pipeline.train_specific_model(symbol, model_type)
                                )
                    else:
                        # Train all models
                        loop.run_until_complete(training_pipeline.train_all_models())
                
                thread = threading.Thread(target=train_models)
                thread.start()
                
                duration_ms = (time.time() - start_time) * 1000
                
                span.set_attribute("success", "true")
                span.set_attribute("training_triggered", "true")
                
                return {
                    'status': 'training_started',
                    'message': 'Model training initiated in background',
                    'symbols': symbols or training_pipeline.training_config.get('symbols', []),
                    'model_types': model_types or training_pipeline.training_config.get('model_types', []),
                    'duration_ms': duration_ms,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"Training trigger failed: {e}")
                raise
    
    async def get_feature_importance(self, symbol: str, model_type: str = "xgboost") -> Dict[str, Any]:
        """Get feature importance for interpretability"""
        try:
            from ..core.ml.feature_store import feature_store
            
            # Get features
            features_df = await feature_store.compute_features(symbol.upper())
            
            if features_df.empty:
                raise ValueError(f"No features available for {symbol}")
            
            # Get feature importance from feature store
            importance = feature_store.get_feature_importance()
            
            return {
                'symbol': symbol.upper(),
                'model_type': model_type,
                'feature_importance': importance,
                'total_features': len(features_df.columns),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Feature importance retrieval failed: {e}")
            raise
    
    async def run_automated_retraining(self) -> Dict[str, Any]:
        """Run automated retraining check"""
        try:
            retrained_models = await self.auto_retrainer.check_and_retrain()
            
            return {
                'status': 'completed',
                'retrained_models': retrained_models,
                'retrained_count': len(retrained_models),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Automated retraining failed: {e}")
            raise
