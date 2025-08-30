"""
Real-time ML inference API for cryptocurrency predictions
"""

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

# Market data should be fetched through database, not directly
from ..storage import get_cache_client
from .models import CryptoPricePredictor, model_registry
from .training import model_evaluator, training_pipeline

# Simple logging for Vercel compatibility
logger = logging.getLogger(__name__)


# Simple metrics tracking
class SimpleMetrics:
    def __init__(self):
        self.counters = {}
        self.histograms = {}

    def counter(self, name: str, value: int = 1, labels: dict = None):
        key = f"{name}:{labels}" if labels else name
        self.counters[key] = self.counters.get(key, 0) + value

    def histogram(self, name: str, value: float, labels: dict = None):
        key = f"{name}:{labels}" if labels else name
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)


business_metrics = SimpleMetrics()


class PredictionRequest(BaseModel):
    symbol: str
    horizon: str = "24h"  # 1h, 24h, 7d
    model_type: Optional[str] = None  # ensemble, neural_network, random_forest
    include_confidence: bool = True
    include_features: bool = False


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    confidence: float
    prediction_time: str
    horizon: str
    model_version: str
    model_type: str
    intervals: Dict[str, Dict[str, float]]
    features_used: Optional[int] = None
    top_features: Optional[List[Dict[str, float]]] = None
    cached: bool = False


class BatchPredictionRequest(BaseModel):
    symbols: List[str]
    horizon: str = "24h"
    model_type: Optional[str] = None


class ModelPerformanceResponse(BaseModel):
    model_name: str
    symbol: str
    horizon: str
    model_type: str
    metrics: Dict[str, float]
    last_trained: str
    backtesting_results: Optional[Dict[str, Any]] = None


class MLInferenceService:
    """Real-time ML inference service"""

    def __init__(self):
        # ML should get data through database, not direct market data clients
        from ...infrastructure.database.unified_database import UnifiedDatabase

        self.database = UnifiedDatabase()
        self.cache_client = get_cache_client()
        self.cache_ttl = 300  # 5 minutes
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("ML Inference Service initialized with database data access")

    def _get_cache_key(self, symbol: str, horizon: str, model_type: Optional[str]) -> str:
        """Generate cache key for predictions"""
        key_parts = [symbol, horizon, model_type or "auto"]
        key_str = ":".join(key_parts)
        return f"ml_prediction:{hashlib.md5(key_str.encode()).hexdigest()}"

    async def get_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Get price prediction for a symbol"""
        try:
            # Check cache
            cache_key = self.cache_client.generate_cache_key(
                request.symbol, request.horizon, request.model_type or "auto"
            )
            cached_result = await self.cache_client.get_prediction(cache_key)

            if cached_result:
                business_metrics.counter("ml.inference.cache_hit", 1, {"symbol": request.symbol})
                cached_result["cached"] = True
                return PredictionResponse(**cached_result)

            # Parse horizon
            horizon_hours = self._parse_horizon(request.horizon)

            # Select model
            if request.model_type:
                model_name = f"{request.symbol}_{request.model_type}_{horizon_hours}h"
            else:
                # Use best performing model
                model_name = training_pipeline.get_best_model(request.symbol, horizon_hours)
                if not model_name:
                    # Default to ensemble
                    model_name = f"{request.symbol}_ensemble_{horizon_hours}h"

            # Get or train model
            try:
                model = model_registry.get_model(model_name)
            except ValueError:
                # Model doesn't exist, train it
                logger.info(f"Model {model_name} not found, training new model")
                await training_pipeline.train_model(
                    request.symbol, request.model_type or "ensemble", horizon_hours
                )
                model = model_registry.get_model(model_name)

            # Get recent data
            data = await self._get_recent_data(request.symbol)
            if data is None:
                raise HTTPException(
                    status_code=404, detail=f"No data available for {request.symbol}"
                )

            # Make prediction
            prediction = model.predict(data)

            # Prepare response
            response_data = {
                "symbol": request.symbol,
                "current_price": prediction["current_price"],
                "predicted_price": prediction["predicted_price"],
                "price_change": prediction["price_change"],
                "price_change_percent": prediction["price_change_percent"],
                "confidence": prediction["confidence"] if request.include_confidence else None,
                "prediction_time": prediction["prediction_time"],
                "horizon": request.horizon,
                "model_version": prediction["model_version"],
                "model_type": prediction["model_type"],
                "intervals": prediction["intervals"],
                "features_used": prediction["features_used"] if request.include_features else None,
                "top_features": prediction["top_features"] if request.include_features else None,
                "cached": False,
            }

            # Cache result
            await self.cache_client.set_prediction(cache_key, response_data, self.cache_ttl)

            # Track metrics
            business_metrics.counter(
                "ml.inference.prediction",
                1,
                {
                    "symbol": request.symbol,
                    "horizon": request.horizon,
                    "model_type": model.model_type,
                },
            )

            business_metrics.histogram(
                "ml.inference.confidence", prediction["confidence"], {"symbol": request.symbol}
            )

            return PredictionResponse(**response_data)

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            business_metrics.counter(
                "ml.inference.error", 1, {"symbol": request.symbol, "error": str(e)}
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def get_batch_predictions(
        self, request: BatchPredictionRequest
    ) -> List[PredictionResponse]:
        """Get predictions for multiple symbols"""
        tasks = []

        for symbol in request.symbols:
            pred_request = PredictionRequest(
                symbol=symbol, horizon=request.horizon, model_type=request.model_type
            )
            tasks.append(self.get_prediction(pred_request))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        predictions = []
        for result in results:
            if isinstance(result, PredictionResponse):
                predictions.append(result)
            else:
                logger.error(f"Batch prediction error: {result}")

        return predictions

    async def get_model_performance(self, symbol: str, horizon: str) -> ModelPerformanceResponse:
        """Get model performance metrics"""
        try:
            horizon_hours = self._parse_horizon(horizon)

            # Find best model
            model_name = training_pipeline.get_best_model(symbol, horizon_hours)
            if not model_name:
                raise HTTPException(
                    status_code=404, detail=f"No model found for {symbol} {horizon}"
                )

            # Get model info
            model = model_registry.get_model(model_name)

            # Get performance from training
            performance = None
            for perf in training_pipeline.performance_history:
                if (
                    perf["symbol"] == symbol
                    and perf["horizon_hours"] == horizon_hours
                    and perf["model_type"] == model.model_type
                ):
                    performance = perf
                    break

            if not performance:
                raise HTTPException(status_code=404, detail="Performance metrics not found")

            # Run backtesting
            backtest_results = await model_evaluator.backtest_model(model_name, test_days=7)

            return ModelPerformanceResponse(
                model_name=model_name,
                symbol=symbol,
                horizon=horizon,
                model_type=model.model_type,
                metrics=performance["metrics"],
                last_trained=performance["trained_at"],
                backtesting_results=backtest_results,
            )

        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_recent_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get recent data for prediction from database"""
        try:
            # Get data from database instead of direct market data client
            data = await self.database.get_historical_data(symbol=symbol, days=90, interval="1h")

            # If not enough data in database, use real data provider
            if data is None or len(data) < 100:
                logger.info(f"Insufficient historical data for {symbol}, fetching from provider")
                from ...data.providers.real_only_provider import RealOnlyDataProvider

                provider = RealOnlyDataProvider()
                # Get recent data directly for ML training
                fresh_data = provider.get_historical_data(symbol, days=90)

                if fresh_data is not None and len(fresh_data) > 0:
                    # Store in database for future use
                    await self.database.store_market_data(symbol, fresh_data)
                    return fresh_data

            return data

        except Exception as e:
            logger.error(f"Error fetching recent data from database: {e}")
            return None

    def _parse_horizon(self, horizon: str) -> int:
        """Parse horizon string to hours"""
        if horizon == "1h":
            return 1
        elif horizon == "24h":
            return 24
        elif horizon == "7d":
            return 168
        else:
            raise ValueError(f"Invalid horizon: {horizon}")

    # Cache methods now handled by self.cache_client


class ModelVersioningService:
    """A/B testing and model versioning"""

    def __init__(self):
        self.version_config = {
            "traffic_split": {"stable": 0.8, "experimental": 0.2},
            "versions": {},
        }

    def register_version(self, symbol: str, horizon: str, version: str, model_name: str):
        """Register a model version"""
        key = f"{symbol}_{horizon}"

        if key not in self.version_config["versions"]:
            self.version_config["versions"][key] = {}

        self.version_config["versions"][key][version] = {
            "model_name": model_name,
            "registered_at": datetime.now().isoformat(),
            "requests": 0,
            "errors": 0,
        }

    def select_version(self, symbol: str, horizon: str) -> str:
        """Select model version based on traffic split"""
        import random

        key = f"{symbol}_{horizon}"
        versions = self.version_config["versions"].get(key, {})

        if not versions:
            return None

        # Simple traffic split
        if (
            "experimental" in versions
            and random.random() < self.version_config["traffic_split"]["experimental"]
        ):
            return versions["experimental"]["model_name"]
        elif "stable" in versions:
            return versions["stable"]["model_name"]
        else:
            # Return any available version
            return list(versions.values())[0]["model_name"]

    def update_metrics(self, model_name: str, success: bool):
        """Update version metrics"""
        for key, versions in self.version_config["versions"].items():
            for version, config in versions.items():
                if config["model_name"] == model_name:
                    config["requests"] += 1
                    if not success:
                        config["errors"] += 1
                    return


# Global instances
inference_service = MLInferenceService()
versioning_service = ModelVersioningService()


# FastAPI endpoints
def create_ml_api():
    """Create FastAPI app for ML inference"""
    app = FastAPI(title="Crypto ML Inference API", version="1.0.0")

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Get price prediction for a cryptocurrency"""
        return await inference_service.get_prediction(request)

    @app.post("/predict/batch", response_model=List[PredictionResponse])
    async def batch_predict(request: BatchPredictionRequest):
        """Get predictions for multiple cryptocurrencies"""
        return await inference_service.get_batch_predictions(request)

    @app.get("/models/{symbol}/{horizon}/performance", response_model=ModelPerformanceResponse)
    async def get_performance(symbol: str, horizon: str):
        """Get model performance metrics"""
        return await inference_service.get_model_performance(symbol, horizon)

    @app.post("/models/train")
    async def trigger_training(background_tasks: BackgroundTasks):
        """Trigger model training in background"""
        background_tasks.add_task(training_pipeline.train_all_models)
        return {"message": "Training started in background"}

    @app.get("/models")
    async def list_models():
        """List all available models"""
        return {"models": model_registry.list_models(), "active_model": model_registry.active_model}

    return app
