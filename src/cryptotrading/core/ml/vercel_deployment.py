"""
Vercel-compatible ML deployment configuration
Optimized for serverless edge functions
"""

import base64
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VercelMLModel:
    """Lightweight ML model for Vercel deployment"""

    def __init__(self, model_data: Optional[Dict] = None):
        self.model_data = model_data or {}
        self.coefficients = None
        self.intercept = None
        self.feature_names = []
        self.scaler_params = {}

    def from_sklearn_model(self, sklearn_model, feature_names: list, scaler=None):
        """Convert sklearn model to lightweight format"""
        # Extract model parameters
        if hasattr(sklearn_model, "coef_"):
            self.coefficients = sklearn_model.coef_.tolist()
        if hasattr(sklearn_model, "intercept_"):
            self.intercept = float(sklearn_model.intercept_)

        self.feature_names = feature_names

        # Extract scaler parameters
        if scaler:
            self.scaler_params = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}

        # Store model type
        self.model_data["model_type"] = type(sklearn_model).__name__
        self.model_data["created_at"] = datetime.now().isoformat()

    def predict(self, features: Dict[str, float]) -> float:
        """Make prediction with lightweight model"""
        # Convert features to array
        X = np.array([features.get(name, 0) for name in self.feature_names])

        # Apply scaling
        if self.scaler_params:
            mean = np.array(self.scaler_params["mean"])
            scale = np.array(self.scaler_params["scale"])
            X = (X - mean) / scale

        # Simple linear prediction
        if self.coefficients:
            prediction = np.dot(X, self.coefficients) + self.intercept
        else:
            # Fallback to simple average
            prediction = np.mean(X)

        return float(prediction)

    def to_json(self) -> str:
        """Serialize model to JSON"""
        return json.dumps(
            {
                "coefficients": self.coefficients,
                "intercept": self.intercept,
                "feature_names": self.feature_names,
                "scaler_params": self.scaler_params,
                "model_data": self.model_data,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "VercelMLModel":
        """Load model from JSON"""
        data = json.loads(json_str)
        model = cls()
        model.coefficients = data.get("coefficients")
        model.intercept = data.get("intercept")
        model.feature_names = data.get("feature_names", [])
        model.scaler_params = data.get("scaler_params", {})
        model.model_data = data.get("model_data", {})
        return model


class VercelMLEngine:
    """ML engine optimized for Vercel edge functions"""

    def __init__(self):
        self.models = {}
        self.feature_extractors = {
            "price_features": self._extract_price_features,
            "technical_indicators": self._extract_technical_indicators,
            "time_features": self._extract_time_features,
        }

    def _extract_price_features(self, data: Dict) -> Dict[str, float]:
        """Extract price-based features"""
        current_price = data.get("price", 0)
        prev_price = data.get("prev_price", current_price)

        return {
            "price": current_price,
            "price_change": current_price - prev_price,
            "price_change_pct": (current_price - prev_price) / prev_price if prev_price > 0 else 0,
            "log_price": np.log(current_price) if current_price > 0 else 0,
        }

    def _extract_technical_indicators(self, data: Dict) -> Dict[str, float]:
        """Extract technical indicators"""
        prices = data.get("price_history", [])

        if len(prices) < 2:
            return {
                "sma_5": data.get("price", 0),
                "sma_20": data.get("price", 0),
                "rsi": 50,
                "momentum": 0,
            }

        prices_array = np.array(prices)

        # Simple moving averages
        sma_5 = np.mean(prices_array[-5:]) if len(prices_array) >= 5 else np.mean(prices_array)
        sma_20 = np.mean(prices_array[-20:]) if len(prices_array) >= 20 else np.mean(prices_array)

        # RSI approximation
        deltas = np.diff(prices_array)
        gains = deltas[deltas > 0].sum()
        losses = -deltas[deltas < 0].sum()
        rs = gains / losses if losses > 0 else 1
        rsi = 100 - (100 / (1 + rs))

        # Momentum
        momentum = prices_array[-1] - prices_array[0]

        return {
            "sma_5": float(sma_5),
            "sma_20": float(sma_20),
            "rsi": float(rsi),
            "momentum": float(momentum),
        }

    def _extract_time_features(self, data: Dict) -> Dict[str, float]:
        """Extract time-based features"""
        timestamp = data.get("timestamp", datetime.now())

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return {
            "hour": float(timestamp.hour),
            "day_of_week": float(timestamp.weekday()),
            "day_of_month": float(timestamp.day),
            "month": float(timestamp.month),
        }

    def extract_features(self, data: Dict) -> Dict[str, float]:
        """Extract all features from input data"""
        features = {}

        for extractor_name, extractor_func in self.feature_extractors.items():
            features.update(extractor_func(data))

        return features

    def predict(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Make prediction for a symbol"""
        # Extract features
        features = self.extract_features(data)

        # Get model
        model = self.models.get(symbol)
        if not model:
            # Use default model or simple heuristic
            return self._heuristic_prediction(symbol, features)

        # Make prediction
        prediction = model.predict(features)

        # Calculate confidence based on feature completeness
        confidence = self._calculate_confidence(features)

        return {
            "symbol": symbol,
            "prediction": prediction,
            "confidence": confidence,
            "features_used": len(features),
            "timestamp": datetime.now().isoformat(),
        }

    def _heuristic_prediction(self, symbol: str, features: Dict[str, float]) -> Dict[str, Any]:
        """No heuristic fallback - fail if no model available"""
        raise ValueError(f"No trained model available for {symbol}. Please train a model first.")

    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate prediction confidence"""
        # Base confidence
        confidence = 80.0

        # Reduce confidence for missing features
        expected_features = 12
        actual_features = len([v for v in features.values() if v != 0])
        feature_completeness = actual_features / expected_features

        confidence *= feature_completeness

        # Adjust for extreme RSI values
        rsi = features.get("rsi", 50)
        if rsi < 20 or rsi > 80:
            confidence *= 0.9

        return min(max(confidence, 50), 95)

    def load_model(self, symbol: str, model_json: str):
        """Load a model from JSON"""
        self.models[symbol] = VercelMLModel.from_json(model_json)

    def save_model(self, symbol: str) -> str:
        """Save a model to JSON"""
        if symbol not in self.models:
            raise ValueError(f"No model found for {symbol}")

        return self.models[symbol].to_json()


# Vercel Edge Function Handler
def create_vercel_handler():
    """Create Vercel edge function handler"""

    # Initialize engine
    engine = VercelMLEngine()

    # Load pre-trained models from environment variables
    models_config = os.environ.get("ML_MODELS_CONFIG")
    if models_config:
        try:
            config = json.loads(base64.b64decode(models_config))
            for symbol, model_json in config.items():
                engine.load_model(symbol, model_json)
                logger.info(f"Loaded model for {symbol}")
        except Exception as e:
            logger.error(f"Error loading models config: {e}")

    async def handler(request):
        """Vercel edge function handler"""
        try:
            # Parse request
            body = await request.json()

            symbol = body.get("symbol", "BTC")
            data = body.get("data", {})

            # Add current timestamp if not provided
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()

            # Make prediction
            result = engine.predict(symbol, data)

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Cache-Control": "public, s-maxage=300",  # Cache for 5 minutes
                },
                "body": json.dumps(result),
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e), "timestamp": datetime.now().isoformat()}),
            }

    return handler


# Model export utilities
def export_models_for_vercel(models_dir: Path, output_file: Path):
    """Export trained models for Vercel deployment"""
    models_config = {}

    # Find all model files
    for model_file in models_dir.glob("*/metadata.json"):
        try:
            # Load metadata
            with open(model_file, "r") as f:
                metadata = json.load(f)

            symbol = model_file.parent.name.split("_")[0]

            # Create lightweight model
            vercel_model = VercelMLModel()

            # Load feature importance as coefficients
            importance_file = model_file.parent / "feature_importance.json"
            if importance_file.exists():
                with open(importance_file, "r") as f:
                    importance = json.load(f)

                # Use feature importance as simple linear coefficients
                vercel_model.coefficients = list(importance.values())
                vercel_model.feature_names = list(importance.keys())
                vercel_model.intercept = 0.0
                vercel_model.model_data = {"source": "feature_importance", "metadata": metadata}

            # Add to config
            models_config[symbol] = vercel_model.to_json()

        except Exception as e:
            logger.error(f"Error exporting model {model_file}: {e}")

    # Save config
    config_str = base64.b64encode(json.dumps(models_config).encode()).decode()

    with open(output_file, "w") as f:
        f.write(f"ML_MODELS_CONFIG={config_str}\n")

    logger.info(f"Exported {len(models_config)} models to {output_file}")

    return models_config
