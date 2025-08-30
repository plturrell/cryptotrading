"""
Production ML/AI system for cryptocurrency predictions
Includes real models, training pipeline, and inference API
"""

from .feature_store import FeatureDefinition, FeatureStore, feature_store
from .inference import (
    BatchPredictionRequest,
    MLInferenceService,
    ModelPerformanceResponse,
    PredictionRequest,
    PredictionResponse,
    create_ml_api,
    inference_service,
)
from .models import CryptoPricePredictor, ModelRegistry, model_registry
from .perplexity import PerplexityClient
from .training import ModelEvaluator, ModelTrainingPipeline, model_evaluator, training_pipeline
from .vercel_deployment import (
    VercelMLEngine,
    VercelMLModel,
    create_vercel_handler,
    export_models_for_vercel,
)

__all__ = [
    # Legacy
    "PerplexityClient",
    # Models
    "CryptoPricePredictor",
    "ModelRegistry",
    "model_registry",
    # Training
    "ModelTrainingPipeline",
    "ModelEvaluator",
    "training_pipeline",
    "model_evaluator",
    # Inference
    "MLInferenceService",
    "inference_service",
    "create_ml_api",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "ModelPerformanceResponse",
    # Deployment
    "VercelMLModel",
    "VercelMLEngine",
    "create_vercel_handler",
    "export_models_for_vercel",
    # Feature Store
    "FeatureStore",
    "FeatureDefinition",
    "feature_store",
]
