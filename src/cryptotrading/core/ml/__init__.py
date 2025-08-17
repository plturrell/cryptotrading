"""
Production ML/AI system for cryptocurrency predictions
Includes real models, training pipeline, and inference API
"""

from .perplexity import PerplexityClient
from .yfinance_client import get_yfinance_client
from .models import CryptoPricePredictor, ModelRegistry, model_registry
from .training import ModelTrainingPipeline, ModelEvaluator, training_pipeline, model_evaluator
from .inference import (
    MLInferenceService, 
    inference_service,
    create_ml_api,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    ModelPerformanceResponse
)
from .vercel_deployment import (
    VercelMLModel,
    VercelMLEngine,
    create_vercel_handler,
    export_models_for_vercel
)
from .feature_store import (
    FeatureStore,
    FeatureDefinition,
    feature_store
)

__all__ = [
    # Legacy
    'PerplexityClient', 
    'get_yfinance_client',
    
    # Models
    'CryptoPricePredictor',
    'ModelRegistry',
    'model_registry',
    
    # Training
    'ModelTrainingPipeline',
    'ModelEvaluator',
    'training_pipeline',
    'model_evaluator',
    
    # Inference
    'MLInferenceService',
    'inference_service',
    'create_ml_api',
    'PredictionRequest',
    'PredictionResponse',
    'BatchPredictionRequest',
    'ModelPerformanceResponse',
    
    # Deployment
    'VercelMLModel',
    'VercelMLEngine',
    'create_vercel_handler',
    'export_models_for_vercel',
    
    # Feature Store
    'FeatureStore',
    'FeatureDefinition',
    'feature_store'
]