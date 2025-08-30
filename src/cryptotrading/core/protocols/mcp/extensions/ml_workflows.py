"""
MCP Protocol Extensions for ML/AI Workflows

Provides tools and resources for distributed ML training, model management,
and inference coordination in the cryptotrading system.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..resources import Resource
from ..tools import MCPTool, ToolResult


class ModelType(Enum):
    """Supported ML model types"""

    PRICE_PREDICTION = "price_prediction"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


class TrainingStatus(Enum):
    """ML training job statuses"""

    PENDING = "pending"
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MLTrainingTool(MCPTool):
    """Tool for coordinating distributed ML training"""

    name = "ml_training_coordinator"
    description = "Coordinate distributed ML model training with resource management"
    parameters = {
        "model_type": {
            "type": "string",
            "enum": [t.value for t in ModelType],
            "description": "Type of model to train",
        },
        "model_config": {
            "type": "object",
            "properties": {
                "architecture": {"type": "string"},
                "hyperparameters": {"type": "object"},
                "features": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string"},
            },
        },
        "training_data": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "train_split": {"type": "number", "default": 0.8},
                "validation_split": {"type": "number", "default": 0.1},
                "test_split": {"type": "number", "default": 0.1},
            },
        },
        "distributed_config": {
            "type": "object",
            "properties": {
                "num_workers": {"type": "integer", "minimum": 1, "default": 1},
                "gpu_per_worker": {"type": "integer", "minimum": 0, "default": 0},
                "batch_size": {"type": "integer", "minimum": 1},
                "epochs": {"type": "integer", "minimum": 1},
                "checkpoint_interval": {"type": "integer", "default": 10},
            },
        },
        "optimization_config": {
            "type": "object",
            "properties": {
                "optimizer": {"type": "string", "enum": ["adam", "sgd", "rmsprop"]},
                "learning_rate": {"type": "number"},
                "early_stopping": {"type": "boolean", "default": True},
                "patience": {"type": "integer", "default": 5},
            },
        },
    }


@dataclass
class ModelDeploymentTool(MCPTool):
    """Tool for deploying trained models"""

    name = "model_deployment"
    description = "Deploy ML models for inference with versioning and A/B testing"
    parameters = {
        "model_id": {"type": "string"},
        "version": {"type": "string"},
        "deployment_config": {
            "type": "object",
            "properties": {
                "environment": {"type": "string", "enum": ["dev", "staging", "production"]},
                "replicas": {"type": "integer", "minimum": 1, "default": 1},
                "autoscaling": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": False},
                        "min_replicas": {"type": "integer", "default": 1},
                        "max_replicas": {"type": "integer", "default": 10},
                        "target_cpu": {"type": "integer", "default": 70},
                    },
                },
                "canary_config": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": False},
                        "traffic_percentage": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                },
            },
        },
    }


@dataclass
class InferenceTool(MCPTool):
    """Tool for running model inference"""

    name = "ml_inference"
    description = "Run inference using deployed models with batching and streaming"
    parameters = {
        "model_id": {"type": "string"},
        "version": {"type": "string", "default": "latest"},
        "input_data": {
            "type": "object",
            "properties": {
                "batch": {"type": "array", "description": "Batch of inputs"},
                "stream_id": {
                    "type": "string",
                    "description": "Stream ID for continuous inference",
                },
            },
        },
        "inference_config": {
            "type": "object",
            "properties": {
                "batch_size": {"type": "integer", "default": 32},
                "timeout_ms": {"type": "integer", "default": 5000},
                "return_probabilities": {"type": "boolean", "default": False},
                "explanation": {"type": "boolean", "default": False},
            },
        },
    }


@dataclass
class ModelMonitoringTool(MCPTool):
    """Tool for monitoring model performance"""

    name = "model_monitoring"
    description = "Monitor model performance, drift, and quality metrics"
    parameters = {
        "model_id": {"type": "string"},
        "monitoring_config": {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["accuracy", "precision", "recall", "f1", "auc", "mse", "mae"],
                    },
                },
                "drift_detection": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "threshold": {"type": "number", "default": 0.1},
                        "window_size": {"type": "integer", "default": 1000},
                    },
                },
                "alert_config": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "channels": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    }


class ModelRegistryResource(Resource):
    """Resource for accessing model registry"""

    def __init__(self):
        super().__init__(
            uri="ml://registry",
            name="Model Registry",
            description="Registry of all trained models with metadata",
            mime_type="application/json",
        )
        self.models = {}

    async def read(self) -> str:
        """Return model registry"""
        import json

        registry = {
            "total_models": len(self.models),
            "model_types": list(set(m["type"] for m in self.models.values())),
            "models": self.models,
        }
        return json.dumps(registry, indent=2, default=str)


class TrainingJobsResource(Resource):
    """Resource for monitoring training jobs"""

    def __init__(self):
        super().__init__(
            uri="ml://training/jobs",
            name="Training Jobs",
            description="Status of all ML training jobs",
            mime_type="application/json",
        )
        self.jobs = {}

    async def read(self) -> str:
        """Return training job status"""
        import json

        status = {
            "active_jobs": len(
                [j for j in self.jobs.values() if j["status"] == TrainingStatus.TRAINING.value]
            ),
            "completed_jobs": len(
                [j for j in self.jobs.values() if j["status"] == TrainingStatus.COMPLETED.value]
            ),
            "jobs": self.jobs,
        }
        return json.dumps(status, indent=2, default=str)


class ModelMetricsResource(Resource):
    """Resource for model performance metrics"""

    def __init__(self):
        super().__init__(
            uri="ml://metrics",
            name="Model Metrics",
            description="Performance metrics for deployed models",
            mime_type="application/json",
        )
        self.metrics = {}

    async def read(self) -> str:
        """Return model metrics"""
        import json

        return json.dumps(self.metrics, indent=2, default=str)
