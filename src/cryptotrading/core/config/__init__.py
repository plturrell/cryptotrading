"""
Configuration package for cryptotrading
Provides environment detection and configuration management
"""

from .environment import (
    DeploymentEnvironment,
    EnvironmentDetector,
    FeatureFlags,
    RuntimeEnvironment,
    get_deployment_environment,
    get_feature_flags,
    get_runtime_environment,
    is_local,
    is_production,
    is_serverless,
    is_vercel,
)

# Try to import production config if it exists
try:
    from .production_config import ProductionConfig
except ImportError:
    ProductionConfig = None

__all__ = [
    # Environment detection
    "DeploymentEnvironment",
    "RuntimeEnvironment",
    "EnvironmentDetector",
    "FeatureFlags",
    "get_deployment_environment",
    "get_runtime_environment",
    "is_vercel",
    "is_local",
    "is_production",
    "is_serverless",
    "get_feature_flags",
    # Config classes
    "ProductionConfig",
]
