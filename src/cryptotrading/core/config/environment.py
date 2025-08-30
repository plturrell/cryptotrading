"""
Environment detection and configuration utilities
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional


class DeploymentEnvironment(Enum):
    """Deployment environment types"""

    LOCAL = "local"
    VERCEL = "vercel"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    UNKNOWN = "unknown"


class RuntimeEnvironment(Enum):
    """Runtime environment types"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class EnvironmentDetector:
    """Detect and provide information about the current environment"""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_deployment_environment() -> DeploymentEnvironment:
        """Detect the deployment environment"""
        # Check for Vercel
        if os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"):
            return DeploymentEnvironment.VERCEL

        # Check for Docker
        if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
            return DeploymentEnvironment.DOCKER

        # Check for Kubernetes
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            return DeploymentEnvironment.KUBERNETES

        # Default to local if no other environment detected
        if os.environ.get("USER") or os.environ.get("USERNAME"):
            return DeploymentEnvironment.LOCAL

        return DeploymentEnvironment.UNKNOWN

    @staticmethod
    @lru_cache(maxsize=1)
    def get_runtime_environment() -> RuntimeEnvironment:
        """Get the runtime environment (dev/staging/prod)"""
        env = os.environ.get("ENVIRONMENT", "").lower()

        if env in ("prod", "production"):
            return RuntimeEnvironment.PRODUCTION
        elif env in ("stage", "staging"):
            return RuntimeEnvironment.STAGING
        elif env in ("test", "testing"):
            return RuntimeEnvironment.TEST
        elif env in ("dev", "development", ""):
            return RuntimeEnvironment.DEVELOPMENT

        # Try to infer from Vercel
        vercel_env = os.environ.get("VERCEL_ENV", "").lower()
        if vercel_env == "production":
            return RuntimeEnvironment.PRODUCTION
        elif vercel_env == "preview":
            return RuntimeEnvironment.STAGING

        return RuntimeEnvironment.DEVELOPMENT

    @staticmethod
    def is_vercel() -> bool:
        """Check if running on Vercel"""
        return EnvironmentDetector.get_deployment_environment() == DeploymentEnvironment.VERCEL

    @staticmethod
    def is_local() -> bool:
        """Check if running locally"""
        return EnvironmentDetector.get_deployment_environment() == DeploymentEnvironment.LOCAL

    @staticmethod
    def is_production() -> bool:
        """Check if running in production"""
        return EnvironmentDetector.get_runtime_environment() == RuntimeEnvironment.PRODUCTION

    @staticmethod
    def is_development() -> bool:
        """Check if running in development"""
        return EnvironmentDetector.get_runtime_environment() == RuntimeEnvironment.DEVELOPMENT

    @staticmethod
    def is_serverless() -> bool:
        """Check if running in a serverless environment"""
        return EnvironmentDetector.is_vercel() or bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        """Get comprehensive environment information"""
        return {
            "deployment": EnvironmentDetector.get_deployment_environment().value,
            "runtime": EnvironmentDetector.get_runtime_environment().value,
            "is_serverless": EnvironmentDetector.is_serverless(),
            "is_production": EnvironmentDetector.is_production(),
            "platform": {
                "vercel": EnvironmentDetector.is_vercel(),
                "local": EnvironmentDetector.is_local(),
                "docker": EnvironmentDetector.get_deployment_environment()
                == DeploymentEnvironment.DOCKER,
                "kubernetes": EnvironmentDetector.get_deployment_environment()
                == DeploymentEnvironment.KUBERNETES,
            },
            "env_vars": {
                "has_vercel_token": bool(os.environ.get("VERCEL_TOKEN")),
                "has_blob_token": bool(os.environ.get("BLOB_READ_WRITE_TOKEN")),
                "has_kv_url": bool(os.environ.get("KV_REST_API_URL")),
                "has_otel_endpoint": bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")),
            },
        }


class FeatureFlags:
    """Feature flags based on environment"""

    def __init__(self):
        self.detector = EnvironmentDetector()

    @property
    def use_full_monitoring(self) -> bool:
        """Whether to use full monitoring with OpenTelemetry"""
        # Use full monitoring only in non-serverless environments with OTEL configured
        return (
            not self.detector.is_serverless()
            and bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
            and os.environ.get("DISABLE_MONITORING", "").lower() != "true"
        )

    @property
    def use_real_apis(self) -> bool:
        """Whether to use real APIs vs mocks"""
        # Use real APIs in production or when explicitly enabled
        return (
            self.detector.is_production()
            or os.environ.get("USE_REAL_APIS", "false").lower() == "true"
        )

    @property
    def enable_parallel_processing(self) -> bool:
        """Whether to enable parallel processing"""
        # Disable parallel processing in serverless environments
        if self.detector.is_serverless():
            return False
        return os.environ.get("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"

    @property
    def max_workers(self) -> int:
        """Maximum number of worker processes/threads"""
        if self.detector.is_vercel():
            return 1  # Vercel serverless limitations
        elif self.detector.is_serverless():
            return 1  # Other serverless limitations
        else:
            # Local/production: use CPU count but cap it
            import multiprocessing

            return min(int(os.environ.get("MAX_WORKERS", multiprocessing.cpu_count())), 8)

    @property
    def memory_limit_mb(self) -> int:
        """Memory limit in MB"""
        if self.detector.is_vercel():
            return 512  # Vercel free tier limit
        elif self.detector.is_serverless():
            return 1024  # Other serverless limits
        else:
            return int(os.environ.get("MEMORY_LIMIT_MB", 2048))

    @property
    def batch_size(self) -> int:
        """Processing batch size"""
        if self.detector.is_vercel():
            return 50  # Smaller batches for Vercel
        elif self.detector.is_serverless():
            return 100
        else:
            return int(os.environ.get("BATCH_SIZE", 500))

    @property
    def cache_ttl_seconds(self) -> int:
        """Cache TTL in seconds"""
        if self.detector.is_development():
            return 60  # Short cache for development
        elif self.detector.is_vercel():
            return 300  # 5 minutes for Vercel
        else:
            return 180  # 3 minutes for production

    @property
    def use_distributed_agents(self) -> bool:
        """Whether to use distributed agent system"""
        # Distributed agents are too heavy for serverless
        return not self.detector.is_serverless()

    @property
    def use_cache(self) -> bool:
        """Whether to use caching"""
        # Always use cache in production
        return self.detector.is_production() or os.environ.get("ENABLE_CACHE", "").lower() == "true"

    @property
    def use_async_storage(self) -> bool:
        """Whether to use async storage operations"""
        # Use async in production and Vercel
        return self.detector.is_production() or self.detector.is_vercel()

    @property
    def enable_debug_endpoints(self) -> bool:
        """Whether to enable debug endpoints"""
        # Only in development or with explicit flag
        return (
            self.detector.is_development()
            or os.environ.get("ENABLE_DEBUG_ENDPOINTS", "").lower() == "true"
        )

    @property
    def max_request_size(self) -> int:
        """Maximum request size in bytes"""
        # Vercel has a 4.5MB limit
        if self.detector.is_vercel():
            return 4_500_000
        return 10_000_000  # 10MB for other environments

    @property
    def request_timeout(self) -> int:
        """Request timeout in seconds"""
        # Vercel has shorter timeouts
        if self.detector.is_vercel():
            return 10
        return 30

    @property
    def enable_ml_features(self) -> bool:
        """Whether to enable ML features"""
        # ML features might be limited in serverless
        if self.detector.is_serverless():
            return os.environ.get("ENABLE_ML_SERVERLESS", "").lower() == "true"
        return True

    def get_feature_flags(self) -> Dict[str, Any]:
        """Get all feature flags as a dictionary"""
        return {
            "use_full_monitoring": self.use_full_monitoring,
            "use_distributed_agents": self.use_distributed_agents,
            "use_cache": self.use_cache,
            "use_async_storage": self.use_async_storage,
            "enable_debug_endpoints": self.enable_debug_endpoints,
            "max_request_size": self.max_request_size,
            "request_timeout": self.request_timeout,
            "enable_ml_features": self.enable_ml_features,
            "use_real_apis": self.use_real_apis,
            "enable_parallel_processing": self.enable_parallel_processing,
            "max_workers": self.max_workers,
            "memory_limit_mb": self.memory_limit_mb,
            "batch_size": self.batch_size,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }

    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source configuration based on environment"""
        if self.use_real_apis:
            return {
                "yahoo_finance": {
                    "enabled": True,
                    "rate_limit": 100 if self.detector.is_vercel() else 200,
                    "timeout": self.request_timeout,
                },
                "binance": {
                    "enabled": bool(os.environ.get("BINANCE_API_KEY")),
                    "api_key": os.environ.get("BINANCE_API_KEY"),
                    "secret_key": os.environ.get("BINANCE_SECRET_KEY"),
                    "rate_limit": 50 if self.detector.is_vercel() else 100,
                    "mock_mode": False,
                },
                "coinbase": {
                    "enabled": bool(os.environ.get("COINBASE_API_KEY")),
                    "api_key": os.environ.get("COINBASE_API_KEY"),
                    "rate_limit": 30 if self.detector.is_vercel() else 60,
                    "mock_mode": False,
                },
            }
        else:
            return {
                "yahoo_finance": {"enabled": True, "mock_mode": True, "rate_limit": 10},
                "binance": {"enabled": True, "mock_mode": True, "rate_limit": 10},
                "coinbase": {"enabled": True, "mock_mode": True, "rate_limit": 10},
            }

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration based on environment"""
        return {
            "parallel_processing": self.enable_parallel_processing,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "memory_limit_mb": self.memory_limit_mb,
            "chunk_size": 50 if self.detector.is_vercel() else 200,
            "use_memory_optimization": self.detector.is_serverless(),
        }


# Singleton instances
_environment_detector = EnvironmentDetector()
_feature_flags = FeatureFlags()


# Convenience functions
def get_deployment_environment() -> DeploymentEnvironment:
    """Get the current deployment environment"""
    return _environment_detector.get_deployment_environment()


def get_runtime_environment() -> RuntimeEnvironment:
    """Get the current runtime environment"""
    return _environment_detector.get_runtime_environment()


def is_vercel() -> bool:
    """Check if running on Vercel"""
    return _environment_detector.is_vercel()


def is_local() -> bool:
    """Check if running locally"""
    return _environment_detector.is_local()


def is_production() -> bool:
    """Check if running in production"""
    return _environment_detector.is_production()


def is_serverless() -> bool:
    """Check if running in serverless environment"""
    return _environment_detector.is_serverless()


def get_feature_flags() -> FeatureFlags:
    """Get feature flags instance"""
    return _feature_flags


def get_data_source_config() -> Dict[str, Any]:
    """Get data source configuration"""
    return _feature_flags.get_data_source_config()


def get_processing_config() -> Dict[str, Any]:
    """Get processing configuration"""
    return _feature_flags.get_processing_config()
