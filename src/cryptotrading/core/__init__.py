"""
Core module for cryptotrading platform
Provides unified abstractions for storage, monitoring, and configuration
"""

# Unified interfaces
from .storage import (
    StorageInterface, SyncStorageInterface, StorageFactory,
    get_storage, get_sync_storage
)
from .monitoring import (
    MonitoringInterface, MetricType, LogLevel, MonitoringFactory,
    get_monitor
)
from .config import (
    DeploymentEnvironment, RuntimeEnvironment, FeatureFlags,
    get_deployment_environment, get_runtime_environment,
    is_vercel, is_local, is_production, is_serverless,
    get_feature_flags
)
from .bootstrap_unified import (
    UnifiedBootstrap, get_bootstrap, setup_flask_app,
    get_unified_monitor, get_unified_storage, get_unified_async_storage
)

__all__ = [
    # Storage
    'StorageInterface',
    'SyncStorageInterface', 
    'StorageFactory',
    'get_storage',
    'get_sync_storage',
    # Monitoring
    'MonitoringInterface',
    'MetricType',
    'LogLevel',
    'MonitoringFactory',
    'get_monitor',
    # Configuration
    'DeploymentEnvironment',
    'RuntimeEnvironment',
    'FeatureFlags',
    'get_deployment_environment',
    'get_runtime_environment',
    'is_vercel',
    'is_local',
    'is_production',
    'is_serverless',
    'get_feature_flags',
    # Bootstrap
    'UnifiedBootstrap',
    'get_bootstrap',
    'setup_flask_app',
    'get_unified_monitor',
    'get_unified_storage',
    'get_unified_async_storage',
]