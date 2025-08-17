"""
Unified bootstrap integration for storage and monitoring abstractions
Provides centralized initialization for all unified components
"""

import os
from typing import Dict, Any, Optional
from .config import (
    get_deployment_environment, 
    get_runtime_environment,
    get_feature_flags,
    is_vercel,
    is_local
)
from .storage import StorageFactory, get_storage, get_sync_storage
from .monitoring import MonitoringFactory, get_monitor


class UnifiedBootstrap:
    """Bootstrap unified storage and monitoring for the application"""
    
    def __init__(self, service_name: str = "cryptotrading", config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.config = config or {}
        self.feature_flags = get_feature_flags()
        
        # Initialize unified components
        self.monitor = get_monitor(service_name)
        self.storage = get_sync_storage()
        self.async_storage = get_storage()
        
        # Set global context
        self.monitor.set_tag("deployment", get_deployment_environment().value)
        self.monitor.set_tag("runtime", get_runtime_environment().value)
        self.monitor.set_tag("service", service_name)
        
        # Log bootstrap completion
        self.monitor.log_info("Unified bootstrap completed", {
            "service_name": service_name,
            "deployment": get_deployment_environment().value,
            "runtime": get_runtime_environment().value,
            "is_vercel": is_vercel(),
            "is_local": is_local(),
            "storage_type": type(self.storage).__name__,
            "monitor_type": type(self.monitor).__name__,
            "features": self.feature_flags.get_feature_flags()
        })
    
    def get_monitor(self) -> 'MonitoringInterface':
        """Get the initialized monitor"""
        return self.monitor
    
    def get_storage(self) -> 'SyncStorageInterface':
        """Get the initialized sync storage"""
        return self.storage
    
    def get_async_storage(self) -> 'StorageInterface':
        """Get the initialized async storage"""
        return self.async_storage
    
    def get_feature_flags(self) -> 'FeatureFlags':
        """Get feature flags"""
        return self.feature_flags
    
    def register_user_context(self, user_id: str, **kwargs):
        """Register user context for monitoring"""
        self.monitor.set_user_context(user_id, **kwargs)
    
    def add_global_tag(self, key: str, value: str):
        """Add a global tag to monitoring"""
        self.monitor.set_tag(key, value)
    
    def flush_monitoring(self):
        """Flush monitoring data"""
        self.monitor.flush()


# Global bootstrap instance
_bootstrap_instance: Optional[UnifiedBootstrap] = None


def get_bootstrap(service_name: str = "cryptotrading") -> UnifiedBootstrap:
    """Get or create the global bootstrap instance"""
    global _bootstrap_instance
    if _bootstrap_instance is None:
        _bootstrap_instance = UnifiedBootstrap(service_name)
    return _bootstrap_instance


def reset_bootstrap():
    """Reset the global bootstrap instance"""
    global _bootstrap_instance
    if _bootstrap_instance:
        _bootstrap_instance.flush_monitoring()
    _bootstrap_instance = None


# Convenience functions
def get_unified_monitor(service_name: str = "cryptotrading"):
    """Get monitor from bootstrap"""
    return get_bootstrap(service_name).get_monitor()


def get_unified_storage():
    """Get sync storage from bootstrap"""
    return get_bootstrap().get_storage()


def get_unified_async_storage():
    """Get async storage from bootstrap"""
    return get_bootstrap().get_async_storage()


def setup_flask_app(app, service_name: str = "cryptotrading"):
    """Setup Flask app with unified monitoring"""
    bootstrap = get_bootstrap(service_name)
    monitor = bootstrap.get_monitor()
    
    # Add request tracking
    @app.before_request
    def before_request():
        from flask import request
        monitor.add_breadcrumb(f"Request: {request.method} {request.path}", "http")
        monitor.increment_counter("http.requests", tags={
            "method": request.method,
            "endpoint": request.endpoint or "unknown"
        })
    
    @app.after_request
    def after_request(response):
        monitor.increment_counter("http.responses", tags={
            "status_code": str(response.status_code)
        })
        return response
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        monitor.record_error(e, {"context": "flask_request"})
        return {"error": "Internal server error"}, 500
    
    return bootstrap