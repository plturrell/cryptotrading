"""
Configuration module for crypto trading platform
Provides secure secret management across all environments
"""

import os
import logging
from typing import Dict, Any, Optional
from .secret_manager import SecretManager, setup_secret_manager, validate_deployment_secrets

# Global secret manager instance
_secret_manager: Optional[SecretManager] = None

def get_secret_manager() -> SecretManager:
    """Get global secret manager instance"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = setup_secret_manager()
    return _secret_manager

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value with fallback to environment variables
    
    Args:
        name: Secret name
        default: Default value if secret not found
        
    Returns:
        Secret value or default
    """
    # Try secret manager first
    try:
        sm = get_secret_manager()
        value = sm.get_secret(name)
        if value:
            return value
    except Exception as e:
        logging.warning(f"Failed to get secret from manager: {e}")
    
    # Fallback to environment variable
    return os.getenv(name, default)

def get_config() -> Dict[str, Any]:
    """
    Get complete configuration for the application
    Combines secrets with application settings
    """
    config = {
        # Environment info
        "environment": os.getenv("ENVIRONMENT", "development"),
        "is_production": os.getenv("ENVIRONMENT") == "production",
        "is_development": os.getenv("ENVIRONMENT", "development") == "development",
        
        # AI Configuration
        "grok4_api_key": get_secret("GROK4_API_KEY"),
        "grok4_base_url": get_secret("GROK4_BASE_URL", "https://api.x.ai/v1"),
        "perplexity_api_key": get_secret("PERPLEXITY_API_KEY"),
        
        # Database Configuration
        "database_url": get_secret("DATABASE_URL"),
        "redis_url": get_secret("REDIS_URL", "redis://localhost:6379"),
        
        # Security
        "jwt_secret": get_secret("JWT_SECRET"),
        "encryption_key": get_secret("ENCRYPTION_KEY"),
        
        # Trading APIs
        "binance_api_key": get_secret("BINANCE_API_KEY"),
        "binance_api_secret": get_secret("BINANCE_API_SECRET"),
        "coinbase_api_key": get_secret("COINBASE_API_KEY"),
        "coinbase_api_secret": get_secret("COINBASE_API_SECRET"),
        
        # Feature Flags
        "use_real_apis": get_secret("USE_REAL_APIS", "false").lower() == "true",
        "enable_parallel_processing": get_secret("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true",
        "enable_caching": get_secret("ENABLE_CACHING", "true").lower() == "true",
        "enable_monitoring": get_secret("ENABLE_MONITORING", "true").lower() == "true",
        
        # Monitoring
        "sentry_dsn": get_secret("SENTRY_DSN"),
        "otel_endpoint": get_secret("OTEL_EXPORTER_OTLP_ENDPOINT"),
        
        # Vercel specific
        "vercel_env": get_secret("VERCEL_ENV"),
        "vercel_url": get_secret("VERCEL_URL"),
        
        # Application settings
        "debug": os.getenv("ENVIRONMENT", "development") == "development",
        "testing": False
    }
    
    return config

# Convenience exports
__all__ = [
    "SecretManager",
    "get_secret_manager",
    "get_secret",
    "get_config",
    "setup_secret_manager",
    "validate_deployment_secrets"
]
