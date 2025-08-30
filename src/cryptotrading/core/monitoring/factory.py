"""
Monitoring factory for environment-aware monitoring selection
"""

import os
from typing import Optional

from .base import MonitoringInterface, NoOpMonitoring
from .full import OTEL_AVAILABLE, FullMonitoring
from .lightweight import LightweightMonitoring


class MonitoringFactory:
    """Factory for creating appropriate monitoring implementation based on environment"""

    # Singleton instances
    _instances = {}

    @classmethod
    def get_monitor(
        cls,
        monitoring_type: Optional[str] = None,
        service_name: str = "cryptotrading",
        environment: Optional[str] = None,
    ) -> MonitoringInterface:
        """
        Get monitoring implementation based on environment

        Args:
            monitoring_type: Override monitoring type ('full', 'lightweight', 'noop')
            service_name: Name of the service
            environment: Environment name (production, development, etc.)

        Returns:
            Monitoring implementation instance
        """
        # Determine environment
        if environment is None:
            environment = os.environ.get("ENVIRONMENT", "development")

        # Create cache key
        cache_key = f"{monitoring_type}:{service_name}:{environment}"

        # Return cached instance if available
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Determine monitoring type from environment if not specified
        if monitoring_type is None:
            if os.environ.get("VERCEL"):
                # On Vercel, use lightweight monitoring
                monitoring_type = "lightweight"
            elif os.environ.get("DISABLE_MONITORING", "").lower() == "true":
                # Monitoring explicitly disabled
                monitoring_type = "noop"
            elif os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") and OTEL_AVAILABLE:
                # OpenTelemetry endpoint configured and available
                monitoring_type = "full"
            else:
                # Default to lightweight in production, full in development
                monitoring_type = "lightweight" if environment == "production" else "full"

        # Create appropriate monitoring instance
        if monitoring_type == "noop":
            instance = NoOpMonitoring()

        elif monitoring_type == "lightweight":
            instance = LightweightMonitoring(service_name, environment)

        elif monitoring_type == "full":
            if not OTEL_AVAILABLE:
                print(
                    "Warning: Full monitoring requested but OpenTelemetry not available. Using lightweight monitoring."
                )
                instance = LightweightMonitoring(service_name, environment)
            else:
                instance = FullMonitoring(service_name, environment)

        else:
            raise ValueError(f"Unknown monitoring type: {monitoring_type}")

        # Cache the instance
        cls._instances[cache_key] = instance

        return instance

    @classmethod
    def clear_cache(cls):
        """Clear cached monitoring instances"""
        cls._instances.clear()


# Convenience function for backward compatibility
def get_monitor(service_name: str = "cryptotrading") -> MonitoringInterface:
    """Get default monitoring for current environment"""
    return MonitoringFactory.get_monitor(service_name=service_name)
