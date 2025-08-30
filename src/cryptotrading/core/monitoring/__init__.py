"""
Unified monitoring package for cryptotrading
Provides environment-aware monitoring implementations
"""

from .base import (
    LogLevel,
    MetricType,
    MonitoringInterface,
    NoOpMonitoring,
    NoOpSpanContext,
    SpanContext,
)
from .factory import MonitoringFactory, get_monitor
from .full import FullMonitoring
from .lightweight import LightweightMonitoring

# Legacy imports for backward compatibility
try:
    from .vercel_monitoring import VercelLogger, VercelMonitor, VercelTracer
except ImportError:
    VercelLogger = VercelMonitor = VercelTracer = None

__all__ = [
    # New unified interface
    "MonitoringInterface",
    "SpanContext",
    "MetricType",
    "LogLevel",
    "NoOpMonitoring",
    "NoOpSpanContext",
    "MonitoringFactory",
    "get_monitor",
    "LightweightMonitoring",
    "FullMonitoring",
    # Legacy exports
    "VercelLogger",
    "VercelMonitor",
    "VercelTracer",
]
