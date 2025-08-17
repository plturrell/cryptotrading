"""
Base monitoring interface for unified monitoring abstraction
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime
from enum import Enum
import time
from contextlib import contextmanager


class MetricType(Enum):
    """Types of metrics that can be recorded"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class LogLevel(Enum):
    """Standard log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringInterface(ABC):
    """Abstract base class for monitoring implementations"""
    
    @abstractmethod
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.COUNTER, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        pass
    
    @abstractmethod
    def log(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a message with structured data"""
        pass
    
    @abstractmethod
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> 'SpanContext':
        """Start a new tracing span"""
        pass
    
    @abstractmethod
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error with context"""
        pass
    
    @abstractmethod
    def add_breadcrumb(self, message: str, category: str = "custom", level: LogLevel = LogLevel.INFO, data: Optional[Dict[str, Any]] = None) -> None:
        """Add a breadcrumb for debugging"""
        pass
    
    @abstractmethod
    def set_user_context(self, user_id: str, email: Optional[str] = None, username: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Set user context for monitoring"""
        pass
    
    @abstractmethod
    def set_tag(self, key: str, value: Any) -> None:
        """Set a global tag for all monitoring data"""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush any pending monitoring data"""
        pass
    
    # Convenience methods with default implementations
    def log_debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message"""
        self.log(LogLevel.DEBUG, message, extra)
    
    def log_info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message"""
        self.log(LogLevel.INFO, message, extra)
    
    def log_warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message"""
        self.log(LogLevel.WARNING, message, extra)
    
    def log_error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message"""
        self.log(LogLevel.ERROR, message, extra)
    
    def log_critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message"""
        self.log(LogLevel.CRITICAL, message, extra)
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric"""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    @contextmanager
    def timed(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager to time operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_histogram(metric_name, duration * 1000, tags)  # Convert to milliseconds
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing spans"""
        span = self.start_span(name, attributes)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()


class SpanContext(ABC):
    """Abstract base class for span contexts"""
    
    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span"""
        pass
    
    @abstractmethod
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span"""
        pass
    
    @abstractmethod
    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the span"""
        pass
    
    @abstractmethod
    def end(self) -> None:
        """End the span"""
        pass
    
    @abstractmethod
    def __enter__(self):
        """Enter context manager"""
        return self
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        if exc_val:
            self.record_exception(exc_val)
        self.end()


class NoOpSpanContext(SpanContext):
    """No-op implementation of SpanContext for when monitoring is disabled"""
    
    def set_attribute(self, key: str, value: Any) -> None:
        pass
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def record_exception(self, exception: Exception) -> None:
        pass
    
    def end(self) -> None:
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class NoOpMonitoring(MonitoringInterface):
    """No-op implementation for when monitoring is disabled"""
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.COUNTER, tags: Optional[Dict[str, str]] = None) -> None:
        pass
    
    def log(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> SpanContext:
        return NoOpSpanContext()
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def add_breadcrumb(self, message: str, category: str = "custom", level: LogLevel = LogLevel.INFO, data: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def set_user_context(self, user_id: str, email: Optional[str] = None, username: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def set_tag(self, key: str, value: Any) -> None:
        pass
    
    def flush(self) -> None:
        pass