"""
Lightweight monitoring implementation for Vercel and serverless environments
"""

import os
import time
import json
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict, deque

from .base import (
    MonitoringInterface, SpanContext, MetricType, LogLevel,
    NoOpSpanContext
)


class LightweightSpanContext(SpanContext):
    """Lightweight span context for basic tracing"""
    
    def __init__(self, name: str, monitor: 'LightweightMonitoring'):
        self.name = name
        self.monitor = monitor
        self.start_time = time.time()
        self.attributes = {}
        self.events = []
        self._ended = False
    
    def set_attribute(self, key: str, value: Any) -> None:
        if not self._ended:
            self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        if not self._ended:
            self.events.append({
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {}
            })
    
    def record_exception(self, exception: Exception) -> None:
        if not self._ended:
            self.add_event("exception", {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception)
            })
    
    def end(self) -> None:
        if not self._ended:
            duration = time.time() - self.start_time
            
            # Log span completion
            self.monitor.log_info(f"Span completed: {self.name}", {
                "span_name": self.name,
                "duration_ms": duration * 1000,
                "attributes": self.attributes,
                "events": self.events
            })
            
            # Record duration metric
            self.monitor.record_histogram(f"span.duration.{self.name}", duration * 1000)
            
            self._ended = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.record_exception(exc_val)
        self.end()


class LightweightMonitoring(MonitoringInterface):
    """Lightweight monitoring for serverless environments"""
    
    def __init__(self, service_name: str = "cryptotrading", environment: str = "production"):
        self.service_name = service_name
        self.environment = environment
        self.global_tags = {
            "service": service_name,
            "environment": environment
        }
        
        # Setup logging
        self._setup_logging()
        
        # In-memory metrics storage (for the lifetime of the function)
        self._metrics = defaultdict(lambda: defaultdict(float))
        self._breadcrumbs = deque(maxlen=100)  # Keep last 100 breadcrumbs
        self._user_context = {}
    
    def _setup_logging(self):
        """Setup simple JSON logging"""
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create simple handler
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_formatter())
        self.logger.addHandler(handler)
    
    def _get_formatter(self):
        """Get formatter for structured logging"""
        monitor = self  # Capture reference to the monitor
        
        class SimpleJSONFormatter(logging.Formatter):
            def format(self, record):
                # Create base log entry
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "service": monitor.service_name,
                    "environment": monitor.environment
                }
                
                # Add extra fields if present
                if hasattr(record, 'extra_fields'):
                    log_entry.update(record.extra_fields)
                
                # Add exception info if present
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                
                return json.dumps(log_entry, default=str)
        
        return SimpleJSONFormatter()
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.COUNTER, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        tags = {**self.global_tags, **(tags or {})}
        tag_str = json.dumps(tags, sort_keys=True)
        
        if metric_type == MetricType.COUNTER:
            self._metrics[name][tag_str] += value
        elif metric_type == MetricType.GAUGE:
            self._metrics[name][tag_str] = value
        elif metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
            # For simplicity, just log the value
            self.log_info(f"Metric: {name}", {
                "metric_name": name,
                "metric_value": value,
                "metric_type": metric_type.value,
                "tags": tags
            })
    
    def log(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a message with structured data"""
        extra_fields = {
            **self.global_tags,
            **self._user_context,
            **(extra or {})
        }
        
        # Add breadcrumbs to error logs
        if level in (LogLevel.ERROR, LogLevel.CRITICAL) and self._breadcrumbs:
            extra_fields["breadcrumbs"] = list(self._breadcrumbs)
        
        log_level = getattr(logging, level.value.upper())
        
        # Create custom log record
        record = logging.LogRecord(
            name=self.logger.name,
            level=log_level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra_fields
        
        self.logger.handle(record)
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> SpanContext:
        """Start a new tracing span"""
        span = LightweightSpanContext(name, self)
        
        # Set initial attributes
        for key, value in self.global_tags.items():
            span.set_attribute(key, value)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return span
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error with context"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context or {},
            "breadcrumbs": list(self._breadcrumbs)
        }
        
        self.log(LogLevel.ERROR, f"{type(error).__name__}: {str(error)}", error_data)
        
        # Increment error counter
        self.increment_counter("errors", tags={"error_type": type(error).__name__})
    
    def add_breadcrumb(self, message: str, category: str = "custom", level: LogLevel = LogLevel.INFO, data: Optional[Dict[str, Any]] = None) -> None:
        """Add a breadcrumb for debugging"""
        breadcrumb = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "category": category,
            "level": level.value,
            "data": data or {}
        }
        
        self._breadcrumbs.append(breadcrumb)
    
    def set_user_context(self, user_id: str, email: Optional[str] = None, username: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Set user context for monitoring"""
        self._user_context = {
            "user_id": user_id
        }
        
        if email:
            self._user_context["user_email"] = email
        if username:
            self._user_context["user_username"] = username
        if extra:
            self._user_context.update(extra)
        
        # Update global tags
        self.global_tags.update(self._user_context)
    
    def set_tag(self, key: str, value: Any) -> None:
        """Set a global tag for all monitoring data"""
        self.global_tags[key] = value
    
    def flush(self) -> None:
        """Flush any pending monitoring data"""
        # Log aggregated metrics
        if self._metrics:
            for metric_name, values in self._metrics.items():
                for tag_str, value in values.items():
                    tags = json.loads(tag_str)
                    self.log_info(f"Aggregated metric: {metric_name}", {
                        "metric_name": metric_name,
                        "metric_value": value,
                        "tags": tags
                    })
        
        # Clear metrics after flush
        self._metrics.clear()
    
    @contextmanager
    def suppress_monitoring(self):
        """Context manager to temporarily suppress monitoring"""
        original_level = self.logger.level
        self.logger.setLevel(logging.CRITICAL + 1)
        try:
            yield
        finally:
            self.logger.setLevel(original_level)