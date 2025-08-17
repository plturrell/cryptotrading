"""
Full monitoring implementation using OpenTelemetry and structured logging
"""

import os
import time
import logging
import json
from typing import Any, Dict, Optional, List
from datetime import datetime
from contextlib import contextmanager

from .base import (
    MonitoringInterface, SpanContext, MetricType, LogLevel,
    NoOpSpanContext
)

# Try to import OpenTelemetry components
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class FullSpanContext(SpanContext):
    """Full span context implementation using OpenTelemetry"""
    
    def __init__(self, span):
        self.span = span
        self._ended = False
    
    def set_attribute(self, key: str, value: Any) -> None:
        if not self._ended and self.span:
            self.span.set_attribute(key, value)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        if not self._ended and self.span:
            self.span.add_event(name, attributes or {})
    
    def record_exception(self, exception: Exception) -> None:
        if not self._ended and self.span:
            self.span.record_exception(exception)
            self.span.set_status(Status(StatusCode.ERROR, str(exception)))
    
    def end(self) -> None:
        if not self._ended and self.span:
            self.span.end()
            self._ended = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.record_exception(exc_val)
        self.end()


class FullMonitoring(MonitoringInterface):
    """Full monitoring implementation with OpenTelemetry and structured logging"""
    
    def __init__(self, service_name: str = "cryptotrading", environment: str = "development"):
        self.service_name = service_name
        self.environment = environment
        self.global_tags = {
            "service": service_name,
            "environment": environment
        }
        
        # Setup structured logging
        self._setup_logging()
        
        # Setup OpenTelemetry if available
        if OTEL_AVAILABLE:
            self._setup_opentelemetry()
        else:
            self.tracer = None
            self.meter = None
            self.logger.warning("OpenTelemetry not available, using logging-only mode")
        
        # Metrics cache for aggregation
        self._metrics_cache = {}
        self._last_flush = time.time()
    
    def _setup_logging(self):
        """Setup structured JSON logging"""
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_json_formatter())
        self.logger.addHandler(handler)
    
    def _get_json_formatter(self):
        """Get JSON formatter for structured logging"""
        monitor = self  # Capture reference to the monitor
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "service": monitor.service_name,
                    "environment": monitor.environment,
                    "logger": record.name,
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add extra fields
                if hasattr(record, 'extra_fields'):
                    log_data.update(record.extra_fields)
                
                # Add exception info if present
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                
                return json.dumps(log_data)
        
        return JSONFormatter()
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing and metrics"""
        # Setup tracing
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # Add OTLP exporter if endpoint is configured
        otlp_endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
        if otlp_endpoint:
            span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        
        self.tracer = trace.get_tracer(self.service_name)
        
        # Setup metrics
        if otlp_endpoint:
            metric_reader = PeriodicExportingMetricReader(
                exporter=OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True),
                export_interval_millis=60000  # Export every minute
            )
            metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
        
        self.meter = metrics.get_meter(self.service_name)
        
        # Create metric instruments
        self._setup_metric_instruments()
    
    def _setup_metric_instruments(self):
        """Setup metric instruments"""
        if not self.meter:
            return
        
        self._counters = {}
        self._gauges = {}
        self._histograms = {}
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.COUNTER, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        tags = {**self.global_tags, **(tags or {})}
        
        # Log the metric
        self.logger.info(f"Metric: {name}", extra={
            "extra_fields": {
                "metric_name": name,
                "metric_value": value,
                "metric_type": metric_type.value,
                "tags": tags
            }
        })
        
        # Record in OpenTelemetry if available
        if self.meter:
            if metric_type == MetricType.COUNTER:
                if name not in self._counters:
                    self._counters[name] = self.meter.create_counter(name)
                self._counters[name].add(value, tags)
            
            elif metric_type == MetricType.GAUGE:
                # Store gauge value for callback
                self._metrics_cache[f"gauge_{name}"] = (value, tags)
                if name not in self._gauges:
                    def gauge_callback(options):
                        cache_key = f"gauge_{options.name}"
                        if cache_key in self._metrics_cache:
                            value, tags = self._metrics_cache[cache_key]
                            yield metrics.Observation(value, tags)
                    
                    self._gauges[name] = self.meter.create_observable_gauge(
                        name,
                        callbacks=[lambda options: gauge_callback(options)]
                    )
            
            elif metric_type == MetricType.HISTOGRAM:
                if name not in self._histograms:
                    self._histograms[name] = self.meter.create_histogram(name)
                self._histograms[name].record(value, tags)
    
    def log(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a message with structured data"""
        extra_fields = {**self.global_tags, **(extra or {})}
        
        log_record = logging.LogRecord(
            name=self.logger.name,
            level=getattr(logging, level.value.upper()),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        log_record.extra_fields = extra_fields
        
        self.logger.handle(log_record)
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> SpanContext:
        """Start a new tracing span"""
        if not self.tracer:
            return NoOpSpanContext()
        
        span = self.tracer.start_span(name)
        
        # Set attributes
        for key, value in self.global_tags.items():
            span.set_attribute(key, value)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return FullSpanContext(span)
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error with context"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.log(LogLevel.ERROR, f"Error: {type(error).__name__}: {str(error)}", error_data)
        
        # Also increment error counter
        self.increment_counter("errors", tags={"error_type": type(error).__name__})
    
    def add_breadcrumb(self, message: str, category: str = "custom", level: LogLevel = LogLevel.INFO, data: Optional[Dict[str, Any]] = None) -> None:
        """Add a breadcrumb for debugging"""
        breadcrumb_data = {
            "breadcrumb": {
                "message": message,
                "category": category,
                "level": level.value,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data or {}
            }
        }
        
        self.log(level, f"Breadcrumb: {message}", breadcrumb_data)
    
    def set_user_context(self, user_id: str, email: Optional[str] = None, username: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Set user context for monitoring"""
        user_context = {
            "user_id": user_id,
            "email": email,
            "username": username,
            **(extra or {})
        }
        
        # Update global tags
        self.global_tags["user_id"] = user_id
        if email:
            self.global_tags["user_email"] = email
        if username:
            self.global_tags["user_username"] = username
        
        self.log(LogLevel.INFO, f"User context set for {user_id}", {"user_context": user_context})
    
    def set_tag(self, key: str, value: Any) -> None:
        """Set a global tag for all monitoring data"""
        self.global_tags[key] = value
    
    def flush(self) -> None:
        """Flush any pending monitoring data"""
        self.logger.info("Flushing monitoring data")
        
        # Force flush OpenTelemetry if available
        if OTEL_AVAILABLE:
            trace.get_tracer_provider().force_flush()
            
        self._last_flush = time.time()
    
    def instrument_flask_app(self, app):
        """Instrument a Flask application"""
        if OTEL_AVAILABLE and FlaskInstrumentor:
            FlaskInstrumentor().instrument_app(app)
            self.logger.info("Flask app instrumented with OpenTelemetry")