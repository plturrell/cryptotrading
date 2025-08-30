"""
Simplified monitoring system for Vercel deployment
Replaces complex OpenTelemetry with lightweight logging and metrics
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional


class VercelLogger:
    """Simple logger for Vercel deployment"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Only add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str, extra: Optional[Dict] = None):
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        self.logger.info(message)

    def warning(self, message: str, extra: Optional[Dict] = None):
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        self.logger.warning(message)

    def error(self, message: str, extra: Optional[Dict] = None):
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        self.logger.error(message)

    def debug(self, message: str, extra: Optional[Dict] = None):
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        self.logger.debug(message)


class VercelMetrics:
    """Simple metrics tracking for Vercel"""

    def __init__(self):
        self.counters = {}
        self.timers = {}
        self.gauges = {}

    def counter(self, name: str, value: int = 1, labels: Optional[Dict] = None):
        """Track counter metrics"""
        key = f"{name}:{labels}" if labels else name
        self.counters[key] = self.counters.get(key, 0) + value

        # Log to Vercel for observability
        print(f"METRIC COUNTER {name}={self.counters[key]} labels={labels}")

    def timer(self, name: str, value: float, labels: Optional[Dict] = None):
        """Track timing metrics"""
        key = f"{name}:{labels}" if labels else name
        if key not in self.timers:
            self.timers[key] = []
        self.timers[key].append(value)

        print(f"METRIC TIMER {name}={value}ms labels={labels}")

    def gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Track gauge metrics"""
        key = f"{name}:{labels}" if labels else name
        self.gauges[key] = value

        print(f"METRIC GAUGE {name}={value} labels={labels}")

    def histogram(self, name: str, value: float, labels: Optional[Dict] = None):
        """Track histogram metrics (simplified as timer)"""
        self.timer(name, value, labels)


class VercelTracer:
    """Simple tracing for Vercel"""

    def __init__(self):
        self.spans = {}

    @contextmanager
    def span(self, name: str):
        """Create a span for tracing"""
        span_id = f"{name}_{int(time.time() * 1000)}"
        start_time = time.time()

        span_data = {"name": name, "start_time": start_time, "attributes": {}}

        self.spans[span_id] = span_data

        class SpanContext:
            def __init__(self, span_data):
                self._span_data = span_data

            def set_attribute(self, key: str, value: Any):
                self._span_data["attributes"][key] = value

        try:
            yield SpanContext(span_data)
        finally:
            duration = (time.time() - start_time) * 1000
            print(
                f"TRACE SPAN {name} duration={duration:.2f}ms attributes={span_data['attributes']}"
            )


# Global instances for Vercel compatibility
_loggers = {}
_metrics = VercelMetrics()
_tracer = VercelTracer()


def get_logger(name: str) -> VercelLogger:
    """Get or create a logger"""
    if name not in _loggers:
        _loggers[name] = VercelLogger(name)
    return _loggers[name]


def get_business_metrics() -> VercelMetrics:
    """Get the global metrics instance"""
    return _metrics


def trace_context(name: str):
    """Create a trace context"""
    return _tracer.span(name)


# Compatibility functions for existing code
def get_metrics():
    """Compatibility function"""
    return _metrics


def get_tracer():
    """Compatibility function"""
    return _tracer


class SimpleErrorTracker:
    """Simple error tracking for Vercel"""

    def __init__(self):
        self.errors = []

    def track_error(self, error: Exception, context: Optional[Dict] = None):
        """Track an error"""
        error_data = {
            "error": str(error),
            "type": type(error).__name__,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }

        self.errors.append(error_data)

        # Log to Vercel
        print(f"ERROR TRACKED {error_data}")

        # Keep only last 100 errors in memory
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]


error_tracker = SimpleErrorTracker()


def track_error(error: Exception, context: Optional[Dict] = None):
    """Track an error globally"""
    error_tracker.track_error(error, context)
