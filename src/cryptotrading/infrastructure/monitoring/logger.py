"""
Structured Logging with Trace Context
Provides JSON-formatted logs with automatic trace correlation
"""

import contextvars
import json
import logging
import sys
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Union

from pythonjsonlogger import jsonlogger

# Context variable for trace information
trace_context = contextvars.ContextVar("trace_context", default={})


class StructuredLogger:
    """Enhanced logger with structured output and trace correlation"""

    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # Create JSON formatter
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s", json_ensure_ascii=False
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler for errors
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            error_handler = logging.FileHandler("logs/errors.log", mode="a")
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add trace context and standard fields to log entry"""
        # Get trace context
        ctx = trace_context.get()

        # Build enhanced extra
        enhanced = {
            "service": self.name,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "hostname": os.getenv("HOSTNAME", "localhost"),
        }

        # Add trace context if available
        if ctx:
            enhanced.update(
                {
                    "trace_id": ctx.get("trace_id"),
                    "span_id": ctx.get("span_id"),
                    "service_name": ctx.get("service"),
                }
            )

        # Add user-provided extra
        if extra:
            enhanced.update(extra)

        return enhanced

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with context"""
        self.logger.debug(message, extra=self._add_context(extra))

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with context"""
        self.logger.info(message, extra=self._add_context(extra))

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with context"""
        self.logger.warning(message, extra=self._add_context(extra))

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log error message with exception details"""
        context_extra = self._add_context(extra or {})

        if error:
            context_extra.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_traceback": traceback.format_exc(),
                }
            )

        self.logger.error(message, extra=context_extra)

    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log critical message with exception details"""
        context_extra = self._add_context(extra or {})

        if error:
            context_extra.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_traceback": traceback.format_exc(),
                }
            )

        self.logger.critical(message, extra=context_extra)

    def log_event(self, event_type: str, event_data: Dict[str, Any], level: str = "INFO"):
        """Log structured event"""
        message = f"Event: {event_type}"
        extra = {"event_type": event_type, "event_data": event_data}

        log_method = getattr(self, level.lower())
        log_method(message, extra=extra)

    def log_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: str = "count",
        tags: Optional[Dict[str, str]] = None,
    ):
        """Log metric data"""
        self.info(
            f"Metric: {metric_name}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                "metric_tags": tags or {},
            },
        )

    def log_api_request(
        self,
        method: str,
        url: str,
        status_code: int,
        duration_ms: float,
        request_id: Optional[str] = None,
    ):
        """Log API request details"""
        self.info(
            f"API Request: {method} {url}",
            extra={
                "api_method": method,
                "api_url": url,
                "api_status_code": status_code,
                "api_duration_ms": duration_ms,
                "api_request_id": request_id,
                "api_success": 200 <= status_code < 300,
            },
        )

    def log_database_query(
        self, query_type: str, table: str, duration_ms: float, rows_affected: int = 0
    ):
        """Log database query details"""
        self.info(
            f"Database Query: {query_type} {table}",
            extra={
                "db_query_type": query_type,
                "db_table": table,
                "db_duration_ms": duration_ms,
                "db_rows_affected": rows_affected,
            },
        )


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional processing"""

    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record"""
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["level"] = record.levelname
        log_record["logger_name"] = record.name

        # Add source location
        log_record["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Clean up message
        if "message" in log_record:
            log_record["message"] = log_record["message"].strip()


# Logger cache
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """Get or create structured logger"""
    if name not in _loggers:
        # Create logs directory if it doesn't exist
        import os

        os.makedirs("logs", exist_ok=True)

        _loggers[name] = StructuredLogger(name, level)

    return _loggers[name]


def log_function_call(logger: Optional[StructuredLogger] = None):
    """Decorator to log function calls with arguments and results"""

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            # Log function entry
            logger.debug(
                f"Calling {func.__name__}",
                extra={
                    "function_name": func.__name__,
                    "function_args": str(args)[:200],  # Truncate long args
                    "function_kwargs": str(kwargs)[:200],
                },
            )

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # Log success
                logger.debug(
                    f"Completed {func.__name__}",
                    extra={
                        "function_name": func.__name__,
                        "function_duration_ms": duration_ms,
                        "function_success": True,
                    },
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # Log error
                logger.error(
                    f"Error in {func.__name__}",
                    error=e,
                    extra={
                        "function_name": func.__name__,
                        "function_duration_ms": duration_ms,
                        "function_success": False,
                    },
                )
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import time

            start_time = time.time()

            # Log function entry
            logger.debug(
                f"Calling {func.__name__}",
                extra={
                    "function_name": func.__name__,
                    "function_args": str(args)[:200],
                    "function_kwargs": str(kwargs)[:200],
                },
            )

            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # Log success
                logger.debug(
                    f"Completed {func.__name__}",
                    extra={
                        "function_name": func.__name__,
                        "function_duration_ms": duration_ms,
                        "function_success": True,
                    },
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # Log error
                logger.error(
                    f"Error in {func.__name__}",
                    error=e,
                    extra={
                        "function_name": func.__name__,
                        "function_duration_ms": duration_ms,
                        "function_success": False,
                    },
                )
                raise

        # Return appropriate wrapper
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


import os

# Import time at module level
import time
