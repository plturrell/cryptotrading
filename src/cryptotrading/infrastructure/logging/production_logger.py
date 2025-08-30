"""
Production-grade structured logging system
Supports JSON formatting, centralized logging, and observability integration
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger


class ProductionFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context"""

    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict):
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["thread"] = threading.current_thread().name
        log_record["process"] = os.getpid()

        # Add application context
        log_record["service"] = "reks-a2a"
        log_record["version"] = "1.0.0"
        log_record["environment"] = os.getenv("ENVIRONMENT", "development")

        # Add trace context if available
        if hasattr(record, "trace_id"):
            log_record["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_record["span_id"] = record.span_id

        # Add user context if available
        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id
        if hasattr(record, "agent_id"):
            log_record["agent_id"] = record.agent_id

        # Add request context if available
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "correlation_id"):
            log_record["correlation_id"] = record.correlation_id

        # Add performance metrics if available
        if hasattr(record, "duration"):
            log_record["duration"] = record.duration
        if hasattr(record, "memory_usage"):
            log_record["memory_usage"] = record.memory_usage


class StructuredLogger:
    """Production structured logger with context management"""

    def __init__(self, name: str = None):
        self.name = name or __name__
        self.logger = logging.getLogger(self.name)
        self._context = {}

    def set_context(self, **kwargs):
        """Set logging context for this logger"""
        self._context.update(kwargs)

    def clear_context(self):
        """Clear logging context"""
        self._context.clear()

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context"""
        # Merge context with kwargs
        extra = {**self._context, **kwargs}

        # Create log record
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Info level logging"""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Warning level logging"""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, error: Exception = None, **kwargs):
        """Error level logging with exception details"""
        if error:
            kwargs.update(
                {
                    "error_type": error.__class__.__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )

        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Critical level logging"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def metric(self, metric_name: str, value: float, tags: Dict[str, str] = None, **kwargs):
        """Log metrics in structured format"""
        metric_data = {
            "metric_name": metric_name,
            "metric_value": value,
            "metric_type": "gauge",
            "metric_tags": tags or {},
        }
        metric_data.update(kwargs)

        self._log_with_context(logging.INFO, f"METRIC: {metric_name}", **metric_data)

    def audit(self, action: str, resource: str = None, user_id: str = None, **kwargs):
        """Audit logging for security events"""
        audit_data = {
            "audit_action": action,
            "audit_resource": resource,
            "audit_user": user_id,
            "audit_timestamp": datetime.utcnow().isoformat(),
        }
        audit_data.update(kwargs)

        self._log_with_context(logging.INFO, f"AUDIT: {action}", **audit_data)

    def performance(self, operation: str, duration: float, **kwargs):
        """Performance logging"""
        perf_data = {
            "performance_operation": operation,
            "performance_duration": duration,
            "performance_unit": "seconds",
        }
        perf_data.update(kwargs)

        self._log_with_context(logging.INFO, f"PERFORMANCE: {operation}", **perf_data)


class LoggingConfig:
    """Logging configuration for production"""

    # Log levels
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # File logging
    LOG_DIR = os.getenv("LOG_DIR", "/var/log/reks")
    LOG_FILE = os.getenv("LOG_FILE", "reks-a2a.log")
    MAX_LOG_SIZE = int(os.getenv("MAX_LOG_SIZE", "100")) * 1024 * 1024  # 100MB
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "10"))

    # Console logging
    CONSOLE_LOGGING = os.getenv("CONSOLE_LOGGING", "true").lower() == "true"

    # Centralized logging
    SYSLOG_HOST = os.getenv("SYSLOG_HOST")
    SYSLOG_PORT = int(os.getenv("SYSLOG_PORT", "514"))

    # ELK Stack integration
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST")
    ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "reks-logs")


def setup_production_logging(config: LoggingConfig = None) -> StructuredLogger:
    """Setup production logging configuration"""
    config = config or LoggingConfig()

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = ProductionFormatter(fmt="%(timestamp)s %(level)s %(logger)s %(message)s")

    # Console handler
    if config.CONSOLE_LOGGING:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    try:
        log_dir = Path(config.LOG_DIR)
        log_dir.mkdir(exist_ok=True, parents=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / config.LOG_FILE,
            maxBytes=config.MAX_LOG_SIZE,
            backupCount=config.LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    except Exception as e:
        print(f"Failed to setup file logging: {e}", file=sys.stderr)

    # Syslog handler for centralized logging
    if config.SYSLOG_HOST:
        try:
            syslog_handler = logging.handlers.SysLogHandler(
                address=(config.SYSLOG_HOST, config.SYSLOG_PORT)
            )
            syslog_handler.setFormatter(formatter)
            root_logger.addHandler(syslog_handler)

        except Exception as e:
            print(f"Failed to setup syslog: {e}", file=sys.stderr)

    # Elasticsearch handler (if available)
    if config.ELASTICSEARCH_HOST:
        try:
            from python_elasticsearch_logger import ElasticsearchHandler

            es_handler = ElasticsearchHandler(
                hosts=[{"host": config.ELASTICSEARCH_HOST, "port": config.ELASTICSEARCH_PORT}],
                auth_type=ElasticsearchHandler.AuthType.NO_AUTH,
                es_index_name=config.ELASTICSEARCH_INDEX,
            )
            es_handler.setFormatter(formatter)
            root_logger.addHandler(es_handler)

        except ImportError:
            print(
                "Elasticsearch logging not available (python-elasticsearch-logger not installed)",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Failed to setup Elasticsearch logging: {e}", file=sys.stderr)

    # Create main application logger
    app_logger = StructuredLogger("reks.a2a")
    app_logger.info(
        "Production logging configured",
        log_level=config.LOG_LEVEL,
        handlers=len(root_logger.handlers),
    )

    return app_logger


class ContextManager:
    """Thread-local context manager for distributed tracing"""

    def __init__(self):
        self._local = threading.local()

    def set_context(self, **kwargs):
        """Set context for current thread"""
        if not hasattr(self._local, "context"):
            self._local.context = {}
        self._local.context.update(kwargs)

    def get_context(self) -> Dict[str, Any]:
        """Get context for current thread"""
        if not hasattr(self._local, "context"):
            self._local.context = {}
        return self._local.context.copy()

    def clear_context(self):
        """Clear context for current thread"""
        if hasattr(self._local, "context"):
            self._local.context.clear()

    def with_context(self, **kwargs):
        """Context manager for temporary context"""

        class TemporaryContext:
            def __init__(self, context_manager, context_kwargs):
                self.context_manager = context_manager
                self.context_kwargs = context_kwargs
                self.original_context = None

            def __enter__(self):
                self.original_context = self.context_manager.get_context()
                self.context_manager.set_context(**self.context_kwargs)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.context_manager.clear_context()
                self.context_manager.set_context(**self.original_context)

        return TemporaryContext(self, kwargs)


def get_logger(name: str = None) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)


def log_function_call(logger: StructuredLogger = None):
    """Decorator to log function calls with parameters and performance"""

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            function_name = f"{func.__module__}.{func.__name__}"

            # Log function entry
            logger.debug(
                f"Entering {function_name}",
                function=function_name,
                args_count=len(args),
                kwargs_count=len(kwargs),
            )

            try:
                result = func(*args, **kwargs)

                # Calculate duration
                duration = (datetime.utcnow() - start_time).total_seconds()

                # Log successful completion
                logger.debug(
                    f"Completed {function_name}",
                    function=function_name,
                    duration=duration,
                    success=True,
                )

                return result

            except Exception as e:
                # Calculate duration
                duration = (datetime.utcnow() - start_time).total_seconds()

                # Log error
                logger.error(
                    f"Failed {function_name}",
                    error=e,
                    function=function_name,
                    duration=duration,
                    success=False,
                )
                raise

        return wrapper

    return decorator


def log_agent_operation(agent_id: str, operation: str):
    """Decorator to log A2A agent operations"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger("reks.a2a.agent")
            logger.set_context(agent_id=agent_id)

            start_time = datetime.utcnow()

            logger.info(
                f"Agent operation started: {operation}", operation=operation, agent_id=agent_id
            )

            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()

                logger.info(
                    f"Agent operation completed: {operation}",
                    operation=operation,
                    agent_id=agent_id,
                    duration=duration,
                    success=True,
                )

                return result

            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()

                logger.error(
                    f"Agent operation failed: {operation}",
                    error=e,
                    operation=operation,
                    agent_id=agent_id,
                    duration=duration,
                    success=False,
                )
                raise
            finally:
                logger.clear_context()

        return wrapper

    return decorator


# Global instances
context_manager = ContextManager()
production_logger = None


def initialize_logging():
    """Initialize production logging system"""
    global production_logger
    production_logger = setup_production_logging()
    return production_logger


# Auto-initialize if imported
if production_logger is None:
    production_logger = initialize_logging()
