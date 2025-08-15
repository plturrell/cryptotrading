"""
Rex Observability - End-to-End Trace Management and Error Logging
Provides distributed tracing, structured logging, and error aggregation
"""

from .tracer import Tracer, get_tracer, trace_context, trace_function
from .logger import StructuredLogger, get_logger, log_function_call
from .error_tracker import ErrorTracker, track_error, get_error_tracker, ErrorSeverity, ErrorCategory
from .metrics import MetricsCollector, get_metrics, get_business_metrics
from .context import TraceContext, get_current_trace, create_trace_context, with_trace_context, A2AContextEnhancer
from .integration import ObservableWorkflow, observable_agent_method

__all__ = [
    'Tracer', 'get_tracer', 'trace_context', 'trace_function',
    'StructuredLogger', 'get_logger', 'log_function_call',
    'ErrorTracker', 'track_error', 'get_error_tracker', 'ErrorSeverity', 'ErrorCategory',
    'MetricsCollector', 'get_metrics', 'get_business_metrics',
    'TraceContext', 'get_current_trace', 'create_trace_context', 'with_trace_context', 'A2AContextEnhancer',
    'ObservableWorkflow', 'observable_agent_method'
]