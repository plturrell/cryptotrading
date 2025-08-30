"""
Simplified Monitoring for Vercel Deployment
Lightweight observability without heavy dependencies
"""

# Try to import Vercel-compatible monitoring first
try:
    from ...core.monitoring.vercel_monitoring import (
        get_business_metrics,
        get_logger,
        trace_context,
        track_error,
    )

    MONITORING_TYPE = "vercel"
except ImportError:
    # Fallback to original monitoring if available
    try:
        from .context import (
            A2AContextEnhancer,
            TraceContext,
            create_trace_context,
            get_current_trace,
            with_trace_context,
        )
        from .error_tracker import (
            ErrorCategory,
            ErrorSeverity,
            ErrorTracker,
            get_error_tracker,
            track_error,
        )
        from .integration import ObservableWorkflow, observable_agent_method
        from .logger import StructuredLogger, get_logger, log_function_call
        from .metrics import MetricsCollector, get_business_metrics, get_metrics
        from .tracer import Tracer, get_tracer, trace_context, trace_function

        MONITORING_TYPE = "full"
    except ImportError:
        # Final fallback - basic logging
        import logging

        def get_logger(name: str):
            return logging.getLogger(name)

        class DummyMetrics:
            def counter(self, *args, **kwargs):
                pass

            def gauge(self, *args, **kwargs):
                pass

            def histogram(self, *args, **kwargs):
                pass

        def get_business_metrics():
            return DummyMetrics()

        def trace_context(name: str):
            from contextlib import nullcontext

            return nullcontext()

        def track_error(error, context=None):
            logging.error(f"Error tracked: {error}")

        MONITORING_TYPE = "fallback"

__all__ = ["get_logger", "get_business_metrics", "trace_context", "track_error", "MONITORING_TYPE"]
