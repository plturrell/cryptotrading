"""
OpenTelemetry Distributed Tracing for MCP Tools
Provides comprehensive tracing across all MCP tool executions
"""

import asyncio
import functools
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from opentelemetry import context, propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)


class MCPTracingSystem:
    """Distributed tracing system for MCP tools"""

    def __init__(self, service_name: str = "mcp-tools", environment: str = "production"):
        self.service_name = service_name
        self.environment = environment
        self.tracer = None
        self.initialized = False
        self.span_processors = []

        # Metrics for tracing
        self.trace_metrics = {
            "spans_created": 0,
            "spans_exported": 0,
            "errors_traced": 0,
            "active_spans": 0,
        }

    def initialize(
        self,
        otlp_endpoint: str = "localhost:4317",
        enable_console: bool = False,
        enable_otlp: bool = True,
    ):
        """Initialize OpenTelemetry tracing"""
        try:
            # Create resource
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": "1.0.0",
                    "deployment.environment": self.environment,
                    "telemetry.sdk.language": "python",
                    "telemetry.sdk.name": "opentelemetry",
                    "host.name": "mcp-server",
                }
            )

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Add console exporter if enabled
            if enable_console:
                console_exporter = ConsoleSpanExporter()
                console_processor = BatchSpanProcessor(console_exporter)
                provider.add_span_processor(console_processor)
                self.span_processors.append(console_processor)

            # Add OTLP exporter if enabled
            if enable_otlp:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint, insecure=True  # Use secure=False for local development
                )
                otlp_processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(otlp_processor)
                self.span_processors.append(otlp_processor)

            # Set global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer
            self.tracer = trace.get_tracer(
                instrumenting_module_name=__name__, instrumenting_library_version="1.0.0"
            )

            # Instrument libraries
            RequestsInstrumentor().instrument()
            AsyncioInstrumentor().instrument()

            # Set propagator
            propagate.set_global_textmap(TraceContextTextMapPropagator())

            self.initialized = True
            logger.info(f"OpenTelemetry tracing initialized for {self.service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.initialized = False

    def create_span(self, name: str, kind: trace.SpanKind = trace.SpanKind.INTERNAL) -> trace.Span:
        """Create a new span"""
        if not self.tracer:
            return None

        span = self.tracer.start_span(
            name=name,
            kind=kind,
            attributes={"mcp.service": self.service_name, "mcp.environment": self.environment},
        )

        self.trace_metrics["spans_created"] += 1
        self.trace_metrics["active_spans"] += 1

        return span

    def trace_tool_execution(self, tool_name: str, method_name: str):
        """Decorator for tracing MCP tool execution"""

        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.tracer:
                    return await func(*args, **kwargs)

                # Create span
                with self.tracer.start_as_current_span(
                    f"mcp.tool.{tool_name}.{method_name}", kind=trace.SpanKind.INTERNAL
                ) as span:
                    # Add attributes
                    span.set_attributes(
                        {
                            "mcp.tool.name": tool_name,
                            "mcp.tool.method": method_name,
                            "mcp.tool.async": True,
                            "mcp.execution.start_time": datetime.now().isoformat(),
                        }
                    )

                    try:
                        # Record parameters
                        if kwargs:
                            span.set_attribute(
                                "mcp.tool.parameters",
                                json.dumps({k: str(v)[:100] for k, v in kwargs.items()}),
                            )

                        # Execute function
                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        execution_time = time.time() - start_time

                        # Record success
                        span.set_attributes(
                            {
                                "mcp.execution.duration_ms": execution_time * 1000,
                                "mcp.execution.success": True,
                            }
                        )
                        span.set_status(Status(StatusCode.OK))

                        return result

                    except Exception as e:
                        # Record error
                        self.trace_metrics["errors_traced"] += 1
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attributes(
                            {"mcp.execution.success": False, "mcp.execution.error": str(e)}
                        )
                        raise

                    finally:
                        self.trace_metrics["active_spans"] -= 1

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.tracer:
                    return func(*args, **kwargs)

                with self.tracer.start_as_current_span(
                    f"mcp.tool.{tool_name}.{method_name}", kind=trace.SpanKind.INTERNAL
                ) as span:
                    span.set_attributes(
                        {
                            "mcp.tool.name": tool_name,
                            "mcp.tool.method": method_name,
                            "mcp.tool.async": False,
                        }
                    )

                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        execution_time = time.time() - start_time

                        span.set_attributes(
                            {
                                "mcp.execution.duration_ms": execution_time * 1000,
                                "mcp.execution.success": True,
                            }
                        )
                        span.set_status(Status(StatusCode.OK))

                        return result

                    except Exception as e:
                        self.trace_metrics["errors_traced"] += 1
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

                    finally:
                        self.trace_metrics["active_spans"] -= 1

            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def trace_agent_communication(self, from_agent: str, to_agent: str):
        """Trace agent-to-agent communication"""

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.tracer:
                    return await func(*args, **kwargs)

                with self.tracer.start_as_current_span(
                    f"a2a.communication.{from_agent}.to.{to_agent}", kind=trace.SpanKind.CLIENT
                ) as span:
                    span.set_attributes(
                        {
                            "a2a.from_agent": from_agent,
                            "a2a.to_agent": to_agent,
                            "a2a.protocol": "MCP",
                            "a2a.timestamp": datetime.now().isoformat(),
                        }
                    )

                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return wrapper

        return decorator

    def create_child_span(self, name: str, parent_span: Optional[trace.Span] = None):
        """Create a child span"""
        if not self.tracer:
            return None

        # Get current span if no parent specified
        if parent_span is None:
            parent_span = trace.get_current_span()

        # Create child span
        ctx = trace.set_span_in_context(parent_span)
        child_span = self.tracer.start_span(name, context=ctx)

        return child_span

    def add_event(self, span: trace.Span, name: str, attributes: Dict[str, Any] = None):
        """Add an event to a span"""
        if span:
            span.add_event(name, attributes=attributes or {})

    def set_span_attributes(self, span: trace.Span, attributes: Dict[str, Any]):
        """Set attributes on a span"""
        if span:
            for key, value in attributes.items():
                # Convert non-primitive types to strings
                if not isinstance(value, (str, int, float, bool)):
                    value = str(value)
                span.set_attribute(key, value)

    def extract_context(self, headers: Dict[str, str]) -> context.Context:
        """Extract trace context from headers"""
        return TraceContextTextMapPropagator().extract(headers)

    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers"""
        TraceContextTextMapPropagator().inject(headers)
        return headers

    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            if span_context.is_valid:
                return format(span_context.trace_id, "032x")
        return None

    def get_span_id(self) -> Optional[str]:
        """Get current span ID"""
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            if span_context.is_valid:
                return format(span_context.span_id, "016x")
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get tracing metrics"""
        return {
            "initialized": self.initialized,
            "service_name": self.service_name,
            "environment": self.environment,
            "metrics": self.trace_metrics,
            "current_trace_id": self.get_trace_id(),
            "current_span_id": self.get_span_id(),
            "timestamp": datetime.now().isoformat(),
        }

    def shutdown(self):
        """Shutdown tracing system"""
        try:
            # Flush and shutdown span processors
            for processor in self.span_processors:
                processor.shutdown()

            logger.info("OpenTelemetry tracing shutdown complete")
        except Exception as e:
            logger.error(f"Error during tracing shutdown: {e}")


class TracedMCPTool:
    """Base class for traced MCP tools"""

    def __init__(self, tool_name: str, tracing_system: MCPTracingSystem):
        self.tool_name = tool_name
        self.tracing_system = tracing_system
        self.execution_count = 0
        self.error_count = 0

    async def execute_with_tracing(self, method_name: str, *args, **kwargs):
        """Execute a method with distributed tracing"""
        self.execution_count += 1

        # Create span for execution
        with self.tracing_system.tracer.start_as_current_span(
            f"mcp.tool.{self.tool_name}.{method_name}", kind=trace.SpanKind.INTERNAL
        ) as span:
            # Set initial attributes
            span.set_attributes(
                {
                    "mcp.tool.name": self.tool_name,
                    "mcp.tool.method": method_name,
                    "mcp.tool.execution_count": self.execution_count,
                    "mcp.execution.timestamp": datetime.now().isoformat(),
                }
            )

            try:
                # Execute the actual method
                method = getattr(self, method_name)
                result = await method(*args, **kwargs)

                # Mark success
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("mcp.execution.success", True)

                return result

            except Exception as e:
                # Record error
                self.error_count += 1
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attributes(
                    {
                        "mcp.execution.success": False,
                        "mcp.execution.error": str(e),
                        "mcp.tool.error_count": self.error_count,
                    }
                )
                raise


# Global tracing instance
_tracing_system = None


def get_tracing_system() -> MCPTracingSystem:
    """Get or create the global tracing system"""
    global _tracing_system

    if _tracing_system is None:
        _tracing_system = MCPTracingSystem()
        _tracing_system.initialize()

    return _tracing_system


def trace_mcp_tool(tool_name: str):
    """Decorator to automatically trace MCP tool methods"""

    def class_decorator(cls):
        tracing = get_tracing_system()

        # Wrap all public methods
        for attr_name in dir(cls):
            if not attr_name.startswith("_"):
                attr = getattr(cls, attr_name)
                if callable(attr):
                    # Apply tracing decorator
                    traced_method = tracing.trace_tool_execution(tool_name, attr_name)(attr)
                    setattr(cls, attr_name, traced_method)

        return cls

    return class_decorator
