"""
Distributed Tracing with OpenTelemetry
Provides end-to-end request tracing across all components
"""

import os
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from functools import wraps
from contextlib import contextmanager
import time
from datetime import datetime
import uuid

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Tracer:
    """Enhanced tracer with automatic error tracking and context propagation"""
    
    def __init__(self, service_name: str = "rex-trading"):
        self.service_name = service_name
        self.tracer = None
        self._initialize_tracing()
        
    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing"""
        # Create resource identifying the service
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            "host.name": os.getenv("HOSTNAME", "localhost")
        })
        
        # Set up the tracer provider
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        
        # Configure exporters based on environment
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            # OTLP exporter for production (e.g., to Datadog, New Relic, etc.)
            otlp_exporter = OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                headers=(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP tracing enabled: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
            
        elif os.getenv("JAEGER_ENDPOINT"):
            # Jaeger for local development
            jaeger_exporter = JaegerExporter(
                agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
                agent_port=int(os.getenv("JAEGER_PORT", "6831"))
            )
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            logger.info("Jaeger tracing enabled")
        else:
            logger.warning("No trace exporter configured. Set OTEL_EXPORTER_OTLP_ENDPOINT or JAEGER_ENDPOINT")
        
        # Set up propagator for distributed tracing
        set_global_textmap(TraceContextTextMapPropagator())
        
        # Auto-instrument HTTP clients
        RequestsInstrumentor().instrument()
        AioHttpClientInstrumentor().instrument()
        
        # Get tracer
        self.tracer = trace.get_tracer(self.service_name, "1.0.0")
    
    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL, 
                   attributes: Optional[Dict[str, Any]] = None) -> trace.Span:
        """Start a new span with attributes"""
        span = self.tracer.start_span(name, kind=kind)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        # Add default attributes
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("span.type", kind.name)
        
        return span
    
    @contextmanager
    def trace_context(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
                     attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing a block of code"""
        span = self.start_span(name, kind, attributes)
        
        try:
            with trace.use_span(span, end_on_exit=True):
                # Add trace ID to logging context
                trace_id = format(span.get_span_context().trace_id, '032x')
                span_id = format(span.get_span_context().span_id, '016x')
                
                # Store in context for logging
                import contextvars
                trace_context = contextvars.ContextVar('trace_context')
                token = trace_context.set({
                    'trace_id': trace_id,
                    'span_id': span_id,
                    'service': self.service_name
                })
                
                try:
                    yield span
                finally:
                    trace_context.reset(token)
                    
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
    
    def trace_function(self, name: Optional[str] = None, 
                      kind: SpanKind = SpanKind.INTERNAL) -> Callable[[T], T]:
        """Decorator to trace function execution"""
        def decorator(func: T) -> T:
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.trace_context(span_name, kind) as span:
                    # Add function arguments as span attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("function.duration_ms", 
                                         (time.time() - start_time) * 1000)
                        return result
                    except Exception as e:
                        span.set_attribute("function.error", True)
                        span.set_attribute("function.error_type", type(e).__name__)
                        raise
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.trace_context(span_name, kind) as span:
                    # Add function arguments as span attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("function.duration_ms", 
                                         (time.time() - start_time) * 1000)
                        return result
                    except Exception as e:
                        span.set_attribute("function.error", True)
                        span.set_attribute("function.error_type", type(e).__name__)
                        raise
            
            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    def trace_a2a_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Add tracing context to A2A messages"""
        current_span = trace.get_current_span()
        
        if current_span.is_recording():
            trace_context = {
                'trace_id': format(current_span.get_span_context().trace_id, '032x'),
                'span_id': format(current_span.get_span_context().span_id, '016x'),
                'flags': current_span.get_span_context().trace_flags
            }
            
            # Add to message metadata
            if 'metadata' not in message:
                message['metadata'] = {}
            message['metadata']['trace_context'] = trace_context
            
        return message
    
    def extract_trace_context(self, message: Dict[str, Any]) -> Optional[trace.SpanContext]:
        """Extract trace context from A2A message"""
        if 'metadata' in message and 'trace_context' in message['metadata']:
            ctx = message['metadata']['trace_context']
            
            try:
                return trace.SpanContext(
                    trace_id=int(ctx['trace_id'], 16),
                    span_id=int(ctx['span_id'], 16),
                    is_remote=True,
                    trace_flags=trace.TraceFlags(ctx.get('flags', 0))
                )
            except (ValueError, KeyError):
                logger.warning("Invalid trace context in message")
                return None
        
        return None
    
    def create_child_span(self, name: str, parent_context: Optional[trace.SpanContext] = None,
                         kind: SpanKind = SpanKind.INTERNAL) -> trace.Span:
        """Create a child span with optional parent context"""
        if parent_context:
            ctx = trace.set_span_in_context(trace.NonRecordingSpan(parent_context))
            return self.tracer.start_span(name, context=ctx, kind=kind)
        else:
            return self.start_span(name, kind)

# Global tracer instance
_tracer = None

def get_tracer(service_name: Optional[str] = None) -> Tracer:
    """Get global tracer instance"""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name or "rex-trading")
    return _tracer

def trace_context(name: str, **kwargs):
    """Convenience function for trace context"""
    return get_tracer().trace_context(name, **kwargs)

def trace_function(name: Optional[str] = None, **kwargs):
    """Convenience decorator for function tracing"""
    return get_tracer().trace_function(name, **kwargs)

# Auto-instrument Flask if available
try:
    from flask import Flask
    def instrument_flask_app(app: Flask):
        """Instrument Flask application for tracing"""
        FlaskInstrumentor().instrument_app(app)
        logger.info("Flask application instrumented for tracing")
except ImportError:
    pass