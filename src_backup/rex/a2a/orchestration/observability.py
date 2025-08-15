"""
OpenTelemetry Observability for A2A Orchestration
Provides distributed tracing, metrics, and logging
"""

import logging
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager
from functools import wraps

# Try to import OpenTelemetry, fallback to no-op if not available
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # No-op implementations
    class NoOpTracer:
        def start_as_current_span(self, name, **kwargs):
            return NoOpSpan()
    
    class NoOpSpan:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_status(self, status): pass
        def record_exception(self, exc): pass
        def add_event(self, name, **kwargs): pass
    
    class NoOpMeter:
        def create_counter(self, *args, **kwargs): return NoOpMetric()
        def create_histogram(self, *args, **kwargs): return NoOpMetric()
        def create_up_down_counter(self, *args, **kwargs): return NoOpMetric()
    
    class NoOpMetric:
        def add(self, *args, **kwargs): pass
        def record(self, *args, **kwargs): pass
    
    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"
    
    class Status:
        def __init__(self, status_code, description=""):
            self.status_code = status_code
            self.description = description

logger = logging.getLogger(__name__)

class ObservabilityProvider:
    """Manages OpenTelemetry instrumentation for A2A orchestration"""
    
    def __init__(self):
        if OTEL_AVAILABLE:
            self.resource = Resource.create({
                "service.name": "a2a-orchestration",
                "service.version": "1.0.0",
                "deployment.environment": os.getenv("VERCEL_ENV", "development")
            })
            
            self._init_tracing()
            self._init_metrics()
            self._init_logging()
            
            self.tracer = trace.get_tracer("a2a.orchestration")
            self.meter = metrics.get_meter("a2a.orchestration")
        else:
            logger.warning("OpenTelemetry not available, using no-op observability")
            self.tracer = NoOpTracer()
            self.meter = NoOpMeter()
        
        # Create metrics
        self._create_metrics()
    
    def _init_tracing(self):
        """Initialize distributed tracing"""
        if not OTEL_AVAILABLE:
            return
            
        # Configure OTLP exporter for Vercel
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        
        if otlp_endpoint:
            exporter = OTLPSpanExporter(
                endpoint=f"{otlp_endpoint}/v1/traces",
                headers={"api-key": os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")}
            )
        else:
            # Console exporter for local development
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        
        provider = TracerProvider(resource=self.resource)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    
    def _init_metrics(self):
        """Initialize metrics collection"""
        if not OTEL_AVAILABLE:
            return
            
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        
        if otlp_endpoint:
            exporter = OTLPMetricExporter(
                endpoint=f"{otlp_endpoint}/v1/metrics",
                headers={"api-key": os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")}
            )
        else:
            # Console exporter for local development
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            exporter = ConsoleMetricExporter()
        
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60000)
        provider = MeterProvider(resource=self.resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
    
    def _init_logging(self):
        """Initialize logging instrumentation"""
        if OTEL_AVAILABLE:
            LoggingInstrumentor().instrument(set_logging_format=True)
    
    def _create_metrics(self):
        """Create metric instruments"""
        # Counters
        self.workflow_counter = self.meter.create_counter(
            "workflow.executions",
            description="Number of workflow executions",
            unit="1"
        )
        
        self.message_counter = self.meter.create_counter(
            "a2a.messages",
            description="Number of A2A messages processed",
            unit="1"
        )
        
        self.error_counter = self.meter.create_counter(
            "workflow.errors",
            description="Number of workflow errors",
            unit="1"
        )
        
        # Histograms
        self.workflow_duration = self.meter.create_histogram(
            "workflow.duration",
            description="Workflow execution duration",
            unit="ms"
        )
        
        self.step_duration = self.meter.create_histogram(
            "workflow.step.duration",
            description="Workflow step execution duration",
            unit="ms"
        )
        
        # Gauges (via UpDownCounter)
        self.active_workflows = self.meter.create_up_down_counter(
            "workflow.active",
            description="Number of active workflows",
            unit="1"
        )
        
        self.queue_size = self.meter.create_up_down_counter(
            "queue.size",
            description="Message queue size",
            unit="1"
        )
    
    @contextmanager
    def trace_workflow(self, workflow_id: str, execution_id: str):
        """Trace workflow execution"""
        with self.tracer.start_as_current_span(
            f"workflow.{workflow_id}",
            attributes={
                "workflow.id": workflow_id,
                "workflow.execution_id": execution_id,
                "workflow.type": "a2a"
            }
        ) as span:
            self.workflow_counter.add(1, {"workflow_id": workflow_id, "status": "started"})
            self.active_workflows.add(1)
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
                self.workflow_counter.add(1, {"workflow_id": workflow_id, "status": "completed"})
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                self.error_counter.add(1, {"workflow_id": workflow_id, "error_type": type(e).__name__})
                raise
            finally:
                self.active_workflows.add(-1)
    
    @contextmanager
    def trace_step(self, step_id: str, agent_id: str, action: str):
        """Trace workflow step execution"""
        with self.tracer.start_as_current_span(
            f"step.{step_id}",
            attributes={
                "step.id": step_id,
                "step.agent_id": agent_id,
                "step.action": action
            }
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def trace_message(self, message_type: str, sender: str, receiver: str):
        """Trace A2A message"""
        span = trace.get_current_span()
        if span:
            span.add_event(
                "a2a_message",
                attributes={
                    "message.type": message_type,
                    "message.sender": sender,
                    "message.receiver": receiver
                }
            )
        
        self.message_counter.add(
            1, 
            {
                "type": message_type,
                "sender": sender,
                "receiver": receiver
            }
        )
    
    def record_workflow_duration(self, workflow_id: str, duration_ms: float):
        """Record workflow execution duration"""
        self.workflow_duration.record(
            duration_ms,
            {"workflow_id": workflow_id}
        )
    
    def record_step_duration(self, step_id: str, agent_id: str, duration_ms: float):
        """Record step execution duration"""
        self.step_duration.record(
            duration_ms,
            {
                "step_id": step_id,
                "agent_id": agent_id
            }
        )
    
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size metric"""
        # Reset to 0 then add actual size
        self.queue_size.add(-10000, {"queue": queue_name})  # Reset
        self.queue_size.add(size, {"queue": queue_name})

# Global observability provider
observability = ObservabilityProvider()

# Decorators for easy instrumentation
def trace_workflow_execution(workflow_id: str):
    """Decorator to trace workflow execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            execution_id = kwargs.get("execution_id", "unknown")
            with observability.trace_workflow(workflow_id, execution_id):
                return await func(self, *args, **kwargs)
        return wrapper
    return decorator

def trace_step_execution(step_id: str, agent_id: str, action: str):
    """Decorator to trace step execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with observability.trace_step(step_id, agent_id, action):
                return await func(*args, **kwargs)
        return wrapper
    return decorator