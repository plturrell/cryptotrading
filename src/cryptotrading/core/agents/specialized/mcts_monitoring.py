"""
MCTS Agent Monitoring and Observability
Provides real-time metrics, health checks, and performance tracking
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ...protocols.mcp.events import EventType, create_event_publisher
from ...protocols.mcp.metrics import mcp_metrics


@dataclass
class PerformanceMetrics:
    """Performance metrics for MCTS calculations"""

    timestamp: datetime
    calculation_id: str
    iterations: int
    execution_time: float
    tree_size: int
    memory_usage_mb: float
    best_action_confidence: float
    expected_value: float
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "calculation_id": self.calculation_id,
            "iterations": self.iterations,
            "execution_time": self.execution_time,
            "tree_size": self.tree_size,
            "memory_usage_mb": self.memory_usage_mb,
            "best_action_confidence": self.best_action_confidence,
            "expected_value": self.expected_value,
            "cache_hit": self.cache_hit,
        }


@dataclass
class HealthStatus:
    """Health status of MCTS agent"""

    status: str  # healthy, degraded, unhealthy
    agent_id: str
    uptime_seconds: float
    memory_ok: bool
    circuit_breaker_state: str
    rate_limit_remaining: int
    last_calculation_time: Optional[datetime] = None
    error_rate: float = 0.0
    average_response_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "agent_id": self.agent_id,
            "uptime_seconds": self.uptime_seconds,
            "memory_ok": self.memory_ok,
            "circuit_breaker_state": self.circuit_breaker_state,
            "rate_limit_remaining": self.rate_limit_remaining,
            "last_calculation_time": self.last_calculation_time.isoformat()
            if self.last_calculation_time
            else None,
            "error_rate": self.error_rate,
            "average_response_time": self.average_response_time,
        }


class MCTSMonitor:
    """Monitor for MCTS agent performance and health"""

    def __init__(
        self,
        agent_id: str,
        retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.agent_id = agent_id
        self.retention_hours = retention_hours
        self.alert_thresholds = alert_thresholds or {
            "error_rate": 0.1,  # 10% error rate
            "response_time": 5.0,  # 5 seconds
            "memory_usage": 400,  # 400 MB
            "circuit_breaker_open": True,
        }

        # Metrics storage
        self.performance_metrics: List[PerformanceMetrics] = []
        self.error_log: List[Dict[str, Any]] = []
        self.calculation_count = 0
        self.error_count = 0
        self.start_time = time.time()

        # Event publisher
        self.event_publisher = create_event_publisher()

        # Start background cleanup
        asyncio.create_task(self._cleanup_old_metrics())

    async def record_calculation(self, metrics: PerformanceMetrics):
        """Record calculation metrics"""
        self.performance_metrics.append(metrics)
        self.calculation_count += 1

        # Check for alerts
        await self._check_alerts(metrics)

        # Publish metrics event
        import time

        from ...protocols.mcp.events import EventType, MCPEvent

        event = MCPEvent(
            event_type=EventType.SYSTEM_STATUS,
            data=metrics.to_dict(),
            timestamp=time.time(),
            source="mcts_agent",
        )
        await self.event_publisher.streamer.publish_event(event)

        # Update MCP metrics
        mcp_metrics.collector.counter(
            "mcts_iterations", metrics.iterations, {"agent_id": self.agent_id}
        )
        mcp_metrics.collector.timer(
            "mcts_execution_time", metrics.execution_time, {"agent_id": self.agent_id}
        )

    async def record_error(self, error: Exception, context: Dict[str, Any]):
        """Record calculation error"""
        self.error_count += 1

        error_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        }

        self.error_log.append(error_entry)

        # Publish error event
        import time

        from ...protocols.mcp.events import EventType, MCPEvent

        error_event = MCPEvent(
            event_type=EventType.ERROR, data=error_entry, timestamp=time.time(), source="mcts_agent"
        )
        await self.event_publisher.streamer.publish_event(error_event)

    async def get_health_status(
        self, circuit_breaker_state: str, rate_limit_remaining: int, memory_ok: bool
    ) -> HealthStatus:
        """Get current health status"""
        uptime = time.time() - self.start_time

        # Calculate metrics
        error_rate = self.error_count / self.calculation_count if self.calculation_count > 0 else 0

        recent_metrics = self._get_recent_metrics(minutes=5)
        avg_response_time = (
            sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
            if recent_metrics
            else 0
        )

        last_calc_time = (
            self.performance_metrics[-1].timestamp if self.performance_metrics else None
        )

        # Determine health status
        if circuit_breaker_state == "open" or not memory_ok:
            status = "unhealthy"
        elif (
            error_rate > self.alert_thresholds["error_rate"]
            or avg_response_time > self.alert_thresholds["response_time"]
        ):
            status = "degraded"
        else:
            status = "healthy"

        return HealthStatus(
            status=status,
            agent_id=self.agent_id,
            uptime_seconds=uptime,
            memory_ok=memory_ok,
            circuit_breaker_state=circuit_breaker_state,
            rate_limit_remaining=rate_limit_remaining,
            last_calculation_time=last_calc_time,
            error_rate=error_rate,
            average_response_time=avg_response_time,
        )

    async def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        recent_metrics = [m for m in self.performance_metrics if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"period_hours": hours, "calculation_count": 0, "error_count": 0, "metrics": {}}

        # Calculate aggregates
        total_iterations = sum(m.iterations for m in recent_metrics)
        avg_iterations = total_iterations / len(recent_metrics)
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        avg_tree_size = sum(m.tree_size for m in recent_metrics) / len(recent_metrics)
        avg_confidence = sum(m.best_action_confidence for m in recent_metrics) / len(recent_metrics)
        avg_value = sum(m.expected_value for m in recent_metrics) / len(recent_metrics)
        cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)

        # Memory statistics
        max_memory = max(m.memory_usage_mb for m in recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)

        return {
            "period_hours": hours,
            "calculation_count": len(recent_metrics),
            "error_count": len(
                [e for e in self.error_log if datetime.fromisoformat(e["timestamp"]) > cutoff_time]
            ),
            "metrics": {
                "average_iterations": avg_iterations,
                "total_iterations": total_iterations,
                "average_execution_time": avg_execution_time,
                "average_tree_size": avg_tree_size,
                "average_confidence": avg_confidence,
                "average_expected_value": avg_value,
                "cache_hit_rate": cache_hit_rate,
                "max_memory_usage_mb": max_memory,
                "average_memory_usage_mb": avg_memory,
                "iterations_per_second": total_iterations
                / sum(m.execution_time for m in recent_metrics),
            },
        }

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts based on thresholds"""
        alerts = []

        # Check error rate
        error_rate = self.error_count / self.calculation_count if self.calculation_count > 0 else 0
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(
                {
                    "type": "error_rate",
                    "severity": "high",
                    "message": f'Error rate {error_rate:.1%} exceeds threshold {self.alert_thresholds["error_rate"]:.1%}',
                    "value": error_rate,
                }
            )

        # Check response time
        recent_metrics = self._get_recent_metrics(minutes=5)
        if recent_metrics:
            avg_response_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
            if avg_response_time > self.alert_thresholds["response_time"]:
                alerts.append(
                    {
                        "type": "response_time",
                        "severity": "medium",
                        "message": f'Average response time {avg_response_time:.2f}s exceeds threshold {self.alert_thresholds["response_time"]}s',
                        "value": avg_response_time,
                    }
                )

        # Check memory usage
        if recent_metrics:
            max_memory = max(m.memory_usage_mb for m in recent_metrics)
            if max_memory > self.alert_thresholds["memory_usage"]:
                alerts.append(
                    {
                        "type": "memory_usage",
                        "severity": "high",
                        "message": f'Memory usage {max_memory:.0f}MB exceeds threshold {self.alert_thresholds["memory_usage"]}MB',
                        "value": max_memory,
                    }
                )

        return alerts

    def _get_recent_metrics(self, minutes: int) -> List[PerformanceMetrics]:
        """Get metrics from recent minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [m for m in self.performance_metrics if m.timestamp > cutoff_time]

    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check if metrics trigger any alerts"""
        # Memory alert
        if metrics.memory_usage_mb > self.alert_thresholds["memory_usage"]:
            import time

            from ...protocols.mcp.events import EventType, MCPEvent

            alert_event = MCPEvent(
                event_type=EventType.ALERT,
                data={
                    "type": "memory_usage",
                    "agent_id": self.agent_id,
                    "value": metrics.memory_usage_mb,
                    "threshold": self.alert_thresholds["memory_usage"],
                },
                timestamp=time.time(),
                source="mcts_agent",
            )
            await self.event_publisher.streamer.publish_event(alert_event)

        # Response time alert
        if metrics.execution_time > self.alert_thresholds["response_time"]:
            import time

            from ...protocols.mcp.events import EventType, MCPEvent

            response_alert = MCPEvent(
                event_type=EventType.ALERT,
                data={
                    "type": "response_time",
                    "agent_id": self.agent_id,
                    "value": metrics.execution_time,
                    "threshold": self.alert_thresholds["response_time"],
                },
                timestamp=time.time(),
                source="mcts_agent",
            )
            await self.event_publisher.streamer.publish_event(response_alert)

    async def _cleanup_old_metrics(self):
        """Background task to clean up old metrics"""
        while True:
            await asyncio.sleep(3600)  # Run hourly

            cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)

            # Clean performance metrics
            self.performance_metrics = [
                m for m in self.performance_metrics if m.timestamp > cutoff_time
            ]

            # Clean error log
            self.error_log = [
                e for e in self.error_log if datetime.fromisoformat(e["timestamp"]) > cutoff_time
            ]

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        data = {
            "agent_id": self.agent_id,
            "export_time": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "total_calculations": self.calculation_count,
            "total_errors": self.error_count,
            "performance_metrics": [m.to_dict() for m in self.performance_metrics],
            "error_log": self.error_log,
        }

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# OpenTelemetry integration for distributed tracing
class MCTSTracer:
    """OpenTelemetry tracer for MCTS calculations"""

    def __init__(self, service_name: str = "mcts-agent"):
        self.service_name = service_name
        self._setup_tracing()

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        # Check if OpenTelemetry is enabled
        if os.getenv("OTEL_ENABLED", "false").lower() != "true":
            self.tracer = None
            return

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Configure tracer
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()

            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"), insecure=True
            )

            # Add span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)

            # Get tracer
            self.tracer = trace.get_tracer(self.service_name)

        except ImportError:
            # OpenTelemetry not installed
            self.tracer = None

    def trace_calculation(self, calculation_type: str, parameters: Dict[str, Any]):
        """Create a trace span for MCTS calculation"""
        if not self.tracer:
            return None

        return self.tracer.start_as_current_span(
            name=f"mcts_calculation_{calculation_type}",
            attributes={
                "calculation.type": calculation_type,
                "calculation.iterations": parameters.get("iterations", 0),
                "calculation.symbols": ",".join(parameters.get("symbols", [])),
                "calculation.portfolio": parameters.get("initial_portfolio", 0),
            },
        )
