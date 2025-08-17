"""
Enterprise-Grade Strands Observability and Monitoring System
Advanced monitoring, metrics collection, alerting, and distributed tracing for Strands agents.
"""
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json
import uuid
import time
from collections import defaultdict, deque
import logging
import traceback
from contextlib import asynccontextmanager

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class TraceLevel(Enum):
    """Tracing levels"""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"

@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    metric_name: str
    condition: str  # ">", "<", "==", "!="
    threshold: Union[int, float]
    severity: AlertSeverity
    duration_minutes: int = 5
    cooldown_minutes: int = 15
    enabled: bool = True

@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TraceSpan:
    """Distributed tracing span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

class StrandsObservabilitySystem:
    """Enterprise observability system for Strands agents"""
    
    def __init__(self, agent: 'EnhancedStrandsAgent'):
        self.agent = agent
        self.logger = logging.getLogger(f"StrandsObservability-{agent.agent_id}")
        
        # Metrics storage
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.metric_buffer_size = 1000
        
        # Alerting system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Distributed tracing
        self.active_traces: Dict[str, TraceSpan] = {}
        self.completed_traces: List[TraceSpan] = []
        self.trace_buffer_size = 500
        
        # Performance monitoring
        self.performance_metrics = {
            "tool_execution_times": deque(maxlen=100),
            "workflow_execution_times": deque(maxlen=50),
            "memory_usage": deque(maxlen=100),
            "error_rates": deque(maxlen=100)
        }
        
        # Health monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_check = datetime.utcnow()
        self.health_check_interval = 60  # seconds
        
        self._setup_default_alerts()
        self._setup_default_health_checks()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                id="high_error_rate",
                name="High Error Rate",
                metric_name="error_rate",
                condition=">",
                threshold=0.1,
                severity=AlertSeverity.WARNING,
                duration_minutes=5
            ),
            AlertRule(
                id="critical_error_rate",
                name="Critical Error Rate",
                metric_name="error_rate",
                condition=">",
                threshold=0.25,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=2
            ),
            AlertRule(
                id="slow_tool_execution",
                name="Slow Tool Execution",
                metric_name="avg_tool_execution_time",
                condition=">",
                threshold=30.0,
                severity=AlertSeverity.WARNING,
                duration_minutes=10
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        self.health_checks.update({
            "agent_responsive": self._check_agent_responsive,
            "memory_usage": self._check_memory_usage,
            "tool_registry": self._check_tool_registry,
            "workflow_registry": self._check_workflow_registry
        })
    
    async def record_metric(self, name: str, value: Union[int, float], 
                          metric_type: MetricType = MetricType.GAUGE,
                          tags: Dict[str, str] = None):
        """Record a metric data point"""
        tags = tags or {}
        
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags,
            metric_type=metric_type
        )
        
        # Add to metrics storage
        self.metrics[name].append(metric_point)
        
        # Maintain buffer size
        if len(self.metrics[name]) > self.metric_buffer_size:
            self.metrics[name] = self.metrics[name][-self.metric_buffer_size:]
        
        # Check for alerts
        await self._check_alerts(name, value)
        
        self.logger.debug(f"Recorded metric: {name}={value} {tags}")
    
    async def _check_alerts(self, metric_name: str, value: Union[int, float]):
        """Check if metric triggers any alerts"""
        for rule_id, rule in self.alert_rules.items():
            if rule.metric_name == metric_name and rule.enabled:
                triggered = False
                
                if rule.condition == ">" and value > rule.threshold:
                    triggered = True
                elif rule.condition == "<" and value < rule.threshold:
                    triggered = True
                elif rule.condition == "==" and value == rule.threshold:
                    triggered = True
                elif rule.condition == "!=" and value != rule.threshold:
                    triggered = True
                
                if triggered:
                    await self._trigger_alert(rule, value)
    
    async def _trigger_alert(self, rule: AlertRule, value: Union[int, float]):
        """Trigger an alert"""
        alert_id = f"{rule.id}_{int(time.time())}"
        
        # Check cooldown
        if rule.id in self.active_alerts:
            last_alert = self.active_alerts[rule.id]
            cooldown_end = last_alert.timestamp + timedelta(minutes=rule.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                return
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            severity=rule.severity,
            message=f"{rule.name}: {rule.metric_name} {rule.condition} {rule.threshold} (current: {value})",
            timestamp=datetime.utcnow(),
            metadata={"metric_value": value, "threshold": rule.threshold}
        )
        
        self.active_alerts[rule.id] = alert
        self.alert_history.append(alert)
        
        self.logger.warning(f"Alert triggered: {alert.message}")
        
        # Notify agent
        if hasattr(self.agent, 'handle_alert'):
            await self.agent.handle_alert(alert)
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager for distributed tracing"""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        tags = tags or {}
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            end_time=None,
            duration_ms=None,
            tags=tags
        )
        
        self.active_traces[span_id] = span
        
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            })
            raise
        finally:
            span.end_time = datetime.utcnow()
            span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
            
            # Move to completed traces
            del self.active_traces[span_id]
            self.completed_traces.append(span)
            
            # Maintain buffer size
            if len(self.completed_traces) > self.trace_buffer_size:
                self.completed_traces = self.completed_traces[-self.trace_buffer_size:]
            
            # Record performance metric
            await self.record_metric(
                f"operation_duration_{operation_name}",
                span.duration_ms,
                MetricType.TIMER,
                tags
            )
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_results = {}
        overall_status = "healthy"
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                health_results[check_name] = result
                
                if not result.get("healthy", True):
                    overall_status = "unhealthy"
                    
            except Exception as e:
                health_results[check_name] = {
                    "healthy": False,
                    "error": str(e)
                }
                overall_status = "unhealthy"
        
        self.last_health_check = datetime.utcnow()
        
        # Record health metric
        await self.record_metric("system_health", 1 if overall_status == "healthy" else 0)
        
        return {
            "overall_status": overall_status,
            "checks": health_results,
            "timestamp": self.last_health_check.isoformat()
        }
    
    async def _check_agent_responsive(self) -> Dict[str, Any]:
        """Check if agent is responsive"""
        try:
            # Simple responsiveness check
            start_time = time.time()
            await asyncio.sleep(0.001)  # Minimal async operation
            response_time = (time.time() - start_time) * 1000
            
            return {
                "healthy": response_time < 100,  # Less than 100ms
                "response_time_ms": response_time
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            return {
                "healthy": memory_percent < 80,  # Less than 80%
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": memory_percent
            }
        except ImportError:
            return {"healthy": True, "note": "psutil not available"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_tool_registry(self) -> Dict[str, Any]:
        """Check tool registry health"""
        try:
            tool_count = len(self.agent.tool_registry)
            return {
                "healthy": tool_count > 0,
                "tool_count": tool_count
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_workflow_registry(self) -> Dict[str, Any]:
        """Check workflow registry health"""
        try:
            workflow_count = len(self.agent.workflow_registry)
            return {
                "healthy": True,  # Workflows are optional
                "workflow_count": workflow_count
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        
        for metric_name, points in self.metrics.items():
            if points:
                values = [p.value for p in points[-10:]]  # Last 10 points
                summary[metric_name] = {
                    "current": values[-1] if values else None,
                    "average": sum(values) / len(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "count": len(points)
                }
        
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [
            {
                "id": alert.id,
                "rule_id": alert.rule_id,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            for alert in self.active_alerts.values()
            if not alert.resolved
        ]
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get tracing summary"""
        if not self.completed_traces:
            return {"traces": 0, "operations": []}
        
        operations = defaultdict(list)
        for trace in self.completed_traces[-50:]:  # Last 50 traces
            operations[trace.operation_name].append(trace.duration_ms)
        
        operation_stats = {}
        for op_name, durations in operations.items():
            operation_stats[op_name] = {
                "count": len(durations),
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations)
            }
        
        return {
            "total_traces": len(self.completed_traces),
            "active_traces": len(self.active_traces),
            "operations": operation_stats
        }
