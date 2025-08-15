"""
MCP Lightweight Metrics Collection
Simple metrics collection for serverless environments
"""
import time
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]
    metric_type: MetricType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "type": self.metric_type.value
        }


class MetricsCollector:
    """Lightweight metrics collector for serverless"""
    
    def __init__(self, max_points: int = 1000):
        self.metrics: List[MetricPoint] = []
        self.max_points = max_points
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        
    def counter(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """Record a counter metric"""
        tags = tags or {}
        key = f"{name}:{json.dumps(tags, sort_keys=True)}"
        
        self.counters[key] = self.counters.get(key, 0) + value
        
        self._add_metric(MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            metric_type=MetricType.COUNTER
        ))
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric"""
        tags = tags or {}
        key = f"{name}:{json.dumps(tags, sort_keys=True)}"
        
        self.gauges[key] = value
        
        self._add_metric(MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            metric_type=MetricType.GAUGE
        ))
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric"""
        tags = tags or {}
        
        self._add_metric(MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            metric_type=MetricType.HISTOGRAM
        ))
    
    def timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer metric"""
        tags = tags or {}
        
        self._add_metric(MetricPoint(
            name=name,
            value=duration,
            timestamp=time.time(),
            tags=tags,
            metric_type=MetricType.TIMER
        ))
    
    def _add_metric(self, metric: MetricPoint):
        """Add metric to collection"""
        self.metrics.append(metric)
        
        # Keep only recent metrics to avoid memory issues
        if len(self.metrics) > self.max_points:
            self.metrics = self.metrics[-self.max_points:]
    
    def get_metrics(self, since: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get metrics since timestamp"""
        if since is None:
            return [m.to_dict() for m in self.metrics]
        
        return [
            m.to_dict() for m in self.metrics
            if m.timestamp >= since
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        now = time.time()
        last_minute = now - 60
        last_hour = now - 3600
        
        recent_metrics = [m for m in self.metrics if m.timestamp >= last_minute]
        hourly_metrics = [m for m in self.metrics if m.timestamp >= last_hour]
        
        return {
            "total_metrics": len(self.metrics),
            "metrics_last_minute": len(recent_metrics),
            "metrics_last_hour": len(hourly_metrics),
            "active_counters": len(self.counters),
            "active_gauges": len(self.gauges),
            "oldest_metric": min(m.timestamp for m in self.metrics) if self.metrics else None,
            "newest_metric": max(m.timestamp for m in self.metrics) if self.metrics else None
        }
    
    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()


class MCPMetrics:
    """MCP-specific metrics collection"""
    
    def __init__(self, collector: MetricsCollector = None):
        self.collector = collector or MetricsCollector()
    
    def tool_execution_start(self, tool_name: str, user_id: str = None):
        """Record tool execution start"""
        tags = {"tool": tool_name}
        if user_id:
            tags["user"] = user_id
        
        self.collector.counter("mcp.tool.executions", tags=tags)
        return time.time()  # Return start time for duration calculation
    
    def tool_execution_end(self, tool_name: str, start_time: float, 
                          success: bool = True, user_id: str = None):
        """Record tool execution completion"""
        duration = time.time() - start_time
        
        tags = {
            "tool": tool_name,
            "status": "success" if success else "error"
        }
        if user_id:
            tags["user"] = user_id
        
        self.collector.timer("mcp.tool.duration", duration, tags=tags)
        self.collector.counter("mcp.tool.completions", tags=tags)
    
    def resource_access(self, uri: str, cached: bool = False):
        """Record resource access"""
        tags = {
            "cached": str(cached).lower()
        }
        
        self.collector.counter("mcp.resource.accesses", tags=tags)
    
    def connection_created(self, transport_type: str):
        """Record connection creation"""
        tags = {"transport": transport_type}
        self.collector.counter("mcp.connections.created", tags=tags)
    
    def connection_error(self, transport_type: str, error_type: str):
        """Record connection error"""
        tags = {
            "transport": transport_type,
            "error_type": error_type
        }
        self.collector.counter("mcp.connections.errors", tags=tags)
    
    def auth_attempt(self, method: str, success: bool):
        """Record authentication attempt"""
        tags = {
            "method": method,
            "status": "success" if success else "failure"
        }
        self.collector.counter("mcp.auth.attempts", tags=tags)
    
    def rate_limit_hit(self, user_id: str, limit_type: str):
        """Record rate limit hit"""
        tags = {
            "user": user_id,
            "limit_type": limit_type
        }
        self.collector.counter("mcp.rate_limits.hits", tags=tags)
    
    def cache_operation(self, operation: str, hit: bool = None):
        """Record cache operation"""
        tags = {"operation": operation}
        if hit is not None:
            tags["result"] = "hit" if hit else "miss"
        
        self.collector.counter("mcp.cache.operations", tags=tags)
    
    def market_data_request(self, symbol: str, timeframe: str):
        """Record market data request"""
        tags = {
            "symbol": symbol,
            "timeframe": timeframe
        }
        self.collector.counter("mcp.market_data.requests", tags=tags)
    
    def trading_operation(self, operation: str, symbol: str, success: bool):
        """Record trading operation"""
        tags = {
            "operation": operation,
            "symbol": symbol,
            "status": "success" if success else "error"
        }
        self.collector.counter("mcp.trading.operations", tags=tags)


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, metrics: MCPMetrics, metric_name: str, tags: Dict[str, str] = None):
        self.metrics = metrics
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            
            tags = {**self.tags, "status": "success" if success else "error"}
            self.metrics.collector.timer(self.metric_name, duration, tags=tags)


def timer(metrics: MCPMetrics, metric_name: str, tags: Dict[str, str] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimerContext(metrics, metric_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class MetricsExporter:
    """Export metrics to external systems"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def export_to_vercel_analytics(self) -> Dict[str, Any]:
        """Export metrics in Vercel Analytics format"""
        metrics = self.collector.get_metrics()
        
        # Group by metric name
        grouped = {}
        for metric in metrics:
            name = metric["name"]
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(metric)
        
        return {
            "timestamp": time.time(),
            "metrics": grouped,
            "summary": self.collector.get_summary()
        }
    
    def export_to_json(self) -> str:
        """Export metrics as JSON"""
        data = {
            "timestamp": time.time(),
            "metrics": self.collector.get_metrics(),
            "summary": self.collector.get_summary()
        }
        return json.dumps(data, indent=2)
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Group metrics by name and type
        metric_groups = {}
        for metric in self.collector.get_metrics():
            name = metric["name"]
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric)
        
        for name, metrics in metric_groups.items():
            # Add help and type comments
            lines.append(f"# HELP {name} MCP metric")
            lines.append(f"# TYPE {name} {metrics[0]['type']}")
            
            # Add metric values
            for metric in metrics:
                tags_str = ""
                if metric["tags"]:
                    tag_pairs = [f'{k}="{v}"' for k, v in metric["tags"].items()]
                    tags_str = "{" + ",".join(tag_pairs) + "}"
                
                lines.append(f"{name}{tags_str} {metric['value']} {int(metric['timestamp'] * 1000)}")
        
        return "\n".join(lines)


# Global metrics instances
global_metrics_collector = MetricsCollector()
mcp_metrics = MCPMetrics(global_metrics_collector)
metrics_exporter = MetricsExporter(global_metrics_collector)


def get_metrics_summary() -> Dict[str, Any]:
    """Get global metrics summary"""
    return global_metrics_collector.get_summary()


def export_metrics(format: str = "json") -> str:
    """Export metrics in specified format"""
    if format == "json":
        return metrics_exporter.export_to_json()
    elif format == "prometheus":
        return metrics_exporter.export_prometheus_format()
    elif format == "vercel":
        return json.dumps(metrics_exporter.export_to_vercel_analytics(), indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def clear_metrics():
    """Clear all metrics"""
    global_metrics_collector.clear()
    logger.info("Global metrics cleared")
