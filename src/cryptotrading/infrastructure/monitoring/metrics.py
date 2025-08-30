"""
Metrics Collection and Monitoring
Provides performance metrics and business metrics tracking
"""

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point"""

    timestamp: datetime
    value: float
    tags: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp.isoformat(), "value": self.value, "tags": self.tags}


class MetricsCollector:
    """Collects and aggregates metrics"""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.metric_types: Dict[str, MetricType] = {}
        self.lock = threading.Lock()

        # Start background export task
        self._start_background_export()

    def counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Record counter metric"""
        self._record_metric(name, value, MetricType.COUNTER, tags or {})

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record gauge metric"""
        self._record_metric(name, value, MetricType.GAUGE, tags or {})

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram metric"""
        self._record_metric(name, value, MetricType.HISTOGRAM, tags or {})

    def timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record timer metric"""
        self._record_metric(name, duration_ms, MetricType.TIMER, tags or {})

    def _record_metric(
        self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str]
    ):
        """Record metric point"""
        point = MetricPoint(timestamp=datetime.utcnow(), value=value, tags=tags)

        with self.lock:
            self.metrics[name].append(point)
            self.metric_types[name] = metric_type

        # Log metric
        logger.log_metric(name, value, metric_type.value, tags)

    def time_function(self, name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Decorator to time function execution"""

        def decorator(func):
            metric_name = name or f"function.{func.__module__}.{func.__name__}.duration_ms"

            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    self.timer(metric_name, duration_ms, tags)
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    error_tags = (tags or {}).copy()
                    error_tags["error"] = "true"
                    error_tags["error_type"] = type(e).__name__
                    self.timer(metric_name, duration_ms, error_tags)
                    raise

            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    self.timer(metric_name, duration_ms, tags)
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    error_tags = (tags or {}).copy()
                    error_tags["error"] = "true"
                    error_tags["error_type"] = type(e).__name__
                    self.timer(metric_name, duration_ms, error_tags)
                    raise

            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def get_metric_summary(self, name: str, hours: int = 1) -> Optional[Dict[str, Any]]:
        """Get metric summary for the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self.lock:
            if name not in self.metrics:
                return None

            points = [p for p in self.metrics[name] if p.timestamp >= cutoff]

        if not points:
            return None

        values = [p.value for p in points]
        metric_type = self.metric_types[name]

        summary = {
            "name": name,
            "type": metric_type.value,
            "count": len(points),
            "time_range": {
                "start": cutoff.isoformat(),
                "end": datetime.utcnow().isoformat(),
                "hours": hours,
            },
        }

        if metric_type == MetricType.COUNTER:
            summary["total"] = sum(values)
            summary["rate_per_minute"] = sum(values) / (hours * 60)

        elif metric_type in [MetricType.GAUGE, MetricType.HISTOGRAM, MetricType.TIMER]:
            summary.update(
                {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1] if values else 0,
                }
            )

            # Calculate percentiles for histograms and timers
            if metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                sorted_values = sorted(values)
                n = len(sorted_values)
                summary["percentiles"] = {
                    "p50": sorted_values[int(n * 0.5)],
                    "p90": sorted_values[int(n * 0.9)],
                    "p95": sorted_values[int(n * 0.95)],
                    "p99": sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
                }

        return summary

    def get_all_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summaries = {}

        with self.lock:
            metric_names = list(self.metrics.keys())

        for name in metric_names:
            summary = self.get_metric_summary(name, hours)
            if summary:
                summaries[name] = summary

        return {
            "time_range": {"hours": hours, "end": datetime.utcnow().isoformat()},
            "total_metrics": len(summaries),
            "metrics": summaries,
        }

    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format"""
        if format == "prometheus":
            return self._export_prometheus_format()
        elif format == "json":
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_prometheus_format(self) -> str:
        """Export in Prometheus format"""
        output = []

        with self.lock:
            for name, points in self.metrics.items():
                if not points:
                    continue

                metric_type = self.metric_types[name]
                latest_point = points[-1]

                # Add help and type
                output.append(f"# HELP {name} {metric_type.value} metric")
                output.append(f"# TYPE {name} {metric_type.value}")

                # Add metric line
                if latest_point.tags:
                    tags_str = ",".join(f'{k}="{v}"' for k, v in latest_point.tags.items())
                    output.append(f"{name}{{{tags_str}}} {latest_point.value}")
                else:
                    output.append(f"{name} {latest_point.value}")

        return "\n".join(output)

    def _export_json_format(self) -> str:
        """Export in JSON format"""
        data = {}

        with self.lock:
            for name, points in self.metrics.items():
                data[name] = {
                    "type": self.metric_types[name].value,
                    "points": [p.to_dict() for p in points],
                }

        return json.dumps(data, indent=2)

    def _start_background_export(self):
        """Start background metric export"""
        self._export_task = None
        self._running = True

        # Start async export task
        try:
            loop = asyncio.get_running_loop()
            self._export_task = loop.create_task(self._async_export_task())
        except RuntimeError:
            # No event loop running, use thread fallback
            import threading

            export_thread = threading.Thread(target=self._sync_export_task, daemon=True)
            export_thread.start()

    async def _async_export_task(self):
        """Async metric export task"""
        while self._running:
            try:
                # Export metrics every 60 seconds
                await asyncio.sleep(60)

                # Send to external monitoring system if configured
                endpoint = os.getenv("METRICS_ENDPOINT")
                if endpoint:
                    await self._async_send_to_external_system(endpoint)

            except Exception as e:
                logger.error(f"Error in metrics export task: {e}")
                await asyncio.sleep(60)  # Continue after error

    def _sync_export_task(self):
        """Synchronous export task for non-async contexts"""
        while self._running:
            try:
                time.sleep(60)
                endpoint = os.getenv("METRICS_ENDPOINT")
                if endpoint:
                    self._send_to_external_system(endpoint)
            except Exception as e:
                logger.error(f"Error in sync metrics export: {e}")

    async def stop_export(self):
        """Stop the export task gracefully"""
        self._running = False
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

    async def _async_send_to_external_system(self, endpoint: str):
        """Send metrics to external monitoring system asynchronously"""
        try:
            import aiohttp

            metrics_data = self.get_all_metrics_summary(hours=1)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=metrics_data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send metrics: {response.status}")

        except Exception as e:
            logger.error(f"Error sending metrics to external system: {e}")

    def _send_to_external_system(self, endpoint: str):
        """Send metrics to external monitoring system"""
        try:
            import requests

            metrics_data = self.get_all_metrics_summary(hours=1)

            response = requests.post(
                endpoint,
                json=metrics_data,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("Metrics exported successfully")
            else:
                logger.warning(f"Metrics export failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending metrics to external system: {e}")


# Business Metrics
class BusinessMetrics:
    """Tracks business-specific metrics"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector

    def track_trade_execution(
        self, symbol: str, side: str, quantity: float, price: float, success: bool
    ):
        """Track trade execution metrics"""
        tags = {"symbol": symbol, "side": side, "success": str(success).lower()}

        # Count trades
        self.metrics.counter("trades.total", 1.0, tags)

        # Track volume
        volume = quantity * price
        self.metrics.histogram("trades.volume_usd", volume, tags)

        # Track price
        self.metrics.gauge(f"prices.{symbol.lower()}", price, {"symbol": symbol})

    def track_api_request(self, endpoint: str, method: str, status_code: int, duration_ms: float):
        """Track API request metrics"""
        tags = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code),
            "success": str(200 <= status_code < 300).lower(),
        }

        # Count requests
        self.metrics.counter("api.requests.total", 1.0, tags)

        # Track duration
        self.metrics.timer("api.requests.duration_ms", duration_ms, tags)

    def track_agent_operation(
        self, agent_id: str, operation: str, success: bool, duration_ms: float
    ):
        """Track A2A agent operations"""
        tags = {"agent_id": agent_id, "operation": operation, "success": str(success).lower()}

        # Count operations
        self.metrics.counter("agents.operations.total", 1.0, tags)

        # Track duration
        self.metrics.timer("agents.operations.duration_ms", duration_ms, tags)

    def track_data_processing(
        self, source: str, symbol: str, records_processed: int, success: bool, duration_ms: float
    ):
        """Track data processing metrics"""
        tags = {"source": source, "symbol": symbol, "success": str(success).lower()}

        # Count processing jobs
        self.metrics.counter("data.processing.jobs.total", 1.0, tags)

        # Track records processed
        self.metrics.histogram("data.processing.records", float(records_processed), tags)

        # Track duration
        self.metrics.timer("data.processing.duration_ms", duration_ms, tags)

    def track_ai_operation(
        self, operation: str, model: str, symbol: str, success: bool, duration_ms: float
    ):
        """Track AI operation metrics"""
        tags = {
            "operation": operation,
            "model": model,
            "symbol": symbol,
            "success": str(success).lower(),
        }

        # Count AI operations
        self.metrics.counter("ai.operations.total", 1.0, tags)

        # Track duration
        self.metrics.timer("ai.operations.duration_ms", duration_ms, tags)


# Global metrics collector
_metrics_collector = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_business_metrics() -> BusinessMetrics:
    """Get business metrics tracker"""
    return BusinessMetrics(get_metrics())
