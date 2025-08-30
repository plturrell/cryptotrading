"""
Advanced Monitoring and Anomaly Detection for MCTS Agent
Implements machine learning-based anomaly detection and intelligent alerting
"""
import asyncio
import json
import logging
import os
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_SPIKE = "memory_spike"
    CONVERGENCE_FAILURE = "convergence_failure"
    EXECUTION_TIME_ANOMALY = "execution_time_anomaly"
    VALUE_QUALITY_DROP = "value_quality_drop"
    ERROR_RATE_SPIKE = "error_rate_spike"
    UNUSUAL_TRAFFIC_PATTERN = "unusual_traffic_pattern"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class AnomalySeverity(Enum):
    """Severity levels for anomalies"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Represents an anomaly alert"""

    alert_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    message: str
    detected_at: datetime
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "detected_at": self.detected_at.isoformat(),
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "expected_range": self.expected_range,
            "confidence": self.confidence,
            "context": self.context,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class MetricSample:
    """Individual metric sample"""

    timestamp: datetime
    value: float
    context: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesAnalyzer:
    """Analyzes time series data for anomalies"""

    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.samples: deque = deque(maxlen=window_size)

    def add_sample(self, value: float, timestamp: datetime = None, context: Dict[str, Any] = None):
        """Add a new sample to the time series"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        sample = MetricSample(timestamp, value, context or {})
        self.samples.append(sample)

    def detect_anomalies(self) -> List[Tuple[float, float, str]]:
        """Detect anomalies using statistical methods"""
        if len(self.samples) < 10:
            return []

        values = [s.value for s in self.samples]
        anomalies = []

        # Statistical outlier detection
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0

        if std_val > 0:
            threshold = self.sensitivity * std_val

            for i, sample in enumerate(self.samples):
                if abs(sample.value - mean_val) > threshold:
                    confidence = min(abs(sample.value - mean_val) / threshold, 1.0)
                    anomalies.append((sample.value, confidence, f"statistical_outlier"))

        # Trend-based anomaly detection
        if len(values) >= 20:
            recent_mean = statistics.mean(values[-10:])
            historical_mean = statistics.mean(values[:-10])

            if historical_mean != 0:
                change_ratio = abs(recent_mean - historical_mean) / abs(historical_mean)
                if change_ratio > 0.5:  # 50% change threshold
                    anomalies.append((recent_mean, change_ratio, "trend_shift"))

        return anomalies

    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics for the time series"""
        if not self.samples:
            return {}

        values = [s.value for s in self.samples]

        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

        if len(values) > 1:
            stats["std"] = statistics.stdev(values)
            stats["variance"] = statistics.variance(values)

        if len(values) >= 4:
            stats["q1"] = statistics.quantiles(values, n=4)[0]
            stats["q3"] = statistics.quantiles(values, n=4)[2]
            stats["iqr"] = stats["q3"] - stats["q1"]

        return stats


class AnomalyDetector:
    """Main anomaly detection system"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.analyzers: Dict[str, TimeSeriesAnalyzer] = {}
        self.alerts: List[AnomalyAlert] = []
        self.alert_handlers: List[Callable] = []

        # Initialize analyzers for different metrics
        self._setup_analyzers()

        # Configuration
        self.alert_cooldown = timedelta(minutes=5)  # Prevent alert spam
        self.last_alerts: Dict[str, datetime] = {}

        logger.info(f"Anomaly detector initialized for agent {agent_id}")

    def _setup_analyzers(self):
        """Setup analyzers for different metrics"""
        metrics = [
            "execution_time",
            "memory_usage",
            "expected_value",
            "confidence",
            "iterations_completed",
            "tree_size",
            "error_rate",
            "convergence_confidence",
        ]

        for metric in metrics:
            self.analyzers[metric] = TimeSeriesAnalyzer(
                window_size=100,
                sensitivity=2.5 if metric in ["error_rate", "memory_usage"] else 2.0,
            )

    async def record_metric(self, metric_name: str, value: float, context: Dict[str, Any] = None):
        """Record a new metric value and check for anomalies"""
        if metric_name not in self.analyzers:
            logger.warning(f"Unknown metric: {metric_name}")
            return

        analyzer = self.analyzers[metric_name]
        analyzer.add_sample(value, context=context)

        # Check for anomalies
        anomalies = analyzer.detect_anomalies()

        for anomaly_value, confidence, detection_method in anomalies:
            await self._handle_anomaly(
                metric_name, anomaly_value, confidence, detection_method, context or {}
            )

    async def _handle_anomaly(
        self,
        metric_name: str,
        value: float,
        confidence: float,
        detection_method: str,
        context: Dict[str, Any],
    ):
        """Handle detected anomaly"""
        # Check cooldown
        cooldown_key = f"{metric_name}_{detection_method}"
        if cooldown_key in self.last_alerts:
            if datetime.utcnow() - self.last_alerts[cooldown_key] < self.alert_cooldown:
                return

        # Determine anomaly type and severity
        anomaly_type, severity = self._classify_anomaly(metric_name, value, context)

        # Get expected range
        analyzer = self.analyzers[metric_name]
        stats = analyzer.get_statistics()
        expected_range = (
            stats.get("mean", 0) - stats.get("std", 0),
            stats.get("mean", 0) + stats.get("std", 0),
        )

        # Create alert
        alert = AnomalyAlert(
            alert_id=f"{self.agent_id}_{metric_name}_{int(time.time())}",
            anomaly_type=anomaly_type,
            severity=severity,
            message=self._generate_alert_message(metric_name, value, detection_method, context),
            detected_at=datetime.utcnow(),
            metric_name=metric_name,
            current_value=value,
            expected_range=expected_range,
            confidence=confidence,
            context=context,
        )

        self.alerts.append(alert)
        self.last_alerts[cooldown_key] = datetime.utcnow()

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"Anomaly detected: {alert.message}")

    def _classify_anomaly(
        self, metric_name: str, value: float, context: Dict[str, Any]
    ) -> Tuple[AnomalyType, AnomalySeverity]:
        """Classify anomaly type and severity"""

        # Get baseline statistics
        analyzer = self.analyzers[metric_name]
        stats = analyzer.get_statistics()

        if metric_name == "execution_time":
            if value > stats.get("mean", 0) * 3:
                return AnomalyType.EXECUTION_TIME_ANOMALY, AnomalySeverity.HIGH
            elif value > stats.get("mean", 0) * 2:
                return AnomalyType.EXECUTION_TIME_ANOMALY, AnomalySeverity.MEDIUM
            else:
                return AnomalyType.PERFORMANCE_DEGRADATION, AnomalySeverity.LOW

        elif metric_name == "memory_usage":
            if value > 400:  # 400MB threshold
                return AnomalyType.MEMORY_SPIKE, AnomalySeverity.CRITICAL
            elif value > 300:
                return AnomalyType.MEMORY_SPIKE, AnomalySeverity.HIGH
            else:
                return AnomalyType.MEMORY_SPIKE, AnomalySeverity.MEDIUM

        elif metric_name == "error_rate":
            if value > 0.2:  # 20% error rate
                return AnomalyType.ERROR_RATE_SPIKE, AnomalySeverity.CRITICAL
            elif value > 0.1:
                return AnomalyType.ERROR_RATE_SPIKE, AnomalySeverity.HIGH
            else:
                return AnomalyType.ERROR_RATE_SPIKE, AnomalySeverity.MEDIUM

        elif metric_name == "convergence_confidence":
            if value < 0.3:
                return AnomalyType.CONVERGENCE_FAILURE, AnomalySeverity.HIGH
            elif value < 0.5:
                return AnomalyType.CONVERGENCE_FAILURE, AnomalySeverity.MEDIUM
            else:
                return AnomalyType.CONVERGENCE_FAILURE, AnomalySeverity.LOW

        elif metric_name == "expected_value":
            mean_val = stats.get("mean", 0)
            if abs(value) < abs(mean_val) * 0.5:  # 50% drop in quality
                return AnomalyType.VALUE_QUALITY_DROP, AnomalySeverity.HIGH
            elif abs(value) < abs(mean_val) * 0.7:
                return AnomalyType.VALUE_QUALITY_DROP, AnomalySeverity.MEDIUM
            else:
                return AnomalyType.VALUE_QUALITY_DROP, AnomalySeverity.LOW

        # Default classification
        return AnomalyType.PERFORMANCE_DEGRADATION, AnomalySeverity.MEDIUM

    def _generate_alert_message(
        self, metric_name: str, value: float, detection_method: str, context: Dict[str, Any]
    ) -> str:
        """Generate human-readable alert message"""
        analyzer = self.analyzers[metric_name]
        stats = analyzer.get_statistics()

        mean_val = stats.get("mean", 0)
        deviation = abs(value - mean_val) / mean_val if mean_val != 0 else 0

        base_msg = f"{metric_name.replace('_', ' ').title()} anomaly detected"

        if metric_name == "execution_time":
            return f"{base_msg}: {value:.2f}s (normal: {mean_val:.2f}s, {deviation:.1%} deviation)"
        elif metric_name == "memory_usage":
            return f"{base_msg}: {value:.1f}MB (normal: {mean_val:.1f}MB)"
        elif metric_name == "error_rate":
            return f"{base_msg}: {value:.1%} error rate (normal: {mean_val:.1%})"
        elif metric_name == "convergence_confidence":
            return f"{base_msg}: {value:.1%} confidence (normal: {mean_val:.1%})"
        else:
            return f"{base_msg}: {value:.3f} (normal: {mean_val:.3f})"

    def add_alert_handler(self, handler: Callable[[AnomalyAlert], None]):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)

    def get_active_alerts(
        self, severity_threshold: AnomalySeverity = AnomalySeverity.LOW
    ) -> List[AnomalyAlert]:
        """Get currently active alerts above severity threshold"""
        severity_order = {
            AnomalySeverity.LOW: 0,
            AnomalySeverity.MEDIUM: 1,
            AnomalySeverity.HIGH: 2,
            AnomalySeverity.CRITICAL: 3,
        }

        threshold_level = severity_order[severity_threshold]

        return [
            alert
            for alert in self.alerts
            if (alert.resolved_at is None and severity_order[alert.severity] >= threshold_level)
        ]

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and alert.resolved_at is None:
                alert.resolved_at = datetime.utcnow()
                logger.info(f"Resolved alert {alert_id}")
                return True
        return False

    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment"""
        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AnomalySeverity.CRITICAL]
        high_alerts = [a for a in active_alerts if a.severity == AnomalySeverity.HIGH]

        # Calculate REAL health score based on actual system metrics
        health_score = await self._calculate_real_health_score(
            critical_alerts, high_alerts, active_alerts
        )

        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        elif health_score >= 25:
            health_status = "poor"
        else:
            health_status = "critical"

        # Get metric summaries
        metric_summaries = {}
        for metric_name, analyzer in self.analyzers.items():
            stats = analyzer.get_statistics()
            if stats:
                metric_summaries[metric_name] = {
                    "current": stats.get("mean", 0),
                    "trend": self._calculate_trend(analyzer),
                    "stability": self._calculate_stability(analyzer),
                }

        return {
            "agent_id": self.agent_id,
            "health_score": health_score,
            "health_status": health_status,
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "high_alerts": len(high_alerts),
            "metric_summaries": metric_summaries,
            "last_updated": datetime.utcnow().isoformat(),
        }

    async def _calculate_real_health_score(
        self, critical_alerts: List, high_alerts: List, active_alerts: List
    ) -> float:
        """Calculate real health score based on actual system performance metrics"""
        try:
            # Start with base score
            base_score = 100.0

            # Get real system metrics
            from ...data.database.performance_monitor import PerformanceMonitor

            perf_monitor = PerformanceMonitor()

            # Get actual performance metrics
            metrics = await perf_monitor.get_current_metrics()

            # Deduct based on real performance issues
            if metrics.get("cpu_usage_percent", 0) > 80:
                base_score -= 15
            elif metrics.get("cpu_usage_percent", 0) > 60:
                base_score -= 8

            if metrics.get("memory_usage_percent", 0) > 85:
                base_score -= 20
            elif metrics.get("memory_usage_percent", 0) > 70:
                base_score -= 10

            # Database performance impact
            db_response_time = metrics.get("avg_db_response_ms", 0)
            if db_response_time > 1000:
                base_score -= 25
            elif db_response_time > 500:
                base_score -= 15

            # Error rate impact
            error_rate = metrics.get("error_rate_percent", 0)
            if error_rate > 5:
                base_score -= 30
            elif error_rate > 2:
                base_score -= 15

            # Alert-based deductions (real impact assessment)
            base_score -= len(critical_alerts) * 20  # Critical alerts have major impact
            base_score -= len(high_alerts) * 10  # High alerts have moderate impact
            base_score -= (
                max(0, len(active_alerts) - len(critical_alerts) - len(high_alerts)) * 3
            )  # Other alerts

            # Ensure score stays within bounds
            return max(0.0, min(100.0, base_score))

        except Exception as e:
            logger.error(f"Failed to calculate real health score: {e}")
            # Fallback to conservative scoring based on alerts only
            fallback_score = 100.0
            fallback_score -= len(critical_alerts) * 25
            fallback_score -= len(high_alerts) * 12
            fallback_score -= len(active_alerts) * 3
            return max(0.0, fallback_score)

    def _calculate_trend(self, analyzer: TimeSeriesAnalyzer) -> str:
        """Calculate trend direction for a metric"""
        if len(analyzer.samples) < 10:
            return "insufficient_data"

        values = [s.value for s in analyzer.samples]
        recent_values = values[-5:]
        older_values = values[-10:-5]

        recent_mean = statistics.mean(recent_values)
        older_mean = statistics.mean(older_values)

        if abs(recent_mean - older_mean) < 0.01:
            return "stable"
        elif recent_mean > older_mean:
            return "increasing"
        else:
            return "decreasing"

    def _calculate_stability(self, analyzer: TimeSeriesAnalyzer) -> float:
        """Calculate stability score (0-1) for a metric"""
        if len(analyzer.samples) < 5:
            return 1.0

        values = [s.value for s in analyzer.samples]
        if len(values) == 1:
            return 1.0

        std_val = statistics.stdev(values)
        mean_val = statistics.mean(values)

        if mean_val == 0:
            return 1.0 if std_val == 0 else 0.0

        coefficient_of_variation = std_val / abs(mean_val)
        stability = max(0, 1 - coefficient_of_variation)

        return min(stability, 1.0)

    def export_alerts(self, format: str = "json") -> str:
        """Export alerts in specified format"""
        if format == "json":
            return json.dumps([alert.to_dict() for alert in self.alerts], indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


class MCTSMonitoringDashboard:
    """Real-time monitoring dashboard for MCTS agent"""

    def __init__(self, anomaly_detector: AnomalyDetector):
        self.anomaly_detector = anomaly_detector
        self.dashboard_data = {}
        self.update_interval = 30  # seconds
        self._running = False

    async def start_monitoring(self):
        """Start the monitoring dashboard"""
        self._running = True
        logger.info("Started MCTS monitoring dashboard")

        while self._running:
            try:
                await self._update_dashboard()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Dashboard update failed: {e}")
                await asyncio.sleep(self.update_interval)

    def stop_monitoring(self):
        """Stop the monitoring dashboard"""
        self._running = False
        logger.info("Stopped MCTS monitoring dashboard")

    async def _update_dashboard(self):
        """Update dashboard data"""
        health = await self.anomaly_detector.get_system_health()
        active_alerts = self.anomaly_detector.get_active_alerts()

        self.dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": health,
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "metrics": self._get_current_metrics(),
            "trends": self._get_trend_analysis(),
        }

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        metrics = {}
        for metric_name, analyzer in self.anomaly_detector.analyzers.items():
            stats = analyzer.get_statistics()
            if stats:
                metrics[metric_name] = {
                    "current": stats.get("mean", 0),
                    "min": stats.get("min", 0),
                    "max": stats.get("max", 0),
                    "std": stats.get("std", 0),
                }
        return metrics

    def _get_trend_analysis(self) -> Dict[str, str]:
        """Get trend analysis for all metrics"""
        trends = {}
        for metric_name, analyzer in self.anomaly_detector.analyzers.items():
            trends[metric_name] = self.anomaly_detector._calculate_trend(analyzer)
        return trends

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data
