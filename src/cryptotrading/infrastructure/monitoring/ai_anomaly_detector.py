"""
AI-Powered Anomaly Detection for Real-Time System Monitoring
Uses machine learning algorithms to detect unusual patterns and behaviors
"""

import numpy as np
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
import asyncio
from dataclasses import dataclass, asdict
import statistics
import math

try:
    # Try to import sklearn for real ML algorithms
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Represents an anomaly detected by the system"""
    id: str
    timestamp: datetime
    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    anomaly_score: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str  # performance, error_rate, resource_usage, business_logic
    description: str
    suggested_action: str
    context: Dict[str, Any]


class AIAnomalyDetector:
    """AI-powered anomaly detection using multiple algorithms"""
    
    def __init__(self, db_path: str = "cryptotrading.db"):
        self.db_path = db_path
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.models = {}
        self.scalers = {}
        self.baseline_stats = {}
        self.alert_cooldown = defaultdict(lambda: datetime.min)
        self._init_database()
        
    def _init_database(self):
        """Initialize database for anomaly tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomaly_events (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                value REAL,
                expected_min REAL,
                expected_max REAL,
                anomaly_score REAL,
                severity TEXT,
                category TEXT,
                description TEXT,
                suggested_action TEXT,
                context TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metric_baselines (
                metric_name TEXT PRIMARY KEY,
                mean_value REAL,
                std_deviation REAL,
                min_value REAL,
                max_value REAL,
                percentile_95 REAL,
                percentile_99 REAL,
                last_updated DATETIME,
                sample_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_metric(self, metric_name: str, value: float, context: Dict = None):
        """Add a metric value for monitoring"""
        timestamp = datetime.now()
        
        # Store in memory for real-time analysis
        self.metric_history[metric_name].append({
            'timestamp': timestamp,
            'value': value,
            'context': context or {}
        })
        
        # Update baseline statistics
        self._update_baseline_stats(metric_name)
        
        # Detect anomalies
        anomaly = self._detect_anomaly(metric_name, value, context)
        
        if anomaly:
            self._store_anomaly(anomaly)
            logger.warning(f"Anomaly detected: {anomaly.description}")
        
        return anomaly
    
    def _detect_anomaly(self, metric_name: str, value: float, context: Dict = None) -> Optional[AnomalyEvent]:
        """Detect anomalies using multiple algorithms"""
        history = list(self.metric_history[metric_name])
        
        if len(history) < 10:  # Need minimum data points
            return None
        
        anomalies = []
        
        # 1. Statistical anomaly detection (Z-score)
        statistical_anomaly = self._detect_statistical_anomaly(metric_name, value, history)
        if statistical_anomaly:
            anomalies.append(statistical_anomaly)
        
        # 2. Time series anomaly detection
        time_series_anomaly = self._detect_time_series_anomaly(metric_name, value, history)
        if time_series_anomaly:
            anomalies.append(time_series_anomaly)
        
        # 3. ML-based anomaly detection (if sklearn available)
        if HAS_SKLEARN:
            ml_anomaly = self._detect_ml_anomaly(metric_name, value, history)
            if ml_anomaly:
                anomalies.append(ml_anomaly)
        
        # 4. Business logic anomaly detection
        business_anomaly = self._detect_business_anomaly(metric_name, value, context)
        if business_anomaly:
            anomalies.append(business_anomaly)
        
        # Return the highest severity anomaly
        if anomalies:
            return max(anomalies, key=lambda a: self._severity_to_score(a.severity))
        
        return None
    
    def _detect_statistical_anomaly(self, metric_name: str, value: float, history: List[Dict]) -> Optional[AnomalyEvent]:
        """Detect anomalies using statistical methods"""
        values = [h['value'] for h in history]
        
        if len(values) < 10:
            return None
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_dev == 0:
            return None
        
        z_score = abs((value - mean) / std_dev)
        
        # Determine severity based on Z-score
        if z_score > 4:
            severity = "CRITICAL"
            description = f"{metric_name} extremely abnormal: {value:.2f} (z-score: {z_score:.2f})"
        elif z_score > 3:
            severity = "HIGH"
            description = f"{metric_name} highly abnormal: {value:.2f} (z-score: {z_score:.2f})"
        elif z_score > 2:
            severity = "MEDIUM"
            description = f"{metric_name} moderately abnormal: {value:.2f} (z-score: {z_score:.2f})"
        else:
            return None  # Not an anomaly
        
        expected_min = mean - 2 * std_dev
        expected_max = mean + 2 * std_dev
        
        return AnomalyEvent(
            id=f"stat_{metric_name}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            expected_range=(expected_min, expected_max),
            anomaly_score=z_score / 4.0,  # Normalize to 0-1
            severity=severity,
            category="statistical",
            description=description,
            suggested_action=self._get_suggested_action(metric_name, "statistical"),
            context={"z_score": z_score, "mean": mean, "std_dev": std_dev}
        )
    
    def _detect_time_series_anomaly(self, metric_name: str, value: float, history: List[Dict]) -> Optional[AnomalyEvent]:
        """Detect anomalies in time series patterns"""
        if len(history) < 20:
            return None
        
        # Get recent values for trend analysis
        recent_values = [h['value'] for h in history[-20:]]
        timestamps = [h['timestamp'] for h in history[-20:]]
        
        # Calculate moving average
        window_size = min(10, len(recent_values) // 2)
        moving_avg = []
        for i in range(window_size - 1, len(recent_values)):
            avg = statistics.mean(recent_values[i - window_size + 1:i + 1])
            moving_avg.append(avg)
        
        if not moving_avg:
            return None
        
        # Compare current value to recent moving average
        recent_avg = moving_avg[-1]
        deviation = abs(value - recent_avg) / (recent_avg if recent_avg != 0 else 1)
        
        # Detect sudden spikes or drops
        if deviation > 0.5:  # 50% deviation
            severity = "HIGH" if deviation > 1.0 else "MEDIUM"
            direction = "spike" if value > recent_avg else "drop"
            
            return AnomalyEvent(
                id=f"ts_{metric_name}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                expected_range=(recent_avg * 0.8, recent_avg * 1.2),
                anomaly_score=min(deviation, 1.0),
                severity=severity,
                category="time_series",
                description=f"Sudden {direction} in {metric_name}: {value:.2f} vs expected {recent_avg:.2f}",
                suggested_action=self._get_suggested_action(metric_name, "time_series"),
                context={"recent_avg": recent_avg, "deviation": deviation, "pattern": direction}
            )
        
        return None
    
    def _detect_ml_anomaly(self, metric_name: str, value: float, history: List[Dict]) -> Optional[AnomalyEvent]:
        """Detect anomalies using machine learning (Isolation Forest)"""
        if not HAS_SKLEARN or len(history) < 50:
            return None
        
        try:
            # Prepare features: value, hour of day, day of week, trend
            features = []
            values = []
            
            for i, h in enumerate(history):
                timestamp = h['timestamp']
                val = h['value']
                
                # Create features
                hour = timestamp.hour
                day_of_week = timestamp.weekday()
                trend = val - history[max(0, i-5)]['value'] if i > 5 else 0
                
                features.append([val, hour, day_of_week, trend])
                values.append(val)
            
            features = np.array(features)
            
            # Scale features
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
                features_scaled = self.scalers[metric_name].fit_transform(features)
            else:
                features_scaled = self.scalers[metric_name].transform(features)
            
            # Train/update Isolation Forest
            if metric_name not in self.models:
                self.models[metric_name] = IsolationForest(contamination=0.1, random_state=42)
                self.models[metric_name].fit(features_scaled)
            
            # Prepare current features
            now = datetime.now()
            current_features = [[
                value,
                now.hour,
                now.weekday(),
                value - history[-1]['value'] if history else 0
            ]]
            current_features_scaled = self.scalers[metric_name].transform(current_features)
            
            # Predict anomaly
            anomaly_score = self.models[metric_name].decision_function(current_features_scaled)[0]
            is_anomaly = self.models[metric_name].predict(current_features_scaled)[0] == -1
            
            if is_anomaly:
                # Convert decision function score to probability-like score
                normalized_score = max(0, min(1, (0.5 - anomaly_score) / 1.0))
                
                severity = "HIGH" if normalized_score > 0.8 else "MEDIUM"
                
                return AnomalyEvent(
                    id=f"ml_{metric_name}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    value=value,
                    expected_range=(min(values), max(values)),
                    anomaly_score=normalized_score,
                    severity=severity,
                    category="machine_learning",
                    description=f"ML detected anomaly in {metric_name}: {value:.2f} (score: {normalized_score:.3f})",
                    suggested_action=self._get_suggested_action(metric_name, "machine_learning"),
                    context={"ml_score": anomaly_score, "model_type": "IsolationForest"}
                )
        
        except Exception as e:
            logger.warning(f"ML anomaly detection failed for {metric_name}: {e}")
        
        return None
    
    def _detect_business_anomaly(self, metric_name: str, value: float, context: Dict = None) -> Optional[AnomalyEvent]:
        """Detect business logic anomalies"""
        context = context or {}
        
        # Define business rules for specific metrics
        business_rules = {
            "api_response_time": lambda v: ("HIGH", "API response time too high") if v > 5000 else None,
            "error_rate": lambda v: ("CRITICAL", "Error rate critically high") if v > 0.1 else ("HIGH", "Error rate elevated") if v > 0.05 else None,
            "memory_usage": lambda v: ("CRITICAL", "Memory usage critical") if v > 0.9 else ("HIGH", "Memory usage high") if v > 0.8 else None,
            "cpu_usage": lambda v: ("HIGH", "CPU usage high") if v > 0.9 else ("MEDIUM", "CPU usage elevated") if v > 0.8 else None,
            "trade_volume": lambda v: ("MEDIUM", "Unusually low trade volume") if v < 1000 else None,
            "active_connections": lambda v: ("HIGH", "Too many active connections") if v > 1000 else None,
            "disk_usage": lambda v: ("CRITICAL", "Disk space critical") if v > 0.95 else ("HIGH", "Disk space low") if v > 0.85 else None
        }
        
        if metric_name in business_rules:
            result = business_rules[metric_name](value)
            if result:
                severity, description = result
                
                return AnomalyEvent(
                    id=f"biz_{metric_name}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    value=value,
                    expected_range=(0, 1),  # Will be updated based on metric type
                    anomaly_score=0.8,
                    severity=severity,
                    category="business_logic",
                    description=f"Business rule violation: {description} (value: {value})",
                    suggested_action=self._get_suggested_action(metric_name, "business_logic"),
                    context=context
                )
        
        return None
    
    def _get_suggested_action(self, metric_name: str, category: str) -> str:
        """Get suggested action for anomaly"""
        suggestions = {
            ("api_response_time", "statistical"): "Check API performance, database queries, and network latency",
            ("api_response_time", "business_logic"): "Scale up servers or optimize slow endpoints",
            ("error_rate", "statistical"): "Investigate recent error logs and check system health",
            ("error_rate", "business_logic"): "Immediate investigation required - system may be failing",
            ("memory_usage", "statistical"): "Check for memory leaks and optimize memory usage",
            ("memory_usage", "business_logic"): "Restart services or scale up memory allocation",
            ("cpu_usage", "statistical"): "Investigate high CPU processes and optimize algorithms",
            ("cpu_usage", "business_logic"): "Scale up CPU resources or optimize workload distribution"
        }
        
        key = (metric_name, category)
        return suggestions.get(key, f"Investigate {metric_name} anomaly - check logs and system resources")
    
    def _severity_to_score(self, severity: str) -> int:
        """Convert severity to numeric score for comparison"""
        scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return scores.get(severity, 0)
    
    def _update_baseline_stats(self, metric_name: str):
        """Update baseline statistics for a metric"""
        history = list(self.metric_history[metric_name])
        if len(history) < 10:
            return
        
        values = [h['value'] for h in history]
        
        stats = {
            'mean_value': statistics.mean(values),
            'std_deviation': statistics.stdev(values) if len(values) > 1 else 0,
            'min_value': min(values),
            'max_value': max(values),
            'sample_count': len(values)
        }
        
        # Calculate percentiles
        sorted_values = sorted(values)
        stats['percentile_95'] = sorted_values[int(0.95 * len(sorted_values))]
        stats['percentile_99'] = sorted_values[int(0.99 * len(sorted_values))]
        
        self.baseline_stats[metric_name] = stats
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO metric_baselines 
            (metric_name, mean_value, std_deviation, min_value, max_value, 
             percentile_95, percentile_99, last_updated, sample_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
        ''', (
            metric_name,
            stats['mean_value'],
            stats['std_deviation'],
            stats['min_value'],
            stats['max_value'],
            stats['percentile_95'],
            stats['percentile_99'],
            stats['sample_count']
        ))
        
        conn.commit()
        conn.close()
    
    def _store_anomaly(self, anomaly: AnomalyEvent):
        """Store anomaly in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO anomaly_events
            (id, metric_name, value, expected_min, expected_max, anomaly_score,
             severity, category, description, suggested_action, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            anomaly.id,
            anomaly.metric_name,
            anomaly.value,
            anomaly.expected_range[0],
            anomaly.expected_range[1],
            anomaly.anomaly_score,
            anomaly.severity,
            anomaly.category,
            anomaly.description,
            anomaly.suggested_action,
            json.dumps(anomaly.context)
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_anomalies(self, hours: int = 24) -> List[Dict]:
        """Get active anomalies from the last N hours"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT * FROM anomaly_events
            WHERE timestamp > ? AND resolved = FALSE
            ORDER BY anomaly_score DESC, timestamp DESC
        ''', (cutoff,))
        
        anomalies = []
        for row in cursor.fetchall():
            anomalies.append({
                "id": row[0],
                "timestamp": row[1],
                "metric_name": row[2],
                "value": row[3],
                "expected_range": [row[4], row[5]],
                "anomaly_score": row[6],
                "severity": row[7],
                "category": row[8],
                "description": row[9],
                "suggested_action": row[10],
                "context": json.loads(row[11] or '{}')
            })
        
        conn.close()
        return anomalies
    
    def get_anomaly_insights(self, hours: int = 24) -> Dict[str, Any]:
        """Get insights about anomalies"""
        active_anomalies = self.get_active_anomalies(hours)
        
        # Group by severity and category
        by_severity = defaultdict(int)
        by_category = defaultdict(int)
        by_metric = defaultdict(int)
        
        for anomaly in active_anomalies:
            by_severity[anomaly["severity"]] += 1
            by_category[anomaly["category"]] += 1
            by_metric[anomaly["metric_name"]] += 1
        
        return {
            "total_active_anomalies": len(active_anomalies),
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "by_metric": dict(by_metric),
            "ml_models_active": len(self.models) if HAS_SKLEARN else 0,
            "metrics_monitored": len(self.metric_history),
            "has_ml_support": HAS_SKLEARN,
            "recent_anomalies": active_anomalies[:5]
        }


# Global instance
anomaly_detector = AIAnomalyDetector()


def detect_anomaly(metric_name: str, value: float, context: Dict = None) -> Optional[AnomalyEvent]:
    """Convenience function to detect anomalies"""
    return anomaly_detector.add_metric(metric_name, value, context)