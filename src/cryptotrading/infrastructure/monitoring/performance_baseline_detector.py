"""
AI-Powered Performance Baseline Auto-Detection
Automatically learns normal performance patterns and detects degradation
"""

import numpy as np
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import statistics
import asyncio

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline for a metric"""
    metric_name: str
    baseline_value: float
    normal_range: Tuple[float, float]
    trend_direction: str  # "improving", "stable", "degrading"
    confidence: float
    samples_used: int
    last_updated: datetime
    seasonal_patterns: Dict[str, float]  # hour_of_day -> expected_multiplier
    business_patterns: Dict[str, float]  # day_of_week -> expected_multiplier


@dataclass
class PerformanceDegradation:
    """Detected performance degradation"""
    id: str
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_percent: float
    severity: str
    detection_time: datetime
    context: Dict[str, Any]
    suggested_actions: List[str]


class PerformanceBaselineDetector:
    """AI-powered performance baseline detection and monitoring"""
    
    def __init__(self, db_path: str = "cryptotrading.db"):
        self.db_path = db_path
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        self.models: Dict[str, Any] = {}
        self.learning_window = timedelta(days=7)  # Learn from 7 days of data
        self.update_interval = timedelta(hours=1)   # Update baselines every hour
        self.last_update = {}
        self._init_database()
        
    def _init_database(self):
        """Initialize database for baseline tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_baselines (
                metric_name TEXT PRIMARY KEY,
                baseline_value REAL,
                normal_min REAL,
                normal_max REAL,
                trend_direction TEXT,
                confidence REAL,
                samples_used INTEGER,
                last_updated DATETIME,
                seasonal_patterns TEXT,
                business_patterns TEXT,
                model_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_degradations (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                current_value REAL,
                baseline_value REAL,
                degradation_percent REAL,
                severity TEXT,
                detection_time DATETIME,
                context TEXT,
                suggested_actions TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metric_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                hour_of_day INTEGER,
                day_of_week INTEGER,
                context TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Load existing baselines
        self._load_baselines()
    
    def add_performance_metric(self, metric_name: str, value: float, context: Dict = None):
        """Add a performance metric sample"""
        timestamp = datetime.now()
        context = context or {}
        
        # Store sample
        sample = {
            'timestamp': timestamp,
            'value': value,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'context': context
        }
        
        self.metric_history[metric_name].append(sample)
        
        # Store in database
        self._store_metric_sample(metric_name, sample)
        
        # Check if we need to update baseline
        if self._should_update_baseline(metric_name):
            baseline = self._learn_baseline(metric_name)
            if baseline:
                self.baselines[metric_name] = baseline
                self._store_baseline(baseline)
        
        # Check for performance degradation
        degradation = self._detect_degradation(metric_name, value, context)
        if degradation:
            self._store_degradation(degradation)
            logger.warning(f"Performance degradation detected: {degradation.metric_name}")
            return degradation
        
        return None
    
    def _should_update_baseline(self, metric_name: str) -> bool:
        """Determine if baseline should be updated"""
        if metric_name not in self.last_update:
            return len(self.metric_history[metric_name]) >= 50  # Minimum samples
        
        time_since_update = datetime.now() - self.last_update[metric_name]
        return time_since_update >= self.update_interval
    
    def _learn_baseline(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """Learn performance baseline from historical data"""
        history = list(self.metric_history[metric_name])
        
        if len(history) < 50:  # Need minimum samples
            return None
        
        try:
            values = [h['value'] for h in history]
            timestamps = [h['timestamp'] for h in history]
            hours = [h['hour_of_day'] for h in history]
            days = [h['day_of_week'] for h in history]
            
            # Basic statistical baseline
            mean_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            
            # Calculate normal range (mean ± 2 std dev)
            normal_min = mean_value - 2 * std_dev
            normal_max = mean_value + 2 * std_dev
            
            # Detect trend
            trend = self._detect_trend(values, timestamps)
            
            # Learn seasonal patterns (hourly)
            seasonal_patterns = self._learn_seasonal_patterns(values, hours)
            
            # Learn business patterns (daily)
            business_patterns = self._learn_business_patterns(values, days)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_baseline_confidence(values, std_dev)
            
            baseline = PerformanceBaseline(
                metric_name=metric_name,
                baseline_value=mean_value,
                normal_range=(normal_min, normal_max),
                trend_direction=trend,
                confidence=confidence,
                samples_used=len(history),
                last_updated=datetime.now(),
                seasonal_patterns=seasonal_patterns,
                business_patterns=business_patterns
            )
            
            self.last_update[metric_name] = datetime.now()
            return baseline
            
        except Exception as e:
            logger.error(f"Failed to learn baseline for {metric_name}: {e}")
            return None
    
    def _detect_trend(self, values: List[float], timestamps: List[datetime]) -> str:
        """Detect performance trend"""
        if len(values) < 10:
            return "stable"
        
        try:
            # Convert timestamps to numeric values (days since first timestamp)
            first_time = timestamps[0]
            x = [(t - first_time).total_seconds() / 86400 for t in timestamps]  # Days
            y = values
            
            if HAS_SKLEARN:
                # Use linear regression to detect trend
                X = np.array(x).reshape(-1, 1)
                y_arr = np.array(y)
                
                model = LinearRegression()
                model.fit(X, y_arr)
                
                slope = model.coef_[0]
                
                # Determine trend based on slope
                if abs(slope) < 0.01:  # Very small slope
                    return "stable"
                elif slope > 0:
                    return "degrading" if self._is_performance_metric_lower_better(values) else "improving"
                else:
                    return "improving" if self._is_performance_metric_lower_better(values) else "degrading"
            else:
                # Simple trend detection using first and last quartiles
                q1_values = values[:len(values)//4]
                q4_values = values[len(values)*3//4:]
                
                q1_avg = statistics.mean(q1_values)
                q4_avg = statistics.mean(q4_values)
                
                change_percent = (q4_avg - q1_avg) / q1_avg if q1_avg != 0 else 0
                
                if abs(change_percent) < 0.05:  # Less than 5% change
                    return "stable"
                elif change_percent > 0:
                    return "degrading" if self._is_performance_metric_lower_better(values) else "improving"
                else:
                    return "improving" if self._is_performance_metric_lower_better(values) else "degrading"
                
        except Exception as e:
            logger.warning(f"Trend detection failed: {e}")
            return "stable"
    
    def _is_performance_metric_lower_better(self, values: List[float]) -> bool:
        """Determine if lower values indicate better performance"""
        # Heuristic: if all values are positive and relatively small, assume lower is better
        # This works for response times, error rates, etc.
        if all(v >= 0 for v in values) and statistics.mean(values) < 1000:
            return True
        return False
    
    def _learn_seasonal_patterns(self, values: List[float], hours: List[int]) -> Dict[str, float]:
        """Learn hourly seasonal patterns"""
        hourly_values = defaultdict(list)
        
        for value, hour in zip(values, hours):
            hourly_values[hour].append(value)
        
        patterns = {}
        overall_mean = statistics.mean(values)
        
        for hour in range(24):
            if hour in hourly_values and len(hourly_values[hour]) > 2:
                hour_mean = statistics.mean(hourly_values[hour])
                multiplier = hour_mean / overall_mean if overall_mean != 0 else 1.0
                patterns[str(hour)] = multiplier
            else:
                patterns[str(hour)] = 1.0
        
        return patterns
    
    def _learn_business_patterns(self, values: List[float], days: List[int]) -> Dict[str, float]:
        """Learn daily business patterns"""
        daily_values = defaultdict(list)
        
        for value, day in zip(values, days):
            daily_values[day].append(value)
        
        patterns = {}
        overall_mean = statistics.mean(values)
        
        for day in range(7):  # 0=Monday, 6=Sunday
            if day in daily_values and len(daily_values[day]) > 2:
                day_mean = statistics.mean(daily_values[day])
                multiplier = day_mean / overall_mean if overall_mean != 0 else 1.0
                patterns[str(day)] = multiplier
            else:
                patterns[str(day)] = 1.0
        
        return patterns
    
    def _calculate_baseline_confidence(self, values: List[float], std_dev: float) -> float:
        """Calculate confidence in the baseline"""
        # Factors affecting confidence:
        # 1. Sample size
        # 2. Data stability (low std dev is good)
        # 3. Data age (more recent is better)
        
        sample_score = min(len(values) / 500, 1.0)  # Max score at 500 samples
        stability_score = max(0, 1.0 - (std_dev / statistics.mean(values))) if statistics.mean(values) != 0 else 0
        
        confidence = (sample_score + stability_score) / 2
        return max(0.1, min(1.0, confidence))
    
    def _detect_degradation(self, metric_name: str, current_value: float, context: Dict) -> Optional[PerformanceDegradation]:
        """Detect performance degradation"""
        if metric_name not in self.baselines:
            return None
        
        baseline = self.baselines[metric_name]
        
        # Get expected value based on time of day and day of week
        now = datetime.now()
        hour_key = str(now.hour)
        day_key = str(now.weekday())
        
        seasonal_multiplier = baseline.seasonal_patterns.get(hour_key, 1.0)
        business_multiplier = baseline.business_patterns.get(day_key, 1.0)
        
        expected_value = baseline.baseline_value * seasonal_multiplier * business_multiplier
        expected_range = (
            baseline.normal_range[0] * seasonal_multiplier * business_multiplier,
            baseline.normal_range[1] * seasonal_multiplier * business_multiplier
        )
        
        # Check if current value is outside normal range
        if expected_range[0] <= current_value <= expected_range[1]:
            return None  # Within normal range
        
        # Calculate degradation percentage
        degradation_percent = abs((current_value - expected_value) / expected_value) * 100 if expected_value != 0 else 0
        
        # Determine severity
        if degradation_percent > 100:
            severity = "CRITICAL"
        elif degradation_percent > 50:
            severity = "HIGH"
        elif degradation_percent > 25:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Generate suggested actions
        suggested_actions = self._get_degradation_suggestions(metric_name, degradation_percent, context)
        
        degradation = PerformanceDegradation(
            id=f"perf_{metric_name}_{datetime.now().timestamp()}",
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=expected_value,
            degradation_percent=degradation_percent,
            severity=severity,
            detection_time=datetime.now(),
            context=context,
            suggested_actions=suggested_actions
        )
        
        return degradation
    
    def _get_degradation_suggestions(self, metric_name: str, degradation_percent: float, context: Dict) -> List[str]:
        """Get suggestions for addressing performance degradation"""
        suggestions = []
        
        # General suggestions based on metric type
        if "response_time" in metric_name.lower() or "latency" in metric_name.lower():
            suggestions.extend([
                "Check database query performance",
                "Review recent code deployments",
                "Monitor server CPU and memory usage",
                "Check network connectivity",
                "Review cache hit rates"
            ])
        elif "memory" in metric_name.lower():
            suggestions.extend([
                "Check for memory leaks",
                "Review object lifecycle management",
                "Monitor garbage collection performance",
                "Check for large data structures",
                "Consider memory optimization"
            ])
        elif "cpu" in metric_name.lower():
            suggestions.extend([
                "Profile CPU-intensive operations",
                "Check for infinite loops or excessive recursion",
                "Review algorithm efficiency",
                "Monitor concurrent processes",
                "Consider load balancing"
            ])
        elif "error" in metric_name.lower():
            suggestions.extend([
                "Review error logs for patterns",
                "Check service dependencies",
                "Verify configuration settings",
                "Review recent changes",
                "Check external API status"
            ])
        
        # Severity-based suggestions
        if degradation_percent > 100:
            suggestions.insert(0, "URGENT: Immediate investigation required")
        elif degradation_percent > 50:
            suggestions.insert(0, "High priority: Investigate within 1 hour")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _store_metric_sample(self, metric_name: str, sample: Dict):
        """Store metric sample in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metric_samples 
            (metric_name, value, timestamp, hour_of_day, day_of_week, context)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metric_name,
            sample['value'],
            sample['timestamp'].isoformat(),
            sample['hour_of_day'],
            sample['day_of_week'],
            json.dumps(sample['context'])
        ))
        
        conn.commit()
        conn.close()
    
    def _store_baseline(self, baseline: PerformanceBaseline):
        """Store baseline in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO performance_baselines
            (metric_name, baseline_value, normal_min, normal_max, trend_direction,
             confidence, samples_used, last_updated, seasonal_patterns, business_patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            baseline.metric_name,
            baseline.baseline_value,
            baseline.normal_range[0],
            baseline.normal_range[1],
            baseline.trend_direction,
            baseline.confidence,
            baseline.samples_used,
            baseline.last_updated.isoformat(),
            json.dumps(baseline.seasonal_patterns),
            json.dumps(baseline.business_patterns)
        ))
        
        conn.commit()
        conn.close()
    
    def _store_degradation(self, degradation: PerformanceDegradation):
        """Store degradation in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_degradations
            (id, metric_name, current_value, baseline_value, degradation_percent,
             severity, detection_time, context, suggested_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            degradation.id,
            degradation.metric_name,
            degradation.current_value,
            degradation.baseline_value,
            degradation.degradation_percent,
            degradation.severity,
            degradation.detection_time.isoformat(),
            json.dumps(degradation.context),
            json.dumps(degradation.suggested_actions)
        ))
        
        conn.commit()
        conn.close()
    
    def _load_baselines(self):
        """Load existing baselines from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM performance_baselines')
        
        for row in cursor.fetchall():
            baseline = PerformanceBaseline(
                metric_name=row[0],
                baseline_value=row[1],
                normal_range=(row[2], row[3]),
                trend_direction=row[4],
                confidence=row[5],
                samples_used=row[6],
                last_updated=datetime.fromisoformat(row[7]),
                seasonal_patterns=json.loads(row[8] or '{}'),
                business_patterns=json.loads(row[9] or '{}')
            )
            self.baselines[baseline.metric_name] = baseline
            self.last_update[baseline.metric_name] = baseline.last_updated
        
        conn.close()
        logger.info(f"Loaded {len(self.baselines)} performance baselines")
    
    def get_baseline_insights(self) -> Dict[str, Any]:
        """Get insights about performance baselines"""
        insights = {
            "total_baselines": len(self.baselines),
            "high_confidence_baselines": 0,
            "trending_metrics": {"improving": 0, "stable": 0, "degrading": 0},
            "recent_degradations": 0,
            "baselines_by_confidence": {"high": [], "medium": [], "low": []},
            "metrics_with_seasonal_patterns": 0,
            "has_ml_support": HAS_SKLEARN
        }
        
        # Analyze baselines
        for baseline in self.baselines.values():
            if baseline.confidence > 0.8:
                insights["high_confidence_baselines"] += 1
                insights["baselines_by_confidence"]["high"].append(baseline.metric_name)
            elif baseline.confidence > 0.5:
                insights["baselines_by_confidence"]["medium"].append(baseline.metric_name)
            else:
                insights["baselines_by_confidence"]["low"].append(baseline.metric_name)
            
            insights["trending_metrics"][baseline.trend_direction] += 1
            
            # Check for seasonal patterns (non-uniform hourly patterns)
            hourly_values = list(baseline.seasonal_patterns.values())
            if max(hourly_values) - min(hourly_values) > 0.2:  # 20% variation
                insights["metrics_with_seasonal_patterns"] += 1
        
        # Get recent degradations
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        cursor.execute('''
            SELECT COUNT(*) FROM performance_degradations 
            WHERE detection_time > ? AND resolved = FALSE
        ''', (cutoff,))
        
        insights["recent_degradations"] = cursor.fetchone()[0]
        
        conn.close()
        
        return insights


# Global instance
performance_detector = PerformanceBaselineDetector()


def track_performance(metric_name: str, value: float, context: Dict = None) -> Optional[PerformanceDegradation]:
    """Convenience function to track performance metrics"""
    return performance_detector.add_performance_metric(metric_name, value, context)