"""
System Metrics Database Persistence
Stores monitoring data for historical analysis and alerting
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
from collections import defaultdict

from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)


class MetricsPersistence:
    """
    Persists system metrics to database for historical tracking
    Provides write-through caching and batch operations
    """
    
    def __init__(self, db: Optional[UnifiedDatabase] = None):
        self.db = db or UnifiedDatabase()
        self._batch_buffer = defaultdict(list)
        self._batch_size = 100
        self._flush_interval = 30  # seconds
        self._flush_task = None
        
    async def start(self):
        """Start the metrics persistence service"""
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Metrics persistence service started")
        
    async def stop(self):
        """Stop the service and flush remaining metrics"""
        if self._flush_task:
            self._flush_task.cancel()
        await self._flush_buffer()
        logger.info("Metrics persistence service stopped")
        
    async def record_metric(self,
                          metric_name: str,
                          metric_value: float,
                          metric_type: str = "gauge",
                          tags: Optional[Dict[str, str]] = None,
                          service_name: Optional[str] = None,
                          environment: Optional[str] = None):
        """Record a single metric"""
        metric = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_type": metric_type,
            "tags": json.dumps(tags) if tags else None,
            "timestamp": datetime.utcnow(),
            "service_name": service_name or "cryptotrading",
            "environment": environment or "production"
        }
        
        # Add to batch buffer
        self._batch_buffer[metric_type].append(metric)
        
        # Flush if buffer is full
        if len(self._batch_buffer[metric_type]) >= self._batch_size:
            await self._flush_buffer(metric_type)
            
    async def record_counter(self, name: str, value: int = 1, 
                           tags: Optional[Dict[str, str]] = None):
        """Record a counter metric"""
        await self.record_metric(name, float(value), "counter", tags)
        
    async def record_gauge(self, name: str, value: float,
                         tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric"""
        await self.record_metric(name, value, "gauge", tags)
        
    async def record_histogram(self, name: str, value: float,
                             tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        await self.record_metric(name, value, "histogram", tags)
        
    async def batch_record(self, metrics: List[Dict[str, Any]]):
        """Record multiple metrics at once"""
        for metric in metrics:
            await self.record_metric(**metric)
            
    async def _flush_buffer(self, metric_type: Optional[str] = None):
        """Flush metrics buffer to database"""
        try:
            types_to_flush = [metric_type] if metric_type else list(self._batch_buffer.keys())
            
            for mtype in types_to_flush:
                if not self._batch_buffer[mtype]:
                    continue
                    
                metrics = self._batch_buffer[mtype]
                self._batch_buffer[mtype] = []
                
                # Batch insert
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    cursor.executemany("""
                        INSERT INTO system_metrics 
                        (metric_name, metric_value, metric_type, tags, 
                         timestamp, service_name, environment)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [
                        (m['metric_name'], m['metric_value'], m['metric_type'],
                         m['tags'], m['timestamp'], m['service_name'], m['environment'])
                        for m in metrics
                    ])
                    
                    conn.commit()
                    
                logger.debug(f"Flushed {len(metrics)} {mtype} metrics to database")
                
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            # Re-add metrics to buffer
            if metric_type:
                self._batch_buffer[metric_type].extend(metrics)
                
    async def _periodic_flush(self):
        """Periodically flush metrics buffer"""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
                
    async def get_metrics(self,
                         metric_name: str,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         tags: Optional[Dict[str, str]] = None,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """Query metrics from database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM system_metrics WHERE metric_name = ?"
                params = [metric_name]
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                    
                if tags:
                    query += " AND tags = ?"
                    params.append(json.dumps(tags))
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append({
                        'id': row[0],
                        'metric_name': row[1],
                        'metric_value': row[2],
                        'metric_type': row[3],
                        'tags': json.loads(row[4]) if row[4] else {},
                        'timestamp': row[5],
                        'service_name': row[6],
                        'environment': row[7]
                    })
                    
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
            
    async def get_aggregated_metrics(self,
                                   metric_name: str,
                                   aggregation: str = "avg",
                                   interval: str = "1h",
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get aggregated metrics (avg, sum, min, max, count)"""
        try:
            if not start_time:
                start_time = datetime.utcnow() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.utcnow()
                
            # Map interval to SQL
            interval_map = {
                "1m": "1 minute",
                "5m": "5 minutes",
                "15m": "15 minutes",
                "1h": "1 hour",
                "1d": "1 day"
            }
            sql_interval = interval_map.get(interval, "1 hour")
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # SQLite datetime functions
                cursor.execute(f"""
                    SELECT 
                        datetime((strftime('%s', timestamp) / 3600) * 3600, 'unixepoch') as time_bucket,
                        {aggregation}(metric_value) as value,
                        COUNT(*) as count,
                        MIN(metric_value) as min_value,
                        MAX(metric_value) as max_value
                    FROM system_metrics
                    WHERE metric_name = ?
                    AND timestamp >= ?
                    AND timestamp <= ?
                    GROUP BY time_bucket
                    ORDER BY time_bucket DESC
                """, (metric_name, start_time, end_time))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'timestamp': row[0],
                        'value': row[1],
                        'count': row[2],
                        'min': row[3],
                        'max': row[4]
                    })
                    
                return results
                
        except Exception as e:
            logger.error(f"Failed to get aggregated metrics: {e}")
            return []
            
    async def cleanup_old_metrics(self, days: int = 30) -> int:
        """Remove metrics older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old metrics")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
            return 0


# Enhanced monitoring integration
class MonitoringIntegration:
    """Integrates with existing monitoring to persist metrics"""
    
    def __init__(self, persistence: Optional[MetricsPersistence] = None):
        self.persistence = persistence or MetricsPersistence()
        self._started = False
        
    async def start(self):
        """Start monitoring integration"""
        if not self._started:
            await self.persistence.start()
            self._started = True
            
    async def stop(self):
        """Stop monitoring integration"""
        if self._started:
            await self.persistence.stop()
            self._started = False
            
    def counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Record counter metric (async in background)"""
        if self._started:
            asyncio.create_task(
                self.persistence.record_counter(name, value, labels)
            )
            
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record gauge metric (async in background)"""
        if self._started:
            asyncio.create_task(
                self.persistence.record_gauge(name, value, labels)
            )
            
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram metric (async in background)"""
        if self._started:
            asyncio.create_task(
                self.persistence.record_histogram(name, value, labels)
            )


# Global instance
_metrics_persistence: Optional[MetricsPersistence] = None

async def get_metrics_persistence() -> MetricsPersistence:
    """Get global metrics persistence instance"""
    global _metrics_persistence
    if _metrics_persistence is None:
        _metrics_persistence = MetricsPersistence()
        await _metrics_persistence.start()
    return _metrics_persistence