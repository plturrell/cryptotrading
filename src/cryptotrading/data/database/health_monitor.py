"""
Database Health Monitoring and Connection Resilience
Provides health checks, connection monitoring, and automatic recovery
"""

import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from contextlib import contextmanager
import threading
import json

logger = logging.getLogger(__name__)

class HealthStatus:
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    
class DatabaseHealthMonitor:
    """Monitors database health and connection resilience"""
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.health_checks: Dict[str, Callable] = {}
        self.check_interval = 60  # seconds
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_check_results = {}
        self.connection_retry_config = {
            'max_retries': 3,
            'initial_delay': 1,
            'max_delay': 30,
            'exponential_base': 2
        }
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check('connection', self._check_connection)
        self.register_check('response_time', self._check_response_time)
        self.register_check('connection_pool', self._check_connection_pool)
        self.register_check('disk_space', self._check_disk_space)
        self.register_check('table_sizes', self._check_table_sizes)
        self.register_check('query_performance', self._check_query_performance)
        self.register_check('replication_lag', self._check_replication_lag)
        self.register_check('lock_contention', self._check_lock_contention)
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check"""
        self.health_checks[name] = check_func
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Database health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Database health monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(10)  # Brief pause on error
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': HealthStatus.HEALTHY,
            'checks': {},
            'metrics': {}
        }
        
        unhealthy_count = 0
        degraded_count = 0
        
        for check_name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                check_result = check_func()
                execution_time = (time.time() - start_time) * 1000
                
                results['checks'][check_name] = {
                    'status': check_result.get('status', HealthStatus.HEALTHY),
                    'message': check_result.get('message', 'OK'),
                    'metrics': check_result.get('metrics', {}),
                    'execution_time_ms': round(execution_time, 2)
                }
                
                # Update counters
                if check_result['status'] == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
                elif check_result['status'] == HealthStatus.DEGRADED:
                    degraded_count += 1
                    
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                results['checks'][check_name] = {
                    'status': HealthStatus.UNHEALTHY,
                    'message': f'Check failed: {str(e)}',
                    'execution_time_ms': 0
                }
                unhealthy_count += 1
        
        # Determine overall status
        if unhealthy_count > 0:
            results['overall_status'] = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            results['overall_status'] = HealthStatus.DEGRADED
        
        # Store results
        self.last_check_results = results
        
        # Log to database
        self._log_health_check(results)
        
        # Alert on status change
        self._check_status_change(results)
        
        return results
    
    def _check_connection(self) -> Dict[str, Any]:
        """Check database connection"""
        try:
            with self.db_client.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                result.fetchone()
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Connection successful'
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Connection failed: {str(e)}'
            }
    
    def _check_response_time(self) -> Dict[str, Any]:
        """Check database response time"""
        try:
            start_time = time.time()
            
            with self.db_client.engine.connect() as conn:
                # Simple query to test response time
                result = conn.execute("SELECT COUNT(*) FROM users")
                result.fetchone()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Define thresholds
            if response_time_ms < 100:
                status = HealthStatus.HEALTHY
                message = f'Response time: {response_time_ms:.2f}ms'
            elif response_time_ms < 500:
                status = HealthStatus.DEGRADED
                message = f'Slow response: {response_time_ms:.2f}ms'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'Very slow response: {response_time_ms:.2f}ms'
            
            return {
                'status': status,
                'message': message,
                'metrics': {'response_time_ms': response_time_ms}
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Response time check failed: {str(e)}'
            }
    
    def _check_connection_pool(self) -> Dict[str, Any]:
        """Check connection pool health"""
        try:
            pool_status = self.db_client.get_pool_status()
            
            # Calculate pool utilization
            total_connections = pool_status['pool_size'] + pool_status['overflow']
            active_connections = pool_status['checked_out']
            utilization = (active_connections / total_connections * 100) if total_connections > 0 else 0
            
            # Define thresholds
            if utilization < 70:
                status = HealthStatus.HEALTHY
                message = f'Pool utilization: {utilization:.1f}%'
            elif utilization < 90:
                status = HealthStatus.DEGRADED
                message = f'High pool utilization: {utilization:.1f}%'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'Critical pool utilization: {utilization:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'metrics': {
                    'pool_utilization_percent': utilization,
                    'active_connections': active_connections,
                    'total_connections': total_connections,
                    **pool_status
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Pool check failed: {str(e)}'
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space for SQLite databases"""
        if not self.db_client.is_sqlite:
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Not applicable for PostgreSQL'
            }
        
        try:
            import shutil
            import os
            
            # Get database file path
            db_path = self.db_client.db_url.replace('sqlite:///', '')
            if not os.path.exists(db_path):
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'message': 'Database file not found'
                }
            
            # Check disk usage
            stat = shutil.disk_usage(os.path.dirname(db_path))
            free_percent = (stat.free / stat.total) * 100
            
            # Define thresholds
            if free_percent > 20:
                status = HealthStatus.HEALTHY
                message = f'Disk space available: {free_percent:.1f}%'
            elif free_percent > 10:
                status = HealthStatus.DEGRADED
                message = f'Low disk space: {free_percent:.1f}%'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'Critical disk space: {free_percent:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'metrics': {
                    'free_space_percent': free_percent,
                    'free_space_gb': stat.free / (1024**3),
                    'total_space_gb': stat.total / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Disk space check failed: {str(e)}'
            }
    
    def _check_table_sizes(self) -> Dict[str, Any]:
        """Check table sizes and row counts"""
        try:
            table_stats = {}
            
            if self.db_client.is_sqlite:
                # SQLite approach
                tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
                with self.db_client.engine.connect() as conn:
                    tables = conn.execute(tables_query).fetchall()
                    
                    for table in tables:
                        table_name = table[0]
                        count_query = f"SELECT COUNT(*) FROM {table_name}"
                        count = conn.execute(count_query).fetchone()[0]
                        table_stats[table_name] = count
            else:
                # PostgreSQL approach
                size_query = """
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                    n_live_tup AS row_count
                FROM pg_stat_user_tables
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                LIMIT 10
                """
                with self.db_client.engine.connect() as conn:
                    result = conn.execute(size_query)
                    for row in result:
                        table_stats[row['tablename']] = {
                            'size': row['size'],
                            'rows': row['row_count']
                        }
            
            # Check for unusually large tables
            large_tables = [t for t, count in table_stats.items() 
                          if isinstance(count, int) and count > 1000000]
            
            if not large_tables:
                status = HealthStatus.HEALTHY
                message = 'Table sizes normal'
            else:
                status = HealthStatus.DEGRADED
                message = f'Large tables detected: {", ".join(large_tables[:3])}'
            
            return {
                'status': status,
                'message': message,
                'metrics': {'table_stats': table_stats}
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Table size check failed: {str(e)}'
            }
    
    def _check_query_performance(self) -> Dict[str, Any]:
        """Check recent query performance"""
        try:
            # Get average query time for last hour
            query = """
            SELECT 
                AVG(execution_time_ms) as avg_time,
                MAX(execution_time_ms) as max_time,
                COUNT(*) as query_count
            FROM query_performance_log
            WHERE created_at >= datetime('now', '-1 hour')
            """
            
            with self.db_client.engine.connect() as conn:
                result = conn.execute(query).fetchone()
                
                if not result or result['query_count'] == 0:
                    return {
                        'status': HealthStatus.HEALTHY,
                        'message': 'No recent queries to analyze'
                    }
                
                avg_time = result['avg_time'] or 0
                max_time = result['max_time'] or 0
                
                # Define thresholds
                if avg_time < 500 and max_time < 5000:
                    status = HealthStatus.HEALTHY
                    message = f'Query performance good (avg: {avg_time:.0f}ms)'
                elif avg_time < 1000 and max_time < 10000:
                    status = HealthStatus.DEGRADED
                    message = f'Query performance degraded (avg: {avg_time:.0f}ms)'
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f'Poor query performance (avg: {avg_time:.0f}ms)'
                
                return {
                    'status': status,
                    'message': message,
                    'metrics': {
                        'avg_query_time_ms': avg_time,
                        'max_query_time_ms': max_time,
                        'query_count_last_hour': result['query_count']
                    }
                }
                
        except Exception as e:
            logger.error(f"Query performance check failed: {e}")
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Query performance check not available'
            }
    
    def _check_replication_lag(self) -> Dict[str, Any]:
        """Check replication lag for PostgreSQL"""
        if self.db_client.is_sqlite:
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Not applicable for SQLite'
            }
        
        try:
            # Check if replication is configured
            query = """
            SELECT 
                client_addr,
                state,
                sent_lsn,
                write_lsn,
                flush_lsn,
                replay_lsn,
                write_lag,
                flush_lag,
                replay_lag
            FROM pg_stat_replication
            """
            
            with self.db_client.engine.connect() as conn:
                result = conn.execute(query).fetchall()
                
                if not result:
                    return {
                        'status': HealthStatus.HEALTHY,
                        'message': 'No replication configured'
                    }
                
                # Check lag for each replica
                max_lag = 0
                for row in result:
                    if row['replay_lag']:
                        lag_seconds = row['replay_lag'].total_seconds()
                        max_lag = max(max_lag, lag_seconds)
                
                if max_lag < 1:
                    status = HealthStatus.HEALTHY
                    message = f'Replication lag: {max_lag:.2f}s'
                elif max_lag < 5:
                    status = HealthStatus.DEGRADED
                    message = f'High replication lag: {max_lag:.2f}s'
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f'Critical replication lag: {max_lag:.2f}s'
                
                return {
                    'status': status,
                    'message': message,
                    'metrics': {'max_replication_lag_seconds': max_lag}
                }
                
        except Exception as e:
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Replication check not available'
            }
    
    def _check_lock_contention(self) -> Dict[str, Any]:
        """Check for lock contention"""
        try:
            if self.db_client.is_postgres:
                # PostgreSQL lock check
                query = """
                SELECT 
                    COUNT(*) as blocked_queries
                FROM pg_stat_activity
                WHERE wait_event_type = 'Lock'
                """
                
                with self.db_client.engine.connect() as conn:
                    result = conn.execute(query).fetchone()
                    blocked_count = result['blocked_queries']
            else:
                # SQLite doesn't have detailed lock info
                blocked_count = 0
            
            if blocked_count == 0:
                status = HealthStatus.HEALTHY
                message = 'No lock contention detected'
            elif blocked_count < 5:
                status = HealthStatus.DEGRADED
                message = f'{blocked_count} queries blocked by locks'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'High lock contention: {blocked_count} blocked queries'
            
            return {
                'status': status,
                'message': message,
                'metrics': {'blocked_queries': blocked_count}
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Lock check not available'
            }
    
    def _log_health_check(self, results: Dict[str, Any]):
        """Log health check results to database"""
        try:
            for check_name, check_result in results['checks'].items():
                insert_sql = """
                INSERT INTO database_health_checks 
                (check_name, check_status, response_time_ms, error_message, metadata)
                VALUES (?, ?, ?, ?, ?)
                """
                
                metadata = json.dumps({
                    'metrics': check_result.get('metrics', {}),
                    'overall_status': results['overall_status']
                })
                
                error_msg = check_result['message'] if check_result['status'] != HealthStatus.HEALTHY else None
                
                with self.db_client.engine.connect() as conn:
                    if self.db_client.is_postgres:
                        insert_sql = insert_sql.replace('?', '%s')
                    
                    conn.execute(insert_sql, (
                        check_name,
                        check_result['status'],
                        check_result.get('execution_time_ms', 0),
                        error_msg,
                        metadata
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to log health check: {e}")
    
    def _check_status_change(self, results: Dict[str, Any]):
        """Check for status changes and alert if needed"""
        # This could be extended to send alerts via email, Slack, etc.
        previous_status = self.last_check_results.get('overall_status') if self.last_check_results else None
        current_status = results['overall_status']
        
        if previous_status and previous_status != current_status:
            if current_status == HealthStatus.UNHEALTHY:
                logger.error(f"Database health degraded: {previous_status} -> {current_status}")
            elif current_status == HealthStatus.HEALTHY and previous_status == HealthStatus.UNHEALTHY:
                logger.info(f"Database health recovered: {previous_status} -> {current_status}")
    
    @contextmanager
    def resilient_connection(self):
        """Get a resilient database connection with retry logic"""
        retry_count = 0
        delay = self.connection_retry_config['initial_delay']
        
        while retry_count < self.connection_retry_config['max_retries']:
            try:
                with self.db_client.get_session() as session:
                    yield session
                    return  # Success
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= self.connection_retry_config['max_retries']:
                    logger.error(f"Failed to establish connection after {retry_count} retries")
                    raise
                
                logger.warning(f"Connection failed (attempt {retry_count}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                
                # Exponential backoff
                delay = min(
                    delay * self.connection_retry_config['exponential_base'],
                    self.connection_retry_config['max_delay']
                )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        if not self.last_check_results:
            return {
                'status': 'unknown',
                'message': 'No health checks run yet'
            }
        
        return {
            'status': self.last_check_results['overall_status'],
            'last_check': self.last_check_results['timestamp'],
            'healthy_checks': sum(1 for c in self.last_check_results['checks'].values() 
                                if c['status'] == HealthStatus.HEALTHY),
            'total_checks': len(self.last_check_results['checks']),
            'issues': [
                {'check': name, 'message': check['message']}
                for name, check in self.last_check_results['checks'].items()
                if check['status'] != HealthStatus.HEALTHY
            ]
        }