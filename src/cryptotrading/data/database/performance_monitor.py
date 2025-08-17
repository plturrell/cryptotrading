"""
Database Performance Monitoring and Benchmarking
Provides comprehensive performance analysis and benchmarking capabilities
"""

import time
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class BenchmarkResult:
    """Results from a benchmark run"""
    operation: str
    total_operations: int
    duration_seconds: float
    operations_per_second: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float
    errors: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Monitors and analyzes database performance"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self._metrics: List[PerformanceMetric] = []
        self._lock = threading.Lock()
        self._operation_counters = {}
        
    @contextmanager
    def measure(self, operation: str, metadata: Optional[Dict] = None):
        """Context manager to measure operation performance"""
        start_time = time.time()
        success = True
        error = None
        
        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            metric = PerformanceMetric(
                operation=operation,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                success=success,
                metadata=metadata or {}
            )
            
            if error:
                metric.metadata['error'] = error
                
            self._add_metric(metric)
    
    def _add_metric(self, metric: PerformanceMetric):
        """Add metric to collection"""
        with self._lock:
            self._metrics.append(metric)
            
            # Maintain size limit
            if len(self._metrics) > self.max_metrics:
                self._metrics = self._metrics[-self.max_metrics:]
            
            # Update counters
            if metric.operation not in self._operation_counters:
                self._operation_counters[metric.operation] = {'total': 0, 'success': 0}
            
            self._operation_counters[metric.operation]['total'] += 1
            if metric.success:
                self._operation_counters[metric.operation]['success'] += 1
    
    def get_metrics(self, operation: str = None, 
                   since: datetime = None) -> List[PerformanceMetric]:
        """Get metrics with optional filtering"""
        with self._lock:
            metrics = self._metrics.copy()
        
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def get_summary(self, operation: str = None, 
                   hours: int = 24) -> Dict[str, Any]:
        """Get performance summary"""
        since = datetime.now() - timedelta(hours=hours)
        metrics = self.get_metrics(operation, since)
        
        if not metrics:
            return {
                'operation': operation or 'all',
                'period_hours': hours,
                'total_operations': 0,
                'no_data': True
            }
        
        durations = [m.duration_ms for m in metrics]
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]
        
        summary = {
            'operation': operation or 'all',
            'period_hours': hours,
            'total_operations': len(metrics),
            'successful_operations': len(successful),
            'failed_operations': len(failed),
            'success_rate': len(successful) / len(metrics) * 100,
            'avg_latency_ms': statistics.mean(durations),
            'min_latency_ms': min(durations),
            'max_latency_ms': max(durations),
            'median_latency_ms': statistics.median(durations),
        }
        
        # Add percentiles if enough data
        if len(durations) >= 10:
            sorted_durations = sorted(durations)
            summary.update({
                'p95_latency_ms': sorted_durations[int(len(sorted_durations) * 0.95)],
                'p99_latency_ms': sorted_durations[int(len(sorted_durations) * 0.99)]
            })
        
        # Add error analysis
        if failed:
            error_types = {}
            for metric in failed:
                error = metric.metadata.get('error', 'Unknown')
                error_types[error] = error_types.get(error, 0) + 1
            summary['error_types'] = error_types
        
        return summary
    
    def get_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations"""
        with self._lock:
            stats = {}
            for operation, counters in self._operation_counters.items():
                recent_metrics = self.get_metrics(
                    operation=operation,
                    since=datetime.now() - timedelta(hours=1)
                )
                
                if recent_metrics:
                    durations = [m.duration_ms for m in recent_metrics]
                    stats[operation] = {
                        'total_ever': counters['total'],
                        'success_ever': counters['success'],
                        'success_rate_ever': counters['success'] / counters['total'] * 100,
                        'recent_count': len(recent_metrics),
                        'recent_avg_ms': statistics.mean(durations),
                        'recent_max_ms': max(durations),
                    }
                else:
                    stats[operation] = {
                        'total_ever': counters['total'],
                        'success_ever': counters['success'],
                        'success_rate_ever': counters['success'] / counters['total'] * 100,
                        'recent_count': 0
                    }
            
            return stats

class DatabaseBenchmark:
    """Database benchmarking utilities"""
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.performance_monitor = PerformanceMonitor()
    
    def benchmark_operation(self, operation_func: Callable, 
                          operation_name: str,
                          iterations: int = 100,
                          concurrent_workers: int = 1) -> BenchmarkResult:
        """Benchmark a database operation"""
        logger.info(f"Starting benchmark: {operation_name} "
                   f"({iterations} iterations, {concurrent_workers} workers)")
        
        start_time = time.time()
        results = []
        errors = []
        
        if concurrent_workers == 1:
            # Sequential execution
            for i in range(iterations):
                try:
                    op_start = time.time()
                    operation_func()
                    op_duration = (time.time() - op_start) * 1000
                    results.append(op_duration)
                except Exception as e:
                    errors.append(str(e))
        else:
            # Concurrent execution
            import concurrent.futures
            
            def run_operation():
                try:
                    op_start = time.time()
                    operation_func()
                    op_duration = (time.time() - op_start) * 1000
                    return op_duration
                except Exception as e:
                    return f"ERROR: {str(e)}"
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
                futures = [executor.submit(run_operation) for _ in range(iterations)]
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if isinstance(result, str) and result.startswith("ERROR:"):
                        errors.append(result[7:])  # Remove "ERROR: " prefix
                    else:
                        results.append(result)
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        if results:
            sorted_results = sorted(results)
            ops_per_second = len(results) / total_duration
            success_rate = len(results) / iterations * 100
            
            benchmark_result = BenchmarkResult(
                operation=operation_name,
                total_operations=iterations,
                duration_seconds=total_duration,
                operations_per_second=ops_per_second,
                avg_latency_ms=statistics.mean(results),
                min_latency_ms=min(results),
                max_latency_ms=max(results),
                p50_latency_ms=statistics.median(results),
                p95_latency_ms=sorted_results[int(len(sorted_results) * 0.95)] if len(sorted_results) >= 20 else max(results),
                p99_latency_ms=sorted_results[int(len(sorted_results) * 0.99)] if len(sorted_results) >= 100 else max(results),
                success_rate=success_rate,
                errors=errors[:10],  # Limit error samples
                metadata={
                    'concurrent_workers': concurrent_workers,
                    'error_count': len(errors)
                }
            )
        else:
            # All operations failed
            benchmark_result = BenchmarkResult(
                operation=operation_name,
                total_operations=iterations,
                duration_seconds=total_duration,
                operations_per_second=0,
                avg_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                success_rate=0,
                errors=errors[:10],
                metadata={
                    'concurrent_workers': concurrent_workers,
                    'error_count': len(errors),
                    'all_failed': True
                }
            )
        
        logger.info(f"Benchmark completed: {operation_name} - "
                   f"{benchmark_result.operations_per_second:.1f} ops/sec, "
                   f"{benchmark_result.avg_latency_ms:.1f}ms avg")
        
        return benchmark_result
    
    def benchmark_crud_operations(self, model_class, sample_data: Dict[str, Any],
                                 iterations: int = 100) -> Dict[str, BenchmarkResult]:
        """Benchmark CRUD operations for a model"""
        results = {}
        created_ids = []
        
        # Benchmark CREATE
        def create_op():
            nonlocal created_ids
            # Add unique identifier to avoid conflicts
            data = sample_data.copy()
            if 'username' in data:
                data['username'] += f"_{len(created_ids)}"
            if 'email' in data:
                name, domain = data['email'].split('@')
                data['email'] = f"{name}_{len(created_ids)}@{domain}"
            
            obj_id = self.db_client.create(model_class, **data)
            created_ids.append(obj_id)
            return obj_id
        
        results['create'] = self.benchmark_operation(
            create_op, f"{model_class.__name__}_create", iterations
        )
        
        # Benchmark READ
        def read_op():
            if created_ids:
                obj_id = created_ids[len(created_ids) // 2]  # Read middle object
                return self.db_client.get_by_id(model_class, obj_id)
            return None
        
        if created_ids:
            results['read'] = self.benchmark_operation(
                read_op, f"{model_class.__name__}_read", iterations
            )
        
        # Benchmark UPDATE
        def update_op():
            if created_ids:
                obj_id = created_ids[0]  # Always update first object
                update_data = {}
                if 'email' in sample_data:
                    update_data['email'] = f"updated_{time.time()}@example.com"
                return self.db_client.update(model_class, obj_id, **update_data)
            return False
        
        if created_ids:
            results['update'] = self.benchmark_operation(
                update_op, f"{model_class.__name__}_update", min(iterations, 50)
            )
        
        # Benchmark DELETE
        def delete_op():
            if created_ids:
                obj_id = created_ids.pop()  # Delete last object
                return self.db_client.delete(model_class, obj_id)
            return False
        
        if created_ids:
            results['delete'] = self.benchmark_operation(
                delete_op, f"{model_class.__name__}_delete", min(iterations, len(created_ids))
            )
        
        return results
    
    def benchmark_query_performance(self, queries: List[tuple],
                                  iterations: int = 50) -> Dict[str, BenchmarkResult]:
        """Benchmark custom queries"""
        results = {}
        
        for query_name, query, params in queries:
            def query_op():
                return self.db_client.execute_query(query, params)
            
            results[query_name] = self.benchmark_operation(
                query_op, f"query_{query_name}", iterations
            )
        
        return results
    
    def benchmark_concurrent_access(self, operation_func: Callable,
                                  operation_name: str,
                                  total_operations: int = 100,
                                  worker_counts: List[int] = [1, 2, 4, 8]) -> Dict[int, BenchmarkResult]:
        """Benchmark operation with different concurrency levels"""
        results = {}
        
        for worker_count in worker_counts:
            ops_per_worker = total_operations // worker_count
            
            result = self.benchmark_operation(
                operation_func,
                f"{operation_name}_concurrent_{worker_count}",
                ops_per_worker,
                worker_count
            )
            
            results[worker_count] = result
        
        return results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        from .models import User, AIAnalysis, MarketData
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_type': 'sqlite' if self.db_client.is_sqlite else 'postgresql',
            'benchmarks': {},
            'performance_summary': {}
        }
        
        # Test data
        user_data = {
            'username': 'benchmark_user',
            'email': 'benchmark@example.com',
            'password_hash': 'benchmark_hash'
        }
        
        analysis_data = {
            'symbol': 'BTC',
            'model': 'benchmark_model',
            'analysis_type': 'benchmark',
            'analysis': 'Benchmark analysis content'
        }
        
        market_data = {
            'symbol': 'ETH',
            'price': 2000.0,
            'volume_24h': 1000000.0
        }
        
        # Benchmark CRUD operations
        try:
            report['benchmarks']['user_crud'] = self.benchmark_crud_operations(
                User, user_data, 50
            )
        except Exception as e:
            report['benchmarks']['user_crud'] = {'error': str(e)}
        
        try:
            report['benchmarks']['analysis_crud'] = self.benchmark_crud_operations(
                AIAnalysis, analysis_data, 50
            )
        except Exception as e:
            report['benchmarks']['analysis_crud'] = {'error': str(e)}
        
        # Benchmark common queries
        common_queries = [
            ('count_users', 'SELECT COUNT(*) FROM users', None),
            ('active_users', 'SELECT * FROM users WHERE is_active = ?', (True,)),
            ('recent_analysis', 'SELECT * FROM ai_analyses ORDER BY created_at DESC LIMIT 10', None)
        ]
        
        try:
            report['benchmarks']['queries'] = self.benchmark_query_performance(
                common_queries, 30
            )
        except Exception as e:
            report['benchmarks']['queries'] = {'error': str(e)}
        
        # Performance summary
        all_results = []
        for category, benchmarks in report['benchmarks'].items():
            if isinstance(benchmarks, dict) and 'error' not in benchmarks:
                for op_name, result in benchmarks.items():
                    if isinstance(result, BenchmarkResult):
                        all_results.append(result)
        
        if all_results:
            avg_ops_per_sec = statistics.mean([r.operations_per_second for r in all_results])
            avg_latency = statistics.mean([r.avg_latency_ms for r in all_results])
            avg_success_rate = statistics.mean([r.success_rate for r in all_results])
            
            report['performance_summary'] = {
                'average_operations_per_second': avg_ops_per_sec,
                'average_latency_ms': avg_latency,
                'average_success_rate': avg_success_rate,
                'total_benchmarks': len(all_results),
                'performance_grade': self._calculate_performance_grade(avg_ops_per_sec, avg_latency, avg_success_rate)
            }
        
        return report
    
    def _calculate_performance_grade(self, ops_per_sec: float, 
                                   avg_latency: float, success_rate: float) -> str:
        """Calculate performance grade based on metrics"""
        score = 0
        
        # Operations per second scoring
        if ops_per_sec > 1000:
            score += 40
        elif ops_per_sec > 500:
            score += 30
        elif ops_per_sec > 100:
            score += 20
        else:
            score += 10
        
        # Latency scoring
        if avg_latency < 10:
            score += 30
        elif avg_latency < 50:
            score += 20
        elif avg_latency < 100:
            score += 10
        else:
            score += 5
        
        # Success rate scoring
        if success_rate > 99:
            score += 30
        elif success_rate > 95:
            score += 20
        elif success_rate > 90:
            score += 10
        else:
            score += 5
        
        # Convert to grade
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save performance report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {filename}")
        return filename