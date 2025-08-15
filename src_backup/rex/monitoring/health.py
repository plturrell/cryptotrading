"""
Production health check endpoints for load balancers and monitoring
Supports readiness, liveness, and detailed health probes
"""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from ..database.client import get_db
from ..database.cache import cache_manager
from ..security.auth import auth_manager
from ..logging.production_logger import get_logger

logger = get_logger(__name__)

class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Individual health check configuration"""
    name: str
    check_function: Callable
    timeout: float = 5.0
    critical: bool = True
    interval: int = 30
    max_failures: int = 3

@dataclass
class HealthResult:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    duration: float
    timestamp: str
    details: Optional[Dict[str, Any]] = None

class HealthMonitor:
    """Production health monitoring system"""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthResult] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_check_times: Dict[str, datetime] = {}
        self.startup_time = datetime.utcnow()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        
        # Database connectivity check
        self.register_check(HealthCheck(
            name="database",
            check_function=self._check_database,
            timeout=10.0,
            critical=True,
            interval=30
        ))
        
        # Cache connectivity check
        self.register_check(HealthCheck(
            name="cache",
            check_function=self._check_cache,
            timeout=5.0,
            critical=False,
            interval=30
        ))
        
        # Memory usage check
        self.register_check(HealthCheck(
            name="memory",
            check_function=self._check_memory,
            timeout=2.0,
            critical=False,
            interval=60
        ))
        
        # Disk space check
        self.register_check(HealthCheck(
            name="disk_space",
            check_function=self._check_disk_space,
            timeout=2.0,
            critical=True,
            interval=120
        ))
        
        # Authentication system check
        self.register_check(HealthCheck(
            name="auth_system",
            check_function=self._check_auth_system,
            timeout=3.0,
            critical=True,
            interval=60
        ))
        
        # Agent registry check
        self.register_check(HealthCheck(
            name="agent_registry",
            check_function=self._check_agent_registry,
            timeout=5.0,
            critical=True,
            interval=30
        ))
    
    def register_check(self, health_check: HealthCheck):
        """Register a health check"""
        self.checks[health_check.name] = health_check
        self.failure_counts[health_check.name] = 0
        logger.info(f"Registered health check: {health_check.name}")
    
    async def run_check(self, check_name: str) -> HealthResult:
        """Run a specific health check"""
        if check_name not in self.checks:
            return HealthResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                message="Check not found",
                duration=0.0,
                timestamp=datetime.utcnow().isoformat()
            )
        
        check = self.checks[check_name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_function(),
                timeout=check.timeout
            )
            
            duration = time.time() - start_time
            
            # Reset failure count on success
            self.failure_counts[check_name] = 0
            
            health_result = HealthResult(
                name=check_name,
                status=HealthStatus.HEALTHY,
                message=result.get('message', 'Check passed'),
                duration=duration,
                timestamp=datetime.utcnow().isoformat(),
                details=result.get('details')
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.failure_counts[check_name] += 1
            
            health_result = HealthResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {check.timeout}s",
                duration=duration,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.failure_counts[check_name] += 1
            
            health_result = HealthResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration=duration,
                timestamp=datetime.utcnow().isoformat(),
                details={'error_type': e.__class__.__name__}
            )
        
        # Store result
        self.results[check_name] = health_result
        self.last_check_times[check_name] = datetime.utcnow()
        
        # Log result
        if health_result.status != HealthStatus.HEALTHY:
            logger.warning(f"Health check failed: {check_name}",
                         check_name=check_name,
                         status=health_result.status.value,
                         message=health_result.message,
                         failure_count=self.failure_counts[check_name])
        
        return health_result
    
    async def run_all_checks(self) -> Dict[str, HealthResult]:
        """Run all registered health checks"""
        tasks = []
        for check_name in self.checks.keys():
            task = asyncio.create_task(self.run_check(check_name))
            tasks.append((check_name, task))
        
        results = {}
        for check_name, task in tasks:
            try:
                result = await task
                results[check_name] = result
            except Exception as e:
                logger.error(f"Failed to run health check {check_name}: {e}")
                results[check_name] = HealthResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(e)}",
                    duration=0.0,
                    timestamp=datetime.utcnow().isoformat()
                )
        
        return results
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        # Run fresh checks for critical components
        fresh_results = await self.run_all_checks()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        critical_failures = []
        degraded_services = []
        
        for check_name, result in fresh_results.items():
            check = self.checks[check_name]
            
            if result.status == HealthStatus.UNHEALTHY:
                if check.critical:
                    overall_status = HealthStatus.UNHEALTHY
                    critical_failures.append(check_name)
                else:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                    degraded_services.append(check_name)
            elif result.status == HealthStatus.DEGRADED:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                degraded_services.append(check_name)
        
        uptime = (datetime.utcnow() - self.startup_time).total_seconds()
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime,
            "checks": {name: {
                "status": result.status.value,
                "message": result.message,
                "duration": result.duration,
                "last_check": result.timestamp,
                "failure_count": self.failure_counts.get(name, 0)
            } for name, result in fresh_results.items()},
            "summary": {
                "total_checks": len(fresh_results),
                "healthy_checks": len([r for r in fresh_results.values() if r.status == HealthStatus.HEALTHY]),
                "degraded_checks": len(degraded_services),
                "unhealthy_checks": len([r for r in fresh_results.values() if r.status == HealthStatus.UNHEALTHY]),
                "critical_failures": critical_failures,
                "degraded_services": degraded_services
            }
        }
    
    async def get_readiness_probe(self) -> Dict[str, Any]:
        """Kubernetes-style readiness probe"""
        # Check only critical components for readiness
        critical_checks = [name for name, check in self.checks.items() if check.critical]
        
        ready = True
        failed_checks = []
        
        for check_name in critical_checks:
            result = await self.run_check(check_name)
            if result.status == HealthStatus.UNHEALTHY:
                ready = False
                failed_checks.append(check_name)
        
        return {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "failed_checks": failed_checks
        }
    
    async def get_liveness_probe(self) -> Dict[str, Any]:
        """Kubernetes-style liveness probe"""
        # Simple check that the application is responding
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds()
        }
    
    # Individual health check implementations
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        db = get_db()
        
        try:
            # Test basic connectivity
            with db.get_session() as session:
                session.execute("SELECT 1").fetchone()
            
            # Check connection pool status
            pool_status = db.get_pool_status()
            
            # Determine health based on pool utilization
            utilization = pool_status['checked_out'] / pool_status['total_connections']
            
            if utilization > 0.9:
                status = HealthStatus.DEGRADED
                message = f"High database connection utilization: {utilization:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = "Database connectivity OK"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'pool_status': pool_status,
                    'utilization': utilization
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Database connection failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_cache(self) -> Dict[str, Any]:
        """Check cache connectivity"""
        try:
            # Test cache connectivity
            test_key = "health_check_test"
            test_value = datetime.utcnow().isoformat()
            
            # Set and get test value
            cache_manager.cache.set(test_key, test_value, ttl=60)
            retrieved_value = cache_manager.cache.get(test_key)
            
            if retrieved_value == test_value:
                # Clean up test key
                cache_manager.cache.delete(test_key)
                
                # Get cache statistics
                stats = cache_manager.get_cache_stats()
                
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'Cache connectivity OK',
                    'details': stats
                }
            else:
                return {
                    'status': HealthStatus.DEGRADED,
                    'message': 'Cache read/write test failed',
                    'details': {'expected': test_value, 'actual': retrieved_value}
                }
        
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED,  # Cache is not critical
                'message': f"Cache check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {memory_percent:.1f}%"
            elif memory_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage OK: {memory_percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'percent_used': memory_percent,
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f"Memory check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {disk_percent:.1f}%"
            elif disk_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage OK: {disk_percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'percent_used': disk_percent,
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f"Disk check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_auth_system(self) -> Dict[str, Any]:
        """Check authentication system"""
        try:
            # Test that auth manager is accessible
            users_count = len(auth_manager.users)
            api_keys_count = len(auth_manager.api_keys)
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Authentication system OK',
                'details': {
                    'users_count': users_count,
                    'api_keys_count': api_keys_count
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Auth system check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_agent_registry(self) -> Dict[str, Any]:
        """Check agent registry"""
        try:
            from ..a2a.registry.registry import agent_registry
            
            # Check if registry is accessible
            agents = agent_registry.get_all_agents()
            active_agents = len([a for a in agents.values() if a.get('status') == 'active'])
            
            if active_agents == 0:
                status = HealthStatus.DEGRADED
                message = "No active agents registered"
            else:
                status = HealthStatus.HEALTHY
                message = f"Agent registry OK ({active_agents} active agents)"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'total_agents': len(agents),
                    'active_agents': active_agents
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Agent registry check failed: {str(e)}",
                'details': {'error': str(e)}
            }

class HealthEndpoints:
    """HTTP endpoints for health checks"""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
    
    async def health_endpoint(self) -> Dict[str, Any]:
        """Main health endpoint - detailed status"""
        return await self.health_monitor.get_health_status()
    
    async def readiness_endpoint(self) -> Dict[str, Any]:
        """Readiness probe endpoint for load balancers"""
        return await self.health_monitor.get_readiness_probe()
    
    async def liveness_endpoint(self) -> Dict[str, Any]:
        """Liveness probe endpoint for container orchestration"""
        return await self.health_monitor.get_liveness_probe()
    
    async def metrics_endpoint(self) -> Dict[str, Any]:
        """Metrics endpoint for monitoring systems"""
        health_status = await self.health_monitor.get_health_status()
        
        # Convert to metrics format
        metrics = {
            "health_check_up": 1 if health_status["status"] == "healthy" else 0,
            "health_check_total": health_status["summary"]["total_checks"],
            "health_check_healthy": health_status["summary"]["healthy_checks"],
            "health_check_unhealthy": health_status["summary"]["unhealthy_checks"],
            "uptime_seconds": health_status["uptime_seconds"]
        }
        
        # Add individual check metrics
        for check_name, check_data in health_status["checks"].items():
            metrics[f"health_check_{check_name}_up"] = 1 if check_data["status"] == "healthy" else 0
            metrics[f"health_check_{check_name}_duration"] = check_data["duration"]
            metrics[f"health_check_{check_name}_failures"] = check_data["failure_count"]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }

# Global health monitor instance
health_monitor = HealthMonitor()
health_endpoints = HealthEndpoints(health_monitor)