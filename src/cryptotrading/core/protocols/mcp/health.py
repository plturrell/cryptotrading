"""
MCP Health Checks and Monitoring
Lightweight health monitoring for serverless environments
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .client import MCPClient
from .connection_pool import ConnectionConfig, ServerlessConnectionManager
from .metrics import mcp_metrics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result"""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: float
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "details": self.details or {},
        }


class HealthChecker:
    """Health checker for MCP components"""

    def __init__(self):
        self.connection_manager = ServerlessConnectionManager()
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks"""
        self.checks["mcp_connection"] = self._check_mcp_connection
        self.checks["auth_system"] = self._check_auth_system
        self.checks["cache_system"] = self._check_cache_system
        self.checks["rate_limiter"] = self._check_rate_limiter
        self.checks["metrics_system"] = self._check_metrics_system

    async def run_check(self, check_name: str) -> HealthCheck:
        """Run a single health check"""
        if check_name not in self.checks:
            return HealthCheck(
                name=check_name,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown health check: {check_name}",
                duration_ms=0,
                timestamp=time.time(),
            )

        start_time = time.time()
        try:
            check_func = self.checks[check_name]
            result = await check_func()
            duration_ms = (time.time() - start_time) * 1000

            health_check = HealthCheck(
                name=check_name,
                status=result.get("status", HealthStatus.UNKNOWN),
                message=result.get("message", "Check completed"),
                duration_ms=duration_ms,
                timestamp=time.time(),
                details=result.get("details", {}),
            )

            self.last_results[check_name] = health_check
            return health_check

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            health_check = HealthCheck(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=time.time(),
                details={"error": str(e)},
            )

            self.last_results[check_name] = health_check
            return health_check

    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        results = {}

        # Run checks concurrently
        tasks = [self.run_check(check_name) for check_name in self.checks.keys()]

        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(check_results):
            check_name = list(self.checks.keys())[i]

            if isinstance(result, Exception):
                results[check_name] = HealthCheck(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed with exception: {str(result)}",
                    duration_ms=0,
                    timestamp=time.time(),
                )
            else:
                results[check_name] = result

        return results

    async def _check_mcp_connection(self) -> Dict[str, Any]:
        """Check MCP connection health"""
        try:
            config = ConnectionConfig(transport_type="stdio", timeout=5.0, retry_attempts=1)

            # Try to create a connection
            client = await self.connection_manager.create_connection(config)

            # Test basic operations
            await client.ping()
            tools = await client.list_tools()

            await client.disconnect()

            return {
                "status": HealthStatus.HEALTHY,
                "message": "MCP connection is working",
                "details": {
                    "tools_available": len(tools),
                    "connection_type": config.transport_type,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"MCP connection failed: {str(e)}",
                "details": {"error": str(e)},
            }

    async def _check_auth_system(self) -> Dict[str, Any]:
        """Check authentication system health"""
        try:
            from .auth import AuthCredentials, MCPAuthenticator

            authenticator = MCPAuthenticator()

            # Test API key generation and validation
            api_key = authenticator.generate_api_key("test_user", "test_tenant")
            credentials = AuthCredentials(api_key=api_key)
            auth_context = authenticator.authenticate(credentials)

            if auth_context and auth_context.user_id == "test_user":
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Authentication system is working",
                    "details": {"api_key_generation": True, "api_key_validation": True},
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "Authentication validation failed",
                    "details": {"validation_failed": True},
                }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Authentication system failed: {str(e)}",
                "details": {"error": str(e)},
            }

    async def _check_cache_system(self) -> Dict[str, Any]:
        """Check cache system health"""
        try:
            from .cache import mcp_cache

            # Test cache operations
            test_key = "health_check_test"
            test_value = {"test": True, "timestamp": time.time()}

            # Set and get
            mcp_cache.cache.set(test_key, test_value, 60)
            retrieved = mcp_cache.cache.get(test_key)

            # Clean up
            mcp_cache.cache.delete(test_key)

            if retrieved == test_value:
                stats = mcp_cache.get_stats()
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Cache system is working",
                    "details": {"cache_operations": True, "cache_stats": stats},
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "Cache retrieval failed",
                    "details": {"retrieval_failed": True},
                }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Cache system failed: {str(e)}",
                "details": {"error": str(e)},
            }

    async def _check_rate_limiter(self) -> Dict[str, Any]:
        """Check rate limiter health"""
        try:
            from .rate_limiter import global_rate_limiter

            # Test rate limiting
            test_user = "health_check_user"
            result = global_rate_limiter.check_rate_limit(test_user, "health_check")

            if result.get("allowed", False):
                stats = global_rate_limiter.get_global_stats()
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Rate limiter is working",
                    "details": {"rate_limit_check": True, "rate_limiter_stats": stats},
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "Rate limiter blocked request",
                    "details": {"blocked_request": True},
                }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Rate limiter failed: {str(e)}",
                "details": {"error": str(e)},
            }

    async def _check_metrics_system(self) -> Dict[str, Any]:
        """Check metrics system health"""
        try:
            from .metrics import global_metrics_collector

            # Test metrics collection
            test_metric = "health_check_metric"
            global_metrics_collector.counter(test_metric, 1, {"test": "true"})

            summary = global_metrics_collector.get_summary()

            return {
                "status": HealthStatus.HEALTHY,
                "message": "Metrics system is working",
                "details": {"metrics_collection": True, "metrics_summary": summary},
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Metrics system failed: {str(e)}",
                "details": {"error": str(e)},
            }

    def register_check(self, name: str, check_func: Callable):
        """Register a custom health check"""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def get_overall_status(self, results: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall system health"""
        if not results:
            return HealthStatus.UNKNOWN

        statuses = [check.status for check in results.values()]

        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED


class HealthMonitor:
    """Monitor system health and provide status endpoints"""

    def __init__(self, health_checker: HealthChecker = None):
        self.health_checker = health_checker or HealthChecker()
        self.monitoring_enabled = True

    async def get_health_status(self, include_details: bool = True) -> Dict[str, Any]:
        """Get current health status"""
        if not self.monitoring_enabled:
            return {
                "status": "monitoring_disabled",
                "timestamp": time.time(),
                "message": "Health monitoring is disabled",
            }

        # Record health check metrics
        start_time = mcp_metrics.tool_execution_start("health_check")

        try:
            results = await self.health_checker.run_all_checks()
            overall_status = self.health_checker.get_overall_status(results)

            mcp_metrics.tool_execution_end("health_check", start_time, True)

            response = {
                "status": overall_status.value,
                "timestamp": time.time(),
                "checks_passed": sum(
                    1 for r in results.values() if r.status == HealthStatus.HEALTHY
                ),
                "total_checks": len(results),
            }

            if include_details:
                response["checks"] = {name: check.to_dict() for name, check in results.items()}

            return response

        except Exception as e:
            mcp_metrics.tool_execution_end("health_check", start_time, False)
            return {
                "status": "error",
                "timestamp": time.time(),
                "message": f"Health check failed: {str(e)}",
                "error": str(e),
            }

    async def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness status (simplified health check)"""
        try:
            # Run only critical checks for readiness
            critical_checks = ["mcp_connection", "auth_system"]
            results = {}

            for check_name in critical_checks:
                if check_name in self.health_checker.checks:
                    results[check_name] = await self.health_checker.run_check(check_name)

            all_healthy = all(check.status == HealthStatus.HEALTHY for check in results.values())

            return {
                "ready": all_healthy,
                "timestamp": time.time(),
                "critical_checks": {name: check.status.value for name, check in results.items()},
            }

        except Exception as e:
            return {"ready": False, "timestamp": time.time(), "error": str(e)}

    async def get_liveness_status(self) -> Dict[str, Any]:
        """Get liveness status (basic system check)"""
        try:
            # Simple liveness check
            return {
                "alive": True,
                "timestamp": time.time(),
                "uptime": time.time(),  # In serverless, this is request time
                "version": "1.0.0",
            }

        except Exception as e:
            return {"alive": False, "timestamp": time.time(), "error": str(e)}


# Global health monitor instance
global_health_checker = HealthChecker()
health_monitor = HealthMonitor(global_health_checker)


async def check_health(include_details: bool = True) -> Dict[str, Any]:
    """Get system health status"""
    return await health_monitor.get_health_status(include_details)


async def check_readiness() -> Dict[str, Any]:
    """Get system readiness status"""
    return await health_monitor.get_readiness_status()


async def check_liveness() -> Dict[str, Any]:
    """Get system liveness status"""
    return await health_monitor.get_liveness_status()


def register_health_check(name: str, check_func: Callable):
    """Register a custom health check"""
    global_health_checker.register_check(name, check_func)
