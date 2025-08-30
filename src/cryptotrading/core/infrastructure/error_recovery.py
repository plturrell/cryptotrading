"""
Comprehensive error recovery and circuit breaker implementation
Handles graceful degradation and automatic recovery for critical systems
"""

import asyncio
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..config.environment import get_feature_flags, is_vercel

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ErrorMetrics:
    """Error tracking metrics"""

    total_requests: int = 0
    failed_requests: int = 0
    success_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


@dataclass
class RecoveryConfig:
    """Recovery configuration"""

    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes to close circuit
    request_timeout: int = 10  # Request timeout seconds
    enable_fallback: bool = True  # Enable fallback mechanisms
    max_retry_attempts: int = 3  # Maximum retry attempts
    retry_delay: float = 1.0  # Base retry delay
    exponential_backoff: bool = True  # Use exponential backoff


class CircuitBreaker:
    """Circuit breaker with automatic recovery"""

    def __init__(self, name: str, config: Optional[RecoveryConfig] = None):
        self.name = name
        self.config = config or RecoveryConfig()
        self.state = CircuitBreakerState.CLOSED
        self.metrics = ErrorMetrics()
        self._lock = threading.RLock()
        self._next_attempt_time = 0

        logger.info(
            f"CircuitBreaker '{name}' initialized: threshold={self.config.failure_threshold}"
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker protection"""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)

        return wrapper

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() < self._next_attempt_time:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    # Try half-open
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' trying HALF_OPEN")

        start_time = time.time()
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.request_timeout
                )
            else:
                result = func(*args, **kwargs)

            # Record success
            await self._record_success()
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            await self._record_failure(e, execution_time)
            raise

    async def _record_success(self):
        """Record successful operation"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.success_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    logger.info(f"Circuit breaker '{self.name}' closed after recovery")

    async def _record_failure(self, error: Exception, execution_time: float):
        """Record failed operation"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()

            # Log failure details
            logger.error(
                f"Circuit breaker '{self.name}' failure: {error} (took {execution_time:.2f}s)"
            )

            if (
                self.state == CircuitBreakerState.CLOSED
                and self.metrics.consecutive_failures >= self.config.failure_threshold
            ):
                # Open circuit
                self.state = CircuitBreakerState.OPEN
                self._next_attempt_time = time.time() + self.config.recovery_timeout
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self.metrics.consecutive_failures} failures"
                )

            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Back to open
                self.state = CircuitBreakerState.OPEN
                self._next_attempt_time = time.time() + self.config.recovery_timeout
                logger.warning(
                    f"Circuit breaker '{self.name}' back to OPEN after half-open failure"
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            failure_rate = (
                self.metrics.failed_requests / self.metrics.total_requests
                if self.metrics.total_requests > 0
                else 0
            )

            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.metrics.total_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_requests": self.metrics.success_requests,
                "failure_rate": failure_rate,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "last_failure": self.metrics.last_failure_time,
                "last_success": self.metrics.last_success_time,
                "next_attempt_time": self._next_attempt_time
                if self.state == CircuitBreakerState.OPEN
                else None,
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""

    pass


class RetryManager:
    """Advanced retry manager with exponential backoff"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter

    async def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.max_attempts - 1:
                    # Last attempt, don't sleep
                    break

                # Calculate delay
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)

        # All attempts failed
        raise RetryExhaustedError(f"All {self.max_attempts} attempts failed") from last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay"""
        if self.exponential_backoff:
            delay = self.base_delay * (2**attempt)
        else:
            delay = self.base_delay

        # Add jitter to prevent thundering herd
        if self.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5  # 50-100% of calculated delay

        return delay


class RetryExhaustedError(Exception):
    """Exception raised when all retry attempts are exhausted"""

    pass


class FallbackManager:
    """Manages fallback strategies for failed services"""

    def __init__(self):
        self.fallbacks: Dict[str, List[Callable]] = {}
        self.flags = get_feature_flags()

    def register_fallback(self, service_name: str, fallback_func: Callable, priority: int = 0):
        """Register a fallback function for a service"""
        if service_name not in self.fallbacks:
            self.fallbacks[service_name] = []

        self.fallbacks[service_name].append((priority, fallback_func))
        # Sort by priority (higher priority first)
        self.fallbacks[service_name].sort(key=lambda x: x[0], reverse=True)

        logger.info(f"Registered fallback for '{service_name}' with priority {priority}")

    async def execute_with_fallback(
        self, service_name: str, primary_func: Callable, *args, **kwargs
    ) -> Any:
        """Execute primary function with fallback on failure"""
        try:
            # Try primary function
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)

        except Exception as primary_error:
            logger.warning(f"Primary function failed for '{service_name}': {primary_error}")

            # Try fallbacks in order of priority
            if service_name in self.fallbacks:
                for priority, fallback_func in self.fallbacks[service_name]:
                    try:
                        logger.info(f"Trying fallback for '{service_name}' (priority {priority})")

                        if asyncio.iscoroutinefunction(fallback_func):
                            return await fallback_func(*args, **kwargs)
                        else:
                            return fallback_func(*args, **kwargs)

                    except Exception as fallback_error:
                        logger.warning(f"Fallback failed for '{service_name}': {fallback_error}")
                        continue

            # No fallbacks available or all failed
            raise FallbackExhaustedError(
                f"No working fallback for '{service_name}'"
            ) from primary_error


class FallbackExhaustedError(Exception):
    """Exception raised when all fallbacks are exhausted"""

    pass


class ErrorRecoverySystem:
    """Comprehensive error recovery system"""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        self.fallback_manager = FallbackManager()
        self.flags = get_feature_flags()

        logger.info("ErrorRecoverySystem initialized")

    def get_circuit_breaker(
        self, name: str, config: Optional[RecoveryConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            # Adjust config for environment
            if config is None:
                config = RecoveryConfig()

            if is_vercel():
                # More aggressive settings for serverless
                config.failure_threshold = 3
                config.recovery_timeout = 30
                config.request_timeout = 5

            self.circuit_breakers[name] = CircuitBreaker(name, config)

        return self.circuit_breakers[name]

    async def execute_with_recovery(
        self,
        service_name: str,
        func: Callable,
        *args,
        use_circuit_breaker: bool = True,
        use_retry: bool = True,
        **kwargs,
    ) -> Any:
        """Execute function with full recovery protection"""
        try:
            if use_circuit_breaker:
                circuit_breaker = self.get_circuit_breaker(service_name)

                if use_retry:
                    # Circuit breaker + retry
                    return await self.retry_manager.retry(
                        circuit_breaker.call, func, *args, **kwargs
                    )
                else:
                    # Circuit breaker only
                    return await circuit_breaker.call(func, *args, **kwargs)
            elif use_retry:
                # Retry only
                return await self.retry_manager.retry(func, *args, **kwargs)
            else:
                # No protection
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

        except Exception as e:
            # Try fallback as last resort
            return await self.fallback_manager.execute_with_fallback(
                service_name, func, *args, **kwargs
            )

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        health = {
            "circuit_breakers": {},
            "total_services": len(self.circuit_breakers),
            "healthy_services": 0,
            "degraded_services": 0,
            "failed_services": 0,
        }

        for name, cb in self.circuit_breakers.items():
            metrics = cb.get_metrics()
            health["circuit_breakers"][name] = metrics

            if metrics["state"] == "closed":
                health["healthy_services"] += 1
            elif metrics["state"] == "half_open":
                health["degraded_services"] += 1
            else:
                health["failed_services"] += 1

        health["overall_health"] = (
            "healthy"
            if health["failed_services"] == 0
            else "degraded"
            if health["healthy_services"] > 0
            else "critical"
        )

        return health


# Global error recovery system
_global_recovery_system: Optional[ErrorRecoverySystem] = None


def get_error_recovery_system() -> ErrorRecoverySystem:
    """Get global error recovery system"""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = ErrorRecoverySystem()
    return _global_recovery_system


# Convenience decorators
def circuit_breaker(name: str, config: Optional[RecoveryConfig] = None):
    """Circuit breaker decorator"""

    def decorator(func: Callable) -> Callable:
        recovery_system = get_error_recovery_system()
        cb = recovery_system.get_circuit_breaker(name, config)
        return cb(func)

    return decorator


def with_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Retry decorator"""

    def decorator(func: Callable) -> Callable:
        retry_manager = RetryManager(max_attempts, base_delay)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_manager.retry(func, *args, **kwargs)

        return wrapper

    return decorator


def with_fallback(service_name: str, fallback_func: Callable, priority: int = 0):
    """Fallback decorator"""

    def decorator(func: Callable) -> Callable:
        recovery_system = get_error_recovery_system()
        recovery_system.fallback_manager.register_fallback(service_name, fallback_func, priority)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await recovery_system.fallback_manager.execute_with_fallback(
                service_name, func, *args, **kwargs
            )

        return wrapper

    return decorator
