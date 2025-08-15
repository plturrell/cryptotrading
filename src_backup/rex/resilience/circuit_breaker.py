"""
Production-grade circuit breaker implementation
Prevents cascade failures and provides graceful degradation
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import threading
from collections import deque

from ..logging.production_logger import get_logger
from ..monitoring.alerts import alert_manager, AlertSeverity

logger = get_logger(__name__)

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Number of failures to trigger opening
    success_threshold: int = 3  # Number of successes to close from half-open
    timeout: float = 60.0  # Seconds before transitioning to half-open
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures
    name: str = "default"
    
    # Advanced configuration
    sliding_window_size: int = 10  # Size of sliding window for failure rate
    minimum_throughput: int = 4  # Minimum requests before considering failure rate
    failure_rate_threshold: float = 0.5  # 50% failure rate to trigger opening
    slow_call_duration_threshold: float = 10.0  # Seconds to consider a call slow
    slow_call_rate_threshold: float = 0.5  # 50% slow calls to trigger opening

class CircuitBreakerError(Exception):
    """Circuit breaker is open"""
    pass

class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is open and blocking requests"""
    pass

class CircuitBreakerStats:
    """Circuit breaker statistics"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.calls = deque(maxlen=window_size)
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_slow_calls = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state_change_time = datetime.utcnow()
    
    def record_call(self, success: bool, duration: float, slow_threshold: float):
        """Record a call result"""
        now = datetime.utcnow()
        is_slow = duration > slow_threshold
        
        call_record = {
            'timestamp': now,
            'success': success,
            'duration': duration,
            'slow': is_slow
        }
        
        self.calls.append(call_record)
        self.total_calls += 1
        
        if success:
            self.total_successes += 1
            self.last_success_time = now
        else:
            self.total_failures += 1
            self.last_failure_time = now
        
        if is_slow:
            self.total_slow_calls += 1
    
    def get_failure_rate(self) -> float:
        """Get current failure rate in sliding window"""
        if len(self.calls) == 0:
            return 0.0
        
        failures = sum(1 for call in self.calls if not call['success'])
        return failures / len(self.calls)
    
    def get_slow_call_rate(self) -> float:
        """Get current slow call rate in sliding window"""
        if len(self.calls) == 0:
            return 0.0
        
        slow_calls = sum(1 for call in self.calls if call['slow'])
        return slow_calls / len(self.calls)
    
    def get_throughput(self) -> int:
        """Get number of calls in current window"""
        return len(self.calls)
    
    def clear_window(self):
        """Clear the sliding window"""
        self.calls.clear()

class CircuitBreaker:
    """Production circuit breaker with advanced failure detection"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats(config.sliding_window_size)
        self.last_failure_time = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.half_open_calls = 0
        self._lock = threading.RLock()
        
        logger.info(f"Circuit breaker initialized: {config.name}",
                   failure_threshold=config.failure_threshold,
                   timeout=config.timeout)
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator usage"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        # Check if circuit should allow the call
        if not self._should_allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.config.name}' is OPEN"
            )
        
        start_time = time.time()
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record successful call
            duration = time.time() - start_time
            await self._record_success(duration)
            
            return result
            
        except self.config.expected_exception as e:
            # Record failed call
            duration = time.time() - start_time
            await self._record_failure(duration, e)
            raise
        except Exception as e:
            # Unexpected exception - don't count as failure
            duration = time.time() - start_time
            logger.warning(f"Unexpected exception in circuit breaker {self.config.name}: {e}")
            raise
    
    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on current state"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if self.last_failure_time:
                    time_since_failure = time.time() - self.last_failure_time
                    if time_since_failure >= self.config.timeout:
                        # Transition to half-open
                        self._transition_to_half_open()
                        return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                return True
            
            return False
    
    async def _record_success(self, duration: float):
        """Record successful call and update state"""
        with self._lock:
            self.stats.record_call(True, duration, self.config.slow_call_duration_threshold)
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                
                # Check if we should close the circuit
                if self.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to_closed()
    
    async def _record_failure(self, duration: float, exception: Exception):
        """Record failed call and update state"""
        with self._lock:
            self.stats.record_call(False, duration, self.config.slow_call_duration_threshold)
            self.last_failure_time = time.time()
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                should_open = self._should_open_circuit()
                if should_open:
                    await self._transition_to_open(exception)
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                await self._transition_to_open(exception)
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on failure criteria"""
        # Check consecutive failures threshold
        if self.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Check failure rate threshold (only if we have minimum throughput)
        if self.stats.get_throughput() >= self.config.minimum_throughput:
            failure_rate = self.stats.get_failure_rate()
            if failure_rate >= self.config.failure_rate_threshold:
                return True
            
            # Check slow call rate threshold
            slow_call_rate = self.stats.get_slow_call_rate()
            if slow_call_rate >= self.config.slow_call_rate_threshold:
                return True
        
        return False
    
    async def _transition_to_open(self, exception: Exception = None):
        """Transition circuit to OPEN state"""
        if self.state != CircuitState.OPEN:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.stats.state_change_time = datetime.utcnow()
            self.half_open_calls = 0
            
            logger.warning(f"Circuit breaker '{self.config.name}' opened",
                          previous_state=old_state.value,
                          consecutive_failures=self.consecutive_failures,
                          failure_rate=self.stats.get_failure_rate(),
                          slow_call_rate=self.stats.get_slow_call_rate())
            
            # Send alert
            await alert_manager.process_event({
                'title': f'Circuit breaker opened: {self.config.name}',
                'component': 'circuit_breaker',
                'severity': AlertSeverity.HIGH.value,
                'source': 'resilience_system',
                'circuit_name': self.config.name,
                'failure_count': self.consecutive_failures,
                'failure_rate': self.stats.get_failure_rate(),
                'exception': str(exception) if exception else None
            })
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        if self.state != CircuitState.HALF_OPEN:
            old_state = self.state
            self.state = CircuitState.HALF_OPEN
            self.stats.state_change_time = datetime.utcnow()
            self.half_open_calls = 0
            self.consecutive_successes = 0
            
            logger.info(f"Circuit breaker '{self.config.name}' half-opened",
                       previous_state=old_state.value)
    
    async def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        if self.state != CircuitState.CLOSED:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.stats.state_change_time = datetime.utcnow()
            self.consecutive_failures = 0
            self.half_open_calls = 0
            
            logger.info(f"Circuit breaker '{self.config.name}' closed",
                       previous_state=old_state.value,
                       consecutive_successes=self.consecutive_successes)
            
            # Send recovery alert
            await alert_manager.process_event({
                'title': f'Circuit breaker closed: {self.config.name}',
                'component': 'circuit_breaker',
                'severity': AlertSeverity.LOW.value,
                'source': 'resilience_system',
                'circuit_name': self.config.name,
                'success_count': self.consecutive_successes
            })
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            time_in_current_state = (
                datetime.utcnow() - self.stats.state_change_time
            ).total_seconds()
            
            return {
                'name': self.config.name,
                'state': self.state.value,
                'time_in_current_state_seconds': time_in_current_state,
                'consecutive_failures': self.consecutive_failures,
                'consecutive_successes': self.consecutive_successes,
                'total_calls': self.stats.total_calls,
                'total_failures': self.stats.total_failures,
                'total_successes': self.stats.total_successes,
                'total_slow_calls': self.stats.total_slow_calls,
                'current_failure_rate': self.stats.get_failure_rate(),
                'current_slow_call_rate': self.stats.get_slow_call_rate(),
                'current_throughput': self.stats.get_throughput(),
                'last_failure_time': self.last_failure_time,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout,
                    'failure_rate_threshold': self.config.failure_rate_threshold,
                    'slow_call_rate_threshold': self.config.slow_call_rate_threshold
                }
            }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.stats = CircuitBreakerStats(self.config.sliding_window_size)
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.half_open_calls = 0
            self.last_failure_time = None
            
            logger.info(f"Circuit breaker '{self.config.name}' reset")

class CircuitBreakerManager:
    """Manage multiple circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._setup_default_breakers()
    
    def _setup_default_breakers(self):
        """Setup default circuit breakers for common services"""
        
        # Database circuit breaker
        self.register(CircuitBreakerConfig(
            name="database",
            failure_threshold=3,
            success_threshold=2,
            timeout=30.0,
            expected_exception=(Exception,),
            failure_rate_threshold=0.3,
            slow_call_duration_threshold=5.0
        ))
        
        # Cache circuit breaker
        self.register(CircuitBreakerConfig(
            name="cache",
            failure_threshold=5,
            success_threshold=3,
            timeout=15.0,
            expected_exception=(Exception,),
            failure_rate_threshold=0.5,
            slow_call_duration_threshold=2.0
        ))
        
        # External API circuit breaker
        self.register(CircuitBreakerConfig(
            name="external_api",
            failure_threshold=5,
            success_threshold=3,
            timeout=60.0,
            expected_exception=(Exception,),
            failure_rate_threshold=0.4,
            slow_call_duration_threshold=10.0
        ))
        
        # AI provider circuit breaker
        self.register(CircuitBreakerConfig(
            name="ai_provider",
            failure_threshold=3,
            success_threshold=2,
            timeout=120.0,  # AI services might take longer to recover
            expected_exception=(Exception,),
            failure_rate_threshold=0.3,
            slow_call_duration_threshold=30.0
        ))
        
        # Agent communication circuit breaker
        self.register(CircuitBreakerConfig(
            name="agent_communication",
            failure_threshold=4,
            success_threshold=2,
            timeout=45.0,
            expected_exception=(Exception,),
            failure_rate_threshold=0.4,
            slow_call_duration_threshold=15.0
        ))
    
    def register(self, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker"""
        circuit_breaker = CircuitBreaker(config)
        self.circuit_breakers[config.name] = circuit_breaker
        
        logger.info(f"Registered circuit breaker: {config.name}")
        return circuit_breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: breaker.get_stats()
            for name, breaker in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        
        logger.info("All circuit breakers reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all circuit breakers"""
        total_breakers = len(self.circuit_breakers)
        open_breakers = [
            name for name, breaker in self.circuit_breakers.items()
            if breaker.get_state() == CircuitState.OPEN
        ]
        half_open_breakers = [
            name for name, breaker in self.circuit_breakers.items()
            if breaker.get_state() == CircuitState.HALF_OPEN
        ]
        
        return {
            'total_circuit_breakers': total_breakers,
            'open_circuit_breakers': len(open_breakers),
            'half_open_circuit_breakers': len(half_open_breakers),
            'closed_circuit_breakers': total_breakers - len(open_breakers) - len(half_open_breakers),
            'open_breaker_names': open_breakers,
            'half_open_breaker_names': half_open_breakers
        }

# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()

def circuit_breaker(
    name: str = None,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout: float = 60.0,
    expected_exception: tuple = (Exception,),
    failure_rate_threshold: float = 0.5,
    slow_call_duration_threshold: float = 10.0
):
    """
    Decorator to add circuit breaker protection to a function
    
    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Number of failures to trigger opening
        success_threshold: Number of successes to close from half-open
        timeout: Seconds before transitioning to half-open
        expected_exception: Exceptions that count as failures
        failure_rate_threshold: Failure rate to trigger opening
        slow_call_duration_threshold: Seconds to consider a call slow
    """
    def decorator(func):
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        
        # Check if circuit breaker already exists
        breaker = circuit_manager.get(breaker_name)
        if not breaker:
            config = CircuitBreakerConfig(
                name=breaker_name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                expected_exception=expected_exception,
                failure_rate_threshold=failure_rate_threshold,
                slow_call_duration_threshold=slow_call_duration_threshold
            )
            breaker = circuit_manager.register(config)
        
        return breaker(func)
    
    return decorator

# Convenience decorators for common services
def database_circuit_breaker(func):
    """Circuit breaker for database operations"""
    return circuit_manager.get("database")(func)

def cache_circuit_breaker(func):
    """Circuit breaker for cache operations"""
    return circuit_manager.get("cache")(func)

def external_api_circuit_breaker(func):
    """Circuit breaker for external API calls"""
    return circuit_manager.get("external_api")(func)

def ai_provider_circuit_breaker(func):
    """Circuit breaker for AI provider calls"""
    return circuit_manager.get("ai_provider")(func)

def agent_communication_circuit_breaker(func):
    """Circuit breaker for agent communication"""
    return circuit_manager.get("agent_communication")(func)