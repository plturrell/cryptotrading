"""
Vercel Runtime Adapter for MCTS
Handles Vercel Edge Runtime limitations and provides compatibility layer
"""
import asyncio
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class VercelRuntimeError(Exception):
    """Vercel-specific runtime errors"""

    pass


class VercelTimeoutError(VercelRuntimeError):
    """Vercel function timeout error"""

    pass


class VercelMemoryError(VercelRuntimeError):
    """Vercel memory limit error"""

    pass


class VercelRuntimeAdapter:
    """Adapts MCTS operations for Vercel Edge Runtime constraints"""

    def __init__(self):
        self.is_vercel = bool(os.getenv("VERCEL_ENV") or os.getenv("VERCEL"))
        self.is_edge_runtime = os.getenv("VERCEL_EDGE", "false").lower() == "true"
        self.max_duration = int(os.getenv("VERCEL_MAX_DURATION", "30"))
        self.region = os.getenv("VERCEL_REGION", "unknown")

        # Runtime capabilities
        self.has_psutil = self._check_psutil()
        self.has_full_asyncio = self._check_asyncio()

        logger.info(
            f"Vercel Runtime Adapter initialized - Vercel: {self.is_vercel}, Edge: {self.is_edge_runtime}"
        )

    def _check_psutil(self) -> bool:
        """Check if psutil is available"""
        try:
            import psutil

            return True
        except ImportError:
            return False

    def _check_asyncio(self) -> bool:
        """Check if full asyncio features are available"""
        try:
            # Test if we can create tasks (Edge runtime limitation)
            # Don't actually create the task, just test the capability
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In running loop, test task creation capability
                return hasattr(asyncio, "create_task")
            else:
                # Not in running loop, assume full capabilities
                return True
        except (RuntimeError, AttributeError):
            return False

    def timeout_handler(self, timeout_seconds: Optional[int] = None):
        """Decorator to handle Vercel timeout limits"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                timeout = timeout_seconds or self.max_duration
                start_time = time.time()

                # Create timeout task
                async def timeout_monitor():
                    remaining = timeout - 5  # Leave 5 seconds for cleanup
                    await asyncio.sleep(remaining)
                    raise VercelTimeoutError(
                        f"Operation approaching Vercel timeout limit ({timeout}s)"
                    )

                if self.is_vercel:
                    # Run with timeout monitoring
                    try:
                        timeout_task = asyncio.create_task(timeout_monitor())
                        result = await func(*args, **kwargs)
                        timeout_task.cancel()
                        return result
                    except VercelTimeoutError:
                        # Clean timeout - return partial results
                        elapsed = time.time() - start_time
                        logger.warning(f"Vercel timeout reached after {elapsed:.1f}s")
                        return {
                            "error": "timeout",
                            "partial_results": True,
                            "elapsed_time": elapsed,
                            "message": "Operation timed out but may have partial results",
                        }
                else:
                    # Local development - no timeout
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    def memory_guard(self, max_memory_mb: int = 512):
        """Decorator to guard against memory limit violations"""

        def decorator(func):
            @wraps(func)
            async def wrapper(self_or_first_arg, *args, **kwargs):
                # Check if this is a method (has self) or function
                if hasattr(self_or_first_arg, "config"):
                    agent_self = self_or_first_arg
                    actual_args = args
                else:
                    agent_self = None
                    actual_args = (self_or_first_arg,) + args

                if self.is_vercel and self.is_edge_runtime:
                    # Periodic memory checks during execution
                    check_interval = 0.5  # seconds
                    last_check = time.time()

                    async def memory_monitor():
                        while True:
                            await asyncio.sleep(check_interval)
                            if agent_self and hasattr(agent_self, "_check_memory_limit"):
                                if agent_self._check_memory_limit():
                                    raise VercelMemoryError("Approaching Vercel memory limit")

                    try:
                        monitor_task = asyncio.create_task(memory_monitor())
                        if agent_self:
                            result = await func(agent_self, *actual_args, **kwargs)
                        else:
                            result = await func(*actual_args, **kwargs)
                        monitor_task.cancel()
                        return result
                    except VercelMemoryError:
                        logger.error("Memory limit exceeded in Vercel runtime")
                        return {
                            "error": "memory_limit",
                            "message": "Operation exceeded memory limits",
                        }
                else:
                    # Local development
                    if agent_self:
                        return await func(agent_self, *actual_args, **kwargs)
                    else:
                        return await func(*actual_args, **kwargs)

            return wrapper

        return decorator

    def edge_compatible(self, func: Callable) -> Callable:
        """Decorator to ensure Edge Runtime compatibility"""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.is_edge_runtime:
                # Remove or adapt incompatible features
                # Filter kwargs to remove Edge-incompatible options
                edge_safe_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["use_multiprocessing", "spawn_workers", "fork_processes"]
                }
                return await func(*args, **edge_safe_kwargs)
            else:
                return await func(*args, **kwargs)

        return wrapper

    def get_memory_usage(self, tree_size: Optional[int] = None) -> Dict[str, Any]:
        """Get memory usage in a Vercel-compatible way"""
        if self.has_psutil and not self.is_edge_runtime:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "available": True,
                "method": "psutil",
            }
        elif tree_size is not None:
            # Estimate based on tree size
            estimated_mb = (tree_size * 200) / (1024 * 1024)
            return {
                "estimated_mb": estimated_mb,
                "available": False,
                "method": "tree_size_estimate",
                "tree_size": tree_size,
            }
        else:
            return {
                "available": False,
                "method": "none",
                "message": "Memory tracking not available in Edge Runtime",
            }

    def handle_vercel_error(self, error: Exception) -> Dict[str, Any]:
        """Convert exceptions to Vercel-friendly error responses"""
        error_response = {
            "error": True,
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": time.time(),
        }

        if isinstance(error, VercelTimeoutError):
            error_response["code"] = "TIMEOUT"
            error_response["status"] = 504
        elif isinstance(error, VercelMemoryError):
            error_response["code"] = "MEMORY_LIMIT"
            error_response["status"] = 507
        elif isinstance(error, ValueError):
            error_response["code"] = "INVALID_INPUT"
            error_response["status"] = 400
        else:
            error_response["code"] = "INTERNAL_ERROR"
            error_response["status"] = 500

        # Add Vercel-specific context
        if self.is_vercel:
            error_response["vercel_context"] = {
                "region": self.region,
                "runtime": "edge" if self.is_edge_runtime else "nodejs",
                "max_duration": self.max_duration,
            }

        return error_response

    async def safe_cache_operation(
        self, operation: str, key: str, value: Any = None, ttl: int = 300
    ) -> Any:
        """Safe cache operations for Vercel KV/Redis"""
        try:
            if operation == "get":
                # Try Vercel KV first
                if os.getenv("KV_REST_API_URL"):
                    # Would integrate with Vercel KV here
                    pass
                return None
            elif operation == "set":
                if os.getenv("KV_REST_API_URL"):
                    # Would integrate with Vercel KV here
                    pass
                return True
        except Exception as e:
            logger.error(f"Cache operation failed: {e}")
            return None if operation == "get" else False

    def get_runtime_info(self) -> Dict[str, Any]:
        """Get current runtime information"""
        return {
            "is_vercel": self.is_vercel,
            "is_edge_runtime": self.is_edge_runtime,
            "has_psutil": self.has_psutil,
            "has_full_asyncio": self.has_full_asyncio,
            "max_duration": self.max_duration,
            "region": self.region,
            "environment": os.getenv("VERCEL_ENV", "development"),
            "runtime_version": os.getenv("AWS_LAMBDA_RUNTIME_API", "local"),
        }


# Global adapter instance
vercel_adapter = VercelRuntimeAdapter()


def vercel_edge_handler(func: Callable) -> Callable:
    """Main decorator for Vercel Edge Function compatibility"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Apply all Vercel adaptations
            result = await vercel_adapter.timeout_handler()(
                vercel_adapter.memory_guard()(vercel_adapter.edge_compatible(func))
            )(*args, **kwargs)

            return result

        except Exception as e:
            # Handle all errors in Vercel-friendly way
            return vercel_adapter.handle_vercel_error(e)

    return wrapper
