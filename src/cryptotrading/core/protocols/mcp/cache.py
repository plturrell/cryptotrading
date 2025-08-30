"""
MCP Lightweight Caching Layer
Simple in-memory and external caching for serverless environments
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL"""

    value: Any
    expires_at: float
    created_at: float

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() > self.expires_at


class InMemoryCache:
    """Simple in-memory cache for serverless functions"""

    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.max_size = 1000  # Limit memory usage

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None

        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            return None

        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        now = time.time()

        # Evict expired entries if cache is getting full
        if len(self.cache) >= self.max_size:
            self._evict_expired()

        # If still full, remove oldest entries
        if len(self.cache) >= self.max_size:
            self._evict_oldest(self.max_size // 4)  # Remove 25%

        self.cache[key] = CacheEntry(value=value, expires_at=now + ttl, created_at=now)

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()

    def _evict_expired(self) -> None:
        """Remove expired entries"""
        now = time.time()
        expired_keys = [key for key, entry in self.cache.items() if entry.expires_at <= now]
        for key in expired_keys:
            del self.cache[key]

    def _evict_oldest(self, count: int) -> None:
        """Remove oldest entries"""
        if count <= 0:
            return

        # Sort by creation time and remove oldest
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1].created_at)

        for key, _ in sorted_items[:count]:
            del self.cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = time.time()
        expired_count = sum(1 for entry in self.cache.values() if entry.expires_at <= now)

        return {
            "total_entries": len(self.cache),
            "expired_entries": expired_count,
            "active_entries": len(self.cache) - expired_count,
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size,
        }


class MCPResourceCache:
    """Cache for MCP resources and tool results"""

    def __init__(self, cache_backend: InMemoryCache = None):
        self.cache = cache_backend or InMemoryCache(default_ttl=300)
        self.resource_ttl = 600  # 10 minutes for resources
        self.tool_result_ttl = 60  # 1 minute for tool results

    def _make_key(self, prefix: str, *args) -> str:
        """Create cache key"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get cached resource"""
        key = self._make_key("resource", uri)
        return self.cache.get(key)

    def set_resource(self, uri: str, content: Dict[str, Any]) -> None:
        """Cache resource content"""
        key = self._make_key("resource", uri)
        self.cache.set(key, content, self.resource_ttl)

    def get_tool_result(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached tool result"""
        args_str = json.dumps(arguments, sort_keys=True)
        key = self._make_key("tool", tool_name, args_str)
        return self.cache.get(key)

    def set_tool_result(
        self, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Cache tool result"""
        args_str = json.dumps(arguments, sort_keys=True)
        key = self._make_key("tool", tool_name, args_str)
        self.cache.set(key, result, self.tool_result_ttl)

    def get_market_data(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get cached market data"""
        key = self._make_key("market", symbol, timeframe)
        return self.cache.get(key)

    def set_market_data(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> None:
        """Cache market data"""
        key = self._make_key("market", symbol, timeframe)
        # Market data expires quickly
        ttl = 30 if timeframe in ["1m", "5m"] else 60
        self.cache.set(key, data, ttl)

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        count = 0
        keys_to_delete = []

        for key in self.cache.cache.keys():
            if pattern in key:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self.cache.delete(key)
            count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        base_stats = self.cache.get_stats()

        # Count entries by type
        resource_count = 0
        tool_count = 0
        market_count = 0

        for key in self.cache.cache.keys():
            if key.startswith("resource"):
                resource_count += 1
            elif key.startswith("tool"):
                tool_count += 1
            elif key.startswith("market"):
                market_count += 1

        return {
            **base_stats,
            "resource_entries": resource_count,
            "tool_entries": tool_count,
            "market_entries": market_count,
        }


class CacheDecorator:
    """Decorator for caching function results"""

    def __init__(self, cache: MCPResourceCache, ttl: int = 300):
        self.cache = cache
        self.ttl = ttl

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            func_name = func.__name__
            cache_key = self.cache._make_key(
                "func", func_name, str(args), json.dumps(kwargs, sort_keys=True)
            )

            # Try to get from cache
            result = self.cache.cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache.cache.set(cache_key, result, self.ttl)
            logger.debug(f"Cache miss for {func_name}, result cached")

            return result

        return wrapper


# Global cache instances
_global_cache = InMemoryCache(default_ttl=300)
mcp_cache = MCPResourceCache(_global_cache)


def cache_tool_result(ttl: int = 60):
    """Decorator to cache tool results"""
    return CacheDecorator(mcp_cache, ttl)


def cache_resource(ttl: int = 600):
    """Decorator to cache resource access"""
    return CacheDecorator(mcp_cache, ttl)


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    return mcp_cache.get_stats()


def clear_cache() -> None:
    """Clear global cache"""
    _global_cache.clear()
    logger.info("Global MCP cache cleared")


def invalidate_cache_pattern(pattern: str) -> int:
    """Invalidate cache entries matching pattern"""
    count = mcp_cache.invalidate_pattern(pattern)
    logger.info(f"Invalidated {count} cache entries matching '{pattern}'")
    return count
