"""
Optimized caching system with memory management for local and Vercel environments
"""

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from ..config.environment import get_feature_flags, is_serverless, is_vercel

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    data: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0


class MemoryLRUCache:
    """Memory-efficient LRU cache with size limits"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_bytes = 0
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "memory_evictions": 0}

        logger.info(
            f"MemoryLRUCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB"
        )

    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes"""
        try:
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, dict):
                return len(json.dumps(data).encode("utf-8"))
            else:
                return len(pickle.dumps(data))
        except Exception:
            # Fallback estimation
            return 1024  # 1KB default

    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [key for key, entry in self.cache.items() if entry.expires_at < current_time]

        for key in expired_keys:
            self._remove_entry(key)

    def _evict_lru(self, target_memory: Optional[int] = None):
        """Evict least recently used entries"""
        if target_memory is None:
            target_memory = int(self.max_memory_bytes * 0.8)  # Target 80% of max memory

        while (
            self.current_memory_bytes > target_memory or len(self.cache) >= self.max_size
        ) and self.cache:
            # Remove least recently used item
            key = next(iter(self.cache))
            self._remove_entry(key)
            self._stats["evictions"] += 1

    def _remove_entry(self, key: str):
        """Remove entry and update memory tracking"""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory_bytes -= entry.size_bytes
            del self.cache[key]

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            self._evict_expired()

            if key in self.cache:
                entry = self.cache[key]

                # Check if expired
                if entry.expires_at < time.time():
                    self._remove_entry(key)
                    self._stats["misses"] += 1
                    return None

                # Move to end (most recently used)
                self.cache.move_to_end(key)

                # Update access stats
                entry.access_count += 1
                entry.last_accessed = time.time()

                self._stats["hits"] += 1
                return entry.data

            self._stats["misses"] += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set item in cache with TTL"""
        with self._lock:
            current_time = time.time()
            size_bytes = self._calculate_size(value)

            # Check if this single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False

            # Evict items if needed
            self._evict_expired()

            # Check if we need to evict for memory
            required_memory = self.current_memory_bytes + size_bytes
            if required_memory > self.max_memory_bytes:
                self._evict_lru(self.max_memory_bytes - size_bytes)
                self._stats["memory_evictions"] += 1

            # Check if we need to evict for count
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            # Create new entry
            entry = CacheEntry(
                data=value,
                created_at=current_time,
                expires_at=current_time + ttl_seconds,
                access_count=1,
                last_accessed=current_time,
                size_bytes=size_bytes,
            )

            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)

            # Add new entry
            self.cache[key] = entry
            self.current_memory_bytes += size_bytes

            return True

    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.current_memory_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = (
                (self._stats["hits"] / (self._stats["hits"] + self._stats["misses"]))
                if (self._stats["hits"] + self._stats["misses"]) > 0
                else 0
            )

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_bytes": self.current_memory_bytes,
                "memory_usage_mb": self.current_memory_bytes / 1024 / 1024,
                "max_memory_mb": self.max_memory_bytes / 1024 / 1024,
                "hit_rate": hit_rate,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "memory_evictions": self._stats["memory_evictions"],
            }


class OptimizedCacheManager:
    """Optimized cache manager with environment-aware configuration"""

    def __init__(self):
        self.flags = get_feature_flags()
        self.config = self._get_cache_config()

        # Initialize appropriate cache based on environment
        if is_serverless():
            # Use memory cache for serverless
            self.cache = MemoryLRUCache(
                max_size=self.config["max_size"], max_memory_mb=self.config["max_memory_mb"]
            )
            self.cache_type = "memory"
        else:
            # Try Redis first, fallback to memory
            try:
                self.cache = self._init_redis_cache()
                self.cache_type = "redis"
            except Exception as e:
                logger.warning(f"Redis cache initialization failed, using memory cache: {e}")
                self.cache = MemoryLRUCache(
                    max_size=self.config["max_size"], max_memory_mb=self.config["max_memory_mb"]
                )
                self.cache_type = "memory"

        logger.info(f"OptimizedCacheManager initialized with {self.cache_type} cache")

    def _get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration based on environment"""
        if is_vercel():
            return {
                "max_size": 500,
                "max_memory_mb": 50,  # Conservative for Vercel
                "default_ttl": 300,  # 5 minutes
                "enable_compression": True,
            }
        elif is_serverless():
            return {
                "max_size": 1000,
                "max_memory_mb": 100,
                "default_ttl": 600,  # 10 minutes
                "enable_compression": True,
            }
        else:
            return {
                "max_size": 5000,
                "max_memory_mb": 256,
                "default_ttl": 1800,  # 30 minutes
                "enable_compression": False,
            }

    def _init_redis_cache(self):
        """Initialize Redis cache if available"""
        import os

        import redis

        redis_url = os.environ.get("REDIS_URL")
        if not redis_url:
            raise Exception("Redis URL not configured")

        redis_client = redis.from_url(redis_url)
        redis_client.ping()  # Test connection

        return RedisCache(redis_client, self.config)

    def _generate_key(self, namespace: str, key: str, **kwargs) -> str:
        """Generate cache key with namespace and parameters"""
        if kwargs:
            # Include parameters in key
            params_str = json.dumps(kwargs, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            return f"{namespace}:{key}:{params_hash}"
        return f"{namespace}:{key}"

    async def get(self, namespace: str, key: str, **kwargs) -> Optional[Any]:
        """Get item from cache"""
        cache_key = self._generate_key(namespace, key, **kwargs)

        try:
            if hasattr(self.cache, "get"):
                return self.cache.get(cache_key)
            else:
                # Async cache
                return await self.cache.get(cache_key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None, **kwargs
    ) -> bool:
        """Set item in cache"""
        cache_key = self._generate_key(namespace, key, **kwargs)
        ttl = ttl_seconds or self.config["default_ttl"]

        try:
            if hasattr(self.cache, "set"):
                return self.cache.set(cache_key, value, ttl)
            else:
                # Async cache
                return await self.cache.set(cache_key, value, ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, namespace: str, key: str, **kwargs) -> bool:
        """Delete item from cache"""
        cache_key = self._generate_key(namespace, key, **kwargs)

        try:
            if hasattr(self.cache, "delete"):
                return self.cache.delete(cache_key)
            else:
                # Async cache
                return await self.cache.delete(cache_key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all items in a namespace"""
        try:
            if self.cache_type == "memory":
                # For memory cache, we need to iterate and delete
                keys_to_delete = [
                    key for key in self.cache.cache.keys() if key.startswith(f"{namespace}:")
                ]
                for key in keys_to_delete:
                    self.cache.delete(key)
                return True
            else:
                # For Redis, use pattern deletion
                return await self.cache.clear_pattern(f"{namespace}:*")
        except Exception as e:
            logger.error(f"Cache clear namespace error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = self.cache.get_stats()
            stats["cache_type"] = self.cache_type
            stats["config"] = self.config
            return stats
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}


class RedisCache:
    """Redis-based cache implementation"""

    def __init__(self, redis_client, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

    def get(self, key: str) -> Optional[Any]:
        """Get item from Redis"""
        try:
            data = self.redis.get(key)
            if data:
                self._stats["hits"] += 1
                return pickle.loads(data)
            else:
                self._stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats["misses"] += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Set item in Redis"""
        try:
            data = pickle.dumps(value)
            self.redis.setex(key, ttl_seconds, data)
            self._stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete item from Redis"""
        try:
            result = self.redis.delete(key)
            self._stats["deletes"] += 1
            return bool(result)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> bool:
        """Clear all keys matching pattern"""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        try:
            info = self.redis.info("memory")
            return {
                "memory_usage_bytes": info.get("used_memory", 0),
                "memory_usage_mb": info.get("used_memory", 0) / 1024 / 1024,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "connected": True,
            }
        except Exception as e:
            return {"error": str(e), "connected": False}


# Global cache manager instance
_global_cache_manager: Optional[OptimizedCacheManager] = None


def get_cache_manager() -> OptimizedCacheManager:
    """Get global cache manager instance"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = OptimizedCacheManager()
    return _global_cache_manager


# Convenience functions
async def cache_get(namespace: str, key: str, **kwargs) -> Optional[Any]:
    """Get item from cache"""
    manager = get_cache_manager()
    return await manager.get(namespace, key, **kwargs)


async def cache_set(
    namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None, **kwargs
) -> bool:
    """Set item in cache"""
    manager = get_cache_manager()
    return await manager.set(namespace, key, value, ttl_seconds, **kwargs)


async def cache_delete(namespace: str, key: str, **kwargs) -> bool:
    """Delete item from cache"""
    manager = get_cache_manager()
    return await manager.delete(namespace, key, **kwargs)
