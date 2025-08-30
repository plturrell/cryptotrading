"""
Redis caching layer for production performance
Only used for caching, not primary storage
"""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import redis

logger = logging.getLogger(__name__)


class CacheConfig:
    """Cache configuration"""

    REDIS_URL = "redis://localhost:6379"
    DEFAULT_TTL = 3600  # 1 hour
    REGISTRY_TTL = 1800  # 30 minutes for registry data
    ANALYSIS_TTL = 7200  # 2 hours for AI analysis
    MARKET_DATA_TTL = 300  # 5 minutes for market data

    # Local cache limits
    MAX_LOCAL_CACHE_SIZE = 1000  # Maximum entries in local cache
    MAX_LOCAL_CACHE_MEMORY_MB = 100  # Maximum memory usage in MB
    CLEANUP_INTERVAL = 300  # Cleanup interval in seconds


class LRUCache:
    """Thread-safe LRU cache with size and memory limits"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0

    def get(self, key: str) -> Optional[Tuple[Any, Optional[datetime]]]:
        """Get value and expiry from cache"""
        with self._lock:
            if key not in self._cache:
                return None

            # Move to end (most recently used)
            value, expires_at, size = self._cache.pop(key)
            self._cache[key] = (value, expires_at, size)

            # Check expiry
            if expires_at and datetime.now() > expires_at:
                self._remove_key(key)
                return None

            return value, expires_at

    def set(self, key: str, value: Any, expires_at: Optional[datetime] = None):
        """Set value with optional expiry"""
        with self._lock:
            # Estimate memory usage
            estimated_size = self._estimate_size(value)

            # Remove existing key if present
            if key in self._cache:
                self._remove_key(key)

            # Ensure we have space
            self._make_space(estimated_size)

            # Add new entry
            self._cache[key] = (value, expires_at, estimated_size)
            self._current_memory += estimated_size

    def delete(self, key: str):
        """Delete key from cache"""
        with self._lock:
            self._remove_key(key)

    def _remove_key(self, key: str):
        """Remove key and update memory counter"""
        if key in self._cache:
            _, _, size = self._cache.pop(key)
            self._current_memory -= size

    def _make_space(self, needed_size: int):
        """Make space by evicting LRU items"""
        # Check size limit
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove_key(oldest_key)

        # Check memory limit
        while (self._current_memory + needed_size) > self.max_memory_bytes and self._cache:
            oldest_key = next(iter(self._cache))
            self._remove_key(oldest_key)

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory usage of value"""
        try:
            # Primary estimation method - JSON serialization
            json_str = json.dumps(value, default=str)
            return len(json_str.encode("utf-8"))
        except (TypeError, ValueError, OverflowError) as e:
            # JSON serialization failed, try pickle
            try:
                import pickle

                return len(pickle.dumps(value))
            except Exception as pickle_error:
                logger.warning(
                    f"Size estimation failed for both JSON and pickle: json_error={e}, pickle_error={pickle_error}"
                )
                # Conservative fallback based on object type
                if isinstance(value, str):
                    return len(value.encode("utf-8"))
                elif isinstance(value, (list, tuple)):
                    return len(value) * 100  # Estimate 100 bytes per item
                elif isinstance(value, dict):
                    return len(value) * 200  # Estimate 200 bytes per key-value pair
                else:
                    return 4096  # 4KB conservative fallback

    def cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            now = datetime.now()
            expired_keys = []

            for key, (_, expires_at, _) in self._cache.items():
                if expires_at and now > expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_key(key)

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_bytes": self._current_memory,
                "max_memory_bytes": self.max_memory_bytes,
                "utilization_percent": (len(self._cache) / self.max_size) * 100,
                "memory_utilization_percent": (self._current_memory / self.max_memory_bytes) * 100,
            }


class RedisCache:
    """Production Redis cache with automatic serialization and LRU fallback"""

    def __init__(self, redis_url: str = None, prefix: str = "cryptotrading"):
        self.redis_url = redis_url or CacheConfig.REDIS_URL
        self.prefix = prefix
        self.redis_client = None

        # LRU fallback cache with size limits
        self.fallback_cache = LRUCache(
            max_size=CacheConfig.MAX_LOCAL_CACHE_SIZE,
            max_memory_mb=CacheConfig.MAX_LOCAL_CACHE_MEMORY_MB,
        )

        # Background cleanup
        self._cleanup_timer = None
        self._start_cleanup_timer()

        self._connect()

    def _start_cleanup_timer(self):
        """Start background cleanup timer"""
        import threading

        def cleanup_expired():
            try:
                removed = self.fallback_cache.cleanup_expired()
                if removed > 0:
                    logger.debug(f"Cleaned up {removed} expired cache entries")
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
            finally:
                # Schedule next cleanup
                self._cleanup_timer = threading.Timer(CacheConfig.CLEANUP_INTERVAL, cleanup_expired)
                self._cleanup_timer.start()

        # Start initial cleanup
        self._cleanup_timer = threading.Timer(CacheConfig.CLEANUP_INTERVAL, cleanup_expired)
        self._cleanup_timer.start()

    def _json_serializer(self, obj):
        """JSON serializer for datetime and other common objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            # Convert objects to dictionaries
            return obj.__dict__
        elif hasattr(obj, "to_dict"):
            # Use to_dict method if available
            return obj.to_dict()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _connect(self):
        """Connect to Redis with fallback"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )

            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using local fallback cache.")
            self.redis_client = None

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key"""
        return f"{self.prefix}:{key}"

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage using secure JSON-only approach"""
        try:
            # Use JSON serialization with proper datetime handling
            json_str = json.dumps(data, default=self._json_serializer)
            return json_str.encode("utf-8")
        except (TypeError, ValueError) as e:
            logger.error(f"Cannot serialize object to JSON: {e}. Object type: {type(data)}")
            raise ValueError(
                f"Object not JSON serializable: {type(data)}. Use simpler data structures."
            )

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage using secure JSON-only approach"""
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Cannot deserialize data from cache: {e}")
            raise ValueError("Cached data is corrupted or not in expected JSON format")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._make_key(key)

        if self.redis_client:
            try:
                data = self.redis_client.get(cache_key)
                if data:
                    return self._deserialize(data)
                return None
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                # Fall through to local cache

        # Use local fallback
        cache_entry = self.fallback_cache.get(key)
        if cache_entry:
            value, expires_at = cache_entry
            return value

        return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        cache_key = self._make_key(key)
        ttl = ttl or CacheConfig.DEFAULT_TTL

        if self.redis_client:
            try:
                serialized = self._serialize(value)
                result = self.redis_client.setex(cache_key, ttl, serialized)
                return bool(result)
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                # Fall through to local cache

        # Use local fallback
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        self.fallback_cache.set(key, value, expires_at)
        return True

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        cache_key = self._make_key(key)

        result = True
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                result = False

        # Also remove from local cache
        self.fallback_cache.delete(key)

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "redis_connected": self.redis_client is not None,
            "local_cache": self.fallback_cache.get_stats(),
        }

        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis"] = {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }

                # Calculate hit rate
                hits = stats["redis"]["keyspace_hits"]
                misses = stats["redis"]["keyspace_misses"]
                total = hits + misses
                stats["redis"]["hit_rate_percent"] = (hits / total * 100) if total > 0 else 0

            except Exception as e:
                stats["redis"] = {"error": str(e)}

        return stats

    def close(self):
        """Close cache connections and cleanup"""
        # Stop cleanup timer
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        # Close Redis connection
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")

        logger.info("Cache closed")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        cache_key = self._make_key(key)

        if self.redis_client:
            try:
                return bool(self.redis_client.exists(cache_key))
            except Exception as e:
                logger.warning(f"Redis exists error: {e}")

        # Check local cache
        cache_entry = self.fallback_cache.get(key)
        if cache_entry:
            value, expires_at = cache_entry
            if expires_at is None or datetime.now() < expires_at:
                return True
            else:
                self.fallback_cache.delete(key)

        return False

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        result = {}

        if self.redis_client:
            try:
                cache_keys = [self._make_key(key) for key in keys]
                values = self.redis_client.mget(cache_keys)

                for key, data in zip(keys, values):
                    if data:
                        try:
                            result[key] = self._deserialize(data)
                        except Exception as e:
                            logger.warning(f"Deserialization error for {key}: {e}")

                return result
            except Exception as e:
                logger.warning(f"Redis mget error: {e}")

        # Fallback to individual gets
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value

        return result

    def set_many(self, mapping: Dict[str, Any], ttl: int = None) -> bool:
        """Set multiple values in cache"""
        ttl = ttl or CacheConfig.DEFAULT_TTL

        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                for key, value in mapping.items():
                    cache_key = self._make_key(key)
                    serialized = self._serialize(value)
                    pipe.setex(cache_key, ttl, serialized)

                pipe.execute()
                return True
            except Exception as e:
                logger.warning(f"Redis mset error: {e}")

        # Fallback to individual sets
        for key, value in mapping.items():
            self.set(key, value, ttl)

        return True

    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        cache_pattern = self._make_key(pattern)

        if self.redis_client:
            try:
                keys = self.redis_client.keys(cache_pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            except Exception as e:
                logger.warning(f"Redis clear pattern error: {e}")

        # Clear from local cache
        cleared = 0
        keys_to_delete = []

        # Get keys safely from cache (LRUCache doesn't have .keys() method)
        with self.fallback_cache._lock:
            pattern_key = pattern.replace("*", "")
            for key in list(self.fallback_cache._cache.keys()):
                if pattern_key in key:
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            self.fallback_cache.delete(key)
            cleared += 1

        return cleared

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        cache_key = self._make_key(key)

        if self.redis_client:
            try:
                return self.redis_client.incrby(cache_key, amount)
            except Exception as e:
                logger.warning(f"Redis increment error: {e}")

        # Local fallback
        current = self.get(key) or 0
        new_value = current + amount
        self.set(key, new_value)
        return new_value

    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on existing key"""
        cache_key = self._make_key(key)

        if self.redis_client:
            try:
                return bool(self.redis_client.expire(cache_key, ttl))
            except Exception as e:
                logger.warning(f"Redis expire error: {e}")

        # Update local cache expiration
        cache_entry = self.fallback_cache.get(key)
        if cache_entry:
            value, _ = cache_entry
            expires_at = datetime.now() + timedelta(seconds=ttl)
            self.fallback_cache.set(key, value, expires_at)
            return True

        return False


class CacheManager:
    """High-level cache management with type-specific TTLs"""

    def __init__(self, redis_url: str = None):
        self.cache = RedisCache(redis_url)

    def cache_registry_data(
        self, registry_type: str, registry_id: str, data: Dict[str, Any]
    ) -> bool:
        """Cache registry data with appropriate TTL"""
        key = f"registry:{registry_type}:{registry_id}"
        return self.cache.set(key, data, CacheConfig.REGISTRY_TTL)

    def get_registry_data(self, registry_type: str, registry_id: str) -> Optional[Dict[str, Any]]:
        """Get cached registry data"""
        key = f"registry:{registry_type}:{registry_id}"
        return self.cache.get(key)

    def cache_analysis(self, symbol: str, provider: str, analysis: Dict[str, Any]) -> bool:
        """Cache AI analysis result"""
        key = f"analysis:{symbol}:{provider}"
        return self.cache.set(key, analysis, CacheConfig.ANALYSIS_TTL)

    def get_cached_analysis(self, symbol: str, provider: str) -> Optional[Dict[str, Any]]:
        """Get cached AI analysis"""
        key = f"analysis:{symbol}:{provider}"
        return self.cache.get(key)

    def cache_market_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Cache market data with short TTL"""
        key = f"market_data:{symbol}"
        return self.cache.set(key, data, CacheConfig.MARKET_DATA_TTL)

    def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data"""
        key = f"market_data:{symbol}"
        return self.cache.get(key)

    def cache_user_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Cache user session data"""
        key = f"session:{session_id}"
        return self.cache.set(key, session_data, 3600)  # 1 hour

    def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session"""
        key = f"session:{session_id}"
        return self.cache.get(key)

    def invalidate_symbol_cache(self, symbol: str):
        """Invalidate all cached data for a symbol"""
        patterns = [
            f"analysis:{symbol}:*",
            f"market_data:{symbol}",
            f"technical_indicators:{symbol}:*",
        ]

        for pattern in patterns:
            self.cache.clear_pattern(pattern)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "connected_to_redis": self.cache.redis_client is not None,
            "local_cache_size": len(self.cache.fallback_cache),
            "redis_url": self.cache.redis_url,
        }

        if self.cache.redis_client:
            try:
                info = self.cache.redis_client.info()
                stats.update(
                    {
                        "redis_memory_used": info.get("used_memory_human"),
                        "redis_connected_clients": info.get("connected_clients"),
                        "redis_total_commands": info.get("total_commands_processed"),
                    }
                )
            except Exception as e:
                stats["redis_error"] = str(e)

        return stats


def cache_result(ttl: int = None, key_func: callable = None):
    """Decorator to cache function results"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Create key from function name and args
                args_str = "_".join(str(arg) for arg in args)
                kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key_parts = [func.__name__, args_str, kwargs_str]
                cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cached_result = cache_manager.cache.get(f"func_cache:{cache_key}")
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.cache.set(f"func_cache:{cache_key}", result, ttl)

            return result

        return wrapper

    return decorator


# Global cache manager instance
cache_manager = CacheManager()


# Convenience functions
def cache_set(key: str, value: Any, ttl: int = None) -> bool:
    """Set value in cache"""
    return cache_manager.cache.set(key, value, ttl)


def cache_get(key: str) -> Optional[Any]:
    """Get value from cache"""
    return cache_manager.cache.get(key)


def cache_delete(key: str) -> bool:
    """Delete key from cache"""
    return cache_manager.cache.delete(key)


def cache_clear_pattern(pattern: str) -> int:
    """Clear keys matching pattern"""
    return cache_manager.cache.clear_pattern(pattern)
