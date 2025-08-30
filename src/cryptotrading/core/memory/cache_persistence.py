"""
Cache Entries Database Persistence
Provides persistent storage for cache entries with TTL support
"""

import asyncio
import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)


class CachePersistence:
    """
    Persists cache entries to database for durability
    Supports TTL, batch operations, and cache warming
    """

    def __init__(self, db: Optional[UnifiedDatabase] = None):
        self.db = db or UnifiedDatabase()
        self._cleanup_interval = 3600  # 1 hour
        self._cleanup_task = None

    async def start(self):
        """Start the cache persistence service"""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Cache persistence service started")

    async def stop(self):
        """Stop the service"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Cache persistence service stopped")

    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate unique cache key"""
        combined = f"{namespace}:{key}"
        return hashlib.md5(combined.encode()).hexdigest()

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store value in persistent cache"""
        try:
            cache_key = self._generate_key(namespace, key)

            # Serialize value
            if isinstance(value, (dict, list, str, int, float, bool)):
                serialized = json.dumps(value)
                serialization_type = "json"
            else:
                serialized = pickle.dumps(value)
                serialization_type = "pickle"

            expires_at = None
            if ttl_seconds:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (cache_key, namespace, key_name, value, serialization_type,
                     metadata, expires_at, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        cache_key,
                        namespace,
                        key,
                        serialized,
                        serialization_type,
                        json.dumps(metadata) if metadata else None,
                        expires_at,
                        datetime.utcnow(),
                        datetime.utcnow(),
                    ),
                )

                conn.commit()

            logger.debug(f"Cached {namespace}:{key} with TTL {ttl_seconds}s")
            return True

        except Exception as e:
            logger.error(f"Failed to set cache entry: {e}")
            return False

    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from persistent cache"""
        try:
            cache_key = self._generate_key(namespace, key)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Check if not expired
                cursor.execute(
                    """
                    SELECT value, serialization_type FROM cache_entries
                    WHERE cache_key = ?
                    AND (expires_at IS NULL OR expires_at > ?)
                """,
                    (cache_key, datetime.utcnow()),
                )

                row = cursor.fetchone()

                if row:
                    # Update last accessed time
                    cursor.execute(
                        """
                        UPDATE cache_entries
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE cache_key = ?
                    """,
                        (datetime.utcnow(), cache_key),
                    )
                    conn.commit()

                    # Deserialize value
                    value, serialization_type = row
                    if serialization_type == "json":
                        return json.loads(value)
                    else:
                        return pickle.loads(value)

            return None

        except Exception as e:
            logger.error(f"Failed to get cache entry: {e}")
            return None

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete entry from cache"""
        try:
            cache_key = self._generate_key(namespace, key)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM cache_entries WHERE cache_key = ?
                """,
                    (cache_key,),
                )

                conn.commit()

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to delete cache entry: {e}")
            return False

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM cache_entries WHERE namespace = ?
                """,
                    (namespace,),
                )

                deleted = cursor.rowcount
                conn.commit()

                logger.info(f"Cleared {deleted} entries from namespace {namespace}")
                return deleted

        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")
            return 0

    async def batch_set(self, entries: List[Dict[str, Any]]) -> int:
        """Set multiple cache entries efficiently"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                data = []
                for entry in entries:
                    cache_key = self._generate_key(entry["namespace"], entry["key"])

                    # Serialize value
                    value = entry["value"]
                    if isinstance(value, (dict, list, str, int, float, bool)):
                        serialized = json.dumps(value)
                        serialization_type = "json"
                    else:
                        serialized = pickle.dumps(value)
                        serialization_type = "pickle"

                    expires_at = None
                    if entry.get("ttl_seconds"):
                        expires_at = datetime.utcnow() + timedelta(seconds=entry["ttl_seconds"])

                    data.append(
                        (
                            cache_key,
                            entry["namespace"],
                            entry["key"],
                            serialized,
                            serialization_type,
                            json.dumps(entry.get("metadata")) if entry.get("metadata") else None,
                            expires_at,
                            datetime.utcnow(),
                            datetime.utcnow(),
                        )
                    )

                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (cache_key, namespace, key_name, value, serialization_type,
                     metadata, expires_at, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    data,
                )

                conn.commit()

                logger.info(f"Batch set {len(data)} cache entries")
                return len(data)

        except Exception as e:
            logger.error(f"Failed to batch set cache entries: {e}")
            return 0

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                base_query = "FROM cache_entries"
                params = []

                if namespace:
                    base_query += " WHERE namespace = ?"
                    params.append(namespace)

                # Total entries
                cursor.execute(f"SELECT COUNT(*) {base_query}", params)
                total_entries = cursor.fetchone()[0]

                # Active entries
                active_query = base_query
                if namespace:
                    active_query += " AND (expires_at IS NULL OR expires_at > ?)"
                else:
                    active_query += " WHERE (expires_at IS NULL OR expires_at > ?)"
                cursor.execute(f"SELECT COUNT(*) {active_query}", params + [datetime.utcnow()])
                active_entries = cursor.fetchone()[0]

                # Most accessed
                access_query = base_query
                if namespace:
                    access_query += " AND access_count > 0"
                else:
                    access_query += " WHERE access_count > 0"
                cursor.execute(
                    f"""
                    SELECT namespace, key_name, access_count 
                    {access_query}
                    ORDER BY access_count DESC
                    LIMIT 10
                """,
                    params,
                )

                most_accessed = []
                for row in cursor.fetchall():
                    most_accessed.append(
                        {"namespace": row[0], "key": row[1], "access_count": row[2]}
                    )

                # Cache size
                cursor.execute(
                    f"""
                    SELECT SUM(LENGTH(value)) {base_query}
                """,
                    params,
                )
                cache_size = cursor.fetchone()[0] or 0

                return {
                    "namespace": namespace,
                    "total_entries": total_entries,
                    "active_entries": active_entries,
                    "expired_entries": total_entries - active_entries,
                    "most_accessed": most_accessed,
                    "cache_size_bytes": cache_size,
                    "cache_size_mb": round(cache_size / 1024 / 1024, 2),
                }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    async def warm_cache(self, namespace: str, min_access_count: int = 5) -> int:
        """Pre-load frequently accessed entries into memory"""
        try:
            # This would integrate with the in-memory cache
            # For now, just return count of entries that would be warmed
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT COUNT(*) FROM cache_entries
                    WHERE namespace = ?
                    AND access_count >= ?
                    AND (expires_at IS NULL OR expires_at > ?)
                """,
                    (namespace, min_access_count, datetime.utcnow()),
                )

                count = cursor.fetchone()[0]

                logger.info(f"Would warm {count} entries for namespace {namespace}")
                return count

        except Exception as e:
            logger.error(f"Failed to warm cache: {e}")
            return 0

    async def _periodic_cleanup(self):
        """Periodically clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)

                with self.db.get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        DELETE FROM cache_entries
                        WHERE expires_at IS NOT NULL AND expires_at < ?
                    """,
                        (datetime.utcnow(),),
                    )

                    deleted = cursor.rowcount
                    conn.commit()

                    if deleted > 0:
                        logger.info(f"Cleaned up {deleted} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")


# Integration with existing cache systems
class PersistentCacheWrapper:
    """Wraps in-memory cache with persistent fallback"""

    def __init__(self, memory_cache, persistence: Optional[CachePersistence] = None):
        self.memory_cache = memory_cache
        self.persistence = persistence or CachePersistence()

    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get from memory cache first, then persistent cache"""
        # Try memory cache
        value = await self.memory_cache.get(namespace, key)
        if value is not None:
            return value

        # Try persistent cache
        value = await self.persistence.get(namespace, key)
        if value is not None:
            # Populate memory cache
            await self.memory_cache.set(namespace, key, value)

        return value

    async def set(
        self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set in both memory and persistent cache"""
        # Set in memory cache
        await self.memory_cache.set(namespace, key, value, ttl_seconds)

        # Set in persistent cache
        return await self.persistence.set(namespace, key, value, ttl_seconds)


# Global cache persistence instance
_cache_persistence: Optional[CachePersistence] = None


async def get_cache_persistence() -> CachePersistence:
    """Get global cache persistence instance"""
    global _cache_persistence
    if _cache_persistence is None:
        _cache_persistence = CachePersistence()
        await _cache_persistence.start()
    return _cache_persistence
