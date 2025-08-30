"""
Persistent Memory System with Database Storage
Replaces RAM-only memory with database-backed persistent storage
Enhanced with Tree library for nested memory structure processing
"""
import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ...data.database.client import get_db
from ...infrastructure.database.unified_database import UnifiedDatabase

# Tree library integration for nested memory processing
try:
    from ...infrastructure.analysis.tree_library import (
        NestedStructure,
        PathOperations,
        StructuralAnalysis,
        TreeDiffMerge,
        TreeOperations,
        TreePath,
    )

    TREE_AVAILABLE = True
except ImportError:
    TREE_AVAILABLE = False

    class TreeOperations:
        @staticmethod
        def flatten(structure):
            return []

        @staticmethod
        def map_structure(fn, structure):
            return structure

    class PathOperations:
        @staticmethod
        def get_path(structure, path):
            return None

        @staticmethod
        def set_path(structure, path, value):
            return structure

    class StructuralAnalysis:
        @staticmethod
        def get_depth(structure):
            return 0

        @staticmethod
        def find_substructures(structure, predicate):
            return []


logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Memory entry with metadata"""

    key: str
    value: Any
    agent_id: str
    memory_type: str = "general"
    importance: float = 0.5
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    expires_at: Optional[datetime] = None


class PersistentMemorySystem:
    """
    Database-backed persistent memory system for agents
    Replaces ephemeral RAM storage with permanent knowledge
    """

    def __init__(self, agent_id: str, db: Optional[UnifiedDatabase] = None):
        self.agent_id = agent_id
        if db:
            self.db = db
        else:
            # Use unified database by default
            self.db = UnifiedDatabase()
        self._cache = {}  # Local cache for performance
        self._cache_size = 1000
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize memory system and load recent memories"""
        try:
            # Load most important recent memories into cache
            await self._load_recent_memories()
            logger.info(f"Persistent memory initialized for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")

    async def store(
        self,
        key: str,
        value: Any,
        memory_type: str = "general",
        importance: float = 0.5,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None,
    ) -> bool:
        """
        Store memory in database with metadata

        Args:
            key: Unique key for memory
            value: Value to store (will be JSON serialized)
            memory_type: Type of memory (general, insight, decision, etc)
            importance: 0-1 importance score for retrieval
            context: Context information
            metadata: Additional metadata
            ttl_hours: Time to live in hours
        """
        async with self._lock:
            try:
                # Serialize value
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)

                # Calculate expiry
                expires_at = None
                if ttl_hours:
                    expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

                # Prepare metadata
                meta_str = json.dumps(metadata) if metadata else None

                # Store in database
                # Ensure database is initialized
                if not hasattr(self.db, "db_conn") or self.db.db_conn is None:
                    await self.db.initialize()

                cursor = self.db.db_conn.cursor()
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO agent_memory 
                        (agent_id, memory_key, memory_value, memory_type, importance, 
                         context, metadata, created_at, accessed_at, access_count, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                        (
                            self.agent_id,
                            key,
                            value_str,
                            memory_type,
                            importance,
                            context,
                            meta_str,
                            datetime.utcnow(),
                            datetime.utcnow(),
                            expires_at,
                        ),
                    )
                    self.db.db_conn.commit()
                finally:
                    cursor.close()

                # Update cache
                self._cache[key] = Memory(
                    key=key,
                    value=value,
                    agent_id=self.agent_id,
                    memory_type=memory_type,
                    importance=importance,
                    context=context,
                    metadata=metadata,
                    created_at=datetime.utcnow(),
                    accessed_at=datetime.utcnow(),
                    access_count=1,
                    expires_at=expires_at,
                )

                # Manage cache size
                if len(self._cache) > self._cache_size:
                    await self._evict_cache()

                logger.debug(f"Stored memory: {key} (importance: {importance})")
                return True

            except Exception as e:
                logger.error(f"Failed to store memory {key}: {e}")
                return False

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve memory by key"""
        async with self._lock:
            # Check cache first
            if key in self._cache:
                memory = self._cache[key]
                memory.accessed_at = datetime.utcnow()
                memory.access_count += 1
                return memory.value

            # Load from database
            cursor = None
            try:
                # Ensure database is initialized
                if not hasattr(self.db, "db_conn") or self.db.db_conn is None:
                    await self.db.initialize()

                cursor = self.db.db_conn.cursor()

                if self.db.config.mode.value == "local":
                    cursor.execute(
                        """
                        SELECT memory_value, memory_type, importance, context, 
                               metadata, created_at, access_count, expires_at
                        FROM agent_memory 
                        WHERE agent_id = ? AND memory_key = ?
                        AND (expires_at IS NULL OR expires_at > ?)
                    """,
                        (self.agent_id, key, datetime.utcnow()),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT memory_value, memory_type, importance, context,
                               metadata, created_at, access_count, expires_at
                        FROM agent_memory 
                        WHERE agent_id = %s AND memory_key = %s
                        AND (expires_at IS NULL OR expires_at > %s)
                    """,
                        (self.agent_id, key, datetime.utcnow()),
                    )

                row = cursor.fetchone()

                if row:
                    # Update access info
                    if self.db.config.mode.value == "local":
                        cursor.execute(
                            """
                            UPDATE agent_memory 
                            SET accessed_at = ?, access_count = access_count + 1
                            WHERE agent_id = ? AND memory_key = ?
                        """,
                            (datetime.utcnow(), self.agent_id, key),
                        )
                    else:
                        cursor.execute(
                            """
                            UPDATE agent_memory 
                            SET accessed_at = %s, access_count = access_count + 1
                            WHERE agent_id = %s AND memory_key = %s
                        """,
                            (datetime.utcnow(), self.agent_id, key),
                        )

                    self.db.db_conn.commit()

                    # Parse value
                    try:
                        value = json.loads(row[0])
                    except (json.JSONDecodeError, TypeError):
                        value = row[0]

                    # Cache it
                    self._cache[key] = Memory(
                        key=key,
                        value=value,
                        agent_id=self.agent_id,
                        memory_type=row[1],
                        importance=row[2],
                        context=row[3],
                        metadata=json.loads(row[4]) if row[4] else None,
                        created_at=row[5],
                        accessed_at=datetime.utcnow(),
                        access_count=row[6] + 1,
                        expires_at=row[7],
                    )

                    return value

                return None

            except Exception as e:
                logger.error(f"Failed to retrieve memory {key}: {e}")
                return None
            finally:
                if cursor:
                    cursor.close()

    async def search(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Search memories by content, type, and importance
        """
        async with self._lock:
            try:
                cursor = self.db.db_conn.cursor()

                # Build query
                conditions = ["agent_id = ?", "(expires_at IS NULL OR expires_at > ?)"]
                params = [self.agent_id, datetime.utcnow()]

                if memory_types:
                    placeholders = ",".join(["?" for _ in memory_types])
                    conditions.append(f"memory_type IN ({placeholders})")
                    params.extend(memory_types)

                conditions.append("importance >= ?")
                params.append(min_importance)

                # Search in value and context
                conditions.append("(memory_value LIKE ? OR context LIKE ?)")
                search_pattern = f"%{query}%"
                params.extend([search_pattern, search_pattern])

                if self.db.config.mode.value == "local":
                    sql = f"""
                        SELECT memory_key, memory_value, memory_type, importance,
                               context, metadata, created_at, accessed_at, 
                               access_count, expires_at
                        FROM agent_memory
                        WHERE {' AND '.join(conditions)}
                        ORDER BY importance DESC, accessed_at DESC
                        LIMIT ?
                    """
                    params.append(limit)
                    cursor.execute(sql, params)
                else:
                    # PostgreSQL version with proper parameter substitution
                    conditions = ["agent_id = %s", "(expires_at IS NULL OR expires_at > %s)"]
                    params = [self.agent_id, datetime.utcnow()]

                    if memory_types:
                        placeholders = ",".join(["%s" for _ in memory_types])
                        conditions.append(f"memory_type IN ({placeholders})")
                        params.extend(memory_types)

                    conditions.append("importance >= %s")
                    params.append(min_importance)
                    conditions.append("(memory_value LIKE %s OR context LIKE %s)")
                    params.extend([search_pattern, search_pattern])

                    sql = f"""
                        SELECT memory_key, memory_value, memory_type, importance,
                               context, metadata, created_at, accessed_at,
                               access_count, expires_at
                        FROM agent_memory
                        WHERE {' AND '.join(conditions)}
                        ORDER BY importance DESC, accessed_at DESC
                        LIMIT %s
                    """
                    params.append(limit)
                    cursor.execute(sql, params)

                memories = []
                for row in cursor.fetchall():
                    try:
                        value = json.loads(row[1])
                    except (json.JSONDecodeError, TypeError):
                        value = row[1]

                    memory = Memory(
                        key=row[0],
                        value=value,
                        agent_id=self.agent_id,
                        memory_type=row[2],
                        importance=row[3],
                        context=row[4],
                        metadata=json.loads(row[5]) if row[5] else None,
                        created_at=row[6],
                        accessed_at=row[7],
                        access_count=row[8],
                        expires_at=row[9],
                    )
                    memories.append(memory)

                return memories

            except Exception as e:
                logger.error(f"Failed to search memories: {e}")
                return []
            finally:
                cursor.close()

    async def get_related_memories(self, key: str, limit: int = 5) -> List[Memory]:
        """Get memories related to a specific key using knowledge graph"""
        # This would query the knowledge_graph table for relationships
        # For now, return similar memories based on context
        memory = await self.retrieve(key)
        if not memory or not hasattr(memory, "context"):
            return []

        return await self.search(memory.context or key, limit=limit)

    async def forget(self, key: str) -> bool:
        """Remove a specific memory"""
        async with self._lock:
            try:
                cursor = self.db.db_conn.cursor()

                if self.db.config.mode.value == "local":
                    cursor.execute(
                        """
                        DELETE FROM agent_memory 
                        WHERE agent_id = ? AND memory_key = ?
                    """,
                        (self.agent_id, key),
                    )
                else:
                    cursor.execute(
                        """
                        DELETE FROM agent_memory 
                        WHERE agent_id = %s AND memory_key = %s
                    """,
                        (self.agent_id, key),
                    )

                self.db.db_conn.commit()

                # Remove from cache
                if key in self._cache:
                    del self._cache[key]

                return True

            except Exception as e:
                logger.error(f"Failed to forget memory {key}: {e}")
                self.db.db_conn.rollback()
                return False
            finally:
                cursor.close()

    async def consolidate_memories(self, older_than_days: int = 30):
        """
        Consolidate old memories into summary memories
        Helps manage memory size while preserving important information
        """
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

        # Get old memories grouped by type
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT memory_type, COUNT(*), AVG(importance)
                    FROM agent_memory
                    WHERE agent_id = ? AND created_at < ?
                    GROUP BY memory_type
                """,
                    (self.agent_id, cutoff_date),
                )
            else:
                cursor.execute(
                    """
                    SELECT memory_type, COUNT(*), AVG(importance)
                    FROM agent_memory
                    WHERE agent_id = %s AND created_at < %s
                    GROUP BY memory_type
                """,
                    (self.agent_id, cutoff_date),
                )

            for memory_type, count, avg_importance in cursor.fetchall():
                if count > 10:  # Only consolidate if many memories
                    # Create summary
                    summary_key = f"summary_{memory_type}_{cutoff_date.strftime('%Y%m')}"
                    summary_value = {
                        "type": "consolidated_summary",
                        "period": f"before_{cutoff_date.strftime('%Y-%m-%d')}",
                        "memory_count": count,
                        "avg_importance": avg_importance,
                        "consolidated_at": datetime.utcnow().isoformat(),
                    }

                    await self.store(
                        summary_key,
                        summary_value,
                        memory_type="summary",
                        importance=avg_importance,
                        context=f"Consolidated {count} {memory_type} memories",
                    )

                    logger.info(f"Consolidated {count} {memory_type} memories")

        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
        finally:
            cursor.close()

    async def _load_recent_memories(self):
        """Load most important recent memories into cache"""
        try:
            # Ensure database is initialized
            if not hasattr(self.db, "db_conn") or self.db.db_conn is None:
                await self.db.initialize()

            cursor = self.db.db_conn.cursor()

            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT memory_key, memory_value, memory_type, importance,
                           context, metadata, created_at, accessed_at,
                           access_count, expires_at
                    FROM agent_memory
                    WHERE agent_id = ? 
                    AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY importance DESC, accessed_at DESC
                    LIMIT ?
                """,
                    (self.agent_id, datetime.utcnow(), 100),
                )
            else:
                cursor.execute(
                    """
                    SELECT memory_key, memory_value, memory_type, importance,
                           context, metadata, created_at, accessed_at,
                           access_count, expires_at
                    FROM agent_memory
                    WHERE agent_id = %s 
                    AND (expires_at IS NULL OR expires_at > %s)
                    ORDER BY importance DESC, accessed_at DESC
                    LIMIT %s
                """,
                    (self.agent_id, datetime.utcnow(), 100),
                )

            for row in cursor.fetchall():
                try:
                    value = json.loads(row[1])
                except (json.JSONDecodeError, TypeError):
                    value = row[1]

                self._cache[row[0]] = Memory(
                    key=row[0],
                    value=value,
                    agent_id=self.agent_id,
                    memory_type=row[2],
                    importance=row[3],
                    context=row[4],
                    metadata=json.loads(row[5]) if row[5] else None,
                    created_at=row[6],
                    accessed_at=row[7],
                    access_count=row[8],
                    expires_at=row[9],
                )

            logger.info(f"Loaded {len(self._cache)} memories into cache")

        except Exception as e:
            logger.error(f"Failed to load recent memories: {e}")
        finally:
            cursor.close()

    async def _evict_cache(self):
        """Evict least important, least accessed memories from cache"""
        # Sort by importance and access count
        sorted_memories = sorted(
            self._cache.items(),
            key=lambda x: (x[1].importance * x[1].access_count, x[1].accessed_at),
        )

        # Remove bottom 20%
        evict_count = len(self._cache) // 5
        for key, _ in sorted_memories[:evict_count]:
            del self._cache[key]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT memory_type) as types,
                        AVG(importance) as avg_importance,
                        MAX(accessed_at) as last_access,
                        SUM(access_count) as total_accesses
                    FROM agent_memory
                    WHERE agent_id = ?
                """,
                    (self.agent_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT memory_type) as types,
                        AVG(importance) as avg_importance,
                        MAX(accessed_at) as last_access,
                        SUM(access_count) as total_accesses
                    FROM agent_memory
                    WHERE agent_id = %s
                """,
                    (self.agent_id,),
                )

            row = cursor.fetchone()

            return {
                "total_memories": row[0] or 0,
                "memory_types": row[1] or 0,
                "avg_importance": row[2] or 0,
                "last_access": row[3],
                "total_accesses": row[4] or 0,
                "cache_size": len(self._cache),
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}
        finally:
            cursor.close()


# Convenience functions
async def create_memory_system(agent_id: str) -> PersistentMemorySystem:
    """Create and initialize a persistent memory system"""
    memory = PersistentMemorySystem(agent_id)
    await memory.initialize()
    return memory
