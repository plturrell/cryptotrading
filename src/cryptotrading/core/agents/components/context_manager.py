"""
Context Management Component for Strands Framework
Handles agent context, memory, and state management
"""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """Individual context entry"""

    key: str
    value: Any
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at

    def touch(self):
        """Update access time and count"""
        self.updated_at = datetime.utcnow()
        self.access_count += 1


@dataclass
class StrandsContext:
    """Enhanced context management for Strands agents"""

    session_id: str
    agent_id: str
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # Core context data
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_executions: List[Dict[str, Any]] = field(default_factory=list)
    workflow_state: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)

    # Enhanced features
    variables: Dict[str, Any] = field(default_factory=dict)
    temporary_storage: Dict[str, Any] = field(default_factory=dict)
    persistent_storage: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()

    def add_conversation_entry(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add entry to conversation history"""
        entry = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.conversation_history.append(entry)
        self.update_activity()

    def add_tool_execution(
        self, tool_name: str, parameters: Dict[str, Any], result: Any = None, error: str = None
    ):
        """Add tool execution to history"""
        entry = {
            "id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.tool_executions.append(entry)
        self.update_activity()


class ContextStorage:
    """Storage backend for context data"""

    def __init__(self, max_memory_size: int = 1000):
        self.max_memory_size = max_memory_size
        self._storage: Dict[str, ContextEntry] = {}
        self._access_order = deque()  # For LRU eviction
        self._lock = asyncio.Lock()

        # Indexes for fast lookup
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._key_patterns: Dict[str, Set[str]] = defaultdict(set)

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None, tags: Set[str] = None
    ) -> bool:
        """Store a value with optional TTL and tags"""
        async with self._lock:
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl) if ttl else None
            tags = tags or set()

            # Create or update entry
            if key in self._storage:
                entry = self._storage[key]
                entry.value = value
                entry.updated_at = now
                entry.expires_at = expires_at
                entry.tags = tags
                entry.touch()
            else:
                entry = ContextEntry(
                    key=key,
                    value=value,
                    created_at=now,
                    updated_at=now,
                    expires_at=expires_at,
                    tags=tags,
                )
                self._storage[key] = entry
                self._access_order.append(key)

            # Update indexes
            for tag in tags:
                self._tag_index[tag].add(key)

            # Evict if necessary
            await self._evict_if_necessary()

            return True

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key"""
        async with self._lock:
            entry = self._storage.get(key)
            if not entry:
                return None

            if entry.is_expired():
                await self._remove_entry(key)
                return None

            entry.touch()

            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return entry.value

    async def delete(self, key: str) -> bool:
        """Delete a value by key"""
        async with self._lock:
            return await self._remove_entry(key)

    async def get_by_tags(self, tags: Set[str]) -> Dict[str, Any]:
        """Get all values that have any of the specified tags"""
        async with self._lock:
            result = {}

            matching_keys = set()
            for tag in tags:
                matching_keys.update(self._tag_index.get(tag, set()))

            for key in matching_keys:
                entry = self._storage.get(key)
                if entry and not entry.is_expired():
                    result[key] = entry.value
                    entry.touch()

            return result

    async def cleanup_expired(self) -> int:
        """Remove expired entries"""
        async with self._lock:
            expired_keys = []
            for key, entry in self._storage.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                await self._remove_entry(key)

            return len(expired_keys)

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        async with self._lock:
            total_size = len(self._storage)
            expired_count = sum(1 for entry in self._storage.values() if entry.is_expired())

            # Memory usage estimate
            memory_usage = sum(
                len(str(entry.value)) + len(entry.key) + 100  # Rough estimate
                for entry in self._storage.values()
            )

            return {
                "total_entries": total_size,
                "expired_entries": expired_count,
                "memory_usage_bytes": memory_usage,
                "max_memory_size": self.max_memory_size,
                "tags_count": len(self._tag_index),
            }

    async def _remove_entry(self, key: str) -> bool:
        """Remove entry and update indexes"""
        if key not in self._storage:
            return False

        entry = self._storage[key]

        # Remove from storage
        del self._storage[key]

        # Remove from access order
        if key in self._access_order:
            self._access_order.remove(key)

        # Update tag index
        for tag in entry.tags:
            self._tag_index[tag].discard(key)
            if not self._tag_index[tag]:
                del self._tag_index[tag]

        return True

    async def _evict_if_necessary(self):
        """Evict oldest entries if memory limit exceeded"""
        while len(self._storage) > self.max_memory_size and self._access_order:
            oldest_key = self._access_order.popleft()
            if oldest_key in self._storage:
                await self._remove_entry(oldest_key)


class MemoryManager:
    """Manages different types of memory and context persistence"""

    def __init__(self, max_short_term: int = 500, max_long_term: int = 10000):
        self.short_term_storage = ContextStorage(max_short_term)
        self.long_term_storage = ContextStorage(max_long_term)

        # Memory categorization
        self.working_memory: Dict[str, Any] = {}  # Current task context
        self.episodic_memory: List[Dict[str, Any]] = []  # Event sequences
        self.semantic_memory: Dict[str, Any] = {}  # Learned knowledge

        self._cleanup_task = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task"""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Clean up every 5 minutes
                    await self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Memory cleanup error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def store_short_term(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Store in short-term memory with TTL"""
        return await self.short_term_storage.set(key, value, ttl=ttl)

    async def store_long_term(self, key: str, value: Any, tags: Set[str] = None) -> bool:
        """Store in long-term memory without TTL"""
        return await self.long_term_storage.set(key, value, tags=tags)

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve from any memory store (short-term first)"""
        # Try short-term first
        value = await self.short_term_storage.get(key)
        if value is not None:
            return value

        # Try long-term
        return await self.long_term_storage.get(key)

    async def store_working_memory(self, key: str, value: Any):
        """Store in working memory (immediate access)"""
        self.working_memory[key] = {"value": value, "timestamp": datetime.utcnow().isoformat()}

        # Limit working memory size
        if len(self.working_memory) > 100:
            # Remove oldest entries
            sorted_items = sorted(self.working_memory.items(), key=lambda x: x[1]["timestamp"])
            self.working_memory = dict(sorted_items[-50:])

    def get_working_memory(self, key: str) -> Optional[Any]:
        """Get from working memory"""
        entry = self.working_memory.get(key)
        return entry["value"] if entry else None

    async def add_episodic_memory(self, event: Dict[str, Any]):
        """Add event to episodic memory"""
        event["timestamp"] = datetime.utcnow().isoformat()
        event["id"] = str(uuid.uuid4())

        self.episodic_memory.append(event)

        # Limit episodic memory size
        if len(self.episodic_memory) > 1000:
            self.episodic_memory = self.episodic_memory[-500:]

    async def update_semantic_memory(self, concept: str, information: Any):
        """Update semantic memory with learned information"""
        if concept not in self.semantic_memory:
            self.semantic_memory[concept] = {
                "data": information,
                "confidence": 1.0,
                "last_updated": datetime.utcnow().isoformat(),
                "update_count": 1,
            }
        else:
            # Update existing concept
            entry = self.semantic_memory[concept]
            entry["data"] = information
            entry["last_updated"] = datetime.utcnow().isoformat()
            entry["update_count"] += 1
            entry["confidence"] = min(1.0, entry["confidence"] + 0.1)

    def get_semantic_memory(self, concept: str) -> Optional[Any]:
        """Get semantic memory for concept"""
        entry = self.semantic_memory.get(concept)
        return entry["data"] if entry else None

    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired memory entries"""
        short_term_cleaned = await self.short_term_storage.cleanup_expired()
        long_term_cleaned = await self.long_term_storage.cleanup_expired()

        return {"short_term_cleaned": short_term_cleaned, "long_term_cleaned": long_term_cleaned}

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        short_term_stats = await self.short_term_storage.get_stats()
        long_term_stats = await self.long_term_storage.get_stats()

        return {
            "short_term": short_term_stats,
            "long_term": long_term_stats,
            "working_memory": {
                "entries": len(self.working_memory),
                "memory_usage_bytes": sum(len(str(v)) for v in self.working_memory.values()),
            },
            "episodic_memory": {
                "events": len(self.episodic_memory),
                "memory_usage_bytes": sum(len(str(event)) for event in self.episodic_memory),
            },
            "semantic_memory": {
                "concepts": len(self.semantic_memory),
                "memory_usage_bytes": sum(
                    len(str(concept)) for concept in self.semantic_memory.values()
                ),
            },
        }

    async def shutdown(self):
        """Graceful shutdown"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class ContextManager:
    """Main context management system for Strands agents"""

    def __init__(self, agent_id: str, max_contexts: int = 100):
        self.agent_id = agent_id
        self.max_contexts = max_contexts

        # Context storage
        self._contexts: Dict[str, StrandsContext] = {}
        self._context_index: Dict[str, str] = {}  # session_id -> context_id

        # Memory management
        self.memory_manager = MemoryManager()

        # Session tracking
        self._active_sessions: Set[str] = set()
        self._session_metadata: Dict[str, Dict[str, Any]] = {}

        # Context lifecycle
        self._context_history: List[str] = []
        self._lock = asyncio.Lock()

        logger.info(f"ContextManager initialized for agent {agent_id}")

    async def create_context(
        self, session_id: str, metadata: Dict[str, Any] = None
    ) -> StrandsContext:
        """Create a new context for a session"""
        async with self._lock:
            context = StrandsContext(
                session_id=session_id, agent_id=self.agent_id, metadata=metadata or {}
            )

            self._contexts[context.context_id] = context
            self._context_index[session_id] = context.context_id
            self._active_sessions.add(session_id)
            self._session_metadata[session_id] = metadata or {}

            # Manage context limits
            await self._evict_old_contexts()

            logger.info(f"Created context {context.context_id} for session {session_id}")
            return context

    async def get_context(self, session_id: str) -> Optional[StrandsContext]:
        """Get context for a session"""
        context_id = self._context_index.get(session_id)
        if not context_id:
            return None

        context = self._contexts.get(context_id)
        if context:
            context.update_activity()

        return context

    async def get_or_create_context(
        self, session_id: str, metadata: Dict[str, Any] = None
    ) -> StrandsContext:
        """Get existing context or create new one"""
        context = await self.get_context(session_id)
        if not context:
            context = await self.create_context(session_id, metadata)
        return context

    async def update_context(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update context with new data"""
        context = await self.get_context(session_id)
        if not context:
            return False

        # Update fields
        for key, value in updates.items():
            if hasattr(context, key):
                setattr(context, key, value)

        context.update_activity()
        return True

    async def delete_context(self, session_id: str) -> bool:
        """Delete a context"""
        async with self._lock:
            context_id = self._context_index.get(session_id)
            if not context_id:
                return False

            # Remove from storage
            if context_id in self._contexts:
                del self._contexts[context_id]

            if session_id in self._context_index:
                del self._context_index[session_id]

            self._active_sessions.discard(session_id)

            if session_id in self._session_metadata:
                del self._session_metadata[session_id]

            logger.info(f"Deleted context for session {session_id}")
            return True

    async def store_memory(
        self, session_id: str, key: str, value: Any, memory_type: str = "short_term"
    ) -> bool:
        """Store data in agent memory"""
        full_key = f"{session_id}:{key}"

        if memory_type == "short_term":
            return await self.memory_manager.store_short_term(full_key, value)
        elif memory_type == "long_term":
            return await self.memory_manager.store_long_term(full_key, value)
        elif memory_type == "working":
            await self.memory_manager.store_working_memory(full_key, value)
            return True
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    async def retrieve_memory(self, session_id: str, key: str) -> Optional[Any]:
        """Retrieve data from agent memory"""
        full_key = f"{session_id}:{key}"

        # Try working memory first
        value = self.memory_manager.get_working_memory(full_key)
        if value is not None:
            return value

        # Try persistent memory
        return await self.memory_manager.retrieve(full_key)

    async def add_conversation_entry(
        self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None
    ):
        """Add entry to conversation history"""
        context = await self.get_context(session_id)
        if context:
            context.add_conversation_entry(role, content, metadata)

            # Also store in episodic memory
            await self.memory_manager.add_episodic_memory(
                {
                    "type": "conversation",
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "metadata": metadata,
                }
            )

    async def add_tool_execution(
        self,
        session_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any = None,
        error: str = None,
    ):
        """Add tool execution to context"""
        context = await self.get_context(session_id)
        if context:
            context.add_tool_execution(tool_name, parameters, result, error)

            # Store in episodic memory
            await self.memory_manager.add_episodic_memory(
                {
                    "type": "tool_execution",
                    "session_id": session_id,
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "result": result,
                    "error": error,
                }
            )

    async def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self._active_sessions)

    async def get_context_stats(self) -> Dict[str, Any]:
        """Get context management statistics"""
        memory_stats = await self.memory_manager.get_memory_stats()

        return {
            "agent_id": self.agent_id,
            "active_contexts": len(self._contexts),
            "active_sessions": len(self._active_sessions),
            "max_contexts": self.max_contexts,
            "memory_stats": memory_stats,
            "context_history_size": len(self._context_history),
        }

    async def _evict_old_contexts(self):
        """Evict oldest contexts if limit exceeded"""
        if len(self._contexts) <= self.max_contexts:
            return

        # Sort contexts by last activity
        sorted_contexts = sorted(self._contexts.items(), key=lambda x: x[1].last_activity)

        # Remove oldest contexts
        contexts_to_remove = sorted_contexts[: len(self._contexts) - self.max_contexts]

        for context_id, context in contexts_to_remove:
            session_id = context.session_id

            # Archive important data before deletion
            await self._archive_context(context)

            # Remove from storage
            del self._contexts[context_id]
            if session_id in self._context_index:
                del self._context_index[session_id]
            self._active_sessions.discard(session_id)

            logger.info(f"Evicted old context {context_id} for session {session_id}")

    async def _archive_context(self, context: StrandsContext):
        """Archive important context data before eviction"""
        # Store conversation history in long-term memory
        if context.conversation_history:
            await self.memory_manager.store_long_term(
                f"archived_conversation_{context.session_id}",
                context.conversation_history,
                tags={"archived", "conversation", context.session_id},
            )

        # Store important variables
        if context.variables:
            await self.memory_manager.store_long_term(
                f"archived_variables_{context.session_id}",
                context.variables,
                tags={"archived", "variables", context.session_id},
            )

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down ContextManager")

        # Archive all active contexts
        for context in self._contexts.values():
            await self._archive_context(context)

        # Shutdown memory manager
        await self.memory_manager.shutdown()

        logger.info("ContextManager shutdown complete")
