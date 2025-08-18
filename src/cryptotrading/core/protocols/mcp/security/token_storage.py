"""
Persistent Token Revocation Storage for MCP Security

This module provides persistent storage for revoked tokens using Redis or database backend.
Ensures revoked tokens remain invalid across server restarts and in distributed deployments.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union
import os

logger = logging.getLogger(__name__)


class TokenStorage(ABC):
    """Abstract base class for token storage backends"""
    
    @abstractmethod
    async def revoke_token(self, token_hash: str, expires_at: Optional[datetime] = None) -> bool:
        """Revoke a token by storing its hash"""
        pass
    
    @abstractmethod
    async def is_token_revoked(self, token_hash: str) -> bool:
        """Check if a token is revoked"""
        pass
    
    @abstractmethod
    async def cleanup_expired_tokens(self) -> int:
        """Remove expired revoked tokens, return count removed"""
        pass
    
    @abstractmethod
    async def get_revoked_count(self) -> int:
        """Get total number of revoked tokens"""
        pass


class RedisTokenStorage(TokenStorage):
    """Redis-based token revocation storage"""
    
    def __init__(self, redis_url: Optional[str] = None, key_prefix: str = "mcp:revoked:"):
        """
        Initialize Redis token storage
        
        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            key_prefix: Prefix for Redis keys
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.key_prefix = key_prefix
        self.redis_client = None
        self._connection_pool = None
    
    async def _get_redis_client(self):
        """Get Redis client with connection pooling"""
        if self.redis_client is None:
            try:
                import redis.asyncio as redis
                
                # Create connection pool
                self._connection_pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=10,
                    retry_on_timeout=True,
                    decode_responses=True
                )
                
                self.redis_client = redis.Redis(connection_pool=self._connection_pool)
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Connected to Redis for token storage")
                
            except ImportError:
                logger.error("redis package not installed. Install with: pip install redis")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self.redis_client
    
    def _get_token_key(self, token_hash: str) -> str:
        """Get Redis key for token hash"""
        return f"{self.key_prefix}{token_hash}"
    
    async def revoke_token(self, token_hash: str, expires_at: Optional[datetime] = None) -> bool:
        """
        Revoke a token by storing its hash in Redis with TTL
        
        Args:
            token_hash: SHA-256 hash of the token
            expires_at: Token expiration time (for TTL calculation)
            
        Returns:
            True if successfully revoked
        """
        try:
            redis_client = await self._get_redis_client()
            key = self._get_token_key(token_hash)
            
            # Calculate TTL based on token expiration
            ttl_seconds = None
            if expires_at:
                ttl_seconds = max(int((expires_at - datetime.utcnow()).total_seconds()), 60)
            else:
                # Default TTL of 24 hours if no expiration
                ttl_seconds = 86400
            
            # Store revocation with metadata
            revocation_data = {
                "revoked_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "token_hash": token_hash
            }
            
            await redis_client.setex(
                key,
                ttl_seconds,
                json.dumps(revocation_data)
            )
            
            logger.info(f"Token revoked: {token_hash[:8]}... (TTL: {ttl_seconds}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token in Redis: {e}")
            return False
    
    async def is_token_revoked(self, token_hash: str) -> bool:
        """Check if token is revoked"""
        try:
            redis_client = await self._get_redis_client()
            key = self._get_token_key(token_hash)
            
            result = await redis_client.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to check token revocation in Redis: {e}")
            # Fail secure - treat as revoked if we can't check
            return True
    
    async def cleanup_expired_tokens(self) -> int:
        """
        Cleanup expired tokens (Redis handles this automatically with TTL)
        Returns estimated count of revoked tokens
        """
        try:
            redis_client = await self._get_redis_client()
            pattern = f"{self.key_prefix}*"
            
            # Count current revoked tokens
            keys = await redis_client.keys(pattern)
            return len(keys)
            
        except Exception as e:
            logger.error(f"Failed to cleanup tokens in Redis: {e}")
            return 0
    
    async def get_revoked_count(self) -> int:
        """Get total number of currently revoked tokens"""
        try:
            redis_client = await self._get_redis_client()
            pattern = f"{self.key_prefix}*"
            
            keys = await redis_client.keys(pattern)
            return len(keys)
            
        except Exception as e:
            logger.error(f"Failed to get revoked token count from Redis: {e}")
            return 0
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()


class DatabaseTokenStorage(TokenStorage):
    """Database-based token revocation storage using UnifiedDatabase"""
    
    def __init__(self, db=None):
        """
        Initialize database token storage
        
        Args:
            db: UnifiedDatabase instance (optional, creates new if None)
        """
        from ....infrastructure.database.unified_database import UnifiedDatabase
        self.db = db or UnifiedDatabase()
        self._initialized = False
    
    async def _initialize_tables(self):
        """Create revoked tokens table if it doesn't exist"""
        if self._initialized:
            return
        
        try:
            # Create table for revoked tokens using UnifiedDatabase
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS revoked_tokens (
                        token_hash VARCHAR(64) PRIMARY KEY,
                        revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires 
                    ON revoked_tokens(expires_at)
                """)
                
                conn.commit()
            
            self._initialized = True
            logger.info("Initialized revoked tokens table")
            
        except Exception as e:
            logger.error(f"Failed to initialize token storage tables: {e}")
            raise
    
    async def revoke_token(self, token_hash: str, expires_at: Optional[datetime] = None) -> bool:
        """Revoke token by inserting into database"""
        await self._initialize_tables()
        
        try:
            metadata = {
                "revoked_by": "mcp_security_system",
                "reason": "manual_revocation"
            }
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO revoked_tokens 
                    (token_hash, revoked_at, expires_at, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    token_hash,
                    datetime.utcnow(),
                    expires_at,
                    json.dumps(metadata)
                ))
                conn.commit()
            
            logger.info(f"Token revoked in database: {token_hash[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token in database: {e}")
            return False
    
    async def is_token_revoked(self, token_hash: str) -> bool:
        """Check if token is revoked"""
        await self._initialize_tables()
        
        try:
            query_sql = """
            SELECT 1 FROM revoked_tokens 
            WHERE token_hash = ? 
            AND (expires_at IS NULL OR expires_at > ?)
            """
            
            result = await self.db_connection.fetch_one(
                query_sql,
                (token_hash, datetime.utcnow())
            )
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Failed to check token revocation in database: {e}")
            # Fail secure
            return True
    
    async def cleanup_expired_tokens(self) -> int:
        """Remove expired revoked tokens"""
        await self._initialize_tables()
        
        try:
            delete_sql = """
            DELETE FROM revoked_tokens 
            WHERE expires_at IS NOT NULL 
            AND expires_at <= ?
            """
            
            result = await self.db_connection.execute(
                delete_sql,
                (datetime.utcnow(),)
            )
            
            deleted_count = result.rowcount if hasattr(result, 'rowcount') else 0
            logger.info(f"Cleaned up {deleted_count} expired revoked tokens")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0
    
    async def get_revoked_count(self) -> int:
        """Get total number of currently revoked tokens"""
        await self._initialize_tables()
        
        try:
            query_sql = """
            SELECT COUNT(*) FROM revoked_tokens 
            WHERE expires_at IS NULL OR expires_at > ?
            """
            
            result = await self.db_connection.fetch_one(
                query_sql,
                (datetime.utcnow(),)
            )
            
            return result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Failed to get revoked token count: {e}")
            return 0


class InMemoryTokenStorage(TokenStorage):
    """In-memory token storage (fallback for development)"""
    
    def __init__(self):
        """Initialize in-memory storage"""
        self._revoked_tokens: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
        logger.warning("Using in-memory token storage - not suitable for production!")
    
    async def revoke_token(self, token_hash: str, expires_at: Optional[datetime] = None) -> bool:
        """Revoke token in memory"""
        async with self._lock:
            self._revoked_tokens[token_hash] = {
                "revoked_at": datetime.utcnow(),
                "expires_at": expires_at
            }
            logger.info(f"Token revoked in memory: {token_hash[:8]}...")
            return True
    
    async def is_token_revoked(self, token_hash: str) -> bool:
        """Check if token is revoked"""
        async with self._lock:
            if token_hash not in self._revoked_tokens:
                return False
            
            revocation = self._revoked_tokens[token_hash]
            expires_at = revocation.get("expires_at")
            
            if expires_at and datetime.utcnow() > expires_at:
                # Token expiration passed, remove from memory
                del self._revoked_tokens[token_hash]
                return False
            
            return True
    
    async def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from memory"""
        async with self._lock:
            now = datetime.utcnow()
            expired_tokens = [
                token_hash for token_hash, data in self._revoked_tokens.items()
                if data.get("expires_at") and now > data["expires_at"]
            ]
            
            for token_hash in expired_tokens:
                del self._revoked_tokens[token_hash]
            
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens from memory")
            return len(expired_tokens)
    
    async def get_revoked_count(self) -> int:
        """Get count of revoked tokens"""
        async with self._lock:
            return len(self._revoked_tokens)


class TokenStorageManager:
    """Manages token storage backend selection and lifecycle"""
    
    def __init__(self, storage_backend: Optional[str] = None):
        """
        Initialize token storage manager
        
        Args:
            storage_backend: Backend type ('redis', 'database', 'memory')
        """
        self.storage_backend = storage_backend or os.getenv("MCP_TOKEN_STORAGE", "redis")
        self.storage: Optional[TokenStorage] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self, db_connection=None) -> TokenStorage:
        """Initialize storage backend"""
        if self.storage is not None:
            return self.storage
        
        if self.storage_backend == "redis":
            self.storage = RedisTokenStorage()
        elif self.storage_backend == "database":
            # Use UnifiedDatabase for database storage
            self.storage = DatabaseTokenStorage(db_connection)
        else:
            # Fallback to in-memory
            self.storage = InMemoryTokenStorage()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info(f"Initialized token storage backend: {self.storage_backend}")
        return self.storage
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired tokens"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                if self.storage:
                    await self.storage.cleanup_expired_tokens()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in token cleanup task: {e}")
    
    async def close(self):
        """Close storage and cleanup tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.storage and hasattr(self.storage, 'close'):
            await self.storage.close()


# Utility functions
def hash_token(token: str) -> str:
    """Create SHA-256 hash of token for storage"""
    return hashlib.sha256(token.encode()).hexdigest()


# Global storage manager instance
_storage_manager = TokenStorageManager()


async def get_token_storage(db_connection=None) -> TokenStorage:
    """Get the global token storage instance"""
    return await _storage_manager.initialize(db_connection)


async def revoke_token(token: str, expires_at: Optional[datetime] = None) -> bool:
    """Convenience function to revoke a token"""
    storage = await get_token_storage()
    token_hash = hash_token(token)
    return await storage.revoke_token(token_hash, expires_at)


async def is_token_revoked(token: str) -> bool:
    """Convenience function to check if token is revoked"""
    storage = await get_token_storage()
    token_hash = hash_token(token)
    return await storage.is_token_revoked(token_hash)