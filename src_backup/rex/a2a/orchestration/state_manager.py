"""
Distributed State Manager using Vercel KV
Provides distributed state management for workflows
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class StateManager:
    """Manages distributed state using Vercel KV or fallback to SQLite"""
    
    def __init__(self):
        self.use_vercel_kv = bool(os.getenv('KV_REST_API_URL'))
        
        if self.use_vercel_kv:
            self._init_vercel_kv()
        else:
            self._init_sqlite_fallback()
            logger.warning("Vercel KV not configured, using SQLite for state management")
    
    def _init_vercel_kv(self):
        """Initialize Vercel KV client"""
        try:
            from vercel_kv import KV
            self.kv = KV()
            logger.info("Initialized Vercel KV for distributed state")
        except Exception as e:
            logger.error(f"Failed to initialize Vercel KV: {e}")
            self.use_vercel_kv = False
            self._init_sqlite_fallback()
    
    def _init_sqlite_fallback(self):
        """Initialize SQLite fallback for local development"""
        from ...database.client import get_db
        self.db = get_db()
        
        # Create state table if needed
        from sqlalchemy import text
        with self.db.get_session() as session:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS workflow_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            session.commit()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed state"""
        try:
            if self.use_vercel_kv:
                value = await self.kv.get(key)
                return json.loads(value) if value else None
            else:
                # SQLite fallback
                from sqlalchemy import text
                with self.db.get_session() as session:
                    result = session.execute(
                        text("SELECT value FROM workflow_state WHERE key = :key AND (expires_at IS NULL OR expires_at > :now)"),
                        {"key": key, "now": datetime.now()}
                    ).fetchone()
                    return json.loads(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error getting state for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in distributed state with optional TTL (seconds)"""
        try:
            value_json = json.dumps(value)
            
            if self.use_vercel_kv:
                if ttl:
                    await self.kv.setex(key, ttl, value_json)
                else:
                    await self.kv.set(key, value_json)
                return True
            else:
                # SQLite fallback
                from sqlalchemy import text
                expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
                with self.db.get_session() as session:
                    session.execute(text("""
                        INSERT OR REPLACE INTO workflow_state (key, value, expires_at, updated_at)
                        VALUES (:key, :value, :expires_at, :updated_at)
                    """), {"key": key, "value": value_json, "expires_at": expires_at, "updated_at": datetime.now()})
                    session.commit()
                return True
        except Exception as e:
            logger.error(f"Error setting state for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from distributed state"""
        try:
            if self.use_vercel_kv:
                await self.kv.delete(key)
            else:
                from sqlalchemy import text
                with self.db.get_session() as session:
                    session.execute(text("DELETE FROM workflow_state WHERE key = :key"), {"key": key})
                    session.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting state for {key}: {e}")
            return False
    
    async def get_pattern(self, pattern: str) -> List[str]:
        """Get all keys matching pattern"""
        try:
            if self.use_vercel_kv:
                return await self.kv.keys(pattern)
            else:
                # SQLite pattern matching
                from sqlalchemy import text
                sql_pattern = pattern.replace('*', '%')
                with self.db.get_session() as session:
                    results = session.execute(
                        text("SELECT key FROM workflow_state WHERE key LIKE :pattern AND (expires_at IS NULL OR expires_at > :now)"),
                        {"pattern": sql_pattern, "now": datetime.now()}
                    ).fetchall()
                    return [r[0] for r in results]
        except Exception as e:
            logger.error(f"Error getting keys for pattern {pattern}: {e}")
            return []
    
    async def cleanup_expired(self):
        """Clean up expired entries (SQLite only)"""
        if not self.use_vercel_kv:
            try:
                from sqlalchemy import text
                with self.db.get_session() as session:
                    session.execute(
                        text("DELETE FROM workflow_state WHERE expires_at < :now"),
                        {"now": datetime.now()}
                    )
                    session.commit()
            except Exception as e:
                logger.error(f"Error cleaning up expired state: {e}")

# Global state manager instance
state_manager = StateManager()