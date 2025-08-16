"""
Unified Database Layer
Supports both local development (SQLite + Redis) and production (Vercel Postgres + Upstash Redis)
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DatabaseMode(Enum):
    LOCAL = "local"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Unified database configuration"""
    mode: DatabaseMode = DatabaseMode.LOCAL
    
    # Local development
    sqlite_path: str = "data/cryptotrading.db"
    redis_url: str = "redis://localhost:6379"
    
    # Production (Vercel)
    postgres_url: Optional[str] = None
    upstash_redis_url: Optional[str] = None
    upstash_redis_token: Optional[str] = None
    
    # Common settings
    cache_ttl: int = 3600
    enable_caching: bool = True

class UnifiedDatabase:
    """
    Unified database layer that automatically switches between:
    - Local: SQLite + Redis (optional)
    - Production: Vercel Postgres + Upstash Redis
    """
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or self._auto_detect_config()
        self.db_conn = None
        self.redis_client = None
        
    def _auto_detect_config(self) -> DatabaseConfig:
        """Auto-detect environment and configure accordingly"""
        config = DatabaseConfig()
        
        # Check if running on Vercel
        if os.getenv('VERCEL') or os.getenv('POSTGRES_URL'):
            config.mode = DatabaseMode.PRODUCTION
            config.postgres_url = os.getenv('POSTGRES_URL')
            config.upstash_redis_url = os.getenv('KV_URL')
            config.upstash_redis_token = os.getenv('KV_REST_API_TOKEN')
        else:
            config.mode = DatabaseMode.LOCAL
            
        logger.info(f"Database mode: {config.mode.value}")
        return config
    
    async def initialize(self):
        """Initialize database based on mode"""
        if self.config.mode == DatabaseMode.LOCAL:
            await self._init_local()
        else:
            await self._init_production()
            
        await self._create_schemas()
        logger.info("Database initialized successfully")
    
    async def _init_local(self):
        """Initialize local SQLite database"""
        # Ensure data directory exists
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to SQLite
        self.db_conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=30.0
        )
        self.db_conn.row_factory = sqlite3.Row
        
        # Enable foreign keys and WAL mode for better performance
        self.db_conn.execute("PRAGMA foreign_keys = ON")
        self.db_conn.execute("PRAGMA journal_mode = WAL")
        
        # Optional Redis for local caching
        if self.config.enable_caching:
            try:
                import redis
                self.redis_client = redis.Redis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info("Local Redis connected")
            except Exception as e:
                logger.warning(f"Redis not available locally: {e}")
                self.redis_client = None
        
        logger.info(f"Local SQLite database: {db_path}")
    
    async def _init_production(self):
        """Initialize production Vercel Postgres + Upstash Redis"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            # Connect to Vercel Postgres
            self.db_conn = psycopg2.connect(
                self.config.postgres_url,
                cursor_factory=RealDictCursor
            )
            logger.info("Vercel Postgres connected")
            
        except ImportError:
            raise ImportError("psycopg2 required for production PostgreSQL")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
        
        # Connect to Upstash Redis
        if self.config.upstash_redis_url and self.config.enable_caching:
            try:
                import redis
                self.redis_client = redis.Redis.from_url(
                    self.config.upstash_redis_url,
                    password=self.config.upstash_redis_token,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Upstash Redis connected")
            except Exception as e:
                logger.warning(f"Upstash Redis not available: {e}")
                self.redis_client = None
    
    async def _create_schemas(self):
        """Create unified schemas that work for both SQLite and PostgreSQL"""
        schemas = [
            self._get_issues_schema(),
            self._get_code_files_schema(),
            self._get_metrics_schema(),
            self._get_monitoring_schema()
        ]
        
        cursor = self.db_conn.cursor()
        try:
            for schema in schemas:
                if self.config.mode == DatabaseMode.LOCAL:
                    # SQLite version
                    cursor.executescript(schema['sqlite'])
                else:
                    # PostgreSQL version
                    cursor.execute(schema['postgres'])
            
            self.db_conn.commit()
            logger.info("Database schemas created")
            
        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            self.db_conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _get_issues_schema(self) -> Dict[str, str]:
        """Issue tracking schema for both SQLite and PostgreSQL"""
        return {
            'sqlite': """
                CREATE TABLE IF NOT EXISTS issues (
                    id TEXT PRIMARY KEY,
                    issue_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'detected',
                    priority TEXT NOT NULL DEFAULT 'medium',
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    description TEXT NOT NULL,
                    suggested_fix TEXT,
                    auto_fixable INTEGER DEFAULT 0,
                    fix_status TEXT DEFAULT 'pending',
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    fixed_at TIMESTAMP NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_issues_type ON issues(issue_type);
                CREATE INDEX IF NOT EXISTS idx_issues_severity ON issues(severity);
                CREATE INDEX IF NOT EXISTS idx_issues_state ON issues(state);
                CREATE INDEX IF NOT EXISTS idx_issues_file ON issues(file_path);
            """,
            'postgres': """
                CREATE TABLE IF NOT EXISTS issues (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    issue_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    state VARCHAR(20) NOT NULL DEFAULT 'detected',
                    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    description TEXT NOT NULL,
                    suggested_fix TEXT,
                    auto_fixable BOOLEAN DEFAULT FALSE,
                    fix_status VARCHAR(20) DEFAULT 'pending',
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    fixed_at TIMESTAMP NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_issues_type ON issues(issue_type);
                CREATE INDEX IF NOT EXISTS idx_issues_severity ON issues(severity);
                CREATE INDEX IF NOT EXISTS idx_issues_state ON issues(state);
                CREATE INDEX IF NOT EXISTS idx_issues_file ON issues(file_path);
            """
        }
    
    def _get_code_files_schema(self) -> Dict[str, str]:
        """Code files tracking schema"""
        return {
            'sqlite': """
                CREATE TABLE IF NOT EXISTS code_files (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL UNIQUE,
                    language TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    last_modified TIMESTAMP NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    facts_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                );
                
                CREATE INDEX IF NOT EXISTS idx_code_files_path ON code_files(file_path);
                CREATE INDEX IF NOT EXISTS idx_code_files_language ON code_files(language);
            """,
            'postgres': """
                CREATE TABLE IF NOT EXISTS code_files (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    file_path TEXT NOT NULL UNIQUE,
                    language VARCHAR(50) NOT NULL,
                    file_size BIGINT NOT NULL,
                    content_hash VARCHAR(64) NOT NULL,
                    last_modified TIMESTAMP NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    facts_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
                
                CREATE INDEX IF NOT EXISTS idx_code_files_path ON code_files(file_path);
                CREATE INDEX IF NOT EXISTS idx_code_files_language ON code_files(language);
            """
        }
    
    def _get_metrics_schema(self) -> Dict[str, str]:
        """Code quality metrics schema"""
        return {
            'sqlite': """
                CREATE TABLE IF NOT EXISTS code_metrics (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold_value REAL,
                    status TEXT DEFAULT 'ok',
                    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_file ON code_metrics(file_path);
                CREATE INDEX IF NOT EXISTS idx_metrics_type ON code_metrics(metric_type);
            """,
            'postgres': """
                CREATE TABLE IF NOT EXISTS code_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    file_path TEXT NOT NULL,
                    metric_type VARCHAR(50) NOT NULL,
                    metric_value DECIMAL NOT NULL,
                    threshold_value DECIMAL,
                    status VARCHAR(20) DEFAULT 'ok',
                    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_file ON code_metrics(file_path);
                CREATE INDEX IF NOT EXISTS idx_metrics_type ON code_metrics(metric_type);
            """
        }
    
    def _get_monitoring_schema(self) -> Dict[str, str]:
        """System monitoring schema"""
        return {
            'sqlite': """
                CREATE TABLE IF NOT EXISTS monitoring_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT DEFAULT '{}',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_monitoring_type ON monitoring_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_monitoring_component ON monitoring_events(component);
            """,
            'postgres': """
                CREATE TABLE IF NOT EXISTS monitoring_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type VARCHAR(50) NOT NULL,
                    component VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    details JSONB DEFAULT '{}'::jsonb,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_monitoring_type ON monitoring_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_monitoring_component ON monitoring_events(component);
            """
        }
    
    # Cache operations
    async def cache_set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set cache value with optional TTL"""
        if not self.redis_client:
            return False
            
        try:
            ttl = ttl or self.config.cache_ttl
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self.redis_client:
            return None
            
        try:
            value = self.redis_client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete cache key"""
        if not self.redis_client:
            return False
            
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        if self.db_conn:
            self.db_conn.close()
        if self.redis_client:
            self.redis_client.close()
