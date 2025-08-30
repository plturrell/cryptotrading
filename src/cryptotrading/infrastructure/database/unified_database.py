"""
Unified Database Layer - Consolidated Implementation
Combines environment-aware design with advanced SQLAlchemy features
Supports both local development (SQLite + Redis) and production (Vercel Postgres + Upstash Redis)
"""

import os
import json
import logging
import sqlite3
import pandas as pd
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

# SQLAlchemy imports for advanced features
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool, StaticPool

logger = logging.getLogger(__name__)

class DatabaseMode(Enum):
    LOCAL = "local"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Unified database configuration with enhanced settings"""
    mode: DatabaseMode = DatabaseMode.LOCAL
    
    # Local development
    sqlite_path: str = "data/cryptotrading.db"
    redis_url: str = "redis://localhost:6379"
    
    # Production (Vercel)
    postgres_url: Optional[str] = None
    upstash_redis_url: Optional[str] = None
    upstash_redis_token: Optional[str] = None
    
    # Advanced connection pooling settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    
    # Common settings
    cache_ttl: int = 3600
    enable_caching: bool = True
    enable_health_monitoring: bool = True
    enable_performance_monitoring: bool = True

class UnifiedDatabase:
    """
    Unified database layer that automatically switches between:
    - Local: SQLite + Redis (optional)
    - Production: Vercel Postgres + Upstash Redis
    
    Features:
    - Environment-aware configuration
    - Advanced SQLAlchemy integration
    - Connection pooling and health monitoring
    - Redis caching support
    - Automatic schema management
    """
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or self._auto_detect_config()
        
        # Initialize connection components
        self.db_conn = None  # Simple connection for direct queries
        self.engine = None   # SQLAlchemy engine for ORM
        self.SessionLocal = None  # SQLAlchemy session factory
        self.Session = None  # Scoped session
        self.redis_client = None
        
        # Advanced features
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._health_monitoring_enabled = config and config.enable_health_monitoring
        self._performance_monitoring_enabled = config and config.enable_performance_monitoring
        
        # Component trackers
        self._migrator = None
        self._health_monitor = None
        self._performance_monitor = None
        # Use threading lock for sync contexts, async lock for async contexts
        try:
            # Try to get current event loop for async contexts
            asyncio.get_running_loop()
            self._db_lock = asyncio.Lock()
            self._is_async_context = True
        except RuntimeError:
            # No event loop running, use threading lock for sync contexts
            self._db_lock = threading.Lock()
            self._is_async_context = False
        
    def _get_database_url(self) -> str:
        """Get database URL from environment or config"""
        if self.config.mode == DatabaseMode.PRODUCTION and self.config.postgres_url:
            return self.config.postgres_url
        elif self.config.mode == DatabaseMode.LOCAL:
            return f'sqlite:///{self.config.sqlite_path}'
        
        # Fallback environment detection
        db_url = os.getenv('DATABASE_URL') or os.getenv('POSTGRES_URL')
        if db_url:
            return db_url
            
        # Development SQLite fallback
        db_path = os.getenv('DATABASE_PATH', 'data/cryptotrading.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f'sqlite:///{db_path}'
    
    def _create_sqlalchemy_engine(self):
        """Create SQLAlchemy engine with appropriate configuration"""
        db_url = self._get_database_url()
        is_sqlite = 'sqlite' in db_url
        
        if is_sqlite:
            # SQLite configuration
            connect_args = {'check_same_thread': False, 'timeout': 30}
            return create_engine(
                db_url,
                poolclass=StaticPool,
                connect_args=connect_args,
                echo=False
            )
        else:
            # PostgreSQL configuration with advanced pooling
            return create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=False
            )
    
    def _setup_sqlalchemy(self):
        """Setup SQLAlchemy session management"""
        if not self.engine:
            self.engine = self._create_sqlalchemy_engine()
            
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False
        )
        self.Session = scoped_session(self.SessionLocal)
        
    @contextmanager
    def get_session(self):
        """Get SQLAlchemy session with automatic cleanup"""
        if not self.Session:
            self._setup_sqlalchemy()
            
        session = self.Session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
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
        """Initialize database based on mode with both simple and SQLAlchemy connections"""
        # Setup SQLAlchemy first for ORM operations
        self._setup_sqlalchemy()
        
        # Setup simple connections for direct queries
        if self.config.mode == DatabaseMode.LOCAL:
            await self._init_local()
        else:
            await self._init_production()
            
        await self._create_schemas()
        
        # Initialize advanced components if enabled
        if self._health_monitoring_enabled:
            await self._setup_health_monitoring()
        if self._performance_monitoring_enabled:
            await self._setup_performance_monitoring()
            
        logger.info("Database initialized successfully with both simple and ORM interfaces")
    
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
            self._get_monitoring_schema(),
            # Add crypto trading schemas
            self._get_market_data_schema(),
            self._get_portfolio_schema(),
            self._get_trading_orders_schema(),
            self._get_historical_data_schema(),
            # Add intelligence schemas
            self._get_intelligence_schemas()
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
    
    def _get_intelligence_schemas(self) -> Dict[str, str]:
        """Intelligence system schemas for AI insights, decisions, and memory"""
        # Import intelligence schemas from the dedicated module
        from ...data.database.intelligence_schema import get_all_intelligence_schemas
        return get_all_intelligence_schemas()
    
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
    
    def _get_market_data_schema(self) -> Dict[str, str]:
        """Market data schema for crypto prices from Yahoo Finance"""
        return {
            'sqlite': """
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'yahoo_finance',
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, source, timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                ON market_data(symbol, timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_market_data_source 
                ON market_data(source);
            """,
            'postgres': """
                CREATE TABLE IF NOT EXISTS market_data (
                    id BIGSERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    source VARCHAR(50) NOT NULL DEFAULT 'yahoo_finance',
                    open DECIMAL(20, 8) NOT NULL,
                    high DECIMAL(20, 8) NOT NULL,
                    low DECIMAL(20, 8) NOT NULL,
                    close DECIMAL(20, 8) NOT NULL,
                    volume DECIMAL(20, 8) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, source, timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                ON market_data(symbol, timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_market_data_source 
                ON market_data(source);
            """
        }
    
    def _get_portfolio_schema(self) -> Dict[str, str]:
        """Portfolio positions schema"""
        return {
            'sqlite': """
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_portfolio_user_symbol 
                ON portfolio_positions(user_id, symbol);
            """,
            'postgres': """
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id BIGSERIAL PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    quantity DECIMAL(20, 8) NOT NULL,
                    average_price DECIMAL(20, 8) NOT NULL,
                    current_price DECIMAL(20, 8),
                    unrealized_pnl DECIMAL(20, 8),
                    realized_pnl DECIMAL(20, 8) DEFAULT 0,
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_portfolio_user_symbol 
                ON portfolio_positions(user_id, symbol);
            """
        }
    
    def _get_trading_orders_schema(self) -> Dict[str, str]:
        """Trading orders schema"""
        return {
            'sqlite': """
                CREATE TABLE IF NOT EXISTS trading_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                    order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
                    quantity REAL NOT NULL,
                    price REAL,
                    executed_quantity REAL DEFAULT 0,
                    executed_price REAL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    fees REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                );
                
                CREATE INDEX IF NOT EXISTS idx_orders_user_symbol 
                ON trading_orders(user_id, symbol, created_at DESC);
                
                CREATE INDEX IF NOT EXISTS idx_orders_status 
                ON trading_orders(status);
            """,
            'postgres': """
                CREATE TABLE IF NOT EXISTS trading_orders (
                    id BIGSERIAL PRIMARY KEY,
                    order_id VARCHAR(100) UNIQUE NOT NULL,
                    user_id VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
                    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
                    quantity DECIMAL(20, 8) NOT NULL,
                    price DECIMAL(20, 8),
                    executed_quantity DECIMAL(20, 8) DEFAULT 0,
                    executed_price DECIMAL(20, 8),
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    fees DECIMAL(20, 8) DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP WITH TIME ZONE,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
                
                CREATE INDEX IF NOT EXISTS idx_orders_user_symbol 
                ON trading_orders(user_id, symbol, created_at DESC);
                
                CREATE INDEX IF NOT EXISTS idx_orders_status 
                ON trading_orders(status);
            """
        }
    
    def _get_historical_data_schema(self) -> Dict[str, str]:
        """Historical data cache schema for Yahoo Finance and FRED data"""
        return {
            'sqlite': """
                CREATE TABLE IF NOT EXISTS historical_data_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_source TEXT NOT NULL CHECK (data_source IN ('yahoo_finance', 'fred')),
                    symbol_or_series TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    value REAL NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(data_source, symbol_or_series, data_type, timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_historical_source_symbol 
                ON historical_data_cache(data_source, symbol_or_series, timestamp DESC);
            """,
            'postgres': """
                CREATE TABLE IF NOT EXISTS historical_data_cache (
                    id BIGSERIAL PRIMARY KEY,
                    data_source VARCHAR(50) NOT NULL CHECK (data_source IN ('yahoo_finance', 'fred')),
                    symbol_or_series VARCHAR(50) NOT NULL,
                    data_type VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    value DECIMAL(20, 8) NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(data_source, symbol_or_series, data_type, timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_historical_source_symbol 
                ON historical_data_cache(data_source, symbol_or_series, timestamp DESC);
            """
        }
    
    # Trading data operations
    async def store_market_data(self, symbol: str, data: Dict[str, Any], source: str = 'yahoo_finance') -> bool:
        """Store market data from Yahoo Finance"""
        async with self._db_lock:  # Fix race condition
            cursor = self.db_conn.cursor()
            try:
                if self.config.mode == DatabaseMode.LOCAL:
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_data 
                        (symbol, source, open, high, low, close, volume, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, source, data['open'], data['high'], data['low'], 
                        data['close'], data['volume'], data['timestamp']
                    ))
                else:
                    cursor.execute("""
                        INSERT INTO market_data 
                        (symbol, source, open, high, low, close, volume, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, source, timestamp) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """, (
                        symbol, source, data['open'], data['high'], data['low'], 
                        data['close'], data['volume'], data['timestamp']
                    ))
                
                self.db_conn.commit()
                return True
                
            except Exception as e:
                logger.error(f"Failed to store market data: {e}")
                self.db_conn.rollback()
                return False
            finally:
                cursor.close()
    
    async def get_latest_market_data(self, symbol: str, source: str = 'yahoo_finance') -> Optional[Dict[str, Any]]:
        """Get latest market data for symbol"""
        async with self._db_lock:  # Fix race condition
            cursor = self.db_conn.cursor()
            try:
                if self.config.mode == DatabaseMode.LOCAL:
                    cursor.execute("""
                        SELECT * FROM market_data 
                        WHERE symbol = ? AND source = ?
                        ORDER BY timestamp DESC LIMIT 1
                    """, (symbol, source))
                else:
                    cursor.execute("""
                        SELECT * FROM market_data 
                        WHERE symbol = %s AND source = %s
                        ORDER BY timestamp DESC LIMIT 1
                    """, (symbol, source))
                
                row = cursor.fetchone()
                if row:
                    if self.config.mode == DatabaseMode.LOCAL:
                        return dict(row)
                    else:
                        return dict(zip([desc[0] for desc in cursor.description], row))
                
                return None
                
            except Exception as e:
                logger.error(f"Failed to get market data: {e}")
                return None
            finally:
                cursor.close()
    
    async def store_historical_data_batch(self, data_source: str, symbol_or_series: str, 
                                        df: pd.DataFrame) -> bool:
        """Store batch historical data from Yahoo Finance or FRED"""
        cursor = self.db_conn.cursor()
        try:
            records = []
            
            # Process dataframe based on source
            if data_source == 'yahoo_finance':
                for timestamp, row in df.iterrows():
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in row and pd.notna(row[col]):
                            records.append((
                                data_source, symbol_or_series, col,
                                timestamp, float(row[col]), '{}'
                            ))
            elif data_source == 'fred':
                # FRED data typically has a single value column
                value_col = symbol_or_series if symbol_or_series in df.columns else df.columns[0]
                for timestamp, row in df.iterrows():
                    if pd.notna(row[value_col]):
                        records.append((
                            data_source, symbol_or_series, 'value',
                            timestamp, float(row[value_col]), '{}'
                        ))
            
            # Batch insert
            if records:
                if self.config.mode == DatabaseMode.LOCAL:
                    cursor.executemany("""
                        INSERT OR REPLACE INTO historical_data_cache
                        (data_source, symbol_or_series, data_type, timestamp, value, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, records)
                else:
                    from psycopg2.extras import execute_values
                    execute_values(cursor, """
                        INSERT INTO historical_data_cache
                        (data_source, symbol_or_series, data_type, timestamp, value, metadata)
                        VALUES %s
                        ON CONFLICT (data_source, symbol_or_series, data_type, timestamp) 
                        DO UPDATE SET value = EXCLUDED.value
                    """, records)
                
                self.db_conn.commit()
                logger.info(f"Stored {len(records)} historical records for {symbol_or_series}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to store historical data batch: {e}")
            self.db_conn.rollback()
            return False
        finally:
            cursor.close()
    
    async def get_portfolio_positions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get portfolio positions for user"""
        cursor = self.db_conn.cursor()
        try:
            if self.config.mode == DatabaseMode.LOCAL:
                cursor.execute("""
                    SELECT p.*, m.close as current_market_price
                    FROM portfolio_positions p
                    LEFT JOIN (
                        SELECT symbol, close, 
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                        FROM market_data
                    ) m ON p.symbol = m.symbol AND m.rn = 1
                    WHERE p.user_id = ? AND p.quantity > 0
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT p.*, m.close as current_market_price
                    FROM portfolio_positions p
                    LEFT JOIN LATERAL (
                        SELECT close FROM market_data 
                        WHERE symbol = p.symbol 
                        ORDER BY timestamp DESC LIMIT 1
                    ) m ON true
                    WHERE p.user_id = %s AND p.quantity > 0
                """, (user_id,))
            
            rows = cursor.fetchall()
            positions = []
            
            for row in rows:
                if self.config.mode == DatabaseMode.LOCAL:
                    pos = dict(row)
                else:
                    pos = dict(zip([desc[0] for desc in cursor.description], row))
                
                # Calculate current metrics
                if pos.get('current_market_price'):
                    current_price = float(pos['current_market_price'])
                    quantity = float(pos['quantity'])
                    avg_price = float(pos['average_price'])
                    
                    pos['current_price'] = current_price
                    pos['current_value'] = quantity * current_price
                    pos['unrealized_pnl'] = quantity * (current_price - avg_price)
                    pos['unrealized_pnl_pct'] = ((current_price - avg_price) / avg_price) * 100
                
                positions.append(pos)
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get portfolio positions: {e}")
            return []
        finally:
            cursor.close()
    
    async def _setup_health_monitoring(self):
        """Setup database health monitoring"""
        try:
            # Import health monitoring components if available
            from ...data.database.health_monitor import DatabaseHealthMonitor
            self._health_monitor = DatabaseHealthMonitor(self)
            logger.info("Health monitoring enabled")
        except ImportError:
            logger.warning("Health monitoring components not available")
    
    async def _setup_performance_monitoring(self):
        """Setup database performance monitoring"""
        try:
            # Import performance monitoring components if available
            from ...data.database.performance_monitor import PerformanceMonitor
            self._performance_monitor = PerformanceMonitor()
            logger.info("Performance monitoring enabled")
        except ImportError:
            logger.warning("Performance monitoring components not available")
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute query with ORM session (legacy compatibility)"""
        from sqlalchemy import text
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return [dict(row) for row in result.fetchall()]
    
    async def execute_query_async(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute query asynchronously using thread pool"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, self.execute_query, query, params
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get database health status"""
        status = {
            "engine_connected": self.engine is not None,
            "simple_connection": self.db_conn is not None,
            "redis_connected": self.redis_client is not None,
            "mode": self.config.mode.value
        }
        
        if self._health_monitor:
            status.update(self._health_monitor.get_status())
            
        return status
    
    async def close(self):
        """Close all database connections and cleanup resources"""
        # Close SQLAlchemy connections
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
            
        # Close simple connections
        if self.db_conn:
            self.db_conn.close()
        if self.redis_client:
            self.redis_client.close()
            
        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=False)
            
        logger.info("Database connections closed")
