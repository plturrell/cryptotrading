"""
Production-ready database connection pool with transaction management
Implements connection pooling, retry logic, and proper transaction boundaries
"""

import asyncio
import contextlib
import logging
from typing import Any, Dict, Optional, List, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
# Handle missing database dependencies gracefully
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    # Create mock asyncpg for testing
    class MockAsyncpg:
        @staticmethod
        async def create_pool(*args, **kwargs):
            raise NotImplementedError("asyncpg not available")
        
        class Record:
            def __init__(self, data):
                self._data = data
            def __getitem__(self, key):
                return self._data.get(key)
            def keys(self):
                return self._data.keys()
            def values(self):
                return self._data.values()
    
    asyncpg = MockAsyncpg()

try:
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    # Create mock psycopg2 for testing
    class MockPsycopg2:
        class pool:
            class ThreadedConnectionPool:
                def __init__(self, *args, **kwargs):
                    raise NotImplementedError("psycopg2 not available")
    psycopg2 = MockPsycopg2()
import sqlite3
from threading import Lock
import time
from functools import wraps
import os

logger = logging.getLogger(__name__)

@dataclass
class PoolConfig:
    """Connection pool configuration"""
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 10.0
    idle_timeout: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0
    
class CircuitBreaker:
    """Circuit breaker pattern for database connections"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = Lock()
        
    def call(self, func: Callable) -> Any:
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise ConnectionError("Circuit breaker is open")
                    
        try:
            result = func()
            with self._lock:
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error("Circuit breaker opened after %d failures", self.failure_count)
                    
            raise e

class RetryPolicy:
    """Retry policy with exponential backoff"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        
    def retry(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < self.max_attempts - 1:
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        logger.warning("Retry attempt %d/%d after %.2fs: %s", 
                                     attempt + 1, self.max_attempts, delay, str(e))
                        await asyncio.sleep(delay)
                    
            raise last_exception
        return wrapper


class DatabaseConnectionPool:
    """Production-ready database connection pool"""
    
    def __init__(self, db_url: str, config: Optional[PoolConfig] = None):
        self.db_url = db_url
        self.config = config or PoolConfig()
        self.circuit_breaker = CircuitBreaker()
        self.retry_policy = RetryPolicy(self.config.retry_attempts, self.config.retry_delay)
        self._pool = None
        self._health_check_task = None
        self._metrics = {
            "connections_created": 0,
            "connections_closed": 0,
            "queries_executed": 0,
            "errors": 0,
            "avg_query_time": 0.0
        }
        
    async def initialize(self):
        """Initialize the connection pool"""
        try:
            if self.db_url.startswith("postgresql://"):
                if not ASYNCPG_AVAILABLE:
                    raise RuntimeError("PostgreSQL requested but asyncpg not available. Install with: pip install asyncpg")
                else:
                    self._pool = await asyncpg.create_pool(
                        self.db_url,
                        min_size=self.config.min_connections,
                        max_size=self.config.max_connections,
                        timeout=self.config.connection_timeout,
                        max_inactive_connection_lifetime=self.config.idle_timeout
                    )
            else:
                # For SQLite, we'll manage our own simple pool
                self._pool = SQLitePool(self.db_url, self.config)
                
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Database connection pool initialized with %d-%d connections",
                       self.config.min_connections, self.config.max_connections)
                       
        except Exception as e:
            logger.error("Failed to initialize connection pool: %s", e)
            raise
            
    async def close(self):
        """Close the connection pool gracefully"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
        if self._pool:
            if hasattr(self._pool, 'close'):
                await self._pool.close()
            logger.info("Database connection pool closed")
            
    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """Acquire a connection from the pool"""
        start_time = time.time()
        connection = None
        
        try:
            connection = await self._acquire_connection()
            self._metrics["connections_created"] += 1
            yield connection
            
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("Error acquiring connection: %s", e)
            raise
            
        finally:
            if connection:
                await self._release_connection(connection)
                self._metrics["connections_closed"] += 1
                
            # Update metrics
            query_time = time.time() - start_time
            self._metrics["queries_executed"] += 1
            self._metrics["avg_query_time"] = (
                (self._metrics["avg_query_time"] * (self._metrics["queries_executed"] - 1) + query_time) /
                self._metrics["queries_executed"]
            )
            
    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        """Execute operations within a transaction"""
        async with self.acquire() as conn:
            if hasattr(conn, 'transaction'):
                # PostgreSQL
                async with conn.transaction():
                    yield conn
            else:
                # SQLite
                try:
                    await conn.execute("BEGIN")
                    yield conn
                    await conn.execute("COMMIT")
                except Exception:
                    await conn.execute("ROLLBACK")
                    raise
                    
    async def execute(self, query: str, *params) -> Any:
        """Execute a query with retry logic"""
        @self.retry_policy.retry
        async def _execute():
            async with self.acquire() as conn:
                return await conn.execute(query, *params)
        return await _execute()
            
    async def fetch(self, query: str, *params) -> List[Any]:
        """Fetch multiple rows with retry logic"""
        @self.retry_policy.retry
        async def _fetch():
            async with self.acquire() as conn:
                return await conn.fetch(query, *params)
        return await _fetch()
            
    async def fetchrow(self, query: str, *params) -> Any:
        """Fetch single row with retry logic"""
        @self.retry_policy.retry
        async def _fetchrow():
            async with self.acquire() as conn:
                return await conn.fetchrow(query, *params)
        return await _fetchrow()
        
    async def fetchone(self, query: str, *params) -> Optional[Any]:
        """Fetch single result with retry logic"""
        @self.retry_policy.retry
        async def _fetchone():
            async with self.acquire() as conn:
                if hasattr(conn, 'fetchrow'):
                    return await conn.fetchrow(query, *params)
                else:
                    # SQLite compatibility
                    cursor = await conn.execute(query, params)
                    return await cursor.fetchone()
        return await _fetchone()
                
    async def _acquire_connection(self) -> Any:
        """Acquire connection with circuit breaker"""
        def acquire():
            if hasattr(self._pool, 'acquire'):
                return self._pool.acquire()
            else:
                return self._pool.get_connection()
                
        return await self.circuit_breaker.call(acquire)
        
    async def _release_connection(self, connection: Any):
        """Release connection back to pool"""
        if hasattr(self._pool, 'release'):
            await self._pool.release(connection)
        else:
            self._pool.return_connection(connection)
            
    async def _health_check_loop(self):
        """Periodic health check for connections"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check failed: %s", e)
                
    async def _health_check(self):
        """Perform health check on pool"""
        try:
            async with self.acquire() as conn:
                if hasattr(conn, 'fetchval'):
                    await conn.fetchval("SELECT 1")
                else:
                    await conn.execute("SELECT 1")
                    
            logger.debug("Database health check passed")
            
        except Exception as e:
            logger.error("Database health check failed: %s", e)
            raise
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics"""
        metrics = self._metrics.copy()
        
        if hasattr(self._pool, 'get_stats'):
            pool_stats = self._pool.get_stats()
            metrics.update({
                "pool_size": pool_stats.get('size', 0),
                "pool_available": pool_stats.get('available', 0),
                "pool_waiting": pool_stats.get('waiting', 0)
            })
            
        return metrics
        
class SQLitePool:
    """Simple connection pool for SQLite"""
    
    def __init__(self, db_path: str, config: PoolConfig):
        self.db_path = db_path
        self.config = config
        self._connections: List[sqlite3.Connection] = []
        self._available: List[sqlite3.Connection] = []
        self._lock = Lock()
        
        # Create initial connections
        for _ in range(config.min_connections):
            conn = self._create_connection()
            self._connections.append(conn)
            self._available.append(conn)
            
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
        
    async def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool"""
        with self._lock:
            if self._available:
                return self._available.pop()
                
            if len(self._connections) < self.config.max_connections:
                conn = self._create_connection()
                self._connections.append(conn)
                return conn
                
        # Wait for available connection
        start_time = time.time()
        while time.time() - start_time < self.config.connection_timeout:
            await asyncio.sleep(0.1)
            with self._lock:
                if self._available:
                    return self._available.pop()
                    
        raise TimeoutError("No available connections in pool")
        
    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        with self._lock:
            if conn in self._connections:
                self._available.append(conn)
                
    async def close(self):
        """Close all connections"""
        with self._lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()
            self._available.clear()
            
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                "size": len(self._connections),
                "available": len(self._available),
                "waiting": 0  # SQLite doesn't queue
            }

# Connection pool factory
_pools: Dict[str, DatabaseConnectionPool] = {}
_pools_lock = Lock()

async def get_connection_pool(db_url: str, config: Optional[PoolConfig] = None) -> DatabaseConnectionPool:
    """Get or create a connection pool for the given database URL"""
    with _pools_lock:
        if db_url not in _pools:
            pool = DatabaseConnectionPool(db_url, config)
            await pool.initialize()
            _pools[db_url] = pool
            
        return _pools[db_url]
        
async def close_all_pools():
    """Close all connection pools"""
    with _pools_lock:
        for pool in _pools.values():
            await pool.close()
        _pools.clear()