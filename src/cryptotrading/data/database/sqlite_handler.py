"""
SQLite-specific handling for connection management
Provides proper write locking and connection pooling for SQLite
"""

import sqlite3
import threading
import queue
import logging
from contextlib import contextmanager
from typing import Optional, Any
import time

logger = logging.getLogger(__name__)

class SQLiteConnectionPool:
    """
    Thread-safe connection pool for SQLite with write serialization
    SQLite only supports one writer at a time, so we serialize writes
    """
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        
        # Connection pool
        self._connections = queue.Queue(maxsize=pool_size)
        
        # Write lock to serialize write operations
        self._write_lock = threading.Lock()
        
        # Initialize connections
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self._connections.put(conn)
            
        logger.info(f"SQLite connection pool initialized with {self.pool_size} connections")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimized settings"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,  # Autocommit mode
            check_same_thread=False
        )
        
        # Set optimizations
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -10000")  # 10MB cache
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        return conn
    
    @contextmanager
    def get_connection(self, for_write: bool = False):
        """
        Get a connection from the pool
        
        Args:
            for_write: If True, acquires write lock for exclusive access
        """
        conn = None
        write_lock_acquired = False
        
        try:
            # Acquire write lock if needed
            if for_write:
                self._write_lock.acquire()
                write_lock_acquired = True
            
            # Get connection from pool with timeout
            conn = self._connections.get(timeout=5.0)
            
            yield conn
            
        except queue.Empty:
            raise RuntimeError("No available connections in pool")
            
        finally:
            # Return connection to pool
            if conn:
                try:
                    # Test connection is still valid
                    conn.execute("SELECT 1")
                    self._connections.put(conn)
                except sqlite3.Error:
                    # Connection is broken, create new one
                    logger.warning("Replacing broken SQLite connection")
                    new_conn = self._create_connection()
                    self._connections.put(new_conn)
            
            # Release write lock
            if write_lock_acquired:
                self._write_lock.release()
    
    def execute_write(self, query: str, params: tuple = None) -> Any:
        """Execute a write query with exclusive access"""
        with self.get_connection(for_write=True) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("BEGIN IMMEDIATE")
                
                if params:
                    result = cursor.execute(query, params)
                else:
                    result = cursor.execute(query)
                
                conn.commit()
                return result.lastrowid if result.lastrowid else None
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()
    
    def execute_read(self, query: str, params: tuple = None) -> list:
        """Execute a read query"""
        with self.get_connection(for_write=False) as conn:
            cursor = conn.cursor()
            
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                return cursor.fetchall()
                
            finally:
                cursor.close()
    
    def execute_many(self, query: str, params_list: list) -> None:
        """Execute multiple write queries in a single transaction"""
        with self.get_connection(for_write=True) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("BEGIN IMMEDIATE")
                cursor.executemany(query, params_list)
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()
    
    def close(self):
        """Close all connections in the pool"""
        while not self._connections.empty():
            try:
                conn = self._connections.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        logger.info("SQLite connection pool closed")


class SQLiteTransactionManager:
    """Manages transactions for SQLite with proper isolation"""
    
    def __init__(self, pool: SQLiteConnectionPool):
        self.pool = pool
    
    @contextmanager
    def transaction(self, isolation_level: str = "DEFERRED"):
        """
        Execute operations within a transaction
        
        Args:
            isolation_level: DEFERRED, IMMEDIATE, or EXCLUSIVE
        """
        with self.pool.get_connection(for_write=True) as conn:
            cursor = conn.cursor()
            
            try:
                # Start transaction with specified isolation
                cursor.execute(f"BEGIN {isolation_level}")
                
                yield conn
                
                # Commit on success
                conn.commit()
                
            except Exception as e:
                # Rollback on error
                conn.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise
                
            finally:
                cursor.close()
    
    @contextmanager
    def savepoint(self, name: str, conn: sqlite3.Connection):
        """Create a savepoint within a transaction"""
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SAVEPOINT {name}")
            yield
            cursor.execute(f"RELEASE SAVEPOINT {name}")
            
        except Exception as e:
            cursor.execute(f"ROLLBACK TO SAVEPOINT {name}")
            raise e
            
        finally:
            cursor.close()


def with_retry(max_attempts: int = 3, delay: float = 0.1):
    """Decorator to retry SQLite operations on busy/locked errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except sqlite3.OperationalError as e:
                    last_error = e
                    if "database is locked" in str(e) or "database table is locked" in str(e):
                        if attempt < max_attempts - 1:
                            wait_time = delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"SQLite busy/locked, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    raise
                    
                except Exception:
                    raise
            
            # All retries exhausted
            raise last_error
            
        return wrapper
    return decorator