"""
Integration tests for database operations using real test databases
No mocks - uses actual database connections with test data
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
import asyncpg
import sqlite3

from cryptotrading.infrastructure.database.connection_pool import (
    DatabaseConnectionPool, PoolConfig, get_connection_pool, close_all_pools
)
from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase
from cryptotrading.data.database import DatabaseClient
from cryptotrading.core.config.production_config import DatabaseConfig

# Test database URLs
TEST_SQLITE_PATH = tempfile.mktemp(suffix=".db")
TEST_POSTGRES_URL = os.getenv("TEST_DATABASE_URL", "postgresql://test:test@localhost:5432/cryptotrading_test")

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def sqlite_pool():
    """Create SQLite test database pool"""
    config = PoolConfig(
        min_connections=2,
        max_connections=5,
        connection_timeout=5.0
    )
    
    pool = await get_connection_pool(f"sqlite:///{TEST_SQLITE_PATH}", config)
    
    # Create test schema
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                amount REAL NOT NULL,
                avg_price REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    yield pool
    
    # Cleanup
    await pool.close()
    os.unlink(TEST_SQLITE_PATH)

@pytest.fixture
async def postgres_pool():
    """Create PostgreSQL test database pool"""
    # Skip if PostgreSQL not available
    try:
        test_conn = await asyncpg.connect(TEST_POSTGRES_URL)
        await test_conn.close()
    except:
        pytest.skip("PostgreSQL test database not available")
        
    config = PoolConfig(
        min_connections=2,
        max_connections=10,
        connection_timeout=10.0
    )
    
    pool = await get_connection_pool(TEST_POSTGRES_URL, config)
    
    # Create test schema
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                amount DECIMAL(20,8) NOT NULL,
                price DECIMAL(20,8),
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) UNIQUE NOT NULL,
                amount DECIMAL(20,8) NOT NULL,
                avg_price DECIMAL(20,8) NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                price DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Clear any existing data
        await conn.execute("TRUNCATE trades, portfolio, market_data CASCADE")
    
    yield pool
    
    # Cleanup
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS trades, portfolio, market_data CASCADE")
    await pool.close()

class TestDatabaseConnectionPool:
    """Test database connection pool functionality"""
    
    @pytest.mark.asyncio
    async def test_connection_acquisition_sqlite(self, sqlite_pool):
        """Test acquiring and releasing connections from SQLite pool"""
        # Acquire multiple connections
        connections = []
        for i in range(3):
            conn = await sqlite_pool.acquire()
            connections.append(conn)
            
            # Verify connection works
            result = await conn.execute("SELECT 1")
            assert result is not None
            
        # Release connections
        for conn in connections:
            await sqlite_pool._release_connection(conn)
            
        # Verify metrics
        metrics = sqlite_pool.get_metrics()
        assert metrics["connections_created"] >= 3
        assert metrics["queries_executed"] >= 3
        
    @pytest.mark.asyncio
    async def test_connection_acquisition_postgres(self, postgres_pool):
        """Test acquiring and releasing connections from PostgreSQL pool"""
        # Test concurrent acquisitions
        async def acquire_and_query():
            async with postgres_pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                assert result == 1
                await asyncio.sleep(0.1)  # Simulate work
                
        # Run concurrent queries
        tasks = [acquire_and_query() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # Check metrics
        metrics = postgres_pool.get_metrics()
        assert metrics["successful_requests"] >= 10
        assert metrics["avg_response_time_ms"] > 0
        
    @pytest.mark.asyncio
    async def test_transaction_rollback_sqlite(self, sqlite_pool):
        """Test transaction rollback on SQLite"""
        async with sqlite_pool.transaction() as conn:
            # Insert test data
            await conn.execute(
                "INSERT INTO trades (symbol, side, amount, price) VALUES (?, ?, ?, ?)",
                ("BTC-USD", "buy", 0.1, 50000.0)
            )
            
            # Verify insertion
            result = await conn.execute("SELECT COUNT(*) as count FROM trades")
            row = await result.fetchone()
            assert row[0] == 1
            
            # Force rollback
            raise Exception("Test rollback")
            
        # Verify rollback worked
        async with sqlite_pool.acquire() as conn:
            result = await conn.execute("SELECT COUNT(*) as count FROM trades")
            row = await result.fetchone()
            assert row[0] == 0
            
    @pytest.mark.asyncio
    async def test_transaction_commit_postgres(self, postgres_pool):
        """Test transaction commit on PostgreSQL"""
        # Insert with transaction
        async with postgres_pool.transaction() as conn:
            await conn.execute(
                "INSERT INTO trades (symbol, side, amount, price) VALUES ($1, $2, $3, $4)",
                "ETH-USD", "sell", 2.5, 3000.0
            )
            
        # Verify committed
        async with postgres_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM trades")
            assert count == 1
            
            # Verify data
            row = await conn.fetchrow("SELECT * FROM trades WHERE symbol = $1", "ETH-USD")
            assert row["side"] == "sell"
            assert float(row["amount"]) == 2.5
            assert float(row["price"]) == 3000.0

class TestUnifiedDatabase:
    """Test UnifiedDatabase abstraction layer"""
    
    @pytest.fixture
    async def unified_db(self):
        """Create UnifiedDatabase instance with test configuration"""
        config = DatabaseConfig(
            host="localhost",
            database=":memory:",
            connection_pool_size=5
        )
        
        db = UnifiedDatabase(config)
        await db.initialize()
        
        yield db
        
        await db.close()
        
    @pytest.mark.asyncio
    async def test_portfolio_operations(self, unified_db):
        """Test portfolio CRUD operations"""
        # Add position
        await unified_db.add_position("BTC-USD", 1.5, 45000.0)
        
        # Get position
        position = await unified_db.get_position("BTC-USD")
        assert position is not None
        assert position["symbol"] == "BTC-USD"
        assert position["amount"] == 1.5
        assert position["avg_price"] == 45000.0
        
        # Update position
        await unified_db.update_position("BTC-USD", 2.0, 46000.0)
        
        # Verify update
        position = await unified_db.get_position("BTC-USD")
        assert position["amount"] == 2.0
        assert position["avg_price"] == 46000.0
        
        # Get all positions
        positions = await unified_db.get_all_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC-USD"
        
    @pytest.mark.asyncio
    async def test_trade_logging(self, unified_db):
        """Test trade logging and retrieval"""
        # Log multiple trades
        trades = [
            {"symbol": "BTC-USD", "side": "buy", "amount": 0.5, "price": 48000.0},
            {"symbol": "ETH-USD", "side": "buy", "amount": 5.0, "price": 2800.0},
            {"symbol": "BTC-USD", "side": "sell", "amount": 0.2, "price": 49000.0}
        ]
        
        for trade in trades:
            await unified_db.log_trade(trade)
            
        # Get trades by symbol
        btc_trades = await unified_db.get_trades_by_symbol("BTC-USD")
        assert len(btc_trades) == 2
        
        # Get recent trades
        recent = await unified_db.get_recent_trades(limit=2)
        assert len(recent) == 2
        
        # Verify trade data
        buy_trade = [t for t in btc_trades if t["side"] == "buy"][0]
        assert buy_trade["amount"] == 0.5
        assert buy_trade["price"] == 48000.0
        
    @pytest.mark.asyncio
    async def test_market_data_storage(self, unified_db):
        """Test market data storage and retrieval"""
        # Store market data points
        now = datetime.utcnow()
        
        for i in range(5):
            await unified_db.store_market_data(
                symbol="BTC-USD",
                price=50000.0 + i * 100,
                volume=1000.0 + i * 50,
                timestamp=now - timedelta(minutes=i)
            )
            
        # Get latest price
        latest = await unified_db.get_latest_price("BTC-USD")
        assert latest == 50400.0  # Last price stored
        
        # Get price history
        history = await unified_db.get_price_history("BTC-USD", hours=1)
        assert len(history) == 5
        assert history[0]["price"] == 50400.0  # Most recent first
        
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, unified_db):
        """Test concurrent database operations"""
        async def add_trade(index: int):
            await unified_db.log_trade({
                "symbol": f"TEST-{index}",
                "side": "buy" if index % 2 == 0 else "sell",
                "amount": float(index),
                "price": 1000.0 * index
            })
            
        # Run concurrent inserts
        tasks = [add_trade(i) for i in range(20)]
        await asyncio.gather(*tasks)
        
        # Verify all inserted
        all_trades = await unified_db.get_recent_trades(limit=50)
        assert len(all_trades) >= 20
        
        # Test concurrent reads
        async def read_trades():
            trades = await unified_db.get_recent_trades(limit=10)
            assert len(trades) > 0
            return trades
            
        read_tasks = [read_trades() for _ in range(10)]
        results = await asyncio.gather(*read_tasks)
        
        # All reads should succeed
        assert all(len(r) > 0 for r in results)

class TestDatabaseClient:
    """Test high-level database client"""
    
    @pytest.fixture
    async def db_client(self, unified_db):
        """Create database client with test database"""
        client = DatabaseClient(unified_db)
        
        # Add test data
        await unified_db.add_position("BTC-USD", 1.0, 50000.0)
        await unified_db.add_position("ETH-USD", 10.0, 3000.0)
        
        yield client
        
    @pytest.mark.asyncio
    async def test_portfolio_summary(self, db_client):
        """Test portfolio summary calculation"""
        summary = await db_client.get_portfolio_summary()
        
        assert "total_value_usd" in summary
        assert "holdings" in summary
        assert len(summary["holdings"]) == 2
        
        # Verify calculations
        btc_holding = [h for h in summary["holdings"] if h["symbol"] == "BTC-USD"][0]
        assert btc_holding["amount"] == 1.0
        assert btc_holding["value"] == 50000.0
        
    @pytest.mark.asyncio  
    async def test_risk_metrics_calculation(self, db_client):
        """Test risk metrics calculation"""
        # Add some trades for history
        trades = [
            {"symbol": "BTC-USD", "side": "buy", "amount": 0.5, "price": 49000.0},
            {"symbol": "BTC-USD", "side": "buy", "amount": 0.5, "price": 51000.0}
        ]
        
        for trade in trades:
            await db_client.db.log_trade(trade)
            
        # Calculate risk metrics
        metrics = await db_client.calculate_risk_metrics()
        
        assert "portfolio_volatility" in metrics
        assert "value_at_risk" in metrics
        assert "sharpe_ratio" in metrics
        assert metrics["portfolio_volatility"] >= 0
        
    @pytest.mark.asyncio
    async def test_performance_tracking(self, db_client):
        """Test performance tracking over time"""
        # Add historical data
        now = datetime.utcnow()
        
        for days_ago in range(7):
            timestamp = now - timedelta(days=days_ago)
            
            # Simulate price changes
            btc_price = 50000.0 * (1 + 0.02 * (days_ago % 3 - 1))
            eth_price = 3000.0 * (1 + 0.03 * (days_ago % 3 - 1))
            
            await db_client.db.store_market_data("BTC-USD", btc_price, 1000.0, timestamp)
            await db_client.db.store_market_data("ETH-USD", eth_price, 500.0, timestamp)
            
        # Get performance history
        history = await db_client.get_portfolio_history(days=7)
        
        assert len(history) > 0
        assert all("timestamp" in h and "total_value" in h for h in history)
        
        # Verify values change over time
        values = [h["total_value"] for h in history]
        assert len(set(values)) > 1  # Values should vary