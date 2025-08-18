"""
Test Database Enhancements
Demonstrates the improved database layer features
"""

import pytest
import time
import asyncio
from datetime import datetime
from sqlalchemy import text
from src.cryptotrading.data.database import (
    UnifiedDatabase, DatabaseConfig, get_db, close_db,
    DatabaseMigrator, QueryOptimizer, 
    DatabaseHealthMonitor, DataValidator, ValidationError, 
    User, AIAnalysis, MarketData
)

@pytest.fixture
async def db():
    """Get database instance for tests"""
    database = get_db()
    # Ensure it's initialized
    if database.db_conn is None:
        await database.initialize()
    yield database
    # Don't close - let the global instance remain

@pytest.mark.asyncio
async def test_database_connection(db):
    """Test basic database connection"""
    # Test simple connection
    cursor = db.db_conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1
    cursor.close()
    
    # Test SQLAlchemy session
    with db.get_session() as session:
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1

@pytest.mark.asyncio
async def test_schema_creation(db):
    """Test that all schemas are created"""
    cursor = db.db_conn.cursor()
    
    # Check tables exist
    if db.config.mode.value == 'local':
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    else:
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
    
    tables = [row[0] for row in cursor.fetchall()]
    cursor.close()
    
    # Verify essential tables exist
    essential_tables = [
        'issues', 'code_files', 'code_metrics', 'monitoring_events',
        'market_data', 'portfolio_positions', 'trading_orders', 
        'historical_data_cache'
    ]
    
    for table in essential_tables:
        assert table in tables, f"Missing table: {table}"

@pytest.mark.asyncio
async def test_market_data_operations(db):
    """Test market data storage and retrieval"""
    # Store test data
    test_data = {
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 1000000000.0,
        'timestamp': datetime.now()
    }
    
    success = await db.store_market_data('BTC-USD', test_data)
    assert success, "Failed to store market data"
    
    # Retrieve data
    latest = await db.get_latest_market_data('BTC-USD')
    assert latest is not None, "Failed to retrieve market data"
    assert latest['close'] == 50500.0

@pytest.mark.asyncio
async def test_cache_operations(db):
    """Test Redis cache operations"""
    # Set cache value
    await db.cache_set('test_key', {'data': 'test_value'}, ttl=60)
    
    # Get cache value
    cached = await db.cache_get('test_key')
    if db.redis_client:
        assert cached is not None
        assert cached['data'] == 'test_value'
    else:
        # Cache might not be available in test environment
        print("Redis cache not available")

@pytest.mark.asyncio
async def test_health_status(db):
    """Test database health monitoring"""
    health = db.get_health_status()
    
    assert 'engine_connected' in health
    assert 'simple_connection' in health
    assert 'mode' in health
    
    assert health['engine_connected'] is True
    assert health['simple_connection'] is True

@pytest.mark.asyncio
async def test_query_execution(db):
    """Test query execution methods"""
    # Test synchronous query
    results = db.execute_query("SELECT 1 as value")
    assert len(results) == 1
    assert results[0]['value'] == 1
    
    # Test async query
    results = await db.execute_query_async("SELECT 2 as value")
    assert len(results) == 1
    assert results[0]['value'] == 2

@pytest.mark.asyncio
async def test_portfolio_operations(db):
    """Test portfolio position operations"""
    # Get positions (should be empty initially)
    positions = await db.get_portfolio_positions('test_user')
    assert isinstance(positions, list)

@pytest.mark.asyncio
async def test_connection_resilience(db):
    """Test that database connections are resilient"""
    # Run multiple operations
    for i in range(5):
        cursor = db.db_conn.cursor()
        cursor.execute("SELECT ?", (i,))
        result = cursor.fetchone()
        assert result[0] == i
        cursor.close()
        
    # Verify connection is still healthy
    health = db.get_health_status()
    assert health['simple_connection'] is True

@pytest.mark.asyncio
async def test_environment_detection(db):
    """Test environment detection works correctly"""
    assert db.config.mode.value in ['local', 'production']
    
    if db.config.mode.value == 'local':
        assert 'sqlite' in db.config.sqlite_path
    else:
        assert db.config.postgres_url is not None

if __name__ == "__main__":
    # Run all tests
    asyncio.run(test_database_connection(get_db()))
    print("✓ Database connection test passed")
    
    asyncio.run(test_schema_creation(get_db()))
    print("✓ Schema creation test passed")
    
    asyncio.run(test_market_data_operations(get_db()))
    print("✓ Market data operations test passed")
    
    asyncio.run(test_cache_operations(get_db()))
    print("✓ Cache operations test passed")
    
    asyncio.run(test_health_status(get_db()))
    print("✓ Health status test passed")
    
    asyncio.run(test_query_execution(get_db()))
    print("✓ Query execution test passed")
    
    asyncio.run(test_portfolio_operations(get_db()))
    print("✓ Portfolio operations test passed")
    
    asyncio.run(test_connection_resilience(get_db()))
    print("✓ Connection resilience test passed")
    
    asyncio.run(test_environment_detection(get_db()))
    print("✓ Environment detection test passed")
    
    print("\n✅ All database enhancement tests passed!")