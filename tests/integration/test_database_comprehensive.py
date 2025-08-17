"""
Comprehensive Database Integration Tests
Tests all database functionality end-to-end with real scenarios
"""

import pytest
import time
import threading
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os

from src.cryptotrading.data.database import (
    UnifiedDatabaseClient, get_unified_db, close_unified_db,
    ValidationError, User, AIAnalysis, MarketData, A2AAgent,
    ConversationSession, ConversationMessage
)

class TestDatabaseIntegration:
    """Comprehensive database integration tests"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        db_url = f'sqlite:///{temp_file.name}'
        client = UnifiedDatabaseClient(db_url)
        
        yield client
        
        # Cleanup
        client.close()
        os.unlink(temp_file.name)
    
    def test_full_user_lifecycle(self, temp_db):
        """Test complete user lifecycle operations"""
        db = temp_db
        
        # Create user
        user_id = db.create(User,
            username="testuser",
            email="test@example.com", 
            password_hash="hashed_password"
        )
        assert user_id is not None
        
        # Read user
        user = db.get_by_id(User, user_id)
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active == True
        
        # Update user
        success = db.update(User, user_id,
            email="newemail@example.com",
            is_active=False
        )
        assert success == True
        
        # Verify update
        updated_user = db.get_by_id(User, user_id)
        assert updated_user.email == "newemail@example.com"
        assert updated_user.is_active == False
        
        # Delete user
        success = db.delete(User, user_id)
        assert success == True
        
        # Verify deletion
        deleted_user = db.get_by_id(User, user_id)
        assert deleted_user is None
    
    def test_data_validation_integration(self, temp_db):
        """Test data validation in real scenarios"""
        db = temp_db
        
        # Test valid email validation
        user_id = db.create(User,
            username="validuser",
            email="valid@example.com",
            password_hash="hash123"
        )
        assert user_id is not None
        
        # Test invalid email validation
        with pytest.raises(ValidationError) as exc_info:
            db.create(User,
                username="invaliduser", 
                email="invalid-email",
                password_hash="hash123"
            )
        assert "email" in str(exc_info.value).lower()
        
        # Test duplicate username constraint
        with pytest.raises(ValidationError) as exc_info:
            db.create(User,
                username="validuser",  # Duplicate
                email="another@example.com",
                password_hash="hash456"
            )
        assert "username" in str(exc_info.value).lower()
    
    def test_concurrent_operations(self, temp_db):
        """Test thread safety with concurrent operations"""
        db = temp_db
        
        def create_users(start_id, count):
            """Create multiple users concurrently"""
            results = []
            for i in range(count):
                try:
                    user_id = db.create(User,
                        username=f"user_{start_id}_{i}",
                        email=f"user_{start_id}_{i}@example.com",
                        password_hash=f"hash_{start_id}_{i}"
                    )
                    results.append(user_id)
                except Exception as e:
                    results.append(f"Error: {e}")
            return results
        
        # Create 50 users across 5 threads (10 each)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for thread_id in range(5):
                future = executor.submit(create_users, thread_id, 10)
                futures.append(future)
            
            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        # Verify all users were created successfully
        successful_creates = [r for r in all_results if isinstance(r, int)]
        failed_creates = [r for r in all_results if isinstance(r, str)]
        
        print(f"Successful creates: {len(successful_creates)}")
        print(f"Failed creates: {len(failed_creates)}")
        
        # Should have high success rate
        assert len(successful_creates) >= 45  # Allow some failures due to concurrency
        
        # Verify users exist in database
        for user_id in successful_creates[:10]:  # Check first 10
            user = db.get_by_id(User, user_id)
            assert user is not None
    
    def test_transaction_integrity(self, temp_db):
        """Test transaction rollback and integrity"""
        db = temp_db
        
        # Create initial user
        user_id = db.create(User,
            username="transactionuser",
            email="trans@example.com",
            password_hash="hash123"
        )
        
        # Test that failed operations don't corrupt database
        initial_count = len(db.execute_query("SELECT * FROM users"))
        
        # Attempt to create user with duplicate username (should fail)
        try:
            db.create(User,
                username="transactionuser",  # Duplicate
                email="different@example.com", 
                password_hash="hash456"
            )
        except ValidationError:
            pass  # Expected
        
        # Verify database state unchanged
        final_count = len(db.execute_query("SELECT * FROM users"))
        assert final_count == initial_count
        
        # Verify original user still exists and unchanged
        user = db.get_by_id(User, user_id)
        assert user.username == "transactionuser"
        assert user.email == "trans@example.com"
    
    def test_complex_relationships(self, temp_db):
        """Test complex model relationships and foreign keys"""
        db = temp_db
        
        # Create user
        user_id = db.create(User,
            username="relationuser",
            email="relation@example.com",
            password_hash="hash123"
        )
        
        # Create conversation session
        session_id = db.create(ConversationSession,
            user_id=user_id,
            session_id="test_session_123",
            agent_type="trading_agent",
            context_summary="Test conversation"
        )
        
        # Create conversation messages
        message_ids = []
        for i in range(5):
            msg_id = db.create(ConversationMessage,
                session_id="test_session_123",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Test message {i}",
                token_count=10 + i
            )
            message_ids.append(msg_id)
        
        # Create AI analysis
        analysis_id = db.create(AIAnalysis,
            symbol="BTC",
            model="gpt-4",
            analysis_type="signal",
            signal="BUY",
            confidence=0.85,
            analysis="Test analysis content"
        )
        
        # Verify all relationships work
        session = db.get_by_id(ConversationSession, session_id)
        assert session.user_id == user_id
        
        analysis = db.get_by_id(AIAnalysis, analysis_id)
        assert analysis.symbol == "BTC"
        assert analysis.confidence == 0.85
        
        # Test cascade deletion would work (don't actually delete)
        user = db.get_by_id(User, user_id)
        assert user is not None
    
    @pytest.mark.asyncio
    async def test_async_operations(self, temp_db):
        """Test asynchronous database operations"""
        db = temp_db
        
        # Test async create
        user_id = await db.async_create(User,
            username="asyncuser",
            email="async@example.com",
            password_hash="asynchash"
        )
        assert user_id is not None
        
        # Test async read
        user = await db.async_get_by_id(User, user_id)
        assert user.username == "asyncuser"
        
        # Test async update
        success = await db.async_update(User, user_id,
            email="newemail@example.com"
        )
        assert success == True
        
        # Test async delete
        success = await db.async_delete(User, user_id)
        assert success == True
        
        # Verify deletion
        deleted_user = await db.async_get_by_id(User, user_id)
        assert deleted_user is None
    
    def test_migration_system_integration(self, temp_db):
        """Test migration system with real scenarios"""
        db = temp_db
        
        # Check initial migration status
        status = db.migrator.get_status()
        assert 'applied_count' in status
        assert 'pending_count' in status
        
        # Run any pending migrations
        if status['pending_count'] > 0:
            result = db.run_migrations()
            assert len(result['failed']) == 0
        
        # Validate migration integrity
        validation = db.migrator.validate_migrations()
        assert validation['valid'] == True
        assert len(validation['issues']) == 0
    
    def test_query_optimization_integration(self, temp_db):
        """Test query optimization in real scenarios"""
        db = temp_db
        
        # Create test data
        for i in range(10):
            db.create(User,
                username=f"perfuser_{i}",
                email=f"perf_{i}@example.com",
                password_hash=f"hash_{i}"
            )
        
        # Test query with monitoring
        test_query = "SELECT COUNT(*) FROM users WHERE is_active = ?"
        
        start_time = time.time()
        with db.query_optimizer.monitor_query(test_query):
            result = db.execute_query(test_query, (True,))
        execution_time = (time.time() - start_time) * 1000
        
        assert len(result) == 1
        assert result[0][0] == 10  # 10 users created
        assert execution_time < 100  # Should be fast
        
        # Test query analysis
        analysis = db.query_optimizer.analyze_query(test_query)
        assert 'issues' in analysis
        assert 'suggestions' in analysis
    
    def test_health_monitoring_integration(self, temp_db):
        """Test health monitoring with real operations"""
        db = temp_db
        
        # Perform some operations to generate metrics
        for i in range(5):
            user_id = db.create(User,
                username=f"healthuser_{i}",
                email=f"health_{i}@example.com",
                password_hash=f"hash_{i}"
            )
            user = db.get_by_id(User, user_id)
            assert user is not None
        
        # Run health checks
        health_results = db.health_monitor.run_health_checks()
        
        # Verify health check structure
        assert 'overall_status' in health_results
        assert 'checks' in health_results
        assert health_results['overall_status'] in ['healthy', 'degraded', 'unhealthy']
        
        # Check specific health checks
        checks = health_results['checks']
        assert 'connection' in checks
        assert 'response_time' in checks
        
        # Connection should be healthy
        assert checks['connection']['status'] == 'healthy'
        
        # Get health summary
        summary = db.get_health_status()
        assert 'status' in summary
        assert 'last_check' in summary
    
    def test_cache_integration(self, temp_db):
        """Test cache integration if available"""
        # This test would need cache system to be integrated
        # For now, verify cache components exist
        assert hasattr(temp_db, '_sqlite_pool')
        if temp_db.is_sqlite:
            assert temp_db._sqlite_pool is not None
    
    def test_performance_under_load(self, temp_db):
        """Test performance under load"""
        db = temp_db
        
        # Measure performance of batch operations
        start_time = time.time()
        
        # Create 100 users
        user_ids = []
        for i in range(100):
            user_id = db.create(User,
                username=f"loaduser_{i}",
                email=f"load_{i}@example.com",
                password_hash=f"hash_{i}"
            )
            user_ids.append(user_id)
        
        create_time = time.time() - start_time
        
        # Read all users
        start_time = time.time()
        for user_id in user_ids:
            user = db.get_by_id(User, user_id)
            assert user is not None
        
        read_time = time.time() - start_time
        
        # Performance assertions
        assert create_time < 5.0  # 100 creates in under 5 seconds
        assert read_time < 2.0    # 100 reads in under 2 seconds
        
        print(f"Performance: {len(user_ids)} creates in {create_time:.2f}s, reads in {read_time:.2f}s")
    
    def test_error_handling_integration(self, temp_db):
        """Test comprehensive error handling"""
        db = temp_db
        
        # Test invalid model data
        with pytest.raises(ValidationError):
            db.create(User,
                username="",  # Invalid empty username
                email="test@example.com",
                password_hash="hash"
            )
        
        # Test non-existent record operations
        assert db.get_by_id(User, 99999) is None
        assert db.update(User, 99999, username="newname") == False
        assert db.delete(User, 99999) == False
        
        # Test malformed queries
        with pytest.raises(Exception):
            db.execute_query("INVALID SQL QUERY")
    
    def test_data_quality_integration(self, temp_db):
        """Test data quality monitoring"""
        db = temp_db
        
        # Create test data with some quality issues
        # Good user
        db.create(User,
            username="gooduser",
            email="good@example.com",
            password_hash="hash123"
        )
        
        # Create AI analysis
        db.create(AIAnalysis,
            symbol="BTC",
            model="gpt-4",
            analysis_type="signal",
            analysis="Good analysis",
            confidence=0.95
        )
        
        # Run data quality checks
        quality_results = db.run_data_quality_checks()
        
        # Verify quality check structure
        assert 'overall_score' in quality_results
        assert 'checks' in quality_results
        assert 'issues' in quality_results
        
        # Should have reasonable quality score
        assert quality_results['overall_score'] >= 70.0
    
    def test_full_system_integration(self, temp_db):
        """Test all systems working together"""
        db = temp_db
        
        print("\n=== Full System Integration Test ===")
        
        # 1. Run migrations
        migration_result = db.run_migrations()
        print(f"✓ Migrations: {len(migration_result['applied'])} applied")
        
        # 2. Create test data
        user_id = db.create(User,
            username="systemuser",
            email="system@example.com",
            password_hash="systemhash"
        )
        print(f"✓ User created: {user_id}")
        
        # 3. Test health
        health = db.get_health_status()
        print(f"✓ Health: {health['status']}")
        assert health['status'] != 'unhealthy'
        
        # 4. Test performance
        perf_report = db.get_performance_report(hours=1)
        print(f"✓ Performance: {perf_report['total_queries']} queries tracked")
        
        # 5. Test data quality
        quality = db.run_data_quality_checks()
        print(f"✓ Data Quality: {quality['overall_score']:.1f}%")
        
        # 6. Test pool status
        pool_status = db.get_pool_status()
        print(f"✓ Pool: {pool_status['total']} connections")
        
        print("✓ All systems integrated successfully")


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])


if __name__ == "__main__":
    run_comprehensive_tests()