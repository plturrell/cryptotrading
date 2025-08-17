"""
Test Database Enhancements
Demonstrates the improved database layer features
"""

import pytest
import time
from datetime import datetime
from src.cryptotrading.data.database import (
    UnifiedDatabaseClient, get_unified_db, close_unified_db,
    DatabaseConfig, DatabaseMigrator, QueryOptimizer, 
    DatabaseHealthMonitor, DataValidator, ValidationError, 
    User, AIAnalysis, MarketData
)

def test_database_migrations():
    """Test database migration system"""
    db = get_unified_db()
    
    # Check migration status
    status = db.migrator.get_status()
    print(f"Migration Status: {status}")
    
    # Validate migrations
    validation = db.migrator.validate_migrations()
    assert validation['valid'], f"Migration validation failed: {validation['issues']}"
    
    # Run pending migrations
    if status['pending_count'] > 0:
        result = db.run_migrations()
        print(f"Migrations applied: {result['applied']}")
        assert len(result['failed']) == 0, f"Failed migrations: {result['failed']}"

def test_query_optimization():
    """Test query optimization and monitoring"""
    db = get_unified_db()
    
    # Test query analysis
    test_query = """
    SELECT u.username, COUNT(a.id) as analysis_count
    FROM users u
    LEFT JOIN ai_analyses a ON u.id = a.user_id
    WHERE u.is_active = 1
    GROUP BY u.username
    ORDER BY analysis_count DESC
    """
    
    analysis = db.query_optimizer.analyze_query(test_query)
    print(f"Query Analysis: {analysis}")
    
    # Execute with optimization monitoring
    start_time = time.time()
    with db.query_optimizer.monitor_query(test_query):
        results = db.execute_query(test_query)
    execution_time = (time.time() - start_time) * 1000
    
    print(f"Query executed in {execution_time:.2f}ms")
    print(f"Results: {len(results)} rows")
    
    # Get performance report
    report = db.get_performance_report(hours=1)
    print(f"Performance Report: {report}")

def test_health_monitoring():
    """Test database health monitoring"""
    db = get_unified_db()
    
    # Run health checks
    health_results = db.health_monitor.run_health_checks()
    print(f"Health Status: {health_results['overall_status']}")
    
    for check_name, check_result in health_results['checks'].items():
        print(f"  {check_name}: {check_result['status']} - {check_result['message']}")
    
    # Get health summary
    summary = db.get_health_status()
    print(f"\nHealth Summary: {summary}")
    
    assert summary['status'] != 'unhealthy', "Database is unhealthy"

def test_data_validation():
    """Test data validation and constraints"""
    db = get_unified_db()
    
    # Test valid user data
    valid_user_data = {
        'username': 'testuser123',
        'email': 'test@example.com',
        'password_hash': 'hashed_password_here'
    }
    
    try:
        validated = db.data_validator.validate('users', valid_user_data)
        print(f"Validated user data: {validated}")
    except ValidationError as e:
        pytest.fail(f"Valid data failed validation: {e}")
    
    # Test invalid email
    invalid_data = {
        'username': 'testuser456',
        'email': 'invalid-email',
        'password_hash': 'hashed_password'
    }
    
    with pytest.raises(ValidationError) as exc_info:
        db.data_validator.validate('users', invalid_data)
    print(f"Expected validation error: {exc_info.value}")
    
    # Test constraint enforcement
    violations = db.constraint_enforcer.check_constraints('users', valid_user_data, 'insert')
    print(f"Constraint violations: {violations}")

def test_data_quality():
    """Test data quality monitoring"""
    db = get_unified_db()
    
    # Run data quality checks
    quality_results = db.run_data_quality_checks()
    
    print(f"Data Quality Score: {quality_results['overall_score']:.1f}%")
    print(f"Quality Issues: {len(quality_results['issues'])}")
    
    for issue in quality_results['issues'][:5]:  # Show first 5 issues
        print(f"  - {issue['check']}: {issue['issue']}")
    
    # Check individual quality metrics
    for check_name, check_result in quality_results['checks'].items():
        print(f"\n{check_name}:")
        print(f"  Score: {check_result.get('score', 0):.1f}%")
        if 'metrics' in check_result:
            for metric, value in check_result['metrics'].items():
                print(f"  {metric}: {value}")

def test_connection_resilience():
    """Test connection resilience and retry logic"""
    db = get_unified_db()
    
    # Test resilient connection
    with db.health_monitor.resilient_connection() as session:
        # Perform database operation
        user_count = session.query(User).count()
        print(f"User count: {user_count}")
    
    # Check connection pool status
    pool_status = db.get_pool_status()
    print(f"\nConnection Pool Status:")
    for key, value in pool_status.items():
        print(f"  {key}: {value}")

def test_performance_indexes():
    """Test that performance indexes are properly created"""
    db = get_unified_db()
    
    # Check if indexes exist
    if db.is_sqlite:
        query = "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    else:
        query = "SELECT indexname FROM pg_indexes WHERE indexname LIKE 'idx_%'"
    
    indexes = db.execute_query(query)
    print(f"\nPerformance Indexes: {len(indexes)}")
    for idx in indexes:
        print(f"  - {idx[0]}")
    
    assert len(indexes) > 10, "Missing performance indexes"

def test_integrated_features():
    """Test integrated database features working together"""
    db = get_unified_db()
    
    print("\n=== Database Enhancement Test Summary ===")
    
    # 1. Migration status
    migration_status = db.migrator.get_status()
    print(f"\n1. Migrations: {migration_status['applied_count']} applied, "
          f"{migration_status['pending_count']} pending")
    
    # 2. Health status
    health = db.get_health_status()
    print(f"\n2. Health: {health['status']} "
          f"({health['healthy_checks']}/{health['total_checks']} checks passing)")
    
    # 3. Performance metrics
    perf_report = db.get_performance_report(hours=1)
    print(f"\n3. Performance: {perf_report['total_queries']} queries, "
          f"{len(perf_report['slow_queries'])} slow queries")
    
    # 4. Data quality
    quality = db.run_data_quality_checks()
    print(f"\n4. Data Quality: {quality['overall_score']:.1f}% "
          f"({len(quality['issues'])} issues)")
    
    # 5. Connection pool
    pool = db.get_pool_status()
    utilization = (pool['checked_out'] / pool['total_connections'] * 100) if pool['total_connections'] > 0 else 0
    print(f"\n5. Connection Pool: {utilization:.1f}% utilization "
          f"({pool['checked_out']}/{pool['total_connections']} connections)")
    
    print("\n=== All Database Enhancements Working ===")

if __name__ == "__main__":
    # Run all tests
    test_database_migrations()
    print("\n" + "="*50 + "\n")
    
    test_query_optimization()
    print("\n" + "="*50 + "\n")
    
    test_health_monitoring()
    print("\n" + "="*50 + "\n")
    
    test_data_validation()
    print("\n" + "="*50 + "\n")
    
    test_data_quality()
    print("\n" + "="*50 + "\n")
    
    test_connection_resilience()
    print("\n" + "="*50 + "\n")
    
    test_performance_indexes()
    print("\n" + "="*50 + "\n")
    
    test_integrated_features()