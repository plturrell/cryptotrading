# Comprehensive Database Layer Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [API Reference](#api-reference)
5. [Performance & Monitoring](#performance--monitoring)
6. [Error Handling](#error-handling)
7. [Testing](#testing)
8. [Configuration](#configuration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The database layer provides a unified, production-ready interface for database operations supporting both SQLite (development) and PostgreSQL (production). It includes comprehensive error handling, performance monitoring, health checks, and data validation.

### Key Features
- ✅ **Unified API**: Single interface for SQLite and PostgreSQL
- ✅ **Thread Safety**: Proper locking and connection pooling
- ✅ **Performance Monitoring**: Real-time metrics and benchmarking
- ✅ **Error Handling**: Standardized error messages and logging
- ✅ **Data Validation**: Built-in validation and constraints
- ✅ **Health Monitoring**: Continuous health checks
- ✅ **Migration System**: Version-controlled schema changes
- ✅ **Security**: SQL injection protection and access control

## Quick Start

### Basic Usage

```python
from src.cryptotrading.data.database import get_unified_db, User, AIAnalysis

# Get database instance
db = get_unified_db()

# Create a user
user_id = db.create(User,
    username="alice",
    email="alice@example.com",
    password_hash="secure_hash"
)

# Read user
user = db.get_by_id(User, user_id)
print(f"User: {user.username} ({user.email})")

# Update user
db.update(User, user_id, email="newemail@example.com")

# Delete user
db.delete(User, user_id)
```

### Async Operations

```python
import asyncio

async def async_operations():
    # All operations have async equivalents
    user_id = await db.async_create(User,
        username="bob",
        email="bob@example.com",
        password_hash="secure_hash"
    )
    
    user = await db.async_get_by_id(User, user_id)
    await db.async_update(User, user_id, email="updated@example.com")
    await db.async_delete(User, user_id)

# Run async operations
asyncio.run(async_operations())
```

### Error Handling

```python
from src.cryptotrading.data.database import ValidationError, DatabaseError

try:
    user_id = db.create(User,
        username="",  # Invalid empty username
        email="invalid-email",  # Invalid email format
        password_hash="hash"
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Details: {e.details}")
except DatabaseError as e:
    print(f"Database error: {e}")
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                UnifiedDatabaseClient                 │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │   Migrator   │ │Query Optimizer│ │Health Monitor│ │
│  │              │ │              │ │              │ │
│  │ • Versioning │ │ • SQL Analysis│ │ • Health Chks│ │
│  │ • Rollback   │ │ • Performance │ │ • Monitoring │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │ Data Validator│ │Error Handler │ │Perf Monitor  │ │
│  │              │ │              │ │              │ │
│  │ • Validation │ │ • Std Errors │ │ • Metrics    │ │
│  │ • Constraints│ │ • Logging    │ │ • Benchmarks │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ │
├─────────────────────────────────────────────────────┤
│                SQLite Handler                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │Connection Pool│ │Write Locking │ │ Transaction  │ │
│  │              │ │              │ │  Manager     │ │
│  │ • Pool Mgmt  │ │ • Serialized │ │ • ACID       │ │
│  │ • Health     │ │ • Retry Logic│ │ • Savepoints │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ │
├─────────────────────────────────────────────────────┤
│         Database Engine (SQLAlchemy)                │
├─────────────────────────────────────────────────────┤
│  SQLite (Development)     │  PostgreSQL (Production) │
│  • Local file storage     │  • Vercel/Cloud hosted   │
│  • Write serialization    │  • Connection pooling    │
│  • WAL mode enabled       │  • Concurrent access     │
└─────────────────────────────────────────────────────┘
```

## API Reference

### Core Methods

#### `create(model_class, **data) -> int`
Create a new record with validation.

```python
user_id = db.create(User,
    username="john",
    email="john@example.com",
    password_hash="hash123"
)
```

**Parameters:**
- `model_class`: SQLAlchemy model class
- `**data`: Field values for the new record

**Returns:** ID of created record

**Raises:** `ValidationError` if data is invalid

#### `get_by_id(model_class, id: int) -> Optional[Model]`
Retrieve a record by its ID.

```python
user = db.get_by_id(User, 123)
if user:
    print(f"Found user: {user.username}")
```

#### `update(model_class, id: int, **data) -> bool`
Update an existing record.

```python
success = db.update(User, 123, email="newemail@example.com")
```

**Returns:** `True` if record was updated, `False` if not found

#### `delete(model_class, id: int) -> bool`
Delete a record.

```python
success = db.delete(User, 123)
```

**Returns:** `True` if record was deleted, `False` if not found

#### `execute_query(query: str, params: Optional[Dict] = None) -> List[Any]`
Execute raw SQL query.

```python
results = db.execute_query(
    "SELECT * FROM users WHERE is_active = ?",
    (True,)
)
```

### Management Methods

#### `run_migrations(target_version: str = None) -> Dict[str, Any]`
Run database migrations.

```python
result = db.run_migrations()
print(f"Applied: {len(result['applied'])}")
print(f"Failed: {len(result['failed'])}")
```

#### `get_health_status() -> Dict[str, Any]`
Get database health status.

```python
health = db.get_health_status()
print(f"Status: {health['status']}")
print(f"Issues: {len(health['issues'])}")
```

#### `get_performance_summary(operation: str = None, hours: int = 24) -> Dict[str, Any]`
Get performance metrics.

```python
perf = db.get_performance_summary("create", hours=1)
print(f"Avg latency: {perf['avg_latency_ms']:.2f}ms")
print(f"Success rate: {perf['success_rate']:.1f}%")
```

#### `run_performance_benchmark() -> Dict[str, Any]`
Run comprehensive performance benchmark.

```python
benchmark = db.run_performance_benchmark()
print(f"Performance grade: {benchmark['performance_summary']['performance_grade']}")
```

#### `get_comprehensive_status() -> Dict[str, Any]`
Get complete database status.

```python
status = db.get_comprehensive_status()
print(f"Database type: {status['connection']['database_type']}")
print(f"Health: {status['health']['status']}")
print(f"Error count: {status['errors']['total_errors']}")
```

## Performance & Monitoring

### Performance Metrics

The database layer automatically tracks:
- **Operation latency** (min, max, avg, p95, p99)
- **Success rates** per operation type
- **Connection pool utilization**
- **Query performance** and slow query detection
- **Error rates** and types

### Benchmarking

```python
# Run full benchmark suite
benchmark_report = db.run_performance_benchmark()

# Key metrics
summary = benchmark_report['performance_summary']
print(f"Operations/sec: {summary['average_operations_per_second']:.1f}")
print(f"Average latency: {summary['average_latency_ms']:.2f}ms")
print(f"Success rate: {summary['average_success_rate']:.1f}%")
print(f"Performance grade: {summary['performance_grade']}")
```

### Performance Targets

| Metric | Target | Excellent |
|--------|--------|-----------|
| Average latency | < 50ms | < 10ms |
| Operations/sec | > 100 | > 1000 |
| Success rate | > 99% | > 99.9% |
| Pool utilization | < 70% | < 50% |

## Error Handling

### Error Types

The database layer provides standardized error handling:

```python
from src.cryptotrading.data.database import (
    DatabaseError,           # Base error class
    DatabaseConnectionError, # Connection issues
    DatabaseValidationError, # Data validation failures
    DatabaseQueryError,      # Query execution problems
    DatabaseTransactionError,# Transaction failures
    DatabaseSecurityError    # Security violations
)
```

### Error Codes

Each error includes a standardized error code:

| Code | Type | Description |
|------|------|-------------|
| `DB_CONN_001` | Connection | Connection failed |
| `DB_VAL_001` | Validation | Data validation failed |
| `DB_QRY_001` | Query | SQL syntax error |
| `DB_TXN_001` | Transaction | Transaction failed |
| `DB_SEC_001` | Security | SQL injection attempt |

### Error Handling Best Practices

```python
from src.cryptotrading.data.database import DatabaseError, format_user_friendly_error

try:
    user_id = db.create(User, username="test", email="invalid")
except DatabaseError as e:
    # Log technical details
    logger.error(f"Database operation failed: {e}")
    logger.error(f"Error code: {e.error_code}")
    logger.error(f"Details: {e.details}")
    
    # Show user-friendly message
    user_message = format_user_friendly_error(e)
    print(f"User message: {user_message}")
```

## Testing

### Running Tests

```bash
# Run comprehensive integration tests
python -m pytest tests/integration/test_database_comprehensive.py -v

# Run specific test categories
python -m pytest tests/integration/test_database_comprehensive.py::TestDatabaseIntegration::test_concurrent_operations -v

# Run performance tests
python -m pytest tests/integration/test_database_comprehensive.py::TestDatabaseIntegration::test_performance_under_load -v
```

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end workflows
3. **Performance Tests**: Load and stress testing
4. **Concurrency Tests**: Thread safety validation
5. **Error Handling Tests**: Error scenarios
6. **Migration Tests**: Schema change validation

### Writing Tests

```python
import pytest
from src.cryptotrading.data.database import get_unified_db, User

class TestMyFeature:
    @pytest.fixture
    def db(self):
        """Get database instance for testing"""
        return get_unified_db()
    
    def test_user_creation(self, db):
        """Test user creation with validation"""
        user_id = db.create(User,
            username="testuser",
            email="test@example.com",
            password_hash="hash123"
        )
        
        user = db.get_by_id(User, user_id)
        assert user.username == "testuser"
        assert user.email == "test@example.com"
    
    def test_validation_error(self, db):
        """Test validation error handling"""
        with pytest.raises(ValidationError) as exc_info:
            db.create(User,
                username="",  # Invalid
                email="invalid-email",
                password_hash="hash"
            )
        
        assert "validation" in str(exc_info.value).lower()
```

## Configuration

### Environment Variables

```bash
# Development (SQLite)
DATABASE_PATH=data/cryptotrading.db

# Production (PostgreSQL)
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Redis cache (optional)
REDIS_URL=redis://localhost:6379

# Vercel deployment
KV_URL=redis://...
KV_REST_API_TOKEN=...
```

### Database Configuration

```python
from src.cryptotrading.data.database import UnifiedDatabaseClient

# Custom configuration
db = UnifiedDatabaseClient(
    db_url="postgresql://user:pass@localhost:5432/mydb"
)

# Auto-detection (recommended)
db = get_unified_db()
```

### Connection Pool Settings

For PostgreSQL:
- **Pool size**: 20 connections
- **Max overflow**: 30 additional connections
- **Pool timeout**: 30 seconds
- **Connection recycling**: 1 hour

For SQLite:
- **Write serialization**: Exclusive locking
- **WAL mode**: Enabled for performance
- **Cache size**: 10MB
- **Memory mapping**: 256MB

## Best Practices

### 1. Use the Unified Client

```python
# ✅ Good - Use unified client
from src.cryptotrading.data.database import get_unified_db
db = get_unified_db()

# ❌ Bad - Direct SQLAlchemy usage
from sqlalchemy import create_engine
engine = create_engine("sqlite:///db.db")
```

### 2. Handle Errors Properly

```python
# ✅ Good - Specific error handling
try:
    user_id = db.create(User, **data)
except ValidationError as e:
    handle_validation_error(e)
except DatabaseConnectionError as e:
    handle_connection_error(e)

# ❌ Bad - Generic error handling
try:
    user_id = db.create(User, **data)
except Exception as e:
    print("Something went wrong")
```

### 3. Monitor Performance

```python
# ✅ Good - Regular monitoring
health = db.get_health_status()
if health['status'] != 'healthy':
    alert_operations_team(health['issues'])

perf = db.get_performance_summary(hours=1)
if perf['avg_latency_ms'] > 100:
    investigate_slow_queries()
```

### 4. Validate Data

```python
# ✅ Good - Let the database validate
user_id = db.create(User, **data)  # Automatic validation

# ❌ Bad - Manual validation
if not data.get('email') or '@' not in data['email']:
    raise Exception("Invalid email")
```

### 5. Use Transactions for Complex Operations

```python
# ✅ Good - Use transaction context
with db.get_session() as session:
    user = User(**user_data)
    session.add(user)
    
    profile = UserProfile(user_id=user.id, **profile_data)
    session.add(profile)
    # Automatic commit on success, rollback on error

# ❌ Bad - Manual transaction management
session = db.Session()
try:
    # ... operations
    session.commit()
except:
    session.rollback()
finally:
    session.close()
```

## Troubleshooting

### Common Issues

#### 1. Connection Pool Exhausted

**Symptoms:**
```
DatabasePerformanceError: [DB_PERF_002] Connection pool exhausted
```

**Solutions:**
- Check for connection leaks in application code
- Increase pool size if needed
- Monitor connection usage patterns

```python
# Check pool status
pool_status = db.get_pool_status()
print(f"Pool utilization: {(pool_status['checked_out']/pool_status['total'])*100:.1f}%")
```

#### 2. Slow Queries

**Symptoms:**
```
DatabasePerformanceError: [DB_PERF_001] Slow query detected
```

**Solutions:**
- Check query execution plans
- Add missing indexes
- Optimize query structure

```python
# Analyze slow queries
perf_report = db.get_performance_report(hours=1)
for slow_query in perf_report['slow_queries']:
    print(f"Query: {slow_query['query']}")
    print(f"Avg time: {slow_query['avg_time_ms']}ms")
```

#### 3. SQLite Lock Errors

**Symptoms:**
```
DatabaseTransactionError: [DB_TXN_002] Database is locked
```

**Solutions:**
- The database layer handles this automatically with retry logic
- Reduce concurrent write operations if persistent
- Consider upgrading to PostgreSQL for high concurrency

#### 4. Validation Errors

**Symptoms:**
```
DatabaseValidationError: [DB_VAL_001] Data validation failed
```

**Solutions:**
- Check data format and constraints
- Review model field requirements
- Validate data before database operations

```python
# Debug validation issues
try:
    user_id = db.create(User, **data)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Details: {e.details}")
    # Fix data and retry
```

### Debugging Tools

#### 1. Health Check

```python
health = db.get_health_status()
if health['status'] != 'healthy':
    for issue in health['issues']:
        print(f"Issue: {issue['check']} - {issue['message']}")
```

#### 2. Performance Analysis

```python
# Get detailed performance metrics
status = db.get_comprehensive_status()

# Connection health
print(f"Database: {status['connection']['database_type']}")
print(f"Pool status: {status['connection']['pool_status']}")

# Performance metrics
print(f"Avg latency: {status['performance']['summary']['avg_latency_ms']:.2f}ms")
print(f"Success rate: {status['performance']['summary']['success_rate']:.1f}%")

# Error analysis
print(f"Total errors: {status['errors']['total_errors']}")
```

#### 3. Query Analysis

```python
# Analyze specific query
query = "SELECT * FROM users WHERE email = ?"
analysis = db.query_optimizer.analyze_query(query)

print(f"Issues: {analysis['issues']}")
print(f"Suggestions: {analysis['suggestions']}")
```

### Performance Tuning

#### 1. Index Optimization

```python
# Run migrations to add performance indexes
result = db.run_migrations()
print(f"Indexes added: {len(result['applied'])}")
```

#### 2. Connection Pool Tuning

For high-load applications:
```python
# Increase pool size
db.engine.pool._pool_size = 30
db.engine.pool._max_overflow = 50
```

#### 3. Query Optimization

```python
# Monitor query performance
with db.query_optimizer.monitor_query(query):
    results = db.execute_query(query, params)

# Get optimization suggestions
analysis = db.query_optimizer.analyze_query(query)
for suggestion in analysis['suggestions']:
    print(f"Suggestion: {suggestion}")
```

### Monitoring & Alerting

Set up monitoring for:

1. **Health status** - Alert if status becomes 'unhealthy'
2. **Performance metrics** - Alert if latency > 100ms
3. **Error rates** - Alert if errors increase suddenly
4. **Connection pool** - Alert if utilization > 90%

```python
def check_database_health():
    """Health check for monitoring systems"""
    try:
        status = db.get_comprehensive_status()
        
        # Check overall health
        if status['health']['status'] == 'unhealthy':
            send_alert("Database unhealthy", status['health']['issues'])
        
        # Check performance
        avg_latency = status['performance']['summary']['avg_latency_ms']
        if avg_latency > 100:
            send_alert(f"High latency: {avg_latency:.2f}ms")
        
        # Check errors
        error_count = status['errors']['total_errors']
        if error_count > 100:
            send_alert(f"High error count: {error_count}")
        
        return True
    except Exception as e:
        send_alert(f"Health check failed: {e}")
        return False
```

---

## Summary

The database layer provides a robust, production-ready foundation with:

✅ **Reliability**: Thread-safe operations, error handling, health monitoring
✅ **Performance**: Connection pooling, query optimization, performance monitoring  
✅ **Security**: SQL injection protection, data validation, access control
✅ **Maintainability**: Standardized errors, comprehensive logging, documentation
✅ **Scalability**: Connection pooling, async support, benchmarking tools

For additional support or questions, refer to the integration tests in `tests/integration/` for real-world usage examples.