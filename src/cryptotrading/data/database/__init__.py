"""
Enhanced Database Module with Enterprise Features
Supports SQLite (local) and PostgreSQL/Vercel (production)
"""

# Use enhanced unified database as the primary interface  
from ...infrastructure.database.unified_database import UnifiedDatabase, DatabaseConfig
from .client import get_db, close_db, reset_db, get_database, initialize_database
from .models import (
    Base, User, AIAnalysis, ConversationSession, 
    ConversationMessage, AgentContext, MemoryFragment, SemanticMemory,
    MarketData, AggregatedMarketData, MarketDataSource, 
    A2AAgent, A2AConnection, A2AWorkflow, A2AWorkflowExecution, A2AMessage,
    EncryptionKeyMetadata, TimeSeries, FactorData, OnChainData, SentimentData,
    MacroData, DataQualityMetrics, DataIngestionJob
)
from .migrations import DatabaseMigrator, Migration, get_migrations
from .query_optimizer import QueryOptimizer, optimize_query
from .health_monitor import DatabaseHealthMonitor, HealthStatus
from .validation import (
    DataValidator, ConstraintEnforcer, DataQualityMonitor,
    ValidationError, validate_model, unique_constraint, range_constraint
)
# from .cache import CacheManager, cache_set, cache_get, cache_delete
# from .backup import DatabaseBackup
# from .transactions import TransactionManager  # Comment out if not available

__all__ = [
    # Database (Enhanced Unified)
    'UnifiedDatabase', 'DatabaseConfig',
    
    # Client functions
    'get_db', 'close_db', 'reset_db', 'get_database', 'initialize_database',
    
    # Models
    'Base', 'User', 'AIAnalysis',
    'ConversationSession', 'ConversationMessage', 'AgentContext', 
    'MemoryFragment', 'SemanticMemory', 'MarketData',
    'AggregatedMarketData', 'MarketDataSource',
    'A2AAgent', 'A2AConnection', 'A2AWorkflow', 'A2AWorkflowExecution',
    'A2AMessage', 'EncryptionKeyMetadata', 'TimeSeries', 'FactorData',
    'OnChainData', 'SentimentData', 'MacroData', 'DataQualityMetrics',
    'DataIngestionJob',
    
    # Migrations
    'DatabaseMigrator', 'Migration', 'get_migrations',
    
    # Performance
    'QueryOptimizer', 'optimize_query',
    
    # Health & Monitoring
    'DatabaseHealthMonitor', 'HealthStatus',
    
    # Validation
    'DataValidator', 'ConstraintEnforcer', 'DataQualityMonitor',
    'ValidationError', 'validate_model', 'unique_constraint', 'range_constraint',
    
    # Features - temporarily disabled due to dependencies
    # 'CacheManager', 'cache_set', 'cache_get', 'cache_delete', 'DatabaseBackup'
]