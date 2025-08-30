"""
Enhanced Database Module with Enterprise Features
Supports SQLite (local) and PostgreSQL/Vercel (production)
"""

# Use enhanced unified database as the primary interface
from ...infrastructure.database.unified_database import DatabaseConfig, UnifiedDatabase
from .client import close_db, get_database, get_db, initialize_database, reset_db
from .health_monitor import DatabaseHealthMonitor, HealthStatus
from .migrations import DatabaseMigrator, Migration, get_migrations
from .models import (
    A2AAgent,
    A2AConnection,
    A2AMessage,
    A2AWorkflow,
    A2AWorkflowExecution,
    AgentContext,
    AggregatedMarketData,
    AIAnalysis,
    Base,
    ConversationMessage,
    ConversationSession,
    DataIngestionJob,
    DataQualityMetrics,
    EncryptionKeyMetadata,
    FactorData,
    MacroData,
    MarketData,
    MarketDataSource,
    MemoryFragment,
    OnChainData,
    SemanticMemory,
    SentimentData,
    TimeSeries,
    User,
)
from .query_optimizer import QueryOptimizer, optimize_query
from .validation import (
    ConstraintEnforcer,
    DataQualityMonitor,
    DataValidator,
    ValidationError,
    range_constraint,
    unique_constraint,
    validate_model,
)

# from .cache import CacheManager, cache_set, cache_get, cache_delete
# from .backup import DatabaseBackup
# from .transactions import TransactionManager  # Comment out if not available

__all__ = [
    # Database (Enhanced Unified)
    "UnifiedDatabase",
    "DatabaseConfig",
    # Client functions
    "get_db",
    "close_db",
    "reset_db",
    "get_database",
    "initialize_database",
    # Models
    "Base",
    "User",
    "AIAnalysis",
    "ConversationSession",
    "ConversationMessage",
    "AgentContext",
    "MemoryFragment",
    "SemanticMemory",
    "MarketData",
    "AggregatedMarketData",
    "MarketDataSource",
    "A2AAgent",
    "A2AConnection",
    "A2AWorkflow",
    "A2AWorkflowExecution",
    "A2AMessage",
    "EncryptionKeyMetadata",
    "TimeSeries",
    "FactorData",
    "OnChainData",
    "SentimentData",
    "MacroData",
    "DataQualityMetrics",
    "DataIngestionJob",
    # Migrations
    "DatabaseMigrator",
    "Migration",
    "get_migrations",
    # Performance
    "QueryOptimizer",
    "optimize_query",
    # Health & Monitoring
    "DatabaseHealthMonitor",
    "HealthStatus",
    # Validation
    "DataValidator",
    "ConstraintEnforcer",
    "DataQualityMonitor",
    "ValidationError",
    "validate_model",
    "unique_constraint",
    "range_constraint",
    # Features - temporarily disabled due to dependencies
    # 'CacheManager', 'cache_set', 'cache_get', 'cache_delete', 'DatabaseBackup'
]
