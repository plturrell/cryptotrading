"""
Extended database models for system monitoring, caching, and management tables
These models complement the core models.py file
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

# ========== SYSTEM MONITORING TABLES ==========

class SystemHealth(Base):
    """System health monitoring metrics"""
    __tablename__ = 'system_health'
    
    id = Column(Integer, primary_key=True)
    component = Column(String(100), nullable=False, index=True)  # api, database, mcp, agents
    status = Column(String(20), nullable=False)  # healthy, degraded, down
    uptime_seconds = Column(Integer)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    network_latency = Column(Float)
    error_rate = Column(Float)
    response_time_ms = Column(Float)
    active_connections = Column(Integer)
    queue_depth = Column(Integer)
    last_error = Column(Text)
    metadata = Column(JSON)
    checked_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_system_health_component_time', 'component', 'checked_at'),
        Index('idx_system_health_status', 'status'),
    )


class SystemMetrics(Base):
    """Detailed system performance metrics"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    value = Column(Float, nullable=False)
    unit = Column(String(20))  # ms, bytes, percentage, count
    component = Column(String(100))
    tags = Column(JSON)  # Additional metric tags
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_metrics_name_time', 'metric_name', 'timestamp'),
        Index('idx_metrics_component', 'component'),
    )


class MonitoringEvents(Base):
    """System monitoring events and alerts"""
    __tablename__ = 'monitoring_events'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False, index=True)  # alert, warning, info
    severity = Column(String(20), nullable=False)  # critical, high, medium, low
    component = Column(String(100), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    metric_value = Column(Float)
    threshold_value = Column(Float)
    action_taken = Column(Text)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_events_type_severity', 'event_type', 'severity'),
        Index('idx_events_resolved', 'resolved'),
    )


class ErrorLogs(Base):
    """Centralized error logging"""
    __tablename__ = 'error_logs'
    
    id = Column(Integer, primary_key=True)
    error_id = Column(String(100), unique=True, nullable=False)
    error_type = Column(String(100), nullable=False, index=True)
    error_message = Column(Text, nullable=False)
    stack_trace = Column(Text)
    component = Column(String(100), nullable=False, index=True)
    function_name = Column(String(200))
    file_path = Column(String(500))
    line_number = Column(Integer)
    user_id = Column(Integer, ForeignKey('users.id'))
    session_id = Column(String(100))
    request_data = Column(JSON)
    context_data = Column(JSON)
    severity = Column(String(20), default='error')  # debug, info, warning, error, critical
    retry_count = Column(Integer, default=0)
    resolved = Column(Boolean, default=False)
    resolution = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_errors_type_component', 'error_type', 'component'),
        Index('idx_errors_severity_time', 'severity', 'created_at'),
    )


# ========== CACHING TABLES ==========

class CacheEntries(Base):
    """General purpose cache storage"""
    __tablename__ = 'cache_entries'
    
    id = Column(Integer, primary_key=True)
    cache_key = Column(String(200), unique=True, nullable=False, index=True)
    cache_type = Column(String(50), nullable=False)  # api_response, calculation, query_result
    value = Column(Text, nullable=False)  # JSON serialized value
    ttl_seconds = Column(Integer, default=3600)
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    expires_at = Column(DateTime, nullable=False, index=True)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_cache_type_expires', 'cache_type', 'expires_at'),
    )


class FeatureCache(Base):
    """ML feature caching for performance"""
    __tablename__ = 'feature_cache'
    
    id = Column(Integer, primary_key=True)
    feature_key = Column(String(200), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Float)
    feature_vector = Column(JSON)  # For multi-dimensional features
    calculation_time_ms = Column(Float)
    data_version = Column(String(20))
    valid_until = Column(DateTime, nullable=False, index=True)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_feature_symbol_name', 'symbol', 'feature_name'),
    )


class HistoricalDataCache(Base):
    """Cache for expensive historical data queries"""
    __tablename__ = 'historical_data_cache'
    
    id = Column(Integer, primary_key=True)
    query_hash = Column(String(64), unique=True, nullable=False, index=True)
    query_params = Column(JSON, nullable=False)
    symbol = Column(String(20), index=True)
    data_type = Column(String(50), nullable=False)  # ohlcv, indicators, sentiment
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    frequency = Column(String(10))
    result_data = Column(Text, nullable=False)  # Compressed JSON
    result_count = Column(Integer)
    compression_ratio = Column(Float)
    query_time_ms = Column(Float)
    expires_at = Column(DateTime, nullable=False, index=True)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_historical_symbol_type', 'symbol', 'data_type'),
    )


# ========== CODE MANAGEMENT TABLES ==========

class CodeFiles(Base):
    """Track code files for analysis and quality monitoring"""
    __tablename__ = 'code_files'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), unique=True, nullable=False, index=True)
    file_name = Column(String(200), nullable=False, index=True)
    file_type = Column(String(20))  # py, js, ts, yaml, json
    file_size = Column(Integer)
    line_count = Column(Integer)
    language = Column(String(50))
    module_name = Column(String(200))
    class_count = Column(Integer)
    function_count = Column(Integer)
    complexity_score = Column(Float)
    test_coverage = Column(Float)
    last_modified = Column(DateTime)
    hash_sha256 = Column(String(64))
    dependencies = Column(JSON)  # List of imports/dependencies
    metadata = Column(JSON)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_code_type_name', 'file_type', 'file_name'),
    )


class CodeMetrics(Base):
    """Code quality and performance metrics"""
    __tablename__ = 'code_metrics'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('code_files.id'))
    metric_type = Column(String(50), nullable=False)  # complexity, coverage, performance
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    baseline_value = Column(Float)
    threshold_value = Column(Float)
    status = Column(String(20))  # pass, warning, fail
    details = Column(JSON)
    measured_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_metrics_file_type', 'file_id', 'metric_type'),
    )


class Issues(Base):
    """Issue tracking for code quality and bugs"""
    __tablename__ = 'issues'
    
    id = Column(Integer, primary_key=True)
    issue_id = Column(String(100), unique=True, nullable=False)
    issue_type = Column(String(50), nullable=False)  # bug, enhancement, security, performance
    severity = Column(String(20), nullable=False)  # critical, high, medium, low
    status = Column(String(20), default='open')  # open, in_progress, resolved, closed
    title = Column(String(500), nullable=False)
    description = Column(Text)
    file_path = Column(String(500))
    line_number = Column(Integer)
    code_snippet = Column(Text)
    suggested_fix = Column(Text)
    assigned_to = Column(String(100))
    labels = Column(JSON)  # List of labels
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_issues_type_status', 'issue_type', 'status'),
        Index('idx_issues_severity', 'severity'),
    )


# ========== AGENT & MEMORY TABLES ==========

class AgentMemory(Base):
    """Extended memory storage for agents"""
    __tablename__ = 'agent_memory'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(100), nullable=False, index=True)
    memory_type = Column(String(50), nullable=False)  # short_term, long_term, episodic
    memory_key = Column(String(200), nullable=False)
    memory_value = Column(Text, nullable=False)
    importance = Column(Float, default=0.5)
    access_frequency = Column(Integer, default=0)
    decay_rate = Column(Float, default=0.1)
    current_strength = Column(Float, default=1.0)
    associations = Column(JSON)  # Related memory IDs
    context = Column(JSON)
    last_accessed = Column(DateTime)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_agent_memory_key', 'agent_id', 'memory_key'),
        Index('idx_memory_importance', 'importance'),
    )


class ConversationHistory(Base):
    """Detailed conversation history with context"""
    __tablename__ = 'conversation_history'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(100), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    agent_id = Column(String(100))
    message_type = Column(String(20), nullable=False)  # user, agent, system
    message = Column(Text, nullable=False)
    intent = Column(String(100))  # Detected user intent
    entities = Column(JSON)  # Extracted entities
    sentiment = Column(Float)  # -1 to 1
    confidence = Column(Float)
    context_before = Column(JSON)
    context_after = Column(JSON)
    actions_taken = Column(JSON)
    feedback = Column(String(20))  # positive, negative, neutral
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_conversation_user', 'conversation_id', 'user_id'),
    )


class KnowledgeGraph(Base):
    """Knowledge graph for semantic relationships"""
    __tablename__ = 'knowledge_graph'
    
    id = Column(Integer, primary_key=True)
    entity_type = Column(String(50), nullable=False)  # symbol, indicator, strategy, concept
    entity_id = Column(String(100), nullable=False)
    entity_name = Column(String(200), nullable=False)
    properties = Column(JSON)
    relationships = Column(JSON)  # List of related entities
    embedding = Column(JSON)  # Vector embedding
    confidence = Column(Float, default=1.0)
    source = Column(String(100))
    validated = Column(Boolean, default=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_knowledge_type_id', 'entity_type', 'entity_id'),
        Index('idx_knowledge_name', 'entity_name'),
    )


# ========== TRADING & PORTFOLIO TABLES ==========

class PortfolioPositions(Base):
    """Current portfolio positions"""
    __tablename__ = 'portfolio_positions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    unrealized_pnl = Column(Float)
    unrealized_pnl_percent = Column(Float)
    realized_pnl = Column(Float)
    cost_basis = Column(Float)
    position_type = Column(String(20))  # long, short
    leverage = Column(Float, default=1.0)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    opened_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_portfolio_user_symbol', 'user_id', 'symbol'),
    )


class TradingOrders(Base):
    """Trading order history"""
    __tablename__ = 'trading_orders'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    order_type = Column(String(20), nullable=False)  # market, limit, stop, stop_limit
    side = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    executed_price = Column(Float)
    executed_quantity = Column(Float)
    status = Column(String(20), nullable=False)  # pending, filled, partial, cancelled
    time_in_force = Column(String(20))  # GTC, IOC, FOK
    stop_price = Column(Float)
    commission = Column(Float)
    commission_asset = Column(String(20))
    exchange = Column(String(50))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_orders_user_symbol', 'user_id', 'symbol'),
        Index('idx_orders_status', 'status'),
    )


class APICredentials(Base):
    """Encrypted API credentials storage"""
    __tablename__ = 'api_credentials'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    service_name = Column(String(50), nullable=False)  # binance, coinbase, etc
    api_key_encrypted = Column(Text, nullable=False)
    api_secret_encrypted = Column(Text, nullable=False)
    passphrase_encrypted = Column(Text)  # For services that require it
    permissions = Column(JSON)  # List of granted permissions
    is_active = Column(Boolean, default=True)
    is_testnet = Column(Boolean, default=False)
    last_used = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_credentials_user_service', 'user_id', 'service_name'),
    )