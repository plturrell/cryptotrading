"""
Database models for rex.com trading platform
"""

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    api_key = Column(String(100), unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    # Trading relationships removed - now an ML platform


class AIAnalysis(Base):
    __tablename__ = "ai_analyses"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    model = Column(String(50), nullable=False)  # grok4, perplexity
    analysis_type = Column(String(50), nullable=False)  # signal, news, market
    signal = Column(String(20))  # BUY/SELL/HOLD
    confidence = Column(Float)
    analysis = Column(Text, nullable=False)
    raw_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    price = Column(Float, nullable=False)
    volume_24h = Column(Float)
    high_24h = Column(Float)
    low_24h = Column(Float)
    change_24h = Column(Float)
    change_percent_24h = Column(Float)
    market_cap = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AggregatedMarketData(Base):
    __tablename__ = "aggregated_market_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    network = Column(String(50))
    avg_price = Column(Float, nullable=False)
    median_price = Column(Float)
    min_price = Column(Float)
    max_price = Column(Float)
    std_dev = Column(Float)
    volume_24h = Column(Float)
    sources_count = Column(Integer)
    raw_data = Column(Text)  # JSON string of all source data
    timestamp = Column(DateTime, default=datetime.utcnow)


class MarketDataSource(Base):
    __tablename__ = "market_data_sources"

    id = Column(Integer, primary_key=True)
    source = Column(String(50), nullable=False)  # geckoterminal, coingecko, etc
    symbol = Column(String(100), nullable=False)
    price = Column(Float)
    volume_24h = Column(Float)
    market_cap = Column(Float)
    liquidity = Column(Float)
    change_1h = Column(Float)
    change_24h = Column(Float)
    change_7d = Column(Float)
    data_type = Column(String(20))  # dex, cex, aggregated
    timestamp = Column(DateTime, default=datetime.utcnow)


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String(100), unique=True, nullable=False)
    agent_type = Column(String(50), nullable=False)  # a2a_coordinator, trading_agent, etc
    context_summary = Column(Text)
    preferences = Column(Text)  # JSON string of user preferences
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User")
    messages = relationship(
        "ConversationMessage", back_populates="session", cascade="all, delete-orphan"
    )
    contexts = relationship("AgentContext", back_populates="session", cascade="all, delete-orphan")


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), ForeignKey("conversation_sessions.session_id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    message_metadata = Column(Text)  # JSON string for additional data
    token_count = Column(Integer)
    embedding = Column(Text)  # JSON array of embeddings for semantic search
    importance_score = Column(Float, default=0.5)  # 0.0-1.0 importance for retention
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ConversationSession", back_populates="messages")


class AgentContext(Base):
    __tablename__ = "agent_contexts"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), ForeignKey("conversation_sessions.session_id"), nullable=False)
    agent_id = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)
    context_type = Column(String(50), nullable=False)  # working_memory, goals, knowledge
    context_data = Column(Text, nullable=False)  # JSON string of context
    version = Column(Integer, default=1)
    active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    session = relationship("ConversationSession", back_populates="contexts")


class MemoryFragment(Base):
    __tablename__ = "memory_fragments"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    fragment_type = Column(String(50), nullable=False)  # preference, fact, pattern, outcome
    content = Column(Text, nullable=False)
    context = Column(Text)  # Additional context
    embedding = Column(Text)  # JSON array of embeddings
    relevance_score = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")


class SemanticMemory(Base):
    __tablename__ = "semantic_memory"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    memory_type = Column(String(50), nullable=False)  # episodic, semantic, procedural
    content = Column(Text, nullable=False)
    keywords = Column(Text)  # Space-separated keywords for quick search
    embedding = Column(Text)  # JSON array of embeddings
    associated_symbols = Column(Text)  # Trading symbols this memory relates to
    confidence = Column(Float, default=0.5)
    reinforcement_count = Column(Integer, default=1)
    last_reinforced = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")


class A2AAgent(Base):
    """Persistent storage for A2A agent registry"""

    __tablename__ = "a2a_agents"

    id = Column(Integer, primary_key=True)
    agent_id = Column(String(100), unique=True, nullable=False)
    agent_type = Column(String(50), nullable=False)
    capabilities = Column(Text, nullable=False)  # JSON array
    config = Column(Text)  # JSON configuration
    status = Column(String(20), default="active")
    blockchain_address = Column(String(100))
    api_key_hash = Column(String(100))
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_heartbeat = Column(DateTime)
    is_active = Column(Boolean, default=True)


class A2AConnection(Base):
    """Persistent storage for agent connections"""

    __tablename__ = "a2a_connections"

    id = Column(Integer, primary_key=True)
    connection_id = Column(String(200), unique=True, nullable=False)
    agent1_id = Column(String(100), ForeignKey("a2a_agents.agent_id"), nullable=False)
    agent2_id = Column(String(100), ForeignKey("a2a_agents.agent_id"), nullable=False)
    protocol = Column(String(50), nullable=False)
    status = Column(String(20), default="active")
    established_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)


class A2AWorkflow(Base):
    """Persistent storage for workflow definitions"""

    __tablename__ = "a2a_workflows"

    id = Column(Integer, primary_key=True)
    workflow_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    definition = Column(Text, nullable=False)  # JSON workflow definition
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class A2AWorkflowExecution(Base):
    """Track workflow executions"""

    __tablename__ = "a2a_workflow_executions"

    id = Column(Integer, primary_key=True)
    execution_id = Column(String(100), unique=True, nullable=False)
    workflow_id = Column(String(100), ForeignKey("a2a_workflows.workflow_id"), nullable=False)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    input_data = Column(Text)  # JSON input data
    result_data = Column(Text)  # JSON result data
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    created_by = Column(String(100))


class A2AMessage(Base):
    """Message log for A2A communications"""

    __tablename__ = "a2a_messages"

    id = Column(Integer, primary_key=True)
    message_id = Column(String(100), unique=True, nullable=False)
    sender_id = Column(String(100), nullable=False)
    receiver_id = Column(String(100), nullable=False)
    message_type = Column(String(50), nullable=False)
    payload = Column(Text)  # JSON payload
    status = Column(String(20), default="sent")  # sent, delivered, processed, failed
    priority = Column(Integer, default=0)
    correlation_id = Column(String(100))
    workflow_context = Column(Text)  # JSON workflow context
    sent_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    error_message = Column(Text)


class EncryptionKeyMetadata(Base):
    """Storage for encryption key metadata (salts for key derivation)"""

    __tablename__ = "encryption_key_metadata"

    id = Column(Integer, primary_key=True)
    key_id = Column(String(100), unique=True, nullable=False)
    salt = Column(String(200), nullable=False)  # Base64 encoded salt
    algorithm = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)


# ========== GRANULAR TIME-SERIES DATA MODELS FOR 58 FACTORS ==========


class FactorFrequencyEnum(enum.Enum):
    """Enum for factor data frequencies"""

    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"


class DataSourceEnum(enum.Enum):
    """Enum for data sources"""

    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    YAHOO = "yahoo"
    COINGECKO = "coingecko"
    GLASSNODE = "glassnode"
    SANTIMENT = "santiment"
    LUNARCRUSH = "lunarcrush"
    FRED = "fred"


class TimeSeries(Base):
    """Base time-series table for all granular data with optimized indexing"""

    __tablename__ = "time_series"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    frequency = Column(Enum(FactorFrequencyEnum), nullable=False, index=True)
    source = Column(Enum(DataSourceEnum), nullable=False, index=True)

    # OHLCV data (for price-based factors)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)

    # Trading data
    trades_count = Column(Integer)
    buy_volume = Column(Float)
    sell_volume = Column(Float)
    large_trades_volume = Column(Float)  # >$100k trades

    # Order book data
    bid_price = Column(Float)
    ask_price = Column(Float)
    bid_size = Column(Float)
    ask_size = Column(Float)
    spread = Column(Float)

    # Additional metadata
    data_quality_score = Column(Float, default=1.0)
    validation_flags = Column(JSON)  # Store validation results
    raw_data = Column(JSON)  # Store additional raw data from source

    created_at = Column(DateTime, default=datetime.utcnow)

    # Optimized composite indexes for common queries
    __table_args__ = (
        Index("idx_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_symbol_frequency_timestamp", "symbol", "frequency", "timestamp"),
        Index("idx_source_symbol_timestamp", "source", "symbol", "timestamp"),
        Index("idx_timestamp_frequency", "timestamp", "frequency"),
    )


class FactorData(Base):
    """Calculated factor values with validation metadata"""

    __tablename__ = "factor_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    factor_name = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)

    # Quality metrics
    quality_score = Column(Float, default=1.0)
    confidence = Column(Float, default=1.0)
    staleness_seconds = Column(Integer, default=0)

    # Calculation metadata
    calculation_method = Column(String(50))  # 'real_time', 'batch', 'backfill'
    input_data_points = Column(Integer)  # Number of data points used
    lookback_period_hours = Column(Float)  # Actual lookback used

    # Validation results
    passed_validation = Column(Boolean, default=True)
    validation_errors = Column(JSON)
    outlier_score = Column(Float)  # Statistical outlier detection

    # Data lineage
    source_data_hash = Column(String(64))  # Hash of input data for reproducibility
    calculation_version = Column(String(20), default="1.0")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Optimized indexes
    __table_args__ = (
        Index("idx_symbol_factor_timestamp", "symbol", "factor_name", "timestamp"),
        Index("idx_factor_timestamp", "factor_name", "timestamp"),
        Index("idx_quality_timestamp", "quality_score", "timestamp"),
    )


class OnChainData(Base):
    """On-chain metrics for blockchain analysis"""

    __tablename__ = "onchain_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(Enum(DataSourceEnum), nullable=False)

    # Network metrics
    active_addresses = Column(Integer)
    transaction_count = Column(Integer)
    transaction_volume_native = Column(Float)
    transaction_volume_usd = Column(Float)
    hash_rate = Column(Float)  # For PoW chains

    # Exchange flow
    exchange_inflow = Column(Float)
    exchange_outflow = Column(Float)
    exchange_balance = Column(Float)

    # Whale activity
    whale_transactions = Column(Integer)  # >$1M transactions
    whale_volume_usd = Column(Float)

    # Network health
    network_difficulty = Column(Float)
    block_time_avg = Column(Float)
    mempool_size = Column(Integer)
    gas_price_avg = Column(Float)  # For Ethereum-like chains

    # DeFi metrics
    total_value_locked = Column(Float)
    staking_ratio = Column(Float)

    # Data quality
    data_completeness = Column(Float, default=1.0)
    last_updated = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_symbol_timestamp_onchain", "symbol", "timestamp"),
        Index("idx_source_timestamp_onchain", "source", "timestamp"),
    )


class SentimentData(Base):
    """Social sentiment and fear/greed metrics"""

    __tablename__ = "sentiment_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(Enum(DataSourceEnum), nullable=False)

    # Social metrics
    social_volume = Column(Integer)  # Mention count
    social_sentiment = Column(Float)  # -1 to 1
    social_dominance = Column(Float)  # Relative mention share

    # Platform-specific metrics
    twitter_mentions = Column(Integer)
    twitter_sentiment = Column(Float)
    reddit_mentions = Column(Integer)
    reddit_sentiment = Column(Float)
    telegram_mentions = Column(Integer)

    # Sentiment indicators
    fear_greed_index = Column(Integer)  # 0-100
    market_sentiment = Column(String(20))  # extreme_fear, fear, neutral, greed, extreme_greed

    # Quality metrics
    sample_size = Column(Integer)
    confidence_score = Column(Float)

    __table_args__ = (Index("idx_symbol_timestamp_sentiment", "symbol", "timestamp"),)


class MacroData(Base):
    """Macroeconomic indicators and correlations"""

    __tablename__ = "macro_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)  # Crypto symbol
    timestamp = Column(DateTime, nullable=False, index=True)

    # Traditional market correlations
    spy_correlation = Column(Float)  # S&P 500 correlation
    dxy_correlation = Column(Float)  # Dollar index correlation
    gold_correlation = Column(Float)  # Gold correlation
    vix_correlation = Column(Float)  # Volatility index correlation

    # Interest rates and bonds
    us_10y_yield = Column(Float)
    yield_curve_2y10y = Column(Float)
    real_rates = Column(Float)

    # Economic indicators
    cpi_yoy = Column(Float)  # Inflation
    unemployment_rate = Column(Float)
    gdp_growth = Column(Float)

    # Monetary policy
    fed_funds_rate = Column(Float)
    money_supply_m2_growth = Column(Float)

    # Calculation metadata
    correlation_window_days = Column(Integer, default=30)
    data_quality = Column(Float, default=1.0)

    __table_args__ = (Index("idx_symbol_timestamp_macro", "symbol", "timestamp"),)


class DataQualityMetrics(Base):
    """Track data quality metrics for monitoring and alerting"""

    __tablename__ = "data_quality_metrics"

    id = Column(Integer, primary_key=True)
    source = Column(Enum(DataSourceEnum), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    factor_name = Column(String(100), index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Quality scores
    completeness_score = Column(Float)  # % of expected data points present
    accuracy_score = Column(Float)  # Deviation from expected ranges
    consistency_score = Column(Float)  # Cross-source consistency
    timeliness_score = Column(Float)  # Data freshness
    overall_quality_score = Column(Float)

    # Specific metrics
    missing_data_points = Column(Integer)
    outlier_count = Column(Integer)
    validation_failures = Column(Integer)
    cross_source_deviation = Column(Float)
    staleness_minutes = Column(Float)

    # Validation details
    failed_rules = Column(JSON)  # List of failed validation rules
    recommendations = Column(JSON)  # Suggested fixes

    # Monitoring
    alert_threshold_breached = Column(Boolean, default=False)
    alert_sent = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_source_symbol_quality", "source", "symbol", "timestamp"),
        Index("idx_quality_score_timestamp", "overall_quality_score", "timestamp"),
    )


class DataIngestionJob(Base):
    """Track data ingestion jobs and their status"""

    __tablename__ = "data_ingestion_jobs"

    id = Column(Integer, primary_key=True)
    job_id = Column(String(100), unique=True, nullable=False)
    job_type = Column(
        String(50), nullable=False
    )  # 'historical_backfill', 'real_time', 'factor_calculation'
    symbol = Column(String(20), nullable=False, index=True)
    source = Column(Enum(DataSourceEnum), nullable=False)

    # Job parameters
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    frequency = Column(Enum(FactorFrequencyEnum))
    factors_requested = Column(JSON)  # List of factor names

    # Status tracking
    status = Column(
        String(20), default="pending", index=True
    )  # pending, running, completed, failed
    progress_percentage = Column(Float, default=0.0)
    records_processed = Column(Integer, default=0)
    records_total = Column(Integer)

    # Results
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    validation_failures = Column(Integer, default=0)
    quality_issues = Column(Integer, default=0)

    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    estimated_completion = Column(DateTime)

    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Worker information
    worker_id = Column(String(100))
    priority = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_status_priority", "status", "priority"),
        Index("idx_symbol_job_type", "symbol", "job_type"),
    )
