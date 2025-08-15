"""
Database models for rex.com trading platform
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    api_key = Column(String(100), unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = relationship("Trade", back_populates="user")
    portfolios = relationship("Portfolio", back_populates="user")
    
class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy/sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    status = Column(String(20), default='pending')  # pending/completed/cancelled
    order_id = Column(String(100))
    exchange = Column(String(50), default='binance')
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="trades")
    
class Portfolio(Base):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    average_price = Column(Float, nullable=False)
    current_price = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    
class AIAnalysis(Base):
    __tablename__ = 'ai_analyses'
    
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
    __tablename__ = 'market_data'
    
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
    
class TradingSignal(Base):
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    signal_type = Column(String(20), nullable=False)  # BUY/SELL/HOLD
    strength = Column(Float)  # 0.0 to 1.0
    rsi = Column(Float)
    macd = Column(Float)
    ma_50 = Column(Float)
    ma_200 = Column(Float)
    volume_signal = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class DexTrade(Base):
    __tablename__ = 'dex_trades'
    
    id = Column(Integer, primary_key=True)
    source = Column(String(50), nullable=False)  # geckoterminal, bitquery
    network = Column(String(50), nullable=False)  # ethereum, bsc, polygon
    token = Column(String(100), nullable=False)
    exchange = Column(String(100))  # Uniswap, Sushiswap, etc
    base_amount = Column(Float)
    quote_amount = Column(Float)
    price = Column(Float, nullable=False)
    side = Column(String(10))  # buy/sell
    tx_hash = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)

class AggregatedMarketData(Base):
    __tablename__ = 'aggregated_market_data'
    
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
    __tablename__ = 'market_data_sources'
    
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
    __tablename__ = 'conversation_sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_id = Column(String(100), unique=True, nullable=False)
    agent_type = Column(String(50), nullable=False)  # a2a_coordinator, trading_agent, etc
    context_summary = Column(Text)
    preferences = Column(Text)  # JSON string of user preferences
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")
    contexts = relationship("AgentContext", back_populates="session", cascade="all, delete-orphan")

class ConversationMessage(Base):
    __tablename__ = 'conversation_messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), ForeignKey('conversation_sessions.session_id'), nullable=False)
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
    __tablename__ = 'agent_contexts'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), ForeignKey('conversation_sessions.session_id'), nullable=False)
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
    __tablename__ = 'memory_fragments'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
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
    __tablename__ = 'semantic_memory'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
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
    __tablename__ = 'a2a_agents'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(100), unique=True, nullable=False)
    agent_type = Column(String(50), nullable=False)
    capabilities = Column(Text, nullable=False)  # JSON array
    config = Column(Text)  # JSON configuration
    status = Column(String(20), default='active')
    blockchain_address = Column(String(100))
    api_key_hash = Column(String(100))
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_heartbeat = Column(DateTime)
    is_active = Column(Boolean, default=True)

class A2AConnection(Base):
    """Persistent storage for agent connections"""
    __tablename__ = 'a2a_connections'
    
    id = Column(Integer, primary_key=True)
    connection_id = Column(String(200), unique=True, nullable=False)
    agent1_id = Column(String(100), ForeignKey('a2a_agents.agent_id'), nullable=False)
    agent2_id = Column(String(100), ForeignKey('a2a_agents.agent_id'), nullable=False)
    protocol = Column(String(50), nullable=False)
    status = Column(String(20), default='active')
    established_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)

class A2AWorkflow(Base):
    """Persistent storage for workflow definitions"""
    __tablename__ = 'a2a_workflows'
    
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
    __tablename__ = 'a2a_workflow_executions'
    
    id = Column(Integer, primary_key=True)
    execution_id = Column(String(100), unique=True, nullable=False)
    workflow_id = Column(String(100), ForeignKey('a2a_workflows.workflow_id'), nullable=False)
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    input_data = Column(Text)  # JSON input data
    result_data = Column(Text)  # JSON result data
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    created_by = Column(String(100))

class A2AMessage(Base):
    """Message log for A2A communications"""
    __tablename__ = 'a2a_messages'
    
    id = Column(Integer, primary_key=True)
    message_id = Column(String(100), unique=True, nullable=False)
    sender_id = Column(String(100), nullable=False)
    receiver_id = Column(String(100), nullable=False)
    message_type = Column(String(50), nullable=False)
    payload = Column(Text)  # JSON payload
    status = Column(String(20), default='sent')  # sent, delivered, processed, failed
    priority = Column(Integer, default=0)
    correlation_id = Column(String(100))
    workflow_context = Column(Text)  # JSON workflow context
    sent_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    error_message = Column(Text)

class EncryptionKeyMetadata(Base):
    """Storage for encryption key metadata (salts for key derivation)"""
    __tablename__ = 'encryption_key_metadata'
    
    id = Column(Integer, primary_key=True)
    key_id = Column(String(100), unique=True, nullable=False)
    salt = Column(String(200), nullable=False)  # Base64 encoded salt
    algorithm = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)