"""
Database schema for persistent AI intelligence and memory
Stores all insights, decisions, and accumulated knowledge
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

class IntelligenceType(Enum):
    """Types of intelligence stored"""
    AI_INSIGHT = "ai_insight"
    MCTS_DECISION = "mcts_decision" 
    ML_PREDICTION = "ml_prediction"
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_EVALUATION = "strategy_evaluation"

class DecisionStatus(Enum):
    """Status of decisions"""
    PROPOSED = "proposed"
    EXECUTED = "executed"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    CANCELLED = "cancelled"

def get_intelligence_schemas() -> Dict[str, Dict[str, str]]:
    """Get database schemas for intelligence storage"""
    return {
        "ai_insights": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS ai_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    score REAL,
                    risk_level TEXT,
                    reasoning TEXT NOT NULL,
                    source TEXT DEFAULT 'grok4',
                    raw_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_ai_insights_symbol ON ai_insights(symbol);
                CREATE INDEX IF NOT EXISTS idx_ai_insights_created ON ai_insights(created_at);
                CREATE INDEX IF NOT EXISTS idx_ai_insights_session ON ai_insights(session_id);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS ai_insights (
                    id SERIAL PRIMARY KEY,
                    insight_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    score REAL,
                    risk_level TEXT,
                    reasoning TEXT NOT NULL,
                    source TEXT DEFAULT 'grok4',
                    raw_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_ai_insights_symbol ON ai_insights(symbol);
                CREATE INDEX IF NOT EXISTS idx_ai_insights_created ON ai_insights(created_at);
                CREATE INDEX IF NOT EXISTS idx_ai_insights_session ON ai_insights(session_id);
            """
        },
        
        "trading_decisions": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS trading_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    amount REAL,
                    price REAL,
                    confidence REAL,
                    expected_value REAL,
                    risk_score REAL,
                    reasoning TEXT,
                    algorithm TEXT,
                    parent_insight_id INTEGER,
                    status TEXT DEFAULT 'proposed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP,
                    session_id TEXT,
                    
                    FOREIGN KEY (parent_insight_id) REFERENCES ai_insights(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON trading_decisions(symbol);
                CREATE INDEX IF NOT EXISTS idx_decisions_status ON trading_decisions(status);
                CREATE INDEX IF NOT EXISTS idx_decisions_created ON trading_decisions(created_at);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS trading_decisions (
                    id SERIAL PRIMARY KEY,
                    decision_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    amount REAL,
                    price REAL,
                    confidence REAL,
                    expected_value REAL,
                    risk_score REAL,
                    reasoning TEXT,
                    algorithm TEXT,
                    parent_insight_id INTEGER REFERENCES ai_insights(id),
                    status TEXT DEFAULT 'proposed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP,
                    session_id TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON trading_decisions(symbol);
                CREATE INDEX IF NOT EXISTS idx_decisions_status ON trading_decisions(status);
                CREATE INDEX IF NOT EXISTS idx_decisions_created ON trading_decisions(created_at);
            """
        },
        
        "ml_predictions": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predicted_value REAL,
                    confidence REAL,
                    time_horizon TEXT,
                    features_used TEXT,
                    model_version TEXT,
                    accuracy_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    target_date TIMESTAMP,
                    actual_value REAL,
                    error REAL,
                    session_id TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON ml_predictions(symbol);
                CREATE INDEX IF NOT EXISTS idx_predictions_created ON ml_predictions(created_at);
                CREATE INDEX IF NOT EXISTS idx_predictions_target ON ml_predictions(target_date);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id SERIAL PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predicted_value REAL,
                    confidence REAL,
                    time_horizon TEXT,
                    features_used TEXT,
                    model_version TEXT,
                    accuracy_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    target_date TIMESTAMP,
                    actual_value REAL,
                    error REAL,
                    session_id TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON ml_predictions(symbol);
                CREATE INDEX IF NOT EXISTS idx_predictions_created ON ml_predictions(created_at);
                CREATE INDEX IF NOT EXISTS idx_predictions_target ON ml_predictions(target_date);
            """
        },
        
        "agent_memory": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    memory_type TEXT,
                    importance REAL DEFAULT 0.5,
                    context TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    expires_at TIMESTAMP,
                    
                    UNIQUE(agent_id, memory_key)
                );
                
                CREATE INDEX IF NOT EXISTS idx_memory_agent ON agent_memory(agent_id);
                CREATE INDEX IF NOT EXISTS idx_memory_importance ON agent_memory(importance);
                CREATE INDEX IF NOT EXISTS idx_memory_accessed ON agent_memory(accessed_at);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id SERIAL PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    memory_type TEXT,
                    importance REAL DEFAULT 0.5,
                    context TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    expires_at TIMESTAMP,
                    
                    UNIQUE(agent_id, memory_key)
                );
                
                CREATE INDEX IF NOT EXISTS idx_memory_agent ON agent_memory(agent_id);
                CREATE INDEX IF NOT EXISTS idx_memory_importance ON agent_memory(importance);
                CREATE INDEX IF NOT EXISTS idx_memory_accessed ON agent_memory(accessed_at);
            """
        },
        
        "knowledge_graph": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    related_entity_type TEXT NOT NULL,
                    related_entity_id TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    evidence TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(entity_type, entity_id, relation_type, related_entity_type, related_entity_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge_graph(entity_type, entity_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_related ON knowledge_graph(related_entity_type, related_entity_id);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    id SERIAL PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    related_entity_type TEXT NOT NULL,
                    related_entity_id TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    evidence TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(entity_type, entity_id, relation_type, related_entity_type, related_entity_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge_graph(entity_type, entity_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_related ON knowledge_graph(related_entity_type, related_entity_id);
            """
        },
        
        "decision_outcomes": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER NOT NULL,
                    outcome_type TEXT NOT NULL,
                    expected_outcome REAL,
                    actual_outcome REAL,
                    profit_loss REAL,
                    execution_price REAL,
                    execution_time TIMESTAMP,
                    market_conditions TEXT,
                    success BOOLEAN,
                    lessons_learned TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (decision_id) REFERENCES trading_decisions(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_outcomes_decision ON decision_outcomes(decision_id);
                CREATE INDEX IF NOT EXISTS idx_outcomes_success ON decision_outcomes(success);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id SERIAL PRIMARY KEY,
                    decision_id INTEGER NOT NULL REFERENCES trading_decisions(id),
                    outcome_type TEXT NOT NULL,
                    expected_outcome REAL,
                    actual_outcome REAL,
                    profit_loss REAL,
                    execution_price REAL,
                    execution_time TIMESTAMP,
                    market_conditions TEXT,
                    success BOOLEAN,
                    lessons_learned TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_outcomes_decision ON decision_outcomes(decision_id);
                CREATE INDEX IF NOT EXISTS idx_outcomes_success ON decision_outcomes(success);
            """
        },
        
        "conversation_history": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    message_content TEXT NOT NULL,
                    context TEXT,
                    parent_message_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_history(session_id);
                CREATE INDEX IF NOT EXISTS idx_conversation_agent ON conversation_history(agent_id);
                CREATE INDEX IF NOT EXISTS idx_conversation_created ON conversation_history(created_at);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    message_content TEXT NOT NULL,
                    context TEXT,
                    parent_message_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_history(session_id);
                CREATE INDEX IF NOT EXISTS idx_conversation_agent ON conversation_history(agent_id);
                CREATE INDEX IF NOT EXISTS idx_conversation_created ON conversation_history(created_at);
            """
        },
        
        "ml_model_registry": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS ml_model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    algorithm TEXT,
                    parameters TEXT,
                    training_metrics TEXT,
                    validation_metrics TEXT,
                    file_path TEXT,
                    blob_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deployed_at TIMESTAMP,
                    status TEXT DEFAULT 'trained',
                    UNIQUE(model_id, version)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_registry_type ON ml_model_registry(model_type);
                CREATE INDEX IF NOT EXISTS idx_model_registry_status ON ml_model_registry(status);
                CREATE INDEX IF NOT EXISTS idx_model_registry_created ON ml_model_registry(created_at);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS ml_model_registry (
                    id SERIAL PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    algorithm TEXT,
                    parameters JSONB,
                    training_metrics JSONB,
                    validation_metrics JSONB,
                    file_path TEXT,
                    blob_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deployed_at TIMESTAMP,
                    status TEXT DEFAULT 'trained',
                    UNIQUE(model_id, version)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_registry_type ON ml_model_registry(model_type);
                CREATE INDEX IF NOT EXISTS idx_model_registry_status ON ml_model_registry(status);
                CREATE INDEX IF NOT EXISTS idx_model_registry_created ON ml_model_registry(created_at);
            """
        },
        
        "system_metrics": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    tags TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    service_name TEXT,
                    environment TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_metrics_service ON system_metrics(service_name);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    metric_value FLOAT NOT NULL,
                    metric_type TEXT NOT NULL,
                    tags JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    service_name TEXT,
                    environment TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_metrics_service ON system_metrics(service_name);
            """
        },
        
        "feature_cache": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS feature_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    feature_vector TEXT,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    cache_key TEXT UNIQUE
                );
                
                CREATE INDEX IF NOT EXISTS idx_feature_cache_symbol ON feature_cache(symbol);
                CREATE INDEX IF NOT EXISTS idx_feature_cache_key ON feature_cache(cache_key);
                CREATE INDEX IF NOT EXISTS idx_feature_cache_expires ON feature_cache(expires_at);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS feature_cache (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value FLOAT,
                    feature_vector JSONB,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    cache_key TEXT UNIQUE
                );
                
                CREATE INDEX IF NOT EXISTS idx_feature_cache_symbol ON feature_cache(symbol);
                CREATE INDEX IF NOT EXISTS idx_feature_cache_key ON feature_cache(cache_key);
                CREATE INDEX IF NOT EXISTS idx_feature_cache_expires ON feature_cache(expires_at);
            """
        },
        
        "error_logs": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    stack_trace TEXT,
                    context TEXT,
                    user_id TEXT,
                    service_name TEXT,
                    environment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_errors_type ON error_logs(error_type);
                CREATE INDEX IF NOT EXISTS idx_errors_created ON error_logs(created_at);
                CREATE INDEX IF NOT EXISTS idx_errors_service ON error_logs(service_name);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS error_logs (
                    id SERIAL PRIMARY KEY,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    stack_trace TEXT,
                    context JSONB,
                    user_id TEXT,
                    service_name TEXT,
                    environment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_errors_type ON error_logs(error_type);
                CREATE INDEX IF NOT EXISTS idx_errors_created ON error_logs(created_at);
                CREATE INDEX IF NOT EXISTS idx_errors_service ON error_logs(service_name);
            """
        },
        
        "cache_entries": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    cache_value TEXT,
                    cache_type TEXT,
                    ttl_seconds INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries(cache_key);
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
                CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id SERIAL PRIMARY KEY,
                    cache_key TEXT UNIQUE NOT NULL,
                    cache_value TEXT,
                    cache_type TEXT,
                    ttl_seconds INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries(cache_key);
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
                CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type);
            """
        },
        
        "system_health": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms REAL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    error_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_health_component ON system_health(component);
                CREATE INDEX IF NOT EXISTS idx_health_status ON system_health(status);
                CREATE INDEX IF NOT EXISTS idx_health_checked ON system_health(checked_at);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS system_health (
                    id SERIAL PRIMARY KEY,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms FLOAT,
                    memory_usage_mb FLOAT,
                    cpu_usage_percent FLOAT,
                    error_count INTEGER DEFAULT 0,
                    metadata JSONB,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_health_component ON system_health(component);
                CREATE INDEX IF NOT EXISTS idx_health_status ON system_health(status);
                CREATE INDEX IF NOT EXISTS idx_health_checked ON system_health(checked_at);
            """
        },
        
        "api_credentials": {
            "sqlite": """
                CREATE TABLE IF NOT EXISTS api_credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT UNIQUE NOT NULL,
                    api_key_hash TEXT NOT NULL,
                    api_secret_hash TEXT,
                    permissions TEXT,
                    rate_limits TEXT,
                    last_rotated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                );
                
                CREATE INDEX IF NOT EXISTS idx_credentials_service ON api_credentials(service_name);
                CREATE INDEX IF NOT EXISTS idx_credentials_active ON api_credentials(is_active);
                CREATE INDEX IF NOT EXISTS idx_credentials_expires ON api_credentials(expires_at);
            """,
            "postgres": """
                CREATE TABLE IF NOT EXISTS api_credentials (
                    id SERIAL PRIMARY KEY,
                    service_name TEXT UNIQUE NOT NULL,
                    api_key_hash TEXT NOT NULL,
                    api_secret_hash TEXT,
                    permissions JSONB,
                    rate_limits JSONB,
                    last_rotated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT true
                );
                
                CREATE INDEX IF NOT EXISTS idx_credentials_service ON api_credentials(service_name);
                CREATE INDEX IF NOT EXISTS idx_credentials_active ON api_credentials(is_active);
                CREATE INDEX IF NOT EXISTS idx_credentials_expires ON api_credentials(expires_at);
            """
        }
    }

def get_all_intelligence_schemas() -> Dict[str, str]:
    """Get all intelligence schemas combined for unified database creation"""
    schemas = get_intelligence_schemas()
    
    # Combine all SQLite schemas
    sqlite_combined = "\n\n".join([schema["sqlite"] for schema in schemas.values()])
    
    # Combine all PostgreSQL schemas
    postgres_combined = "\n\n".join([schema["postgres"] for schema in schemas.values()])
    
    return {
        "sqlite": sqlite_combined,
        "postgres": postgres_combined
    }