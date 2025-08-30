# Complete Database Schema Documentation

## Overview
This document provides comprehensive documentation of all database tables in the cryptotrading platform, including core models, extended models, and intelligence schemas.

## Database Architecture

### Database Files
1. **cryptotrading.db** - Main application database (46 tables)
2. **real_market_data.db** - High-frequency market data storage (1 table)
3. **rex.db** - Parallel/backup database (30 tables)

## Table Categories

### 1. User & Authentication Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `users` | User accounts | id, username, email, password_hash, api_key |
| `api_credentials` | Encrypted API keys | user_id, service_name, api_key_encrypted |

### 2. Market Data Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `market_data` | Current market prices | symbol, price, volume_24h, timestamp |
| `aggregated_market_data` | Cross-exchange aggregated data | symbol, avg_price, sources_count |
| `market_data_sources` | Individual source data | source, symbol, price, liquidity |
| `time_series` | Granular OHLCV data | symbol, timestamp, frequency, OHLCV |

### 3. AI & Intelligence Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `ai_analyses` | AI analysis results | symbol, model, analysis_type, signal |
| `ai_insights` | AI-generated insights | insight_type, recommendation, confidence |
| `trading_decisions` | Trading decision tracking | action, symbol, confidence, status |
| `decision_outcomes` | Decision results | decision_id, actual_outcome, profit_loss |
| `ml_predictions` | ML model predictions | model_id, prediction, confidence |
| `ml_model_registry` | ML model metadata | model_name, version, performance_metrics |

### 4. Agent & Communication Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `a2a_agents` | Agent registry | agent_id, agent_type, capabilities, status |
| `a2a_connections` | Agent connections | agent1_id, agent2_id, protocol |
| `a2a_messages` | Agent messages | sender_id, receiver_id, payload |
| `a2a_workflows` | Workflow definitions | workflow_id, definition, version |
| `a2a_workflow_executions` | Workflow runs | workflow_id, status, result_data |
| `agent_contexts` | Agent context storage | agent_id, context_type, context_data |
| `agent_memory` | Extended agent memory | agent_id, memory_type, importance |

### 5. Conversation & Memory Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `conversation_sessions` | User sessions | user_id, session_id, agent_type |
| `conversation_messages` | Chat messages | session_id, role, content, embedding |
| `conversation_history` | Detailed history | conversation_id, intent, sentiment |
| `memory_fragments` | User preferences/facts | user_id, fragment_type, content |
| `semantic_memory` | Long-term memory | user_id, memory_type, embedding |
| `knowledge_graph` | Semantic relationships | entity_type, entity_id, relationships |

### 6. Factor & Analytics Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `factor_definitions` | Factor metadata | name, category, formula, lookback_hours |
| `factor_data` | Calculated factors | symbol, factor_name, value, quality_score |
| `onchain_data` | Blockchain metrics | symbol, active_addresses, transaction_volume |
| `sentiment_data` | Social sentiment | symbol, social_volume, fear_greed_index |
| `macro_data` | Macro indicators | symbol, spy_correlation, interest_rates |

### 7. System Monitoring Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `system_health` | Component health | component, status, cpu_usage, memory_usage |
| `system_metrics` | Performance metrics | metric_name, value, component |
| `monitoring_events` | Alerts & events | event_type, severity, component |
| `error_logs` | Error tracking | error_type, stack_trace, component |

### 8. Caching Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `cache_entries` | General cache | cache_key, value, ttl_seconds, expires_at |
| `feature_cache` | ML feature cache | feature_key, symbol, feature_value |
| `historical_data_cache` | Historical queries | query_hash, result_data, expires_at |

### 9. Code Management Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `code_files` | File tracking | file_path, complexity_score, test_coverage |
| `code_metrics` | Quality metrics | file_id, metric_type, metric_value |
| `issues` | Issue tracking | issue_type, severity, status, file_path |

### 10. Trading & Portfolio Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `portfolio_positions` | Current positions | user_id, symbol, quantity, unrealized_pnl |
| `trading_orders` | Order history | order_id, symbol, order_type, status |

### 11. Data Quality & Ingestion Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `data_quality_metrics` | Quality tracking | source, symbol, quality_scores |
| `data_ingestion_jobs` | ETL job tracking | job_id, status, progress_percentage |

### 12. Security Tables
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `encryption_key_metadata` | Key management | key_id, salt, algorithm |

## Indexes

### Performance-Critical Indexes
1. **Time-series queries**: `(symbol, timestamp)`, `(symbol, frequency, timestamp)`
2. **Factor lookups**: `(symbol, factor_name, timestamp)`
3. **Agent operations**: `(agent_id)`, `(sender_id, receiver_id)`
4. **User queries**: `(user_id)`, `(session_id)`
5. **Cache lookups**: `(cache_key)`, `(expires_at)`

## Data Relationships

### Primary Relationships
```
users ─┬─> conversation_sessions -> conversation_messages
       ├─> memory_fragments
       ├─> semantic_memory
       ├─> portfolio_positions
       └─> trading_orders

a2a_agents ─┬─> a2a_connections
            ├─> a2a_messages
            └─> agent_contexts

a2a_workflows -> a2a_workflow_executions

ai_insights -> trading_decisions -> decision_outcomes
```

## Migration Path

### Current State
- All tables exist and are functional
- Three separate database files serve different purposes
- No missing critical tables

### Recommended Consolidation (Optional)
```python
# Option 1: Keep current structure (RECOMMENDED)
# - cryptotrading.db for application data
# - real_market_data.db for high-frequency data
# - rex.db as backup/parallel

# Option 2: Consolidate to two databases
# - app.db (all application tables)
# - market.db (time-series and high-frequency data)

# Option 3: Single database with table partitioning
# - main.db with partitioned time_series table
```

## Best Practices

### 1. Query Optimization
- Use composite indexes for common query patterns
- Partition time-series data by date ranges
- Archive old data to separate tables/databases

### 2. Data Integrity
- Use foreign key constraints where applicable
- Implement soft deletes for audit trails
- Regular VACUUM operations for SQLite

### 3. Backup Strategy
- Daily backups of cryptotrading.db
- Continuous replication of real_market_data.db
- Weekly full backups with transaction logs

### 4. Monitoring
- Track table sizes and growth rates
- Monitor query performance
- Alert on data quality issues

## Schema Version Control

### Current Version: 2.0.0
- Added extended_models.py for system tables
- Documented all 46 tables
- Established clear categorization

### Migration History
- v1.0.0: Initial schema with core models
- v1.5.0: Added intelligence schema tables
- v2.0.0: Complete documentation and extended models

## Maintenance Commands

```bash
# Check database integrity
sqlite3 data/cryptotrading.db "PRAGMA integrity_check"

# Vacuum and analyze
sqlite3 data/cryptotrading.db "VACUUM; ANALYZE"

# Export schema
sqlite3 data/cryptotrading.db ".schema" > schema.sql

# Backup database
sqlite3 data/cryptotrading.db ".backup backup.db"
```