# Database to CDS Service Coverage Audit

## 📊 Complete Coverage Matrix

### ✅ FULLY COVERED (Database → CDS Model → CDS Service)

| Database Table | CDS Model Entity | CDS Service | Service Path |
|----------------|------------------|-------------|--------------|
| **ai_insights** | AIInsights | IntelligenceService | `/api/odata/v4/IntelligenceService` |
| **trading_decisions** | TradingDecisions | IntelligenceService | `/api/odata/v4/IntelligenceService` |
| **ml_predictions** | MLPredictions | IntelligenceService | `/api/odata/v4/IntelligenceService` |
| **agent_memory** | AgentMemory | IntelligenceService | `/api/odata/v4/IntelligenceService` |
| **knowledge_graph** | KnowledgeGraph | IntelligenceService | `/api/odata/v4/IntelligenceService` |
| **decision_outcomes** | DecisionOutcomes | IntelligenceService | `/api/odata/v4/IntelligenceService` |
| **code_files** | CodeFiles | CodeAnalysisService | `/api/odata/v4/CodeAnalysisService` |
| **code_metrics** | IndexerStats | CodeAnalysisService | `/api/odata/v4/CodeAnalysisService` |
| **issues** | BlindSpots | CodeAnalysisService | `/api/odata/v4/CodeAnalysisService` |
| **trading_orders** | Orders | TradingService | `/api/odata/v4/TradingService` |
| **portfolio_positions** | Holdings | TradingService | `/api/odata/v4/TradingService` |
| **market_data** | MarketData | TradingService | `/api/odata/v4/TradingService` |

### ⚠️ MISSING CDS COVERAGE (Database tables without CDS models)

| Database Table | Suggested CDS Model | Priority |
|----------------|-------------------|----------|
| **a2a_agents** | A2AAgents | High |
| **a2a_connections** | A2AConnections | High |
| **a2a_messages** | A2AMessages | High |
| **a2a_workflows** | A2AWorkflows | High |
| **a2a_workflow_executions** | A2AWorkflowExecutions | High |
| **conversation_messages** | ConversationMessages | Medium |
| **conversation_sessions** | ConversationSessions | Medium |
| **conversation_history** | ConversationHistory | Medium |
| **data_ingestion_jobs** | DataIngestionJobs | Medium |
| **data_quality_metrics** | DataQualityMetrics | Medium |
| **encryption_key_metadata** | EncryptionKeyMetadata | Low |
| **error_logs** | ErrorLogs | Medium |
| **factor_data** | FactorData | High |
| **factor_definitions** | FactorDefinitions | High |
| **feature_cache** | FeatureCache | Low |
| **historical_data_cache** | HistoricalDataCache | Low |
| **cache_entries** | CacheEntries | Low |
| **macro_data** | MacroData | Medium |
| **market_data_sources** | MarketDataSources | Medium |
| **memory_fragments** | MemoryFragments | Medium |
| **ml_model_registry** | MLModelRegistry | High |
| **monitoring_events** | MonitoringEvents | Medium |
| **onchain_data** | OnchainData | High |
| **semantic_memory** | SemanticMemory | Medium |
| **sentiment_data** | SentimentData | Medium |
| **system_health** | SystemHealth | Medium |
| **system_metrics** | SystemMetrics | Medium |
| **time_series** | TimeSeries | High |
| **users** | Users | High |
| **aggregated_market_data** | AggregatedMarketData | Medium |
| **api_credentials** | APICredentials | Low |
| **agent_contexts** | AgentContexts | Medium |
| **ai_analyses** | AIAnalyses | High |

## 📈 CDS Entity Coverage in Services

### ✅ Code Analysis Service Entities
All entities from `code-analysis-model.cds` are exposed:
- Projects ✅
- IndexingSessions ✅
- CodeFiles ✅
- AnalysisResults ✅
- IndexerStats ✅
- BlindSpots ✅
- Views: ProjectAnalytics, LanguageBreakdown, SessionProgress ✅

### ✅ Trading Service Entities
All entities from `trading-model.cds` are exposed:
- TradingPairs ✅
- Orders ✅
- OrderExecutions ✅
- PriceHistory ✅
- Portfolio ✅
- Holdings ✅
- Transactions ✅
- OrderBook ✅
- MarketData ✅
- Views: ActiveOrders, CompletedOrders, PortfolioSummary, TopCryptocurrencies ✅

### ✅ Intelligence Service Entities
All entities from `intelligence-model.cds` are exposed:
- AIInsights ✅
- TradingDecisions ✅
- MLPredictions ✅
- AgentMemory ✅
- KnowledgeGraph ✅
- DecisionOutcomes ✅
- Views: InsightPerformance, ModelAccuracy, DecisionSuccessRate ✅

## 📋 Summary Statistics

- **Total Database Tables Found**: 47
- **Tables with CDS Models**: 12 (25.5%)
- **Tables without CDS Models**: 35 (74.5%)
- **CDS Models with Services**: 21 (100% of CDS models)
- **Service Endpoints**: 3
  - `/api/odata/v4/CodeAnalysisService`
  - `/api/odata/v4/TradingService`
  - `/api/odata/v4/IntelligenceService`

## 🎯 Recommendations for Perfect Coverage

### High Priority (Core Business Logic)
1. Create `a2a-model.cds` and `a2a-service.cds` for A2A agent tables
2. Create `conversation-model.cds` and `conversation-service.cds` for conversation management
3. Create `factor-model.cds` and `factor-service.cds` for factor definitions and data
4. Create `user-model.cds` and `user-service.cds` for user management

### Medium Priority (Supporting Systems)
5. Create `monitoring-model.cds` and `monitoring-service.cds` for system monitoring
6. Create `data-pipeline-model.cds` and `data-pipeline-service.cds` for data ingestion

### Low Priority (Internal/Cache)
7. Consider if cache tables need CDS exposure (usually not needed for OData)

## ✅ Current Status
- All existing CDS models have corresponding services ✅
- All services are properly exposed with endpoints ✅
- Redirection targets are configured to avoid ambiguity ✅
- Build completes successfully without errors ✅

## 🚀 Next Steps to Achieve Perfect Coverage
1. Prioritize creating CDS models for the 35 missing tables based on business needs
2. Group related tables into logical service domains
3. Consider which tables actually need OData/REST exposure vs internal-only access
4. Some cache and system tables may not need CDS services if they're internal-only