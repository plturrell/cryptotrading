using { com.rex.cryptotrading.intelligence as intel } from './intelligence-model';

namespace com.rex.cryptotrading.intelligence.service;

/**
 * AI Intelligence Service - RESTful API for AI/ML Trading Intelligence
 */
@path: '/api/odata/v4/IntelligenceService'
service IntelligenceService {
    
    // Core Intelligence Entities
    @odata.draft.enabled
    @cds.redirection.target
    entity AIInsights as projection on intel.AIInsights;
    
    @cds.redirection.target
    entity TradingDecisions as projection on intel.TradingDecisions;
    
    entity MLPredictions as projection on intel.MLPredictions;
    
    entity AgentMemory as projection on intel.AgentMemory;
    
    entity KnowledgeGraph as projection on intel.KnowledgeGraph;
    
    entity DecisionOutcomes as projection on intel.DecisionOutcomes;
    
    // Analytics Views
    @readonly
    entity InsightPerformance as projection on intel.InsightPerformance;
    
    @readonly
    entity ModelAccuracy as projection on intel.ModelAccuracy;
    
    @readonly
    entity DecisionSuccessRate as projection on intel.DecisionSuccessRate;
    
    // AI Actions
    action generateInsight(
        symbol: String,
        analysisType: String,
        timeframe: String
    ) returns {
        insightId: String;
        recommendation: String;
        confidence: Decimal;
        reasoning: String;
    };
    
    action executeDecision(
        decisionId: String,
        confirmExecution: Boolean
    ) returns {
        success: Boolean;
        executionId: String;
        message: String;
    };
    
    action trainModel(
        modelType: String,
        trainingData: String,
        hyperparameters: String
    ) returns {
        modelId: String;
        trainingStatus: String;
        estimatedTime: Integer;
    };
    
    action updateAgentMemory(
        agentId: String,
        memories: array of {
            memoryKey: String;
            memoryValue: String;
            importance: Decimal;
        }
    ) returns {
        updatedCount: Integer;
        message: String;
    };
    
    // Intelligence Functions
    function getInsightHistory(
        symbol: String,
        startDate: DateTime,
        endDate: DateTime
    ) returns array of {
        timestamp: DateTime;
        insightType: String;
        recommendation: String;
        confidence: Decimal;
        outcome: String;
    };
    
    function getPredictionAccuracy(
        modelType: String,
        period: String
    ) returns {
        accuracy: Decimal;
        precision: Decimal;
        recall: Decimal;
        f1Score: Decimal;
        totalPredictions: Integer;
        correctPredictions: Integer;
    };
    
    function getDecisionAnalytics(
        algorithm: String,
        period: String
    ) returns {
        totalDecisions: Integer;
        successRate: Decimal;
        avgProfitLoss: Decimal;
        bestDecision: Decimal;
        worstDecision: Decimal;
        sharpeRatio: Decimal;
    };
    
    function queryKnowledgeGraph(
        entityType: String,
        entityId: String,
        depth: Integer
    ) returns array of {
        relationType: String;
        relatedEntity: String;
        strength: Decimal;
        path: array of String;
    };
    
    function getAgentPerformance(
        agentId: String
    ) returns {
        totalMemories: Integer;
        avgImportance: Decimal;
        recentActivity: array of {
            timestamp: DateTime;
            action: String;
            result: String;
        };
        successMetrics: {
            decisions: Integer;
            successRate: Decimal;
            profitability: Decimal;
        };
    };
    
    // Backtesting Functions
    function backtest(
        strategy: String,
        startDate: DateTime,
        endDate: DateTime,
        initialCapital: Decimal
    ) returns {
        finalCapital: Decimal;
        totalReturn: Decimal;
        annualizedReturn: Decimal;
        maxDrawdown: Decimal;
        winRate: Decimal;
        profitFactor: Decimal;
        trades: array of {
            timestamp: DateTime;
            action: String;
            symbol: String;
            quantity: Decimal;
            price: Decimal;
            pnl: Decimal;
        };
    };
    
    // Real-time Intelligence
    function getRealtimeSignals() returns array of {
        symbol: String;
        signal: String;
        strength: Decimal;
        timeframe: String;
        indicators: array of {
            name: String;
            value: Decimal;
            signal: String;
        };
    };
    
    function getMarketSentiment(symbol: String) returns {
        overall: String;
        score: Decimal;
        sources: array of {
            source: String;
            sentiment: String;
            confidence: Decimal;
        };
        newsImpact: String;
        socialMediaTrend: String;
    };
}