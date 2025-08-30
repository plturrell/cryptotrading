namespace com.rex.cryptotrading.intelligence;

using { managed, cuid } from '@sap/cds/common';

/**
 * AI Insights Entity
 * Stores AI-generated insights and recommendations
 */
entity AIInsights : managed {
    key id          : Integer @title: 'ID';
    insightType     : String(50) @title: 'Insight Type' not null;
    symbol          : String(20) @title: 'Symbol' not null;
    recommendation  : String(50) @title: 'Recommendation' not null;
    confidence      : Decimal(5,4) @title: 'Confidence Score' not null;
    score           : Decimal(10,4) @title: 'Analysis Score';
    riskLevel       : String(20) @title: 'Risk Level';
    reasoning       : LargeString @title: 'Reasoning' not null;
    source          : String(50) @title: 'AI Source' default 'grok4';
    rawResponse     : LargeString @title: 'Raw AI Response';
    sessionId       : String(50) @title: 'Session ID';
    
    // Navigation
    decisions       : Composition of many TradingDecisions on decisions.parentInsight = $self;
}

/**
 * Trading Decisions Entity
 * Tracks trading decisions made based on AI insights
 */
entity TradingDecisions : managed {
    key id          : Integer @title: 'ID';
    decisionType    : String(50) @title: 'Decision Type' not null;
    action          : String(50) @title: 'Action' not null;
    symbol          : String(20) @title: 'Symbol' not null;
    amount          : Decimal(15,8) @title: 'Amount';
    price           : Decimal(15,8) @title: 'Price';
    confidence      : Decimal(5,4) @title: 'Confidence';
    expectedValue   : Decimal(20,4) @title: 'Expected Value';
    riskScore       : Decimal(5,4) @title: 'Risk Score';
    reasoning       : LargeString @title: 'Reasoning';
    algorithm       : String(50) @title: 'Algorithm Used';
    parentInsightId : Integer @title: 'Parent Insight ID';
    status          : String(20) @title: 'Status' default 'proposed';
    executedAt      : Timestamp @title: 'Execution Time';
    sessionId       : String(50) @title: 'Session ID';
    
    // Navigation
    parentInsight   : Association to AIInsights on parentInsight.id = parentInsightId;
    outcomes        : Composition of many DecisionOutcomes on outcomes.decision = $self;
}

/**
 * ML Predictions Entity
 * Machine learning model predictions
 */
entity MLPredictions : cuid, managed {
    modelType       : String(50) @title: 'Model Type';
    symbol          : String(20) @title: 'Symbol';
    predictionType  : String(50) @title: 'Prediction Type';
    predictedValue  : Decimal(20,8) @title: 'Predicted Value';
    confidence      : Decimal(5,4) @title: 'Confidence';
    timeHorizon     : String(20) @title: 'Time Horizon';
    featuresUsed    : LargeString @title: 'Features Used (JSON)';
    modelVersion    : String(20) @title: 'Model Version';
    accuracyScore   : Decimal(5,4) @title: 'Accuracy Score';
    targetDate      : Timestamp @title: 'Target Date';
    actualValue     : Decimal(20,8) @title: 'Actual Value';
    error           : Decimal(10,4) @title: 'Prediction Error';
    sessionId       : String(50) @title: 'Session ID';
}

/**
 * Agent Memory Entity
 * Persistent memory storage for AI agents
 */
entity AgentMemory : cuid, managed {
    agentId         : String(50) @title: 'Agent ID';
    memoryKey       : String(100) @title: 'Memory Key';
    memoryValue     : LargeString @title: 'Memory Value';
    memoryType      : String(50) @title: 'Memory Type';
    importance      : Decimal(3,2) @title: 'Importance' default 0.5;
    context         : LargeString @title: 'Context';
    metadata        : LargeString @title: 'Metadata (JSON)';
    accessedAt      : Timestamp @title: 'Last Accessed';
    accessCount     : Integer @title: 'Access Count' default 1;
    expiresAt       : Timestamp @title: 'Expiration Date';
}

/**
 * Knowledge Graph Entity
 * Relationships between entities in the knowledge base
 */
entity KnowledgeGraph : cuid, managed {
    entityType      : String(50) @title: 'Entity Type';
    entityId        : String(100) @title: 'Entity ID';
    relationType    : String(50) @title: 'Relation Type';
    relatedEntityType : String(50) @title: 'Related Entity Type';
    relatedEntityId : String(100) @title: 'Related Entity ID';
    strength        : Decimal(3,2) @title: 'Relation Strength' default 1.0;
    confidence      : Decimal(3,2) @title: 'Confidence' default 1.0;
    evidence        : LargeString @title: 'Evidence';
}

/**
 * Decision Outcomes Entity
 * Tracks the outcomes of trading decisions
 */
entity DecisionOutcomes : cuid, managed {
    decision        : Association to TradingDecisions @title: 'Decision';
    outcomeType     : String(50) @title: 'Outcome Type';
    expectedOutcome : Decimal(20,4) @title: 'Expected Outcome';
    actualOutcome   : Decimal(20,4) @title: 'Actual Outcome';
    profitLoss      : Decimal(20,4) @title: 'Profit/Loss';
    executionPrice  : Decimal(15,8) @title: 'Execution Price';
    executionTime   : Timestamp @title: 'Execution Time';
    marketConditions : LargeString @title: 'Market Conditions (JSON)';
    success         : Boolean @title: 'Success';
    lessonsLearned  : LargeString @title: 'Lessons Learned';
}

// Analytics Views
view InsightPerformance as select from AIInsights {
    insightType,
    source,
    count(*) as totalInsights : Integer,
    avg(confidence) as avgConfidence : Decimal(5,4),
    sum(case when decisions.status = 'SUCCESSFUL' then 1 else 0 end) as successfulDecisions : Integer
} group by insightType, source;

view ModelAccuracy as select from MLPredictions {
    modelType,
    modelVersion,
    count(*) as totalPredictions : Integer,
    avg(confidence) as avgConfidence : Decimal(5,4),
    avg(accuracyScore) as avgAccuracy : Decimal(5,4),
    avg(error) as avgError : Decimal(10,4)
} group by modelType, modelVersion;

view DecisionSuccessRate as select from TradingDecisions {
    algorithm,
    action,
    count(*) as totalDecisions : Integer,
    sum(case when status = 'SUCCESSFUL' then 1 else 0 end) as successfulCount : Integer,
    cast(sum(case when status = 'SUCCESSFUL' then 1 else 0 end) * 100.0 / count(*) as Decimal(5,2)) as successRate : Decimal(5,2)
} group by algorithm, action;