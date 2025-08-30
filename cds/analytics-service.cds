using { com.rex.cryptotrading.analytics as analytics } from './analytics-model';

namespace com.rex.cryptotrading.analytics.service;

/**
 * Analytics Service - RESTful API for Analytics and Time Series Data
 */
@path: '/api/odata/v4/AnalyticsService'
service AnalyticsService {
    
    // Core Analytics Entities
    @odata.draft.enabled
    @cds.redirection.target
    entity FactorDefinitions as projection on analytics.FactorDefinitions;
    
    entity FactorData as projection on analytics.FactorData;
    
    entity TimeSeries as projection on analytics.TimeSeries;
    
    entity MacroData as projection on analytics.MacroData;
    
    entity SentimentData as projection on analytics.SentimentData;
    
    entity MemoryFragments as projection on analytics.MemoryFragments;
    
    entity SemanticMemory as projection on analytics.SemanticMemory;
    
    // Analytics Views
    @readonly
    entity ActiveFactors as projection on analytics.ActiveFactors;
    
    @readonly
    entity RecentFactorData as projection on analytics.RecentFactorData;
    
    @readonly
    entity MarketSentimentSummary as projection on analytics.MarketSentimentSummary;
    
    // Analytics Actions
    action calculateFactor(
        factorId: String,
        symbol: String,
        timestamp: DateTime
    ) returns {
        value: Decimal;
        signal: String;
        confidence: Decimal;
    };
    
    action updateTimeSeries(
        seriesName: String,
        dataPoints: array of {
            timestamp: DateTime;
            value: Decimal;
            volume: Decimal;
        }
    ) returns {
        recordsAdded: Integer;
        seriesLength: Integer;
    };
    
    action analyzeSentiment(
        symbol: String,
        sources: array of String
    ) returns {
        overallSentiment: String;
        score: Decimal;
        breakdown: array of {
            source: String;
            sentiment: String;
            confidence: Decimal;
        };
    };
    
    action storeSemanticMemory(
        context: String,
        content: String,
        embedding: array of Decimal
    ) returns {
        memoryId: String;
        similarity: Decimal;
        relatedMemories: array of String;
    };
    
    // Analytics Functions
    function getFactorAnalysis(
        symbol: String,
        factors: array of String
    ) returns {
        symbol: String;
        timestamp: DateTime;
        factors: array of {
            name: String;
            value: Decimal;
            signal: String;
            weight: Decimal;
            contribution: Decimal;
        };
        compositeScore: Decimal;
        recommendation: String;
    };
    
    function getTimeSeriesData(
        seriesName: String,
        startDate: DateTime,
        endDate: DateTime,
        aggregation: String
    ) returns array of {
        timestamp: DateTime;
        value: Decimal;
        volume: Decimal;
        metadata: String;
    };
    
    function getMacroIndicators(
        country: String,
        indicators: array of String
    ) returns array of {
        indicator: String;
        value: Decimal;
        previousValue: Decimal;
        change: Decimal;
        impact: String;
        releaseDate: Date;
    };
    
    function getSentimentTrends(
        symbol: String,
        period: String
    ) returns {
        currentSentiment: Decimal;
        trend: String;
        history: array of {
            date: Date;
            sentiment: Decimal;
            volume: Integer;
        };
        drivers: array of {
            keyword: String;
            impact: Decimal;
            frequency: Integer;
        };
    };
    
    function searchSemanticMemory(
        query: String,
        context: String,
        limit: Integer
    ) returns array of {
        memoryId: String;
        content: String;
        similarity: Decimal;
        importance: Decimal;
        timestamp: DateTime;
    };
    
    function reconstructMemoryFragments(
        parentKey: String
    ) returns {
        complete: Boolean;
        data: String;
        fragments: Integer;
        totalSize: Integer;
    };
}