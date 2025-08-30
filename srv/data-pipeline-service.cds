using { com.rex.cryptotrading.datapipeline as pipeline } from './data-pipeline-model';

namespace com.rex.cryptotrading.datapipeline.service;

/**
 * Data Pipeline Service - RESTful API for Data Pipeline Management
 */
@path: '/api/odata/v4/DataPipelineService'
service DataPipelineService {
    
    // Core Pipeline Entities
    @odata.draft.enabled
    @cds.redirection.target
    entity DataIngestionJobs as projection on pipeline.DataIngestionJobs;
    
    entity DataQualityMetrics as projection on pipeline.DataQualityMetrics;
    
    entity MarketDataSources as projection on pipeline.MarketDataSources;
    
    entity AggregatedMarketData as projection on pipeline.AggregatedMarketData;
    
    @odata.draft.enabled
    entity MLModelRegistry as projection on pipeline.MLModelRegistry;
    
    entity OnchainData as projection on pipeline.OnchainData;
    
    entity AIAnalyses as projection on pipeline.AIAnalyses;
    
    // Analytics Views
    @readonly
    entity ActiveDataJobs as projection on pipeline.ActiveDataJobs;
    
    @readonly
    entity DataQualityDashboard as projection on pipeline.DataQualityDashboard;
    
    @readonly
    entity ModelPerformance as projection on pipeline.ModelPerformance;
    
    // Pipeline Management Actions
    action startIngestionJob(
        jobName: String,
        source: String,
        destination: String
    ) returns {
        jobId: String;
        status: String;
        estimatedTime: Integer;
    };
    
    action stopIngestionJob(
        jobId: String,
        reason: String
    ) returns {
        success: Boolean;
        recordsProcessed: Integer;
        message: String;
    };
    
    action validateDataQuality(
        dataSource: String,
        tableName: String
    ) returns {
        qualityScore: Decimal;
        issues: array of {
            metricName: String;
            status: String;
            value: Decimal;
            threshold: Decimal;
        };
    };
    
    action deployModel(
        modelId: String,
        targetEnvironment: String
    ) returns {
        deploymentId: String;
        status: String;
        endpoint: String;
    };
    
    action syncMarketData(
        sourceId: String,
        symbols: array of String
    ) returns {
        recordsSynced: Integer;
        errors: Integer;
        nextSync: DateTime;
    };
    
    // Data Query Functions
    function getJobStatus(jobId: String) returns {
        status: String;
        progress: Decimal;
        recordsProcessed: Integer;
        estimatedCompletion: DateTime;
        errors: array of String;
    };
    
    function getDataQualityReport(
        period: String
    ) returns {
        overallScore: Decimal;
        bySource: array of {
            source: String;
            score: Decimal;
            passRate: Decimal;
            issues: Integer;
        };
        trends: array of {
            date: Date;
            score: Decimal;
        };
    };
    
    function getModelMetrics(
        modelId: String
    ) returns {
        accuracy: Decimal;
        precision: Decimal;
        recall: Decimal;
        f1Score: Decimal;
        latency: Integer;
        throughput: Integer;
        lastPrediction: DateTime;
    };
    
    function getOnchainStats(
        chainName: String,
        period: String
    ) returns {
        totalTransactions: Integer;
        totalVolume: Decimal;
        avgGasPrice: Decimal;
        uniqueAddresses: Integer;
        topContracts: array of {
            address: String;
            transactions: Integer;
            volume: Decimal;
        };
    };
}