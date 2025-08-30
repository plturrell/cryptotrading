namespace com.rex.cryptotrading.datapipeline;

using { managed, cuid } from '@sap/cds/common';

/**
 * Data Ingestion Jobs Entity
 * Manages data ingestion pipeline jobs
 */
entity DataIngestionJobs : managed {
    key id          : Integer @title: 'ID';
    jobId           : String(100) @title: 'Job ID' not null;
    jobType         : String(50) @title: 'Job Type' not null;
    symbol          : String(20) @title: 'Symbol' not null;
    source          : String(10) @title: 'Source' not null;
    startDate       : DateTime @title: 'Start Date';
    endDate         : DateTime @title: 'End Date';
    frequency       : String(14) @title: 'Frequency';
    factorsRequested: LargeString @title: 'Factors Requested (JSON)';
    status          : String(20) @title: 'Status';
    progressPercentage: Decimal(5,2) @title: 'Progress %';
    recordsProcessed: Integer @title: 'Records Processed';
    recordsTotal    : Integer @title: 'Records Total';
    recordsInserted : Integer @title: 'Records Inserted';
    recordsUpdated  : Integer @title: 'Records Updated';
    validationFailures: Integer @title: 'Validation Failures';
    qualityIssues   : Integer @title: 'Quality Issues';
    startedAt       : Timestamp @title: 'Started At';
    completedAt     : Timestamp @title: 'Completed At';
    estimatedCompletion: Timestamp @title: 'Estimated Completion';
    errorMessage    : LargeString @title: 'Error Message';
    retryCount      : Integer @title: 'Retry Count';
    maxRetries      : Integer @title: 'Max Retries';
    workerId        : String(100) @title: 'Worker ID';
    priority        : Integer @title: 'Priority';
    updatedAt       : Timestamp @title: 'Updated At';
}

/**
 * Data Quality Metrics Entity
 * Tracks data quality across the pipeline
 */
entity DataQualityMetrics : cuid, managed {
    dataSource      : String(100) @title: 'Data Source';
    tableName       : String(100) @title: 'Table Name';
    metricType      : String(50) @title: 'Metric Type';
    metricName      : String(100) @title: 'Metric Name';
    metricValue     : Decimal(20,4) @title: 'Metric Value';
    threshold       : Decimal(20,4) @title: 'Threshold';
    status          : String(20) @title: 'Status' @assert.range enum {
        PASS;
        WARNING;
        FAIL;
    };
    checkTime       : Timestamp @title: 'Check Time';
    details         : LargeString @title: 'Details (JSON)';
    recordCount     : Integer @title: 'Record Count';
    nullCount       : Integer @title: 'Null Count';
    duplicateCount  : Integer @title: 'Duplicate Count';
    anomalyCount    : Integer @title: 'Anomaly Count';
}

/**
 * Market Data Sources Entity
 * Configuration for market data sources
 */
entity MarketDataSources : cuid, managed {
    sourceName      : String(100) @title: 'Source Name';
    sourceType      : String(50) @title: 'Source Type';
    apiEndpoint     : String(500) @title: 'API Endpoint';
    authentication  : String(50) @title: 'Auth Type';
    rateLimit       : Integer @title: 'Rate Limit (req/min)';
    dataTypes       : LargeString @title: 'Data Types (JSON)';
    symbols         : LargeString @title: 'Symbols (JSON)';
    isActive        : Boolean @title: 'Is Active' default true;
    priority        : Integer @title: 'Priority' default 5;
    lastSync        : Timestamp @title: 'Last Sync';
    syncFrequency   : Integer @title: 'Sync Frequency (seconds)';
    errorCount      : Integer @title: 'Error Count' default 0;
    successRate     : Decimal(5,2) @title: 'Success Rate %';
}

/**
 * Aggregated Market Data Entity
 * Pre-aggregated market data for performance
 */
entity AggregatedMarketData : cuid, managed {
    symbol          : String(20) @title: 'Symbol';
    timeframe       : String(20) @title: 'Timeframe';
    periodStart     : Timestamp @title: 'Period Start';
    periodEnd       : Timestamp @title: 'Period End';
    openPrice       : Decimal(20,8) @title: 'Open Price';
    highPrice       : Decimal(20,8) @title: 'High Price';
    lowPrice        : Decimal(20,8) @title: 'Low Price';
    closePrice      : Decimal(20,8) @title: 'Close Price';
    volume          : Decimal(30,8) @title: 'Volume';
    trades          : Integer @title: 'Number of Trades';
    vwap            : Decimal(20,8) @title: 'VWAP';
    spread          : Decimal(10,8) @title: 'Spread';
    volatility      : Decimal(10,4) @title: 'Volatility';
}

/**
 * ML Model Registry Entity
 * Registry of ML models used in the system
 */
entity MLModelRegistry : managed {
    key id          : Integer @title: 'ID';
    modelId         : String(100) @title: 'Model ID' not null;
    version         : String(20) @title: 'Version' not null;
    modelType       : String(50) @title: 'Model Type' not null;
    algorithm       : String(50) @title: 'Algorithm';
    parameters      : LargeString @title: 'Parameters';
    trainingMetrics : LargeString @title: 'Training Metrics';
    validationMetrics: LargeString @title: 'Validation Metrics';
    filePath        : String(500) @title: 'File Path';
    blobUrl         : String(500) @title: 'Blob URL';
    deployedAt      : Timestamp @title: 'Deployed At';
    status          : String(20) @title: 'Status' default 'trained';
}

/**
 * Onchain Data Entity
 * Blockchain and on-chain data
 */
entity OnchainData : cuid, managed {
    chainName       : String(50) @title: 'Chain Name';
    blockNumber     : Integer @title: 'Block Number';
    blockHash       : String(100) @title: 'Block Hash';
    transactionHash : String(100) @title: 'Transaction Hash';
    contractAddress : String(100) @title: 'Contract Address';
    eventName       : String(100) @title: 'Event Name';
    eventData       : LargeString @title: 'Event Data (JSON)';
    timestamp       : Timestamp @title: 'Timestamp';
    gasUsed         : Integer @title: 'Gas Used';
    gasPrice        : Decimal(20,8) @title: 'Gas Price';
    value           : Decimal(30,8) @title: 'Value';
    fromAddress     : String(100) @title: 'From Address';
    toAddress       : String(100) @title: 'To Address';
    status          : String(20) @title: 'Status';
}

/**
 * AI Analyses Entity
 * Results from AI analysis runs
 */
entity AIAnalyses : cuid, managed {
    analysisType    : String(50) @title: 'Analysis Type';
    targetSymbol    : String(20) @title: 'Target Symbol';
    modelUsed       : String(100) @title: 'Model Used';
    inputData       : LargeString @title: 'Input Data (JSON)';
    results         : LargeString @title: 'Results (JSON)';
    confidence      : Decimal(5,4) @title: 'Confidence Score';
    recommendations : LargeString @title: 'Recommendations (JSON)';
    performedAt     : Timestamp @title: 'Performed At';
    duration        : Integer @title: 'Duration (ms)';
    status          : String(20) @title: 'Status';
    errorDetails    : String(1000) @title: 'Error Details';
}

// Analytics Views
view ActiveDataJobs as select from DataIngestionJobs {
    *
} where status in ('SCHEDULED', 'RUNNING');

view DataQualityDashboard as select from DataQualityMetrics {
    dataSource,
    count(*) as totalChecks : Integer,
    sum(case when status = 'PASS' then 1 else 0 end) as passCount : Integer,
    sum(case when status = 'FAIL' then 1 else 0 end) as failCount : Integer,
    cast(sum(case when status = 'PASS' then 1 else 0 end) * 100.0 / count(*) as Decimal(5,2)) as passRate : Decimal(5,2)
} group by dataSource;

view ModelPerformance as select from MLModelRegistry {
    modelId,
    modelType,
    version,
    status,
    deployedAt
} where status = 'DEPLOYED';