namespace com.rex.cryptotrading.datapipeline;

using { managed, cuid } from '@sap/cds/common';

/**
 * Data Ingestion Jobs Entity
 * Manages data ingestion pipeline jobs
 */
entity DataIngestionJobs : cuid, managed {
    jobName         : String(100) @title: 'Job Name';
    jobType         : String(50) @title: 'Job Type';
    source          : String(100) @title: 'Data Source';
    destination     : String(100) @title: 'Destination';
    schedule        : String(100) @title: 'Schedule (Cron)';
    status          : String(20) @title: 'Status' @assert.range enum {
        SCHEDULED;
        RUNNING;
        COMPLETED;
        FAILED;
        CANCELLED;
    };
    startTime       : Timestamp @title: 'Start Time';
    endTime         : Timestamp @title: 'End Time';
    recordsProcessed: Integer @title: 'Records Processed';
    recordsFailed   : Integer @title: 'Records Failed';
    errorMessage    : String(1000) @title: 'Error Message';
    configuration   : LargeString @title: 'Configuration (JSON)';
    lastRun         : Timestamp @title: 'Last Run';
    nextRun         : Timestamp @title: 'Next Run';
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
entity MLModelRegistry : cuid, managed {
    modelName       : String(100) @title: 'Model Name';
    modelType       : String(50) @title: 'Model Type';
    version         : String(20) @title: 'Version';
    framework       : String(50) @title: 'Framework';
    description     : String(500) @title: 'Description';
    modelPath       : String(500) @title: 'Model Path';
    parameters      : LargeString @title: 'Parameters (JSON)';
    metrics         : LargeString @title: 'Metrics (JSON)';
    status          : String(20) @title: 'Status' @assert.range enum {
        TRAINING;
        VALIDATING;
        DEPLOYED;
        ARCHIVED;
        FAILED;
    };
    trainedAt       : Timestamp @title: 'Trained At';
    deployedAt      : Timestamp @title: 'Deployed At';
    accuracy        : Decimal(5,4) @title: 'Accuracy';
    performance     : LargeString @title: 'Performance Metrics (JSON)';
    inputFeatures   : LargeString @title: 'Input Features (JSON)';
    outputFormat    : String(100) @title: 'Output Format';
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
    ![from]         : String(100) @title: 'From Address';
    ![to]           : String(100) @title: 'To Address';
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
    key dataSource : String @(title: 'Data Source'),
    count(*) as totalChecks : Integer,
    sum(case when status = 'PASS' then 1 else 0 end) as passCount : Integer,
    sum(case when status = 'FAIL' then 1 else 0 end) as failCount : Integer,
    cast(sum(case when status = 'PASS' then 1 else 0 end) * 100.0 / count(*) as Decimal(5,2)) as passRate : Decimal(5,2)
} group by dataSource;

view ModelPerformance as select from MLModelRegistry {
    key ID,
    modelName,
    modelType,
    version,
    status,
    accuracy,
    trainedAt,
    deployedAt
} where status = 'DEPLOYED';