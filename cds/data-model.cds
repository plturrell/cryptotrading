namespace com.rex.cryptotrading.data;

using { cuid, managed } from '@sap/cds/common';

// Data source configuration
entity DataSources : cuid, managed {
    name: String(50) not null;  // yahoo, fred, geckoterminal
    type: String(20);  // market, economic, dex
    baseUrl: String(500);
    apiKey: String(200);
    isActive: Boolean default true;
    rateLimit: Integer;  // requests per minute
    lastSync: DateTime;
    config: LargeString;  // JSON configuration
}

// Data loading jobs
entity LoadingJobs : cuid, managed {
    source: Association to DataSources;
    jobType: String(50);  // historical, realtime, incremental
    status: String(20) default 'pending';  // pending, running, completed, failed
    priority: Integer default 0;
    
    // Job parameters
    symbols: LargeString;  // JSON array of symbols
    startDate: DateTime;
    endDate: DateTime;
    interval: String(10);
    parameters: LargeString;  // JSON parameters
    
    // Progress tracking
    totalRecords: Integer default 0;
    processedRecords: Integer default 0;
    failedRecords: Integer default 0;
    progress: Integer default 0;  // 0-100
    
    // Timing
    scheduledAt: DateTime;
    startedAt: DateTime;
    completedAt: DateTime;
    estimatedCompletion: DateTime;
    
    // Error handling
    errorMessage: LargeString;
    retryCount: Integer default 0;
    maxRetries: Integer default 3;
}

// Loading status tracking
entity LoadingStatus : cuid {
    job: Association to LoadingJobs;
    symbol: String(50);
    dataSource: String(50);
    recordsLoaded: Integer;
    status: String(20);  // success, failed, skipped
    message: String(500);
    timestamp: DateTime;
}