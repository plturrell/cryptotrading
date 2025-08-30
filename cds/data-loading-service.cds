using { com.rex.cryptotrading.data as data } from './data-model';

service DataLoadingService @(path: '/api/odata/v4/DataLoadingService') {
    
    // Data Loading Entities
    entity DataSources as projection on data.DataSources;
    entity LoadingJobs as projection on data.LoadingJobs;
    entity LoadingStatus as projection on data.LoadingStatus;
    
    // Data Loading Actions
    
    // Load Yahoo Finance Data
    action loadYahooFinanceData(
        symbols: array of String,
        startDate: DateTime,
        endDate: DateTime,
        interval: String  // 1m, 5m, 15m, 1h, 1d, 1wk, 1mo
    ) returns {
        jobId: String;
        status: String;
        message: String;
        recordsQueued: Integer;
    };
    
    // Load FRED Economic Data
    action loadFREDData(
        series: array of String,  // DGS10, WALCL, M2SL, etc.
        startDate: DateTime,
        endDate: DateTime
    ) returns {
        jobId: String;
        status: String;
        message: String;
        seriesQueued: Integer;
    };
    
    // Load GeckoTerminal DEX Data
    action loadGeckoTerminalData(
        networks: array of String,  // ethereum, bsc, polygon, etc.
        poolCount: Integer,
        includeVolume: Boolean,
        includeLiquidity: Boolean
    ) returns {
        jobId: String;
        status: String;
        message: String;
        poolsQueued: Integer;
    };
    
    // Bulk Load All Sources
    action loadAllMarketData(
        cryptoSymbols: array of String,
        fredSeries: array of String,
        dexNetworks: array of String,
        startDate: DateTime,
        endDate: DateTime
    ) returns {
        jobIds: array of String;
        totalJobs: Integer;
        status: String;
        message: String;
    };
    
    // Job Management Actions
    action cancelLoadingJob(jobId: String) returns {
        success: Boolean;
        message: String;
    };
    
    action retryFailedJobs(since: DateTime) returns {
        retriedCount: Integer;
        jobIds: array of String;
        message: String;
    };
    
    // Data Loading Functions
    
    function getLoadingStatus(jobId: String) returns {
        jobId: String;
        status: String;  // pending, running, completed, failed
        progress: Integer;  // 0-100
        recordsLoaded: Integer;
        recordsFailed: Integer;
        startTime: DateTime;
        endTime: DateTime;
        errorMessage: String;
    };
    
    function getActiveJobs() returns array of {
        jobId: String;
        source: String;
        status: String;
        progress: Integer;
        startTime: DateTime;
    };
    
    function getDataSourceStatus() returns array of {
        source: String;  // yahoo, fred, geckoterminal
        isAvailable: Boolean;
        lastSync: DateTime;
        recordCount: Integer;
        apiStatus: String;
        rateLimit: String;
    };
    
    function getHistoricalDataRange(symbol: String) returns {
        symbol: String;
        earliestDate: DateTime;
        latestDate: DateTime;
        recordCount: Integer;
        dataSources: array of String;
        gaps: array of {
            startDate: DateTime;
            endDate: DateTime;
        };
    };
    
    // Data Validation Functions
    
    function validateDataIntegrity(
        symbol: String,
        startDate: DateTime,
        endDate: DateTime
    ) returns {
        isValid: Boolean;
        missingDates: array of DateTime;
        duplicates: Integer;
        outliers: Integer;
        qualityScore: Decimal;
    };
    
    function getDataStatistics(
        source: String,
        period: String  // today, week, month
    ) returns {
        source: String;
        period: String;
        totalRecords: Integer;
        uniqueSymbols: Integer;
        avgRecordsPerDay: Decimal;
        dataQuality: Decimal;
        lastUpdate: DateTime;
    };
}