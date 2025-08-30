namespace com.rex.cryptotrading.monitoring;

using { managed, cuid } from '@sap/cds/common';

/**
 * System Health Entity
 * Monitors overall system health
 */
entity SystemHealth : cuid, managed {
    componentName   : String(100) @title: 'Component Name';
    status          : String(20) @title: 'Status' @assert.range enum {
        HEALTHY;
        WARNING;
        CRITICAL;
        OFFLINE;
    };
    cpuUsage        : Decimal(5,2) @title: 'CPU Usage %';
    memoryUsage     : Decimal(5,2) @title: 'Memory Usage %';
    diskUsage       : Decimal(5,2) @title: 'Disk Usage %';
    uptime          : Integer @title: 'Uptime (seconds)';
    lastCheck       : Timestamp @title: 'Last Check';
    checkInterval   : Integer @title: 'Check Interval (seconds)';
    alerts          : LargeString @title: 'Active Alerts (JSON)';
}

/**
 * System Metrics Entity
 * Detailed system performance metrics
 */
entity SystemMetrics : cuid, managed {
    metricType      : String(50) @title: 'Metric Type';
    metricName      : String(100) @title: 'Metric Name';
    metricValue     : Decimal(20,4) @title: 'Metric Value';
    unit            : String(20) @title: 'Unit';
    timestamp       : Timestamp @title: 'Timestamp';
    source          : String(100) @title: 'Source';
    tags            : LargeString @title: 'Tags (JSON)';
    threshold       : Decimal(20,4) @title: 'Threshold';
    isAlert         : Boolean @title: 'Is Alert' default false;
}

/**
 * Monitoring Events Entity
 * System monitoring events and alerts
 */
entity MonitoringEvents : cuid, managed {
    eventType       : String(50) @title: 'Event Type';
    severity        : String(20) @title: 'Severity' @assert.range enum {
        INFO;
        WARNING;
        ERROR;
        CRITICAL;
    };
    source          : String(100) @title: 'Source';
    message         : String(1000) @title: 'Message';
    details         : LargeString @title: 'Details (JSON)';
    timestamp       : Timestamp @title: 'Timestamp';
    resolved        : Boolean @title: 'Resolved' default false;
    resolvedAt      : Timestamp @title: 'Resolved At';
    resolvedBy      : String(100) @title: 'Resolved By';
    actionTaken     : String(500) @title: 'Action Taken';
}

/**
 * Error Logs Entity
 * Application error logging
 */
entity ErrorLogs : cuid, managed {
    errorCode       : String(50) @title: 'Error Code';
    errorType       : String(100) @title: 'Error Type';
    message         : String(1000) @title: 'Error Message';
    stackTrace      : LargeString @title: 'Stack Trace';
    source          : String(200) @title: 'Source';
    userId          : String(50) @title: 'User ID';
    sessionId       : String(100) @title: 'Session ID';
    requestId       : String(100) @title: 'Request ID';
    timestamp       : Timestamp @title: 'Timestamp';
    severity        : String(20) @title: 'Severity';
    resolved        : Boolean @title: 'Resolved' default false;
    resolution      : String(500) @title: 'Resolution';
}

/**
 * Cache Entries Entity
 * General cache management
 */
entity CacheEntries : cuid, managed {
    cacheKey        : String(200) @title: 'Cache Key';
    cacheValue      : LargeString @title: 'Cache Value';
    cacheType       : String(50) @title: 'Cache Type';
    ttl             : Integer @title: 'TTL (seconds)';
    expiresAt       : Timestamp @title: 'Expires At';
    hits            : Integer @title: 'Hit Count' default 0;
    lastAccessed    : Timestamp @title: 'Last Accessed';
    size            : Integer @title: 'Size (bytes)';
    tags            : String(500) @title: 'Tags';
}

/**
 * Feature Cache Entity
 * ML feature caching
 */
entity FeatureCache : cuid, managed {
    featureName     : String(100) @title: 'Feature Name';
    featureVersion  : String(20) @title: 'Feature Version';
    symbol          : String(20) @title: 'Symbol';
    featureData     : LargeString @title: 'Feature Data (JSON)';
    computedAt      : Timestamp @title: 'Computed At';
    validUntil      : Timestamp @title: 'Valid Until';
    computeTime     : Integer @title: 'Compute Time (ms)';
    dependencies    : LargeString @title: 'Dependencies (JSON)';
    quality         : Decimal(3,2) @title: 'Quality Score';
}

/**
 * Historical Data Cache Entity
 * Historical data caching
 */
entity HistoricalDataCache : cuid, managed {
    dataType        : String(50) @title: 'Data Type';
    symbol          : String(20) @title: 'Symbol';
    timeframe       : String(20) @title: 'Timeframe';
    startDate       : Date @title: 'Start Date';
    endDate         : Date @title: 'End Date';
    dataPoints      : Integer @title: 'Data Points';
    cachedData      : LargeString @title: 'Cached Data (JSON)';
    compression     : String(20) @title: 'Compression Type';
    sizeBytes       : Integer @title: 'Size (bytes)';
    lastUpdated     : Timestamp @title: 'Last Updated';
    accessCount     : Integer @title: 'Access Count' default 0;
}

/**
 * Encryption Key Metadata Entity
 * Metadata for encryption keys
 */
entity EncryptionKeyMetadata : cuid, managed {
    keyId           : String(100) @title: 'Key ID';
    keyType         : String(50) @title: 'Key Type';
    algorithm       : String(50) @title: 'Algorithm';
    keyLength       : Integer @title: 'Key Length (bits)';
    purpose         : String(100) @title: 'Purpose';
    status          : String(20) @title: 'Status' @assert.range enum {
        ACTIVE;
        ROTATED;
        EXPIRED;
        REVOKED;
    };
    createdAt       : Timestamp @title: 'Created At';
    rotatedAt       : Timestamp @title: 'Rotated At';
    expiresAt       : Timestamp @title: 'Expires At';
    lastUsed        : Timestamp @title: 'Last Used';
    usageCount      : Integer @title: 'Usage Count' default 0;
}

// Analytics Views
view SystemHealthDashboard as select from SystemHealth {
    key componentName : String @(title: 'Component Name'),
    status,
    cpuUsage,
    memoryUsage,
    diskUsage,
    uptime,
    lastCheck
} where lastCheck >= $now - 300000; // Last 5 minutes

view CriticalErrors as select from ErrorLogs {
    *
} where severity = 'CRITICAL' and resolved = false;

view CacheEfficiency as select from CacheEntries {
    key cacheType : String @(title: 'Cache Type'),
    count(*) as totalEntries : Integer,
    sum(hits) as totalHits : Integer,
    avg(size) as avgSize : Integer
} group by cacheType;