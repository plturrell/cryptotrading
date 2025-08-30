using { com.rex.cryptotrading.monitoring as monitoring } from './monitoring-system-model';

namespace com.rex.cryptotrading.monitoring.service;

/**
 * Monitoring System Service - RESTful API for System Monitoring
 */
@path: '/api/odata/v4/MonitoringService'
service MonitoringService {
    
    // Core Monitoring Entities
    @cds.redirection.target
    entity SystemHealth as projection on monitoring.SystemHealth;
    
    entity SystemMetrics as projection on monitoring.SystemMetrics;
    
    entity MonitoringEvents as projection on monitoring.MonitoringEvents;
    
    entity ErrorLogs as projection on monitoring.ErrorLogs;
    
    entity CacheEntries as projection on monitoring.CacheEntries;
    
    entity FeatureCache as projection on monitoring.FeatureCache;
    
    entity HistoricalDataCache as projection on monitoring.HistoricalDataCache;
    
    @restrict: [
        { grant: ['READ'], to: 'admin' }
    ]
    entity EncryptionKeyMetadata as projection on monitoring.EncryptionKeyMetadata;
    
    // Analytics Views
    @readonly
    entity SystemHealthDashboard as projection on monitoring.SystemHealthDashboard;
    
    @readonly
    entity CriticalErrors as projection on monitoring.CriticalErrors;
    
    @readonly
    entity CacheEfficiency as projection on monitoring.CacheEfficiency;
    
    // Monitoring Actions
    action checkSystemHealth() returns {
        overallStatus: String;
        components: array of {
            name: String;
            status: String;
            metrics: {
                cpu: Decimal;
                memory: Decimal;
                disk: Decimal;
            };
        };
    };
    
    action clearCache(
        cacheType: String,
        olderThan: DateTime
    ) returns {
        entriesCleared: Integer;
        spaceFreed: Integer;
    };
    
    action resolveError(
        errorId: String,
        resolution: String
    ) returns {
        success: Boolean;
        message: String;
    };
    
    action triggerAlert(
        severity: String,
        message: String,
        source: String
    ) returns {
        eventId: String;
        notificationsSent: Integer;
    };
    
    // Monitoring Functions
    function getSystemMetrics(
        metricType: String,
        period: String
    ) returns array of {
        timestamp: DateTime;
        metricName: String;
        value: Decimal;
        unit: String;
    };
    
    function getCacheStatistics() returns {
        totalEntries: Integer;
        totalSize: Integer;
        hitRate: Decimal;
        missRate: Decimal;
        evictionRate: Decimal;
        byType: array of {
            cacheType: String;
            entries: Integer;
            hits: Integer;
            efficiency: Decimal;
        };
    };
    
    function getErrorAnalytics(
        period: String
    ) returns {
        totalErrors: Integer;
        criticalErrors: Integer;
        resolvedErrors: Integer;
        avgResolutionTime: Integer;
        topErrors: array of {
            errorType: String;
            count: Integer;
            lastOccurrence: DateTime;
        };
    };
}