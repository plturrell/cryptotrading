using { com.rex.cryptotrading.codeanalysis as ca } from './code-analysis-model';

namespace com.rex.cryptotrading.codeanalysis.service;

/**
 * Code Analysis Service - RESTful API for Multi-Language Indexer
 */
service CodeAnalysisService {
    
    // Main entities
    entity Projects as projection on ca.Projects;
    entity IndexingSessions as projection on ca.IndexingSessions;
    entity CodeFiles as projection on ca.CodeFiles;
    entity AnalysisResults as projection on ca.AnalysisResults;
    entity IndexerStats as projection on ca.IndexerStats;
    entity BlindSpots as projection on ca.BlindSpots;
    
    // Analytics Views
    view ProjectAnalytics as select from ca.Projects {
        ID,
        name,
        status,
        count(indexingSessions.ID) as totalSessions : Integer,
        sum(indexingSessions.totalFacts) as totalFacts : Integer,
        avg(indexingSessions.coveragePercent) as avgCoverage : Decimal(5,2),
        count(codeFiles.ID) as totalFiles : Integer
    } group by ID, name, status;
    
    view LanguageBreakdown as select from ca.CodeFiles {
        language,
        count(*) as fileCount : Integer,
        sum(factsGenerated) as totalFacts : Integer,
        avg(factsGenerated) as avgFactsPerFile : Decimal(8,2),
        sum(case when parseStatus = 'SUCCESS' then 1 else 0 end) as successCount : Integer,
        cast(sum(case when parseStatus = 'SUCCESS' then 1 else 0 end) * 100.0 / count(*) as Decimal(5,2)) as successRate : Decimal(5,2)
    } group by language;
    
    view SessionProgress as select from ca.IndexingSessions {
        ID,
        sessionName,
        project.name as projectName,
        status,
        startTime,
        endTime,
        totalFiles,
        processedFiles,
        cast(processedFiles * 100.0 / totalFiles as Decimal(5,2)) as progressPercent : Decimal(5,2),
        totalFacts,
        coveragePercent,
        blindSpots
    };
    
    // Actions for indexing operations
    action startIndexing(projectId: String, sessionName: String) returns {
        sessionId: String;
        status: String;
        message: String;
    };
    
    action stopIndexing(sessionId: String) returns {
        status: String;
        message: String;
    };
    
    action validateResults(sessionId: String) returns {
        validationScore: Decimal(5,2);
        issues: many {
            type: String;
            severity: String;
            description: String;
            recommendation: String;
        };
    };
    
    action exportResults(sessionId: String, format: String) returns {
        downloadUrl: String;
        fileSize: Integer;
        recordCount: Integer;
    };
    
    // Functions for analytics
    function getAnalytics() returns {
        totalProjects: Integer;
        totalFiles: Integer;
        totalFacts: Integer;
        coveragePercent: Decimal(5,2);
        languages: many {
            language: String;
            files: Integer;
            successRate: Decimal(5,2);
            facts: Integer;
        };
        recentSessions: many {
            sessionName: String;
            status: String;
            duration: String;
            facts: Integer;
        };
    };
    
    function getBlindSpotAnalysis() returns {
        totalBlindSpots: Integer;
        criticalCount: Integer;
        highCount: Integer;
        mediumCount: Integer;
        lowCount: Integer;
        topPatterns: many {
            pattern: String;
            count: Integer;
            severity: String;
        };
        recommendations: many {
            priority: String;
            action: String;
            impact: String;
        };
    };
    
    function getPerformanceMetrics() returns {
        avgProcessingTime: Integer;
        throughputPerHour: Integer;
        errorRate: Decimal(5,2);
        memoryUsage: Integer;
        cpuUtilization: Decimal(5,2);
        queueLength: Integer;
    };
}
