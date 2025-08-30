using { managed, cuid } from '@sap/cds/common';

namespace com.rex.cryptotrading.codeanalysis;

/**
 * Code Analysis Project Entity
 */
entity Projects : managed, cuid {
    name        : String(100) @title: 'Project Name';
    description : String(500) @title: 'Description';
    path        : String(1000) @title: 'Project Path';
    status      : String(20) @title: 'Status' @assert.range enum {
        ACTIVE;
        INDEXING;
        COMPLETED;
        ERROR;
    };
    
    // Composition relationships
    indexingSessions : Composition of many IndexingSessions on indexingSessions.project = $self;
    codeFiles       : Composition of many CodeFiles on codeFiles.project = $self;
    analysisResults : Composition of many AnalysisResults on analysisResults.project = $self;
}

/**
 * Multi-Language Indexing Sessions
 */
entity IndexingSessions : managed, cuid {
    project         : Association to Projects;
    sessionName     : String(100) @title: 'Session Name';
    startTime       : Timestamp @title: 'Start Time';
    endTime         : Timestamp @title: 'End Time';
    status          : String(20) @title: 'Status' @assert.range enum {
        RUNNING;
        COMPLETED;
        FAILED;
        CANCELLED;
    };
    
    // Indexing statistics
    totalFiles      : Integer @title: 'Total Files';
    processedFiles  : Integer @title: 'Processed Files';
    totalFacts      : Integer @title: 'Total Facts Generated';
    coveragePercent : Decimal(5,2) @title: 'Coverage Percentage';
    blindSpots      : Integer @title: 'Blind Spots Count';
    
    // Language breakdown
    pythonFiles     : Integer @title: 'Python Files';
    jsFiles         : Integer @title: 'JavaScript Files';
    tsFiles         : Integer @title: 'TypeScript Files';
    capFiles        : Integer @title: 'SAP CAP Files';
    xmlFiles        : Integer @title: 'XML Files';
    configFiles     : Integer @title: 'Config Files';
    
    // Error tracking
    errorMessage    : String(1000) @title: 'Error Message';
    
    // Associations
    codeFiles       : Association to many CodeFiles on codeFiles.indexingSession = $self;
    analysisResults : Association to many AnalysisResults on analysisResults.indexingSession = $self;
}

/**
 * Code Files Entity
 */
entity CodeFiles : managed, cuid {
    project         : Association to Projects;
    indexingSession : Association to IndexingSessions;
    
    fileName        : String(255) @title: 'File Name';
    relativePath    : String(1000) @title: 'Relative Path';
    language        : String(20) @title: 'Language' @assert.range enum {
        PYTHON;
        JAVASCRIPT;
        TYPESCRIPT;
        SAP_CAP;
        XML;
        JSON;
        YAML;
        OTHER;
    };
    
    fileSize        : Integer @title: 'File Size (bytes)';
    linesOfCode     : Integer @title: 'Lines of Code';
    factsGenerated  : Integer @title: 'Facts Generated';
    
    // Parsing status
    parseStatus     : String(20) @title: 'Parse Status' @assert.range enum {
        SUCCESS;
        PARTIAL;
        FAILED;
        SKIPPED;
    };
    parseError      : String(1000) @title: 'Parse Error';
    
    // Language-specific metadata
    declarations    : Integer @title: 'Declarations Found';
    references      : Integer @title: 'References Found';
    imports         : Integer @title: 'Imports Found';
    exports         : Integer @title: 'Exports Found';
    
    // UI5 specific
    ui5Controllers  : Integer @title: 'UI5 Controllers';
    ui5Views        : Integer @title: 'UI5 Views';
    
    // CAP specific
    capEntities     : Integer @title: 'CAP Entities';
    capServices     : Integer @title: 'CAP Services';
}

/**
 * Analysis Results and Facts
 */
entity AnalysisResults : managed, cuid {
    project         : Association to Projects;
    indexingSession : Association to IndexingSessions;
    
    factType        : String(50) @title: 'Fact Type';
    predicate       : String(100) @title: 'Predicate';
    symbolName      : String(255) @title: 'Symbol Name';
    symbolType      : String(50) @title: 'Symbol Type';
    
    // Location information
    fileName        : String(255) @title: 'File Name';
    lineNumber      : Integer @title: 'Line Number';
    columnNumber    : Integer @title: 'Column Number';
    
    // Fact content
    factData        : LargeString @title: 'Fact Data (JSON)';
    
    // Quality metrics
    confidence      : Decimal(3,2) @title: 'Confidence Score';
    validated       : Boolean @title: 'Validated';
}

/**
 * Language Indexer Statistics
 */
entity IndexerStats : managed, cuid {
    language        : String(20) @title: 'Language';
    totalFiles      : Integer @title: 'Total Files Indexed';
    successRate     : Decimal(5,2) @title: 'Success Rate %';
    avgFactsPerFile : Decimal(8,2) @title: 'Avg Facts per File';
    totalFacts      : Integer @title: 'Total Facts Generated';
    
    // Performance metrics
    avgProcessingTime : Integer @title: 'Avg Processing Time (ms)';
    totalProcessingTime : Integer @title: 'Total Processing Time (ms)';
    
    // Error statistics
    parseErrors     : Integer @title: 'Parse Errors';
    commonErrors    : LargeString @title: 'Common Error Patterns (JSON)';
    
    lastUpdated     : Timestamp @title: 'Last Updated';
}

/**
 * Blind Spots and Coverage Analysis
 */
entity BlindSpots : managed, cuid {
    project         : Association to Projects;
    indexingSession : Association to IndexingSessions;
    
    filePattern     : String(255) @title: 'File Pattern';
    language        : String(50) @title: 'Detected Language';
    fileCount       : Integer @title: 'Affected Files';
    severity        : String(20) @title: 'Severity' @assert.range enum {
        LOW;
        MEDIUM;
        HIGH;
        CRITICAL;
    };
    
    description     : String(500) @title: 'Description';
    recommendation  : String(1000) @title: 'Recommendation';
    
    resolved        : Boolean @title: 'Resolved';
    resolvedAt      : Timestamp @title: 'Resolved At';
    resolvedBy      : String(100) @title: 'Resolved By';
}
