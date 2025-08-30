namespace com.rex.cryptotrading.analytics;

using { managed, cuid } from '@sap/cds/common';

/**
 * Factor Definitions Entity
 * Defines analytical factors used in trading
 */
entity FactorDefinitions : cuid, managed {
    factorName      : String(100) @title: 'Factor Name';
    factorType      : String(50) @title: 'Factor Type';
    category        : String(50) @title: 'Category';
    description     : String(500) @title: 'Description';
    formula         : LargeString @title: 'Formula';
    parameters      : LargeString @title: 'Parameters (JSON)';
    dataRequirements: LargeString @title: 'Data Requirements (JSON)';
    updateFrequency : String(20) @title: 'Update Frequency';
    version         : String(20) @title: 'Version';
    isActive        : Boolean @title: 'Is Active' default true;
    weight          : Decimal(5,4) @title: 'Weight';
    threshold       : Decimal(20,4) @title: 'Threshold';
}

/**
 * Factor Data Entity
 * Calculated factor values
 */
entity FactorData : cuid, managed {
    factor          : Association to FactorDefinitions @title: 'Factor';
    symbol          : String(20) @title: 'Symbol';
    timestamp       : Timestamp @title: 'Timestamp';
    value           : Decimal(30,8) @title: 'Factor Value';
    normalizedValue : Decimal(10,8) @title: 'Normalized Value';
    zscore          : Decimal(10,4) @title: 'Z-Score';
    percentile      : Decimal(5,2) @title: 'Percentile';
    signal          : String(20) @title: 'Signal';
    confidence      : Decimal(5,4) @title: 'Confidence';
    metadata        : LargeString @title: 'Metadata (JSON)';
}

/**
 * Time Series Entity
 * Generic time series data storage
 */
entity TimeSeries : cuid, managed {
    seriesName      : String(100) @title: 'Series Name';
    seriesType      : String(50) @title: 'Series Type';
    symbol          : String(20) @title: 'Symbol';
    timestamp       : Timestamp @title: 'Timestamp';
    value           : Decimal(30,8) @title: 'Value';
    volume          : Decimal(30,8) @title: 'Volume';
    metadata        : LargeString @title: 'Metadata (JSON)';
    quality         : String(20) @title: 'Data Quality';
    source          : String(100) @title: 'Data Source';
    adjusted        : Boolean @title: 'Is Adjusted' default false;
}

/**
 * Macro Data Entity
 * Macroeconomic indicators
 */
entity MacroData : cuid, managed {
    indicatorName   : String(100) @title: 'Indicator Name';
    indicatorCode   : String(50) @title: 'Indicator Code';
    country         : String(50) @title: 'Country';
    region          : String(50) @title: 'Region';
    value           : Decimal(30,8) @title: 'Value';
    unit            : String(20) @title: 'Unit';
    period          : String(20) @title: 'Period';
    releaseDate     : Date @title: 'Release Date';
    previousValue   : Decimal(30,8) @title: 'Previous Value';
    forecast        : Decimal(30,8) @title: 'Forecast';
    actual          : Decimal(30,8) @title: 'Actual';
    impact          : String(20) @title: 'Market Impact';
    source          : String(100) @title: 'Source';
}

/**
 * Sentiment Data Entity
 * Market sentiment indicators
 */
entity SentimentData : cuid, managed {
    symbol          : String(20) @title: 'Symbol';
    source          : String(100) @title: 'Source';
    sentimentType   : String(50) @title: 'Sentiment Type';
    score           : Decimal(5,4) @title: 'Sentiment Score';
    bullish         : Decimal(5,2) @title: 'Bullish %';
    bearish         : Decimal(5,2) @title: 'Bearish %';
    neutral         : Decimal(5,2) @title: 'Neutral %';
    volume          : Integer @title: 'Sample Volume';
    confidence      : Decimal(5,4) @title: 'Confidence';
    timestamp       : Timestamp @title: 'Timestamp';
    keywords        : LargeString @title: 'Keywords (JSON)';
    trends          : LargeString @title: 'Trends (JSON)';
}

/**
 * Memory Fragments Entity
 * Fragmented memory storage for distributed processing
 */
entity MemoryFragments : cuid, managed {
    fragmentKey     : String(200) @title: 'Fragment Key';
    parentKey       : String(200) @title: 'Parent Key';
    fragmentIndex   : Integer @title: 'Fragment Index';
    totalFragments  : Integer @title: 'Total Fragments';
    data            : LargeString @title: 'Fragment Data';
    dataType        : String(50) @title: 'Data Type';
    encoding        : String(20) @title: 'Encoding';
    checksum        : String(100) @title: 'Checksum';
    sizeBytes       : Integer @title: 'Size (bytes)';
    isComplete      : Boolean @title: 'Is Complete' default false;
}

/**
 * Semantic Memory Entity
 * AI semantic memory storage
 */
entity SemanticMemory : cuid, managed {
    memoryType      : String(50) @title: 'Memory Type';
    context         : String(200) @title: 'Context';
    embedding       : LargeString @title: 'Embedding Vector';
    content         : LargeString @title: 'Content';
    metadata        : LargeString @title: 'Metadata (JSON)';
    similarity      : Decimal(5,4) @title: 'Similarity Score';
    importance      : Decimal(5,4) @title: 'Importance';
    accessCount     : Integer @title: 'Access Count' default 0;
    lastAccessed    : Timestamp @title: 'Last Accessed';
    decayFactor     : Decimal(5,4) @title: 'Decay Factor';
}

// Analytics Views
view ActiveFactors as select from FactorDefinitions {
    *
} where isActive = true;

view RecentFactorData as select from FactorData {
    factor.factorName as factorName,
    symbol,
    timestamp,
    value,
    signal,
    confidence
} where timestamp >= $now - 3600000; // Last hour

view MarketSentimentSummary as select from SentimentData {
    symbol,
    avg(score) as avgSentiment : Decimal(5,4),
    avg(bullish) as avgBullish : Decimal(5,2),
    avg(bearish) as avgBearish : Decimal(5,2),
    count(*) as dataPoints : Integer
} group by symbol;