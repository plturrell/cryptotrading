namespace com.rex.cryptotrading.news;

using { managed, cuid } from '@sap/cds/common';

/**
 * News Articles Entity
 * Store cryptocurrency news articles with Russian translation support
 */
entity NewsArticles : cuid, managed {
    title           : String(500) @title: 'Article Title' not null;
    content         : LargeString @title: 'Article Content';
    summary         : LargeString @title: 'Article Summary';
    url             : String(1000) @title: 'Source URL';
    source          : String(100) @title: 'News Source';
    author          : String(100) @title: 'Author';
    publishedAt     : Timestamp @title: 'Published Date';
    language        : String(5) @title: 'Language' default 'en';
    category        : String(50) @title: 'Category';
    symbols         : LargeString @title: 'Related Symbols (JSON)';
    sentiment       : String(20) @title: 'Sentiment' @assert.range enum {
        POSITIVE;
        NEGATIVE;
        NEUTRAL;
    } default 'NEUTRAL';
    relevanceScore  : Decimal(3,2) @title: 'Relevance Score';
    
    // Russian translation fields
    translatedTitle : String(500) @title: 'Russian Title';
    translatedContent: LargeString @title: 'Russian Content';
    translatedSummary: LargeString @title: 'Russian Summary';
    translationStatus: String(20) @title: 'Translation Status' @assert.range enum {
        PENDING;
        COMPLETED;
        FAILED;
        NOT_REQUIRED;
    } default 'NOT_REQUIRED';
    
    // Image support
    images          : LargeString @title: 'Article Images (JSON)';
    hasImages       : Boolean @title: 'Has Images' default false;
    imageCount      : Integer @title: 'Image Count' default 0;
    
    // Metadata
    tags            : LargeString @title: 'Tags (JSON)';
    metadata        : LargeString @title: 'Additional Metadata (JSON)';
    isActive        : Boolean @title: 'Is Active' default true;
    viewCount       : Integer @title: 'View Count' default 0;
    
    // Navigation
    userInteractions: Composition of many NewsUserInteractions on userInteractions.article = $self;
    categories      : Composition of many NewsCategories on categories.article = $self;
}

/**
 * News Categories Entity
 * Categorization of news articles
 */
entity NewsCategories : cuid, managed {
    article         : Association to NewsArticles @title: 'Article';
    categoryName    : String(50) @title: 'Category Name' not null;
    categoryType    : String(20) @title: 'Category Type' @assert.range enum {
        PRIMARY;
        SECONDARY;
        TAG;
    } default 'PRIMARY';
    confidence      : Decimal(3,2) @title: 'Confidence Score';
}

/**
 * News User Interactions Entity
 * Track user interactions with news articles
 */
entity NewsUserInteractions : cuid, managed {
    article         : Association to NewsArticles @title: 'Article';
    userId          : Integer @title: 'User ID';
    interactionType : String(20) @title: 'Interaction Type' @assert.range enum {
        VIEW;
        LIKE;
        SHARE;
        BOOKMARK;
        COMMENT;
    };
    timestamp       : Timestamp @title: 'Interaction Time';
    metadata        : LargeString @title: 'Interaction Metadata (JSON)';
}

/**
 * News Sources Entity
 * Manage news sources and their configurations
 */
entity NewsSources : cuid, managed {
    name            : String(100) @title: 'Source Name' not null;
    url             : String(500) @title: 'Source URL';
    type            : String(20) @title: 'Source Type' @assert.range enum {
        RSS;
        API;
        SCRAPER;
        PERPLEXITY;
    };
    language        : String(5) @title: 'Primary Language' default 'en';
    country         : String(5) @title: 'Country Code';
    reliability     : Decimal(3,2) @title: 'Reliability Score';
    isActive        : Boolean @title: 'Is Active' default true;
    lastFetched     : Timestamp @title: 'Last Fetched';
    fetchFrequency  : Integer @title: 'Fetch Frequency (minutes)' default 60;
    
    // Configuration
    apiKey          : String(200) @title: 'API Key (Encrypted)';
    configuration   : LargeString @title: 'Source Configuration (JSON)';
    
    // Navigation
    articles        : Association to many NewsArticles on articles.source = name;
}

/**
 * News Fetch Jobs Entity
 * Track news fetching jobs and their status
 */
entity NewsFetchJobs : cuid, managed {
    jobType         : String(20) @title: 'Job Type' @assert.range enum {
        LATEST;
        CATEGORY;
        SYMBOL;
        RUSSIAN;
        TRANSLATION;
    };
    status          : String(20) @title: 'Status' @assert.range enum {
        PENDING;
        RUNNING;
        COMPLETED;
        FAILED;
        CANCELLED;
    } default 'PENDING';
    parameters      : LargeString @title: 'Job Parameters (JSON)';
    startedAt       : Timestamp @title: 'Started At';
    completedAt     : Timestamp @title: 'Completed At';
    articlesCount   : Integer @title: 'Articles Fetched' default 0;
    errorMessage    : String(1000) @title: 'Error Message';
    
    // Progress tracking
    progress        : Integer @title: 'Progress Percentage' default 0;
    currentStep     : String(100) @title: 'Current Step';
}

/**
 * News Analytics Entity
 * Store analytics data for news articles
 */
entity NewsAnalytics : cuid, managed {
    date            : Date @title: 'Analytics Date';
    totalArticles   : Integer @title: 'Total Articles' default 0;
    englishArticles : Integer @title: 'English Articles' default 0;
    russianArticles : Integer @title: 'Russian Articles' default 0;
    translatedArticles: Integer @title: 'Translated Articles' default 0;
    
    // Category breakdown
    categoryBreakdown: LargeString @title: 'Category Breakdown (JSON)';
    
    // Sentiment analysis
    positiveCount   : Integer @title: 'Positive Articles' default 0;
    negativeCount   : Integer @title: 'Negative Articles' default 0;
    neutralCount    : Integer @title: 'Neutral Articles' default 0;
    
    // User engagement
    totalViews      : Integer @title: 'Total Views' default 0;
    totalLikes      : Integer @title: 'Total Likes' default 0;
    totalShares     : Integer @title: 'Total Shares' default 0;
    totalBookmarks  : Integer @title: 'Total Bookmarks' default 0;
}

// Views for common queries
view LatestNews as select from NewsArticles {
    *
} where isActive = true
order by publishedAt desc;

view RussianNews as select from NewsArticles {
    *
} where language = 'ru' and isActive = true
order by publishedAt desc;

view TranslatedNews as select from NewsArticles {
    *
} where translationStatus = 'COMPLETED' and isActive = true
order by publishedAt desc;

view TrendingNews as select from NewsArticles {
    *
} where isActive = true
order by viewCount desc, publishedAt desc;

view NewsByCategory as select from NewsArticles {
    category,
    count(*) as articleCount : Integer,
    avg(relevanceScore) as avgRelevance : Decimal(3,2),
    max(publishedAt) as latestArticle : Timestamp
} where isActive = true
group by category;

view NewsSourceStats as select from NewsSources {
    name,
    language,
    reliability,
    count(articles.ID) as totalArticles : Integer,
    max(articles.publishedAt) as latestArticle : Timestamp
} where isActive = true
group by name, language, reliability;
