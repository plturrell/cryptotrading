namespace com.rex.cryptotrading.news;

using { NewsArticles, NewsCategories, NewsUserInteractions, NewsSources, NewsFetchJobs, NewsAnalytics } from './news-model';

/**
 * News Service Definition
 * Comprehensive news management and retrieval service
 */
service NewsService {
    
    // Core News Entities
    entity Articles as projection on NewsArticles;
    entity Categories as projection on NewsCategories;
    entity Sources as projection on NewsSources;
    entity FetchJobs as projection on NewsFetchJobs;
    entity Analytics as projection on NewsAnalytics;
    entity UserInteractions as projection on NewsUserInteractions;
    
    // News Retrieval Actions
    
    /**
     * Get Latest News
     */
    action getLatestNews(
        limit: Integer default 10,
        language: String default 'en',
        category: String
    ) returns array of {
        id: String;
        title: String;
        content: String;
        summary: String;
        url: String;
        source: String;
        author: String;
        publishedAt: Timestamp;
        language: String;
        category: String;
        symbols: String;
        sentiment: String;
        relevanceScore: Decimal;
        translatedTitle: String;
        translatedContent: String;
        viewCount: Integer;
        images: String;
        hasImages: Boolean;
        imageCount: Integer;
        featuredImage: String;
    };
    
    /**
     * Get News by Category
     */
    action getNewsByCategory(
        category: String not null,
        limit: Integer default 10,
        language: String default 'en'
    ) returns array of {
        id: String;
        title: String;
        content: String;
        summary: String;
        url: String;
        source: String;
        publishedAt: Timestamp;
        language: String;
        sentiment: String;
        relevanceScore: Decimal;
        translatedTitle: String;
        translatedContent: String;
        images: String;
        hasImages: Boolean;
        imageCount: Integer;
        featuredImage: String;
    };
    
    /**
     * Get News by Symbol
     */
    action getNewsBySymbol(
        symbol: String not null,
        limit: Integer default 10,
        language: String default 'en'
    ) returns array of {
        id: String;
        title: String;
        content: String;
        summary: String;
        url: String;
        source: String;
        publishedAt: Timestamp;
        language: String;
        symbols: String;
        sentiment: String;
        relevanceScore: Decimal;
        translatedTitle: String;
        translatedContent: String;
        images: String;
        hasImages: Boolean;
        imageCount: Integer;
        featuredImage: String;
    };
    
    /**
     * Get Russian News
     */
    action getRussianNews(
        limit: Integer default 10,
        category: String,
        symbol: String
    ) returns array of {
        id: String;
        title: String;
        content: String;
        summary: String;
        url: String;
        source: String;
        publishedAt: Timestamp;
        language: String;
        category: String;
        symbols: String;
        sentiment: String;
        translatedTitle: String;
        translatedContent: String;
        translatedSummary: String;
        translationStatus: String;
        images: String;
        hasImages: Boolean;
        imageCount: Integer;
        featuredImage: String;
    };
    
    /**
     * Search News
     */
    action searchNews(
        query: String not null,
        limit: Integer default 10,
        language: String default 'en',
        category: String,
        dateFrom: Date,
        dateTo: Date
    ) returns array of {
        id: String;
        title: String;
        content: String;
        summary: String;
        url: String;
        source: String;
        publishedAt: Timestamp;
        language: String;
        category: String;
        symbols: String;
        sentiment: String;
        relevanceScore: Decimal;
        translatedTitle: String;
        translatedContent: String;
    };
    
    /**
     * Get Market Sentiment
     */
    action getMarketSentiment(
        limit: Integer default 20,
        timeframe: String default '24h'
    ) returns {
        overallSentiment: String;
        positiveCount: Integer;
        negativeCount: Integer;
        neutralCount: Integer;
        sentimentScore: Decimal;
        articles: array of {
            id: String;
            title: String;
            sentiment: String;
            relevanceScore: Decimal;
            publishedAt: Timestamp;
            source: String;
        };
    };
    
    // Translation Actions
    
    /**
     * Translate Article to Russian
     */
    action translateToRussian(
        articleId: String not null
    ) returns {
        success: Boolean;
        translatedTitle: String;
        translatedContent: String;
        translatedSummary: String;
        translationStatus: String;
    };
    
    /**
     * Batch Translate Articles
     */
    action batchTranslate(
        articleIds: array of String,
        targetLanguage: String default 'ru'
    ) returns {
        success: Boolean;
        processedCount: Integer;
        successCount: Integer;
        failedCount: Integer;
        jobId: String;
    };
    
    // News Fetching Actions
    
    /**
     * Fetch Latest News from API
     */
    action fetchLatestNews(
        limit: Integer default 10,
        sources: array of String,
        categories: array of String
    ) returns {
        success: Boolean;
        articlesCount: Integer;
        jobId: String;
        message: String;
    };
    
    /**
     * Fetch Russian Crypto News
     */
    action fetchRussianCryptoNews(
        limit: Integer default 10
    ) returns {
        success: Boolean;
        articlesCount: Integer;
        jobId: String;
        message: String;
    };
    
    // Analytics Actions
    
    /**
     * Get News Analytics
     */
    action getNewsAnalytics(
        dateFrom: Date,
        dateTo: Date,
        groupBy: String default 'day'
    ) returns {
        totalArticles: Integer;
        englishArticles: Integer;
        russianArticles: Integer;
        translatedArticles: Integer;
        categoryBreakdown: String;
        sentimentDistribution: {
            positive: Integer;
            negative: Integer;
            neutral: Integer;
        };
        dailyStats: array of {
            date: Date;
            articleCount: Integer;
            avgSentiment: Decimal;
            topCategory: String;
        };
    };
    
    /**
     * Get Trending Topics
     */
    action getTrendingTopics(
        timeframe: String default '24h',
        limit: Integer default 10
    ) returns array of {
        topic: String;
        articleCount: Integer;
        sentiment: String;
        relevanceScore: Decimal;
        symbols: array of String;
    };
    
    // User Interaction Actions
    
    /**
     * Record User Interaction
     */
    action recordInteraction(
        articleId: String not null,
        userId: Integer not null,
        interactionType: String not null,
        metadata: String
    ) returns {
        success: Boolean;
        message: String;
    };
    
    /**
     * Get User Reading History
     */
    action getUserReadingHistory(
        userId: Integer not null,
        limit: Integer default 20
    ) returns array of {
        articleId: String;
        title: String;
        category: String;
        readAt: Timestamp;
        interactionType: String;
    };
    
    // Configuration Actions
    
    /**
     * Get Available Categories
     */
    function getAvailableCategories() returns array of {
        name: String;
        description: String;
        articleCount: Integer;
        language: String;
    };
    
    /**
     * Get Supported Languages
     */
    function getSupportedLanguages() returns array of {
        code: String;
        name: String;
        nativeName: String;
        translationSupported: Boolean;
    };
    
    /**
     * Get News Sources
     */
    function getNewsSources() returns array of {
        name: String;
        url: String;
        language: String;
        reliability: Decimal;
        isActive: Boolean;
        lastFetched: Timestamp;
    };
}

// Annotations for UI
annotate NewsService.Articles with @(
    UI.LineItem: [
        { $Type: 'UI.DataField', Value: title, Label: 'Title' },
        { $Type: 'UI.DataField', Value: source, Label: 'Source' },
        { $Type: 'UI.DataField', Value: category, Label: 'Category' },
        { $Type: 'UI.DataField', Value: language, Label: 'Language' },
        { $Type: 'UI.DataField', Value: sentiment, Label: 'Sentiment' },
        { $Type: 'UI.DataField', Value: publishedAt, Label: 'Published' },
        { $Type: 'UI.DataField', Value: viewCount, Label: 'Views' }
    ],
    UI.SelectionFields: [
        category,
        language,
        sentiment,
        source
    ],
    UI.HeaderInfo: {
        TypeName: 'News Article',
        TypeNamePlural: 'News Articles',
        Title: { Value: title },
        Description: { Value: summary }
    }
);

annotate NewsService.Articles with @(
    UI.Facets: [
        {
            $Type: 'UI.ReferenceFacet',
            Label: 'Article Details',
            Target: '@UI.FieldGroup#ArticleDetails'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Label: 'Translation',
            Target: '@UI.FieldGroup#Translation'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Label: 'Analytics',
            Target: '@UI.FieldGroup#Analytics'
        }
    ],
    UI.FieldGroup#ArticleDetails: {
        Data: [
            { $Type: 'UI.DataField', Value: title },
            { $Type: 'UI.DataField', Value: summary },
            { $Type: 'UI.DataField', Value: url },
            { $Type: 'UI.DataField', Value: source },
            { $Type: 'UI.DataField', Value: author },
            { $Type: 'UI.DataField', Value: publishedAt },
            { $Type: 'UI.DataField', Value: category },
            { $Type: 'UI.DataField', Value: symbols },
            { $Type: 'UI.DataField', Value: sentiment },
            { $Type: 'UI.DataField', Value: relevanceScore }
        ]
    },
    UI.FieldGroup#Translation: {
        Data: [
            { $Type: 'UI.DataField', Value: language },
            { $Type: 'UI.DataField', Value: translatedTitle },
            { $Type: 'UI.DataField', Value: translatedSummary },
            { $Type: 'UI.DataField', Value: translationStatus }
        ]
    },
    UI.FieldGroup#Analytics: {
        Data: [
            { $Type: 'UI.DataField', Value: viewCount },
            { $Type: 'UI.DataField', Value: relevanceScore },
            { $Type: 'UI.DataField', Value: createdAt },
            { $Type: 'UI.DataField', Value: modifiedAt }
        ]
    }
);
