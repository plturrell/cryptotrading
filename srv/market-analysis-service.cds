using { com.рекс.cryptotrading.market as market } from './market-data-model';

service MarketAnalysisService @(path: '/api/odata/v4/MarketAnalysisService') {
    
    // Market Data Entities (Read-Only)
    @readonly
    @cds.redirection.target
    entity MarketPairs as projection on market.MarketPairs;
    
    @readonly
    entity PriceHistory as projection on market.PriceHistory;
    
    @readonly
    entity MarketData as projection on market.MarketData;
    
    @readonly
    entity MarketIndicators as projection on market.MarketIndicators;
    
    @readonly
    entity MarketSentiment as projection on market.MarketSentiment;
    
    // Analytics Views
    @readonly
    entity TopCryptocurrencies as projection on market.TopCryptocurrencies;
    
    // Market Analysis Functions
    function getMarketSummary() returns {
        totalMarketCap: Decimal;
        totalVolume24h: Decimal;
        btcDominance: Decimal;
        fearGreedIndex: Integer;
        activeMarkets: Integer;
    };
    
    function getPriceHistory(
        symbol: String,
        timeframe: String,
        startDate: DateTime,
        endDate: DateTime
    ) returns array of {
        timestamp: DateTime;
        open: Decimal;
        high: Decimal;
        low: Decimal;
        close: Decimal;
        volume: Decimal;
    };
    
    function getMarketTrends(period: String) returns {
        topGainers: array of {
            symbol: String;
            priceChange: Decimal;
            volumeChange: Decimal;
        };
        topLosers: array of {
            symbol: String;
            priceChange: Decimal;
            volumeChange: Decimal;
        };
        trendingCoins: array of {
            symbol: String;
            mentions: Integer;
            sentiment: Decimal;
        };
    };
    
    function getMarketIndicators(symbol: String) returns {
        rsi: Decimal;
        macd: {
            value: Decimal;
            signal: Decimal;
            histogram: Decimal;
        };
        movingAverages: {
            ma20: Decimal;
            ma50: Decimal;
            ma200: Decimal;
        };
        bollingerBands: {
            upper: Decimal;
            middle: Decimal;
            lower: Decimal;
        };
        volume: {
            current: Decimal;
            average: Decimal;
            trend: String;
        };
    };
    
    function getCorrelationMatrix(symbols: array of String) returns array of {
        symbol1: String;
        symbol2: String;
        correlation: Decimal;
        period: String;
    };
    
    function getVolatilityAnalysis(symbol: String, period: String) returns {
        historicalVolatility: Decimal;
        impliedVolatility: Decimal;
        volatilityRank: Decimal;
        volatilityPercentile: Decimal;
        atr: Decimal; // Average True Range
    };
    
    // Real-time Data Subscriptions (WebSocket info only)
    function getDataStreamInfo(dataType: String) returns {
        websocketUrl: String;
        supportedSymbols: array of String;
        updateFrequency: Integer;
        dataFormat: String;
    };
}