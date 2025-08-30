namespace com.рекс.cryptotrading.market;

using { cuid, managed } from '@sap/cds/common';

// Market Data Entities (Read-Only for Analysis)
entity MarketPairs : managed {
    key symbol     : String(20) @title: 'Symbol';
    baseCurrency   : String(10) @title: 'Base Currency';
    quoteCurrency  : String(10) @title: 'Quote Currency';
    displayName    : String(100) @title: 'Display Name';
    lastPrice      : Decimal(15,8) @title: 'Last Price';
    priceChange24h : Decimal(10,2) @title: '24h Price Change %';
    volume24h      : Decimal(20,8) @title: '24h Volume';
    marketCap      : Decimal(20,2) @title: 'Market Cap';
    circulatingSupply : Decimal(20,8) @title: 'Circulating Supply';
    totalSupply    : Decimal(20,8) @title: 'Total Supply';
    
    // Navigation
    priceHistory   : Composition of many PriceHistory on priceHistory.marketPair = $self;
}

entity PriceHistory : cuid, managed {
    marketPair     : Association to MarketPairs @title: 'Market Pair';
    timestamp      : Timestamp @title: 'Timestamp';
    open           : Decimal(15,8) @title: 'Open Price';
    high           : Decimal(15,8) @title: 'High Price';
    low            : Decimal(15,8) @title: 'Low Price';
    close          : Decimal(15,8) @title: 'Close Price';
    volume         : Decimal(20,8) @title: 'Volume';
    numberOfTrades : Integer @title: 'Number of Trades';
    timeframe      : String(10) @title: 'Timeframe'; // 1m, 5m, 15m, 1h, 4h, 1d, 1w
}

entity MarketData : managed {
    key id         : Integer @title: 'ID';
    symbol         : String(20) @title: 'Symbol';
    source         : String(50) @title: 'Data Source';
    open           : Decimal(15,8) @title: 'Open';
    high           : Decimal(15,8) @title: 'High';
    low            : Decimal(15,8) @title: 'Low';
    close          : Decimal(15,8) @title: 'Close';
    volume         : Decimal(20,8) @title: 'Volume';
    timestamp      : Timestamp @title: 'Timestamp';
    fetchedAt      : Timestamp @title: 'Fetched At';
}

entity MarketIndicators : cuid, managed {
    symbol         : String(20) @title: 'Symbol';
    timestamp      : Timestamp @title: 'Timestamp';
    rsi            : Decimal(5,2) @title: 'RSI';
    macd           : Decimal(15,8) @title: 'MACD';
    macdSignal     : Decimal(15,8) @title: 'MACD Signal';
    macdHistogram  : Decimal(15,8) @title: 'MACD Histogram';
    sma20          : Decimal(15,8) @title: 'SMA 20';
    sma50          : Decimal(15,8) @title: 'SMA 50';
    sma200         : Decimal(15,8) @title: 'SMA 200';
    ema12          : Decimal(15,8) @title: 'EMA 12';
    ema26          : Decimal(15,8) @title: 'EMA 26';
    bbUpper        : Decimal(15,8) @title: 'Bollinger Band Upper';
    bbMiddle       : Decimal(15,8) @title: 'Bollinger Band Middle';
    bbLower        : Decimal(15,8) @title: 'Bollinger Band Lower';
    atr            : Decimal(15,8) @title: 'Average True Range';
    volatility     : Decimal(10,4) @title: 'Volatility';
}

entity MarketSentiment : cuid, managed {
    symbol         : String(20) @title: 'Symbol';
    timestamp      : Timestamp @title: 'Timestamp';
    sentimentScore : Decimal(5,2) @title: 'Sentiment Score'; // -100 to 100
    fearGreedIndex : Integer @title: 'Fear & Greed Index'; // 0 to 100
    socialVolume   : Integer @title: 'Social Media Volume';
    newsVolume     : Integer @title: 'News Volume';
    redditMentions : Integer @title: 'Reddit Mentions';
    twitterMentions: Integer @title: 'Twitter Mentions';
    googleTrends   : Integer @title: 'Google Trends Score';
}

// Analytics Views
view TopCryptocurrencies as select from MarketPairs {
    *
} order by marketCap desc limit 100;

view TrendingPairs as select from MarketPairs {
    *
} where priceChange24h > 10 or priceChange24h < -10
order by abs(priceChange24h) desc;

view HighVolumePairs as select from MarketPairs {
    *
} order by volume24h desc limit 50;

view MarketOverview as select from MarketPairs {
    count(*) as totalPairs,
    sum(marketCap) as totalMarketCap,
    sum(volume24h) as totalVolume24h,
    avg(priceChange24h) as avgPriceChange24h
};