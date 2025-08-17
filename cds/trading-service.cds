using { com.рекс.cryptotrading.trading as trading } from './trading-model';

service TradingService @(path: '/api/odata/v4/TradingService') {
    
    // Core Trading Entities
    @odata.draft.enabled
    entity TradingPairs as projection on trading.TradingPairs;
    
    @odata.draft.enabled
    @cds.redirection.target
    entity Orders as projection on trading.Orders;
    
    entity OrderExecutions as projection on trading.OrderExecutions;
    
    entity PriceHistory as projection on trading.PriceHistory;
    
    // Portfolio Management
    @odata.draft.enabled
    entity Portfolio as projection on trading.Portfolio;
    
    entity Holdings as projection on trading.Holdings;
    
    entity Transactions as projection on trading.Transactions;
    
    // Market Data
    entity OrderBook as projection on trading.OrderBook;
    
    entity MarketData as projection on trading.MarketData;
    
    // Analytics Views
    @readonly
    entity ActiveOrders as projection on trading.ActiveOrders;
    
    @readonly
    entity CompletedOrders as projection on trading.CompletedOrders;
    
    @readonly
    entity PortfolioSummary as projection on trading.PortfolioSummary;
    
    @readonly
    entity TopCryptocurrencies as projection on trading.TopCryptocurrencies;
    
    // Trading Actions
    action submitOrder(
        tradingPair: String,
        orderType: String,
        orderMethod: String,
        quantity: Decimal,
        price: Decimal,
        stopPrice: Decimal,
        timeInForce: String
    ) returns {
        orderId: String;
        status: String;
        message: String;
    };
    
    action cancelOrder(orderId: String) returns {
        success: Boolean;
        message: String;
    };
    
    action cancelAllOrders(tradingPair: String) returns {
        cancelledCount: Integer;
        message: String;
    };
    
    action quickTrade(
        symbol: String,
        orderType: String,
        amount: Decimal
    ) returns {
        orderId: String;
        executedPrice: Decimal;
        message: String;
    };
    
    // Portfolio Actions
    action rebalancePortfolio(
        targetAllocations: array of {
            symbol: String;
            targetPercent: Decimal;
        }
    ) returns {
        rebalanceId: String;
        ordersCreated: Integer;
        message: String;
    };
    
    action calculatePortfolioMetrics(portfolioId: String) returns {
        totalValue: Decimal;
        totalPnL: Decimal;
        dailyPnL: Decimal;
        sharpeRatio: Decimal;
        maxDrawdown: Decimal;
        volatility: Decimal;
    };
    
    // Market Data Functions
    function getOrderBook(tradingPair: String) returns array of {
        side: String;
        price: Decimal;
        amount: Decimal;
        total: Decimal;
    };
    
    function getPriceHistory(
        tradingPair: String,
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
    
    function getMarketSummary() returns {
        totalMarketCap: Decimal;
        totalVolume24h: Decimal;
        btcDominance: Decimal;
        fearGreedIndex: Integer;
        activeMarkets: Integer;
    };
    
    // Risk Management Functions
    function calculateRiskMetrics(portfolioId: String) returns {
        valueAtRisk: Decimal;
        expectedShortfall: Decimal;
        beta: Decimal;
        correlation: Decimal;
        concentration: Decimal;
    };
    
    function validateOrder(
        tradingPair: String,
        orderType: String,
        quantity: Decimal,
        price: Decimal
    ) returns {
        isValid: Boolean;
        errors: array of String;
        warnings: array of String;
        estimatedFee: Decimal;
        estimatedTotal: Decimal;
    };
    
    // Analytics Functions
    function getPortfolioPerformance(
        portfolioId: String,
        period: String
    ) returns {
        returns: Decimal;
        volatility: Decimal;
        sharpeRatio: Decimal;
        maxDrawdown: Decimal;
        winRate: Decimal;
        profitFactor: Decimal;
    };
    
    function getTradingStatistics(
        userId: String,
        period: String
    ) returns {
        totalTrades: Integer;
        winningTrades: Integer;
        losingTrades: Integer;
        totalVolume: Decimal;
        totalFees: Decimal;
        averageHoldTime: Decimal;
        bestTrade: Decimal;
        worstTrade: Decimal;
    };
    
    // Real-time Data Subscriptions
    function subscribeToPrice(tradingPair: String) returns {
        subscriptionId: String;
        websocketUrl: String;
    };
    
    function subscribeToOrderBook(tradingPair: String) returns {
        subscriptionId: String;
        websocketUrl: String;
    };
    
    function subscribeToTrades(tradingPair: String) returns {
        subscriptionId: String;
        websocketUrl: String;
    };
}
