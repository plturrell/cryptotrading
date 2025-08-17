namespace com.рекс.cryptotrading.trading;

using { managed, cuid } from '@sap/cds/common';

entity TradingPairs : cuid, managed {
    symbol          : String(20) @title: 'Trading Pair';
    name           : String(100) @title: 'Full Name';
    baseAsset      : String(10) @title: 'Base Asset';
    quoteAsset     : String(10) @title: 'Quote Asset';
    currentPrice   : Decimal(15,8) @title: 'Current Price';
    priceChange24h : Decimal(15,8) @title: '24h Price Change';
    volume24h      : Decimal(20,8) @title: '24h Volume';
    marketCap      : Decimal(20,2) @title: 'Market Cap';
    isActive       : Boolean @title: 'Active';
    minOrderSize   : Decimal(15,8) @title: 'Minimum Order Size';
    maxOrderSize   : Decimal(15,8) @title: 'Maximum Order Size';
    tickSize       : Decimal(15,8) @title: 'Tick Size';
    
    // Navigation
    orders         : Composition of many Orders on orders.tradingPair = $self;
    priceHistory   : Composition of many PriceHistory on priceHistory.tradingPair = $self;
}

entity Orders : cuid, managed {
    orderId        : String(50) @title: 'Order ID';
    tradingPair    : Association to TradingPairs @title: 'Trading Pair';
    orderType      : String(10) @title: 'Order Type'; // BUY, SELL
    orderMethod    : String(20) @title: 'Order Method'; // MARKET, LIMIT, STOP_LOSS
    quantity       : Decimal(15,8) @title: 'Quantity';
    price          : Decimal(15,8) @title: 'Price';
    stopPrice      : Decimal(15,8) @title: 'Stop Price';
    filledQuantity : Decimal(15,8) @title: 'Filled Quantity';
    remainingQuantity : Decimal(15,8) @title: 'Remaining Quantity';
    averagePrice   : Decimal(15,8) @title: 'Average Fill Price';
    totalValue     : Decimal(20,2) @title: 'Total Value';
    fee            : Decimal(10,2) @title: 'Trading Fee';
    status         : String(20) @title: 'Status'; // PENDING, PARTIAL, FILLED, CANCELLED, REJECTED
    timeInForce    : String(10) @title: 'Time in Force'; // GTC, IOC, FOK
    clientOrderId  : String(50) @title: 'Client Order ID';
    executedAt     : Timestamp @title: 'Execution Time';
    cancelledAt    : Timestamp @title: 'Cancellation Time';
    
    // Navigation
    executions     : Composition of many OrderExecutions on executions.order = $self;
}

entity OrderExecutions : cuid, managed {
    order          : Association to Orders @title: 'Order';
    executionId    : String(50) @title: 'Execution ID';
    quantity       : Decimal(15,8) @title: 'Executed Quantity';
    price          : Decimal(15,8) @title: 'Execution Price';
    fee            : Decimal(10,2) @title: 'Execution Fee';
    executedAt     : Timestamp @title: 'Execution Time';
    tradeId        : String(50) @title: 'Trade ID';
}

entity PriceHistory : cuid, managed {
    tradingPair    : Association to TradingPairs @title: 'Trading Pair';
    timestamp      : Timestamp @title: 'Timestamp';
    openPrice      : Decimal(15,8) @title: 'Open Price';
    highPrice      : Decimal(15,8) @title: 'High Price';
    lowPrice       : Decimal(15,8) @title: 'Low Price';
    closePrice     : Decimal(15,8) @title: 'Close Price';
    volume         : Decimal(20,8) @title: 'Volume';
    timeframe      : String(10) @title: 'Timeframe'; // 1m, 5m, 15m, 1h, 4h, 1d, 1w
}

entity Portfolio : cuid, managed {
    userId         : String(50) @title: 'User ID';
    totalValue     : Decimal(20,2) @title: 'Total Portfolio Value';
    totalPnL       : Decimal(20,2) @title: 'Total P&L';
    dailyPnL       : Decimal(20,2) @title: 'Daily P&L';
    assetCount     : Integer @title: 'Number of Assets';
    lastUpdated    : Timestamp @title: 'Last Updated';
    
    // Navigation
    holdings       : Composition of many Holdings on holdings.portfolio = $self;
    transactions   : Composition of many Transactions on transactions.portfolio = $self;
}

entity Holdings : cuid, managed {
    portfolio      : Association to Portfolio @title: 'Portfolio';
    symbol         : String(20) @title: 'Asset Symbol';
    name           : String(100) @title: 'Asset Name';
    amount         : Decimal(15,8) @title: 'Amount Held';
    avgPrice       : Decimal(15,8) @title: 'Average Purchase Price';
    currentPrice   : Decimal(15,8) @title: 'Current Price';
    value          : Decimal(20,2) @title: 'Current Value';
    pnl            : Decimal(20,2) @title: 'Unrealized P&L';
    pnlPercent     : Decimal(10,4) @title: 'P&L Percentage';
    allocation     : Decimal(10,4) @title: 'Portfolio Allocation %';
    icon           : String(200) @title: 'Asset Icon URL';
}

entity Transactions : cuid, managed {
    portfolio      : Association to Portfolio @title: 'Portfolio';
    transactionId  : String(50) @title: 'Transaction ID';
    type           : String(10) @title: 'Transaction Type'; // BUY, SELL, DEPOSIT, WITHDRAW
    asset          : String(20) @title: 'Asset Symbol';
    amount         : Decimal(15,8) @title: 'Amount';
    price          : Decimal(15,8) @title: 'Price per Unit';
    total          : Decimal(20,2) @title: 'Total Value';
    fee            : Decimal(10,2) @title: 'Transaction Fee';
    status         : String(20) @title: 'Status'; // PENDING, COMPLETED, FAILED
    timestamp      : Timestamp @title: 'Transaction Time';
    orderId        : String(50) @title: 'Related Order ID';
}

entity OrderBook : cuid {
    tradingPair    : Association to TradingPairs @title: 'Trading Pair';
    side           : String(10) @title: 'Order Side'; // BUY, SELL
    price          : Decimal(15,8) @title: 'Price Level';
    amount         : Decimal(15,8) @title: 'Total Amount';
    total          : Decimal(20,2) @title: 'Total Value';
    orderCount     : Integer @title: 'Number of Orders';
    timestamp      : Timestamp @title: 'Last Updated';
}

entity MarketData : cuid {
    symbol         : String(20) @title: 'Symbol';
    name           : String(100) @title: 'Name';
    currentPrice   : Decimal(15,8) @title: 'Current Price';
    priceChange24h : Decimal(15,8) @title: '24h Change';
    priceChangePercent24h : Decimal(10,4) @title: '24h Change %';
    high24h        : Decimal(15,8) @title: '24h High';
    low24h         : Decimal(15,8) @title: '24h Low';
    volume24h      : Decimal(20,8) @title: '24h Volume';
    marketCap      : Decimal(20,2) @title: 'Market Cap';
    rank           : Integer @title: 'Market Cap Rank';
    lastUpdated    : Timestamp @title: 'Last Updated';
}

// Views for analytics
view ActiveOrders as select from Orders {
    *
} where status in ('PENDING', 'PARTIAL');

view CompletedOrders as select from Orders {
    *
} where status = 'FILLED';

view PortfolioSummary as select from Portfolio {
    userId,
    totalValue,
    totalPnL,
    dailyPnL,
    assetCount,
    lastUpdated
};

view TopCryptocurrencies as select from MarketData {
    *
} where rank <= 100 order by rank asc;
