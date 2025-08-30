// Interval tracking for memory leak prevention
const intervals = [];
const timeouts = [];

function trackInterval(intervalId) {
    intervals.push(intervalId);
    return intervalId;
}

function trackTimeout(timeoutId) {
    timeouts.push(timeoutId);
    return timeoutId;
}

function cleanupIntervals() {
    intervals.forEach(clearInterval);
    intervals.length = 0;
    timeouts.forEach(clearTimeout);
    timeouts.length = 0;
}

// Cleanup on exit
if (typeof process !== "undefined") {
    process.on("exit", cleanupIntervals);
    process.on("SIGTERM", cleanupIntervals);
    process.on("SIGINT", cleanupIntervals);
}

/**
 * Enterprise Component for Crypto Trading Application
 * Enhanced with memory leak prevention, performance monitoring, and enterprise patterns
 */

sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "sap/ui/model/json/JSONModel",
    "sap/base/Log",
    "./utils/SharedErrorHandlingUtils",
    "./utils/SharedSecurityUtils",
    "./utils/MarketDataService",
    "./utils/TradingService",
    "./utils/CacheManager"
], function (UIComponent, Device, JSONModel, Log, SharedErrorHandlingUtils, SharedSecurityUtils, MarketDataService, TradingService, CacheManager) {
    "use strict";

    return UIComponent.extend("com.rex.cryptotrading.Component", {

        metadata: {
            manifest: "json"
        },

        /**
         * The component is initialized by UI5 automatically during the startup of the app and calls the init method once.
         * @public
         * @override
         */
        init: function () {
            // Initialize cleanup tasks array
            this._aCleanupTasks = [];
            
            // Initialize performance monitoring
            this._initPerformanceMonitoring();
            
            // Initialize memory leak prevention
            this._initMemoryLeakPrevention();
            
            // Initialize shared utilities
            this._initSharedUtils();
            
            // Initialize services
            this._initServices();
            
            // Initialize WebSocket connections
            this._initWebSocketConnections();
            
            // Initialize global error handling
            this._initGlobalErrorHandling();
            
            // Initialize user preferences
            this._initUserPreferences();
            
            // Disable UI5 flexibility features for performance
            if (sap.ui.fl && sap.ui.fl.Utils) {
                if (typeof sap.ui.fl.Utils.isApplicationVariant === "undefined") {
                    sap.ui.fl.Utils.isApplicationVariant = function() { return false; };
                    sap.ui.fl.Utils.isVariantByStartupParameter = function() { return false; };
                }
                
                // Block Storage methods
                if (sap.ui.fl.apply && sap.ui.fl.apply._internal && sap.ui.fl.apply._internal.Storage) {
                    var Storage = sap.ui.fl.apply._internal.Storage;
                    Storage.loadFlexData = function() {
                        return Promise.resolve([]);
                    };
                }
                
                // Disable flexibility completely
                if (sap.ui.fl.Utils) {
                    sap.ui.fl.Utils.isApplicationVariant = function() {
                        return false;
                    };
                }
            }

            // Call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);

            // Initialize global error handling
            this._initializeErrorHandling();

            // Create device model
            this.setModel(this._createDeviceModel(), "device");

            // Create app model
            this.setModel(this._createAppModel(), "app");

            // Create trading model
            this.setModel(this._createTradingModel(), "trading");

            // Initialize services
            this._initializeServices();

            // Initialize performance monitoring
            this._initializePerformanceMonitoring();

            // Apply user preferences
            this._applyUserPreferences();

            // Enable routing
            this.getRouter().initialize();

            Log.info("Crypto Trading Platform Component initialized");
        },

        /**
         * The component is destroyed by UI5 automatically.
         * @public
         * @override
         */
        destroy: function () {
            // Execute cleanup tasks
            this._executeCleanupTasks();

            // Cleanup intervals and timeouts
            cleanupIntervals();

            // Destroy services
            if (this._marketDataService) {
                this._marketDataService.destroy();
                this._marketDataService = null;
            }

            if (this._tradingService) {
                this._tradingService.destroy();
                this._tradingService = null;
            }

            if (this._cacheManager) {
                this._cacheManager.destroy();
                this._cacheManager = null;
            }

            // Disconnect WebSocket
            if (this._websocket) {
                this._websocket.close();
                this._websocket = null;
            }

            // Call the base component's destroy function
            UIComponent.prototype.destroy.apply(this, arguments);

            Log.info("Crypto Trading Platform Component destroyed");
        },

        /**
         * Register a cleanup task to be executed on component destruction
         * @public
         * @param {function} fnCleanup Cleanup function
         */
        registerForCleanup: function (fnCleanup) {
            if (typeof fnCleanup === "function") {
                this._aCleanupTasks.push(fnCleanup);
            }
        },

        /**
         * Gets the market data service instance
         * @public
         * @returns {Object} Market data service
         */
        getMarketDataService: function () {
            return this._marketDataService;
        },

        /**
         * Gets the trading service instance
         * @public
         * @returns {Object} Trading service
         */
        getTradingService: function () {
            return this._tradingService;
        },

        /**
         * Gets the cache manager instance
         * @public
         * @returns {Object} Cache manager
         */
        getCacheManager: function () {
            return this._cacheManager;
        },

        /**
         * This method can be called to determine whether the sapUiSizeCompact or sapUiSizeCozy
         * design mode class should be set, which influences the size appearance of some controls.
         * @public
         * @returns {string} css class, either "sapUiSizeCompact" or "sapUiSizeCozy" -
         * or an empty string if no css class should be set
         */
        getContentDensityClass: function () {
            if (this._sContentDensityClass === undefined) {
                // Check whether FLP has already set the content density class; do nothing in this case
                if (document.body.classList.contains("sapUiSizeCozy") ||
                    document.body.classList.contains("sapUiSizeCompact")) {
                    this._sContentDensityClass = "";
                } else if (!Device.support.touch) { // Apply "compact" mode if touch is not supported
                    this._sContentDensityClass = "sapUiSizeCompact";
                } else {
                    // "cozy" in case of touch support; default for most sap.m controls,
                    // but needed for desktop-first controls like sap.ui.table.Table
                    this._sContentDensityClass = "sapUiSizeCozy";
                }
            }
            return this._sContentDensityClass;
        },

        /* =========================================================== */
        /* Internal Methods                                            */
        /* =========================================================== */

        /**
         * Creates the device model.
         * @private
         * @returns {sap.ui.model.json.JSONModel} the device model
         */
        _createDeviceModel: function () {
            const oModel = new JSONModel(Device);
            oModel.setDefaultBindingMode("OneWay");
            return oModel;
        },

        /**
         * Creates the app model.
         * @private
         * @returns {sap.ui.model.json.JSONModel} the app model
         */
        _createAppModel: function () {
            const oModel = new JSONModel({
                busy: false,
                delay: 0,
                currentUser: null,
                environment: this._getEnvironmentInfo(),
                connectionStatus: "disconnected",
                marketStatus: "closed",
                theme: "sap_horizon",
                language: "en",
                notifications: [],
                settings: {
                    autoRefresh: true,
                    refreshInterval: 5000,
                    soundEnabled: true,
                    darkMode: false
                }
            });
            oModel.setDefaultBindingMode("TwoWay");
            return oModel;
        },

        /**
         * Creates the trading model.
         * @private
         * @returns {sap.ui.model.json.JSONModel} the trading model
         */
        _createTradingModel: function () {
            const oModel = new JSONModel({
                portfolio: {
                    totalValue: 0,
                    availableBalance: 0,
                    totalPnL: 0,
                    totalPnLPercent: 0,
                    positions: []
                },
                watchlist: [],
                openOrders: [],
                recentTrades: [],
                marketData: {},
                alerts: [],
                performance: {
                    dailyPnL: 0,
                    weeklyPnL: 0,
                    monthlyPnL: 0,
                    winRate: 0,
                    totalTrades: 0
                }
            });
            oModel.setDefaultBindingMode("TwoWay");
            return oModel;
        },

        /**
         * Initializes all services
         * @private
         */
        _initializeServices: function () {
            try {
                // Initialize cache manager
                this._cacheManager = new CacheManager();
                this._cacheManager.initialize({
                    maxMemorySize: 100 * 1024 * 1024, // 100MB
                    cleanupInterval: 5 * 60 * 1000 // 5 minutes
                });

                // Initialize market data service
                this._marketDataService = new MarketDataService();
                this._marketDataService.initialize({
                    wsUrl: this._getWebSocketUrl(),
                    apiUrl: "/api/market",
                    cacheTimeout: 30000
                });

                // Initialize trading service
                this._tradingService = new TradingService();
                this._tradingService.initialize({
                    apiUrl: "/api/trading",
                    maxOrderSize: 1000000,
                    riskLimits: {
                        dailyLoss: 50000,
                        maxLeverage: 10,
                        maxOpenOrders: 50
                    }
                });

                // Initialize WebSocket connection
                this._initializeWebSocket();

                // Load initial data
                this._loadInitialData();

                Log.info("All services initialized successfully");
            } catch (error) {
                ErrorHandler.handleError(error, {
                    context: "service_initialization",
                    severity: ErrorHandler.SEVERITY.CRITICAL
                });
            }
        },

        /**
         * Initializes WebSocket connection for real-time updates
         * @private
         */
        _initializeWebSocket: function () {
            const wsUrl = this._getWebSocketUrl();
            if (!wsUrl) {
                Log.warning("WebSocket URL not configured - real-time updates disabled");
                return;
            }

            try {
                this._websocket = new WebSocket(wsUrl);

                this._websocket.onopen = () => {
                    Log.info("WebSocket connected");
                    this.getModel("app").setProperty("/connectionStatus", "connected");
                    
                    // Subscribe to essential data streams
                    this._subscribeToDataStreams();
                };

                this._websocket.onmessage = (event) => {
                    this._handleWebSocketMessage(event);
                };

                this._websocket.onclose = () => {
                    Log.warning("WebSocket disconnected");
                    this.getModel("app").setProperty("/connectionStatus", "disconnected");
                    this._scheduleReconnect();
                };

                this._websocket.onerror = (error) => {
                    ErrorHandler.handleConnectionError(error, () => {
                        this._initializeWebSocket();
                    });
                };

            } catch (error) {
                ErrorHandler.handleConnectionError(error, () => {
                    trackTimeout(setTimeout(() => {
                        this._initializeWebSocket();
                    }, 5000));
                });
            }
        },

        /**
         * Initializes error handling
         * @private
         */
        _initializeErrorHandling: function () {
            // Global error handler for unhandled promise rejections
            window.addEventListener("unhandledrejection", (event) => {
                ErrorHandler.handleError(event.reason, {
                    context: "unhandled_promise_rejection",
                    severity: ErrorHandler.SEVERITY.HIGH
                });
            });

            // Global error handler for JavaScript errors
            window.addEventListener("error", (event) => {
                ErrorHandler.handleError(event.error || event.message, {
                    context: "javascript_error",
                    severity: ErrorHandler.SEVERITY.MEDIUM
                });
            });
        },

        /**
         * Initializes performance monitoring
         * @private
         */
        _initializePerformanceMonitoring: function () {
            this._performanceMonitor = {
                startTime: performance.now(),
                metrics: {
                    componentInitTime: 0,
                    firstRenderTime: 0,
                    dataLoadTime: 0
                }
            };

            // Monitor component initialization time
            trackTimeout(setTimeout(() => {
                this._performanceMonitor.metrics.componentInitTime = 
                    performance.now() - this._performanceMonitor.startTime;
                Log.info(`Component initialization took ${this._performanceMonitor.metrics.componentInitTime}ms`);
            }, 0));
        },

        /**
         * Loads initial application data
         * @private
         */
        _loadInitialData: function () {
            const startTime = performance.now();

            Promise.allSettled([
                this._loadPortfolioData(),
                this._loadWatchlistData(),
                this._loadMarketStatus()
            ]).then(() => {
                this._performanceMonitor.metrics.dataLoadTime = performance.now() - startTime;
                Log.info(`Initial data loading took ${this._performanceMonitor.metrics.dataLoadTime}ms`);
            }).catch((error) => {
                ErrorHandler.handleError(error, {
                    context: "initial_data_load",
                    severity: ErrorHandler.SEVERITY.MEDIUM
                });
            });
        },

        /**
         * Loads portfolio data
         * @private
         */
        _loadPortfolioData: async function () {
            try {
                const portfolio = await this._tradingService.getPortfolio();
                this.getModel("trading").setProperty("/portfolio", portfolio);
            } catch (error) {
                ErrorHandler.handleTradingError(error, { context: "portfolio_load" });
            }
        },

        /**
         * Loads watchlist data
         * @private
         */
        _loadWatchlistData: async function () {
            try {
                const symbols = await this._marketDataService.getSymbols();
                const defaultWatchlist = symbols.slice(0, 10); // Top 10 symbols
                this.getModel("trading").setProperty("/watchlist", defaultWatchlist);
            } catch (error) {
                ErrorHandler.handleMarketDataError(error, { context: "watchlist_load" });
            }
        },

        /**
         * Loads market status
         * @private
         */
        _loadMarketStatus: async function () {
            try {
                const status = await this._marketDataService.getMarketStatus();
                this.getModel("app").setProperty("/marketStatus", status.status);
            } catch (error) {
                ErrorHandler.handleMarketDataError(error, { context: "market_status_load" });
            }
        },

        /**
         * Subscribes to real-time data streams
         * @private
         */
        _subscribeToDataStreams: function () {
            if (!this._websocket || this._websocket.readyState !== WebSocket.OPEN) {
                return;
            }

            // Subscribe to portfolio updates
            this._websocket.send(JSON.stringify({
                type: "subscribe",
                channels: ["portfolio", "orders", "trades"]
            }));
        },

        /**
         * Handles WebSocket messages
         * @private
         */
        _handleWebSocketMessage: function (event) {
            try {
                const data = JSON.parse(event.data);
                
                switch (data.type) {
                    case "portfolio_update":
                        this.getModel("trading").setProperty("/portfolio", data.payload);
                        break;
                    case "order_update":
                        this._handleOrderUpdate(data.payload);
                        break;
                    case "trade_update":
                        this._handleTradeUpdate(data.payload);
                        break;
                    case "market_data":
                        this._handleMarketDataUpdate(data.payload);
                        break;
                    default:
                        Log.info("Unknown WebSocket message type:", data.type);
                }
            } catch (error) {
                ErrorHandler.handleError(error, {
                    context: "websocket_message_handling",
                    severity: ErrorHandler.SEVERITY.LOW
                });
            }
        },

        /**
         * Handles order updates
         * @private
         */
        _handleOrderUpdate: function (orderData) {
            const tradingModel = this.getModel("trading");
            const openOrders = tradingModel.getProperty("/openOrders") || [];
            
            const existingIndex = openOrders.findIndex(order => order.id === orderData.id);
            if (existingIndex >= 0) {
                if (orderData.status === "FILLED" || orderData.status === "CANCELLED") {
                    openOrders.splice(existingIndex, 1);
                } else {
                    openOrders[existingIndex] = orderData;
                }
            } else if (orderData.status === "NEW" || orderData.status === "PARTIALLY_FILLED") {
                openOrders.push(orderData);
            }
            
            tradingModel.setProperty("/openOrders", openOrders);
        },

        /**
         * Handles trade updates
         * @private
         */
        _handleTradeUpdate: function (tradeData) {
            const tradingModel = this.getModel("trading");
            const recentTrades = tradingModel.getProperty("/recentTrades") || [];
            
            recentTrades.unshift(tradeData);
            if (recentTrades.length > 100) {
                recentTrades.splice(100); // Keep only last 100 trades
            }
            
            tradingModel.setProperty("/recentTrades", recentTrades);
        },

        /**
         * Handles market data updates
         * @private
         */
        _handleMarketDataUpdate: function (marketData) {
            const tradingModel = this.getModel("trading");
            const currentMarketData = tradingModel.getProperty("/marketData") || {};
            
            currentMarketData[marketData.symbol] = marketData;
            tradingModel.setProperty("/marketData", currentMarketData);
        },

        /**
         * Schedules WebSocket reconnection
         * @private
         */
        _scheduleReconnect: function () {
            trackTimeout(setTimeout(() => {
                if (!this._websocket || this._websocket.readyState === WebSocket.CLOSED) {
                    this._initializeWebSocket();
                }
            }, 5000));
        },

        /**
         * Applies user preferences
         * @private
         */
        _applyUserPreferences: function () {
            try {
                const preferences = JSON.parse(localStorage.getItem("cryptoTradingPreferences") || "{}");
                const appModel = this.getModel("app");
                
                if (preferences.theme) {
                    appModel.setProperty("/theme", preferences.theme);
                }
                if (preferences.language) {
                    appModel.setProperty("/language", preferences.language);
                }
                if (preferences.settings) {
                    appModel.setProperty("/settings", { ...appModel.getProperty("/settings"), ...preferences.settings });
                }
            } catch (error) {
                Log.warning("Failed to load user preferences:", error);
            }
        },

        /**
         * Gets environment information
         * @private
         */
        _getEnvironmentInfo: function () {
            return {
                hostname: window.location.hostname,
                protocol: window.location.protocol,
                userAgent: navigator.userAgent,
                timestamp: new Date().toISOString(),
                version: "1.0.0"
            };
        },

        /**
         * Gets WebSocket URL based on environment
         * @private
         */
        _getWebSocketUrl: function () {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const host = window.location.host;
            return `${protocol}//${host}/ws`;
        },

        /**
         * Execute all registered cleanup tasks
         * @private
         */
        _executeCleanupTasks: function () {
            this._aCleanupTasks.forEach((fnCleanup) => {
                try {
                    fnCleanup();
                } catch (error) {
                    Log.error("Error during cleanup task execution", error);
                }
            });
            this._aCleanupTasks = [];
        },

        /**
         * Register a cleanup task to be executed on component destruction
         * @public
         * @param {function} fnCleanup Cleanup function
         */
        registerForCleanup: function (fnCleanup) {
            if (typeof fnCleanup === "function") {
                this._aCleanupTasks.push(fnCleanup);
            }
        },

        /**
         * Gets the market data service instance
         * @public
         * @returns {Object} Market data service
         */
        getMarketDataService: function () {
            return this._marketDataService;
        },

        /**
         * Gets the trading service instance
         * @public
         * @returns {Object} Trading service
         */
        getTradingService: function () {
            return this._tradingService;
        },

        /**
         * Gets the cache manager instance
         * @public
         * @returns {Object} Cache manager
         */
        getCacheManager: function () {
            return this._cacheManager;
        }
    });
});

            // Enable routing after models are set up
            this.getRouter().initialize();

        },
        
        _initializeModelsAndData: function() {
            // Set device model
            var oDeviceModel = new JSONModel(Device);
            oDeviceModel.setDefaultBindingMode("OneWay");
            this.setModel(oDeviceModel, "device");
            
            // Initialize SAP-standard data managers first
            this._initializeDataManagers();

            // Real Data Model - No fake data, all loaded from APIs
            var oAppModel = new JSONModel({
                // User Profile - loaded after wallet connection
                user: {
                    name: "Trading User",
                    role: "Trader",
                    greeting: "Good morning"
                },
                
                // Wallet Integration - real MetaMask data only
                wallet: {
                    address: null,
                    balance: {
                        ETH: null,
                        BTC: null,
                        total: null
                    },
                    connected: false
                },
                
                // Real-time Market Data - loaded from APIs
                marketData: {
                    btcPrice: null,
                    btcChange: null,
                    btcIndicator: null,
                    ethPrice: null,
                    ethChange: null,
                    ethIndicator: null,
                    totalMarketCap: null,
                    lastUpdated: null,
                    loading: true
                },
                
                // Portfolio Metrics - calculated from real wallet data
                portfolio: {
                    totalValue: null,
                    change24h: null,
                    changePercent: null,
                    positions: null,
                    profitLoss: null,
                    loading: true
                },
                
                // AI Analysis Results - real Claude-4-Sonnet responses
                aiAnalysis: {
                    signal: null,
                    confidence: null,
                    recommendation: null,
                    lastUpdate: null,
                    loading: true
                },
                
                
                // Risk Metrics - calculated from real portfolio data
                riskMetrics: {
                    valueAtRisk: null,
                    confidence: null,
                    maxDrawdown: null,
                    sharpeRatio: null,
                    loading: true
                },
                
                // ML Predictions - real model predictions
                mlPredictions: {
                    btc: {
                        current_price: null,
                        predicted_price: null,
                        price_change_percent: null,
                        confidence: null,
                        model_accuracy: null,
                        loading: true
                    },
                    eth: {
                        current_price: null,
                        predicted_price: null,
                        price_change_percent: null,
                        confidence: null,
                        model_accuracy: null,
                        loading: true
                    },
                    models_trained: 0,
                    last_training: null,
                    loading: true
                },
                
                // Feature Store - ML feature engineering
                featureStore: {
                    total_features: null,
                    feature_importance: {},
                    last_computed: null,
                    loading: true
                }
            });
            this.setModel(oAppModel, "app");

            // Set up real-time data updates with SAP patterns
            this._startDataUpdates();
        },
        
        _startDataUpdates: function() {
            var that = this;
            
            // Start auto-refresh for market data
            this._oMarketDataManager.startAutoRefresh(30000);
            
            // Setup event handlers for data synchronization
            this._setupDataEventHandlers();
            
            // Load initial data
            this._loadInitialData();
        },
        
        _setupDataEventHandlers: function() {
            var that = this;
            
            // Subscribe to market data updates
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.REFRESH_REQUESTED,
                this._onMarketRefreshRequested.bind(this),
                this
            );
            
            // Subscribe to wallet events
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.WALLET,
                this._oEventBusManager.EVENTS.WALLET.CONNECTION_CHANGED,
                this._onWalletConnectionChanged.bind(this),
                this
            );
            
            // Subscribe to ML events
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.ML,
                this._oEventBusManager.EVENTS.ML.PREDICTIONS_UPDATED,
                this._onMLPredictionsUpdated.bind(this),
                this
            );
        },
        
        _initializeDataManagers: function() {
            var that = this;
            
            // Create EventBus manager for cross-component communication
            this._oEventBusManager = new EventBusManager();
            
            // Create specialized data managers
            this._oMarketDataManager = new MarketDataManager();
            this._oWalletDataManager = new WalletDataManager();
            this._oMLDataManager = new MLDataManager();
            
            // Set models on component
            this.setModel(this._oMarketDataManager.getModel(), "market");
            this.setModel(this._oWalletDataManager.getModel(), "wallet");
            this.setModel(this._oMLDataManager.getModel(), "ml");
            
            // Create global references for controllers (SAP pattern)
            sap.ui.getCore().setModel(this._oMarketDataManager.getModel(), "market");
            sap.ui.getCore().setModel(this._oWalletDataManager.getModel(), "wallet");
            sap.ui.getCore().setModel(this._oMLDataManager.getModel(), "ml");
            sap.ui.getCore().EventBusManager = this._oEventBusManager;
            
            // Subscribe to data manager changes to sync with legacy app model
            this._setupModelSynchronization();
            
            // Publish component ready event
            this._oEventBusManager.publishComponentReady("Component");
        },
        
        _setupModelSynchronization: function() {
            var that = this;
            
            // Sync market data changes
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.DATA_UPDATED,
                function(sChannel, sEvent, oData) {
                    that._syncMarketDataWithAppModel(oData.marketData);
                },
                this
            );
            
            // Sync wallet changes
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.WALLET,
                this._oEventBusManager.EVENTS.WALLET.CONNECTION_CHANGED,
                function(sChannel, sEvent, oData) {
                    that._syncWalletDataWithAppModel(oData);
                },
                this
            );
        },
        
        _syncMarketDataWithAppModel: function(oMarketData) {
            var oAppModel = this.getModel("app");
            
            // Sync market data to legacy app model structure
            if (oMarketData) {
                var oBtcData = oMarketData.BTC || {};
                var oEthData = oMarketData.ETH || {};
                
                oAppModel.setProperty("/marketData", {
                    btcPrice: oBtcData.price || null,
                    btcChange: oBtcData.priceChange24h || null,
                    btcIndicator: (oBtcData.priceChange24h || 0) > 0 ? "Up" : "Down",
                    ethPrice: oEthData.price || null,
                    ethChange: oEthData.priceChange24h || null,
                    ethIndicator: (oEthData.priceChange24h || 0) > 0 ? "Up" : "Down",
                    lastUpdated: new Date().toISOString(),
                    loading: false
                });
            }
        },
        
        _syncWalletDataWithAppModel: function(oWalletData) {
            var oAppModel = this.getModel("app");
            
            // Sync wallet data to legacy app model structure
            oAppModel.setProperty("/wallet/connected", oWalletData.connected || false);
            oAppModel.setProperty("/wallet/address", oWalletData.address || null);
        },
        
        _loadInitialData: function() {
            // Market data will be loaded automatically by MarketDataManager
            // Just trigger the initial refresh
            this._onMarketRefreshRequested();
            
            // Load other initial data
            this._loadAIAnalysis();
            this._loadFeatureStore();
        },
        
        _onMarketRefreshRequested: function() {
            var that = this;
            
            this._oMarketDataManager.setLoading(true);
            
            // Use jQuery to fetch market data (SAP standard)
            jQuery.ajax({
                url: "/api/market/overview?symbols=BTC,ETH,BNB,SOL,XRP",
                type: "GET",
                success: function(data) {
                    if (data && data.symbols) {
                        var oTransformedData = {};
                        Object.keys(data.symbols).forEach(function(key) {
                            var sSymbol = key.toUpperCase();
                            var oSymbolData = data.symbols[key];
                            oTransformedData[sSymbol] = {
                                price: oSymbolData.prices ? oSymbolData.prices.average : 0,
                                priceChange24h: oSymbolData.market_data ? oSymbolData.market_data.price_change_percentage_24h : 0,
                                volume24h: oSymbolData.market_data ? oSymbolData.market_data.total_volume : 0,
                                marketCap: oSymbolData.market_data ? oSymbolData.market_data.market_cap : 0,
                                high24h: oSymbolData.prices ? oSymbolData.prices.high : 0,
                                low24h: oSymbolData.prices ? oSymbolData.prices.low : 0
                            };
                        });
                        
                        that._oMarketDataManager.updateMarketData(oTransformedData);
                    }
                    
                    that._oMarketDataManager.setLoading(false);
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load market data:", error);
                    that._oMarketDataManager.setError(error);
                    that._oMarketDataManager.setLoading(false);
                }
            });
        },
        
        _onWalletConnectionChanged: function(sChannel, sEvent, oData) {
            // Handle wallet connection changes if needed
            console.log("Wallet connection changed:", oData.connected);
        },
        
        _onMLPredictionsUpdated: function(sChannel, sEvent, oData) {
            // Sync ML predictions with legacy app model
            var oAppModel = this.getModel("app");
            var oPredictions = oData.predictions || {};
            
            // Update BTC and ETH predictions in legacy format
            if (oPredictions.BTC) {
                oAppModel.setProperty("/mlPredictions/btc", {
                    current_price: oPredictions.BTC.currentPrice,
                    predicted_price: oPredictions.BTC.predictedPrice,
                    price_change_percent: oPredictions.BTC.predictedChange,
                    confidence: oPredictions.BTC.confidence,
                    loading: false
                });
            }
            
            if (oPredictions.ETH) {
                oAppModel.setProperty("/mlPredictions/eth", {
                    current_price: oPredictions.ETH.currentPrice,
                    predicted_price: oPredictions.ETH.predictedPrice,
                    price_change_percent: oPredictions.ETH.predictedChange,
                    confidence: oPredictions.ETH.confidence,
                    loading: false
                });
            }
        },
        

        _startDataPolling: function() {
            // Initial load
            this._loadMarketData();
            this._loadAIAnalysis();
            this._loadMLPredictions();
            this._loadFeatureStore();
            
            // Set up polling intervals
            setInterval(this._loadMarketData.bind(this), 30000); // 30 seconds
            setInterval(this._loadAIAnalysis.bind(this), 60000); // 1 minute
            setInterval(this._loadMLPredictions.bind(this), 300000); // 5 minutes
            setInterval(this._loadFeatureStore.bind(this), 600000); // 10 minutes
        },
        
        _loadMarketData: function() {
            var that = this;
            var oModel = that.getModel("app");
            
            jQuery.ajax({
                url: "/api/market/overview?symbols=bitcoin,ethereum",
                type: "GET",
                success: function(data) {
                    if (data.symbols && data.symbols.bitcoin) {
                        var btcData = data.symbols.bitcoin;
                        var ethData = data.symbols.ethereum;
                        
                        // Update with real data only
                        oModel.setProperty("/marketData", {
                            btcPrice: btcData.prices ? Math.round(btcData.prices.average) : null,
                            btcChange: btcData.market_data ? btcData.market_data.price_change_percentage_24h : null,
                            btcIndicator: btcData.market_data && btcData.market_data.price_change_percentage_24h > 0 ? "Up" : "Down",
                            ethPrice: ethData && ethData.prices ? Math.round(ethData.prices.average) : null,
                            ethChange: ethData && ethData.market_data ? ethData.market_data.price_change_percentage_24h : null,
                            ethIndicator: ethData && ethData.market_data && ethData.market_data.price_change_percentage_24h > 0 ? "Up" : "Down",
                            lastUpdated: new Date().toISOString(),
                            loading: false
                        });
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load market data:", error);
                    oModel.setProperty("/marketData/loading", false);
                }
            });
        },
        
        _loadAIAnalysis: function() {
            var that = this;
            var oModel = that.getModel("app");
            
            // Check if model exists and is initialized
            if (!oModel) {
                console.log("App model not yet available for AI analysis");
                return;
            }
            
            // Get current market data for AI analysis
            var marketData = oModel.getProperty("/marketData");
            if (!marketData || !marketData.btcPrice) {
                console.log("Waiting for market data before AI analysis");
                return;
            }
            
            jQuery.ajax({
                url: "/api/ai/analyze",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    symbol: "BTC",
                    price: marketData.btcPrice,
                    change_24h: marketData.btcChange || 0,
                    volume_24h: 0, // Will be filled with real data when available
                    indicators: {} // Real technical indicators would come from market data
                }),
                success: function(data) {
                    // Update with real AI response only
                    oModel.setProperty("/aiAnalysis", {
                        signal: data.signal || null,
                        confidence: data.confidence || null,
                        recommendation: data.analysis || null,
                        lastUpdate: new Date().toISOString(),
                        loading: false
                    });
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load AI analysis:", error);
                    oModel.setProperty("/aiAnalysis/loading", false);
                }
            });
        },
        
        
        // Add method to load real wallet data
        _loadWalletData: function(address) {
            if (!address) return;
            
            var that = this;
            var oModel = that.getModel("app");
            
            jQuery.ajax({
                url: "/api/wallet/balance",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({ address: address }),
                success: function(data) {
                    if (data.balance) {
                        oModel.setProperty("/wallet", {
                            address: address,
                            balance: {
                                ETH: data.balance.ETH || 0,
                                BTC: data.balance.BTC || 0,
                                total: data.totalValue || 0
                            },
                            connected: true
                        });
                        
                        // Calculate portfolio metrics from real wallet data
                        that._calculatePortfolioMetrics(data);
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load wallet data:", error);
                }
            });
        },
        
        // Calculate real portfolio metrics
        _calculatePortfolioMetrics: function(walletData) {
            var oModel = this.getModel("app");
            
            if (walletData && walletData.totalValue) {
                oModel.setProperty("/portfolio", {
                    totalValue: walletData.totalValue,
                    change24h: walletData.change24h || null,
                    changePercent: walletData.changePercent || null,
                    positions: Object.keys(walletData.balance || {}).length,
                    profitLoss: walletData.profitLoss || null,
                    loading: false
                });
            }
        },
        
        _loadMLPredictions: function() {
            var that = this;
            var oModel = that.getModel("app");
            
            // Load batch predictions for BTC and ETH
            jQuery.ajax({
                url: "/api/ml/predict/batch",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    symbols: ["BTC", "ETH"],
                    horizon: "24h",
                    model_type: "ensemble"
                }),
                success: function(predictions) {
                    if (Array.isArray(predictions)) {
                        predictions.forEach(function(pred) {
                            var symbol = pred.symbol.toLowerCase();
                            if (symbol === 'btc' || symbol === 'eth') {
                                oModel.setProperty("/mlPredictions/" + symbol, {
                                    current_price: pred.current_price,
                                    predicted_price: pred.predicted_price,
                                    price_change_percent: pred.price_change_percent,
                                    confidence: pred.confidence,
                                    model_accuracy: null, // Will be loaded separately
                                    loading: false
                                });
                            }
                        });
                        
                        oModel.setProperty("/mlPredictions/loading", false);
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load ML predictions:", error);
                    // Set error state instead of dummy values
                    oModel.setProperty("/mlPredictions/loading", false);
                    oModel.setProperty("/mlPredictions/error", true);
                    oModel.setProperty("/mlPredictions/errorMessage", "Unable to load predictions - models may need training");
                }
            });
            
            // Load model performance for BTC
            jQuery.ajax({
                url: "/api/ml/performance/BTC?horizon=24h",
                type: "GET",
                success: function(data) {
                    if (data && data.metrics && oModel) {
                        var accuracy = data.metrics.r2 ? (data.metrics.r2 * 100).toFixed(1) : null;
                        oModel.setProperty("/mlPredictions/btc/model_accuracy", accuracy);
                        
                        // Safe getProperty call with fallback
                        var currentCount = 0;
                        try {
                            currentCount = oModel.getProperty("/mlPredictions/models_trained") || 0;
                        } catch(e) {
                            console.warn("Could not get models_trained count:", e);
                        }
                        
                        oModel.setProperty("/mlPredictions/models_trained", currentCount + 1);
                        oModel.setProperty("/mlPredictions/last_training", data.last_trained);
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load model performance:", error);
                }
            });
        },
        
        _loadFeatureStore: function() {
            var that = this;
            var oModel = that.getModel("app");
            
            // Load feature information for BTC
            jQuery.ajax({
                url: "/api/ml/features/BTC?features=rsi_14,macd_signal,volatility_20,price_change_24h,volume_ratio_20",
                type: "GET",
                success: function(data) {
                    if (data && data.total_features) {
                        oModel.setProperty("/featureStore", {
                            total_features: data.total_features,
                            feature_importance: data.importance || {},
                            last_computed: new Date().toISOString(),
                            loading: false
                        });
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load feature store data:", error);
                    oModel.setProperty("/featureStore/loading", false);
                }
            });
        }
    });
});