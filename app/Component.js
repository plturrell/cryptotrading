sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "sap/ui/model/json/JSONModel",
    "com/rex/cryptotrading/model/EventBusManager",
    "com/rex/cryptotrading/model/MarketDataManager",
    "com/rex/cryptotrading/model/WalletDataManager",
    "com/rex/cryptotrading/model/MLDataManager",
], function (UIComponent, Device, JSONModel, EventBusManager, MarketDataManager, 
             WalletDataManager, MLDataManager) {
    "use strict";

    return UIComponent.extend("com.rex.cryptotrading.Component", {

        metadata: {
            manifest: "json"
        },

        init: function () {
            // Call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);

            // Enable routing
            this.getRouter().initialize();

            // Set device model
            var oDeviceModel = new JSONModel(Device);
            oDeviceModel.setDefaultBindingMode("OneWay");
            this.setModel(oDeviceModel, "device");
            
            // Initialize SAP-standard data managers
            this._initializeDataManagers();

            // Real Data Model - No fake data, all loaded from APIs
            var oAppModel = new JSONModel({
                // User Profile - loaded after wallet connection
                user: {
                    name: null,
                    role: null,
                    greeting: null
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
                },
                
                // Launchpad tiles configuration
                tiles: [
                    {
                        title: "Market Analysis",
                        subtitle: "Real-time market data",
                        number: "6",
                        unit: "pairs",
                        state: "Good",
                        icon: "sap-icon://line-chart"
                    },
                    {
                        title: "Portfolio",
                        subtitle: "Your holdings",
                        number: "0",
                        unit: "USD",
                        state: "Neutral",
                        icon: "sap-icon://wallet"
                    },
                    {
                        title: "Trading",
                        subtitle: "Execute trades",
                        number: "0",
                        unit: "orders",
                        state: "Neutral", 
                        icon: "sap-icon://sales-order"
                    }
                ]
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
                    if (data && data.metrics) {
                        var accuracy = data.metrics.r2 ? (data.metrics.r2 * 100).toFixed(1) : null;
                        oModel.setProperty("/mlPredictions/btc/model_accuracy", accuracy);
                        oModel.setProperty("/mlPredictions/models_trained", 
                            (oModel.getProperty("/mlPredictions/models_trained") || 0) + 1);
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