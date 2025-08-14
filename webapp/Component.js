sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "sap/ui/model/json/JSONModel"
], function (UIComponent, Device, JSONModel) {
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
                
                // DEX Data - real GeckoTerminal data
                dexData: {
                    activePools: null,
                    topPairs: [],
                    totalLiquidity: null,
                    loading: true
                },
                
                // Risk Metrics - calculated from real portfolio data
                riskMetrics: {
                    valueAtRisk: null,
                    confidence: null,
                    maxDrawdown: null,
                    sharpeRatio: null,
                    loading: true
                }
            });
            this.setModel(oAppModel, "app");

            // Set up real-time data updates
            this._startDataPolling();
        },

        _startDataPolling: function() {
            // Initial load
            this._loadMarketData();
            this._loadAIAnalysis();
            this._loadDEXData();
            
            // Set up polling intervals
            setInterval(this._loadMarketData.bind(this), 30000); // 30 seconds
            setInterval(this._loadAIAnalysis.bind(this), 60000); // 1 minute
            setInterval(this._loadDEXData.bind(this), 45000); // 45 seconds
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
        
        _loadDEXData: function() {
            var that = this;
            var oModel = that.getModel("app");
            
            jQuery.ajax({
                url: "/api/market/dex/trending",
                type: "GET",
                success: function(data) {
                    // Update with real DEX data only
                    if (data.data && Array.isArray(data.data)) {
                        oModel.setProperty("/dexData", {
                            activePools: data.data.length || 0,
                            topPairs: data.data.slice(0, 3).map(item => item.name || "Unknown"),
                            totalLiquidity: null, // Would need separate API call
                            loading: false
                        });
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load DEX data:", error);
                    oModel.setProperty("/dexData/loading", false);
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
        }
    });
});