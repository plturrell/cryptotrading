sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "sap/ui/model/json/JSONModel"
], function (UIComponent, Device, JSONModel) {
    "use strict";

    return UIComponent.extend("com.рекс.cryptotrading.Component", {

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

            // Official SAP Data Model Structure
            var oAppModel = new JSONModel({
                // User Profile
                user: {
                    name: "Professional Trader",
                    role: "Senior Analyst",
                    greeting: "Good Morning"
                },
                
                // Wallet Integration
                wallet: {
                    address: "0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1",
                    balance: {
                        ETH: 0,
                        BTC: 0,
                        total: 0
                    },
                    connected: false
                },
                
                // Real-time Market Data
                marketData: {
                    btcPrice: "65,432",
                    btcChange: "+2.5",
                    btcIndicator: "Up",
                    ethPrice: "3,524",
                    ethChange: "+1.8",
                    ethIndicator: "Up",
                    totalMarketCap: "2.8T",
                    lastUpdated: new Date().toISOString()
                },
                
                // Portfolio Metrics
                portfolio: {
                    totalValue: "125,650",
                    change24h: "+12.5",
                    changePercent: "12.5",
                    positions: 8,
                    profitLoss: "+15,432"
                },
                
                // AI Analysis Results
                aiAnalysis: {
                    signal: "BUY",
                    confidence: "85",
                    recommendation: "Strong bullish momentum detected. RSI at 65 indicates healthy correction. MACD positive divergence.",
                    lastUpdate: new Date().toISOString()
                },
                
                // DEX Data
                dexData: {
                    activePools: "1,247",
                    topPairs: ["ETH/USDC", "BTC/ETH", "MATIC/USDC"],
                    totalLiquidity: "45.2B"
                },
                
                // Risk Metrics
                riskMetrics: {
                    valueAtRisk: "12.3",
                    confidence: "95",
                    maxDrawdown: "8.7",
                    sharpeRatio: "1.85"
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
            jQuery.ajax({
                url: "/api/market/overview?symbols=bitcoin,ethereum",
                type: "GET",
                success: function(data) {
                    var oModel = that.getModel("app");
                    var aTiles = oModel.getProperty("/tiles");
                    
                    if (data.symbols && data.symbols.bitcoin) {
                        var btcPrice = Math.round(data.symbols.bitcoin.prices.average);
                        aTiles[0].number = btcPrice.toLocaleString();
                        aTiles[0].info = data.symbols.bitcoin.market_data.price_change_percentage_24h > 0 ? "↑ " + data.symbols.bitcoin.market_data.price_change_percentage_24h.toFixed(2) + "%" : "↓ " + Math.abs(data.symbols.bitcoin.market_data.price_change_percentage_24h).toFixed(2) + "%";
                        aTiles[0].infoState = data.symbols.bitcoin.market_data.price_change_percentage_24h > 0 ? "Success" : "Error";
                        aTiles[0].stateArrow = data.symbols.bitcoin.market_data.price_change_percentage_24h > 0 ? "Up" : "Down";
                    }
                    
                    oModel.setProperty("/tiles", aTiles);
                },
                error: function() {
                    console.error("Failed to load market data");
                }
            });
        },
        
        _loadAIAnalysis: function() {
            var that = this;
            jQuery.ajax({
                url: "/api/ai/analyze",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    symbol: "BTC",
                    price: 65000,
                    volume_24h: 28500000000,
                    change_24h: 2.5,
                    indicators: {
                        rsi: 65,
                        macd: 150,
                        ma_50: 63000,
                        ma_200: 58000
                    }
                }),
                success: function(data) {
                    var oModel = that.getModel("app");
                    var aTiles = oModel.getProperty("/tiles");
                    
                    // Update AI tile
                    var aiTile = aTiles[3];
                    aiTile.number = data.confidence || "85";
                    aiTile.newsContent = [{
                        title: data.signal || "HOLD",
                        text: (data.analysis || "Analyzing market conditions...").substring(0, 100) + "..."
                    }];
                    aiTile.state = "Loaded";
                    
                    oModel.setProperty("/tiles", aTiles);
                },
                error: function() {
                    console.error("Failed to load AI analysis");
                }
            });
        },
        
        _loadDEXData: function() {
            var that = this;
            jQuery.ajax({
                url: "/api/market/dex/trending",
                type: "GET",
                success: function(data) {
                    var oModel = that.getModel("app");
                    var aTiles = oModel.getProperty("/tiles");
                    
                    // Update DEX tile
                    if (data.data && data.data.length > 0) {
                        aTiles[4].number = data.data.length.toLocaleString();
                        aTiles[4].info = "Trending";
                    }
                    
                    oModel.setProperty("/tiles", aTiles);
                },
                error: function() {
                    console.error("Failed to load DEX data");
                }
            });
        }
    });
});