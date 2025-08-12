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

            // Set initial app data model with professional SAP tiles
            var oAppModel = new JSONModel({
                wallet: {
                    address: "0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1",
                    balance: 0,
                    connected: false
                },
                user: {
                    name: "Professional Trader",
                    role: "Senior Analyst"
                },
                tiles: [
                    {
                        title: "Market Overview",
                        subtitle: "Real-time cryptocurrency prices",
                        number: "65,432",
                        numberUnit: "USD",
                        info: "Bitcoin",
                        infoState: "Success",
                        icon: "sap-icon://line-chart-dual-axis",
                        type: "Monitor",
                        frameType: "TwoByOne",
                        state: "Loaded",
                        stateArrow: "Up",
                        targetParams: "display",
                        press: "marketOverview"
                    },
                    {
                        title: "Portfolio Management",
                        subtitle: "Total holdings and performance",
                        number: "125,650",
                        numberUnit: "USD",
                        info: "+12.5%",
                        infoState: "Success",
                        icon: "sap-icon://wallet",
                        type: "Monitor",
                        frameType: "OneByOne",
                        state: "Loaded",
                        stateArrow: "Up",
                        press: "portfolio"
                    },
                    {
                        title: "Trading Console",
                        subtitle: "Execute buy and sell orders",
                        number: "24",
                        numberUnit: "Active",
                        info: "Orders",
                        infoState: "None",
                        icon: "sap-icon://sales-order",
                        type: "Create",
                        frameType: "OneByOne",
                        state: "Loaded",
                        press: "trading"
                    },
                    {
                        title: "AI Market Intelligence",
                        subtitle: "Claude-4-Sonnet analysis engine",
                        number: "85",
                        numberUnit: "%",
                        info: "Confidence",
                        infoState: "Success",
                        icon: "sap-icon://business-objects-experience",
                        type: "Monitor",
                        frameType: "TwoByOne",
                        state: "Loading",
                        newsContent: [{
                            title: "BTC Analysis",
                            text: "Strong bullish momentum detected. RSI oversold conditions resolved."
                        }],
                        press: "aiAnalysis"
                    },
                    {
                        title: "DEX Analytics",
                        subtitle: "Decentralized exchange monitor",
                        number: "1,247",
                        numberUnit: "Pools",
                        info: "Active",
                        infoState: "None",
                        icon: "sap-icon://chain-link",
                        type: "Monitor",
                        frameType: "OneByOne",
                        state: "Loaded",
                        press: "dexMonitor"
                    },
                    {
                        title: "Risk Management",
                        subtitle: "Portfolio risk metrics",
                        number: "12.3",
                        numberUnit: "% VaR",
                        info: "95% CI",
                        infoState: "Error",
                        icon: "sap-icon://alert",
                        type: "Monitor",
                        frameType: "OneByOne",
                        state: "Loaded",
                        stateArrow: "Up",
                        press: "riskAnalytics"
                    },
                    {
                        title: "Historical Data",
                        subtitle: "Download market datasets",
                        number: "2.3",
                        numberUnit: "TB",
                        info: "Available",
                        infoState: "None",
                        icon: "sap-icon://download",
                        type: "Create",
                        frameType: "OneByOne",
                        state: "Loaded",
                        press: "historicalData"
                    },
                    {
                        title: "System Settings",
                        subtitle: "Configure platform parameters",
                        icon: "sap-icon://action-settings",
                        type: "Create",
                        frameType: "OneByOne",
                        state: "Loaded",
                        press: "settings"
                    }
                ]
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