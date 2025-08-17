sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function(Controller, MessageToast, MessageBox, Fragment, JSONModel) {
    "use strict";

    return Controller.extend("com.rex.cryptotrading.controller.MarketOverview", {
        
        onInit: function() {
            // Initialize EventBus and data managers
            this._oEventBusManager = sap.ui.getCore().EventBusManager;
            this._oMarketModel = sap.ui.getCore().getModel("market");
            
            // Initialize local models
            this._initializeModels();
            
            // Setup event handlers
            this._setupEventHandlers();
        },
        
        onExit: function() {
            // Cleanup EventBus subscriptions
            if (this._oEventBusManager) {
                this._oEventBusManager.unsubscribeAll(this);
            }
        },
        
        _setupEventHandlers: function() {
            // Subscribe to market data updates
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.DATA_UPDATED,
                this._onMarketDataUpdated.bind(this),
                this
            );
            
            // Subscribe to price changes
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.PRICE_CHANGED,
                this._onPriceChanged.bind(this),
                this
            );
        },
        
        _onMarketDataUpdated: function(sChannel, sEvent, oData) {
            // Update local view model when market data changes
            var oMarketModel = this.getView().getModel("market");
            var oMarketData = oData.marketData || {};
            
            // Update model with new data
            Object.keys(oMarketData).forEach(function(sSymbol) {
                var oSymbolData = oMarketData[sSymbol];
                oMarketModel.setProperty("/" + sSymbol.toLowerCase(), {
                    price: oSymbolData.price,
                    change24h: oSymbolData.priceChange24h,
                    volume: oSymbolData.volume24h,
                    marketCap: oSymbolData.marketCap,
                    high24h: oSymbolData.high24h,
                    low24h: oSymbolData.low24h
                });
            });
            
            MessageToast.show("Market data updated");
        },
        
        _onPriceChanged: function(sChannel, sEvent, oData) {
            // Handle individual price changes
            console.log("Price changed for " + oData.symbol + ": " + oData.newPrice);
        },
        
        _initializeModels: function() {
            // Create local view model for this controller
            var oViewModel = new JSONModel({
                refreshing: false,
                lastRefresh: null,
                selectedTimeframe: "24h",
                viewMode: "grid" // or "list"
            });
            this.getView().setModel(oViewModel, "view");
            
            // The market model is already set globally, just reference it
            // Local initialization with default data
            var oLocalMarketModel = new JSONModel({
                btc: { price: 0, change24h: 0 },
                eth: { price: 0, change24h: 0 },
                bnb: { price: 0, change24h: 0 },
                sol: { price: 0, change24h: 0 },
                xrp: { price: 0, change24h: 0 }
            });
            this.getView().setModel(oLocalMarketModel, "localMarket");
        }
        
        _loadMarketData: function() {
            // Trigger market data refresh via EventBus
            this._oEventBusManager.publish(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.REFRESH_REQUESTED,
                { timestamp: new Date().toISOString() }
            );
        },
        
        onRefreshPress: function() {
            // Manual refresh button handler
            this._loadMarketData();
        },
        
        onSymbolPress: function(oEvent) {
            // Handle symbol tile press
            var oBindingContext = oEvent.getSource().getBindingContext("market");
            var sSymbol = oBindingContext.getProperty("symbol");
            
            MessageToast.show("Viewing details for " + sSymbol);
            
            // Could navigate to detailed view
            // this.getRouter().navTo("symbolDetail", { symbol: sSymbol });
        },
        
        onExit: function() {
            if (this._refreshTimer) {
                clearInterval(this._refreshTimer);
            }
        },
        
        onNavBack: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("launchpad");
        },
        
        onRefreshData: function() {
            this._loadMarketData();
            MessageToast.show("Market data refreshed");
        },
        
        onCryptoPress: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("market");
            var sCrypto = oContext.getProperty("symbol");
            MessageToast.show("Opening " + sCrypto + " details");
        },
        
        onBuyCrypto: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("market");
            var sCrypto = oContext.getProperty("symbol");
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("tradingConsole", { crypto: sCrypto });
        },
        
        // Formatters
        formatPriceColor: function(fChange) {
            if (fChange > 0) return "Good";
            if (fChange < 0) return "Error";
            return "Neutral";
        },
        
        formatIndicator: function(fChange) {
            if (fChange > 0) return "Up";
            if (fChange < 0) return "Down";
            return "None";
        },
        
        formatSentimentColor: function(iIndex) {
            if (iIndex >= 75) return "Good";
            if (iIndex >= 50) return "Critical";
            if (iIndex >= 25) return "Error";
            return "Error";
        },
        
        formatChangeState: function(fChange) {
            if (fChange > 0) return "Success";
            if (fChange < 0) return "Error";
            return "None";
        }
    });
});
