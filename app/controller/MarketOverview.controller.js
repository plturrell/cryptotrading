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
            // Initialize OData model for Trading Service
            this._oTradingModel = this.getOwnerComponent().getModel("trading");
            
            // Initialize EventBus and data managers (keep for backward compatibility)
            this._oEventBusManager = sap.ui.getCore().EventBusManager;
            this._oMarketModel = sap.ui.getCore().getModel("market");
            
            // Initialize local models
            this._initializeModels();
            
            // Setup event handlers
            this._setupEventHandlers();
            
            // Load initial market data from CDS service
            this._loadMarketDataFromCDS();
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
        },
        
        _loadMarketData: function() {
            // Use CDS service for market data
            this._loadMarketDataFromCDS();
            
            // Also trigger EventBus for backward compatibility
            if (this._oEventBusManager) {
                this._oEventBusManager.publish(
                    this._oEventBusManager.CHANNELS.MARKET,
                    this._oEventBusManager.EVENTS.MARKET.REFRESH_REQUESTED,
                    { timestamp: new Date().toISOString() }
                );
            }
        },
        
        _loadMarketDataFromCDS: function() {
            var that = this;
            
            // Call MarketDataService to get real market data (READ ONLY)
            jQuery.ajax({
                url: "/api/odata/v4/MarketDataService/MarketData",
                method: "GET",
                success: function(oData) {
                    that._updateMarketData(oData);
                },
                error: function(oError) {
                    console.error("Failed to load market data:", oError);
                }
            });
        },
        
        _updateMarketData: function(aMarketData) {
            var oLocalMarketModel = this.getView().getModel("localMarket");
            if (oLocalMarketModel && aMarketData) {
                // Update local model with market data
                aMarketData.forEach(function(oData) {
                    var sKey = oData.symbol.toLowerCase();
                    oLocalMarketModel.setProperty("/" + sKey, oData);
                });
            }
        },
        
        _updateMarketSummary: function(oSummary) {
            var oViewModel = this.getView().getModel("view");
            if (oViewModel && oSummary) {
                oViewModel.setProperty("/marketCap", oSummary.totalMarketCap);
                oViewModel.setProperty("/volume24h", oSummary.totalVolume24h);
                oViewModel.setProperty("/btcDominance", oSummary.btcDominance);
                oViewModel.setProperty("/fearGreedIndex", oSummary.fearGreedIndex);
            }
        },
        
        _updateTradingPairs: function(aTradingPairs) {
            var oLocalMarketModel = this.getView().getModel("localMarket");
            if (oLocalMarketModel && aTradingPairs) {
                aTradingPairs.forEach(function(oPair) {
                    var sKey = oPair.base_currency.toLowerCase();
                    oLocalMarketModel.setProperty("/" + sKey, {
                        price: oPair.current_price || 0,
                        change24h: oPair.price_change_24h || 0,
                        volume: oPair.volume_24h || 0,
                        high24h: oPair.high_24h || 0,
                        low24h: oPair.low_24h || 0
                    });
                });
            }
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
            
            // Display market data only - no actual trading
            MessageToast.show("Viewing " + sCrypto + " market data");
            
            // Navigate to detailed view or refresh data
            this._loadMarketDataFromCDS();
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
