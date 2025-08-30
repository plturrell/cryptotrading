sap.ui.define([
    "com/rex/cryptotrading/model/DataManager",
    "com/rex/cryptotrading/model/EventBusManager"
], function(DataManager, EventBusManager) {
    "use strict";

    /**
     * Market Data Manager - Specialized data manager for market data
     * Follows SAP UI5 patterns with structured data handling
     */
    return DataManager.extend("com.rex.cryptotrading.model.MarketDataManager", {
        
        varructor: function() {
            var oInitialData = this._getInitialMarketData();
            var oConfig = {
                enableChangeDetection: true,
                enableValidation: true,
                enableHistory: false // Market data changes too frequently for history
            };
            
            DataManager.prototype.varructor.call(this, "market", oInitialData, oConfig);
            
            this._oEventBusManager = new EventBusManager();
            this._mSymbolValidators = {};
            this._iRefreshInterval = 30000; // 30 seconds
            this._oRefreshTimer = null;
            
            this._initializeValidationSchema();
            this._setupEventHandlers();
        },
        
        /**
         * Start automatic data refresh
         * @param {number} iInterval - Refresh interval in milliseconds
         */
        startAutoRefresh: function(iInterval) {
            this.stopAutoRefresh();
            
            this._iRefreshInterval = iInterval || this._iRefreshInterval;
            var that = this;
            this._oRefreshTimer = setInterval(function() {
                that.refreshMarketData();
            }, this._iRefreshInterval);
            
            // Initial load
            this.refreshMarketData();
        },
        
        /**
         * Stop automatic data refresh
         */
        stopAutoRefresh: function() {
            if (this._oRefreshTimer) {
                clearInterval(this._oRefreshTimer);
                this._oRefreshTimer = null;
            }
        },
        
        /**
         * Update market data for multiple symbols
         * @param {Object} oMarketData - Market data object
         */
        updateMarketData: function(oMarketData) {
            var oCurrentData = this.getProperty("/symbols") || {};
            var oUpdatedData = Object.assign({}, oCurrentData);
            
            // Process each symbol
            var that = this;
            Object.keys(oMarketData).forEach(function(sSymbol) {
                var oSymbolData = oMarketData[sSymbol];
                var oOldData = oUpdatedData[sSymbol] || {};
                
                // Validate symbol data
                if (this._validateSymbolData(sSymbol, oSymbolData)) {
                    oUpdatedData[sSymbol] = this._processSymbolData(oSymbolData, oOldData);
                    
                    // Fire price change event if price changed
                    if (oOldData.price && oOldData.price !== oSymbolData.price) {
                        this._oEventBusManager.publishPriceChanged(
                            sSymbol, oOldData.price, oSymbolData.price
                        );
                    }
                }
            });
            
            // Update the model
            this.setProperty("/symbols", oUpdatedData);
            this.setProperty("/lastUpdate", new Date().toISOString());
            
            // Calculate and update market statistics
            this._updateMarketStatistics(oUpdatedData);
            
            // Publish market data updated event
            this._oEventBusManager.publishMarketDataUpdated(oUpdatedData);
        },
        
        /**
         * Update single symbol data
         * @param {string} sSymbol - Symbol to update
         * @param {Object} oSymbolData - Symbol data
         */
        updateSymbol: function(sSymbol, oSymbolData) {
            var sPath = "/symbols/" + sSymbol;
            var oOldData = this.getProperty(sPath) || {};
            
            if (this._validateSymbolData(sSymbol, oSymbolData)) {
                var oProcessedData = this._processSymbolData(oSymbolData, oOldData);
                this.updateObject(sPath, oProcessedData);
                
                // Fire price change event
                if (oOldData.price && oOldData.price !== oSymbolData.price) {
                    this._oEventBusManager.publishPriceChanged(
                        sSymbol, oOldData.price, oSymbolData.price
                    );
                }
            }
        },
        
        /**
         * Get symbol data
         * @param {string} sSymbol - Symbol to get
         * @returns {Object} Symbol data
         */
        getSymbolData: function(sSymbol) {
            return this.getProperty("/symbols/" + sSymbol) || {};
        },
        
        /**
         * Get all symbols
         * @returns {Array} Array of symbol names
         */
        getSymbols: function() {
            var oSymbols = this.getProperty("/symbols") || {};
            return Object.keys(oSymbols);
        },
        
        /**
         * Get market statistics
         * @returns {Object} Market statistics
         */
        getMarketStatistics: function() {
            return this.getProperty("/statistics") || {};
        },
        
        /**
         * Add symbol to watchlist
         * @param {string} sSymbol - Symbol to add
         */
        addToWatchlist: function(sSymbol) {
            this.addToArray("/watchlist", sSymbol);
        },
        
        /**
         * Remove symbol from watchlist
         * @param {string} sSymbol - Symbol to remove
         */
        removeFromWatchlist: function(sSymbol) {
            this.removeFromArray("/watchlist", (item) => item === sSymbol);
        },
        
        /**
         * Set connection status
         * @param {boolean} bConnected - Connection status
         */
        setConnectionStatus: function(bConnected) {
            this.setProperty("/connectionStatus", bConnected);
            this._oEventBusManager.publish(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.CONNECTION_STATUS,
                { connected: bConnected }
            );
        },
        
        /**
         * Set loading state
         * @param {boolean} bLoading - Loading state
         */
        setLoading: function(bLoading) {
            this.setProperty("/loading", bLoading);
        },
        
        /**
         * Set error state
         * @param {Object|string} vError - Error object or message
         */
        setError: function(vError) {
            var oError = typeof vError === 'string' ? { message: vError } : vError;
            this.setProperty("/error", oError);
            
            if (oError) {
                this._oEventBusManager.publish(
                    this._oEventBusManager.CHANNELS.MARKET,
                    this._oEventBusManager.EVENTS.MARKET.ERROR_OCCURRED,
                    { error: oError }
                );
            }
        },
        
        /**
         * Refresh market data
         */
        refreshMarketData: function() {
            this.setLoading(true);
            this.setError(null);
            
            // Fire refresh requested event
            this._oEventBusManager.publish(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.REFRESH_REQUESTED,
                { timestamp: new Date().toISOString() }
            );
            
            // Actual refresh logic would be handled by the component
            // This is just the data model management
        },
        
        /**
         * Get formatted price
         * @param {string} sSymbol - Symbol
         * @returns {string} Formatted price
         */
        getFormattedPrice: function(sSymbol) {
            var oSymbolData = this.getSymbolData(sSymbol);
            if (!oSymbolData.price) return "N/A";
            
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 2,
                maximumFractionDigits: oSymbolData.price > 1 ? 2 : 6
            }).format(oSymbolData.price);
        },
        
        /**
         * Get price change indicator
         * @param {string} sSymbol - Symbol
         * @returns {string} Indicator (Up, Down, Neutral)
         */
        getPriceIndicator: function(sSymbol) {
            var oSymbolData = this.getSymbolData(sSymbol);
            var nChange = oSymbolData.change24h || 0;
            
            if (nChange > 0) return "Up";
            if (nChange < 0) return "Down";
            return "Neutral";
        },
        
        /**
         * Initialize market data structure
         * @private
         */
        _getInitialMarketData: function() {
            return {
                symbols: {},
                statistics: {
                    totalMarketCap: 0,
                    totalVolume24h: 0,
                    activeSymbols: 0,
                    gainers: [],
                    losers: []
                },
                watchlist: ["BTC", "ETH", "BNB", "SOL", "XRP"],
                connectionStatus: false,
                loading: false,
                error: null,
                lastUpdate: null
            };
        },
        
        /**
         * Process symbol data with calculations
         * @private
         */
        _processSymbolData: function(oNewData, oOldData) {
            var oProcessed = Object.assign({}, oNewData);
            
            // Calculate additional fields
            if (oProcessed.price && oOldData.price) {
                oProcessed.priceChange = oProcessed.price - oOldData.price;
                oProcessed.priceChangePercent = 
                    ((oProcessed.price - oOldData.price) / oOldData.price * 100).toFixed(2);
            }
            
            // Add timestamp
            oProcessed.lastUpdate = new Date().toISOString();
            
            // Preserve price history (last 24 hours)
            oProcessed.priceHistory = oOldData.priceHistory || [];
            if (oProcessed.price) {
                oProcessed.priceHistory.push({
                    price: oProcessed.price,
                    timestamp: new Date().toISOString()
                });
                
                // Keep only last 24 hours (assuming 5-minute intervals)
                if (oProcessed.priceHistory.length > 288) {
                    oProcessed.priceHistory = oProcessed.priceHistory.slice(-288);
                }
            }
            
            return oProcessed;
        },
        
        /**
         * Update market statistics
         * @private
         */
        _updateMarketStatistics: function(oSymbolsData) {
            var aSymbols = Object.values(oSymbolsData);
            
            var oStatistics = {
                totalMarketCap: aSymbols.reduce((sum, symbol) => 
                    sum + (symbol.marketCap || 0), 0),
                totalVolume24h: aSymbols.reduce((sum, symbol) => 
                    sum + (symbol.volume24h || 0), 0),
                activeSymbols: aSymbols.length,
                gainers: aSymbols
                    .filter(symbol => (symbol.change24h || 0) > 0)
                    .sort((a, b) => (b.change24h || 0) - (a.change24h || 0))
                    .slice(0, 5),
                losers: aSymbols
                    .filter(symbol => (symbol.change24h || 0) < 0)
                    .sort((a, b) => (a.change24h || 0) - (b.change24h || 0))
                    .slice(0, 5)
            };
            
            this.updateObject("/statistics", oStatistics);
        },
        
        /**
         * Initialize validation schema
         * @private
         */
        _initializeValidationSchema: function() {
            this._oValidationSchema = {
                symbols: {
                    type: 'object',
                    required: false
                }
            };
            
            this._oSymbolSchema = {
                price: { type: 'number', required: true },
                change24h: { type: 'number', required: false },
                volume24h: { type: 'number', required: false },
                marketCap: { type: 'number', required: false },
                high24h: { type: 'number', required: false },
                low24h: { type: 'number', required: false }
            };
        },
        
        /**
         * Validate symbol data
         * @private
         */
        _validateSymbolData: function(sSymbol, oData) {
            if (!sSymbol || typeof sSymbol !== 'string') {
                console.error("Invalid symbol:", sSymbol);
                return false;
            }
            
            if (!oData || typeof oData !== 'object') {
                console.error("Invalid symbol data for", sSymbol);
                return false;
            }
            
            // Validate required fields
            if (typeof oData.price !== 'number' || oData.price <= 0) {
                console.error("Invalid price for", sSymbol, oData.price);
                return false;
            }
            
            return true;
        },
        
        /**
         * Setup event handlers
         * @private
         */
        _setupEventHandlers: function() {
            // Listen for system events that might affect market data
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.SYSTEM,
                this._oEventBusManager.EVENTS.SYSTEM.ERROR_GLOBAL,
                this._onSystemError.bind(this),
                this
            );
        },
        
        /**
         * Handle system errors
         * @private
         */
        _onSystemError: function(sChannel, sEvent, oData) {
            if (oData.component === 'market') {
                this.setError(oData.error);
                this.setLoading(false);
            }
        },
        
        /**
         * Cleanup
         */
        destroy: function() {
            this.stopAutoRefresh();
            
            if (this._oEventBusManager) {
                this._oEventBusManager.destroy();
            }
            
            DataManager.prototype.destroy.apply(this, arguments);
        }
    });
});