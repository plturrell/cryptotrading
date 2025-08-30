sap.ui.define([
    "sap/ui/base/Object",
    "./SharedErrorHandlingUtils",
    "./SharedSecurityUtils"
], (BaseObject, ErrorHandler, SecurityUtils) => {
    "use strict";

    /**
     * Market Data Service for Crypto Trading Platform
     * Provides real-time market data management including:
     * - WebSocket connections for live price feeds
     * - Market data caching and optimization
     * - Symbol management and validation
     * - Historical data retrieval
     * - Market status monitoring
     */
    return BaseObject.extend("com.rex.cryptotrading.utils.MarketDataService", {

        constructor: function() {
            BaseObject.prototype.constructor.apply(this, arguments);
            this._websocket = null;
            this._subscriptions = new Map();
            this._cache = new Map();
            this._reconnectAttempts = 0;
            this._maxReconnectAttempts = 5;
            this._reconnectDelay = 1000;
            this._isConnected = false;
            this._eventBus = sap.ui.getCore().getEventBus();
        },

        /**
         * Initializes market data service
         * @param {Object} config - Service configuration
         */
        initialize(config = {}) {
            this._config = {
                wsUrl: config.wsUrl || "wss://stream.binance.com:9443/ws/btcusdt@ticker",
                apiUrl: config.apiUrl || "/api/market",
                cacheTimeout: config.cacheTimeout || 30000, // 30 seconds
                reconnectDelay: config.reconnectDelay || 1000,
                maxReconnectAttempts: config.maxReconnectAttempts || 5,
                ...config
            };

            this._initializeWebSocket();
            this._startCacheCleanup();
        },

        /**
         * Subscribes to real-time data for a symbol
         * @param {string} symbol - Trading symbol
         * @param {Function} callback - Data callback function
         * @returns {string} - Subscription ID
         */
        subscribe(symbol, callback) {
            const subscriptionId = this._generateSubscriptionId();
            const sanitizedSymbol = SecurityUtils.sanitizeData(symbol).toUpperCase();

            this._subscriptions.set(subscriptionId, {
                symbol: sanitizedSymbol,
                callback: callback,
                timestamp: Date.now()
            });

            // Send subscription message if connected
            if (this._isConnected && this._websocket) {
                this._sendSubscription(sanitizedSymbol);
            }

            // Return cached data immediately if available
            const cachedData = this._cache.get(sanitizedSymbol);
            if (cachedData && this._isCacheValid(cachedData)) {
                setTimeout(() => callback(cachedData.data), 0);
            }

            return subscriptionId;
        },

        /**
         * Unsubscribes from real-time data
         * @param {string} subscriptionId - Subscription ID to remove
         */
        unsubscribe(subscriptionId) {
            const subscription = this._subscriptions.get(subscriptionId);
            if (subscription) {
                this._subscriptions.delete(subscriptionId);

                // Check if no more subscriptions for this symbol
                const hasOtherSubs = Array.from(this._subscriptions.values())
                    .some(sub => sub.symbol === subscription.symbol);

                if (!hasOtherSubs && this._isConnected && this._websocket) {
                    this._sendUnsubscription(subscription.symbol);
                }
            }
        },

        /**
         * Gets current market data for a symbol
         * @param {string} symbol - Trading symbol
         * @returns {Promise} - Market data promise
         */
        async getMarketData(symbol) {
            const sanitizedSymbol = SecurityUtils.sanitizeData(symbol).toUpperCase();

            // Check cache firs
            const cachedData = this._cache.get(sanitizedSymbol);
            if (cachedData && this._isCacheValid(cachedData)) {
                return Promise.resolve(cachedData.data);
            }

            try {
                const response = await fetch(`${this._config.apiUrl}/ticker/${sanitizedSymbol}`, {
                    headers: SecurityUtils.createSecureHeaders()
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                const marketData = this._processMarketData(data);

                // Cache the data
                this._cache.set(sanitizedSymbol, {
                    data: marketData,
                    timestamp: Date.now()
                });

                return marketData;
            } catch (error) {
                ErrorHandler.handleMarketDataError(error, { symbol: sanitizedSymbol });
                throw error;
            }
        },

        /**
         * Gets historical data for a symbol
         * @param {string} symbol - Trading symbol
         * @param {string} timeframe - Timeframe (1m, 5m, 1h, 1d, etc.)
         * @param {number} limit - Number of data points
         * @returns {Promise} - Historical data promise
         */
        async getHistoricalData(symbol, timeframe = "1h", limit = 100) {
            const sanitizedSymbol = SecurityUtils.sanitizeData(symbol).toUpperCase();
            const sanitizedTimeframe = SecurityUtils.sanitizeData(timeframe);
            const sanitizedLimit = Math.min(Math.max(parseInt(limit, 10) || 100, 1), 1000);

            try {
                const response = await fetch(
                    `${this._config.apiUrl}/klines/${sanitizedSymbol}?interval=${sanitizedTimeframe}&limit=${sanitizedLimit}`,
                    {
                        headers: SecurityUtils.createSecureHeaders()
                    }
                );

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                return this._processHistoricalData(data);
            } catch (error) {
                ErrorHandler.handleMarketDataError(error, {
                    symbol: sanitizedSymbol,
                    timeframe: sanitizedTimeframe
                });
                throw error;
            }
        },

        /**
         * Gets list of available trading symbols
         * @returns {Promise} - Symbols list promise
         */
        async getSymbols() {
            try {
                const response = await fetch(`${this._config.apiUrl}/symbols`, {
                    headers: SecurityUtils.createSecureHeaders()
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                return data.symbols || [];
            } catch (error) {
                ErrorHandler.handleMarketDataError(error, { context: "symbols" });
                throw error;
            }
        },

        /**
         * Gets market status
         * @returns {Promise} - Market status promise
         */
        async getMarketStatus() {
            try {
                const response = await fetch(`${this._config.apiUrl}/status`, {
                    headers: SecurityUtils.createSecureHeaders()
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                return {
                    status: data.status || "unknown",
                    timestamp: data.timestamp || Date.now()
                };
            } catch (error) {
                ErrorHandler.handleMarketDataError(error, { context: "market_status" });
                return { status: "unknown", timestamp: Date.now() };
            }
        },

        /**
         * Initializes WebSocket connection
         * @private
         */
        _initializeWebSocket() {
            try {
                this._websocket = new WebSocket(this._config.wsUrl);

                this._websocket.onopen = () => {
                    this._isConnected = true;
                    this._reconnectAttempts = 0;
                    this._eventBus.publish("MarketData", "Connected", {});

                    // Resubscribe to all active subscriptions
                    this._resubscribeAll();
                };

                this._websocket.onmessage = (event) => {
                    this._handleWebSocketMessage(event);
                };

                this._websocket.onclose = () => {
                    this._isConnected = false;
                    this._eventBus.publish("MarketData", "Disconnected", {});
                    this._scheduleReconnect();
                };

                this._websocket.onerror = (error) => {
                    ErrorHandler.handleConnectionError(error, () => {
                        this._initializeWebSocket();
                    });
                };

            } catch (error) {
                ErrorHandler.handleConnectionError(error, () => {
                    this._scheduleReconnect();
                });
            }
        },

        /**
         * Handles WebSocket messages
         * @private
         */
        _handleWebSocketMessage(event) {
            try {
                const data = JSON.parse(event.data);
                const processedData = this._processMarketData(data);

                if (processedData.symbol) {
                    // Update cache
                    this._cache.set(processedData.symbol, {
                        data: processedData,
                        timestamp: Date.now()
                    });

                    // Notify subscribers
                    this._notifySubscribers(processedData.symbol, processedData);
                }
            } catch (error) {
                ErrorHandler.handleMarketDataError(error, { context: "websocket_message" });
            }
        },

        /**
         * Processes raw market data
         * @private
         */
        _processMarketData(rawData) {
            return {
                symbol: rawData.s || rawData.symbol,
                price: parseFloat(rawData.c || rawData.price || 0),
                change: parseFloat(rawData.P || rawData.priceChange || 0),
                changePercent: parseFloat(rawData.p || rawData.priceChangePercent || 0),
                volume: parseFloat(rawData.v || rawData.volume || 0),
                high: parseFloat(rawData.h || rawData.high || 0),
                low: parseFloat(rawData.l || rawData.low || 0),
                open: parseFloat(rawData.o || rawData.open || 0),
                timestamp: parseInt(rawData.E || rawData.timestamp || Date.now(), 10)
            };
        },

        /**
         * Processes historical data
         * @private
         */
        _processHistoricalData(rawData) {
            if (!Array.isArray(rawData)) {
                return [];
            }

            return rawData.map(item => ({
                timestamp: parseInt(item[0] || item.timestamp, 10),
                open: parseFloat(item[1] || item.open),
                high: parseFloat(item[2] || item.high),
                low: parseFloat(item[3] || item.low),
                close: parseFloat(item[4] || item.close),
                volume: parseFloat(item[5] || item.volume)
            }));
        },

        /**
         * Notifies subscribers of data updates
         * @private
         */
        _notifySubscribers(symbol, data) {
            for (const [id, subscription] of this._subscriptions.entries()) {
                if (subscription.symbol === symbol && subscription.callback) {
                    try {
                        subscription.callback(data);
                    } catch (error) {
                        console.error(`Error in subscription callback ${id}:`, error);
                    }
                }
            }
        },

        /**
         * Resubscribes to all active subscriptions
         * @private
         */
        _resubscribeAll() {
            const symbols = new Set();
            for (const subscription of this._subscriptions.values()) {
                symbols.add(subscription.symbol);
            }

            for (const symbol of symbols) {
                this._sendSubscription(symbol);
            }
        },

        /**
         * Sends subscription message
         * @private
         */
        _sendSubscription(symbol) {
            if (this._websocket && this._websocket.readyState === WebSocket.OPEN) {
                const message = {
                    method: "SUBSCRIBE",
                    params: [`${symbol.toLowerCase()}@ticker`],
                    id: Date.now()
                };
                this._websocket.send(JSON.stringify(message));
            }
        },

        /**
         * Sends unsubscription message
         * @private
         */
        _sendUnsubscription(symbol) {
            if (this._websocket && this._websocket.readyState === WebSocket.OPEN) {
                const message = {
                    method: "UNSUBSCRIBE",
                    params: [`${symbol.toLowerCase()}@ticker`],
                    id: Date.now()
                };
                this._websocket.send(JSON.stringify(message));
            }
        },

        /**
         * Schedules reconnection attemp
         * @private
         */
        _scheduleReconnect() {
            if (this._reconnectAttempts < this._maxReconnectAttempts) {
                this._reconnectAttempts++;
                const delay = this._reconnectDelay * Math.pow(2, this._reconnectAttempts - 1);

                setTimeout(() => {
                    this._initializeWebSocket();
                }, delay);
            }
        },

        /**
         * Checks if cached data is still valid
         * @private
         */
        _isCacheValid(cachedItem) {
            return (Date.now() - cachedItem.timestamp) < this._config.cacheTimeout;
        },

        /**
         * Starts cache cleanup interval
         * @private
         */
        _startCacheCleanup() {
            setInterval(() => {
                for (const [key, item] of this._cache.entries()) {
                    if (!this._isCacheValid(item)) {
                        this._cache.delete(key);
                    }
                }
            }, this._config.cacheTimeout);
        },

        /**
         * Generates unique subscription ID
         * @private
         */
        _generateSubscriptionId() {
            return "sub_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
        },

        /**
         * Destroys the service and cleans up resources
         */
        destroy() {
            if (this._websocket) {
                this._websocket.close();
                this._websocket = null;
            }

            this._subscriptions.clear();
            this._cache.clear();
            this._isConnected = false;
        }
    });
});
