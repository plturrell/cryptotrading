sap.ui.define([
    "sap/ui/base/Object",
    "./SharedErrorHandlingUtils",
    "./SharedSecurityUtils"
], (BaseObject, ErrorHandler, SecurityUtils) => {
    "use strict";

    /**
     * Trading Service for Crypto Trading Platform
     * Provides trading functionality including:
     * - Order management (place, cancel, modify)
     * - Portfolio tracking and managemen
     * - Risk management and validation
     * - Trading history and analytics
     * - Position managemen
     */
    return BaseObject.extend("com.rex.cryptotrading.utils.TradingService", {

        constructor: function() {
            BaseObject.prototype.constructor.apply(this, arguments);
            this._orders = new Map();
            this._positions = new Map();
            this._eventBus = sap.ui.getCore().getEventBus();
        },

        /**
         * Initializes trading service
         * @param {Object} config - Service configuration
         */
        initialize(config = {}) {
            this._config = {
                apiUrl: config.apiUrl || "/api/trading",
                maxOrderSize: config.maxOrderSize || 1000000, // $1M defaul
                maxPositionSize: config.maxPositionSize || 5000000, // $5M defaul
                riskLimits: {
                    dailyLoss: config.dailyLossLimit || 50000, // $50K
                    maxLeverage: config.maxLeverage || 10,
                    maxOpenOrders: config.maxOpenOrders || 50
                },
                ...config
            };
        },

        /**
         * Places a new trading order
         * @param {Object} orderData - Order data
         * @returns {Promise} - Order placement promise
         */
        async placeOrder(orderData) {
            try {
                // Validate order data
                const validation = SecurityUtils.validateTradingData(orderData, "order");
                if (!validation.isValid) {
                    throw new Error(`Order validation failed: ${validation.errors.join(", ")}`);
                }

                const sanitizedOrder = validation.sanitizedData;

                // Risk checks
                await this._performRiskChecks(sanitizedOrder);

                // Place order via API
                const response = await fetch(`${this._config.apiUrl}/orders`, {
                    method: "POST",
                    headers: SecurityUtils.createSecureHeaders({
                        csrfToken: this._getCsrfToken(),
                        correlationId: SecurityUtils.generateSecureRandom(16)
                    }),
                    body: JSON.stringify(sanitizedOrder)
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.message || `HTTP ${response.status}`);
                }

                const orderResult = await response.json();
                const processedOrder = this._processOrderResponse(orderResult);

                // Store order locally
                this._orders.set(processedOrder.id, processedOrder);

                // Notify listeners
                this._eventBus.publish("Trading", "OrderPlaced", { order: processedOrder });

                return processedOrder;

            } catch (error) {
                ErrorHandler.handleTradingError(error, {
                    symbol: orderData.symbol,
                    side: orderData.side,
                    type: orderData.type
                });
                throw error;
            }
        },

        /**
         * Cancels an existing order
         * @param {string} orderId - Order ID to cancel
         * @returns {Promise} - Cancellation promise
         */
        async cancelOrder(orderId) {
            try {
                const sanitizedOrderId = SecurityUtils.sanitizeData(orderId);

                const response = await fetch(`${this._config.apiUrl}/orders/${sanitizedOrderId}`, {
                    method: "DELETE",
                    headers: SecurityUtils.createSecureHeaders({
                        csrfToken: this._getCsrfToken(),
                        correlationId: SecurityUtils.generateSecureRandom(16)
                    })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.message || `HTTP ${response.status}`);
                }

                const result = await response.json();

                // Update local order status
                if (this._orders.has(orderId)) {
                    const order = this._orders.get(orderId);
                    order.status = "CANCELLED";
                    order.updatedAt = Date.now();
                }

                // Notify listeners
                this._eventBus.publish("Trading", "OrderCancelled", { orderId: orderId });

                return result;

            } catch (error) {
                ErrorHandler.handleTradingError(error, { orderId: orderId });
                throw error;
            }
        },

        /**
         * Gets current portfolio
         * @returns {Promise} - Portfolio data promise
         */
        async getPortfolio() {
            try {
                const response = await fetch(`${this._config.apiUrl}/portfolio`, {
                    headers: SecurityUtils.createSecureHeaders({
                        correlationId: SecurityUtils.generateSecureRandom(16)
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const portfolioData = await response.json();
                return this._processPortfolioData(portfolioData);

            } catch (error) {
                ErrorHandler.handleTradingError(error, { context: "portfolio" });
                throw error;
            }
        },

        /**
         * Gets open orders
         * @param {string} symbol - Optional symbol filter
         * @returns {Promise} - Open orders promise
         */
        async getOpenOrders(symbol = null) {
            try {
                let url = `${this._config.apiUrl}/orders/open`;
                if (symbol) {
                    const sanitizedSymbol = SecurityUtils.sanitizeData(symbol).toUpperCase();
                    url += `?symbol=${sanitizedSymbol}`;
                }

                const response = await fetch(url, {
                    headers: SecurityUtils.createSecureHeaders({
                        correlationId: SecurityUtils.generateSecureRandom(16)
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const ordersData = await response.json();
                return ordersData.orders?.map(order => this._processOrderResponse(order)) || [];

            } catch (error) {
                ErrorHandler.handleTradingError(error, { context: "open_orders", symbol: symbol });
                throw error;
            }
        },

        /**
         * Gets trading history
         * @param {Object} options - Query options
         * @returns {Promise} - Trading history promise
         */
        async getTradingHistory(options = {}) {
            try {
                const params = new URLSearchParams();

                if (options.symbol) {
                    params.append("symbol", SecurityUtils.sanitizeData(options.symbol).toUpperCase());
                }
                if (options.startTime) {
                    params.append("startTime", parseInt(options.startTime, 10));
                }
                if (options.endTime) {
                    params.append("endTime", parseInt(options.endTime, 10));
                }
                if (options.limit) {
                    params.append("limit", Math.min(parseInt(options.limit, 10) || 100, 1000));
                }

                const response = await fetch(`${this._config.apiUrl}/history?${params.toString()}`, {
                    headers: SecurityUtils.createSecureHeaders({
                        correlationId: SecurityUtils.generateSecureRandom(16)
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const historyData = await response.json();
                return historyData.trades?.map(trade => this._processTrade(trade)) || [];

            } catch (error) {
                ErrorHandler.handleTradingError(error, { context: "trading_history" });
                throw error;
            }
        },

        /**
         * Calculates position size based on risk parameters
         * @param {Object} params - Calculation parameters
         * @returns {Object} - Position size calculation
         */
        calculatePositionSize(params) {
            const {
                accountBalance,
                riskPercentage = 2, // 2% default risk
                entryPrice,
                stopLoss,
                symbol: _symbol
            } = params;

            if (!accountBalance || !entryPrice || !stopLoss) {
                throw new Error("Missing required parameters for position size calculation");
            }

            const riskAmount = accountBalance * (riskPercentage / 100);
            const priceRisk = Math.abs(entryPrice - stopLoss);
            const positionSize = riskAmount / priceRisk;

            // Apply maximum position size limits
            const maxPositionValue = Math.min(
                this._config.maxPositionSize,
                accountBalance * 0.5 // Max 50% of accoun
            );
            const maxQuantity = maxPositionValue / entryPrice;

            return {
                recommendedSize: Math.min(positionSize, maxQuantity),
                maxSize: maxQuantity,
                riskAmount: riskAmount,
                priceRisk: priceRisk,
                riskPercentage: riskPercentage
            };
        },

        /**
         * Performs risk checks before placing order
         * @private
         */
        async _performRiskChecks(orderData) {
            // Check order size limits
            const orderValue = orderData.quantity * (orderData.price || 0);
            if (orderValue > this._config.maxOrderSize) {
                throw new Error(`Order size exceeds maximum limit of ${this._config.maxOrderSize}`);
            }

            // Check open orders limi
            const openOrders = await this.getOpenOrders();
            if (openOrders.length >= this._config.riskLimits.maxOpenOrders) {
                throw new Error(`Maximum open orders limit (${this._config.riskLimits.maxOpenOrders}) reached`);
            }

            // Additional risk checks would go here
            // - Daily loss limits
            // - Position concentration limits
            // - Leverage limits
        },

        /**
         * Processes order response from API
         * @private
         */
        _processOrderResponse(orderData) {
            return {
                id: orderData.orderId || orderData.id,
                symbol: orderData.symbol,
                side: orderData.side,
                type: orderData.type,
                quantity: parseFloat(orderData.origQty || orderData.quantity || 0),
                price: parseFloat(orderData.price || 0),
                executedQuantity: parseFloat(orderData.executedQty || orderData.executedQuantity || 0),
                status: orderData.status,
                timeInForce: orderData.timeInForce,
                createdAt: parseInt(orderData.time || orderData.createdAt || Date.now(), 10),
                updatedAt: parseInt(orderData.updateTime || orderData.updatedAt || Date.now(), 10)
            };
        },

        /**
         * Processes portfolio data from API
         * @private
         */
        _processPortfolioData(portfolioData) {
            const positions = portfolioData.balances?.map(balance => ({
                asset: balance.asset,
                free: parseFloat(balance.free || 0),
                locked: parseFloat(balance.locked || 0),
                total: parseFloat(balance.free || 0) + parseFloat(balance.locked || 0)
            })).filter(pos => pos.total > 0) || [];

            return {
                totalValue: parseFloat(portfolioData.totalWalletBalance || 0),
                availableBalance: parseFloat(portfolioData.availableBalance || 0),
                totalPnL: parseFloat(portfolioData.totalUnrealizedProfit || 0),
                positions: positions,
                updatedAt: Date.now()
            };
        },

        /**
         * Processes trade data from API
         * @private
         */
        _processTrade(tradeData) {
            return {
                id: tradeData.id,
                orderId: tradeData.orderId,
                symbol: tradeData.symbol,
                side: tradeData.isBuyer ? "BUY" : "SELL",
                quantity: parseFloat(tradeData.qty || tradeData.quantity || 0),
                price: parseFloat(tradeData.price || 0),
                commission: parseFloat(tradeData.commission || 0),
                commissionAsset: tradeData.commissionAsset,
                timestamp: parseInt(tradeData.time || tradeData.timestamp || 0, 10)
            };
        },

        /**
         * Gets CSRF token for secure requests
         * @private
         */
        _getCsrfToken() {
            // This would typically be retrieved from the BaseController or security service
            return window.csrfToken || null;
        },

        /**
         * Destroys the service and cleans up resources
         */
        destroy() {
            this._orders.clear();
            this._positions.clear();
        }
    });
});
