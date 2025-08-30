sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/core/EventBus"
], function(BaseObject, EventBus) {
    "use strict";

    /**
     * EventBus Manager - Centralized event handling following SAP UI5 patterns
     * Provides structured event communication between components
     */
    return BaseObject.extend("com.rex.cryptotrading.model.EventBusManager", {

        constructor: function() {
            BaseObject.prototype.constructor.apply(this, arguments);
            this._oEventBus = EventBus.getInstance();
            this._mEventChannels = {};
            this._aSubscriptions = [];
        },

        /**
         * Event channel definitions following SAP naming conventions
         */
        CHANNELS: {
            MARKET: "market",
            WALLET: "wallet",
            ML: "ml",
            UI: "ui",
            NAVIGATION: "navigation",
            SYSTEM: "system"
        },

        /**
         * Event types for each channel
         */
        EVENTS: {
            MARKET: {
                DATA_UPDATED: "dataUpdated",
                PRICE_CHANGED: "priceChanged",
                CONNECTION_STATUS: "connectionStatus",
                REFRESH_REQUESTED: "refreshRequested",
                ERROR_OCCURRED: "errorOccurred"
            },
            WALLET: {
                CONNECTION_CHANGED: "connectionChanged",
                BALANCE_UPDATED: "balanceUpdated",
                TRANSACTION_SUBMITTED: "transactionSubmitted",
                TRANSACTION_CONFIRMED: "transactionConfirmed",
                ERROR_OCCURRED: "errorOccurred"
            },
            ML: {
                PREDICTIONS_UPDATED: "predictionsUpdated",
                MODEL_TRAINED: "modelTrained",
                ANALYSIS_COMPLETE: "analysisCompvare",
                TRAINING_PROGRESS: "trainingProgress",
                ERROR_OCCURRED: "errorOccurred"
            },
            UI: {
                THEME_CHANGED: "themeChanged",
                LANGUAGE_CHANGED: "languageChanged",
                LAYOUT_CHANGED: "layoutChanged",
                LOADING_STATE: "loadingState",
                NOTIFICATION: "notification"
            },
            NAVIGATION: {
                ROUTE_CHANGED: "routeChanged",
                TAB_SELECTED: "tabSelected",
                DIALOG_OPENED: "dialogOpened",
                DIALOG_CLOSED: "dialogClosed"
            },
            SYSTEM: {
                COMPONENT_READY: "componentReady",
                DATA_SYNC: "dataSync",
                ERROR_GLOBAL: "errorGlobal",
                PERFORMANCE_METRIC: "performanceMetric"
            }
        },

        /**
         * Publish event to specific channel
         * @param {string} sChannel - Event channel
         * @param {string} sEvent - Event name
         * @param {Object} oData - Event data
         */
        publish: function(sChannel, sEvent, oData) {
            const oEventData = this._enhanceEventData(oData, sChannel, sEvent);

            this._oEventBus.publish(sChannel, sEvent, oEventData);

            // Log event for debugging (in development)
            if (window.location.hostname === "localhost") {
                // [EventBus] ${sChannel}.${sEvent}: oEventData
            }
        },

        /**
         * Subscribe to events on a channel
         * @param {string} sChannel - Event channel
         * @param {string} sEvent - Event name
         * @param {function} fnCallback - Callback function
         * @param {Object} oContext - Context for callback
         * @returns {Object} Subscription object with unsubscribe method
         */
        subscribe: function(sChannel, sEvent, fnCallback, oContext) {
            this._oEventBus.subscribe(sChannel, sEvent, fnCallback, oContext);

            const oSubscription = {
                channel: sChannel,
                event: sEvent,
                callback: fnCallback,
                context: oContext,
                timestamp: new Date(),
                unsubscribe: () => {
                    this._oEventBus.unsubscribe(sChannel, sEvent, fnCallback, oContext);
                    this._removeSubscription(oSubscription);
                }
            };

            this._aSubscriptions.push(oSubscription);
            return oSubscription;
        },

        /**
         * Unsubscribe from all events for a contex
         * @param {Object} oContext - Context to unsubscribe
         */
        unsubscribeAll: function(oContext) {
            const aToRemove = this._aSubscriptions.filter(sub => sub.context === oContext);

            aToRemove.forEach(sub => {
                this._oEventBus.unsubscribe(sub.channel, sub.event, sub.callback, sub.context);
            });

            this._aSubscriptions = this._aSubscriptions.filter(sub => sub.context !== oContext);
        },

        /**
         * Market-specific event publishers
         */
        publishMarketDataUpdated: function(oMarketData) {
            this.publish(this.CHANNELS.MARKET, this.EVENTS.MARKET.DATA_UPDATED, {
                marketData: oMarketData,
                symbols: Object.keys(oMarketData),
                updateTime: new Date().toISOString()
            });
        },

        publishPriceChanged: function(sSymbol, nOldPrice, nNewPrice) {
            this.publish(this.CHANNELS.MARKET, this.EVENTS.MARKET.PRICE_CHANGED, {
                symbol: sSymbol,
                oldPrice: nOldPrice,
                newPrice: nNewPrice,
                change: nNewPrice - nOldPrice,
                changePercent: ((nNewPrice - nOldPrice) / nOldPrice * 100).toFixed(2)
            });
        },

        /**
         * Walvar-specific event publishers
         */
        publishWalvarConnectionChanged: function(bConnected, sAddress, sProvider) {
            this.publish(this.CHANNELS.WALLET, this.EVENTS.WALLET.CONNECTION_CHANGED, {
                connected: bConnected,
                address: sAddress,
                provider: sProvider,
                timestamp: new Date().toISOString()
            });
        },

        publishBalanceUpdated: function(oBalances) {
            this.publish(this.CHANNELS.WALLET, this.EVENTS.WALLET.BALANCE_UPDATED, {
                balances: oBalances,
                totalValue: this._calculateTotalValue(oBalances),
                timestamp: new Date().toISOString()
            });
        },

        /**
         * ML-specific event publishers
         */
        publishPredictionsUpdated: function(oPredictions) {
            this.publish(this.CHANNELS.ML, this.EVENTS.ML.PREDICTIONS_UPDATED, {
                predictions: oPredictions,
                symbols: Object.keys(oPredictions),
                timestamp: new Date().toISOString()
            });
        },

        publishModelTrained: function(sModel, oMetrics) {
            this.publish(this.CHANNELS.ML, this.EVENTS.ML.MODEL_TRAINED, {
                modelName: sModel,
                metrics: oMetrics,
                timestamp: new Date().toISOString()
            });
        },

        /**
         * UI-specific event publishers
         */
        publishNotification: function(sType, sMessage, oOptions) {
            this.publish(this.CHANNELS.UI, this.EVENTS.UI.NOTIFICATION, {
                type: sType, // success, error, warning, info
                message: sMessage,
                options: oOptions || {},
                timestamp: new Date().toISOString()
            });
        },

        publishLoadingState: function(bLoading, sComponent, sOperation) {
            this.publish(this.CHANNELS.UI, this.EVENTS.UI.LOADING_STATE, {
                loading: bLoading,
                component: sComponent,
                operation: sOperation,
                timestamp: new Date().toISOString()
            });
        },

        /**
         * System-specific event publishers
         */
        publishComponentReady: function(sComponentName) {
            this.publish(this.CHANNELS.SYSTEM, this.EVENTS.SYSTEM.COMPONENT_READY, {
                componentName: sComponentName,
                timestamp: new Date().toISOString()
            });
        },

        publishError: function(oError, sComponent, sOperation) {
            this.publish(this.CHANNELS.SYSTEM, this.EVENTS.SYSTEM.ERROR_GLOBAL, {
                error: oError,
                component: sComponent,
                operation: sOperation,
                timestamp: new Date().toISOString(),
                severity: this._getErrorSeverity(oError)
            });
        },

        /**
         * Convenience method for subscribing to multiple events
         * @param {Object} mEventHandlers - Map of channel.event to handler
         * @param {Object} oContext - Context for handlers
         * @returns {Array} Array of subscription objects
         */
        subscribeToEvents: function(mEventHandlers, oContext) {
            const aSubscriptions = [];

            Object.keys(mEventHandlers).forEach(sEventKey => {
                const [sChannel, sEvent] = sEventKey.split(".");
                const fnHandler = mEventHandlers[sEventKey];

                if (sChannel && sEvent && fnHandler) {
                    const oSub = this.subscribe(sChannel, sEvent, fnHandler, oContext);
                    aSubscriptions.push(oSub);
                }
            });

            return aSubscriptions;
        },

        /**
         * Get subscription statistics
         */
        getSubscriptionStats: function() {
            const mStats = {};

            this._aSubscriptions.forEach(sub => {
                const sKey = `${sub.channel}.${sub.event}`;
                mStats[sKey] = (mStats[sKey] || 0) + 1;
            });

            return {
                totalSubscriptions: this._aSubscriptions.length,
                byEvent: mStats,
                channels: Object.keys(this.CHANNELS)
            };
        },

        /**
         * Enhance event data with metadata
         * @private
         */
        _enhanceEventData: function(oData, sChannel, sEvent) {
            return Object.assign({}, oData, {
                _metadata: {
                    channel: sChannel,
                    event: sEvent,
                    timestamp: new Date().toISOString(),
                    sequence: this._getNextSequence()
                }
            });
        },

        /**
         * Calculate total portfolio value
         * @private
         */
        _calculateTotalValue: function(oBalances) {
            // This would typically use current market prices
            return Object.values(oBalances).reduce((total, balance) => {
                return total + (parseFloat(balance) || 0);
            }, 0);
        },

        /**
         * Determine error severity
         * @private
         */
        _getErrorSeverity: function(oError) {
            if (oError.name === "NetworkError") return "high";
            if (oError.name === "ValidationError") return "medium";
            return "low";
        },

        /**
         * Get next sequence number
         * @private
         */
        _getNextSequence: function() {
            this._iSequence = (this._iSequence || 0) + 1;
            return this._iSequence;
        },

        /**
         * Remove subscription from tracking
         * @private
         */
        _removeSubscription: function(oSubscription) {
            const iIndex = this._aSubscriptions.indexOf(oSubscription);
            if (iIndex > -1) {
                this._aSubscriptions.splice(iIndex, 1);
            }
        },

        /**
         * Cleanup all subscriptions
         */
        destroy: function() {
            this._aSubscriptions.forEach(sub => {
                this._oEventBus.unsubscribe(sub.channel, sub.event, sub.callback, sub.context);
            });
            this._aSubscriptions = [];

            BaseObject.prototype.destroy.apply(this, arguments);
        }
    });
});
