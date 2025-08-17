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
        
        varructor: function() {
            BaseObject.prototype.varructor.apply(this, arguments);
            this._oEventBus = EventBus.getInstance();
            this._mEventChannels = {};
            this._aSubscriptions = [];
        },
        
        /**
         * Event channel definitions following SAP naming conventions
         */
        CHANNELS: {
            MARKET: "market",
            WALLET: "walvar", 
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
            }\n        },\n        \n        /**\n         * Publish event to specific channel\n         * @param {string} sChannel - Event channel\n         * @param {string} sEvent - Event name\n         * @param {Object} oData - Event data\n         */\n        publish: function(sChannel, sEvent, oData) {\n            var oEventData = this._enhanceEventData(oData, sChannel, sEvent);\n            \n            this._oEventBus.publish(sChannel, sEvent, oEventData);\n            \n            // Log event for debugging (in development)\n            if (window.location.hostname === 'localhost') {\n                console.log(`[EventBus] ${sChannel}.${sEvent}:`, oEventData);\n            }\n        },\n        \n        /**\n         * Subscribe to events on a channel\n         * @param {string} sChannel - Event channel\n         * @param {string} sEvent - Event name\n         * @param {function} fnCallback - Callback function\n         * @param {Object} oContext - Context for callback\n         * @returns {Object} Subscription object with unsubscribe method\n         */\n        subscribe: function(sChannel, sEvent, fnCallback, oContext) {\n            this._oEventBus.subscribe(sChannel, sEvent, fnCallback, oContext);\n            \n            var oSubscription = {\n                channel: sChannel,\n                event: sEvent,\n                callback: fnCallback,\n                context: oContext,\n                timestamp: new Date(),\n                unsubscribe: () => {\n                    this._oEventBus.unsubscribe(sChannel, sEvent, fnCallback, oContext);\n                    this._removeSubscription(oSubscription);\n                }\n            };\n            \n            this._aSubscriptions.push(oSubscription);\n            return oSubscription;\n        },\n        \n        /**\n         * Unsubscribe from all events for a context\n         * @param {Object} oContext - Context to unsubscribe\n         */\n        unsubscribeAll: function(oContext) {\n            var aToRemove = this._aSubscriptions.filter(sub => sub.context === oContext);\n            \n            aToRemove.forEach(sub => {\n                this._oEventBus.unsubscribe(sub.channel, sub.event, sub.callback, sub.context);\n            });\n            \n            this._aSubscriptions = this._aSubscriptions.filter(sub => sub.context !== oContext);\n        },\n        \n        /**\n         * Market-specific event publishers\n         */\n        publishMarketDataUpdated: function(oMarketData) {\n            this.publish(this.CHANNELS.MARKET, this.EVENTS.MARKET.DATA_UPDATED, {\n                marketData: oMarketData,\n                symbols: Object.keys(oMarketData),\n                updateTime: new Date().toISOString()\n            });\n        },\n        \n        publishPriceChanged: function(sSymbol, nOldPrice, nNewPrice) {\n            this.publish(this.CHANNELS.MARKET, this.EVENTS.MARKET.PRICE_CHANGED, {\n                symbol: sSymbol,\n                oldPrice: nOldPrice,\n                newPrice: nNewPrice,\n                change: nNewPrice - nOldPrice,\n                changePercent: ((nNewPrice - nOldPrice) / nOldPrice * 100).toFixed(2)\n            });\n        },\n        \n        /**\n         * Walvar-specific event publishers\n         */\n        publishWalvarConnectionChanged: function(bConnected, sAddress, sProvider) {\n            this.publish(this.CHANNELS.WALLET, this.EVENTS.WALLET.CONNECTION_CHANGED, {\n                connected: bConnected,\n                address: sAddress,\n                provider: sProvider,\n                timestamp: new Date().toISOString()\n            });\n        },\n        \n        publishBalanceUpdated: function(oBalances) {\n            this.publish(this.CHANNELS.WALLET, this.EVENTS.WALLET.BALANCE_UPDATED, {\n                balances: oBalances,\n                totalValue: this._calculateTotalValue(oBalances),\n                timestamp: new Date().toISOString()\n            });\n        },\n        \n        /**\n         * ML-specific event publishers\n         */\n        publishPredictionsUpdated: function(oPredictions) {\n            this.publish(this.CHANNELS.ML, this.EVENTS.ML.PREDICTIONS_UPDATED, {\n                predictions: oPredictions,\n                symbols: Object.keys(oPredictions),\n                timestamp: new Date().toISOString()\n            });\n        },\n        \n        publishModelTrained: function(sModel, oMetrics) {\n            this.publish(this.CHANNELS.ML, this.EVENTS.ML.MODEL_TRAINED, {\n                modelName: sModel,\n                metrics: oMetrics,\n                timestamp: new Date().toISOString()\n            });\n        },\n        \n        /**\n         * UI-specific event publishers\n         */\n        publishNotification: function(sType, sMessage, oOptions) {\n            this.publish(this.CHANNELS.UI, this.EVENTS.UI.NOTIFICATION, {\n                type: sType, // success, error, warning, info\n                message: sMessage,\n                options: oOptions || {},\n                timestamp: new Date().toISOString()\n            });\n        },\n        \n        publishLoadingState: function(bLoading, sComponent, sOperation) {\n            this.publish(this.CHANNELS.UI, this.EVENTS.UI.LOADING_STATE, {\n                loading: bLoading,\n                component: sComponent,\n                operation: sOperation,\n                timestamp: new Date().toISOString()\n            });\n        },\n        \n        /**\n         * System-specific event publishers\n         */\n        publishComponentReady: function(sComponentName) {\n            this.publish(this.CHANNELS.SYSTEM, this.EVENTS.SYSTEM.COMPONENT_READY, {\n                componentName: sComponentName,\n                timestamp: new Date().toISOString()\n            });\n        },\n        \n        publishError: function(oError, sComponent, sOperation) {\n            this.publish(this.CHANNELS.SYSTEM, this.EVENTS.SYSTEM.ERROR_GLOBAL, {\n                error: oError,\n                component: sComponent,\n                operation: sOperation,\n                timestamp: new Date().toISOString(),\n                severity: this._getErrorSeverity(oError)\n            });\n        },\n        \n        /**\n         * Convenience method for subscribing to multiple events\n         * @param {Object} mEventHandlers - Map of channel.event to handler\n         * @param {Object} oContext - Context for handlers\n         * @returns {Array} Array of subscription objects\n         */\n        subscribeToEvents: function(mEventHandlers, oContext) {\n            var aSubscriptions = [];\n            \n            Object.keys(mEventHandlers).forEach(sEventKey => {\n                var [sChannel, sEvent] = sEventKey.split('.');\n                var fnHandler = mEventHandlers[sEventKey];\n                \n                if (sChannel && sEvent && fnHandler) {\n                    var oSub = this.subscribe(sChannel, sEvent, fnHandler, oContext);\n                    aSubscriptions.push(oSub);\n                }\n            });\n            \n            return aSubscriptions;\n        },\n        \n        /**\n         * Get subscription statistics\n         */\n        getSubscriptionStats: function() {\n            var mStats = {};\n            \n            this._aSubscriptions.forEach(sub => {\n                var sKey = `${sub.channel}.${sub.event}`;\n                mStats[sKey] = (mStats[sKey] || 0) + 1;\n            });\n            \n            return {\n                totalSubscriptions: this._aSubscriptions.length,\n                byEvent: mStats,\n                channels: Object.keys(this.CHANNELS)\n            };\n        },\n        \n        /**\n         * Enhance event data with metadata\n         * @private\n         */\n        _enhanceEventData: function(oData, sChannel, sEvent) {\n            return Object.assign({}, oData, {\n                _metadata: {\n                    channel: sChannel,\n                    event: sEvent,\n                    timestamp: new Date().toISOString(),\n                    sequence: this._getNextSequence()\n                }\n            });\n        },\n        \n        /**\n         * Calculate total portfolio value\n         * @private\n         */\n        _calculateTotalValue: function(oBalances) {\n            // This would typically use current market prices\n            return Object.values(oBalances).reduce((total, balance) => {\n                return total + (parseFloat(balance) || 0);\n            }, 0);\n        },\n        \n        /**\n         * Determine error severity\n         * @private\n         */\n        _getErrorSeverity: function(oError) {\n            if (oError.name === 'NetworkError') return 'high';\n            if (oError.name === 'ValidationError') return 'medium';\n            return 'low';\n        },\n        \n        /**\n         * Get next sequence number\n         * @private\n         */\n        _getNextSequence: function() {\n            this._iSequence = (this._iSequence || 0) + 1;\n            return this._iSequence;\n        },\n        \n        /**\n         * Remove subscription from tracking\n         * @private\n         */\n        _removeSubscription: function(oSubscription) {\n            var iIndex = this._aSubscriptions.indexOf(oSubscription);\n            if (iIndex > -1) {\n                this._aSubscriptions.splice(iIndex, 1);\n            }\n        },\n        \n        /**\n         * Cleanup all subscriptions\n         */\n        destroy: function() {\n            this._aSubscriptions.forEach(sub => {\n                this._oEventBus.unsubscribe(sub.channel, sub.event, sub.callback, sub.context);\n            });\n            this._aSubscriptions = [];\n            \n            BaseObject.prototype.destroy.apply(this, arguments);\n        }\n    });\n});