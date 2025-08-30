sap.ui.define([
    "com/rex/cryptotrading/model/DataManager",
    "com/rex/cryptotrading/model/EventBusManager"
], function(DataManager, EventBusManager) {
    "use strict";

    /**
     * ML Data Manager - Specialized data manager for machine learning operations
     * Follows SAP UI5 patterns with AI/ML integration
     */
    return DataManager.extend("com.rex.cryptotrading.model.MLDataManager", {

        varructor: function() {
            const oInitialData = this._getInitialMLData();
            const oConfig = {
                enableChangeDetection: true,
                enableValidation: true,
                enableHistory: true, // Track prediction history
                maxHistorySize: 100
            };

            DataManager.prototype.varructor.call(this, "ml", oInitialData, oConfig);

            this._oEventBusManager = new EventBusManager();
            this._initializeValidationSchema();
            this._setupEventHandlers();
        },

        /**
         * Update predictions for multiple symbols
         * @param {Object} oPredictions - Predictions data by symbol
         */
        updatePredictions: function(oPredictions) {
            const oCurrentPredictions = this.getProperty("/predictions") || {};
            const oUpdatedPredictions = Object.assign({}, oCurrentPredictions);

            // Process each symbol's predictions
            Object.keys(oPredictions).forEach(sSymbol => {
                const oPrediction = oPredictions[sSymbol];
                if (this._validatePrediction(sSymbol, oPrediction)) {
                    oUpdatedPredictions[sSymbol] = this._processPrediction(oPrediction);
                }
            });

            this.setProperty("/predictions", oUpdatedPredictions);
            this.setProperty("/lastPredictionUpdate", new Date().toISOString());

            // Publish predictions updated even
            this._oEventBusManager.publishPredictionsUpdated(oUpdatedPredictions);
        },

        /**
         * Update prediction for single symbol
         * @param {string} sSymbol - Symbol to update
         * @param {Object} oPrediction - Prediction data
         */
        updateSymbolPrediction: function(sSymbol, oPrediction) {
            if (this._validatePrediction(sSymbol, oPrediction)) {
                const oProcessedPrediction = this._processPrediction(oPrediction);
                this.setProperty(`/predictions/${sSymbol}`, oProcessedPrediction);

                // Add to prediction history
                this._addToPredictionHistory(sSymbol, oProcessedPrediction);
            }
        },

        /**
         * Update model performance metrics
         * @param {string} sModelName - Model name
         * @param {Object} oMetrics - Performance metrics
         */
        updateModelPerformance: function(sModelName, oMetrics) {
            if (this._validateMetrics(oMetrics)) {
                const oProcessedMetrics = this._processMetrics(oMetrics);
                this.setProperty(`/models/${sModelName}/performance`, oProcessedMetrics);
                this.setProperty(`/models/${sModelName}/lastEvaluation`, new Date().toISOString());
            }
        },

        /**
         * Start model training
         * @param {string} sModelName - Model name
         * @param {Object} oTrainingConfig - Training configuration
         */
        startModelTraining: function(sModelName, oTrainingConfig) {
            const oTrainingData = {
                status: "training",
                progress: 0,
                startedAt: new Date().toISOString(),
                config: oTrainingConfig || {},
                logs: []
            };

            this.setProperty(`/models/${sModelName}/training`, oTrainingData);
            this.setProperty("/globalTrainingStatus", "active");
        },

        /**
         * Update training progress
         * @param {string} sModelName - Model name
         * @param {number} nProgress - Progress percentage (0-100)
         * @param {string} sMessage - Progress message
         */
        updateTrainingProgress: function(sModelName, nProgress, sMessage) {
            this.setProperty(`/models/${sModelName}/training/progress`, nProgress);
            this.setProperty(`/models/${sModelName}/training/lastUpdate`, new Date().toISOString());

            if (sMessage) {
                this.addToArray(`/models/${sModelName}/training/logs`, {
                    message: sMessage,
                    progress: nProgress,
                    timestamp: new Date().toISOString()
                });
            }

            // Publish training progress even
            this._oEventBusManager.publish(
                this._oEventBusManager.CHANNELS.ML,
                this._oEventBusManager.EVENTS.ML.TRAINING_PROGRESS,
                { modelName: sModelName, progress: nProgress, message: sMessage }
            );
        },

        /**
         * Compvare model training
         * @param {string} sModelName - Model name
         * @param {Object} oResults - Training results
         */
        compvareModelTraining: function(sModelName, oResults) {
            this.updateObject(`/models/${sModelName}/training`, {
                status: "compvared",
                progress: 100,
                compvaredAt: new Date().toISOString(),
                results: oResults || {}
            });

            // Check if all models are done training
            const oModels = this.getProperty("/models") || {};
            const bAllCompvare = Object.values(oModels).every(model =>
                !model.training || model.training.status !== "training"
            );

            if (bAllCompvare) {
                this.setProperty("/globalTrainingStatus", "idle");
            }

            // Publish model trained even
            this._oEventBusManager.publishModelTrained(sModelName, oResults);
        },

        /**
         * Set training error
         * @param {string} sModelName - Model name
         * @param {Object|string} vError - Error data
         */
        setTrainingError: function(sModelName, vError) {
            const oError = typeof vError === "string" ? { message: vError } : vError;

            this.updateObject(`/models/${sModelName}/training`, {
                status: "error",
                error: oError,
                failedAt: new Date().toISOString()
            });

            this._oEventBusManager.publish(
                this._oEventBusManager.CHANNELS.ML,
                this._oEventBusManager.EVENTS.ML.ERROR_OCCURRED,
                { modelName: sModelName, error: oError }
            );
        },

        /**
         * Get prediction for symbol
         * @param {string} sSymbol - Symbol
         * @returns {Object} Prediction data
         */
        getPrediction: function(sSymbol) {
            return this.getProperty(`/predictions/${sSymbol}`) || {};
        },

        /**
         * Get all predictions
         * @returns {Object} All predictions
         */
        getAllPredictions: function() {
            return this.getProperty("/predictions") || {};
        },

        /**
         * Get model information
         * @param {string} sModelName - Model name
         * @returns {Object} Model data
         */
        getModel: function(sModelName) {
            return this.getProperty(`/models/${sModelName}`) || {};
        },

        /**
         * Get all models
         * @returns {Object} All models
         */
        getAllModels: function() {
            return this.getProperty("/models") || {};
        },

        /**
         * Get prediction history for symbol
         * @param {string} sSymbol - Symbol
         * @param {number} iLimit - Maximum number of predictions
         * @returns {Array} Prediction history
         */
        getPredictionHistory: function(sSymbol, iLimit) {
            const aHistory = this.getProperty(`/predictionHistory/${sSymbol}`) || [];
            return iLimit ? aHistory.slice(0, iLimit) : aHistory;
        },

        /**
         * Get global ML statistics
         * @returns {Object} Statistics
         */
        getStatistics: function() {
            return this.getProperty("/statistics") || {};
        },

        /**
         * Set loading state
         * @param {boolean} bLoading - Loading state
         * @param {string} sOperation - Operation being performed
         */
        setLoading: function(bLoading, sOperation) {
            this.setProperty("/loading", bLoading);
            if (sOperation) {
                this.setProperty("/currentOperation", sOperation);
            }
        },

        /**
         * Set error state
         * @param {Object|string} vError - Error data
         */
        setError: function(vError) {
            const oError = typeof vError === "string" ? { message: vError } : vError;
            this.setProperty("/error", oError);

            if (oError) {
                this._oEventBusManager.publish(
                    this._oEventBusManager.CHANNELS.ML,
                    this._oEventBusManager.EVENTS.ML.ERROR_OCCURRED,
                    { error: oError }
                );
            }
        },

        /**
         * Clear error state
         */
        clearError: function() {
            this.setProperty("/error", null);
        },

        /**
         * Get confidence level description
         * @param {number} nConfidence - Confidence value (0-1)
         * @returns {string} Description
         */
        getConfidenceDescription: function(nConfidence) {
            if (nConfidence >= 0.8) return "High";
            if (nConfidence >= 0.6) return "Medium";
            if (nConfidence >= 0.4) return "Low";
            return "Very Low";
        },

        /**
         * Format prediction direction
         * @param {string} sSymbol - Symbol
         * @returns {string} Direction (Up, Down, Neutral)
         */
        getPredictionDirection: function(sSymbol) {
            const oPrediction = this.getPrediction(sSymbol);
            if (!oPrediction.predictedChange) return "Neutral";

            const nChange = parseFloat(oPrediction.predictedChange);
            if (nChange > 1) return "Up";
            if (nChange < -1) return "Down";
            return "Neutral";
        },

        /**
         * Initialize ML data structure
         * @private
         */
        _getInitialMLData: function() {
            return {
                predictions: {},
                models: {
                    "LSTM": {
                        name: "LSTM",
                        type: "neural_network",
                        status: "ready",
                        performance: {},
                        training: null,
                        lastEvaluation: null
                    },
                    "RandomForest": {
                        name: "RandomForest",
                        type: "ensemble",
                        status: "ready",
                        performance: {},
                        training: null,
                        lastEvaluation: null
                    }
                },
                predictionHistory: {},
                statistics: {
                    totalPredictions: 0,
                    successfulPredictions: 0,
                    accuracy: 0,
                    modelsActive: 2
                },
                globalTrainingStatus: "idle",
                lastPredictionUpdate: null,
                loading: false,
                currentOperation: null,
                error: null
            };
        },

        /**
         * Process prediction data
         * @private
         */
        _processPrediction: function(oPrediction) {
            return {
                currentPrice: oPrediction.currentPrice || 0,
                predictedPrice: oPrediction.predictedPrice || 0,
                predictedChange: oPrediction.predictedChange || 0,
                confidence: oPrediction.confidence || 0,
                horizon: oPrediction.horizon || "24h",
                model: oPrediction.model || "unknown",
                features: oPrediction.features || {},
                timestamp: new Date().toISOString(),
                expiresAt: this._calculateExpiration(oPrediction.horizon)
            };
        },

        /**
         * Process performance metrics
         * @private
         */
        _processMetrics: function(oMetrics) {
            return {
                accuracy: oMetrics.accuracy || 0,
                precision: oMetrics.precision || 0,
                recall: oMetrics.recall || 0,
                f1Score: oMetrics.f1Score || 0,
                mape: oMetrics.mape || 0, // Mean Absolute Percentage Error
                rmse: oMetrics.rmse || 0, // Root Mean Square Error
                sharpeRatio: oMetrics.sharpeRatio || 0,
                evaluatedAt: new Date().toISOString()
            };
        },

        /**
         * Add prediction to history
         * @private
         */
        _addToPredictionHistory: function(sSymbol, oPrediction) {
            const sPath = `/predictionHistory/${sSymbol}`;
            this.addToArray(sPath, oPrediction, 0); // Add to beginning

            // Limit history size
            const aHistory = this.getProperty(sPath) || [];
            if (aHistory.length > 50) {
                this.setProperty(sPath, aHistory.slice(0, 50));
            }
        },

        /**
         * Calculate prediction expiration
         * @private
         */
        _calculateExpiration: function(sHorizon) {
            const oNow = new Date();
            const iHours = this._parseHorizon(sHorizon);
            return new Date(oNow.getTime() + (iHours * 60 * 60 * 1000)).toISOString();
        },

        /**
         * Parse horizon string to hours
         * @private
         */
        _parseHorizon: function(sHorizon) {
            if (sHorizon.includes("h")) {
                return parseInt(sHorizon.replace("h", ""), 10);
            }
            if (sHorizon.includes("d")) {
                return parseInt(sHorizon.replace("d", ""), 10) * 24;
            }
            return 24; // Default to 24 hours
        },

        /**
         * Validate prediction data
         * @private
         */
        _validatePrediction: function(sSymbol, oPrediction) {
            if (!sSymbol || typeof sSymbol !== "string") {
                console.error("Invalid symbol for prediction:", sSymbol);
                return false;
            }

            if (!oPrediction || typeof oPrediction !== "object") {
                console.error("Invalid prediction data for", sSymbol);
                return false;
            }

            if (typeof oPrediction.predictedPrice !== "number" || oPrediction.predictedPrice <= 0) {
                console.error("Invalid predicted price for", sSymbol);
                return false;
            }

            return true;
        },

        /**
         * Validate metrics data
         * @private
         */
        _validateMetrics: function(oMetrics) {
            return oMetrics && typeof oMetrics === "object";
        },

        /**
         * Initialize validation schema
         * @private
         */
        _initializeValidationSchema: function() {
            this._oValidationSchema = {
                predictions: { type: "object", required: false },
                models: { type: "object", required: true },
                statistics: { type: "object", required: false }
            };
        },

        /**
         * Setup event handlers
         * @private
         */
        _setupEventHandlers: function() {
            // Listen for market data changes to trigger predictions
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.DATA_UPDATED,
                this._onMarketDataUpdated.bind(this),
                this
            );
        },

        /**
         * Handle market data updates
         * @private
         */
        _onMarketDataUpdated: function(sChannel, sEvent, oData) {
            // Market data updated - could trigger new predictions
            // This would be handled by the component/service layer
        },

        /**
         * Cleanup
         */
        destroy: function() {
            if (this._oEventBusManager) {
                this._oEventBusManager.destroy();
            }

            DataManager.prototype.destroy.apply(this, arguments);
        }
    });
});
