sap.ui.define([
    "sap/base/Log",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], (Log, MessageBox, MessageToast) => {
    "use strict";

    /**
     * Shared Error Handling Utilities for Crypto Trading Platform
     * Provides comprehensive error handling with trading-specific features including:
     * - Trading error categorization and handling
     * - Market data error recovery
     * - Connection error management
     * - User-friendly error presentation
     * - Retry mechanisms with exponential backoff
     */
    return {

        /**
         * Error severity levels
         */
        SEVERITY: {
            CRITICAL: "critical",
            HIGH: "high", 
            MEDIUM: "medium",
            LOW: "low",
            INFO: "info"
        },

        /**
         * Trading-specific error types
         */
        ERROR_TYPES: {
            TRADING: "trading",
            MARKET_DATA: "market_data",
            CONNECTION: "connection",
            VALIDATION: "validation",
            AUTHENTICATION: "authentication",
            RATE_LIMIT: "rate_limit",
            INSUFFICIENT_FUNDS: "insufficient_funds",
            NETWORK: "network"
        },

        /**
         * Handles errors with trading-specific logic and user-friendly presentation
         * @param {Error|Object} error - Error object or error information
         * @param {Object} options - Error handling options
         * @param {string} options.context - Context where error occurred (trading, market, etc.)
         * @param {string} options.severity - Error severity level
         * @param {boolean} options.showToUser - Whether to show error to user
         * @param {boolean} options.retry - Whether to enable retry functionality
         * @param {Function} options.onRetry - Retry callback function
         */
        handleError(error, options = {}) {
            const errorInfo = this._parseError(error);
            
            // Enhance error info with trading context
            errorInfo.context = options.context || "general";
            errorInfo.timestamp = new Date().toISOString();
            errorInfo.correlationId = options.correlationId || this._generateCorrelationId();

            // Log error securely
            this._logError(errorInfo, options);

            // Handle based on error type and severity
            if (options.showToUser !== false) {
                this._displayErrorToUser(errorInfo, options);
            }

            // Track error for analytics
            if (options.trackError !== false) {
                this._trackError(errorInfo, options);
            }

            return errorInfo;
        },

        /**
         * Handles trading-specific errors
         * @param {Error} error - Trading error
         * @param {Object} tradeContext - Trading context (symbol, amount, etc.)
         */
        handleTradingError(error, tradeContext = {}) {
            const options = {
                context: "trading",
                severity: this._determineTradingSeverity(error),
                showToUser: true,
                retry: this._canRetryTradingError(error),
                tradeContext: tradeContext
            };

            return this.handleError(error, options);
        },

        /**
         * Handles market data errors with automatic retry
         * @param {Error} error - Market data error
         * @param {Object} marketContext - Market context (symbol, timeframe, etc.)
         */
        handleMarketDataError(error, marketContext = {}) {
            const options = {
                context: "market_data",
                severity: this.SEVERITY.MEDIUM,
                showToUser: false, // Usually silent for market data
                retry: true,
                retryDelay: 1000,
                maxRetries: 3,
                marketContext: marketContext
            };

            return this.handleError(error, options);
        },

        /**
         * Handles connection errors with reconnection logic
         * @param {Error} error - Connection error
         * @param {Function} reconnectCallback - Reconnection callback
         */
        handleConnectionError(error, reconnectCallback) {
            const options = {
                context: "connection",
                severity: this.SEVERITY.HIGH,
                showToUser: true,
                retry: true,
                retryDelay: 2000,
                maxRetries: 5,
                onRetry: reconnectCallback
            };

            return this.handleError(error, options);
        },

        /**
         * Shows error dialog with retry option
         * @param {Object} errorInfo - Parsed error information
         * @param {Object} options - Display options
         */
        showErrorDialog(errorInfo, options = {}) {
            const actions = [MessageBox.Action.OK];
            
            if (options.retry && options.onRetry) {
                actions.unshift("Retry");
            }

            MessageBox.show(
                this._formatErrorMessage(errorInfo),
                {
                    icon: this._getErrorIcon(errorInfo.severity),
                    title: this._getErrorTitle(errorInfo),
                    actions: actions,
                    onClose: (sAction) => {
                        if (sAction === "Retry" && options.onRetry) {
                            options.onRetry();
                        }
                    }
                }
            );
        },

        /**
         * Shows error toast for non-critical errors
         * @param {Object} errorInfo - Parsed error information
         */
        showErrorToast(errorInfo) {
            MessageToast.show(
                this._formatErrorMessage(errorInfo, true),
                {
                    duration: 4000,
                    at: MessageToast.BOTTOM_CENTER
                }
            );
        },

        /**
         * Parses error object into standardized format
         * @private
         */
        _parseError(error) {
            if (!error) {
                return {
                    message: "Unknown error occurred",
                    type: this.ERROR_TYPES.NETWORK,
                    severity: this.SEVERITY.MEDIUM,
                    code: "UNKNOWN_ERROR"
                };
            }

            if (typeof error === "string") {
                return {
                    message: error,
                    type: this.ERROR_TYPES.NETWORK,
                    severity: this.SEVERITY.MEDIUM,
                    code: "STRING_ERROR"
                };
            }

            // Parse different error formats
            const errorInfo = {
                message: error.message || error.error || "An error occurred",
                type: error.type || this._inferErrorType(error),
                severity: error.severity || this._inferSeverity(error),
                code: error.code || error.status || "GENERIC_ERROR",
                details: error.details || error.stack,
                originalError: error
            };

            return errorInfo;
        },

        /**
         * Logs error securely without exposing sensitive data
         * @private
         */
        _logError(errorInfo, options) {
            const logData = {
                message: errorInfo.message,
                type: errorInfo.type,
                severity: errorInfo.severity,
                code: errorInfo.code,
                context: errorInfo.context,
                timestamp: errorInfo.timestamp,
                correlationId: errorInfo.correlationId
            };

            // Don't log sensitive trading data
            if (options.tradeContext) {
                logData.tradeSymbol = options.tradeContext.symbol;
                // Omit amounts and other sensitive data
            }

            switch (errorInfo.severity) {
                case this.SEVERITY.CRITICAL:
                    Log.error("CRITICAL ERROR", logData);
                    break;
                case this.SEVERITY.HIGH:
                    Log.error("HIGH SEVERITY ERROR", logData);
                    break;
                case this.SEVERITY.MEDIUM:
                    Log.warning("MEDIUM SEVERITY ERROR", logData);
                    break;
                default:
                    Log.info("ERROR", logData);
            }
        },

        /**
         * Displays error to user based on severity
         * @private
         */
        _displayErrorToUser(errorInfo, options) {
            switch (errorInfo.severity) {
                case this.SEVERITY.CRITICAL:
                case this.SEVERITY.HIGH:
                    this.showErrorDialog(errorInfo, options);
                    break;
                case this.SEVERITY.MEDIUM:
                    if (errorInfo.type === this.ERROR_TYPES.TRADING) {
                        this.showErrorDialog(errorInfo, options);
                    } else {
                        this.showErrorToast(errorInfo);
                    }
                    break;
                default:
                    this.showErrorToast(errorInfo);
            }
        },

        /**
         * Determines trading error severity
         * @private
         */
        _determineTradingSeverity(error) {
            const message = error.message?.toLowerCase() || "";
            
            if (message.includes("insufficient funds") || 
                message.includes("balance") ||
                message.includes("limit exceeded")) {
                return this.SEVERITY.HIGH;
            }
            
            if (message.includes("invalid") || 
                message.includes("validation")) {
                return this.SEVERITY.MEDIUM;
            }
            
            return this.SEVERITY.HIGH; // Default for trading errors
        },

        /**
         * Determines if trading error can be retried
         * @private
         */
        _canRetryTradingError(error) {
            const message = error.message?.toLowerCase() || "";
            
            // Don't retry validation or insufficient funds errors
            if (message.includes("insufficient funds") ||
                message.includes("invalid") ||
                message.includes("validation")) {
                return false;
            }
            
            // Retry network and temporary errors
            return message.includes("network") ||
                   message.includes("timeout") ||
                   message.includes("temporary");
        },

        /**
         * Infers error type from error object
         * @private
         */
        _inferErrorType(error) {
            const message = error.message?.toLowerCase() || "";
            
            if (message.includes("trading") || message.includes("order")) {
                return this.ERROR_TYPES.TRADING;
            }
            if (message.includes("market") || message.includes("price")) {
                return this.ERROR_TYPES.MARKET_DATA;
            }
            if (message.includes("connection") || message.includes("websocket")) {
                return this.ERROR_TYPES.CONNECTION;
            }
            if (message.includes("validation") || message.includes("invalid")) {
                return this.ERROR_TYPES.VALIDATION;
            }
            if (message.includes("auth") || message.includes("token")) {
                return this.ERROR_TYPES.AUTHENTICATION;
            }
            if (message.includes("rate limit") || message.includes("too many")) {
                return this.ERROR_TYPES.RATE_LIMIT;
            }
            
            return this.ERROR_TYPES.NETWORK;
        },

        /**
         * Infers error severity from error object
         * @private
         */
        _inferSeverity(error) {
            if (error.status >= 500) {
                return this.SEVERITY.HIGH;
            }
            if (error.status >= 400) {
                return this.SEVERITY.MEDIUM;
            }
            return this.SEVERITY.LOW;
        },

        /**
         * Formats error message for display
         * @private
         */
        _formatErrorMessage(errorInfo, isToast = false) {
            let message = errorInfo.message;
            
            // Add context for trading errors
            if (errorInfo.type === this.ERROR_TYPES.TRADING) {
                message = `Trading Error: ${message}`;
            } else if (errorInfo.type === this.ERROR_TYPES.MARKET_DATA) {
                message = `Market Data Error: ${message}`;
            }
            
            // Truncate for toasts
            if (isToast && message.length > 100) {
                message = message.substring(0, 97) + "...";
            }
            
            return message;
        },

        /**
         * Gets appropriate error icon
         * @private
         */
        _getErrorIcon(severity) {
            switch (severity) {
                case this.SEVERITY.CRITICAL:
                case this.SEVERITY.HIGH:
                    return MessageBox.Icon.ERROR;
                case this.SEVERITY.MEDIUM:
                    return MessageBox.Icon.WARNING;
                default:
                    return MessageBox.Icon.INFORMATION;
            }
        },

        /**
         * Gets error dialog title
         * @private
         */
        _getErrorTitle(errorInfo) {
            switch (errorInfo.type) {
                case this.ERROR_TYPES.TRADING:
                    return "Trading Error";
                case this.ERROR_TYPES.MARKET_DATA:
                    return "Market Data Error";
                case this.ERROR_TYPES.CONNECTION:
                    return "Connection Error";
                case this.ERROR_TYPES.AUTHENTICATION:
                    return "Authentication Error";
                default:
                    return "Error";
            }
        },

        /**
         * Tracks error for analytics
         * @private
         */
        _trackError(errorInfo, options) {
            // Implementation would send to analytics service
            // For now, just log to console in development
            if (window.location.hostname === "localhost") {
                console.group(`Error Tracking: ${errorInfo.type}`);
                console.log("Error Info:", errorInfo);
                console.log("Options:", options);
                console.groupEnd();
            }
        },

        /**
         * Generates correlation ID for error tracking
         * @private
         */
        _generateCorrelationId() {
            return "err-" + Date.now() + "-" + Math.random().toString(36).substr(2, 9);
        }
    };
});
