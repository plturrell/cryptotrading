/**
 * Crypto Trading Platform Base Controller
 * 
 * Enterprise-grade base controller providing standardized patterns for the crypto trading platform.
 * Adapted from FINSIGHT UI5 Template Framework standards.
 */

sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel"
], (Controller, MessageToast, MessageBox, JSONModel) => {
    "use strict";

    /**
     * Base Controller for Crypto Trading Platform
     *
     * Provides common functionality and enterprise patterns for all controllers in the crypto trading application.
     * This controller implements SAP standard patterns for loading states, error handling, navigation,
     * resource bundle management, and crypto-specific functionality.
     *
     * @namespace com.rex.cryptotrading.controller
     * @class
     * @extends sap.ui.core.mvc.Controller
     * @public
     * @author rex.com
     * @since 1.0.0
     * @version 1.0.0
     *
     * @example
     * // Extending BaseController in your controller
     * sap.ui.define([
     *     "./BaseController"
     * ], function(BaseController) {
     *     "use strict";
     *     return BaseController.extend("com.rex.cryptotrading.controller.MyController", {
     *         onInit: function() {
     *             BaseController.prototype.onInit.apply(this, arguments);
     *             // Your initialization code here
     *         }
     *     });
     * });
     */

    return Controller.extend("com.rex.cryptotrading.controller.BaseController", {

        /**
         * Called when a controller is instantiated and its View controls have been created.
         * Initializes the UI model for loading states and common controller functionality.
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @since 1.0.0
         */
        onInit() {
            // Initialize UI state model for loading states
            this.oUIModel = new JSONModel({
                // Loading states
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingMarketData: false,
                isLoadingTrade: false,
                loadingMessage: "",
                loadingSubMessage: "",
                progressTitle: "",
                progressValue: 0,
                progressText: "",
                progressState: "None",
                progressDescription: "",
                marketDataStep: "",
                tradeStep: "",

                // Error states
                hasError: false,
                errorMessage: "",
                errorTitle: "",
                errorType: "", // trading, market, connection, validation

                // Data states
                hasNoData: false,
                noDataMessage: "",
                noDataIcon: "sap-icon://product",

                // UI states
                busy: false,
                editable: false,
                hasChanges: false,
                showDetails: false,
                selectedItems: [],

                // Trading states
                tradingEnabled: true,
                connectionStatus: "connected", // connected, disconnected, reconnecting
                marketStatus: "open", // open, closed, pre-market, after-hours
                riskLevel: "medium", // low, medium, high, critical

                // Security states
                csrfToken: null,
                sessionId: null,
                correlationId: null,
                securityInitialized: false,
                apiKeyValid: false
            });

            this.getView().setModel(this.oUIModel, "ui");

            // Initialize app model for navigation state
            this.oAppModel = new JSONModel({
                selectedKey: "launchpad",
                connectionStatus: "connected",
                busy: false,
                currentSymbol: "BTC",
                currentTimeframe: "1h"
            });
            this.getView().setModel(this.oAppModel, "app");

            // Initialize trading model
            this.oTradingModel = new JSONModel({
                portfolio: {
                    totalValue: 0,
                    totalPnL: 0,
                    totalPnLPercent: 0,
                    positions: []
                },
                watchlist: [],
                alerts: [],
                orders: []
            });
            this.getView().setModel(this.oTradingModel, "trading");

            // Initialize security
            this._initializeSecurity();

            // Initialize market data connection
            this._initializeMarketData();
        },

        /**
         * Gets the resource bundle for internationalization
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @returns {sap.base.i18n.ResourceBundle} The resource bundle
         * @since 1.0.0
         */
        getResourceBundle() {
            const oComponent = this.getOwnerComponent();
            if (!oComponent) {
                console.warn("Owner component not available, using fallback resource bundle");
                return null;
            }
            const oI18nModel = oComponent.getModel("i18n");
            return oI18nModel ? oI18nModel.getResourceBundle() : null;
        },

        /**
         * Gets the router instance
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @returns {sap.ui.core.routing.Router} The router instance
         * @since 1.0.0
         */
        getRouter() {
            const oComponent = this.getOwnerComponent();
            if (!oComponent) {
                console.warn("Owner component not available, router functionality limited");
                return null;
            }
            return oComponent.getRouter();
        },

        /**
         * Gets the model by name, or the default model if no name is provided
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} [sModelName] The name of the model
         * @returns {sap.ui.model.Model} The model instance
         * @since 1.0.0
         */
        getModel(sModelName) {
            return this.getView().getModel(sModelName);
        },

        /**
         * Sets a model on the view
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {sap.ui.model.Model} oModel The model to set
         * @param {string} [sModelName] The name of the model
         * @since 1.0.0
         */
        setModel(oModel, sModelName) {
            this.getView().setModel(oModel, sModelName);
        },

        /* =========================================================== */
        /* Loading State Methods                                       */
        /* =========================================================== */

        /**
         * Shows skeleton loading state for lists and tables
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} [sMessage] Loading message to display
         * @since 1.0.0
         */
        showSkeletonLoading(sMessage) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: true,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingMarketData: false,
                isLoadingTrade: false,
                hasError: false,
                hasNoData: false,
                loadingMessage: sMessage || this._getText("common.loading")
            });
        },

        /**
         * Shows spinner loading state for actions
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} [sMessage] Loading message to display
         * @param {string} [sSubMessage] Additional loading context
         * @since 1.0.0
         */
        showSpinnerLoading(sMessage, sSubMessage) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: true,
                isLoadingProgress: false,
                isLoadingMarketData: false,
                isLoadingTrade: false,
                hasError: false,
                hasNoData: false,
                loadingMessage: sMessage || this._getText("common.processing"),
                loadingSubMessage: sSubMessage || ""
            });
        },

        /**
         * Shows market data loading state
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} [sStep] Current market data operation step
         * @since 1.0.0
         */
        showMarketDataLoading(sStep) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingMarketData: true,
                isLoadingTrade: false,
                hasError: false,
                hasNoData: false,
                marketDataStep: sStep || this._getText("trading.loadingMarketData")
            });
        },

        /**
         * Shows trading operation loading state
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} [sStep] Current trading operation step
         * @since 1.0.0
         */
        showTradeLoading(sStep) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingMarketData: false,
                isLoadingTrade: true,
                hasError: false,
                hasNoData: false,
                tradeStep: sStep || this._getText("trading.processingTrade")
            });
        },

        /**
         * Shows progress loading state for multi-step operations
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {object} oOptions Progress configuration options
         * @param {string} [oOptions.title] Progress title
         * @param {string} [oOptions.message] Progress message
         * @param {number} [oOptions.value] Progress percentage (0-100)
         * @param {string} [oOptions.state] Progress state (None|Success|Warning|Error)
         * @since 1.0.0
         */
        showProgressLoading(oOptions = {}) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: true,
                isLoadingMarketData: false,
                isLoadingTrade: false,
                hasError: false,
                hasNoData: false,
                progressTitle: oOptions.title || this._getText("common.processing"),
                loadingMessage: oOptions.message || "",
                progressValue: oOptions.value || 0,
                progressText: `${oOptions.value || 0}%`,
                progressState: oOptions.state || "None",
                progressDescription: oOptions.description || ""
            });
        },

        /**
         * Hides all loading states
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @since 1.0.0
         */
        hideLoading() {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingMarketData: false,
                isLoadingTrade: false
            });
        },

        /* =========================================================== */
        /* Error Handling Methods                                      */
        /* =========================================================== */

        /**
         * Shows error state with message
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} sMessage Error message
         * @param {string} [sTitle] Error title
         * @param {string} [sType] Error type (trading, market, connection, validation)
         * @since 1.0.0
         */
        showError(sMessage, sTitle, sType) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingMarketData: false,
                isLoadingTrade: false,
                hasError: true,
                hasNoData: false,
                errorMessage: sMessage,
                errorTitle: sTitle || this._getText("common.error"),
                errorType: sType || "general"
            });

            // Log error for debugging
            console.error(`[${sType || "GENERAL"}] ${sTitle || "Error"}: ${sMessage}`);
        },

        /**
         * Shows no data state
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} [sMessage] No data message
         * @param {string} [sIcon] No data icon
         * @since 1.0.0
         */
        showNoData(sMessage, sIcon) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingMarketData: false,
                isLoadingTrade: false,
                hasError: false,
                hasNoData: true,
                noDataMessage: sMessage || this._getText("common.noDataAvailable"),
                noDataIcon: sIcon || "sap-icon://product"
            });
        },

        /* =========================================================== */
        /* Navigation Methods                                          */
        /* =========================================================== */

        /**
         * Navigates back in browser history or to a specific route
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} [sDefaultRoute] Default route if no history
         * @since 1.0.0
         */
        onNavBack(sDefaultRoute) {
            const oHistory = sap.ui.core.routing.History.getInstance();
            const sPreviousHash = oHistory.getPreviousHash();
            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getRouter().navTo(sDefaultRoute || "launchpad", {}, true);
            }
        },

        /* =========================================================== */
        /* Message Methods                                             */
        /* =========================================================== */

        /**
         * Shows a message toast
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} sMessage Message to display
         * @param {object} [oOptions] Toast options
         * @since 1.0.0
         */
        showMessageToast(sMessage, oOptions = {}) {
            MessageToast.show(sMessage, {
                duration: oOptions.duration || 3000,
                at: oOptions.at || MessageToast.BOTTOM_CENTER,
                ...oOptions
            });
        },

        /**
         * Shows a message box
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} sMessage Message to display
         * @param {object} [oOptions] MessageBox options
         * @since 1.0.0
         */
        showMessageBox(sMessage, oOptions = {}) {
            MessageBox.show(sMessage, {
                icon: oOptions.icon || MessageBox.Icon.INFORMATION,
                title: oOptions.title || this._getText("common.information"),
                actions: oOptions.actions || [MessageBox.Action.OK],
                onClose: oOptions.onClose,
                ...oOptions
            });
        },

        /* =========================================================== */
        /* Trading-Specific Methods                                    */
        /* =========================================================== */

        /**
         * Updates connection status
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} sStatus Connection status (connected, disconnected, reconnecting)
         * @since 1.0.0
         */
        updateConnectionStatus(sStatus) {
            this.oUIModel.setProperty("/connectionStatus", sStatus);
            this.oAppModel.setProperty("/connectionStatus", sStatus);
        },

        /**
         * Updates market status
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} sStatus Market status (open, closed, pre-market, after-hours)
         * @since 1.0.0
         */
        updateMarketStatus(sStatus) {
            this.oUIModel.setProperty("/marketStatus", sStatus);
        },

        /**
         * Updates risk level
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {string} sLevel Risk level (low, medium, high, critical)
         * @since 1.0.0
         */
        updateRiskLevel(sLevel) {
            this.oUIModel.setProperty("/riskLevel", sLevel);
        },

        /**
         * Formats currency values for display
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {number} nValue Numeric value
         * @param {string} [sCurrency] Currency code (default: USD)
         * @param {number} [nDecimals] Number of decimal places
         * @returns {string} Formatted currency string
         * @since 1.0.0
         */
        formatCurrency(nValue, sCurrency = "USD", nDecimals = 2) {
            if (typeof nValue !== "number" || isNaN(nValue)) {
                return "0.00";
            }
            return new Intl.NumberFormat("en-US", {
                style: "currency",
                currency: sCurrency,
                minimumFractionDigits: nDecimals,
                maximumFractionDigits: nDecimals
            }).format(nValue);
        },

        /**
         * Formats percentage values for display
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @public
         * @param {number} nValue Numeric value (as decimal, e.g., 0.05 for 5%)
         * @param {number} [nDecimals] Number of decimal places
         * @returns {string} Formatted percentage string
         * @since 1.0.0
         */
        formatPercentage(nValue, nDecimals = 2) {
            if (typeof nValue !== "number" || isNaN(nValue)) {
                return "0.00%";
            }
            return new Intl.NumberFormat("en-US", {
                style: "percent",
                minimumFractionDigits: nDecimals,
                maximumFractionDigits: nDecimals
            }).format(nValue);
        },

        /* =========================================================== */
        /* Security Methods                                            */
        /* =========================================================== */

        /**
         * Initializes security features including CSRF token retrieval
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @since 1.0.0
         */
        _initializeSecurity() {
            // Generate correlation ID for request tracking
            this.oUIModel.setProperty("/correlationId", this._generateCorrelationId());

            // Fetch CSRF token for secure operations
            this._fetchCSRFToken();
        },

        /**
         * Initializes market data connection
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @since 1.0.0
         */
        _initializeMarketData() {
            // Initialize WebSocket connection for real-time data
            // This will be implemented by specific controllers as needed
            this.updateConnectionStatus("connected");
            this.updateMarketStatus("open");
        },

        /**
         * Generates a unique correlation ID for request tracking
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @returns {string} Unique correlation ID
         * @since 1.0.0
         */
        _generateCorrelationId() {
            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
                const r = Math.random() * 16 | 0, v = c === "x" ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        },

        /**
         * Fetches CSRF token from the server
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @since 1.0.0
         */
        async _fetchCSRFToken() {
            try {
                const response = await fetch("/api/csrf-token", {
                    method: "GET",
                    credentials: "same-origin",
                    headers: {
                        "X-Correlation-ID": this.oUIModel.getProperty("/correlationId")
                    }
                });

                if (response.ok) {
                    const data = await response.json();
                    this.oUIModel.setProperty("/csrfToken", data.csrfToken);
                    this.oUIModel.setProperty("/sessionId", data.sessionId);
                    this.oUIModel.setProperty("/securityInitialized", true);
                }
            } catch (error) {
                console.warn("Failed to fetch CSRF token:", error);
                // Continue without CSRF in development
                this.oUIModel.setProperty("/securityInitialized", true);
            }
        },

        /**
         * Makes a secure API request with proper error handling
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @protected
         * @param {string} sUrl Request URL
         * @param {object} [oOptions] Request options
         * @returns {Promise} Request promise
         * @since 1.0.0
         */
        async secureRequest(sUrl, oOptions = {}) {
            const {
                method = "GET",
                data = null,
                headers = {},
                sanitize = true
            } = oOptions;

            // Prepare headers
            const requestHeaders = {
                "Content-Type": "application/json",
                "X-Correlation-ID": this.oUIModel.getProperty("/correlationId"),
                ...headers
            };

            // Add CSRF token for write operations
            if (["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
                const csrfToken = this.oUIModel.getProperty("/csrfToken");
                if (csrfToken) {
                    requestHeaders["X-CSRF-Token"] = csrfToken;
                }
            }

            // Sanitize request data
            let sanitizedData = data;
            if (sanitize && data) {
                sanitizedData = this._sanitizeRequestData(data);
            }

            const requestOptions = {
                method: method.toUpperCase(),
                credentials: "same-origin",
                headers: requestHeaders
            };

            if (sanitizedData && method.toUpperCase() !== "GET") {
                requestOptions.body = JSON.stringify(sanitizedData);
            }

            try {
                const response = await fetch(sUrl, requestOptions);
                return await this._handleSecureResponse(response);
            } catch (error) {
                return this._handleRequestError(error, sUrl, method);
            }
        },

        /* =========================================================== */
        /* Private Helper Methods                                      */
        /* =========================================================== */

        /**
         * Gets localized text from resource bundle
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @param {string} sKey Text key
         * @param {array} [aArgs] Arguments for placeholders
         * @returns {string} Localized text
         * @since 1.0.0
         */
        _getText(sKey, aArgs) {
            const oResourceBundle = this.getResourceBundle();
            if (oResourceBundle) {
                return oResourceBundle.getText(sKey, aArgs);
            }
            return sKey; // Fallback to key if no resource bundle
        },

        /**
         * Sanitizes request data to prevent XSS and injection attacks
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @param {*} data Data to sanitize
         * @returns {*} Sanitized data
         * @since 1.0.0
         */
        _sanitizeRequestData(data) {
            if (typeof data === "string") {
                return this._sanitizeString(data);
            }

            if (Array.isArray(data)) {
                return data.map(item => this._sanitizeRequestData(item));
            }

            if (data && typeof data === "object") {
                const sanitized = {};
                Object.keys(data).forEach(key => {
                    const sanitizedKey = this._sanitizeString(key);
                    sanitized[sanitizedKey] = this._sanitizeRequestData(data[key]);
                });
                return sanitized;
            }

            return data;
        },

        /**
         * Sanitizes a string to prevent XSS attacks
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @param {string} sInput String to sanitize
         * @returns {string} Sanitized string
         * @since 1.0.0
         */
        _sanitizeString(sInput) {
            if (typeof sInput !== "string") {
                return sInput;
            }

            return sInput
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#x27;")
                .replace(/\//g, "&#x2F;")
                .replace(/\\/g, "&#x5C;")
                .replace(/&/g, "&amp;");
        },

        /**
         * Handles secure response processing
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @param {Response} response Fetch response
         * @returns {Promise} Response data or error
         * @since 1.0.0
         */
        async _handleSecureResponse(response) {
            if (!response.ok) {
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    errorData = { error: "Network error", status: response.status };
                }

                // Log security-related errors
                if (response.status === 403 || response.status === 401) {
                    console.warn(`[SECURITY] ${response.status} response`);
                }

                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            try {
                const data = await response.json();
                return data;
            } catch (error) {
                // Handle non-JSON responses
                return response.text();
            }
        },

        /**
         * Handles request errors with proper logging
         *
         * @function
         * @memberOf com.rex.cryptotrading.controller.BaseController
         * @private
         * @param {Error} error Request error
         * @param {string} sUrl Request URL
         * @param {string} sMethod HTTP method
         * @returns {Promise} Rejected promise with sanitized error
         * @since 1.0.0
         */
        _handleRequestError(error, sUrl, sMethod) {
            const correlationId = this.oUIModel.getProperty("/correlationId");

            // Log error with correlation ID
            console.error(`[REQUEST ERROR] ${sMethod} ${sUrl} - Correlation ID: ${correlationId}`, error);

            // Return sanitized error message
            const sanitizedMessage = error.message.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            return Promise.reject(new Error(sanitizedMessage));
        }
    });
});
