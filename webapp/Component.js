/**
 * Enterprise Component for Crypto Trading Application
 * Enhanced with memory leak prevention, performance monitoring, and enterprise patterns
 */
sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "sap/ui/model/json/JSONModel",
    "sap/base/Log",
    "./utils/SharedErrorHandlingUtils",
    "./utils/SharedSecurityUtils",
    "./utils/MarketDataService",
    "./utils/TradingService",
    "./utils/CacheManager",
    "./service/ServiceRegistry",
    "./extensions/ExtensionManager"
], function (
    UIComponent,
    Device,
    JSONModel,
    Log,
    SharedErrorHandlingUtils,
    SharedSecurityUtils,
    MarketDataService,
    TradingService,
    CacheManager,
    ServiceRegistry,
    ExtensionManager
) {
    "use strict";

    return UIComponent.extend("com.rex.cryptotrading.Component", {

        metadata: {
            manifest: "json"
        },

        /**
         * The component is initialized by UI5 automatically during the startup of the app
         * and calls the init method once.
         * @public
         * @override
         */
        init: function () {
            // Initialize cleanup tasks array
            this._aCleanupTasks = [];
            this._aIntervals = [];
            this._aTimeouts = [];

            // Initialize performance monitoring
            this._initPerformanceMonitoring();

            // Initialize memory leak prevention
            this._initMemoryLeakPrevention();

            // Initialize shared utilities
            this._initSharedUtils();

            // Initialize service registry
            this._initServiceRegistry();

            // Initialize extension manager
            this._initExtensionManager();

            // Initialize services
            this._initServices();

            // Initialize WebSocket connections
            this._initWebSocketConnections();

            // Initialize global error handling
            this._initGlobalErrorHandling();

            // Initialize user preferences
            this._initUserPreferences();

            // Disable UI5 flexibility features for performance
            if (sap.ui.fl && sap.ui.fl.Utils) {
                if (typeof sap.ui.fl.Utils.isApplicationVariant === "undefined") {
                    sap.ui.fl.Utils.isApplicationVariant = function() { return false; };
                    sap.ui.fl.Utils.isVariantByStartupParameter = function() { return false; };
                }
            }

            // Call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);

            // Initialize models first before routing
            this._initializeModelsAndData();

            // Enable routing after models are set up
            this.getRouter().initialize();
        },

        /**
         * Initialize performance monitoring
         * @private
         */
        _initPerformanceMonitoring: function () {
            this._oPerformanceMetrics = {
                startTime: Date.now(),
                navigationTimes: [],
                renderTimes: [],
                apiCallTimes: []
            };

            // Track component initialization time
            this.registerForCleanup(() => {
                const initTime = Date.now() - this._oPerformanceMetrics.startTime;
                Log.info("Component initialization completed in " + initTime + "ms");
            });
        },

        /**
         * Initialize memory leak prevention
         * @private
         */
        _initMemoryLeakPrevention: function () {
            // Override setInterval to track intervals
            const originalSetInterval = window.setInterval;
            const _that = this;

            window.setInterval = function (callback, delay) {
                const intervalId = originalSetInterval(callback, delay);
                that._aIntervals.push(intervalId);
                return intervalId;
            };

            // Override setTimeout to track timeouts
            const originalSetTimeout = window.setTimeout;

            window.setTimeout = function (callback, delay) {
                const timeoutId = originalSetTimeout(callback, delay);
                that._aTimeouts.push(timeoutId);
                return timeoutId;
            };

            // Register cleanup for intervals and timeouts
            this.registerForCleanup(() => {
                that._aIntervals.forEach(clearInterval);
                that._aTimeouts.forEach(clearTimeout);
                that._aIntervals = [];
                that._aTimeouts = [];
            });
        },

        /**
         * Initialize shared utilities
         * @private
         */
        _initSharedUtils: function () {
            try {
                // Initialize error handling utils
                this._errorHandlingUtils = SharedErrorHandlingUtils;

                // Initialize security utils
                this._securityUtils = SharedSecurityUtils;

                // Register cleanup for utilities
                this.registerForCleanup(() => {
                    if (this._errorHandlingUtils && this._errorHandlingUtils.cleanup) {
                        this._errorHandlingUtils.cleanup();
                    }
                    if (this._securityUtils && this._securityUtils.cleanup) {
                        this._securityUtils.cleanup();
                    }
                });

                Log.info("Shared utilities initialized successfully");
            } catch (error) {
                Log.error("Failed to initialize shared utilities", error);
            }
        },

        /**
         * Initialize services
         * @private
         */
        _initServices: function () {
            try {
                // Initialize cache manager firs
                this._cacheManager = new CacheManager();

                // Initialize market data service
                this._marketDataService = new MarketDataService({
                    cacheManager: this._cacheManager
                });

                // Initialize trading service
                this._tradingService = new TradingService({
                    cacheManager: this._cacheManager,
                    marketDataService: this._marketDataService
                });

                // Register cleanup for services
                this.registerForCleanup(() => {
                    if (this._tradingService && this._tradingService.destroy) {
                        this._tradingService.destroy();
                    }
                    if (this._marketDataService && this._marketDataService.destroy) {
                        this._marketDataService.destroy();
                    }
                    if (this._cacheManager && this._cacheManager.destroy) {
                        this._cacheManager.destroy();
                    }
                });

                Log.info("Services initialized successfully");
            } catch (error) {
                Log.error("Failed to initialize services", error);
            }
        },

        /**
         * Initialize WebSocket connections
         * @private
         */
        _initWebSocketConnections: function () {
            try {
                // WebSocket URL based on current environmen
                const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
                const wsUrl = protocol + "//" + window.location.host + "/ws/market-data";

                this._webSocket = new WebSocket(wsUrl);

                this._webSocket.onopen = () => {
                    Log.info("WebSocket connection established");
                };

                this._webSocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this._handleWebSocketMessage(data);
                    } catch (error) {
                        Log.error("Failed to parse WebSocket message", error);
                    }
                };

                this._webSocket.onerror = (error) => {
                    Log.error("WebSocket error", error);
                };

                this._webSocket.onclose = () => {
                    Log.info("WebSocket connection closed");
                };

                // Register cleanup for WebSocke
                this.registerForCleanup(() => {
                    if (this._webSocket && this._webSocket.readyState === WebSocket.OPEN) {
                        this._webSocket.close();
                    }
                });

            } catch (error) {
                Log.error("Failed to initialize WebSocket connections", error);
            }
        },

        /**
         * Handle WebSocket messages
         * @private
         * @param {Object} data Message data
         */
        _handleWebSocketMessage: function (data) {
            if (data.type === "market-data" && this._marketDataService) {
                this._marketDataService.updateMarketData(data.payload);
            } else if (data.type === "trading-update" && this._tradingService) {
                this._tradingService.handleTradingUpdate(data.payload);
            }
        },

        /**
         * Initialize global error handling
         * @private
         */
        _initGlobalErrorHandling: function () {
            // Global error handler for unhandled promises
            window.addEventListener("unhandledrejection", (event) => {
                Log.error("Unhandled promise rejection", event.reason);
                if (this._errorHandlingUtils) {
                    this._errorHandlingUtils.handleError(event.reason, "UNHANDLED_PROMISE");
                }
            });

            // Global error handler for JavaScript errors
            window.addEventListener("error", (event) => {
                Log.error("Global JavaScript error", event.error);
                if (this._errorHandlingUtils) {
                    this._errorHandlingUtils.handleError(event.error, "JAVASCRIPT_ERROR");
                }
            });
        },

        /**
         * Initialize user preferences
         * @private
         */
        _initUserPreferences: function () {
            try {
                // Load user preferences from localStorage
                const storedPreferences = localStorage.getItem("cryptotrading.userPreferences");
                this._userPreferences = storedPreferences ? JSON.parse(storedPreferences) : {
                    theme: "sap_horizon",
                    language: "en",
                    currency: "USD",
                    notifications: true,
                    autoRefresh: true,
                    refreshInterval: 30000
                };

                // Apply preferences
                this._applyUserPreferences();

                Log.info("User preferences initialized");
            } catch (error) {
                Log.error("Failed to initialize user preferences", error);
                // Fallback to default preferences
                this._userPreferences = {
                    theme: "sap_horizon",
                    language: "en",
                    currency: "USD",
                    notifications: true,
                    autoRefresh: true,
                    refreshInterval: 30000
                };
            }
        },

        /**
         * Apply user preferences
         * @private
         */
        _applyUserPreferences: function () {
            // Apply theme
            if (this._userPreferences.theme) {
                sap.ui.getCore().applyTheme(this._userPreferences.theme);
            }

            // Apply language
            if (this._userPreferences.language) {
                sap.ui.getCore().getConfiguration().setLanguage(this._userPreferences.language);
            }
        },

        /**
         * Initialize models and data
         * @private
         */
        _initializeModelsAndData: function () {
            // Set device model
            const oDeviceModel = new JSONModel(Device);
            oDeviceModel.setDefaultBindingMode("OneWay");
            this.setModel(oDeviceModel, "device");

            // Initialize application model with real data structure
            const oAppModel = new JSONModel({
                user: {
                    name: "Trading User",
                    role: "Trader",
                    greeting: "Good morning"
                },
                wallet: {
                    address: null,
                    balance: {
                        ETH: null,
                        BTC: null,
                        total: null
                    },
                    connected: false
                },
                marketData: {
                    btcPrice: null,
                    btcChange: null,
                    btcIndicator: null,
                    ethPrice: null,
                    ethChange: null,
                    ethIndicator: null,
                    totalMarketCap: null,
                    lastUpdated: null,
                    loading: true
                },
                portfolio: {
                    totalValue: null,
                    change24h: null,
                    changePercent: null,
                    positions: null,
                    profitLoss: null,
                    loading: true
                },
                aiAnalysis: {
                    signal: null,
                    confidence: null,
                    recommendation: null,
                    lastUpdate: null,
                    loading: true
                },
                riskMetrics: {
                    valueAtRisk: null,
                    confidence: null,
                    maxDrawdown: null,
                    sharpeRatio: null,
                    loading: true
                }
            });

            this.setModel(oAppModel, "app");

            // Initialize UI state model
            const oUIModel = new JSONModel({
                isLoading: false,
                hasError: false,
                errorMessage: "",
                currentView: "dashboard",
                sidebarExpanded: true,
                notifications: []
            });

            this.setModel(oUIModel, "ui");
        },

        /**
         * Component destruction
         * @public
         * @override
         */
        destroy: function () {
            // Execute cleanup tasks
            this._executeCleanupTasks();

            // Call parent destroy
            UIComponent.prototype.destroy.apply(this, arguments);

            Log.info("Component destroyed successfully");
        },

        /**
         * Execute cleanup tasks
         * @private
         */
        _executeCleanupTasks: function () {
            this._aCleanupTasks.forEach((fnCleanup) => {
                try {
                    fnCleanup();
                } catch (error) {
                    Log.error("Error during cleanup task execution", error);
                }
            });
            this._aCleanupTasks = [];
        },

        /**
         * Register a cleanup task to be executed on component destruction
         * @public
         * @param {function} fnCleanup Cleanup function
         */
        registerForCleanup: function (fnCleanup) {
            if (typeof fnCleanup === "function") {
                this._aCleanupTasks.push(fnCleanup);
            }
        },

        /**
         * Gets the market data service instance
         * @public
         * @returns {Object} Market data service
         */
        getMarketDataService: function () {
            return this._marketDataService;
        },

        /**
         * Gets the trading service instance
         * @public
         * @returns {Object} Trading service
         */
        getTradingService: function () {
            return this._tradingService;
        },

        /**
         * Gets the cache manager instance
         * @public
         * @returns {Object} Cache manager
         */
        getCacheManager: function () {
            return this._cacheManager;
        },

        /**
         * Gets the service registry instance
         * @public
         * @returns {Object} Service registry
         */
        getServiceRegistry: function () {
            return this._serviceRegistry;
        },

        /**
         * Gets the extension manager instance
         * @public
         * @returns {Object} Extension manager
         */
        getExtensionManager: function () {
            return this._extensionManager;
        },

        /**
         * Initialize service registry
         * @private
         */
        _initServiceRegistry: function () {
            this._serviceRegistry = new ServiceRegistry();
            this._serviceRegistry.init();

            this.registerForCleanup(() => {
                if (this._serviceRegistry) {
                    this._serviceRegistry.destroy();
                    this._serviceRegistry = null;
                }
            });

            Log.info("Service Registry initialized");
        },

        /**
         * Initialize extension manager
         * @private
         */
        _initExtensionManager: function () {
            this._extensionManager = new ExtensionManager();
            this._extensionManager.init();
            this._extensionManager.initializeExtensions();

            this.registerForCleanup(() => {
                if (this._extensionManager) {
                    this._extensionManager.destroy();
                    this._extensionManager = null;
                }
            });

            Log.info("Extension Manager initialized");
        }
    });
});
