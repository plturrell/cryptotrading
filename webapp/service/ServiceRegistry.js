/**
 * Service Registry for Crypto Trading Application
 * Centralized service management and configuration
 */
sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function (BaseObject, Log) {
    "use strict";

    return BaseObject.extend("com.rex.cryptotrading.service.ServiceRegistry", {

        constructor: function () {
            BaseObject.prototype.constructor.apply(this, arguments);
            this._mServices = {};
            this._mConfigurations = {};
            this._bInitialized = false;
        },

        /**
         * Initialize the service registry
         * @public
         */
        init: function () {
            if (this._bInitialized) {
                return;
            }

            Log.info("Initializing Service Registry");
            this._setupDefaultConfigurations();
            this._registerCoreServices();
            this._bInitialized = true;
        },

        /**
         * Register a service
         * @public
         * @param {string} sServiceName Service name
         * @param {Object} oServiceConfig Service configuration
         */
        registerService: function (sServiceName, oServiceConfig) {
            if (!sServiceName || !oServiceConfig) {
                Log.error("Service name and configuration are required");
                return false;
            }

            this._mConfigurations[sServiceName] = oServiceConfig;
            Log.info("Service '" + sServiceName + "' registered successfully");
            return true;
        },

        /**
         * Get service configuration
         * @public
         * @param {string} sServiceName Service name
         * @returns {Object|null} Service configuration
         */
        getServiceConfig: function (sServiceName) {
            return this._mConfigurations[sServiceName] || null;
        },

        /**
         * Get service instance
         * @public
         * @param {string} sServiceName Service name
         * @returns {Object|null} Service instance
         */
        getService: function (sServiceName) {
            if (!this._mServices[sServiceName]) {
                var oConfig = this._mConfigurations[sServiceName];
                if (oConfig) {
                    this._mServices[sServiceName] = this._createServiceInstance(sServiceName, oConfig);
                }
            }
            return this._mServices[sServiceName] || null;
        },

        /**
         * Setup default service configurations
         * @private
         */
        _setupDefaultConfigurations: function () {
            // Market Data Service Configuration
            this.registerService("marketData", {
                type: "websocket",
                endpoint: "/ws/market-data",
                fallbackEndpoint: "/api/v1/market-data",
                reconnectInterval: 5000,
                maxReconnectAttempts: 10,
                supportedSymbols: ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD"],
                updateInterval: 1000
            });

            // Trading Service Configuration
            this.registerService("trading", {
                type: "rest",
                baseUrl: "/api/v1/trading",
                endpoints: {
                    orders: "/orders",
                    positions: "/positions",
                    history: "/history",
                    balance: "/balance"
                },
                timeout: 30000,
                retryAttempts: 3,
                riskLimits: {
                    maxPositionSize: 0.1,
                    maxDailyLoss: 0.05
                }
            });

            // Portfolio Service Configuration
            this.registerService("portfolio", {
                type: "rest",
                baseUrl: "/api/v1/portfolio",
                endpoints: {
                    summary: "/summary",
                    holdings: "/holdings",
                    performance: "/performance",
                    allocation: "/allocation"
                },
                cacheTimeout: 60000
            });

            // Analytics Service Configuration
            this.registerService("analytics", {
                type: "rest",
                baseUrl: "/api/v1/analytics",
                endpoints: {
                    indicators: "/indicators",
                    signals: "/signals",
                    backtesting: "/backtesting",
                    reports: "/reports"
                },
                supportedIndicators: ["SMA", "EMA", "RSI", "MACD", "BOLLINGER"],
                cacheDuration: 300000
            });

            // Risk Management Service Configuration
            this.registerService("risk", {
                type: "rest",
                baseUrl: "/api/v1/risk",
                endpoints: {
                    assessment: "/assessment",
                    limits: "/limits",
                    monitoring: "/monitoring",
                    alerts: "/alerts"
                },
                monitoringInterval: 10000,
                alertThresholds: {
                    drawdown: 0.15,
                    volatility: 0.05,
                    concentration: 0.25
                }
            });

            // Wallet Service Configuration
            this.registerService("wallet", {
                type: "blockchain",
                networks: {
                    ethereum: {
                        rpcUrl: "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
                        chainId: 1
                    },
                    polygon: {
                        rpcUrl: "https://polygon-rpc.com",
                        chainId: 137
                    }
                },
                supportedWallets: ["MetaMask", "WalletConnect", "Coinbase"],
                gasLimits: {
                    transfer: 21000,
                    swap: 150000,
                    approve: 50000
                }
            });

            // News Service Configuration
            this.registerService("news", {
                type: "rest",
                baseUrl: "/api/v1/news",
                endpoints: {
                    latest: "/latest",
                    search: "/search",
                    categories: "/categories",
                    sentiment: "/sentiment"
                },
                refreshInterval: 300000,
                categories: ["market", "regulation", "technology", "adoption"]
            });
        },

        /**
         * Register core services
         * @private
         */
        _registerCoreServices: function () {
            // Core services are registered but instances created on demand
            Log.info("Core services registered in configuration");
        },

        /**
         * Create service instance
         * @private
         * @param {string} sServiceName Service name
         * @param {Object} oConfig Service configuration
         * @returns {Object} Service instance
         */
        _createServiceInstance: function (sServiceName, oConfig) {
            var oServiceInstance = {
                name: sServiceName,
                config: oConfig,
                status: "initialized",
                lastUpdate: new Date(),

                // Common service methods
                isHealthy: function () {
                    return this.status === "connected" || this.status === "ready";
                },

                getEndpoint: function (sEndpointName) {
                    if (this.config.endpoints && this.config.endpoints[sEndpointName]) {
                        return this.config.baseUrl + this.config.endpoints[sEndpointName];
                    }
                    return this.config.baseUrl || this.config.endpoint;
                },

                updateStatus: function (sStatus) {
                    this.status = sStatus;
                    this.lastUpdate = new Date();
                    Log.info("Service '" + this.name + "' status updated to: " + sStatus);
                }
            };

            // Add service-specific methods based on type
            switch (oConfig.type) {
                case "websocket":
                    oServiceInstance.connect = function () {
                        this.updateStatus("connecting");
                        // WebSocket connection logic would go here
                        this.updateStatus("connected");
                    };
                    oServiceInstance.disconnect = function () {
                        this.updateStatus("disconnected");
                    };
                    break;

                case "rest":
                    oServiceInstance.request = function (sMethod, sEndpoint, oData) {
                        return new Promise(function (resolve, reject) {
                            // REST request logic would go here
                            setTimeout(function () {
                                resolve({ success: true, data: {} });
                            }, 100);
                        });
                    };
                    break;

                case "blockchain":
                    oServiceInstance.connectWallet = function (sWalletType) {
                        return new Promise(function (resolve, reject) {
                            // Wallet connection logic would go here
                            resolve({ connected: true, address: "0x..." });
                        });
                    };
                    break;
            }

            return oServiceInstance;
        },

        /**
         * Get all registered services
         * @public
         * @returns {Array} Array of service names
         */
        getRegisteredServices: function () {
            return Object.keys(this._mConfigurations);
        },

        /**
         * Check service health
         * @public
         * @param {string} sServiceName Service name
         * @returns {boolean} Health status
         */
        isServiceHealthy: function (sServiceName) {
            var oService = this.getService(sServiceName);
            return oService ? oService.isHealthy() : false;
        },

        /**
         * Get service status
         * @public
         * @param {string} sServiceName Service name
         * @returns {string} Service status
         */
        getServiceStatus: function (sServiceName) {
            var oService = this.getService(sServiceName);
            return oService ? oService.status : "not_found";
        },

        /**
         * Destroy the service registry
         * @public
         */
        destroy: function () {
            // Cleanup all service instances
            Object.keys(this._mServices).forEach(function (sServiceName) {
                var oService = this._mServices[sServiceName];
                if (oService && oService.destroy) {
                    oService.destroy();
                }
            }.bind(this));

            this._mServices = {};
            this._mConfigurations = {};
            this._bInitialized = false;

            BaseObject.prototype.destroy.apply(this, arguments);
        }
    });
});
