/**
 * Extension Manager for Crypto Trading Application
 * Provides enterprise-level extension framework for modular functionality
 */
sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function (BaseObject, Log) {
    "use strict";

    return BaseObject.extend("com.rex.cryptotrading.extensions.ExtensionManager", {

        constructor: function () {
            BaseObject.prototype.constructor.apply(this, arguments);
            this._aExtensions = [];
            this._mExtensionRegistry = {};
            this._bInitialized = false;
        },

        /**
         * Initialize the extension manager
         * @public
         */
        init: function () {
            if (this._bInitialized) {
                return;
            }

            Log.info("Initializing Extension Manager");
            this._loadCoreExtensions();
            this._bInitialized = true;
        },

        /**
         * Register an extension
         * @public
         * @param {string} sName Extension name
         * @param {Object} oExtension Extension object
         * @param {Object} oConfig Extension configuration
         */
        registerExtension: function (sName, oExtension, oConfig) {
            if (!sName || !oExtension) {
                Log.error("Extension name and object are required");
                return false;
            }

            if (this._mExtensionRegistry[sName]) {
                Log.warning("Extension '" + sName + "' is already registered");
                return false;
            }

            var oExtensionWrapper = {
                name: sName,
                extension: oExtension,
                config: oConfig || {},
                enabled: true,
                initialized: false,
                dependencies: oConfig?.dependencies || []
            };

            this._mExtensionRegistry[sName] = oExtensionWrapper;
            this._aExtensions.push(oExtensionWrapper);

            Log.info("Extension '" + sName + "' registered successfully");
            return true;
        },

        /**
         * Get an extension by name
         * @public
         * @param {string} sName Extension name
         * @returns {Object|null} Extension object or null
         */
        getExtension: function (sName) {
            var oWrapper = this._mExtensionRegistry[sName];
            return oWrapper ? oWrapper.extension : null;
        },

        /**
         * Enable an extension
         * @public
         * @param {string} sName Extension name
         */
        enableExtension: function (sName) {
            var oWrapper = this._mExtensionRegistry[sName];
            if (oWrapper) {
                oWrapper.enabled = true;
                Log.info("Extension '" + sName + "' enabled");
            }
        },

        /**
         * Disable an extension
         * @public
         * @param {string} sName Extension name
         */
        disableExtension: function (sName) {
            var oWrapper = this._mExtensionRegistry[sName];
            if (oWrapper) {
                oWrapper.enabled = false;
                Log.info("Extension '" + sName + "' disabled");
            }
        },

        /**
         * Initialize all registered extensions
         * @public
         */
        initializeExtensions: function () {
            // Sort extensions by dependencies
            var aSortedExtensions = this._sortExtensionsByDependencies();

            aSortedExtensions.forEach(function (oWrapper) {
                if (oWrapper.enabled && !oWrapper.initialized) {
                    this._initializeExtension(oWrapper);
                }
            }.bind(this));
        },

        /**
         * Load core extensions
         * @private
         */
        _loadCoreExtensions: function () {
            // Register core trading extensions
            this.registerExtension("marketData", {
                init: function () {
                    Log.info("Market Data extension initialized");
                },
                getMarketData: function (symbol) {
                    return this._fetchMarketData(symbol);
                },
                _fetchMarketData: function (symbol) {
                    // Implementation would go here
                    return Promise.resolve({});
                }
            }, {
                dependencies: [],
                autoInit: true
            });

            this.registerExtension("trading", {
                init: function () {
                    Log.info("Trading extension initialized");
                },
                executeTrade: function (order) {
                    return this._executeOrder(order);
                },
                _executeOrder: function (order) {
                    // Implementation would go here
                    return Promise.resolve({});
                }
            }, {
                dependencies: ["marketData"],
                autoInit: true
            });

            this.registerExtension("analytics", {
                init: function () {
                    Log.info("Analytics extension initialized");
                },
                calculateIndicators: function (data) {
                    return this._performAnalysis(data);
                },
                _performAnalysis: function (data) {
                    // Implementation would go here
                    return {};
                }
            }, {
                dependencies: ["marketData"],
                autoInit: true
            });

            this.registerExtension("riskManagement", {
                init: function () {
                    Log.info("Risk Management extension initialized");
                },
                assessRisk: function (portfolio) {
                    return this._calculateRisk(portfolio);
                },
                _calculateRisk: function (portfolio) {
                    // Implementation would go here
                    return {};
                }
            }, {
                dependencies: ["trading", "analytics"],
                autoInit: true
            });
        },

        /**
         * Initialize a single extension
         * @private
         * @param {Object} oWrapper Extension wrapper
         */
        _initializeExtension: function (oWrapper) {
            try {
                if (oWrapper.extension.init && typeof oWrapper.extension.init === "function") {
                    oWrapper.extension.init();
                }
                oWrapper.initialized = true;
                Log.info("Extension '" + oWrapper.name + "' initialized successfully");
            } catch (error) {
                Log.error("Failed to initialize extension '" + oWrapper.name + "'", error);
            }
        },

        /**
         * Sort extensions by dependencies
         * @private
         * @returns {Array} Sorted extensions array
         */
        _sortExtensionsByDependencies: function () {
            var aResult = [];
            var mVisited = {};
            var mInProgress = {};

            var fnVisit = function (oWrapper) {
                if (mInProgress[oWrapper.name]) {
                    Log.error("Circular dependency detected for extension: " + oWrapper.name);
                    return;
                }

                if (mVisited[oWrapper.name]) {
                    return;
                }

                mInProgress[oWrapper.name] = true;

                // Visit dependencies first
                oWrapper.dependencies.forEach(function (sDependency) {
                    var oDependencyWrapper = this._mExtensionRegistry[sDependency];
                    if (oDependencyWrapper) {
                        fnVisit(oDependencyWrapper);
                    }
                }.bind(this));

                mInProgress[oWrapper.name] = false;
                mVisited[oWrapper.name] = true;
                aResult.push(oWrapper);
            }.bind(this);

            this._aExtensions.forEach(fnVisit);
            return aResult;
        },

        /**
         * Get all registered extensions
         * @public
         * @returns {Array} Array of extension names
         */
        getRegisteredExtensions: function () {
            return Object.keys(this._mExtensionRegistry);
        },

        /**
         * Check if extension is enabled
         * @public
         * @param {string} sName Extension name
         * @returns {boolean} True if enabled
         */
        isExtensionEnabled: function (sName) {
            var oWrapper = this._mExtensionRegistry[sName];
            return oWrapper ? oWrapper.enabled : false;
        },

        /**
         * Destroy the extension manager
         * @public
         */
        destroy: function () {
            // Cleanup extensions
            this._aExtensions.forEach(function (oWrapper) {
                if (oWrapper.extension.destroy && typeof oWrapper.extension.destroy === "function") {
                    try {
                        oWrapper.extension.destroy();
                    } catch (error) {
                        Log.error("Error destroying extension '" + oWrapper.name + "'", error);
                    }
                }
            });

            this._aExtensions = [];
            this._mExtensionRegistry = {};
            this._bInitialized = false;

            BaseObject.prototype.destroy.apply(this, arguments);
        }
    });
});
