sap.ui.define([
    "com/rex/cryptotrading/controller/BaseController",
    "../utils/Constants"
], function (BaseController, Constants) {
    "use strict";

    return BaseController.extend("com.rex.cryptotrading.controller.Dashboard", {

        onInit: function () {
            BaseController.prototype.onInit.apply(this, arguments);

            // Initialize dashboard models
            this._initializeDashboardModels();

            // Set up real-time data updates
            this._setupRealTimeUpdates();
        },

        _initializeDashboardModels: function () {
            // Initialize dashboard-specific models with sample data
            const oDashboardModel = new sap.ui.model.json.JSONModel({
                marketData: {
                    totalMarketCap: 2.1
                },
                portfolio: {
                    totalValue: 125.5,
                    performanceState: "Good"
                },
                trading: {
                    activeOrders: 3
                },
                analytics: {
                    signals: 7
                },
                news: {
                    unreadCount: 12
                },
                risk: {
                    riskScore: 25,
                    riskState: "Good"
                }
            });

            this.getView().setModel(oDashboardModel, "dashboard");
        },

        _setupRealTimeUpdates: function () {
            // Set up periodic updates for dashboard tiles
            this._updateInterval = setInterval(() => {
                this._updateDashboardData();
            }, Constants.TIME.REFRESH_INTERVAL); // Update every 30 seconds
        },

        _updateDashboardData: function () {
            // Update dashboard data from services
            const oServiceRegistry = this.getServiceRegistry();

            if (oServiceRegistry) {
                // Update market data
                const oMarketService = oServiceRegistry.getService("marketData");
                if (oMarketService) {
                    // Fetch latest market data
                }

                // Update portfolio data
                const oPortfolioService = oServiceRegistry.getService("portfolio");
                if (oPortfolioService) {
                    // Fetch latest portfolio data
                }
            }
        },

        // Navigation handlers
        onNavigateToMarket: function () {
            this.getRouter().navTo("market");
        },

        onNavigateToPortfolio: function () {
            this.getRouter().navTo("portfolio");
        },

        onNavigateToTrading: function () {
            this.getRouter().navTo("trading");
        },

        onNavigateToAnalysis: function () {
            this.getRouter().navTo("analytics");
        },

        onNavigateToNews: function () {
            this.getRouter().navTo("news");
        },

        onNavigateToRisk: function () {
            this.getRouter().navTo("risk");
        },

        onExit: function () {
            // Clean up intervals
            if (this._updateInterval) {
                clearInterval(this._updateInterval);
            }

            BaseController.prototype.onExit.apply(this, arguments);
        }
    });
});
