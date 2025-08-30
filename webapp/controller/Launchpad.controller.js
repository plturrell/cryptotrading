sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/Text",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function(Controller, MessageToast, Dialog, Button, Text, Fragment, JSONModel) {
    "use strict";

    return Controller.extend("com.rex.cryptotrading.controller.Launchpad", {
        
        onInit: function() {
            // Very defensive initialization
            try {
                console.log("Launchpad controller onInit started");
                
                // Wait for models to be available before proceeding
                var that = this;
                var checkModels = function() {
                    try {
                        var oComponent = that.getOwnerComponent();
                        var oAppModel = oComponent ? oComponent.getModel("app") : null;
                        
                        if (oAppModel) {
                            console.log("App model is available");
                            that._initializeController();
                        } else {
                            console.log("Waiting for app model...");
                            setTimeout(checkModels, 100);
                        }
                    } catch (e) {
                        console.warn("Model check failed, retrying:", e.message);
                        setTimeout(checkModels, 100);
                    }
                };
                
                // Start checking for models
                setTimeout(checkModels, 10);
                
            } catch (e) {
                console.error("Controller onInit error:", e);
            }
        },
        
        _initializeController: function() {
            try {
                console.log("Initializing controller with available models");
                
                // Initialize basic properties safely
                this._oEventBusManager = null;
                this._oWalletDataManager = {
                    status: "disconnected",
                    connection: { connected: false, provider: null, address: null, network: null, chainId: null },
                    balances: {},
                    error: null
                };
                
                console.log("Controller initialization completed successfully");
            } catch (e) {
                console.error("Controller initialization error:", e);
            }
        },
        
        onExit: function() {
            // Cleanup EventBus subscriptions
            if (this._oEventBusManager) {
                this._oEventBusManager.unsubscribeAll(this);
            }
        },
        
        _setupEventHandlers: function() {
            if (!this._oEventBusManager || !this._oEventBusManager.subscribe) {
                return;
            }
            
            try {
                // Subscribe to wallet connection events
                this._oEventBusManager.subscribe(
                    this._oEventBusManager.CHANNELS.WALLET,
                    this._oEventBusManager.EVENTS.WALLET.CONNECTION_CHANGED,
                    this._onWalletConnectionChanged.bind(this),
                    this
                );
                
                // Subscribe to UI notifications
                this._oEventBusManager.subscribe(
                    this._oEventBusManager.CHANNELS.UI,
                    this._oEventBusManager.EVENTS.UI.NOTIFICATION,
                    this._onNotification.bind(this),
                    this
                );
            } catch (e) {
                console.warn("EventBus not available:", e);
            }
        },
        
        _onWalletConnectionChanged: function(sChannel, sEvent, oData) {
            if (oData.connected) {
                MessageToast.show("Wallet connected: " + oData.address.substring(0, 8) + "...");
            } else {
                MessageToast.show("Wallet disconnected");
            }
        },
        
        _onNotification: function(sChannel, sEvent, oData) {
            MessageToast.show(oData.message);
        },
        
        onTilePress: function(oEvent) {
            var oTile = oEvent.getSource();
            var oBindingContext = oTile.getBindingContext("app");
            
            // Check if binding context exists
            if (!oBindingContext) {
                console.warn("No binding context found for tile");
                MessageToast.show("Navigation not configured for this tile");
                return;
            }
            
            var sTilePress = oBindingContext.getProperty("press");
            var sTitle = oBindingContext.getProperty("title");
            
            if (!sTilePress) {
                console.warn("No press action configured for tile");
                MessageToast.show("Navigation not configured for this tile");
                return;
            }
            
            MessageToast.show("Opening " + (sTitle || "Application") + "...");
            
            // Navigate to the appropriate view
            try {
                var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
                oRouter.navTo(sTilePress);
            } catch (e) {
                console.error("Navigation error:", e);
                MessageToast.show("Navigation failed: " + e.message);
            }
        },
        
        onMenuPress: function() {
            MessageToast.show("Menu functionality coming soon");
        },
        
        onNotificationPress: function() {
            if (!this._oNotificationPopover) {
                this._oNotificationPopover = new Dialog({
                    title: "Notifications",
                    content: [
                        new Text({ text: "BTC price alert: $65,432" }),
                        new Text({ text: "New AI signal available" }),
                        new Text({ text: "ML model retrained successfully" })
                    ],
                    beginButton: new Button({
                        text: "Close",
                        press: function() {
                            this._oNotificationPopover.close();
                        }.bind(this)
                    })
                });
            }
            this._oNotificationPopover.open();
        },
        
        onAvatarPress: function(oEvent) {
            var oButton = oEvent.getSource();
            
            if (!this._oUserPopover) {
                Fragment.load({
                    name: "com.rex.cryptotrading.fragment.UserMenu",
                    controller: this
                }).then(function(oPopover) {
                    this._oUserPopover = oPopover;
                    this.getView().addDependent(this._oUserPopover);
                    this._oUserPopover.openBy(oButton);
                }.bind(this));
            } else {
                this._oUserPopover.openBy(oButton);
            }
        },
        
        onConnectWallet: function() {
            var that = this;
            
            if (typeof window.ethereum !== 'undefined') {
                // Get wallet data manager
                var oWalletModel = sap.ui.getCore().getModel("wallet");
                var oWalletDataManager = oWalletModel.getProperty("/");
                
                // Set connecting status
                oWalletModel.setProperty("/status", "connecting");
                
                window.ethereum.request({ method: 'eth_requestAccounts' })
                    .then(function(accounts) {
                        if (accounts.length > 0) {
                            // Use wallet data manager to connect
                            var oWalletManager = that._getWalletDataManager();
                            oWalletManager.connectWallet('metamask', accounts[0], 'mainnet', 1);
                            
                            // Update balances
                            that._updateWalletBalance(accounts[0]);
                            
                            // Legacy model update for backward compatibility
                            var oAppModel = that.getView().getModel("app");
                            oAppModel.setProperty("/wallet/connected", true);
                            oAppModel.setProperty("/wallet/address", accounts[0]);
                        }
                    })
                    .catch(function(error) {
                        var oWalletManager = that._getWalletDataManager();
                        oWalletManager.setError(error.message);
                        MessageToast.show("Failed to connect wallet: " + error.message);
                    });
            } else {
                MessageToast.show("Please install MetaMask to connect your wallet");
            }
        },
        
        _getWalletDataManager: function() {
            // Helper to get wallet data manager instance
            // In a real implementation, this would access the manager instance
            return {
                connectWallet: function(provider, address, network, chainId) {
                    var oWalletModel = sap.ui.getCore().getModel("wallet");
                    oWalletModel.setProperty("/connection/connected", true);
                    oWalletModel.setProperty("/connection/provider", provider);
                    oWalletModel.setProperty("/connection/address", address);
                    oWalletModel.setProperty("/connection/network", network);
                    oWalletModel.setProperty("/connection/chainId", chainId);
                    oWalletModel.setProperty("/status", "connected");
                },
                setError: function(error) {
                    var oWalletModel = sap.ui.getCore().getModel("wallet");
                    oWalletModel.setProperty("/error", { message: error });
                    oWalletModel.setProperty("/status", "error");
                }
            };
        },
        
        _updateWalletBalance: function(address) {
            var that = this;
            
            // Use modern data manager approach
            jQuery.ajax({
                url: "/api/wallet/balance",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({ address: address }),
                success: function(data) {
                    // Update wallet model
                    var oWalletModel = sap.ui.getCore().getModel("wallet");
                    if (data && data.balances) {
                        oWalletModel.setProperty("/balances", data.balances);
                        
                        // Publish balance updated event
                        that._oEventBusManager.publishBalanceUpdated(data.balances);
                    }
                    
                    // Keep legacy model update
                    var oModel = that.getView().getModel("app");
                    if (data.balance) {
                        oModel.setProperty("/wallet/balance", data.balance.ETH);
                    }
                }.bind(this),
                error: function() {
                    console.error("Failed to update wallet balance");
                }
            });
        },
        
        onPortfolioPress: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("portfolio");
        },
        
        onCodeAnalysisPress: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("codeAnalysis");
        },
        
        onTradingConsolePress: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("tradingConsole");
        },
        
        onTechnicalAnalysisPress: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("technicalAnalysis", {
                symbol: "BTC-USD"
            });
        },
        
        // Additional press handlers for static tiles
        onMarketOverviewPress: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("market");
        },
        
        onMLPredictionsPress: function() {
            MessageToast.show("ML Predictions dashboard coming soon");
        },
        
        onModelTrainingPress: function() {
            MessageToast.show("Model Training interface coming soon");
        },
        
        onAIMarketIntelligencePress: function() {
            MessageToast.show("AI Market Intelligence coming soon");
        },
        
        onFeatureStorePress: function() {
            MessageToast.show("Feature Store interface coming soon");
        },
        
        onRiskManagementPress: function() {
            MessageToast.show("Risk Management dashboard coming soon");
        },
        
        onHistoricalDataPress: function() {
            MessageToast.show("Historical Data viewer coming soon");
        },
        
        onSystemSettingsPress: function() {
            MessageToast.show("System Settings coming soon");
        }
    });
});