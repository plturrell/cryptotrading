sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/Text",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function(BaseController, MessageToast, Dialog, Button, Text, Fragment, JSONModel) {
    "use strict";

    return BaseController.extend("com.rex.cryptotrading.controller.Launchpad", {
        
        onInit: function() {
            // Call parent onInit
            BaseController.prototype.onInit.apply(this, arguments);
            
            try {
                // Initialize controller-specific properties
                this._initializeController();
                
            } catch (e) {
                this.handleError(e, "Controller initialization failed");
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
            
            // Navigate using BaseController method
            try {
                this.navTo(sTilePress);
            } catch (e) {
                this.handleError(e, "Navigation failed");
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
            // Use BaseController's secure request method
            this.makeSecureRequest("/api/wallet/balance", {
                method: "POST",
                body: JSON.stringify({ address: address })
            }).then(function(data) {
                // Update wallet model
                var oWalletModel = sap.ui.getCore().getModel("wallet");
                if (data && data.balances) {
                    oWalletModel.setProperty("/balances", data.balances);
                    
                    // Publish balance updated event
                    if (this._oEventBusManager && this._oEventBusManager.publishBalanceUpdated) {
                        this._oEventBusManager.publishBalanceUpdated(data.balances);
                    }
                }
                
                // Keep legacy model update
                var oModel = this.getView().getModel("app");
                if (data.balance) {
                    oModel.setProperty("/wallet/balance", data.balance.ETH);
                }
            }.bind(this)).catch(function(error) {
                this.handleError(error, "Failed to update wallet balance");
            }.bind(this));
        },
        
        onPortfolioPress: function() {
            this.navTo("portfolio");
        },
        
        onCodeAnalysisPress: function() {
            this.navTo("codeAnalysis");
        },
        
        onTradingConsolePress: function() {
            this.navTo("trading");
        },
        
        onTechnicalAnalysisPress: function() {
            this.navTo("analytics");
        },
        
        // Additional press handlers for static tiles
        onMarketOverviewPress: function() {
            this.navTo("market");
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