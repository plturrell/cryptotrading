sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/Text",
    "sap/ui/core/Fragment"
], function(Controller, MessageToast, Dialog, Button, Text, Fragment) {
    "use strict";

    return Controller.extend("com.рекс.cryptotrading.controller.Launchpad", {
        
        onInit: function() {
            // Initialize controller
        },
        
        onTilePress: function(oEvent) {
            var oTile = oEvent.getSource();
            var oBindingContext = oTile.getBindingContext("app");
            var sTilePress = oBindingContext.getProperty("press");
            
            MessageToast.show("Opening " + oBindingContext.getProperty("title") + "...");
            
            // Navigate to the appropriate view
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo(sTilePress);
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
                        new Text({ text: "DEX opportunity detected" })
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
                    name: "com.рекс.cryptotrading.fragment.UserMenu",
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
            if (typeof window.ethereum !== 'undefined') {
                window.ethereum.request({ method: 'eth_requestAccounts' })
                    .then(function(accounts) {
                        if (accounts.length > 0) {
                            var oModel = this.getView().getModel("app");
                            oModel.setProperty("/wallet/connected", true);
                            oModel.setProperty("/wallet/address", accounts[0]);
                            MessageToast.show("Wallet connected successfully");
                            
                            // Update wallet balance
                            this._updateWalletBalance(accounts[0]);
                        }
                    }.bind(this))
                    .catch(function(error) {
                        MessageToast.show("Failed to connect wallet: " + error.message);
                    });
            } else {
                MessageToast.show("Please install MetaMask to connect your wallet");
            }
        },
        
        _updateWalletBalance: function(address) {
            jQuery.ajax({
                url: "/api/wallet/balance",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({ address: address }),
                success: function(data) {
                    var oModel = this.getView().getModel("app");
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
        }
    });
});