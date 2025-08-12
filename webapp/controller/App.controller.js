sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Controller, MessageToast, MessageBox) {
    "use strict";

    return Controller.extend("com.рекс.cryptotrading.controller.App", {

        onInit: function () {
            // Set tile model
            var oView = this.getView();
            var oModel = this.getOwnerComponent().getModel("app");
            var aTiles = oModel.getProperty("/tiles");
            
            // Bind tiles to grid container
            var oGridContainer = this.byId("gridContainer");
            aTiles.forEach(function(oTile) {
                var oGenericTile = new sap.m.GenericTile({
                    header: oTile.title,
                    subheader: oTile.subtitle,
                    press: this.onTilePress.bind(this),
                    frameType: "TwoByOne"
                });
                
                oGenericTile.data("route", oTile.route);
                
                var oTileContent = new sap.m.TileContent({
                    content: new sap.m.NumericContent({
                        value: oTile.number,
                        scale: oTile.unit,
                        state: oTile.state,
                        icon: oTile.icon
                    })
                });
                
                oGenericTile.addTileContent(oTileContent);
                
                oGenericTile.setLayoutData(new sap.f.GridContainerItemLayoutData({
                    columns: 4,
                    rows: 2
                }));
                
                oGridContainer.addItem(oGenericTile);
            }.bind(this));

            // Start real-time updates
            this._startRealtimeUpdates();
        },

        onTilePress: function (oEvent) {
            var oTile = oEvent.getSource();
            var sRoute = oTile.data("route");
            
            if (sRoute) {
                var oRouter = this.getOwnerComponent().getRouter();
                
                // For now, show message since routes aren't implemented yet
                MessageToast.show("Navigating to " + oTile.getHeader());
                
                // When routes are ready:
                // oRouter.navTo(sRoute);
            }
        },

        onWalletPress: function () {
            var oModel = this.getOwnerComponent().getModel("app");
            var bConnected = oModel.getProperty("/wallet/connected");
            
            if (!bConnected) {
                // Connect to MetaMask
                this._connectWallet();
            } else {
                // Show wallet details
                var sAddress = oModel.getProperty("/wallet/address");
                var fBalance = oModel.getProperty("/wallet/balance");
                
                MessageBox.information(
                    "Address: " + sAddress + "\n" +
                    "Balance: " + fBalance + " ETH",
                    {
                        title: "Wallet Information"
                    }
                );
            }
        },

        onAvatarPress: function () {
            MessageToast.show("User profile clicked");
        },

        _connectWallet: function () {
            if (typeof window.ethereum !== 'undefined') {
                // MetaMask is installed
                window.ethereum.request({ method: 'eth_requestAccounts' })
                    .then(function(accounts) {
                        if (accounts.length > 0) {
                            var oModel = this.getOwnerComponent().getModel("app");
                            oModel.setProperty("/wallet/address", accounts[0]);
                            oModel.setProperty("/wallet/connected", true);
                            
                            // Get balance
                            window.ethereum.request({
                                method: 'eth_getBalance',
                                params: [accounts[0], 'latest']
                            }).then(function(balance) {
                                // Convert from wei to ETH
                                var ethBalance = parseInt(balance, 16) / 1e18;
                                oModel.setProperty("/wallet/balance", ethBalance.toFixed(4));
                            });
                            
                            MessageToast.show("Wallet connected successfully!");
                        }
                    }.bind(this))
                    .catch(function(error) {
                        MessageBox.error("Failed to connect wallet: " + error.message);
                    });
            } else {
                MessageBox.warning("Please install MetaMask to connect your wallet");
            }
        },

        _startRealtimeUpdates: function () {
            // Update market data every 30 seconds
            setInterval(function() {
                this._updateMarketData();
            }.bind(this), 30000);

            // Update other tiles
            this._updateDEXData();
            this._updateAISignals();
        },

        _updateMarketData: function () {
            fetch('/api/market/overview?symbols=bitcoin,ethereum,binancecoin')
                .then(response => response.json())
                .then(data => {
                    if (data.symbols && data.symbols.bitcoin) {
                        var oGridContainer = this.byId("gridContainer");
                        var oMarketTile = oGridContainer.getItems()[0];
                        
                        if (oMarketTile) {
                            var oNumericContent = oMarketTile.getTileContent()[0].getContent();
                            oNumericContent.setValue(Math.round(data.symbols.bitcoin.prices.average).toLocaleString());
                            oNumericContent.setState("Success");
                        }
                    }
                }.bind(this))
                .catch(function(error) {
                    console.error("Error updating market data:", error);
                });
        },

        _updateDEXData: function () {
            fetch('/api/market/dex/trending')
                .then(function(response) {
                    return response.json();
                })
                .then(function(data) {
                    if (data.data && data.data.length > 0) {
                        var oGridContainer = this.byId("gridContainer");
                        var oDEXTile = oGridContainer.getItems()[4];
                        
                        if (oDEXTile) {
                            var oNumericContent = oDEXTile.getTileContent()[0].getContent();
                            oNumericContent.setValue(data.data.length);
                            oNumericContent.setState("Success");
                        }
                    }
                }.bind(this))
                .catch(function(error) {
                    console.error("Error updating DEX data:", error);
                });
        },

        _updateAISignals: function () {
            // Simulate AI signals for now
            var oGridContainer = this.byId("gridContainer");
            var oAITile = oGridContainer.getItems()[3];
            
            if (oAITile) {
                var oNumericContent = oAITile.getTileContent()[0].getContent();
                oNumericContent.setValue("3");
                oNumericContent.setState("Warning");
            }
        }
    });
});