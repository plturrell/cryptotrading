sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "sap/ui/model/json/JSONModel"
], function (UIComponent, Device, JSONModel) {
    "use strict";

    return UIComponent.extend("com.рекс.cryptotrading.Component", {

        metadata: {
            manifest: "json"
        },

        init: function () {
            // Call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);

            // Enable routing
            this.getRouter().initialize();

            // Set device model
            var oDeviceModel = new JSONModel(Device);
            oDeviceModel.setDefaultBindingMode("OneWay");
            this.setModel(oDeviceModel, "device");

            // Set initial app data model
            var oAppModel = new JSONModel({
                wallet: {
                    address: "0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1",
                    balance: 0,
                    connected: false
                },
                user: {
                    name: "Crypto Trader"
                },
                tiles: [
                    {
                        title: "Market Overview",
                        subtitle: "Real-time prices",
                        number: "0",
                        unit: "USD",
                        icon: "sap-icon://line-chart",
                        state: "None",
                        route: "marketOverview"
                    },
                    {
                        title: "Portfolio",
                        subtitle: "Your holdings",
                        number: "0",
                        unit: "USD",
                        icon: "sap-icon://wallet",
                        state: "None",
                        route: "portfolio"
                    },
                    {
                        title: "Trading",
                        subtitle: "Buy/Sell crypto",
                        number: "0",
                        unit: "Trades",
                        icon: "sap-icon://sales-order",
                        state: "None",
                        route: "trading"
                    },
                    {
                        title: "AI Analysis",
                        subtitle: "Market insights",
                        number: "0",
                        unit: "Signals",
                        icon: "sap-icon://business-objects-experience",
                        state: "None",
                        route: "aiAnalysis"
                    },
                    {
                        title: "DEX Monitor",
                        subtitle: "DeFi opportunities",
                        number: "0",
                        unit: "Pools",
                        icon: "sap-icon://chain-link",
                        state: "None",
                        route: "dexMonitor"
                    },
                    {
                        title: "Historical Data",
                        subtitle: "Download datasets",
                        number: "0",
                        unit: "Datasets",
                        icon: "sap-icon://download",
                        state: "None",
                        route: "historicalData"
                    }
                ]
            });
            this.setModel(oAppModel, "app");

            // Load initial data
            this._loadInitialData();
        },

        _loadInitialData: function() {
            // Load market overview
            fetch('/api/market/overview?symbols=bitcoin,ethereum,binancecoin')
                .then(response => response.json())
                .then(data => {
                    var oModel = this.getModel("app");
                    var aTiles = oModel.getProperty("/tiles");
                    
                    // Update market overview tile
                    if (data.symbols && data.symbols.bitcoin) {
                        aTiles[0].number = Math.round(data.symbols.bitcoin.prices.average).toLocaleString();
                        aTiles[0].state = "Success";
                    }
                    
                    oModel.setProperty("/tiles", aTiles);
                })
                .catch(error => console.error("Error loading market data:", error));

            // Check wallet status
            fetch('/api/wallet/balance')
                .then(response => response.json())
                .then(data => {
                    var oModel = this.getModel("app");
                    if (data.balance) {
                        oModel.setProperty("/wallet/balance", data.balance.ETH);
                        oModel.setProperty("/wallet/connected", true);
                    }
                })
                .catch(error => console.error("Error loading wallet:", error));
        }
    });
});