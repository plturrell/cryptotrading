sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "../utils/Constants"
], function (BaseController, MessageToast, MessageBox, Constants) {
    "use strict";

    return BaseController.extend("com.rex.cryptotrading.controller.App", {

        onInit: function () {
            BaseController.prototype.onInit.apply(this, arguments);

            // Initialize app model
            this._initializeAppModel();

            // Set up routing
            this._setupRouting();

            // Initialize services
            this._initializeServices();

            // Set up WebSocket connections
            this._setupWebSocketConnections();

            // Initialize navigation
            this._initializeNavigation();
        },

        _initializeAppModel: function () {
            const oAppModel = new sap.ui.model.json.JSONModel({
                busy: false,
                selectedKey: "home",
                user: {
                    initials: "CT"
                },
                wallet: {
                    connected: false,
                    address: "",
                    balance: "0.0000"
                },
                networkStatus: {
                    text: "Mainnet",
                    icon: "sap-icon://connected"
                },
                notificationCount: "3"
            });

            this.getView().setModel(oAppModel, "app");
        },

        _setupRouting: function () {
            const oRouter = this.getRouter();
            oRouter.attachRouteMatched(this._onRouteMatched, this);
        },

        _initializeServices: function () {
            // Initialize ServiceRegistry and ExtensionManager
            const oComponent = this.getOwnerComponent();
            const oServiceRegistry = oComponent.getServiceRegistry();
            const oExtensionManager = oComponent.getExtensionManager();

            if (oServiceRegistry) {
                // Services are already initialized in Component.js
                this.getLogger().info("ServiceRegistry initialized with services", oServiceRegistry.getRegisteredServices());
            }

            if (oExtensionManager) {
                // Extensions are already loaded in Component.js
                this.getLogger().info("ExtensionManager initialized with plugins", oExtensionManager.getRegisteredPlugins());
            }
        },

        _setupWebSocketConnections: function () {
            // WebSocket setup is handled in Component.js
            this.getLogger().info("WebSocket connections managed by Component");
        },

        _initializeNavigation: function () {
            // Set default navigation
            const oRouter = this.getRouter();
            oRouter.navTo("dashboard");
        },

        _onRouteMatched: function (oEvent) {
            const sRouteName = oEvent.getParameter("name");
            const oAppModel = this.getView().getModel("app");

            // Update selected navigation key based on route
            let sSelectedKey = "home";
            switch (sRouteName) {
            case "dashboard":
                sSelectedKey = "home";
                break;
            case "market":
                sSelectedKey = "marketOverview";
                break;
            case "trading":
                sSelectedKey = "trading";
                break;
            case "analytics":
                sSelectedKey = "technicalAnalysis";
                break;
            case "portfolio":
                sSelectedKey = "portfolio";
                break;
            case "news":
                sSelectedKey = "news";
                break;
            case "risk":
                sSelectedKey = "risk";
                break;
            case "settings":
                sSelectedKey = "settings";
                break;
            }

            oAppModel.setProperty("/selectedKey", sSelectedKey);
        },

        // Navigation handlers
        onSideNavButtonPress: function () {
            const oToolPage = this.byId("toolPage");
            const bSideExpanded = oToolPage.getSideExpanded();
            oToolPage.setSideExpanded(!bSideExpanded);
        },

        onItemSelect: function (oEvent) {
            const oItem = oEvent.getParameter("item");
            const sKey = oItem.getKey();

            // Navigate based on selected key
            switch (sKey) {
            case "home":
                this.navTo("dashboard");
                break;
            case "marketOverview":
                this.navTo("market");
                break;
            case "trading":
                this.navTo("trading");
                break;
            case "technicalAnalysis":
                this.navTo("analytics");
                break;
            case "portfolio":
                this.navTo("portfolio");
                break;
            case "news":
                this.navTo("news");
                break;
            case "risk":
                this.navTo("risk");
                break;
            case "settings":
                this.navTo("settings");
                break;
            }
        },

        onWalletPress: function () {
            const _oModel = this.getView().getModel("app");
            const bConnected = oModel.getProperty("/wallet/connected");

            if (!bConnected) {
                // Connect to MetaMask
                this._connectWallet();
            } else {
                // Show wallet details
                const sAddress = oModel.getProperty("/wallet/address");
                const fBalance = oModel.getProperty("/wallet/balance");

                MessageBox.information(
                    "Address: " + sAddress + "\n" +
                    "Balance: " + fBalance + " ETH",
                    {
                        title: "Wallet Information"
                    }
                );
            }
        },

        onNotificationPress: function () {
            MessageToast.show("Notifications clicked");
        },

        onAvatarPress: function () {
            MessageToast.show("User profile clicked");
        },

        _connectWallet: function () {
            if (typeof window.ethereum !== "undefined") {
                // MetaMask is installed
                window.ethereum.request({ method: "eth_requestAccounts" })
                    .then(function(accounts) {
                        if (accounts.length > 0) {
                            const _oModel = this.getView().getModel("app");
                            oModel.setProperty("/wallet/address", accounts[0]);
                            oModel.setProperty("/wallet/connected", true);

                            // Get balance
                            window.ethereum.request({
                                method: "eth_getBalance",
                                params: [accounts[0], "latest"]
                            }).then(function(balance) {
                                // Convert from wei to ETH
                                const ethBalance = parseInt(balance, Constants.NUMBERS.HEX_BASE) /
                                    Constants.NUMBERS.WEI_TO_ETH;
                                oModel.setProperty("/wallet/balance", ethBalance.toFixed(Constants.NUMBERS.DECIMAL_PLACES));
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
        }
    });
});
