sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Controller, JSONModel, MessageToast, MessageBox) {
    "use strict";

    return Controller.extend("com.rex.cryptotrading.controller.DataLoading", {

        onInit: function () {
            // Initialize model
            var oModel = new JSONModel({
                dataSources: [],
                activeJobs: [],
                availableSymbols: [
                    { key: "BTC", text: "Bitcoin (BTC)" },
                    { key: "ETH", text: "Ethereum (ETH)" },
                    { key: "BNB", text: "Binance Coin (BNB)" },
                    { key: "SOL", text: "Solana (SOL)" },
                    { key: "ADA", text: "Cardano (ADA)" },
                    { key: "XRP", text: "Ripple (XRP)" },
                    { key: "DOGE", text: "Dogecoin (DOGE)" },
                    { key: "DOT", text: "Polkadot (DOT)" },
                    { key: "MATIC", text: "Polygon (MATIC)" },
                    { key: "AVAX", text: "Avalanche (AVAX)" }
                ],
                fredSeries: [
                    { key: "DGS10", text: "10-Year Treasury Rate" },
                    { key: "WALCL", text: "Fed Balance Sheet" },
                    { key: "M2SL", text: "M2 Money Supply" },
                    { key: "RRPONTSYD", text: "Reverse Repo" },
                    { key: "EFFR", text: "Fed Funds Rate" },
                    { key: "T10Y2Y", text: "Yield Curve (10Y-2Y)" },
                    { key: "CPIAUCSL", text: "Consumer Price Index" },
                    { key: "UNRATE", text: "Unemployment Rate" }
                ],
                dexNetworks: [
                    { key: "ethereum", text: "Ethereum" },
                    { key: "bsc", text: "Binance Smart Chain" },
                    { key: "polygon", text: "Polygon" },
                    { key: "arbitrum", text: "Arbitrum" },
                    { key: "optimism", text: "Optimism" },
                    { key: "avalanche", text: "Avalanche" },
                    { key: "base", text: "Base" }
                ],
                yahooStartDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
                yahooEndDate: new Date(),
                fredStartDate: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), // 1 year ago
                fredEndDate: new Date()
            });
            
            this.getView().setModel(oModel, "dataLoading");
            
            // Load initial status
            this._loadDataSourceStatus();
            this._loadActiveJobs();
            
            // Set up auto-refresh
            this._startAutoRefresh();
        },
        
        onExit: function() {
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
            }
        },
        
        _startAutoRefresh: function() {
            // Refresh every 5 seconds
            this._refreshInterval = setInterval(() => {
                this._loadActiveJobs();
            }, 5000);
        },
        
        _loadDataSourceStatus: function() {
            var that = this;
            
            jQuery.ajax({
                url: "/api/odata/v4/DataLoadingService/getDataSourceStatus",
                method: "GET",
                success: function(data) {
                    that.getView().getModel("dataLoading").setProperty("/dataSources", data);
                },
                error: function(error) {
                    console.error("Failed to load data source status:", error);
                }
            });
        },
        
        _loadActiveJobs: function() {
            var that = this;
            
            jQuery.ajax({
                url: "/api/odata/v4/DataLoadingService/getActiveJobs",
                method: "GET",
                success: function(data) {
                    that.getView().getModel("dataLoading").setProperty("/activeJobs", data);
                },
                error: function(error) {
                    console.error("Failed to load active jobs:", error);
                }
            });
        },
        
        onLoadYahooData: function() {
            var oModel = this.getView().getModel("dataLoading");
            var aSymbols = this.byId("yahooSymbols").getSelectedKeys();
            var oDateRange = this.byId("yahooDateRange");
            var sInterval = this.byId("yahooInterval").getSelectedKey();
            
            if (aSymbols.length === 0) {
                MessageBox.warning("Please select at least one symbol");
                return;
            }
            
            var oData = {
                symbols: aSymbols,
                startDate: oDateRange.getDateValue().toISOString(),
                endDate: oDateRange.getSecondDateValue().toISOString(),
                interval: sInterval
            };
            
            jQuery.ajax({
                url: "/api/odata/v4/DataLoadingService/loadYahooFinanceData",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(result) {
                    MessageToast.show("Yahoo Finance data loading started: " + result.message);
                    this._loadActiveJobs();
                }.bind(this),
                error: function(error) {
                    MessageBox.error("Failed to start Yahoo data loading");
                }
            });
        },
        
        onLoadFREDData: function() {
            var oModel = this.getView().getModel("dataLoading");
            var aSeries = this.byId("fredSeries").getSelectedKeys();
            var oDateRange = this.byId("fredDateRange");
            
            if (aSeries.length === 0) {
                MessageBox.warning("Please select at least one FRED series");
                return;
            }
            
            var oData = {
                series: aSeries,
                startDate: oDateRange.getDateValue().toISOString(),
                endDate: oDateRange.getSecondDateValue().toISOString()
            };
            
            jQuery.ajax({
                url: "/api/odata/v4/DataLoadingService/loadFREDData",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(result) {
                    MessageToast.show("FRED data loading started: " + result.message);
                    this._loadActiveJobs();
                }.bind(this),
                error: function(error) {
                    MessageBox.error("Failed to start FRED data loading");
                }
            });
        },
        
        onLoadGeckoData: function() {
            var aNetworks = this.byId("geckoNetworks").getSelectedKeys();
            var iPoolCount = this.byId("poolCount").getValue();
            var bIncludeVolume = this.byId("includeVolume").getSelected();
            var bIncludeLiquidity = this.byId("includeLiquidity").getSelected();
            
            if (aNetworks.length === 0) {
                MessageBox.warning("Please select at least one network");
                return;
            }
            
            var oData = {
                networks: aNetworks,
                poolCount: iPoolCount,
                includeVolume: bIncludeVolume,
                includeLiquidity: bIncludeLiquidity
            };
            
            jQuery.ajax({
                url: "/api/odata/v4/DataLoadingService/loadGeckoTerminalData",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(result) {
                    MessageToast.show("GeckoTerminal data loading started: " + result.message);
                    this._loadActiveJobs();
                }.bind(this),
                error: function(error) {
                    MessageBox.error("Failed to start GeckoTerminal data loading");
                }
            });
        },
        
        onLoadAllData: function() {
            var that = this;
            
            MessageBox.confirm("This will load data from all sources. Continue?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        that._loadAllData();
                    }
                }
            });
        },
        
        _loadAllData: function() {
            var aYahooSymbols = this.byId("yahooSymbols").getSelectedKeys();
            var aFredSeries = this.byId("fredSeries").getSelectedKeys();
            var aNetworks = this.byId("geckoNetworks").getSelectedKeys();
            var oYahooDateRange = this.byId("yahooDateRange");
            
            var oData = {
                cryptoSymbols: aYahooSymbols,
                fredSeries: aFredSeries,
                dexNetworks: aNetworks,
                startDate: oYahooDateRange.getDateValue().toISOString(),
                endDate: oYahooDateRange.getSecondDateValue().toISOString()
            };
            
            jQuery.ajax({
                url: "/api/odata/v4/DataLoadingService/loadAllMarketData",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(result) {
                    MessageToast.show("All data sources loading started: " + result.totalJobs + " jobs queued");
                    this._loadActiveJobs();
                }.bind(this),
                error: function(error) {
                    MessageBox.error("Failed to start bulk data loading");
                }
            });
        },
        
        onCancelJob: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("dataLoading");
            var sJobId = oContext.getProperty("jobId");
            
            jQuery.ajax({
                url: "/api/odata/v4/DataLoadingService/cancelLoadingJob",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({ jobId: sJobId }),
                success: function(result) {
                    MessageToast.show("Job cancelled: " + sJobId);
                    this._loadActiveJobs();
                }.bind(this),
                error: function(error) {
                    MessageBox.error("Failed to cancel job");
                }
            });
        },
        
        onCancelAllJobs: function() {
            MessageBox.confirm("Cancel all active jobs?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        // Cancel each active job
                        var aJobs = this.getView().getModel("dataLoading").getProperty("/activeJobs");
                        aJobs.forEach(function(job) {
                            this.onCancelJob({ 
                                getSource: function() { 
                                    return { 
                                        getBindingContext: function() { 
                                            return { 
                                                getProperty: function() { return job.jobId; } 
                                            }; 
                                        } 
                                    }; 
                                } 
                            });
                        }.bind(this));
                    }
                }.bind(this)
            });
        },
        
        onRefreshStatus: function() {
            this._loadDataSourceStatus();
            this._loadActiveJobs();
            MessageToast.show("Status refreshed");
        },
        
        onViewLogs: function() {
            MessageBox.information("Log viewing will be implemented in a future update");
        },
        
        onNavBack: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("launchpad");
        }
    });
});