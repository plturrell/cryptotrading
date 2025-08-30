sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel"
], function (Controller, MessageToast, MessageBox, JSONModel) {
    "use strict";

    return Controller.extend("cryptotrading.controller.AWSDataExchange", {

        onInit: function () {
            // Initialize models
            this._initializeModels();
            this._checkServiceStatus();
        },

        _initializeModels: function () {
            // AWS Data Exchange model
            var oAWSModel = new JSONModel({
                serviceStatus: {
                    available: false,
                    awsCredentials: false,
                    loading: true
                },
                datasets: {
                    all: [],
                    crypto: [],
                    economic: []
                },
                selectedDataset: null,
                assets: [],
                selectedAsset: null,
                loading: {
                    datasets: false,
                    assets: false,
                    dataLoad: false
                }
            });
            this.getView().setModel(oAWSModel, "aws");
        },

        _checkServiceStatus: function () {
            var oModel = this.getView().getModel("aws");
            
            jQuery.ajax({
                url: "/api/odata/v4/AWSDataExchange/getServiceStatus",
                method: "GET",
                success: function (data) {
                    oModel.setProperty("/serviceStatus", {
                        available: data.service_available,
                        awsCredentials: data.aws_credentials_valid,
                        awsAccount: data.aws_account,
                        environmentVars: data.environment_variables,
                        setupInstructions: data.setup_instructions,
                        error: data.error,
                        loading: false
                    });
                    
                    if (data.service_available) {
                        this._loadAvailableDatasets();
                    }
                }.bind(this),
                error: function () {
                    oModel.setProperty("/serviceStatus/loading", false);
                    MessageToast.show("Failed to check AWS Data Exchange service status");
                }
            });
        },

        _loadAvailableDatasets: function () {
            var oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/datasets", true);
            
            // Load all datasets
            jQuery.ajax({
                url: "/api/odata/v4/AWSDataExchange/getAvailableDatasets?type=all",
                method: "GET",
                success: function (data) {
                    oModel.setProperty("/datasets/all", data.datasets || []);
                    oModel.setProperty("/loading/datasets", false);
                },
                error: function () {
                    oModel.setProperty("/loading/datasets", false);
                    MessageToast.show("Failed to load AWS datasets");
                }
            });
        },

        onDiscoverCryptoData: function () {
            var oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/datasets", true);
            
            jQuery.ajax({
                url: "/api/odata/v4/AWSDataExchange/discoverCryptoData",
                method: "GET",
                success: function (data) {
                    oModel.setProperty("/datasets/crypto", data.crypto_datasets || []);
                    oModel.setProperty("/loading/datasets", false);
                    MessageToast.show(`Found ${data.dataset_count} crypto datasets`);
                },
                error: function () {
                    oModel.setProperty("/loading/datasets", false);
                    MessageToast.show("Failed to discover crypto datasets");
                }
            });
        },

        onDiscoverEconomicData: function () {
            var oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/datasets", true);
            
            jQuery.ajax({
                url: "/api/odata/v4/AWSDataExchange/discoverEconomicData",
                method: "GET",
                success: function (data) {
                    oModel.setProperty("/datasets/economic", data.economic_datasets || []);
                    oModel.setProperty("/loading/datasets", false);
                    MessageToast.show(`Found ${data.dataset_count} economic datasets`);
                },
                error: function () {
                    oModel.setProperty("/loading/datasets", false);
                    MessageToast.show("Failed to discover economic datasets");
                }
            });
        },

        onDatasetSelect: function (oEvent) {
            var oSelectedItem = oEvent.getParameter("listItem");
            var oContext = oSelectedItem.getBindingContext("aws");
            var oDataset = oContext.getObject();
            
            var oModel = this.getView().getModel("aws");
            oModel.setProperty("/selectedDataset", oDataset);
            
            // Load assets for selected dataset
            this._loadDatasetAssets(oDataset.dataset_id);
        },

        _loadDatasetAssets: function (sDatasetId) {
            var oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/assets", true);
            oModel.setProperty("/assets", []);
            
            jQuery.ajax({
                url: "/api/odata/v4/AWSDataExchange/getDatasetAssets",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    dataset_id: sDatasetId
                }),
                success: function (data) {
                    oModel.setProperty("/assets", data.assets || []);
                    oModel.setProperty("/loading/assets", false);
                    MessageToast.show(`Found ${data.asset_count} assets in dataset`);
                },
                error: function () {
                    oModel.setProperty("/loading/assets", false);
                    MessageToast.show("Failed to load dataset assets");
                }
            });
        },

        onAssetSelect: function (oEvent) {
            var oSelectedItem = oEvent.getParameter("listItem");
            var oContext = oSelectedItem.getBindingContext("aws");
            var oAsset = oContext.getObject();
            
            var oModel = this.getView().getModel("aws");
            oModel.setProperty("/selectedAsset", oAsset);
        },

        onLoadDatasetToDatabase: function () {
            var oModel = this.getView().getModel("aws");
            var oSelectedDataset = oModel.getProperty("/selectedDataset");
            var oSelectedAsset = oModel.getProperty("/selectedAsset");
            
            if (!oSelectedDataset || !oSelectedAsset) {
                MessageToast.show("Please select both a dataset and an asset");
                return;
            }
            
            // Confirm loading
            MessageBox.confirm(
                `Load "${oSelectedAsset.name}" from "${oSelectedDataset.name}" to database?\n\n` +
                `Asset size: ${oSelectedAsset.size_mb} MB\n` +
                `Provider: ${oSelectedDataset.provider}`,
                {
                    title: "Confirm Data Loading",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._executeDataLoad(oSelectedDataset, oSelectedAsset);
                        }
                    }.bind(this)
                }
            );
        },

        _executeDataLoad: function (oDataset, oAsset) {
            var oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/dataLoad", true);
            
            // Generate table name
            var sTableName = `aws_${oDataset.dataset_id}_${oAsset.asset_id}`.substring(0, 50);
            
            jQuery.ajax({
                url: "/api/odata/v4/AWSDataExchange/loadDatasetToDatabase",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    dataset_id: oDataset.dataset_id,
                    asset_id: oAsset.asset_id,
                    table_name: sTableName
                }),
                success: function (data) {
                    oModel.setProperty("/loading/dataLoad", false);
                    
                    MessageBox.success(
                        `Successfully loaded data!\n\n` +
                        `Records loaded: ${data.records_loaded}\n` +
                        `Table: ${data.table_name}\n` +
                        `Columns: ${data.columns.length}\n` +
                        `Data shape: ${data.data_shape[0]} rows × ${data.data_shape[1]} columns`,
                        {
                            title: "Data Loading Complete"
                        }
                    );
                },
                error: function (xhr) {
                    oModel.setProperty("/loading/dataLoad", false);
                    
                    var sErrorMsg = "Failed to load dataset";
                    try {
                        var oError = JSON.parse(xhr.responseText);
                        sErrorMsg = oError.error || sErrorMsg;
                    } catch (e) {}
                    
                    MessageBox.error(sErrorMsg, {
                        title: "Data Loading Failed"
                    });
                }
            });
        },

        onRefreshDatasets: function () {
            this._loadAvailableDatasets();
        },

        onShowSetupInstructions: function () {
            var oModel = this.getView().getModel("aws");
            var oStatus = oModel.getProperty("/serviceStatus");
            
            var sInstructions = "AWS Data Exchange Setup Instructions:\n\n";
            
            if (oStatus.setupInstructions) {
                Object.keys(oStatus.setupInstructions).forEach(function (sKey) {
                    sInstructions += `${sKey.replace('_', ' ').toUpperCase()}: ${oStatus.setupInstructions[sKey]}\n`;
                });
            }
            
            sInstructions += "\nEnvironment Variables:\n";
            if (oStatus.environmentVars) {
                Object.keys(oStatus.environmentVars).forEach(function (sKey) {
                    var sValue = oStatus.environmentVars[sKey];
                    sInstructions += `${sKey}: ${sValue === true ? '✓ Set' : sValue === false ? '✗ Not set' : sValue}\n`;
                });
            }
            
            MessageBox.information(sInstructions, {
                title: "Setup Instructions"
            });
        },

        formatDatasetProvider: function (sProvider) {
            return sProvider || "Unknown Provider";
        },

        formatDatasetDescription: function (sDescription) {
            if (!sDescription) return "No description available";
            return sDescription.length > 150 ? sDescription.substring(0, 150) + "..." : sDescription;
        },

        formatFileSize: function (iSizeMB) {
            if (!iSizeMB) return "Unknown size";
            return iSizeMB < 1 ? "<1 MB" : `${iSizeMB} MB`;
        },

        formatDateTime: function (sDateTime) {
            if (!sDateTime) return "";
            try {
                return new Date(sDateTime).toLocaleString();
            } catch (e) {
                return sDateTime;
            }
        }

    });

});