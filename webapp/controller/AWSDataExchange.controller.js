sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel",
    "../utils/Constants"
], function (Controller, MessageToast, MessageBox, JSONModel, Constants) {
    "use strict";

    return Controller.extend("cryptotrading.controller.AWSDataExchange", {

        onInit: function () {
            // Initialize models
            this._initializeModels();
            this._checkServiceStatus();
        },

        _initializeModels: function () {
            // AWS Data Exchange model
            const oAWSModel = new JSONModel({
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
            const oModel = this.getView().getModel("aws");

            this._makeAjaxCall("/api/odata/v4/AWSDataExchange/getServiceStatus", "GET", null,
                (data) => {
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
                },
                () => {
                    oModel.setProperty("/serviceStatus/loading", false);
                    MessageToast.show("Failed to check AWS Data Exchange service status");
                }
            );
        },

        _makeAjaxCall: function(sUrl, sMethod, oData, fnSuccess, fnError) {
            const oSettings = {
                url: sUrl,
                method: sMethod,
                success: fnSuccess.bind(this),
                error: fnError.bind(this)
            };

            if (oData && sMethod !== "GET") {
                oSettings.contentType = "application/json";
                oSettings.data = JSON.stringify(oData);
            }

            jQuery.ajax(oSettings);
        },

        _loadAvailableDatasets: function () {
            const oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/datasets", true);

            this._makeAjaxCall("/api/odata/v4/AWSDataExchange/getAvailableDatasets?type=all", "GET", null,
                (data) => {
                    oModel.setProperty("/datasets/all", data.datasets || []);
                    oModel.setProperty("/loading/datasets", false);
                },
                () => {
                    oModel.setProperty("/loading/datasets", false);
                    MessageToast.show("Failed to load AWS datasets");
                }
            );
        },

        onDiscoverCryptoData: function () {
            this._discoverDatasets("/api/odata/v4/AWSDataExchange/discoverCryptoData", "crypto", "crypto datasets");
        },

        _discoverDatasets: function(sUrl, sDatasetType, sTypeDesc) {
            const oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/datasets", true);

            this._makeAjaxCall(sUrl, "GET", null,
                (data) => {
                    const sKey = sDatasetType === "crypto" ? "crypto_datasets" : "economic_datasets";
                    oModel.setProperty(`/datasets/${sDatasetType}`, data[sKey] || []);
                    oModel.setProperty("/loading/datasets", false);
                    MessageToast.show(`Found ${data.dataset_count} ${sTypeDesc}`);
                },
                () => {
                    oModel.setProperty("/loading/datasets", false);
                    MessageToast.show(`Failed to discover ${sTypeDesc}`);
                }
            );
        },

        onDiscoverEconomicData: function () {
            this._discoverDatasets("/api/odata/v4/AWSDataExchange/discoverEconomicData", "economic", "economic datasets");
        },

        onDatasetSelect: function (oEvent) {
            const oSelectedItem = oEvent.getParameter("listItem");
            const oContext = oSelectedItem.getBindingContext("aws");
            const oDataset = oContext.getObject();

            const _oModel = this.getView().getModel("aws");
            oModel.setProperty("/selectedDataset", oDataset);

            // Load assets for selected datase
            this._loadDatasetAssets(oDataset.dataset_id);
        },

        _loadDatasetAssets: function (sDatasetId) {
            const oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/assets", true);
            oModel.setProperty("/assets", []);

            this._makeAjaxCall("/api/odata/v4/AWSDataExchange/getDatasetAssets", "POST",
                { dataset_id: sDatasetId },
                (data) => {
                    oModel.setProperty("/assets", data.assets || []);
                    oModel.setProperty("/loading/assets", false);
                    MessageToast.show(`Found ${data.asset_count} assets in dataset`);
                },
                () => {
                    oModel.setProperty("/loading/assets", false);
                    MessageToast.show("Failed to load dataset assets");
                }
            );
        },

        onAssetSelect: function (oEvent) {
            const oSelectedItem = oEvent.getParameter("listItem");
            const oContext = oSelectedItem.getBindingContext("aws");
            const oAsset = oContext.getObject();

            const _oModel = this.getView().getModel("aws");
            oModel.setProperty("/selectedAsset", oAsset);
        },

        onLoadDatasetToDatabase: function () {
            const _oModel = this.getView().getModel("aws");
            const oSelectedDataset = oModel.getProperty("/selectedDataset");
            const oSelectedAsset = oModel.getProperty("/selectedAsset");

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
            const oModel = this.getView().getModel("aws");
            oModel.setProperty("/loading/dataLoad", true);

            const sTableName = this._generateTableName(oDataset.dataset_id, oAsset.asset_id);
            const oLoadData = {
                dataset_id: oDataset.dataset_id,
                asset_id: oAsset.asset_id,
                table_name: sTableName
            };

            this._makeAjaxCall("/api/odata/v4/AWSDataExchange/loadDatasetToDatabase", "POST", oLoadData,
                (data) => this._handleDataLoadSuccess(data, oModel),
                (xhr) => this._handleDataLoadError(xhr, oModel)
            );
        },

        _generateTableName: function(sDatasetId, sAssetId) {
            return `aws_${sDatasetId}_${sAssetId}`.substring(Constants.NUMBERS.ZERO, Constants.NUMBERS.MAX_FILE_SIZE);
        },

        _handleDataLoadSuccess: function(data, oModel) {
            oModel.setProperty("/loading/dataLoad", false);
            MessageBox.success(
                "Successfully loaded data!\n\n" +
                `Records loaded: ${data.records_loaded}\n` +
                `Table: ${data.table_name}\n` +
                `Columns: ${data.columns.length}\n` +
                `Data shape: ${data.data_shape[0]} rows × ${data.data_shape[1]} columns`,
                { title: "Data Loading Complete" }
            );
        },

        _handleDataLoadError: function(xhr, oModel) {
            oModel.setProperty("/loading/dataLoad", false);

            let sErrorMsg = "Failed to load dataset";
            try {
                const oError = JSON.parse(xhr.responseText);
                sErrorMsg = oError.error || sErrorMsg;
            } catch (e) {
                // Ignore JSON parsing errors, use default message
            }

            MessageBox.error(sErrorMsg, { title: "Data Loading Failed" });
        },

        onRefreshDatasets: function () {
            this._loadAvailableDatasets();
        },

        onShowSetupInstructions: function () {
            const _oModel = this.getView().getModel("aws");
            const oStatus = oModel.getProperty("/serviceStatus");

            let sInstructions = "AWS Data Exchange Setup Instructions:\n\n";

            if (oStatus.setupInstructions) {
                Object.keys(oStatus.setupInstructions).forEach(function (sKey) {
                    sInstructions += `${sKey.replace("_", " ").toUpperCase()}: ${oStatus.setupInstructions[sKey]}\n`;
                });
            }

            sInstructions += "\nEnvironment Variables:\n";
            if (oStatus.environmentVars) {
                Object.keys(oStatus.environmentVars).forEach(function (sKey) {
                    const sValue = oStatus.environmentVars[sKey];
                    sInstructions += `${sKey}: ${sValue === true ? "✓ Set" : sValue === false ? "✗ Not set" : sValue}\n`;
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
            return sDescription.length > Constants.NUMBERS.LARGE_LIMIT ? sDescription.substring(Constants.NUMBERS.ZERO, Constants.NUMBERS.LARGE_LIMIT) + "..." : sDescription;
        },

        formatFileSize: function (iSizeMB) {
            if (!iSizeMB) return "Unknown size";
            return iSizeMB < Constants.NUMBERS.ONE ? "<1 MB" : `${iSizeMB} MB`;
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
