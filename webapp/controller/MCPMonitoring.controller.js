sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/ui/core/format/DateFormat",
    "sap/ui/export/Spreadsheet",
    "sap/ui/export/library",
    "../utils/Constants"
], function (Controller, JSONModel, MessageToast, DateFormat, Spreadsheet, exportLibrary, Constants) {
    "use strict";

    const EdmType = exportLibrary.EdmType;

    return Controller.extend("com.rex.cryptotrading.controller.MCPMonitoring", {
        onInit: function () {
            // Initialize model with monitoring data
            const _oModel = new JSONModel({
                activeTools: 16,
                totalRequests: 45678,
                avgResponseTime: 125,
                successRate: 98.7,
                activeTraces: 234,
                totalTools: 16,
                healthyTools: 15,
                toolVersions: 8,
                activeServers: 7,
                loadDistribution: "Balanced",
                failovers: 2,
                requestTimeline: this._generateTimelineData(),
                traces: this._generateTraceData(),
                tools: this._getToolsData()
            });

            this.getView().setModel(oModel);

            // Start real-time updates
            this._startRealTimeUpdates();

            // Initialize charts
            this._initializeCharts();
        },

        onAfterRendering: function() {
            // Configure chart properties
            const oVizFrame = this.byId("requestChart");
            if (oVizFrame) {
                oVizFrame.setVizProperties({
                    plotArea: {
                        window: {
                            start: "firstDataPoint",
                            end: "lastDataPoint"
                        },
                        dataLabel: {
                            formatString: "#,##0"
                        }
                    },
                    valueAxis: {
                        label: {
                            formatString: "#,##0"
                        },
                        title: {
                            visible: true,
                            text: "Requests"
                        }
                    },
                    timeAxis: {
                        title: {
                            visible: true,
                            text: "Time"
                        },
                        interval: {
                            unit: ""
                        }
                    },
                    title: {
                        visible: false
                    }
                });
            }
        },

        _initializeCharts: function() {
            // Initialize any additional chart configurations
            const oVizFrame = this.byId("requestChart");
            if (oVizFrame) {
                oVizFrame.setVizType("timeseries_line");
            }
        },

        _startRealTimeUpdates: function() {
            // Update data every 5 seconds
            this._updateInterval = setInterval(() => {
                this._updateMonitoringData();
            }, Constants.TIME.AUTO_REFRESH);
        },

        _updateMonitoringData: function() {
            const _oModel = this.getView().getModel();
            const currentData = oModel.getData();

            // Simulate real-time updates
            currentData.totalRequests += Math.floor(Math.random() * 100);
            currentData.activeTraces = Math.floor(Math.random() * Constants.NUMBERS.DEFAULT_LIMIT) +
                Constants.NUMBERS.LARGE_LIMIT;
            currentData.avgResponseTime = Math.floor(Math.random() * Constants.NUMBERS.DEFAULT_LIMIT) +
                Constants.NUMBERS.PERCENTAGE_MULTIPLIER;
            currentData.successRate = (Constants.NUMBERS.PERCENTAGE_98 +
                Math.random() * Constants.NUMBERS.PERCENTAGE_PLACES).toFixed(1);

            // Update timeline with new data poin
            const timeline = currentData.requestTimeline;
            if (timeline.length > Constants.NUMBERS.DEFAULT_LIMIT) {
                timeline.shift(); // Remove oldest poin
            }
            timeline.push({
                time: new Date().toISOString(),
                requests: Math.floor(Math.random() * 500) + 200,
                errors: Math.floor(Math.random() * 10)
            });

            oModel.setData(currentData);
        },

        _generateTimelineData: function() {
            const data = [];
            const _now = new Date();

            for (let i = 24; i >= 0; i--) {
                const time = new Date(now.getTime() - i * 3600000);
                data.push({
                    time: time.toISOString(),
                    requests: Math.floor(Math.random() * 500) + 200,
                    errors: Math.floor(Math.random() * 20)
                });
            }

            return data;
        },

        _generateTraceData: function() {
            const tools = ["TechnicalAnalysisTool", "MLModelsTool", "HistoricalDataTool", "FeatureEngineeringTool"];
            const methods = ["execute", "analyze", "predict", "fetch", "train", "optimize"];
            const statuses = ["Success", "Success", "Success", "Warning", "Error"];
            const traces = [];

            for (let i = 0; i < 50; i++) {
                traces.push({
                    traceId: this._generateTraceId(),
                    tool: tools[Math.floor(Math.random() * tools.length)],
                    method: methods[Math.floor(Math.random() * methods.length)],
                    duration: Math.floor(Math.random() * 1000) + 50,
                    status: statuses[Math.floor(Math.random() * statuses.length)],
                    timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString()
                });
            }

            return traces;
        },

        _generateTraceId: function() {
            return "trace-" + Math.random().toString(36).substr(2, 9);
        },

        _getToolsData: function() {
            return [
                {
                    name: "TechnicalAnalysisTool",
                    version: "2.1.0",
                    status: "Healthy",
                    requests: 245,
                    latency: 45,
                    successRate: 99.8,
                    cpu: 85,
                    memory: 45
                },
                {
                    name: "MLModelsTool",
                    version: "1.5.3",
                    status: "Healthy",
                    requests: 156,
                    latency: 320,
                    successRate: 98.5,
                    cpu: 60,
                    memory: 75
                }
            ];
        },

        onRefresh: function() {
            this._updateMonitoringData();
            MessageToast.show("Dashboard refreshed");
        },

        onSettings: function() {
            // Open settings dialog
            if (!this._settingsDialog) {
                this._settingsDialog = sap.ui.xmlfragment(
                    "com.rex.cryptotrading.fragment.MCPSettings",
                    this
                );
                this.getView().addDependent(this._settingsDialog);
            }
            this._settingsDialog.open();
        },

        onExport: function() {
            const _oModel = this.getView().getModel();
            const aTraces = oModel.getProperty("/traces");

            const aCols = [
                {
                    label: "Trace ID",
                    property: "traceId",
                    type: EdmType.String
                },
                {
                    label: "Tool",
                    property: "tool",
                    type: EdmType.String
                },
                {
                    label: "Method",
                    property: "method",
                    type: EdmType.String
                },
                {
                    label: "Duration (ms)",
                    property: "duration",
                    type: EdmType.Number
                },
                {
                    label: "Status",
                    property: "status",
                    type: EdmType.String
                },
                {
                    label: "Timestamp",
                    property: "timestamp",
                    type: EdmType.DateTime
                }
            ];

            const oSettings = {
                workbook: {
                    columns: aCols
                },
                dataSource: aTraces,
                fileName: "MCP_Traces_" + new Date().getTime() + ".xlsx"
            };

            const oSheet = new Spreadsheet(oSettings);
            oSheet.build().finally(function() {
                oSheet.destroy();
            });

            MessageToast.show("Export started");
        },

        onToolDetails: function(oEvent) {
            const oSource = oEvent.getSource();
            const oCard = oSource.getParent().getParent();
            const sToolName = oCard.getHeader().getTitle();

            // Navigate to tool details
            const oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("toolDetails", {
                toolId: sToolName
            });
        },

        onSearchTraces: function(oEvent) {
            const sQuery = oEvent.getParameter("newValue");
            const oTable = this.byId("tracesTable");
            const oBinding = oTable.getBinding("items");

            if (oBinding) {
                const aFilters = [];
                if (sQuery) {
                    aFilters.push(new sap.ui.model.Filter({
                        filters: [
                            new sap.ui.model.Filter("traceId", sap.ui.model.FilterOperator.Contains, sQuery),
                            new sap.ui.model.Filter("tool", sap.ui.model.FilterOperator.Contains, sQuery),
                            new sap.ui.model.Filter("method", sap.ui.model.FilterOperator.Contains, sQuery)
                        ],
                        and: false
                    }));
                }
                oBinding.filter(aFilters);
            }
        },

        onFilterTraces: function() {
            // Open filter dialog
            MessageToast.show("Filter dialog would open here");
        },

        onTraceDetails: function(oEvent) {
            const oItem = oEvent.getSource();
            const oContext = oItem.getBindingContext();
            const oTrace = oContext.getObject();

            // Show trace details in a dialog
            MessageToast.show("Trace ID: " + oTrace.traceId);
        },

        formatDurationState: function(duration) {
            if (duration < 100) return "Success";
            if (duration < 500) return "Warning";
            return "Error";
        },

        formatStatusState: function(status) {
            switch(status) {
            case "Success": return "Success";
            case "Warning": return "Warning";
            case "Error": return "Error";
            default: return "None";
            }
        },

        formatTimestamp: function(timestamp) {
            const oDateFormat = DateFormat.getDateTimeInstance({
                pattern: "yyyy-MM-dd HH:mm:ss"
            });
            return oDateFormat.format(new Date(timestamp));
        },

        onExit: function() {
            // Clean up
            if (this._updateInterval) {
                clearInterval(this._updateInterval);
            }

            if (this._settingsDialog) {
                this._settingsDialog.destroy();
            }
        }
    });
});
