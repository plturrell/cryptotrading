sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/Input",
    "sap/m/Label",
    "sap/m/VBox"
], function(Controller, MessageToast, MessageBox, Fragment, JSONModel, Dialog, Button, Input, Label, VBox) {
    "use strict";

    return Controller.extend("com.рекс.cryptotrading.controller.CodeAnalysis", {
        
        onInit: function() {
            this._initializeModels();
            this._loadAnalyticsData();
            this._loadIndexingSessions();
            this._loadBlindSpots();
            this._loadRecentResults();
            
            // Set up auto-refresh for active sessions
            this._setupAutoRefresh();
        },
        
        _loadRealAnalyticsData: function() {
            var that = this;
            var oAnalyticsModel = this.getView().getModel("analytics");
            
            // Load real project analytics from API
            jQuery.ajax({
                url: "/api/analytics/project-stats",
                type: "GET",
                success: function(data) {
                    if (data && !data.error) {
                        oAnalyticsModel.setData({
                            totalProjects: data.totalProjects || 0,
                            totalFiles: data.totalFiles || 0,
                            totalFacts: data.totalFacts || 0,
                            coveragePercent: data.coveragePercent || 0,
                            languages: data.languages || {},
                            loading: false,
                            error: null
                        });
                    } else {
                        throw new Error(data.error || "Invalid analytics data");
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Failed to load real analytics data:", error);
                    oAnalyticsModel.setProperty("/loading", false);
                    oAnalyticsModel.setProperty("/error", "Unable to load real analytics data");
                }
            });
        },
        
        _initializeModels: function() {
            // Analytics model - load real data from API
            var oAnalyticsModel = new JSONModel({
                totalProjects: null,
                totalFiles: null,
                totalFacts: null,
                coveragePercent: null,
                languages: {},
                loading: true,
                error: null
            });
            this.getView().setModel(oAnalyticsModel, "analytics");
            
            // Load real analytics data
            this._loadRealAnalyticsData();
            
            // Sessions model
            var oSessionsModel = new JSONModel([]);
            this.getView().setModel(oSessionsModel, "sessions");
            
            // Blind spots model
            var oBlindSpotsModel = new JSONModel([]);
            this.getView().setModel(oBlindSpotsModel, "blindSpots");
            
            // Results model
            var oResultsModel = new JSONModel([]);
            this.getView().setModel(oResultsModel, "results");
        },
        
        _loadAnalyticsData: function() {
            // Load real-time analytics from multi-language indexer
            jQuery.ajax({
                url: "/api/code-analysis/analytics",
                type: "GET",
                success: function(data) {
                    this.getView().getModel("analytics").setData(data);
                }.bind(this),
                error: function() {
                    console.error("Failed to load analytics data");
                }
            });
        },
        
        _loadIndexingSessions: function() {
            jQuery.ajax({
                url: "/api/code-analysis/sessions",
                type: "GET",
                success: function(data) {
                    this.getView().getModel("sessions").setData(data);
                }.bind(this),
                error: function() {
                    console.error("Failed to load indexing sessions");
                }
            });
        },
        
        _loadBlindSpots: function() {
            jQuery.ajax({
                url: "/api/code-analysis/blind-spots",
                type: "GET",
                success: function(data) {
                    this.getView().getModel("blindSpots").setData(data);
                }.bind(this),
                error: function() {
                    console.error("Failed to load blind spots data");
                }
            });
        },
        
        _loadRecentResults: function() {
            jQuery.ajax({
                url: "/api/code-analysis/results?limit=50",
                type: "GET",
                success: function(data) {
                    this.getView().getModel("results").setData(data);
                }.bind(this),
                error: function() {
                    console.error("Failed to load analysis results");
                }
            });
        },
        
        _setupAutoRefresh: function() {
            // Refresh active sessions every 30 seconds
            this._refreshTimer = setInterval(function() {
                this._loadIndexingSessions();
                this._loadAnalyticsData();
            }.bind(this), 30000);
        },
        
        onExit: function() {
            if (this._refreshTimer) {
                clearInterval(this._refreshTimer);
            }
        },
        
        onNavBack: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("launchpad");
        },
        
        onNewAnalysis: function() {
            if (!this._oNewAnalysisDialog) {
                this._oNewAnalysisDialog = new Dialog({
                    title: "New Code Analysis Project",
                    contentWidth: "500px",
                    content: [
                        new VBox({
                            items: [
                                new Label({ text: "Project Name:" }),
                                new Input({ id: "projectNameInput", placeholder: "Enter project name" }),
                                new Label({ text: "Project Path:" }),
                                new Input({ id: "projectPathInput", placeholder: "/path/to/project" }),
                                new Label({ text: "Description:" }),
                                new Input({ id: "projectDescInput", placeholder: "Optional description" })
                            ]
                        })
                    ],
                    beginButton: new Button({
                        text: "Create & Start",
                        type: "Emphasized",
                        press: this._onCreateProject.bind(this)
                    }),
                    endButton: new Button({
                        text: "Cancel",
                        press: function() {
                            this._oNewAnalysisDialog.close();
                        }.bind(this)
                    })
                });
                this.getView().addDependent(this._oNewAnalysisDialog);
            }
            this._oNewAnalysisDialog.open();
        },
        
        _onCreateProject: function() {
            var sName = sap.ui.getCore().byId("projectNameInput").getValue();
            var sPath = sap.ui.getCore().byId("projectPathInput").getValue();
            var sDesc = sap.ui.getCore().byId("projectDescInput").getValue();
            
            if (!sName || !sPath) {
                MessageToast.show("Please fill in required fields");
                return;
            }
            
            jQuery.ajax({
                url: "/api/code-analysis/projects",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    name: sName,
                    path: sPath,
                    description: sDesc
                }),
                success: function(data) {
                    MessageToast.show("Project created successfully");
                    this._oNewAnalysisDialog.close();
                    this._startIndexing(data.projectId);
                }.bind(this),
                error: function() {
                    MessageBox.error("Failed to create project");
                }
            });
        },
        
        onStartIndexing: function() {
            MessageBox.confirm(
                "Start a new indexing session for the selected project?",
                {
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._startIndexing();
                        }
                    }.bind(this)
                }
            );
        },
        
        _startIndexing: function(sProjectId) {
            jQuery.ajax({
                url: "/api/code-analysis/start-indexing",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    projectId: sProjectId || "default"
                }),
                success: function(data) {
                    MessageToast.show("Indexing session started");
                    this._loadIndexingSessions();
                }.bind(this),
                error: function() {
                    MessageBox.error("Failed to start indexing session");
                }
            });
        },
        
        onStopSession: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("sessions");
            var sSessionId = oContext.getProperty("id");
            
            MessageBox.confirm(
                "Stop the selected indexing session?",
                {
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._stopSession(sSessionId);
                        }
                    }.bind(this)
                }
            );
        },
        
        _stopSession: function(sSessionId) {
            jQuery.ajax({
                url: "/api/code-analysis/sessions/" + sSessionId + "/stop",
                type: "POST",
                success: function() {
                    MessageToast.show("Session stopped");
                    this._loadIndexingSessions();
                }.bind(this),
                error: function() {
                    MessageBox.error("Failed to stop session");
                }
            });
        },
        
        onViewSession: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("sessions");
            var sSessionId = oContext.getProperty("id");
            
            // Navigate to detailed session view
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.navTo("sessionDetails", { sessionId: sSessionId });
        },
        
        onSessionPress: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("sessions");
            var sSessionId = oContext.getProperty("id");
            this.onViewSession({ getSource: function() { return { getBindingContext: function() { return oContext; } }; } });
        },
        
        onExportResults: function() {
            jQuery.ajax({
                url: "/api/code-analysis/export",
                type: "GET",
                success: function(data) {
                    // Create and download CSV
                    var csvContent = this._generateCSV(data);
                    var blob = new Blob([csvContent], { type: 'text/csv' });
                    var url = window.URL.createObjectURL(blob);
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = 'code-analysis-results.csv';
                    a.click();
                    window.URL.revokeObjectURL(url);
                    MessageToast.show("Results exported successfully");
                }.bind(this),
                error: function() {
                    MessageBox.error("Failed to export results");
                }
            });
        },
        
        _generateCSV: function(data) {
            var csv = "Symbol,Type,File,Line,Confidence,Validated\n";
            data.forEach(function(item) {
                csv += [
                    item.symbolName,
                    item.symbolType,
                    item.fileName,
                    item.lineNumber,
                    item.confidence,
                    item.validated
                ].join(",") + "\n";
            });
            return csv;
        },
        
        onRefresh: function() {
            this._loadAnalyticsData();
            this._loadIndexingSessions();
            this._loadBlindSpots();
            this._loadRecentResults();
            MessageToast.show("Data refreshed");
        },
        
        // Formatters
        formatCoverageColor: function(fCoverage) {
            if (fCoverage >= 90) return "Good";
            if (fCoverage >= 70) return "Critical";
            return "Error";
        },
        
        formatSuccessState: function(fRate) {
            if (fRate >= 95) return "Success";
            if (fRate >= 85) return "Warning";
            return "Error";
        },
        
        formatSessionStatus: function(sStatus) {
            switch (sStatus) {
                case "RUNNING": return "Success";
                case "COMPLETED": return "Success";
                case "FAILED": return "Error";
                case "CANCELLED": return "Warning";
                default: return "None";
            }
        },
        
        formatProgressState: function(sStatus) {
            return sStatus === "RUNNING" ? "Success" : "None";
        },
        
        calculateProgress: function(iProcessed, iTotal) {
            return iTotal > 0 ? (iProcessed / iTotal) * 100 : 0;
        },
        
        formatProgress: function(iProcessed, iTotal) {
            return iProcessed + " / " + iTotal;
        },
        
        formatDuration: function(sStartTime) {
            if (!sStartTime) return "";
            var start = new Date(sStartTime);
            var now = new Date();
            var diff = Math.floor((now - start) / 1000 / 60); // minutes
            return diff + " min";
        },
        
        isRunning: function(sStatus) {
            return sStatus === "RUNNING";
        },
        
        formatSeverityState: function(sSeverity) {
            switch (sSeverity) {
                case "CRITICAL": return "Error";
                case "HIGH": return "Error";
                case "MEDIUM": return "Warning";
                case "LOW": return "Success";
                default: return "None";
            }
        },
        
        formatResolvedStatus: function(bResolved) {
            return bResolved ? "Resolved" : "Open";
        },
        
        formatResolvedState: function(bResolved) {
            return bResolved ? "Success" : "Error";
        },
        
        formatConfidencePercent: function(fConfidence) {
            return fConfidence * 100;
        },
        
        formatConfidenceState: function(fConfidence) {
            if (fConfidence >= 0.9) return "Success";
            if (fConfidence >= 0.7) return "Warning";
            return "Error";
        },
        
        formatValidatedIcon: function(bValidated) {
            return bValidated ? "sap-icon://accept" : "sap-icon://pending";
        },
        
        formatValidatedColor: function(bValidated) {
            return bValidated ? "Positive" : "Critical";
        }
    });
});
