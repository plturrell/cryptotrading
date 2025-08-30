sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/routing/History",
    "sap/ui/export/Spreadsheet"
], function (Controller, JSONModel, MessageToast, MessageBox, History, Spreadsheet) {
    "use strict";

    return Controller.extend("com.rex.cryptotrading.controller.TechnicalAnalysis", {

        onInit: function () {
            // Initialize OData model for Trading Service
            this._oTradingModel = this.getOwnerComponent().getModel("trading");
            
            this._initializeModel();
            this._setupRouting();
            this._loadInitialData();
            this._startDataRefresh();
        },

        onExit: function () {
            if (this._refreshTimer) {
                clearInterval(this._refreshTimer);
            }
        },

        _initializeModel: function () {
            const oModel = new JSONModel({
                selectedSymbol: "BTC-USD",
                selectedTimeframe: "1d",
                availableSymbols: [
                    { symbol: "BTC-USD", displayName: "Bitcoin (BTC)" },
                    { symbol: "ETH-USD", displayName: "Ethereum (ETH)" },
                    { symbol: "ADA-USD", displayName: "Cardano (ADA)" },
                    { symbol: "SOL-USD", displayName: "Solana (SOL)" },
                    { symbol: "DOT-USD", displayName: "Polkadot (DOT)" }
                ],
                availableTimeframes: [
                    { value: "1h", label: "1H" },
                    { value: "4h", label: "4H" },
                    { value: "1d", label: "1D" },
                    { value: "1w", label: "1W" }
                ],
                currentPrice: 0,
                priceIndicator: "None",
                priceColor: "Neutral",
                indicators: {
                    RSI: 0,
                    RSI_indicator: "None",
                    RSI_color: "Neutral"
                },
                analysis: {
                    confidence_score: 0,
                    sentiment_indicator: "None",
                    sentiment_color: "Neutral"
                },
                chartData: {
                    price: [],
                    volume: [],
                    rsi: [],
                    macd: [],
                    bollinger: []
                },
                chartConfig: {
                    priceChart: {
                        title: { text: "Price Chart with Moving Averages" },
                        plotArea: {
                            dataLabel: { visible: false },
                            colorPalette: ["#5cbae6", "#b6d957", "#fac364"]
                        },
                        categoryAxis: {
                            title: { text: "Date" }
                        },
                        valueAxis: {
                            title: { text: "Price (USD)" }
                        }
                    },
                    volumeChart: {
                        title: { text: "Volume Analysis" },
                        plotArea: {
                            dataLabel: { visible: false },
                            colorPalette: ["#b6d957", "#fac364"]
                        }
                    },
                    rsiChart: {
                        title: { text: "RSI (14)" },
                        plotArea: {
                            dataLabel: { visible: false },
                            colorPalette: ["#5cbae6"]
                        },
                        categoryAxis: {
                            title: { text: "Date" }
                        },
                        valueAxis: {
                            title: { text: "RSI" },
                            scale: { min: 0, max: 100 }
                        }
                    },
                    macdChart: {
                        title: { text: "MACD" },
                        plotArea: {
                            dataLabel: { visible: false },
                            colorPalette: ["#5cbae6", "#b6d957", "#fac364"]
                        }
                    },
                    bollingerChart: {
                        title: { text: "Bollinger Bands" },
                        plotArea: {
                            dataLabel: { visible: false },
                            colorPalette: ["#5cbae6", "#b6d957", "#fac364", "#ff6f6f"]
                        }
                    }
                },
                signals: [],
                patterns: [],
                supportResistance: [],
                aiInsights: {
                    market_summary: "Loading AI analysis...",
                    key_signals: [],
                    risk_assessment: "Analyzing market conditions...",
                    risk_type: "Information",
                    market_regime: "Unknown",
                    regime_state: "None",
                    confidence: 0
                },
                performance: {
                    operations: []
                },
                isLoading: false,
                lastUpdated: new Date()
            });
            
            this.getView().setModel(oModel, "ta");
        },

        _setupRouting: function () {
            const oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.getRoute("technicalAnalysis").attachPatternMatched(this._onObjectMatched, this);
        },

        _onObjectMatched: function (oEvent) {
            const oArguments = oEvent.getParameter("arguments");
            if (oArguments.symbol) {
                this.getModel("ta").setProperty("/selectedSymbol", oArguments.symbol);
                this._loadTechnicalAnalysis();
            }
        },

        _loadInitialData: function () {
            this._loadTechnicalAnalysis();
        },

        _startDataRefresh: function () {
            // Refresh data every 30 seconds
            this._refreshTimer = setInterval(() => {
                if (!this.getModel("ta").getProperty("/isLoading")) {
                    this._loadTechnicalAnalysis();
                }
            }, 30000);
        },

        _loadTechnicalAnalysis: function () {
            const oModel = this.getModel("ta");
            const sSymbol = oModel.getProperty("/selectedSymbol");
            const sTimeframe = oModel.getProperty("/selectedTimeframe");
            
            oModel.setProperty("/isLoading", true);
            
            // Call backend Technical Analysis API
            this._callTechnicalAnalysisAPI(sSymbol, sTimeframe)
                .then((oData) => {
                    this._updateModelWithAnalysis(oData);
                    oModel.setProperty("/lastUpdated", new Date());
                    MessageToast.show("Technical analysis updated successfully");
                })
                .catch((oError) => {
                    console.error("Failed to load technical analysis:", oError);
                    MessageToast.show("Failed to load technical analysis data");
                })
                .finally(() => {
                    oModel.setProperty("/isLoading", false);
                });
        },

        _callTechnicalAnalysisAPI: function (sSymbol, sTimeframe) {
            return new Promise((resolve, reject) => {
                // Call the backend Technical Analysis STRAND agent
                fetch(`/api/technical-analysis/comprehensive`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        symbol: sSymbol,
                        timeframe: sTimeframe,
                        analysis_type: "comprehensive",
                        include_ai_insights: true,
                        include_patterns: true,
                        include_performance: true
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    resolve(data);
                })
                .catch(error => {
                    reject(error);
                });
            });
        },

        _updateModelWithAnalysis: function (oData) {
            const oModel = this.getModel("ta");
            
            // Update current price and indicators
            if (oData.current_price) {
                oModel.setProperty("/currentPrice", oData.current_price.toFixed(2));
                oModel.setProperty("/priceIndicator", this._getPriceIndicator(oData.price_change_24h));
                oModel.setProperty("/priceColor", this._getPriceColor(oData.price_change_24h));
            }
            
            // Update technical indicators
            if (oData.indicators) {
                const indicators = oData.indicators;
                oModel.setProperty("/indicators/RSI", indicators.RSI ? indicators.RSI.toFixed(1) : 0);
                oModel.setProperty("/indicators/RSI_indicator", this._getRSIIndicator(indicators.RSI));
                oModel.setProperty("/indicators/RSI_color", this._getRSIColor(indicators.RSI));
            }
            
            // Update analysis summary
            if (oData.analysis_summary) {
                const analysis = oData.analysis_summary;
                oModel.setProperty("/analysis/confidence_score", analysis.confidence_score ? analysis.confidence_score.toFixed(1) : 0);
                oModel.setProperty("/analysis/sentiment_indicator", this._getSentimentIndicator(analysis.overall_sentiment));
                oModel.setProperty("/analysis/sentiment_color", this._getSentimentColor(analysis.overall_sentiment));
            }
            
            // Update chart data
            if (oData.chart_data) {
                oModel.setProperty("/chartData", oData.chart_data);
            }
            
            // Update signals
            if (oData.signals) {
                const formattedSignals = oData.signals.map(signal => ({
                    ...signal,
                    signalState: this._getSignalState(signal.signal),
                    strengthPercent: signal.strength * 100,
                    strengthState: this._getStrengthState(signal.strength),
                    hasInsight: !!signal.ai_insight
                }));
                oModel.setProperty("/signals", formattedSignals);
            }
            
            // Update patterns
            if (oData.patterns) {
                const formattedPatterns = oData.patterns.map(pattern => ({
                    ...pattern,
                    reliabilityState: this._getReliabilityState(pattern.reliability)
                }));
                oModel.setProperty("/patterns", formattedPatterns);
            }
            
            // Update support/resistance levels
            if (oData.support_resistance) {
                const formattedLevels = oData.support_resistance.map(level => ({
                    ...level,
                    typeState: level.type === "Support" ? "Success" : "Error",
                    strengthPercent: level.strength * 100
                }));
                oModel.setProperty("/supportResistance", formattedLevels);
            }
            
            // Update AI insights
            if (oData.ai_insights) {
                const insights = oData.ai_insights;
                oModel.setProperty("/aiInsights", {
                    market_summary: insights.market_summary || "No AI analysis available",
                    key_signals: insights.key_signals || [],
                    risk_assessment: insights.risk_assessment || "Risk analysis unavailable",
                    risk_type: this._getRiskType(insights.risk_level),
                    market_regime: insights.market_regime || "Unknown",
                    regime_state: this._getRegimeState(insights.market_regime),
                    confidence: insights.confidence || 0
                });
            }
            
            // Update performance metrics
            if (oData.performance_metrics) {
                const formattedMetrics = oData.performance_metrics.map(metric => ({
                    ...metric,
                    statusState: metric.status === "Success" ? "Success" : "Error"
                }));
                oModel.setProperty("/performance/operations", formattedMetrics);
            }
        },

        // Event Handlers
        onSymbolChange: function () {
            this._loadTechnicalAnalysis();
        },

        onTimeframeChange: function () {
            this._loadTechnicalAnalysis();
        },

        onRefresh: function () {
            this._loadTechnicalAnalysis();
        },

        onNavBack: function () {
            const oHistory = History.getInstance();
            const sPreviousHash = oHistory.getPreviousHash();
            
            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                const oRouter = sap.ui.core.UIComponent.getRouterFor(this);
                oRouter.navTo("launchpad", {}, true);
            }
        },

        onTabSelect: function (oEvent) {
            const sSelectedKey = oEvent.getParameter("selectedKey");
            // Handle tab-specific loading if needed
            console.log("Selected tab:", sSelectedKey);
        },

        onChartSelect: function (oEvent) {
            const aSelectedData = oEvent.getParameter("data");
            if (aSelectedData && aSelectedData.length > 0) {
                const oSelectedPoint = aSelectedData[0];
                MessageToast.show(`Selected: ${oSelectedPoint.data.Date} - $${oSelectedPoint.data.Price}`);
            }
        },

        onSignalPress: function (oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext("ta");
            const oSignal = oBindingContext.getObject();
            
            MessageBox.information(
                `Signal: ${oSignal.signal}\nStrength: ${oSignal.strength}\nValue: ${oSignal.value}`,
                {
                    title: oSignal.indicator
                }
            );
        },

        onInsightPress: function (oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext("ta");
            const oSignal = oBindingContext.getObject();
            
            if (oSignal.ai_insight) {
                MessageBox.information(oSignal.ai_insight, {
                    title: `AI Insight: ${oSignal.indicator}`
                });
            }
        },

        onPatternPress: function (oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext("ta");
            const oPattern = oBindingContext.getObject();
            
            MessageBox.information(
                `${oPattern.description}\n\nConfidence: ${oPattern.confidence}%\nReliability: ${oPattern.reliability}`,
                {
                    title: oPattern.pattern_name
                }
            );
        },

        onExportAnalysis: function () {
            const oModel = this.getModel("ta");
            const aSignals = oModel.getProperty("/signals");
            const aPatterns = oModel.getProperty("/patterns");
            
            const aExportData = [
                ...aSignals.map(signal => ({
                    Type: "Signal",
                    Name: signal.indicator,
                    Value: signal.signal,
                    Strength: signal.strength,
                    Details: signal.value
                })),
                ...aPatterns.map(pattern => ({
                    Type: "Pattern",
                    Name: pattern.pattern_name,
                    Value: pattern.reliability,
                    Strength: pattern.confidence / 100,
                    Details: pattern.description
                }))
            ];
            
            const oSpreadsheet = new Spreadsheet({
                workbook: {
                    columns: [
                        { label: "Type", property: "Type" },
                        { label: "Name", property: "Name" },
                        { label: "Value", property: "Value" },
                        { label: "Strength", property: "Strength" },
                        { label: "Details", property: "Details" }
                    ]
                },
                dataSource: aExportData,
                fileName: `TA_Analysis_${oModel.getProperty("/selectedSymbol")}_${new Date().toISOString().split('T')[0]}.xlsx`
            });
            
            oSpreadsheet.build();
            MessageToast.show("Analysis exported successfully");
        },

        onScheduleReport: function () {
            MessageBox.information("Report scheduling feature will be available soon.");
        },

        onConfigureAlerts: function () {
            MessageBox.information("Alert configuration feature will be available soon.");
        },

        // Helper Methods
        _getPriceIndicator: function (fChange) {
            if (fChange > 0) return "Up";
            if (fChange < 0) return "Down";
            return "None";
        },

        _getPriceColor: function (fChange) {
            if (fChange > 0) return "Good";
            if (fChange < 0) return "Error";
            return "Neutral";
        },

        _getRSIIndicator: function (fRSI) {
            if (fRSI > 70) return "Down";
            if (fRSI < 30) return "Up";
            return "None";
        },

        _getRSIColor: function (fRSI) {
            if (fRSI > 70) return "Error";
            if (fRSI < 30) return "Good";
            return "Neutral";
        },

        _getSentimentIndicator: function (sSentiment) {
            if (sSentiment === "bullish") return "Up";
            if (sSentiment === "bearish") return "Down";
            return "None";
        },

        _getSentimentColor: function (sSentiment) {
            if (sSentiment === "bullish") return "Good";
            if (sSentiment === "bearish") return "Error";
            return "Neutral";
        },

        _getSignalState: function (sSignal) {
            if (sSignal === "BUY" || sSignal === "STRONG_BUY") return "Success";
            if (sSignal === "SELL" || sSignal === "STRONG_SELL") return "Error";
            return "Warning";
        },

        _getStrengthState: function (fStrength) {
            if (fStrength > 0.7) return "Success";
            if (fStrength < 0.3) return "Error";
            return "Warning";
        },

        _getReliabilityState: function (sReliability) {
            if (sReliability === "High") return "Success";
            if (sReliability === "Low") return "Error";
            return "Warning";
        },

        _getRiskType: function (sRiskLevel) {
            if (sRiskLevel === "High") return "Error";
            if (sRiskLevel === "Medium") return "Warning";
            return "Success";
        },

        _getRegimeState: function (sRegime) {
            if (sRegime === "Bull Market") return "Success";
            if (sRegime === "Bear Market") return "Error";
            return "Warning";
        }
    });
});
