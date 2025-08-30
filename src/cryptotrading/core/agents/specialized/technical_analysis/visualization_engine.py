"""
Technical Analysis Visualization Engine
Provides interactive charts and visualizations for TA indicators and patterns
"""

import base64
import json
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Visualization imports
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Polygon, Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Install with: pip install matplotlib seaborn")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly")

logger = logging.getLogger(__name__)


class TechnicalAnalysisVisualizer:
    """Main visualization engine for technical analysis"""

    def __init__(self, theme: str = "dark", figsize: Tuple[int, int] = (15, 10)):
        self.theme = theme
        self.figsize = figsize
        self.colors = self._get_color_scheme()

        if MATPLOTLIB_AVAILABLE:
            plt.style.use("dark_background" if theme == "dark" else "default")
            sns.set_palette("husl")

    def _get_color_scheme(self) -> Dict[str, str]:
        """Get color scheme based on theme"""
        if self.theme == "dark":
            return {
                "background": "#1e1e1e",
                "grid": "#333333",
                "text": "#ffffff",
                "bullish": "#00ff88",
                "bearish": "#ff4444",
                "neutral": "#888888",
                "volume": "#4488ff",
                "indicators": ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"],
            }
        else:
            return {
                "background": "#ffffff",
                "grid": "#cccccc",
                "text": "#000000",
                "bullish": "#00aa44",
                "bearish": "#cc3333",
                "neutral": "#666666",
                "volume": "#3366cc",
                "indicators": ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
            }

    def create_candlestick_chart(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, Any] = None,
        patterns: List[Dict[str, Any]] = None,
        title: str = "Price Chart",
    ) -> Dict[str, Any]:
        """
        Create interactive candlestick chart with indicators

        Args:
            data: OHLCV DataFrame
            indicators: Dictionary of indicator data
            patterns: List of detected patterns
            title: Chart title

        Returns:
            Chart data and metadata
        """
        if not PLOTLY_AVAILABLE:
            return self._fallback_chart_data(data, title)

        try:
            # Create subplots
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=("Price & Indicators", "Volume", "Oscillators"),
            )

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name="Price",
                    increasing_line_color=self.colors["bullish"],
                    decreasing_line_color=self.colors["bearish"],
                ),
                row=1,
                col=1,
            )

            # Add volume bars
            colors = [
                self.colors["bullish"] if close >= open else self.colors["bearish"]
                for close, open in zip(data["close"], data["open"])
            ]

            fig.add_trace(
                go.Bar(
                    x=data.index, y=data["volume"], name="Volume", marker_color=colors, opacity=0.7
                ),
                row=2,
                col=1,
            )

            # Add indicators if provided
            if indicators:
                self._add_indicators_to_chart(fig, data, indicators)

            # Add patterns if provided
            if patterns:
                self._add_patterns_to_chart(fig, data, patterns)

            # Update layout
            fig.update_layout(
                title=title,
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
            )

            # Convert to JSON for storage/transmission
            chart_json = fig.to_json()

            return {
                "success": True,
                "chart_type": "candlestick",
                "chart_data": chart_json,
                "interactive": True,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Candlestick chart creation failed: {e}")
            return self._fallback_chart_data(data, title)

    def create_indicator_chart(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        indicator_data: Dict[str, Any],
        title: str = None,
    ) -> Dict[str, Any]:
        """
        Create specialized chart for specific indicators

        Args:
            data: OHLCV DataFrame
            indicator_name: Name of the indicator
            indicator_data: Indicator calculation results
            title: Chart title

        Returns:
            Indicator chart data
        """
        if not PLOTLY_AVAILABLE:
            return self._fallback_chart_data(data, title or indicator_name)

        try:
            title = title or f"{indicator_name} Analysis"

            if indicator_name.upper() == "RSI":
                return self._create_rsi_chart(data, indicator_data, title)
            elif indicator_name.upper() == "MACD":
                return self._create_macd_chart(data, indicator_data, title)
            elif "BOLLINGER" in indicator_name.upper():
                return self._create_bollinger_chart(data, indicator_data, title)
            else:
                return self._create_generic_indicator_chart(data, indicator_data, title)

        except Exception as e:
            logger.error(f"Indicator chart creation failed: {e}")
            return self._fallback_chart_data(data, title or indicator_name)

    def create_pattern_visualization(
        self, data: pd.DataFrame, pattern_data: Dict[str, Any], title: str = "Pattern Analysis"
    ) -> Dict[str, Any]:
        """
        Create visualization for detected patterns

        Args:
            data: OHLCV DataFrame
            pattern_data: Pattern detection results
            title: Chart title

        Returns:
            Pattern visualization data
        """
        if not PLOTLY_AVAILABLE:
            return self._fallback_chart_data(data, title)

        try:
            fig = go.Figure()

            # Add candlestick base
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name="Price",
                )
            )

            # Add pattern annotations
            if "patterns" in pattern_data:
                for pattern in pattern_data["patterns"]:
                    self._add_pattern_annotation(fig, pattern)

            fig.update_layout(
                title=title,
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                height=600,
            )

            return {
                "success": True,
                "chart_type": "pattern",
                "chart_data": fig.to_json(),
                "patterns_detected": len(pattern_data.get("patterns", [])),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Pattern visualization failed: {e}")
            return self._fallback_chart_data(data, title)

    def create_dashboard_summary(
        self, analysis_results: Dict[str, Any], market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create comprehensive dashboard visualization

        Args:
            analysis_results: Complete TA analysis results
            market_data: OHLCV market data

        Returns:
            Dashboard visualization data
        """
        try:
            dashboard_components = []

            # Main price chart with all indicators
            main_chart = self.create_candlestick_chart(
                market_data,
                analysis_results.get("indicators", {}),
                analysis_results.get("patterns", []),
                "Technical Analysis Dashboard",
            )
            dashboard_components.append(main_chart)

            # Signal strength heatmap
            signal_heatmap = self._create_signal_heatmap(analysis_results)
            dashboard_components.append(signal_heatmap)

            # Performance metrics
            performance_chart = self._create_performance_metrics(analysis_results)
            dashboard_components.append(performance_chart)

            return {
                "success": True,
                "dashboard_type": "comprehensive",
                "components": dashboard_components,
                "component_count": len(dashboard_components),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return {"success": False, "error": str(e)}

    def _add_indicators_to_chart(self, fig, data: pd.DataFrame, indicators: Dict[str, Any]):
        """Add indicators to existing chart"""
        color_idx = 0

        for indicator_name, values in indicators.items():
            if isinstance(values, list) and len(values) == len(data):
                color = self.colors["indicators"][color_idx % len(self.colors["indicators"])]

                if "RSI" in indicator_name:
                    # Add RSI to oscillator subplot
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=values,
                            name=indicator_name,
                            line=dict(color=color),
                            yaxis="y3",
                        ),
                        row=3,
                        col=1,
                    )
                    # Add RSI reference lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

                elif any(x in indicator_name for x in ["SMA", "EMA", "VWAP"]):
                    # Add moving averages to price chart
                    fig.add_trace(
                        go.Scatter(
                            x=data.index, y=values, name=indicator_name, line=dict(color=color)
                        ),
                        row=1,
                        col=1,
                    )

                color_idx += 1

    def _add_patterns_to_chart(self, fig, data: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """Add pattern annotations to chart"""
        for pattern in patterns:
            if pattern.get("type") == "support_resistance":
                # Add horizontal lines for S/R levels
                level = pattern.get("level", 0)
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="yellow",
                    annotation_text=f"S/R: {level:.2f}",
                    row=1,
                    col=1,
                )

    def _create_rsi_chart(self, data: pd.DataFrame, rsi_data: Dict[str, Any], title: str):
        """Create specialized RSI chart"""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "RSI"),
        )

        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # RSI chart
        rsi_values = rsi_data.get("indicators", {}).get("RSI", [])
        if rsi_values:
            fig.add_trace(
                go.Scatter(x=data.index, y=rsi_values, name="RSI", line=dict(color="purple")),
                row=2,
                col=1,
            )

            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(title=title, height=600)

        return {
            "success": True,
            "chart_type": "rsi",
            "chart_data": fig.to_json(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    def _create_macd_chart(self, data: pd.DataFrame, macd_data: Dict[str, Any], title: str):
        """Create specialized MACD chart"""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "MACD"),
        )

        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # MACD components
        indicators = macd_data.get("indicators", {})
        if "MACD_line" in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=indicators["MACD_line"], name="MACD", line=dict(color="blue")
                ),
                row=2,
                col=1,
            )

        if "MACD_signal" in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=indicators["MACD_signal"], name="Signal", line=dict(color="red")
                ),
                row=2,
                col=1,
            )

        if "MACD_histogram" in indicators:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=indicators["MACD_histogram"],
                    name="Histogram",
                    marker_color="green",
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(title=title, height=600)

        return {
            "success": True,
            "chart_type": "macd",
            "chart_data": fig.to_json(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    def _create_bollinger_chart(self, data: pd.DataFrame, bb_data: Dict[str, Any], title: str):
        """Create Bollinger Bands chart"""
        fig = go.Figure()

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
            )
        )

        # Bollinger Bands
        indicators = bb_data.get("indicators", {})
        if all(band in indicators for band in ["BB_upper", "BB_middle", "BB_lower"]):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators["BB_upper"],
                    name="Upper Band",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators["BB_middle"],
                    name="Middle Band",
                    line=dict(color="blue"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators["BB_lower"],
                    name="Lower Band",
                    line=dict(color="green", dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(0,100,80,0.1)",
                )
            )

        fig.update_layout(title=title, height=600)

        return {
            "success": True,
            "chart_type": "bollinger",
            "chart_data": fig.to_json(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    def _create_generic_indicator_chart(
        self, data: pd.DataFrame, indicator_data: Dict[str, Any], title: str
    ):
        """Create generic indicator chart"""
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3]
        )

        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Indicators
        indicators = indicator_data.get("indicators", {})
        for name, values in indicators.items():
            if isinstance(values, list) and len(values) == len(data):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=values,
                        name=name,
                        line=dict(color=self.colors["indicators"][0]),
                    ),
                    row=2,
                    col=1,
                )

        fig.update_layout(title=title, height=600)

        return {
            "success": True,
            "chart_type": "generic",
            "chart_data": fig.to_json(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    def _create_signal_heatmap(self, analysis_results: Dict[str, Any]):
        """Create signal strength heatmap"""
        try:
            signals = analysis_results.get("signals", [])
            if not signals:
                return {"success": False, "error": "No signals to visualize"}

            # Process signals into heatmap data
            signal_matrix = {}
            for signal in signals:
                indicator = signal.get("indicator", "Unknown")
                strength = signal.get("strength", "medium")
                signal_type = signal.get("signal", "neutral")

                if indicator not in signal_matrix:
                    signal_matrix[indicator] = {"buy": 0, "sell": 0, "neutral": 0}

                signal_matrix[indicator][signal_type] += 1

            return {
                "success": True,
                "chart_type": "heatmap",
                "data": signal_matrix,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Signal heatmap creation failed: {e}")
            return {"success": False, "error": str(e)}

    def _create_performance_metrics(self, analysis_results: Dict[str, Any]):
        """Create performance metrics visualization"""
        try:
            metrics = {
                "total_indicators": len(analysis_results.get("indicators", {})),
                "total_signals": len(analysis_results.get("signals", [])),
                "buy_signals": len(
                    [s for s in analysis_results.get("signals", []) if s.get("signal") == "buy"]
                ),
                "sell_signals": len(
                    [s for s in analysis_results.get("signals", []) if s.get("signal") == "sell"]
                ),
                "confidence_score": analysis_results.get("analysis", {}).get(
                    "confidence_score", 0.5
                ),
            }

            return {
                "success": True,
                "chart_type": "metrics",
                "data": metrics,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Performance metrics creation failed: {e}")
            return {"success": False, "error": str(e)}

    def _add_pattern_annotation(self, fig, pattern: Dict[str, Any]):
        """Add pattern annotation to chart"""
        pattern_type = pattern.get("type", "unknown")

        if pattern_type == "triangle":
            # Add triangle pattern lines
            points = pattern.get("points", [])
            if len(points) >= 3:
                fig.add_shape(
                    type="line",
                    x0=points[0]["x"],
                    y0=points[0]["y"],
                    x1=points[1]["x"],
                    y1=points[1]["y"],
                    line=dict(color="yellow", width=2, dash="dash"),
                )

    def _fallback_chart_data(self, data: pd.DataFrame, title: str):
        """Fallback chart data when visualization libraries unavailable"""
        return {
            "success": False,
            "error": "Visualization libraries not available",
            "fallback_data": {
                "title": title,
                "data_points": len(data),
                "date_range": f"{data.index[0]} to {data.index[-1]}",
                "price_range": f"{data['low'].min():.2f} - {data['high'].max():.2f}",
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }


# Global visualizer instance
visualizer = TechnicalAnalysisVisualizer()


def create_candlestick_chart(
    data: pd.DataFrame,
    indicators: Dict[str, Any] = None,
    patterns: List[Dict[str, Any]] = None,
    title: str = "Price Chart",
) -> Dict[str, Any]:
    """
    STRAND Tool: Create interactive candlestick chart

    Args:
        data: OHLCV DataFrame
        indicators: Dictionary of indicator data
        patterns: List of detected patterns
        title: Chart title

    Returns:
        Chart visualization data
    """
    return visualizer.create_candlestick_chart(data, indicators, patterns, title)


def create_indicator_chart(
    data: pd.DataFrame, indicator_name: str, indicator_data: Dict[str, Any], title: str = None
) -> Dict[str, Any]:
    """
    STRAND Tool: Create specialized indicator chart

    Args:
        data: OHLCV DataFrame
        indicator_name: Name of the indicator
        indicator_data: Indicator calculation results
        title: Chart title

    Returns:
        Indicator chart data
    """
    return visualizer.create_indicator_chart(data, indicator_name, indicator_data, title)


def create_pattern_visualization(
    data: pd.DataFrame, pattern_data: Dict[str, Any], title: str = "Pattern Analysis"
) -> Dict[str, Any]:
    """
    STRAND Tool: Create pattern visualization

    Args:
        data: OHLCV DataFrame
        pattern_data: Pattern detection results
        title: Chart title

    Returns:
        Pattern visualization data
    """
    return visualizer.create_pattern_visualization(data, pattern_data, title)


def create_dashboard_summary(
    analysis_results: Dict[str, Any], market_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    STRAND Tool: Create comprehensive dashboard

    Args:
        analysis_results: Complete TA analysis results
        market_data: OHLCV market data

    Returns:
        Dashboard visualization data
    """
    return visualizer.create_dashboard_summary(analysis_results, market_data)


def create_visualization_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for visualization

    Returns:
        List of tool specifications for STRAND framework
    """
    return [
        {
            "name": "create_candlestick_chart",
            "function": create_candlestick_chart,
            "description": "Create interactive candlestick chart with indicators and patterns",
            "parameters": {
                "data": "OHLCV DataFrame",
                "indicators": "Dictionary of indicator data",
                "patterns": "List of detected patterns",
                "title": "Chart title",
            },
            "category": "visualization",
            "skill": "charting",
        },
        {
            "name": "create_indicator_chart",
            "function": create_indicator_chart,
            "description": "Create specialized chart for specific indicators",
            "parameters": {
                "data": "OHLCV DataFrame",
                "indicator_name": "Name of the indicator",
                "indicator_data": "Indicator calculation results",
                "title": "Chart title",
            },
            "category": "visualization",
            "skill": "charting",
        },
        {
            "name": "create_pattern_visualization",
            "function": create_pattern_visualization,
            "description": "Create visualization for detected patterns",
            "parameters": {
                "data": "OHLCV DataFrame",
                "pattern_data": "Pattern detection results",
                "title": "Chart title",
            },
            "category": "visualization",
            "skill": "charting",
        },
        {
            "name": "create_dashboard_summary",
            "function": create_dashboard_summary,
            "description": "Create comprehensive dashboard visualization",
            "parameters": {
                "analysis_results": "Complete TA analysis results",
                "market_data": "OHLCV market data",
            },
            "category": "visualization",
            "skill": "dashboard",
        },
    ]
