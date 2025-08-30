"""
Technical Analysis MCP Tools
MCP server integration for Technical Analysis skills to enable inter-agent communication
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ...core.agents.specialized.technical_analysis import (
    TechnicalAnalysisAgent,
    create_technical_analysis_agent,
    get_all_ta_skills,
)

logger = logging.getLogger(__name__)


class TechnicalAnalysisMCPServer:
    """MCP Server for Technical Analysis Tools"""

    def __init__(self):
        self.server = Server("technical-analysis")
        self.ta_agent: Optional[TechnicalAnalysisAgent] = None
        self.skill_tools = get_all_ta_skills()

        # Register MCP tools
        self._register_mcp_tools()

    def _register_mcp_tools(self):
        """Register all TA tools as MCP tools"""

        # Comprehensive analysis tool
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available TA tools"""
            tools = [
                Tool(
                    name="analyze_market_comprehensive",
                    description="Perform comprehensive technical analysis on market data using all 8 TA skills",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "market_data": {
                                "type": "string",
                                "description": "JSON string containing OHLCV market data",
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol (e.g., BTC-USD)",
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["basic", "comprehensive", "quick", "dashboard"],
                                "default": "comprehensive",
                                "description": "Type of analysis to perform",
                            },
                        },
                        "required": ["market_data", "symbol"],
                    },
                ),
                Tool(
                    name="analyze_momentum_indicators",
                    description="Analyze momentum technical indicators (SMA, EMA, RSI)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "market_data": {
                                "type": "string",
                                "description": "JSON string containing OHLCV market data",
                            },
                            "symbol": {"type": "string", "description": "Trading symbol"},
                        },
                        "required": ["market_data", "symbol"],
                    },
                ),
                Tool(
                    name="analyze_support_resistance",
                    description="Detect support and resistance levels using pivot points, psychological levels, and Fibonacci",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "market_data": {
                                "type": "string",
                                "description": "JSON string containing OHLCV market data",
                            },
                            "symbol": {"type": "string", "description": "Trading symbol"},
                        },
                        "required": ["market_data", "symbol"],
                    },
                ),
                Tool(
                    name="detect_chart_patterns",
                    description="Detect chart patterns like triangles, head & shoulders, double tops/bottoms",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "market_data": {
                                "type": "string",
                                "description": "JSON string containing OHLCV market data",
                            },
                            "symbol": {"type": "string", "description": "Trading symbol"},
                        },
                        "required": ["market_data", "symbol"],
                    },
                ),
                Tool(
                    name="create_ta_dashboard",
                    description="Create comprehensive TA dashboard with visualizations and alerts",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "market_data": {
                                "type": "string",
                                "description": "JSON string containing OHLCV market data",
                            },
                            "symbol": {"type": "string", "description": "Trading symbol"},
                            "export_format": {
                                "type": "string",
                                "enum": ["json", "html", "csv"],
                                "default": "json",
                            },
                        },
                        "required": ["market_data", "symbol"],
                    },
                ),
            ]
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle MCP tool calls"""
            try:
                # Initialize TA agent if not exists
                if self.ta_agent is None:
                    self.ta_agent = create_technical_analysis_agent("mcp_ta_agent")

                # Parse market data
                market_data_json = arguments.get("market_data")
                symbol = arguments.get("symbol", "UNKNOWN")

                if not market_data_json:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"success": False, "error": "No market data provided"}),
                        )
                    ]

                # Convert JSON to DataFrame
                try:
                    market_data_dict = json.loads(market_data_json)
                    data = pd.DataFrame(market_data_dict)

                    # Ensure proper datetime index if timestamps provided
                    if "timestamp" in data.columns:
                        data["timestamp"] = pd.to_datetime(data["timestamp"])
                        data.set_index("timestamp", inplace=True)
                    elif "date" in data.columns:
                        data["date"] = pd.to_datetime(data["date"])
                        data.set_index("date", inplace=True)

                except Exception as e:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "success": False,
                                    "error": f"Failed to parse market data: {str(e)}",
                                }
                            ),
                        )
                    ]

                # Route to appropriate handler
                if name == "analyze_market_comprehensive":
                    result = await self._handle_comprehensive_analysis(data, arguments)
                elif name == "analyze_momentum_indicators":
                    result = await self._handle_momentum_indicators(data, symbol)
                elif name == "analyze_support_resistance":
                    result = await self._handle_support_resistance(data, symbol)
                elif name == "detect_chart_patterns":
                    result = await self._handle_chart_patterns(data, symbol)
                elif name == "create_ta_dashboard":
                    result = await self._handle_dashboard_creation(data, arguments)
                else:
                    result = {"success": False, "error": f"Unknown tool: {name}"}

                return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

            except Exception as e:
                logger.error(f"MCP tool call failed: {e}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": f"Tool execution failed: {str(e)}"}
                        ),
                    )
                ]

    async def _handle_comprehensive_analysis(
        self, data: pd.DataFrame, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle comprehensive market analysis"""
        analysis_type = arguments.get("analysis_type", "comprehensive")
        symbol = arguments.get("symbol", "UNKNOWN")

        result = await self.ta_agent.analyze_market_data(data=data, analysis_type=analysis_type)

        # Add symbol metadata
        if result.get("success"):
            result["analysis"]["metadata"]["symbol"] = symbol
            result["analysis"]["metadata"]["mcp_tool"] = "analyze_market_comprehensive"

        return result

    async def _handle_momentum_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Handle momentum indicators analysis using technical analysis agent.

        Args:
            data: OHLCV market data as pandas DataFrame
            symbol: Trading symbol identifier (e.g., 'BTC-USD')

        Returns:
            Dictionary containing analysis results with success status, indicators, and signals
        """
        try:
            # Execute momentum indicators analysis
            momentum_result = await self.ta_agent.execute_tool(
                "analyze_momentum_indicators", {"data": data}
            )

            if momentum_result.success:
                return {
                    "success": True,
                    "symbol": symbol,
                    "analysis_type": "momentum_indicators",
                    "indicators": momentum_result.result.get("indicators", {}),
                    "signals": momentum_result.result.get("signals", []),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {"success": False, "error": "Momentum indicators analysis failed"}

        except Exception as e:
            return {"success": False, "error": f"Momentum indicators analysis error: {str(e)}"}

    async def _handle_support_resistance(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Handle support and resistance level analysis using technical analysis agent.

        Args:
            data: OHLCV market data as pandas DataFrame
            symbol: Trading symbol identifier (e.g., 'BTC-USD')

        Returns:
            Dictionary containing analysis results with success status, levels, and signals
        """
        try:
            sr_result = await self.ta_agent.execute_tool(
                "analyze_support_resistance", {"data": data}
            )

            if sr_result.success:
                return {
                    "success": True,
                    "symbol": symbol,
                    "analysis_type": "support_resistance",
                    "levels": sr_result.result.get("levels", {}),
                    "signals": sr_result.result.get("signals", []),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {"success": False, "error": "Support/resistance analysis failed"}

        except Exception as e:
            return {"success": False, "error": f"Support/resistance analysis error: {str(e)}"}

    async def _handle_chart_patterns(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Handle chart pattern detection analysis using technical analysis agent.

        Args:
            data: OHLCV market data as pandas DataFrame
            symbol: Trading symbol identifier (e.g., 'BTC-USD')

        Returns:
            Dictionary containing analysis results with success status, patterns, and signals
        """
        try:
            pattern_result = await self.ta_agent.execute_tool(
                "analyze_chart_patterns", {"data": data}
            )

            if pattern_result.success:
                return {
                    "success": True,
                    "symbol": symbol,
                    "analysis_type": "chart_patterns",
                    "patterns": pattern_result.result.get("patterns", {}),
                    "signals": pattern_result.result.get("signals", []),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {"success": False, "error": "Chart pattern analysis failed"}

        except Exception as e:
            return {"success": False, "error": f"Chart pattern analysis error: {str(e)}"}

    async def _handle_dashboard_creation(
        self, data: pd.DataFrame, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle technical analysis dashboard creation with comprehensive visualizations.

        Args:
            data: OHLCV market data as pandas DataFrame
            arguments: Dictionary containing symbol and export_format parameters

        Returns:
            Dictionary containing dashboard data, content type, and export format
        """
        try:
            symbol = arguments.get("symbol", "UNKNOWN")
            export_format = arguments.get("export_format", "json")

            # Run comprehensive analysis first
            analysis_result = await self.ta_agent.analyze_market_data(
                data=data, analysis_type="dashboard"
            )

            if analysis_result.get("success"):
                # Export dashboard data
                export_result = await self.ta_agent.execute_tool(
                    "export_dashboard_data",
                    {
                        "data": data,
                        "all_analysis": analysis_result["analysis"],
                        "export_format": export_format,
                    },
                )

                if export_result.success:
                    return {
                        "success": True,
                        "symbol": symbol,
                        "analysis_type": "dashboard",
                        "dashboard_data": export_result.result.get("exported_data"),
                        "content_type": export_result.result.get("content_type"),
                        "format": export_format,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {"success": False, "error": "Dashboard export failed"}
            else:
                return analysis_result

        except Exception as e:
            return {"success": False, "error": f"Dashboard creation error: {str(e)}"}


async def run_ta_mcp_server():
    """Run the Technical Analysis MCP Server"""
    ta_server = TechnicalAnalysisMCPServer()

    async with stdio_server() as (read_stream, write_stream):
        await ta_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="technical-analysis",
                server_version="1.0.0",
                capabilities=ta_server.server.get_capabilities(
                    notification_options=None, experimental_capabilities=None
                ),
            ),
        )


def create_ta_mcp_tools_config() -> Dict[str, Any]:
    """Create MCP tools configuration for Technical Analysis"""
    return {
        "technical_analysis": {
            "command": "python",
            "args": ["-m", "cryptotrading.infrastructure.mcp.technical_analysis_mcp_tools"],
            "description": "Technical Analysis tools for crypto trading",
            "capabilities": [
                "comprehensive_market_analysis",
                "momentum_indicators",
                "support_resistance_detection",
                "chart_pattern_recognition",
                "trading_signal_generation",
                "dashboard_creation",
            ],
            "tools": [
                "analyze_market_comprehensive",
                "analyze_momentum_indicators",
                "analyze_support_resistance",
                "detect_chart_patterns",
                "create_ta_dashboard",
            ],
        }
    }


if __name__ == "__main__":
    asyncio.run(run_ta_mcp_server())
