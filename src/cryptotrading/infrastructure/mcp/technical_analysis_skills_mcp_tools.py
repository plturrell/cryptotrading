"""
Technical Analysis Skills MCP Tools - All TA skill calculations
"""
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import pandas as pd

from ...core.agents.specialized.technical_analysis.skill_1_momentum_indicators import MomentumIndicatorsSkill
from ...core.agents.specialized.technical_analysis.skill_2_momentum_volatility import MomentumVolatilitySkill
from ...core.agents.specialized.technical_analysis.skill_3_volume_analysis import VolumeAnalysisSkill
from ...core.agents.specialized.technical_analysis.skill_4_support_resistance import SupportResistanceSkill
from ...core.agents.specialized.technical_analysis.skill_6_harmonic_patterns import AdvancedPatternsSkill
from ...core.agents.specialized.technical_analysis.skill_7_comprehensive_system import ComprehensiveSystemSkill

logger = logging.getLogger(__name__)


class TechnicalAnalysisSkillsMCPTools:
    """MCP tools for all technical analysis skill calculations"""
    
    def __init__(self):
        self.momentum_indicators = MomentumIndicatorsSkill()
        self.momentum_volatility = MomentumVolatilitySkill()
        self.volume_analysis = VolumeAnalysisSkill()
        self.support_resistance = SupportResistanceSkill()
        self.harmonic_patterns = AdvancedPatternsSkill()
        self.comprehensive_system = ComprehensiveSystemSkill()
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions for TA skills"""
        return [
            {
                "name": "calculate_momentum_indicators",
                "description": "Calculate momentum technical indicators (SMA, EMA, RSI, MACD)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "market_data": {"type": "string", "description": "JSON string containing OHLCV data"},
                        "indicators": {"type": "array", "items": {"type": "string"}, "description": "List of indicators to calculate"},
                        "periods": {"type": "object", "description": "Period settings for indicators"}
                    },
                    "required": ["market_data"]
                }
            },
            {
                "name": "calculate_momentum_volatility",
                "description": "Calculate momentum and volatility indicators",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "market_data": {"type": "string", "description": "JSON string containing OHLCV data"},
                        "period": {"type": "integer", "default": 14, "description": "Calculation period"},
                        "include_bollinger": {"type": "boolean", "default": True}
                    },
                    "required": ["market_data"]
                }
            },
            {
                "name": "analyze_volume_patterns",
                "description": "Analyze volume patterns and trends",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "market_data": {"type": "string", "description": "JSON string containing OHLCV data"},
                        "lookback_period": {"type": "integer", "default": 20}
                    },
                    "required": ["market_data"]
                }
            },
            {
                "name": "identify_support_resistance",
                "description": "Identify support and resistance levels",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "market_data": {"type": "string", "description": "JSON string containing OHLCV data"},
                        "min_touches": {"type": "integer", "default": 2},
                        "tolerance": {"type": "number", "default": 0.01}
                    },
                    "required": ["market_data"]
                }
            },
            {
                "name": "detect_chart_patterns",
                "description": "Detect advanced chart patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "market_data": {"type": "string", "description": "JSON string containing OHLCV data"},
                        "pattern_types": {"type": "array", "items": {"type": "string"}, "description": "Pattern types to detect"},
                        "sensitivity": {"type": "number", "default": 0.5}
                    },
                    "required": ["market_data"]
                }
            },
            {
                "name": "comprehensive_analysis",
                "description": "Perform comprehensive technical analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "market_data": {"type": "string", "description": "JSON string containing OHLCV data"},
                        "symbol": {"type": "string", "description": "Trading symbol"},
                        "timeframe": {"type": "string", "default": "1d"}
                    },
                    "required": ["market_data", "symbol"]
                }
            }
        ]
    
    def register_tools(self, server):
        """Register all TA skill tools with MCP server"""
        for tool_def in self.tools:
            tool_name = tool_def["name"]
            
            @server.call_tool()
            async def handle_tool(name: str, arguments: dict) -> dict:
                if name == tool_name:
                    return await self.handle_tool_call(tool_name, arguments)
                return {"error": f"Unknown tool: {name}"}
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls for TA skills"""
        try:
            if tool_name == "calculate_momentum_volatility":
                return await self._handle_momentum_volatility(arguments)
            elif tool_name == "calculate_momentum_indicators":
                return await self._handle_momentum_indicators(arguments)
            elif tool_name == "analyze_volume_patterns":
                return await self._handle_volume_patterns(arguments)
            elif tool_name == "identify_support_resistance":
                return await self._handle_support_resistance(arguments)
            elif tool_name == "detect_chart_patterns":
                return await self._handle_chart_patterns(arguments)
            elif tool_name == "comprehensive_analysis":
                return await self._handle_comprehensive_analysis(arguments)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Error in TA skill tool {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_momentum_indicators(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle momentum indicators calculation"""
        try:
            market_data_str = args["market_data"]
            market_data = pd.read_json(market_data_str)
            
            indicators = args.get("indicators", ["sma", "ema", "rsi", "macd"])
            periods = args.get("periods", {})
            
            results = await self.momentum_indicators.calculate_indicators(
                market_data, indicators, periods
            )
            
            return {
                "success": True,
                "indicators": results,
                "skill": "momentum_indicators",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_momentum_volatility(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle momentum and volatility calculation"""
        try:
            market_data_str = args["market_data"]
            market_data = pd.read_json(market_data_str)
            
            period = args.get("period", 14)
            include_bollinger = args.get("include_bollinger", True)
            
            results = await self.momentum_volatility.calculate_momentum_volatility(
                market_data, period, include_bollinger
            )
            
            return {
                "success": True,
                "momentum_volatility": results,
                "skill": "momentum_volatility",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_volume_patterns(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle volume pattern analysis"""
        try:
            market_data_str = args["market_data"]
            market_data = pd.read_json(market_data_str)
            
            lookback_period = args.get("lookback_period", 20)
            
            results = await self.volume_analysis.analyze_volume_patterns(
                market_data, lookback_period
            )
            
            return {
                "success": True,
                "volume_analysis": results,
                "skill": "volume_analysis",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_support_resistance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle support/resistance identification"""
        try:
            market_data_str = args["market_data"]
            market_data = pd.read_json(market_data_str)
            
            min_touches = args.get("min_touches", 2)
            tolerance = args.get("tolerance", 0.01)
            
            results = await self.support_resistance.identify_levels(
                market_data, min_touches, tolerance
            )
            
            return {
                "success": True,
                "support_resistance": results,
                "skill": "support_resistance",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_chart_patterns(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chart pattern detection"""
        try:
            market_data_str = args["market_data"]
            market_data = pd.read_json(market_data_str)
            
            pattern_types = args.get("pattern_types", ["head_shoulders", "triangles", "flags"])
            sensitivity = args.get("sensitivity", 0.5)
            
            results = await self.harmonic_patterns.detect_patterns(
                market_data, pattern_types, sensitivity
            )
            
            return {
                "success": True,
                "chart_patterns": results,
                "skill": "harmonic_patterns",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_comprehensive_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive technical analysis"""
        try:
            market_data_str = args["market_data"]
            market_data = pd.read_json(market_data_str)
            
            symbol = args["symbol"]
            timeframe = args.get("timeframe", "1d")
            
            results = await self.comprehensive_system.perform_analysis(
                market_data, symbol, timeframe
            )
            
            return {
                "success": True,
                "comprehensive_analysis": results,
                "skill": "comprehensive_system",
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
