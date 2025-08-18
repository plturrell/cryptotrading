"""
Technical Analysis STRAND Agent
Main orchestrator agent that uses all TA skills for comprehensive crypto trading analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from ...strands import StrandsAgent, AgentConfig
from ....protocols.a2a.a2a_protocol import A2AAgentRegistry, A2A_CAPABILITIES
from .skill_1_momentum_indicators import create_momentum_indicators_tools
from .skill_2_momentum_volatility import create_momentum_volatility_tools
from .skill_3_volume_analysis import create_volume_analysis_tools
from .skill_4_support_resistance import create_support_resistance_tools
from .skill_5_chart_patterns import create_chart_pattern_tools
from .skill_6_harmonic_patterns import create_advanced_pattern_tools
from .skill_7_comprehensive_system import create_comprehensive_system_tools
from .skill_8_dashboard import create_dashboard_tools
from .grok_insights_integration import create_grok_insights_tools
from .visualization_engine import create_visualization_tools
from .performance_optimization import create_performance_tools

logger = logging.getLogger(__name__)

class TechnicalAnalysisAgent(StrandsAgent):
    """
    Technical Analysis STRAND Agent
    A2A Registered Agent ID: technical_analysis_agent
    
    Comprehensive crypto trading technical analysis agent that orchestrates
    8 specialized TA skills using the STRAND framework for modular analysis.
    """
    
    def __init__(self, agent_id: str = "technical_analysis_agent", **kwargs):
        """
        Initialize Technical Analysis Agent
        
        Args:
            agent_id: Unique identifier for the agent
            **kwargs: Additional configuration parameters
        """
        config = AgentConfig(
            agent_id=agent_id,
            agent_type="technical_analysis",
            description="Comprehensive technical analysis agent for crypto trading",
            capabilities=[
                "momentum_indicators", "momentum_volatility", "volume_analysis",
                "support_resistance", "chart_patterns", "harmonic_patterns",
                "comprehensive_system", "dashboard"
            ],
            max_concurrent_tools=5,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=60
        )
        
        super().__init__(
            agent_id=agent_id,
            agent_type="technical_analysis",
            config=config,
            **kwargs
        )
        
        # Register all skill tools
        self.register_skill_tools()
        
        # Register with A2A protocol
        capabilities = A2A_CAPABILITIES.get(agent_id, [])
        A2AAgentRegistry.register_agent(agent_id, capabilities, self)
        
        # Initialize memory for analysis caching and learning
        self._initialize_memory_system()
    
    async def initialize(self) -> bool:
        """Initialize the Technical Analysis Agent"""
        try:
            logger.info(f"Initializing Technical Analysis Agent {self.agent_id}")
            
            # Verify all skills are loaded
            if not self.skills:
                logger.error("No technical analysis skills loaded")
                return False
            
            # Initialize each skill
            for skill_name, skill in self.skills.items():
                if hasattr(skill, 'initialize'):
                    await skill.initialize()
                logger.debug(f"Skill {skill_name} ready")
            
            # Test basic functionality
            test_data = pd.DataFrame({
                'close': [100, 101, 102, 101, 103],
                'volume': [1000, 1100, 1200, 1050, 1300]
            })
            
            # Quick validation test
            try:
                await self.quick_analysis(test_data)
                logger.info("Technical analysis validation successful")
            except Exception as e:
                logger.warning(f"Technical analysis validation failed: {e}")
            
            logger.info(f"Technical Analysis Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Technical Analysis Agent {self.agent_id}: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the Technical Analysis Agent"""
        try:
            logger.info(f"Starting Technical Analysis Agent {self.agent_id}")
            
            # Start any background processes if needed
            # For now, technical analysis is primarily request-driven
            
            logger.info(f"Technical Analysis Agent {self.agent_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Technical Analysis Agent {self.agent_id}: {e}")
            return False
    
    async def _initialize_memory_system(self):
        """Initialize memory system for analysis caching and learning"""
        try:
            # Store agent configuration
            await self.store_memory(
                "agent_config",
                {
                    "agent_id": self.agent_id,
                    "skills_enabled": list(self.skill_states.keys()),
                    "initialized_at": datetime.now().isoformat()
                },
                {"type": "configuration", "persistent": True}
            )
            
            # Initialize analysis cache
            await self.store_memory(
                "analysis_cache", 
                {},
                {"type": "cache", "max_entries": 100}
            )
            
            # Initialize learning patterns
            await self.store_memory(
                "successful_patterns",
                {},
                {"type": "learning", "persistent": True}
            )
            
            logger.info(f"Memory system initialized for {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
    
    def _register_ta_tools(self):
        """Register all technical analysis skill tools"""
        try:
            # Register all skill tools
            all_tools = []
            all_tools.extend(create_momentum_indicators_tools())
            all_tools.extend(create_momentum_volatility_tools())
            all_tools.extend(create_volume_analysis_tools())
            all_tools.extend(create_support_resistance_tools())
            all_tools.extend(create_chart_pattern_tools())
            all_tools.extend(create_advanced_pattern_tools())
            all_tools.extend(create_comprehensive_system_tools())
            all_tools.extend(create_dashboard_tools())
            all_tools.extend(create_grok_insights_tools())
            all_tools.extend(create_visualization_tools())
            all_tools.extend(create_performance_tools())
            
            # Register tools from each skill
            for tool in all_tools:
                self.tool_registry.register_tool(
                    name=tool["name"],
                    function=tool["function"],
                    description=tool["description"],
                    parameters=tool.get("parameters", {}),
                    category=tool.get("category", "technical_analysis"),
                    skill=tool.get("skill", "unknown")
                )
            
            logger.info(f"Registered {len(self.tool_registry.tools)} TA tools across 8 skills")
            
        except Exception as e:
            logger.error(f"Failed to register TA tools: {e}")
            raise
    
    async def analyze_market_data(self, data: pd.DataFrame, analysis_type: str = "comprehensive", 
                                 risk_tolerance: str = "medium") -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis on market data
        
        Args:
            data: OHLCV DataFrame
            analysis_type: Type of analysis (basic, comprehensive, signals_only, dashboard)
            risk_tolerance: Risk tolerance level (low, medium, high)
        
        Returns:
            Dictionary with complete technical analysis results
        """
        try:
            analysis_start = datetime.now()
            
            # Validate input data
            if not self._validate_market_data(data):
                return {
                    "success": False,
                    "error": "Invalid market data provided",
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            
            logger.info(f"Starting {analysis_type} market analysis with risk tolerance: {risk_tolerance}")
            
            # Check memory cache for recent analysis
            cache_key = f"analysis_{hash(str(data.tail(10).values.tobytes()))}_{analysis_type}_{risk_tolerance}"
            cached_result = await self.retrieve_memory(cache_key)
            if cached_result:
                logger.info("Retrieved analysis from memory cache")
                return cached_result
            
            # Initialize results structure
            analysis_results = {
                "metadata": {
                    "analysis_type": analysis_type,
                    "risk_tolerance": risk_tolerance,
                    "data_points": len(data),
                    "timeframe": self._detect_timeframe(data),
                    "start_time": analysis_start.isoformat()
                },
                "indicators": {},
                "patterns": {},
                "signals": [],
                "levels": {},
                "risk_assessment": {},
                "recommendation": {},
                "dashboard": {}
            }
            
            # Execute analysis based on type
            if analysis_type in ["basic", "comprehensive"]:
                await self._run_momentum_analysis(data, analysis_results)
            
            if analysis_type in ["comprehensive"]:
                await self._run_advanced_analysis(data, analysis_results)
                await self._run_comprehensive_system_analysis(data, analysis_results)
            
            if analysis_type in ["dashboard", "comprehensive"]:
                await self._run_dashboard_analysis(data, analysis_results)
            
            # Generate final recommendation
            if analysis_type != "signals_only":
                await self._generate_final_recommendation(data, analysis_results, risk_tolerance)
            
            # Add completion metadata
            analysis_end = datetime.now()
            analysis_results["metadata"]["end_time"] = analysis_end.isoformat()
            analysis_results["metadata"]["duration_seconds"] = (analysis_end - analysis_start).total_seconds()
            analysis_results["metadata"]["tools_used"] = len([s for s in self.skill_states.values() if s["last_used"]])
            
            logger.info(f"Technical analysis completed in {analysis_results['metadata']['duration_seconds']:.2f}s")
            
            # Aggregate all results
            comprehensive_analysis = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type,
                "risk_tolerance": risk_tolerance,
                "data_points": len(data),
                "skills_used": list(self.skill_states.keys()),
                "results": analysis_results,
                "summary": self._generate_analysis_summary(analysis_results),
                "recommendations": self._generate_recommendations(analysis_results, risk_tolerance)
            }
            
            # Store analysis in memory cache
            await self.store_memory(
                cache_key,
                comprehensive_analysis,
                {"type": "analysis_cache", "expires_at": (datetime.now() + timedelta(minutes=15)).isoformat()}
            )
            
            # Learn from successful patterns
            await self._learn_from_analysis(comprehensive_analysis)
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
                "analysis_type": analysis_type
            }
    
    async def _learn_from_analysis(self, analysis: Dict[str, Any]):
        """Learn from successful analysis patterns"""
        try:
            # Extract successful patterns
            if analysis.get("results", {}).get("success", False):
                patterns = await self.retrieve_memory("successful_patterns") or {}
                
                # Store successful signal patterns
                for signal in analysis.get("results", {}).get("signals", []):
                    if signal.get("confidence", 0) > 0.7:
                        pattern_key = f"{signal.get('type', 'unknown')}_{signal.get('direction', 'neutral')}"
                        if pattern_key not in patterns:
                            patterns[pattern_key] = {"count": 0, "avg_confidence": 0}
                        
                        patterns[pattern_key]["count"] += 1
                        patterns[pattern_key]["avg_confidence"] = (
                            patterns[pattern_key]["avg_confidence"] + signal.get("confidence", 0)
                        ) / 2
                
                await self.store_memory("successful_patterns", patterns, {"type": "learning"})
                
        except Exception as e:
            logger.error(f"Failed to learn from analysis: {e}")
    
    async def _run_momentum_analysis(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Run momentum technical analysis (Skills 1-3)"""
        try:
            # Skill 1: Basic Indicators
            momentum_result = await self.execute_tool("analyze_momentum_indicators", {"data": data})
            if momentum_result.success:
                results["indicators"]["momentum"] = momentum_result.result
                results["signals"].extend(momentum_result.result.get("signals", []))
                self.skill_states["momentum_indicators"]["last_used"] = datetime.now()
            
            # Skill 2: Momentum/Volatility
            momentum_result = await self.execute_tool("analyze_momentum_volatility", {"data": data})
            if momentum_result.success:
                results["indicators"]["momentum_volatility"] = momentum_result.result
                results["signals"].extend(momentum_result.result.get("signals", []))
                self.skill_states["momentum_volatility"]["last_used"] = datetime.now()
            
            # Skill 3: Volume Analysis
            volume_result = await self.execute_tool("analyze_volume_indicators", {"data": data})
            if volume_result.success:
                results["indicators"]["volume"] = volume_result.result
                results["signals"].extend(volume_result.result.get("signals", []))
                self.skill_states["volume_analysis"]["last_used"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Basic analysis failed: {e}")
    
    async def _run_advanced_analysis(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Run advanced technical analysis (Skills 4-6)"""
        try:
            # Skill 4: Support/Resistance
            sr_result = await self.execute_tool("analyze_support_resistance", {"data": data})
            if sr_result.success:
                results["levels"]["support_resistance"] = sr_result.result
                results["signals"].extend(sr_result.result.get("signals", []))
                self.skill_states["support_resistance"]["last_used"] = datetime.now()
            
            # Skill 5: Chart Patterns
            patterns_result = await self.execute_tool("analyze_chart_patterns", {"data": data})
            if patterns_result.success:
                results["patterns"]["chart_patterns"] = patterns_result.result
                results["signals"].extend(patterns_result.result.get("signals", []))
                self.skill_states["chart_patterns"]["last_used"] = datetime.now()
            
            # Skill 6: Advanced Patterns
            advanced_result = await self.execute_tool("analyze_advanced_patterns", {"data": data})
            if advanced_result.success:
                results["patterns"]["harmonic_patterns"] = advanced_result.result
                results["signals"].extend(advanced_result.result.get("signals", []))
                self.skill_states["harmonic_patterns"]["last_used"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
    
    async def _run_comprehensive_system_analysis(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Run comprehensive system analysis (Skill 7)"""
        try:
            # Calculate signal strength
            signal_strength_result = await self.execute_tool(
                "calculate_signal_strength", 
                {"signals": results["signals"]}
            )
            if signal_strength_result.success:
                results["signal_analysis"] = signal_strength_result.result
            
            # Assess market regime
            regime_result = await self.execute_tool("assess_market_regime", {"data": data})
            if regime_result.success:
                results["market_regime"] = regime_result.result
            
            # Calculate risk metrics
            risk_result = await self.execute_tool(
                "calculate_risk_metrics", 
                {"data": data, "signals": results["signals"]}
            )
            if risk_result.success:
                results["risk_assessment"] = risk_result.result
            
            self.skill_states["comprehensive_system"]["last_used"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Comprehensive system analysis failed: {e}")
    
    async def _run_dashboard_analysis(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Run dashboard analysis (Skill 8)"""
        try:
            # Generate summary report
            summary_result = await self.execute_tool(
                "generate_ta_summary_report",
                {"data": data, "all_analysis": results}
            )
            if summary_result.success:
                results["dashboard"]["summary"] = summary_result.result
            
            # Create signal heatmap
            heatmap_result = await self.execute_tool(
                "create_signal_heatmap",
                {"signals": results["signals"]}
            )
            if heatmap_result.success:
                results["dashboard"]["heatmap"] = heatmap_result.result
            
            # Generate performance metrics
            performance_result = await self.execute_tool(
                "generate_performance_metrics",
                {"data": data, "signals": results["signals"]}
            )
            if performance_result.success:
                results["dashboard"]["performance"] = performance_result.result
            
            # Create alerts
            alerts_result = await self.execute_tool(
                "create_alert_system",
                {"signals": results["signals"]}
            )
            if alerts_result.success:
                results["dashboard"]["alerts"] = alerts_result.result
            
            self.skill_states["dashboard"]["last_used"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Dashboard analysis failed: {e}")
    
    async def _generate_final_recommendation(self, data: pd.DataFrame, 
                                           results: Dict[str, Any], 
                                           risk_tolerance: str):
        """Generate final trading recommendation"""
        try:
            recommendation_result = await self.execute_tool(
                "generate_trading_recommendation",
                {
                    "data": data,
                    "all_signals": results["signals"],
                    "risk_tolerance": risk_tolerance
                }
            )
            
            if recommendation_result.success:
                results["recommendation"] = recommendation_result.result
            
        except Exception as e:
            logger.error(f"Final recommendation generation failed: {e}")
    
    def _validate_market_data(self, data: pd.DataFrame) -> bool:
        """Validate market data format and completeness"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns. Expected: {required_columns}")
                return False
            
            if len(data) < 20:
                logger.error("Insufficient data points for analysis (minimum 20 required)")
                return False
            
            if data[required_columns].isnull().any().any():
                logger.warning("Data contains null values")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect timeframe of the data"""
        try:
            if len(data) < 2:
                return "unknown"
            
            # Calculate average time difference
            time_diffs = pd.Series(data.index).diff().dropna()
            avg_diff = time_diffs.mean()
            
            if avg_diff <= pd.Timedelta(minutes=1):
                return "1m"
            elif avg_diff <= pd.Timedelta(minutes=5):
                return "5m"
            elif avg_diff <= pd.Timedelta(minutes=15):
                return "15m"
            elif avg_diff <= pd.Timedelta(hours=1):
                return "1h"
            elif avg_diff <= pd.Timedelta(hours=4):
                return "4h"
            elif avg_diff <= pd.Timedelta(days=1):
                return "1d"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    async def get_skill_status(self) -> Dict[str, Any]:
        """Get status of all TA skills"""
        return {
            "skill_states": self.skill_states,
            "total_tools": len(self.tool_registry.tools),
            "tools_by_skill": {
                skill: len([t for t in self.tool_registry.tools.values() 
                           if t.metadata.get("skill") == skill])
                for skill in self.skill_states.keys()
            },
            "circuit_breaker_status": {
                name: breaker.is_open for name, breaker in self.circuit_breakers.items()
            }
        }
    
    async def quick_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform quick technical analysis with essential indicators only"""
        try:
            # Run only essential tools for quick analysis
            essential_results = {
                "signals": [],
                "key_levels": {},
                "trend": "unknown",
                "volatility": "unknown"
            }
            
            # Basic indicators (SMA, RSI)
            basic_result = await self.execute_tool("calculate_sma", {"data": data, "period": 20})
            if basic_result.success:
                essential_results["sma_20"] = basic_result.result
            
            rsi_result = await self.execute_tool("calculate_rsi", {"data": data})
            if rsi_result.success:
                essential_results["rsi"] = rsi_result.result
                essential_results["signals"].extend(rsi_result.result.get("signals", []))
            
            # Support/Resistance
            sr_result = await self.execute_tool("detect_pivot_points", {"data": data})
            if sr_result.success:
                essential_results["key_levels"] = sr_result.result
            
            # Signal strength
            if essential_results["signals"]:
                strength_result = await self.execute_tool(
                    "calculate_signal_strength",
                    {"signals": essential_results["signals"]}
                )
                if strength_result.success:
                    essential_results["overall_signal"] = strength_result.result["overall_signal"]
            
            return {
                "success": True,
                "analysis": essential_results,
                "analysis_type": "quick",
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quick analysis failed: {e}")
            return {"success": False, "error": str(e)}

def create_technical_analysis_agent(agent_id: str = None, **kwargs) -> TechnicalAnalysisAgent:
    """
    Factory function to create Technical Analysis Agent
    
    Args:
        agent_id: Optional agent ID (auto-generated if not provided)
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured TechnicalAnalysisAgent instance
    """
    if agent_id is None:
        agent_id = f"ta_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return TechnicalAnalysisAgent(agent_id=agent_id, **kwargs)
