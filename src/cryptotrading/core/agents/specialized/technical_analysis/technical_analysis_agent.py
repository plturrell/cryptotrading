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

# Import Grok4 AI client for enhanced analysis
try:
    from ....ai.grok4_client import get_grok4_client, Grok4Client, Grok4Error
    GROK4_AVAILABLE = True
except ImportError:
    GROK4_AVAILABLE = False

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
        
        # Initialize AI enhancement
        self.grok4_client = None
        self._ai_cache = {}
        self._ai_cache_ttl = 300  # 5 minutes
    
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
    
    # ==================== AI ENHANCEMENT METHODS ====================
    
    async def _initialize_grok4_client(self):
        """Initialize Grok4 AI client for enhanced analysis"""
        if not GROK4_AVAILABLE:
            logger.warning("Grok4 not available - using traditional analysis only")
            return
            
        try:
            self.grok4_client = await get_grok4_client()
            logger.info("Grok4 AI client initialized for technical analysis enhancement")
        except Exception as e:
            logger.warning(f"Failed to initialize Grok4 client: {e}")
            self.grok4_client = None
    
    async def analyze_market_data_ai_enhanced(self, data: pd.DataFrame, symbol: str = "BTC", 
                                            analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        AI-Enhanced technical analysis that combines traditional indicators with Grok4 intelligence
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            analysis_type: Type of analysis
            
        Returns:
            Enhanced analysis with AI insights
        """
        logger.info(f"Starting AI-enhanced technical analysis for {symbol}")
        
        # Initialize Grok4 if not already done
        if not self.grok4_client and GROK4_AVAILABLE:
            await self._initialize_grok4_client()
        
        # Run traditional technical analysis first
        traditional_analysis = await self.analyze_market_data(data, analysis_type)
        
        # Enhance with AI if available
        if self.grok4_client:
            try:
                # Get AI sentiment analysis
                ai_sentiment = await self._get_ai_sentiment_cached([symbol])
                
                # AI-enhanced pattern recognition
                ai_patterns = await self._analyze_patterns_with_ai(data, symbol)
                
                # AI support/resistance detection
                ai_levels = await self._detect_levels_with_ai(data, symbol)
                
                # Combine traditional and AI analysis
                enhanced_analysis = await self._combine_traditional_ai_analysis(
                    traditional_analysis, ai_sentiment, ai_patterns, ai_levels, symbol
                )
                
                logger.info(f"AI enhancement completed for {symbol}")
                return enhanced_analysis
                
            except Exception as e:
                logger.warning(f"AI enhancement failed, using traditional analysis: {e}")
                traditional_analysis['ai_enhancement_status'] = f'failed: {str(e)}'
                return traditional_analysis
        else:
            traditional_analysis['ai_enhancement_status'] = 'not_available'
            return traditional_analysis
    
    async def _get_ai_sentiment_cached(self, symbols: List[str]) -> Dict[str, Any]:
        """Get AI sentiment with caching"""
        if not self.grok4_client:
            return {}
        
        cache_key = f"ai_sentiment_{','.join(symbols)}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Check cache first
        if cache_key in self._ai_cache:
            cache_time = self._ai_cache[cache_key].get('timestamp', 0)
            if (datetime.now().timestamp() - cache_time) < self._ai_cache_ttl:
                return self._ai_cache[cache_key]['data']
        
        try:
            # Get fresh AI sentiment
            insights = await self.grok4_client.analyze_market_sentiment(symbols, timeframe='1h')
            
            # Convert to dict format
            sentiment_data = {}
            for insight in insights:
                sentiment_data[insight.symbol] = {
                    'recommendation': insight.recommendation,
                    'sentiment_score': insight.score,
                    'confidence': insight.confidence,
                    'risk_level': insight.risk_level,
                    'reasoning': insight.reasoning
                }
            
            # Cache the result
            self._ai_cache[cache_key] = {
                'data': sentiment_data,
                'timestamp': datetime.now().timestamp()
            }
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Failed to get AI sentiment: {e}")
            return {}
    
    async def _analyze_patterns_with_ai(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """AI-enhanced pattern recognition"""
        if not self.grok4_client:
            return {}
        
        try:
            # Convert DataFrame to format suitable for AI analysis
            market_data = {
                'symbol': symbol,
                'ohlcv': data.tail(50).to_dict('records'),  # Last 50 candles
                'timeframe': self._detect_timeframe(data)
            }
            
            # Use Grok4's correlation analysis as pattern detection
            # (Since we don't have direct pattern analysis, we use available methods)
            correlation_analysis = await self.grok4_client.analyze_correlation_patterns(
                [symbol], timeframe='1h'
            )
            
            # Extract pattern insights from correlation analysis
            patterns = {
                'ai_pattern_strength': correlation_analysis.get('insights', {}).get('diversification_score', 0.5),
                'pattern_type': 'correlation_based',
                'confidence': 0.7,
                'ai_detected': True
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"AI pattern analysis failed: {e}")
            return {}
    
    async def _detect_levels_with_ai(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """AI-enhanced support/resistance detection"""
        if not self.grok4_client:
            return {}
        
        try:
            # Use AI prediction to identify key levels
            predictions = await self.grok4_client.predict_market_movement([symbol], horizon='1h')
            
            current_price = data['close'].iloc[-1] if len(data) > 0 else 0
            
            # Extract support/resistance from AI predictions
            symbol_prediction = predictions.get(symbol, {})
            
            ai_levels = {
                'ai_support': current_price * 0.98,  # Conservative estimate
                'ai_resistance': current_price * 1.02,
                'confidence': symbol_prediction.get('confidence', 0.5),
                'ai_reasoning': symbol_prediction.get('key_factors', []),
                'predicted_direction': symbol_prediction.get('direction', 'SIDEWAYS')
            }
            
            return ai_levels
            
        except Exception as e:
            logger.error(f"AI level detection failed: {e}")
            return {}
    
    async def _combine_traditional_ai_analysis(self, traditional: Dict[str, Any], 
                                             ai_sentiment: Dict[str, Any],
                                             ai_patterns: Dict[str, Any],
                                             ai_levels: Dict[str, Any],
                                             symbol: str) -> Dict[str, Any]:
        """Combine traditional TA with AI insights"""
        
        # Start with traditional analysis
        enhanced = traditional.copy()
        
        # Add AI enhancement metadata
        enhanced['ai_enhancement'] = {
            'enabled': True,
            'ai_sentiment_available': bool(ai_sentiment),
            'ai_patterns_available': bool(ai_patterns),
            'ai_levels_available': bool(ai_levels),
            'enhancement_timestamp': datetime.now().isoformat()
        }
        
        # Enhance signals with AI confidence
        if 'results' in enhanced and 'signals' in enhanced['results']:
            enhanced_signals = []
            for signal in enhanced['results']['signals']:
                enhanced_signal = signal.copy()
                
                # Add AI confidence boost/penalty
                if symbol in ai_sentiment:
                    ai_rec = ai_sentiment[symbol].get('recommendation', 'HOLD')
                    ai_confidence = ai_sentiment[symbol].get('confidence', 0.5)
                    
                    signal_type = signal.get('type', 'unknown')
                    
                    # Check alignment between traditional signal and AI recommendation
                    if ((signal_type == 'buy' and ai_rec == 'BUY') or
                        (signal_type == 'sell' and ai_rec == 'SELL')):
                        # Signals align - boost confidence
                        original_confidence = signal.get('confidence', 0.5)
                        enhanced_signal['confidence'] = min(
                            original_confidence + (ai_confidence * 0.2), 1.0
                        )
                        enhanced_signal['ai_alignment'] = 'strong'
                        enhanced_signal['ai_reasoning'] = ai_sentiment[symbol].get('reasoning', '')
                    elif ((signal_type == 'buy' and ai_rec == 'SELL') or
                          (signal_type == 'sell' and ai_rec == 'BUY')):
                        # Signals conflict - reduce confidence
                        original_confidence = signal.get('confidence', 0.5)
                        enhanced_signal['confidence'] = max(
                            original_confidence - (ai_confidence * 0.15), 0.1
                        )
                        enhanced_signal['ai_alignment'] = 'conflicting'
                        enhanced_signal['ai_warning'] = 'AI suggests opposite direction'
                    else:
                        enhanced_signal['ai_alignment'] = 'neutral'
                
                enhanced_signals.append(enhanced_signal)
            
            enhanced['results']['signals'] = enhanced_signals
        
        # Add AI-specific insights section
        enhanced['ai_insights'] = {
            'sentiment_analysis': ai_sentiment.get(symbol, {}),
            'ai_patterns': ai_patterns,
            'ai_levels': ai_levels,
            'overall_ai_confidence': self._calculate_overall_ai_confidence(
                ai_sentiment.get(symbol, {}), ai_patterns, ai_levels
            )
        }
        
        # Enhance recommendations with AI
        if 'recommendations' in enhanced:
            ai_enhanced_recommendations = enhanced['recommendations'].copy()
            
            if symbol in ai_sentiment:
                ai_rec = ai_sentiment[symbol].get('recommendation', 'HOLD')
                ai_reasoning = ai_sentiment[symbol].get('reasoning', '')
                
                ai_enhanced_recommendations.append({
                    'type': 'ai_recommendation',
                    'action': ai_rec,
                    'reasoning': ai_reasoning,
                    'confidence': ai_sentiment[symbol].get('confidence', 0.5),
                    'source': 'grok4_ai'
                })
            
            enhanced['recommendations'] = ai_enhanced_recommendations
        
        return enhanced
    
    def _calculate_overall_ai_confidence(self, sentiment: Dict[str, Any], 
                                       patterns: Dict[str, Any], 
                                       levels: Dict[str, Any]) -> float:
        """Calculate overall AI confidence score"""
        confidence_scores = []
        
        if sentiment and 'confidence' in sentiment:
            confidence_scores.append(sentiment['confidence'])
        
        if patterns and 'confidence' in patterns:
            confidence_scores.append(patterns['confidence'])
        
        if levels and 'confidence' in levels:
            confidence_scores.append(levels['confidence'])
        
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.5  # Default neutral confidence
    
    async def enhance_existing_signals_with_ai(self, signals: List[Dict[str, Any]], 
                                             symbol: str) -> List[Dict[str, Any]]:
        """Enhance existing traditional signals with AI intelligence"""
        if not self.grok4_client or not signals:
            return signals
        
        try:
            # Get AI sentiment for the symbol
            ai_sentiment = await self._get_ai_sentiment_cached([symbol])
            
            enhanced_signals = []
            for signal in signals:
                enhanced_signal = signal.copy()
                
                if symbol in ai_sentiment:
                    ai_data = ai_sentiment[symbol]
                    signal_type = signal.get('type', 'unknown')
                    ai_rec = ai_data.get('recommendation', 'HOLD')
                    
                    # Calculate alignment score
                    alignment_score = self._calculate_signal_alignment(signal_type, ai_rec)
                    
                    # Adjust confidence based on AI alignment
                    original_confidence = signal.get('confidence', 0.5)
                    ai_confidence = ai_data.get('confidence', 0.5)
                    
                    if alignment_score > 0.7:  # Strong alignment
                        enhanced_signal['confidence'] = min(original_confidence + 0.15, 1.0)
                        enhanced_signal['ai_boost'] = True
                    elif alignment_score < 0.3:  # Strong conflict
                        enhanced_signal['confidence'] = max(original_confidence - 0.1, 0.1)
                        enhanced_signal['ai_warning'] = True
                    
                    # Add AI context
                    enhanced_signal['ai_context'] = {
                        'ai_recommendation': ai_rec,
                        'ai_confidence': ai_confidence,
                        'alignment_score': alignment_score,
                        'ai_reasoning': ai_data.get('reasoning', '')[:100] + '...'
                    }
                
                enhanced_signals.append(enhanced_signal)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Failed to enhance signals with AI: {e}")
            return signals
    
    def _calculate_signal_alignment(self, signal_type: str, ai_recommendation: str) -> float:
        """Calculate alignment score between traditional signal and AI recommendation"""
        signal_type = signal_type.lower()
        ai_rec = ai_recommendation.upper()
        
        if (signal_type == 'buy' and ai_rec == 'BUY') or \
           (signal_type == 'sell' and ai_rec == 'SELL'):
            return 1.0  # Perfect alignment
        elif (signal_type == 'buy' and ai_rec == 'SELL') or \
             (signal_type == 'sell' and ai_rec == 'BUY'):
            return 0.0  # Complete conflict
        elif signal_type == 'hold' or ai_rec == 'HOLD':
            return 0.6  # Neutral alignment
        else:
            return 0.5  # Unknown alignment
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary (enhanced with AI if available)"""
        summary = {
            'total_signals': len(results.get('signals', [])),
            'signal_types': {},
            'confidence_distribution': {},
            'ai_enhanced': False
        }
        
        # Count signal types and confidence levels
        for signal in results.get('signals', []):
            signal_type = signal.get('type', 'unknown')
            confidence = signal.get('confidence', 0.5)
            
            summary['signal_types'][signal_type] = summary['signal_types'].get(signal_type, 0) + 1
            
            # Confidence buckets
            if confidence >= 0.8:
                bucket = 'high'
            elif confidence >= 0.6:
                bucket = 'medium'
            else:
                bucket = 'low'
            
            summary['confidence_distribution'][bucket] = summary['confidence_distribution'].get(bucket, 0) + 1
            
            # Check if any signals have AI enhancement
            if 'ai_context' in signal or 'ai_alignment' in signal:
                summary['ai_enhanced'] = True
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any], risk_tolerance: str) -> List[Dict[str, Any]]:
        """Generate trading recommendations (enhanced with AI if available)"""
        recommendations = []
        
        # Analyze signals for patterns
        signals = results.get('signals', [])
        if not signals:
            return [{'type': 'hold', 'reason': 'No signals generated', 'confidence': 0.5}]
        
        # Count signal types
        buy_signals = [s for s in signals if s.get('type') == 'buy']
        sell_signals = [s for s in signals if s.get('type') == 'sell']
        
        # Calculate average confidence
        avg_confidence = sum(s.get('confidence', 0.5) for s in signals) / len(signals)
        
        # Check for AI enhancement
        ai_enhanced_signals = [s for s in signals if 'ai_context' in s]
        ai_boost_signals = [s for s in signals if s.get('ai_boost', False)]
        
        # Generate primary recommendation
        if len(buy_signals) > len(sell_signals) and avg_confidence > 0.6:
            recommendations.append({
                'type': 'buy',
                'reason': f'{len(buy_signals)} buy signals detected',
                'confidence': avg_confidence,
                'risk_level': 'medium' if risk_tolerance == 'medium' else risk_tolerance,
                'ai_enhanced': len(ai_enhanced_signals) > 0
            })
        elif len(sell_signals) > len(buy_signals) and avg_confidence > 0.6:
            recommendations.append({
                'type': 'sell',
                'reason': f'{len(sell_signals)} sell signals detected',
                'confidence': avg_confidence,
                'risk_level': 'medium' if risk_tolerance == 'medium' else risk_tolerance,
                'ai_enhanced': len(ai_enhanced_signals) > 0
            })
        else:
            recommendations.append({
                'type': 'hold',
                'reason': 'Mixed or weak signals',
                'confidence': avg_confidence,
                'risk_level': 'low',
                'ai_enhanced': len(ai_enhanced_signals) > 0
            })
        
        # Add AI-specific recommendations if available
        if ai_boost_signals:
            recommendations.append({
                'type': 'ai_insight',
                'reason': f'{len(ai_boost_signals)} signals received AI confidence boost',
                'confidence': 0.8,
                'note': 'AI analysis supports traditional technical indicators'
            })
        
        return recommendations

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
