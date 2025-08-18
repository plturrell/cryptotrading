"""
Data Analysis Agent - STRANDS Integration
Specialized agent for statistical analysis and data quality validation
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from ..base_agent import BaseAgent, AgentConfig
from ...data_ingestion.quality_validator import FactorQualityValidator
from ...processing.parallel_executor import ParallelExecutor
from ...infrastructure.mcp.data_analysis_mcp_tools import data_analysis_mcp_tools

# Import Grok4 AI client for enhanced data analysis
try:
    from ...ai.grok4_client import get_grok4_client, Grok4Client, Grok4Error
    GROK4_AVAILABLE = True
except ImportError:
    GROK4_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataAnalysisAgent(BaseAgent):
    """STRANDS agent for data analysis operations"""
    
    def __init__(self, agent_id: str = "data_analysis_agent", **kwargs):
        """Initialize Data Analysis Agent"""
        config = AgentConfig(
            agent_id=agent_id,
            agent_type="data_analysis",
            description="Statistical analysis and data quality validation agent",
            capabilities=[
                "validate_data_quality", "analyze_data_distribution", "compute_correlation_matrix",
                "detect_outliers", "compute_rolling_statistics"
            ],
            max_concurrent_tools=4,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=30
        )
        
        super().__init__(
            agent_id=agent_id,
            agent_type="data_analysis",
            config=config,
            **kwargs
        )
        
        self.quality_validator = FactorQualityValidator()
        self.parallel_executor = ParallelExecutor()
        self.mcp_tools = data_analysis_mcp_tools
        
        # Initialize AI enhancement
        self.grok4_client = None
        self._ai_cache = {}
        self._ai_cache_ttl = 300  # 5 minutes
        
        # Register MCP tools as STRANDS tools
        self._register_strands_tools()
        
        logger.info(f"Data Analysis Agent {agent_id} initialized")
    
    def _register_strands_tools(self):
        """Register MCP tools as STRANDS tools"""
        for tool_def in self.mcp_tools.tools:
            tool_name = tool_def["name"]
            
            # Create STRANDS tool wrapper
            async def tool_wrapper(tool_name=tool_name, **kwargs):
                return await self.mcp_tools.handle_tool_call(tool_name, kwargs)
            
            # Register with STRANDS
            self.register_tool(
                name=tool_name,
                description=tool_def["description"],
                func=tool_wrapper,
                input_schema=tool_def["inputSchema"]
            )
    
    async def initialize(self) -> bool:
        """Initialize the Data Analysis Agent"""
        try:
            logger.info(f"Initializing Data Analysis Agent {self.agent_id}")
            
            # Test quality validator
            test_data = pd.Series([1, 2, 3, 4, 5])
            test_result = await self.quality_validator.validate_factor("test", test_data, {})
            logger.info(f"Quality validator test: {test_result.get('score', 0)}")
            
            logger.info(f"Data Analysis Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Data Analysis Agent {self.agent_id}: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the Data Analysis Agent"""
        try:
            logger.info(f"Starting Data Analysis Agent {self.agent_id}")
            
            # Data analysis is primarily request-driven
            # No background processes needed
            
            logger.info(f"Data Analysis Agent {self.agent_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Data Analysis Agent {self.agent_id}: {e}")
            return False
    
    async def validate_factor_quality(self, data: Dict[str, Any], factor_names: List[str], 
                                    validation_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate quality of factors in data"""
        try:
            result = await self.execute_tool("validate_data_quality", {
                "data": data,
                "factor_names": factor_names,
                "validation_rules": validation_rules or {}
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating factor quality: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_statistical_distribution(self, data: Dict[str, Any], 
                                             columns: List[str] = None) -> Dict[str, Any]:
        """Analyze statistical distribution of data"""
        try:
            result = await self.execute_tool("analyze_data_distribution", {
                "data": data,
                "columns": columns
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing distribution: {e}")
            return {"success": False, "error": str(e)}
    
    async def compute_correlations(self, data: Dict[str, Any], 
                                 method: str = "pearson") -> Dict[str, Any]:
        """Compute correlation matrix"""
        try:
            result = await self.execute_tool("compute_correlation_matrix", {
                "data": data,
                "method": method
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing correlations: {e}")
            return {"success": False, "error": str(e)}
    
    async def find_outliers(self, data: Dict[str, Any], method: str = "iqr", 
                          threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers in data"""
        try:
            result = await self.execute_tool("detect_outliers", {
                "data": data,
                "method": method,
                "threshold": threshold
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return {"success": False, "error": str(e)}
    
    async def compute_time_series_stats(self, data: Dict[str, Any], window: int = 20, 
                                      statistics: List[str] = None) -> Dict[str, Any]:
        """Compute rolling statistics for time series"""
        try:
            result = await self.execute_tool("compute_rolling_statistics", {
                "data": data,
                "window": window,
                "statistics": statistics or ["mean", "std"]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing rolling statistics: {e}")
            return {"success": False, "error": str(e)}
    
    async def comprehensive_data_analysis(self, data: Dict[str, Any], 
                                        factor_names: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "comprehensive"
            }
            
            # Distribution analysis
            distribution_result = await self.analyze_statistical_distribution(data)
            results["distribution_analysis"] = distribution_result
            
            # Correlation analysis
            correlation_result = await self.compute_correlations(data)
            results["correlation_analysis"] = correlation_result
            
            # Outlier detection
            outlier_result = await self.find_outliers(data)
            results["outlier_analysis"] = outlier_result
            
            # Quality validation if factor names provided
            if factor_names:
                quality_result = await self.validate_factor_quality(data, factor_names)
                results["quality_validation"] = quality_result
            
            # Rolling statistics for time series data
            rolling_result = await self.compute_time_series_stats(data)
            results["rolling_statistics"] = rolling_result
            
            return {
                "success": True,
                "comprehensive_analysis": results
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages with data analysis operations"""
        try:
            message_type = message.get("type", "unknown")
            
            if message_type == "validate_quality":
                data = message.get("data", {})
                factor_names = message.get("factor_names", [])
                validation_rules = message.get("validation_rules")
                return await self.validate_factor_quality(data, factor_names, validation_rules)
                
            elif message_type == "analyze_distribution":
                data = message.get("data", {})
                columns = message.get("columns")
                return await self.analyze_statistical_distribution(data, columns)
                
            elif message_type == "compute_correlations":
                data = message.get("data", {})
                method = message.get("method", "pearson")
                return await self.compute_correlations(data, method)
                
            elif message_type == "detect_outliers":
                data = message.get("data", {})
                method = message.get("method", "iqr")
                threshold = message.get("threshold", 3.0)
                return await self.find_outliers(data, method, threshold)
                
            elif message_type == "rolling_statistics":
                data = message.get("data", {})
                window = message.get("window", 20)
                statistics = message.get("statistics")
                return await self.compute_time_series_stats(data, window, statistics)
                
            elif message_type == "comprehensive_analysis":
                data = message.get("data", {})
                factor_names = message.get("factor_names")
                return await self.comprehensive_data_analysis(data, factor_names)
                
            else:
                return await super().process_message(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== AI ENHANCEMENT METHODS ====================
    
    async def _initialize_grok4_client(self):
        """Initialize Grok4 AI client for enhanced data analysis"""
        if not GROK4_AVAILABLE:
            logger.warning("Grok4 not available - using traditional analysis only")
            return
            
        try:
            self.grok4_client = await get_grok4_client()
            logger.info("Grok4 AI client initialized for data analysis enhancement")
        except Exception as e:
            logger.warning(f"Failed to initialize Grok4 client: {e}")
            self.grok4_client = None
    
    async def analyze_data_quality_ai_enhanced(self, data: Dict[str, Any], 
                                             symbols: List[str] = None) -> Dict[str, Any]:
        """
        AI-Enhanced data quality analysis that combines traditional validation with AI intelligence
        
        Args:
            data: Market data to analyze
            symbols: Trading symbols (for context)
            
        Returns:
            Enhanced quality analysis with AI insights
        """
        logger.info("Starting AI-enhanced data quality analysis")
        
        # Initialize Grok4 if not already done
        if not self.grok4_client and GROK4_AVAILABLE:
            await self._initialize_grok4_client()
        
        # Run traditional data analysis first
        traditional_analysis = await self.comprehensive_data_analysis(data, symbols)
        
        # Enhance with AI if available
        if self.grok4_client and symbols:
            try:
                # AI-powered anomaly detection
                ai_anomalies = await self._detect_anomalies_with_ai(data, symbols)
                
                # AI data quality scoring
                ai_quality_score = await self._assess_data_quality_with_ai(data, symbols)
                
                # AI correlation insights
                ai_correlations = await self._analyze_correlations_with_ai(data, symbols)
                
                # Combine traditional and AI analysis
                enhanced_analysis = await self._combine_traditional_ai_data_analysis(
                    traditional_analysis, ai_anomalies, ai_quality_score, ai_correlations
                )
                
                logger.info("AI enhancement completed for data analysis")
                return enhanced_analysis
                
            except Exception as e:
                logger.warning(f"AI enhancement failed, using traditional analysis: {e}")
                traditional_analysis['ai_enhancement_status'] = f'failed: {str(e)}'
                return traditional_analysis
        else:
            traditional_analysis['ai_enhancement_status'] = 'not_available'
            return traditional_analysis
    
    async def _detect_anomalies_with_ai(self, data: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """AI-powered anomaly detection in market data"""
        if not self.grok4_client:
            return {}
        
        try:
            # Use AI correlation analysis to detect unusual patterns
            correlation_analysis = await self.grok4_client.analyze_correlation_patterns(
                symbols, timeframe='1h'
            )
            
            # Extract anomaly insights from correlation analysis
            anomalies = {
                'correlation_anomalies': [],
                'pattern_breaks': [],
                'ai_confidence': 0.7
            }
            
            # Check for correlation matrix anomalies
            correlation_matrix = correlation_analysis.get('correlation_matrix', {})
            for symbol_pair, correlation in correlation_matrix.items():
                if isinstance(correlation, dict):
                    for other_symbol, corr_value in correlation.items():
                        if isinstance(corr_value, (int, float)):
                            if abs(corr_value) > 0.95:  # Unusually high correlation
                                anomalies['correlation_anomalies'].append({
                                    'symbols': [symbol_pair, other_symbol],
                                    'correlation': corr_value,
                                    'anomaly_type': 'high_correlation',
                                    'severity': 'medium'
                                })
            
            # Check diversification score for anomalies
            diversification_score = correlation_analysis.get('insights', {}).get('diversification_score', 0.5)
            if diversification_score < 0.3:
                anomalies['pattern_breaks'].append({
                    'type': 'low_diversification',
                    'score': diversification_score,
                    'severity': 'high',
                    'description': 'Portfolio shows unusually low diversification'
                })
            
            anomalies['total_anomalies'] = len(anomalies['correlation_anomalies']) + len(anomalies['pattern_breaks'])
            
            return anomalies
            
        except Exception as e:
            logger.error(f"AI anomaly detection failed: {e}")
            return {}
    
    async def _assess_data_quality_with_ai(self, data: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """AI-powered data quality assessment"""
        if not self.grok4_client:
            return {}
        
        try:
            # Use AI sentiment analysis to validate data consistency
            sentiment_analysis = await self.grok4_client.analyze_market_sentiment(symbols, timeframe='1h')
            
            # Calculate AI quality score based on sentiment consistency
            quality_metrics = {
                'ai_quality_score': 0.8,  # Default good score
                'consistency_check': True,
                'sentiment_alignment': 'good',
                'data_freshness': 'current'
            }
            
            # Analyze sentiment consistency
            sentiment_scores = [insight.score for insight in sentiment_analysis]
            if sentiment_scores:
                score_variance = np.var(sentiment_scores)
                if score_variance > 0.1:  # High variance indicates inconsistency
                    quality_metrics['ai_quality_score'] -= 0.2
                    quality_metrics['sentiment_alignment'] = 'inconsistent'
                
                # Check confidence levels
                avg_confidence = np.mean([insight.confidence for insight in sentiment_analysis])
                quality_metrics['ai_confidence'] = avg_confidence
                
                if avg_confidence < 0.6:
                    quality_metrics['ai_quality_score'] -= 0.1
                    quality_metrics['quality_warning'] = 'Low AI confidence in data'
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"AI quality assessment failed: {e}")
            return {}
    
    async def _analyze_correlations_with_ai(self, data: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """AI-enhanced correlation analysis"""
        if not self.grok4_client:
            return {}
        
        try:
            # Get AI correlation insights
            correlation_analysis = await self.grok4_client.analyze_correlation_patterns(symbols)
            
            # Extract meaningful insights
            ai_correlations = {
                'ai_correlation_matrix': correlation_analysis.get('correlation_matrix', {}),
                'diversification_insights': correlation_analysis.get('insights', {}),
                'recommendations': correlation_analysis.get('recommendations', []),
                'ai_detected_patterns': []
            }
            
            # Identify interesting patterns
            insights = correlation_analysis.get('insights', {})
            diversification_score = insights.get('diversification_score', 0.5)
            
            if diversification_score > 0.8:
                ai_correlations['ai_detected_patterns'].append({
                    'pattern': 'high_diversification',
                    'score': diversification_score,
                    'description': 'Portfolio shows excellent diversification'
                })
            elif diversification_score < 0.4:
                ai_correlations['ai_detected_patterns'].append({
                    'pattern': 'concentration_risk',
                    'score': diversification_score,
                    'description': 'Portfolio may have concentration risk'
                })
            
            return ai_correlations
            
        except Exception as e:
            logger.error(f"AI correlation analysis failed: {e}")
            return {}
    
    async def _combine_traditional_ai_data_analysis(self, traditional: Dict[str, Any],
                                                   ai_anomalies: Dict[str, Any],
                                                   ai_quality: Dict[str, Any],
                                                   ai_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Combine traditional data analysis with AI insights"""
        
        # Start with traditional analysis
        enhanced = traditional.copy()
        
        # Add AI enhancement metadata
        enhanced['ai_enhancement'] = {
            'enabled': True,
            'ai_anomalies_available': bool(ai_anomalies),
            'ai_quality_available': bool(ai_quality),
            'ai_correlations_available': bool(ai_correlations),
            'enhancement_timestamp': datetime.now().isoformat()
        }
        
        # Enhance quality validation with AI insights
        if 'comprehensive_analysis' in enhanced:
            analysis = enhanced['comprehensive_analysis']
            
            # Add AI quality assessment
            if ai_quality:
                analysis['ai_quality_assessment'] = ai_quality
                
                # Combine traditional and AI quality scores
                traditional_quality = analysis.get('quality_validation', {})
                if traditional_quality and 'score' in traditional_quality:
                    ai_score = ai_quality.get('ai_quality_score', 0.8)
                    traditional_score = traditional_quality['score']
                    
                    # Weighted combination (70% traditional, 30% AI)
                    combined_score = (traditional_score * 0.7) + (ai_score * 0.3)
                    analysis['combined_quality_score'] = combined_score
                    
                    if combined_score > traditional_score:
                        analysis['ai_quality_boost'] = True
                    elif combined_score < traditional_score:
                        analysis['ai_quality_warning'] = True
            
            # Add AI anomaly detection results
            if ai_anomalies:
                analysis['ai_anomaly_detection'] = ai_anomalies
                
                # Flag data issues based on AI anomalies
                total_anomalies = ai_anomalies.get('total_anomalies', 0)
                if total_anomalies > 0:
                    analysis['data_quality_alerts'] = analysis.get('data_quality_alerts', [])
                    analysis['data_quality_alerts'].append({
                        'type': 'ai_anomaly_detected',
                        'count': total_anomalies,
                        'severity': 'medium' if total_anomalies < 3 else 'high',
                        'description': f'AI detected {total_anomalies} data anomalies'
                    })
            
            # Enhance correlation analysis with AI insights
            if ai_correlations and 'correlation_analysis' in analysis:
                traditional_corr = analysis['correlation_analysis']
                traditional_corr['ai_correlation_insights'] = ai_correlations
                
                # Compare traditional vs AI correlations
                ai_patterns = ai_correlations.get('ai_detected_patterns', [])
                if ai_patterns:
                    traditional_corr['ai_pattern_insights'] = ai_patterns
        
        # Add AI recommendations section
        enhanced['ai_recommendations'] = []
        
        if ai_quality:
            ai_score = ai_quality.get('ai_quality_score', 0.8)
            if ai_score < 0.7:
                enhanced['ai_recommendations'].append({
                    'type': 'data_quality_improvement',
                    'priority': 'high',
                    'description': 'AI suggests improving data quality',
                    'ai_score': ai_score
                })
        
        if ai_anomalies:
            anomaly_count = ai_anomalies.get('total_anomalies', 0)
            if anomaly_count > 0:
                enhanced['ai_recommendations'].append({
                    'type': 'investigate_anomalies',
                    'priority': 'medium',
                    'description': f'Investigate {anomaly_count} AI-detected anomalies',
                    'anomaly_details': ai_anomalies
                })
        
        return enhanced
    
    async def enhance_outlier_detection_with_ai(self, data: Dict[str, Any], 
                                              symbols: List[str] = None) -> Dict[str, Any]:
        """Enhance traditional outlier detection with AI insights"""
        if not self.grok4_client:
            return await self.find_outliers(data)
        
        try:
            # Run traditional outlier detection
            traditional_outliers = await self.find_outliers(data)
            
            # Get AI anomaly detection
            if symbols:
                ai_anomalies = await self._detect_anomalies_with_ai(data, symbols)
                
                # Combine results
                enhanced_outliers = traditional_outliers.copy()
                enhanced_outliers['ai_anomalies'] = ai_anomalies
                
                # Cross-validate outliers with AI anomalies
                traditional_count = len(traditional_outliers.get('outliers', []))
                ai_count = ai_anomalies.get('total_anomalies', 0)
                
                enhanced_outliers['analysis_summary'] = {
                    'traditional_outliers': traditional_count,
                    'ai_anomalies': ai_count,
                    'validation_status': 'confirmed' if ai_count > 0 else 'traditional_only',
                    'confidence_boost': ai_count > 0
                }
                
                return enhanced_outliers
            else:
                return traditional_outliers
                
        except Exception as e:
            logger.error(f"Enhanced outlier detection failed: {e}")
            return await self.find_outliers(data)
    
    async def smart_data_validation(self, data: Dict[str, Any], 
                                  symbols: List[str], 
                                  validation_context: str = "trading") -> Dict[str, Any]:
        """Smart data validation combining multiple AI and traditional approaches"""
        logger.info(f"Starting smart data validation for {len(symbols)} symbols")
        
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'context': validation_context,
            'symbols_analyzed': symbols,
            'validation_summary': {},
            'recommendations': []
        }
        
        try:
            # Traditional comprehensive analysis
            traditional_analysis = await self.comprehensive_data_analysis(data, symbols)
            validation_results['traditional_analysis'] = traditional_analysis
            
            # AI-enhanced quality analysis
            if GROK4_AVAILABLE:
                ai_enhanced_analysis = await self.analyze_data_quality_ai_enhanced(data, symbols)
                validation_results['ai_enhanced_analysis'] = ai_enhanced_analysis
                
                # Create validation summary
                traditional_success = traditional_analysis.get('success', False)
                ai_enhancement_status = ai_enhanced_analysis.get('ai_enhancement_status', 'not_available')
                
                validation_results['validation_summary'] = {
                    'traditional_validation': 'passed' if traditional_success else 'failed',
                    'ai_enhancement': ai_enhancement_status,
                    'overall_quality': 'good',  # Will be calculated below
                    'confidence_level': 'medium'
                }
                
                # Calculate overall quality score
                if 'comprehensive_analysis' in ai_enhanced_analysis:
                    analysis = ai_enhanced_analysis['comprehensive_analysis']
                    combined_score = analysis.get('combined_quality_score')
                    
                    if combined_score is not None:
                        if combined_score >= 0.8:
                            validation_results['validation_summary']['overall_quality'] = 'excellent'
                            validation_results['validation_summary']['confidence_level'] = 'high'
                        elif combined_score >= 0.6:
                            validation_results['validation_summary']['overall_quality'] = 'good'
                            validation_results['validation_summary']['confidence_level'] = 'medium'
                        else:
                            validation_results['validation_summary']['overall_quality'] = 'needs_improvement'
                            validation_results['validation_summary']['confidence_level'] = 'low'
                        
                        validation_results['validation_summary']['combined_quality_score'] = combined_score
                
                # Generate smart recommendations
                recommendations = validation_results.get('ai_recommendations', [])
                if recommendations:
                    validation_results['recommendations'].extend(recommendations)
            
            else:
                validation_results['validation_summary'] = {
                    'traditional_validation': 'passed' if traditional_analysis.get('success') else 'failed',
                    'ai_enhancement': 'not_available',
                    'overall_quality': 'traditional_only',
                    'confidence_level': 'medium'
                }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Smart data validation failed: {e}")
            validation_results['error'] = str(e)
            validation_results['validation_summary']['overall_quality'] = 'error'
            return validation_results

# Global agent instance
data_analysis_agent = DataAnalysisAgent()
