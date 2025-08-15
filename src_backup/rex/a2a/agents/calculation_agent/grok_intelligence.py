"""
GROK Intelligence Module for Calculation Agent

This module provides real AI-powered decision making using the GROK model
for intelligent calculation strategy selection, method optimization, and
result interpretation.

Key Features:
- Real GROK model integration for decision making
- AI-powered method selection based on problem analysis
- Intelligent result interpretation and explanation
- Learning from calculation patterns and outcomes
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.strands.agent import Agent
from src.strands.models import Model
from .types import ComputationMethod, CalculationType, CalculationResult
from .utils import classify_calculation_type, assess_expression_complexity


logger = logging.getLogger(__name__)


class GrokIntelligence:
    """GROK-powered intelligent decision making for calculations"""
    
    def __init__(self, agent_instance):
        """Initialize with the main agent's GROK model"""
        self.agent = agent_instance
        self.model = agent_instance.agent.model if hasattr(agent_instance.agent, 'model') else None
        self.decision_history = []
        self.performance_patterns = {
            "successful_strategies": {},
            "failed_strategies": {},
            "problem_categories": {}
        }
    
    def analyze_calculation_problem(self, expression: str, variables: Dict[str, Any] = None, 
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use GROK to analyze the calculation problem and recommend approach
        
        Args:
            expression: Mathematical expression to analyze
            variables: Variable values and types
            context: Additional context about the calculation
            
        Returns:
            Dict with GROK's analysis and recommendations
        """
        try:
            # Create analysis prompt for GROK
            analysis_prompt = self._create_analysis_prompt(expression, variables, context)
            
            # Get GROK's analysis
            if self.model:
                response = self.model(analysis_prompt)
            else:
                # Fallback to agent processing
                response = self.agent.agent(analysis_prompt)
            
            # Parse GROK's response into structured recommendations
            parsed_analysis = self._parse_grok_response(str(response))
            
            # Store decision in history for learning
            decision_record = {
                "timestamp": datetime.now().isoformat(),
                "expression": expression,
                "variables": variables,
                "context": context,
                "grok_analysis": parsed_analysis,
                "recommended_method": parsed_analysis.get("recommended_method"),
                "reasoning": parsed_analysis.get("reasoning")
            }
            self.decision_history.append(decision_record)
            
            logger.info(f"GROK analysis completed for: {expression}")
            return parsed_analysis
            
        except Exception as e:
            logger.error(f"GROK analysis failed: {e}")
            # Fallback to rule-based analysis
            return self._fallback_analysis(expression, variables, context)
    
    def select_computation_method(self, expression: str, variables: Dict[str, Any] = None,
                                requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use GROK to intelligently select the best computation method
        
        Args:
            expression: Mathematical expression
            variables: Variables and their types/values
            requirements: Performance/accuracy requirements
            
        Returns:
            Dict with selected method and reasoning
        """
        try:
            # Create method selection prompt
            selection_prompt = self._create_method_selection_prompt(expression, variables, requirements)
            
            # Get GROK's recommendation
            if self.model:
                response = self.model(selection_prompt)
            else:
                response = self.agent.agent(selection_prompt)
            
            # Parse method selection
            method_selection = self._parse_method_selection(str(response))
            
            # Validate the selection
            selected_method = self._validate_method_selection(method_selection, expression)
            
            logger.info(f"GROK selected method: {selected_method['method']} for {expression}")
            return selected_method
            
        except Exception as e:
            logger.error(f"GROK method selection failed: {e}")
            # Fallback to rule-based selection
            return self._fallback_method_selection(expression, variables, requirements)
    
    def interpret_calculation_result(self, result: Dict[str, Any], 
                                   original_expression: str,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use GROK to interpret and explain calculation results
        
        Args:
            result: Calculation result to interpret
            original_expression: Original mathematical expression
            context: Additional context for interpretation
            
        Returns:
            Dict with GROK's interpretation and explanation
        """
        try:
            # Create interpretation prompt
            interpretation_prompt = self._create_interpretation_prompt(result, original_expression, context)
            
            # Get GROK's interpretation
            if self.model:
                response = self.model(interpretation_prompt)
            else:
                response = self.agent.agent(interpretation_prompt)
            
            # Parse interpretation
            interpretation = self._parse_interpretation(str(response), result)
            
            logger.info(f"GROK interpreted result for: {original_expression}")
            return interpretation
            
        except Exception as e:
            logger.error(f"GROK interpretation failed: {e}")
            return self._fallback_interpretation(result, original_expression)
    
    def learn_from_calculation(self, expression: str, method_used: str, 
                             result: Dict[str, Any], success: bool,
                             performance_metrics: Dict[str, Any] = None) -> None:
        """
        Learn from calculation outcomes to improve future decisions
        
        Args:
            expression: Mathematical expression that was calculated
            method_used: Computation method that was used
            result: Calculation result
            success: Whether the calculation was successful
            performance_metrics: Performance data (time, accuracy, etc.)
        """
        try:
            # Categorize the problem
            calc_type = classify_calculation_type(expression)
            complexity = assess_expression_complexity(expression)
            
            # Update performance patterns
            problem_key = f"{calc_type.value}_{complexity}"
            
            if problem_key not in self.performance_patterns["problem_categories"]:
                self.performance_patterns["problem_categories"][problem_key] = {
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "method_performance": {}
                }
            
            category = self.performance_patterns["problem_categories"][problem_key]
            category["total_attempts"] += 1
            
            if success:
                category["successful_attempts"] += 1
                
                # Track successful strategies
                if method_used not in self.performance_patterns["successful_strategies"]:
                    self.performance_patterns["successful_strategies"][method_used] = []
                
                self.performance_patterns["successful_strategies"][method_used].append({
                    "expression": expression,
                    "problem_type": problem_key,
                    "performance": performance_metrics,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                # Track failed strategies
                if method_used not in self.performance_patterns["failed_strategies"]:
                    self.performance_patterns["failed_strategies"][method_used] = []
                
                self.performance_patterns["failed_strategies"][method_used].append({
                    "expression": expression,
                    "problem_type": problem_key,
                    "error": result.get("error"),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Update method performance for this problem category
            if method_used not in category["method_performance"]:
                category["method_performance"][method_used] = {
                    "attempts": 0,
                    "successes": 0,
                    "average_time": 0.0,
                    "success_rate": 0.0
                }
            
            method_perf = category["method_performance"][method_used]
            method_perf["attempts"] += 1
            if success:
                method_perf["successes"] += 1
            
            method_perf["success_rate"] = method_perf["successes"] / method_perf["attempts"]
            
            # Update average time if available
            if performance_metrics and "computation_time" in performance_metrics:
                old_avg = method_perf["average_time"]
                new_time = performance_metrics["computation_time"]
                method_perf["average_time"] = (old_avg * (method_perf["attempts"] - 1) + new_time) / method_perf["attempts"]
            
            logger.info(f"Learning updated for {method_used} on {problem_key}")
            
        except Exception as e:
            logger.error(f"Learning update failed: {e}")
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights from learned performance patterns"""
        try:
            insights = {
                "total_decisions": len(self.decision_history),
                "problem_categories_learned": len(self.performance_patterns["problem_categories"]),
                "successful_strategies_count": sum(len(v) for v in self.performance_patterns["successful_strategies"].values()),
                "failed_strategies_count": sum(len(v) for v in self.performance_patterns["failed_strategies"].values()),
                "recommendations": []
            }
            
            # Generate recommendations based on learned patterns
            for problem_type, category_data in self.performance_patterns["problem_categories"].items():
                if category_data["total_attempts"] > 5:  # Only analyze categories with sufficient data
                    best_method = None
                    best_success_rate = 0
                    
                    for method, perf in category_data["method_performance"].items():
                        if perf["success_rate"] > best_success_rate:
                            best_success_rate = perf["success_rate"]
                            best_method = method
                    
                    if best_method and best_success_rate > 0.8:
                        insights["recommendations"].append({
                            "problem_type": problem_type,
                            "recommended_method": best_method,
                            "success_rate": best_success_rate,
                            "sample_size": category_data["total_attempts"]
                        })
            
            return insights
            
        except Exception as e:
            logger.error(f"Performance insights generation failed: {e}")
            return {"error": str(e)}
    
    def analyze_financial_problem(self, problem_type: str, data_description: Dict[str, Any],
                                 requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use GROK to analyze financial analysis problems and recommend approach
        
        Args:
            problem_type: Type of financial analysis (correlation, risk_metrics, backtesting, etc.)
            data_description: Description of available data
            requirements: Analysis requirements and constraints
            
        Returns:
            Dict with GROK's analysis and recommendations
        """
        try:
            if not self.model:
                return self._fallback_financial_analysis(problem_type, data_description, requirements)
            
            # Create specialized prompt for financial analysis
            prompt = self._create_financial_analysis_prompt(problem_type, data_description, requirements)
            
            # Get GROK's financial analysis
            response = self.model.query(prompt)
            analysis = self._parse_financial_grok_response(response.text if hasattr(response, 'text') else str(response))
            
            # Store decision for learning
            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "problem_type": "financial_analysis",
                "analysis_type": problem_type,
                "data_description": data_description,
                "requirements": requirements,
                "grok_analysis": analysis,
                "decision_type": "financial_method_selection"
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Financial problem analysis failed: {e}")
            return self._fallback_financial_analysis(problem_type, data_description, requirements)
    
    def recommend_financial_strategy(self, market_conditions: Dict[str, Any],
                                   available_tools: List[str],
                                   risk_tolerance: str = "medium") -> Dict[str, Any]:
        """
        Use GROK to recommend FX-crypto correlation trading strategy
        
        Args:
            market_conditions: Current market state and correlations
            available_tools: Available financial analysis tools
            risk_tolerance: Risk tolerance level (low, medium, high)
            
        Returns:
            Dict with strategy recommendations
        """
        try:
            if not self.model:
                return self._fallback_strategy_recommendation(market_conditions, available_tools, risk_tolerance)
            
            # Create strategy recommendation prompt
            prompt = self._create_strategy_prompt(market_conditions, available_tools, risk_tolerance)
            
            # Get GROK's strategy recommendation
            response = self.model.query(prompt)
            strategy = self._parse_strategy_grok_response(response.text if hasattr(response, 'text') else str(response))
            
            # Store recommendation for learning
            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "problem_type": "strategy_recommendation",
                "market_conditions": market_conditions,
                "available_tools": available_tools,
                "risk_tolerance": risk_tolerance,
                "grok_strategy": strategy,
                "decision_type": "trading_strategy_selection"
            })
            
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy recommendation failed: {e}")
            return self._fallback_strategy_recommendation(market_conditions, available_tools, risk_tolerance)
    
    def _create_analysis_prompt(self, expression: str, variables: Dict[str, Any], 
                               context: Dict[str, Any]) -> str:
        """Create a prompt for GROK to analyze the calculation problem"""
        prompt = f"""Analyze this mathematical calculation problem:

Expression: {expression}
Variables: {variables or 'None'}
Context: {context or 'None'}

Please analyze:
1. What type of mathematical problem is this? (arithmetic, algebraic, calculus, statistical, etc.)
2. What is the complexity level? (simple, moderate, complex, very complex)
3. What computation method would work best? (symbolic for exact results, numeric for performance, hybrid for verification)
4. Are there any special considerations or potential issues?
5. What accuracy vs speed trade-offs should be considered?

Provide your analysis in a structured format with clear recommendations."""
        
        return prompt
    
    def _create_method_selection_prompt(self, expression: str, variables: Dict[str, Any], 
                                      requirements: Dict[str, Any]) -> str:
        """Create a prompt for GROK to select computation method"""
        prompt = f"""Select the best computation method for this calculation:

Expression: {expression}
Variables: {variables or 'None'}
Requirements: {requirements or 'None'}

Available methods:
1. SYMBOLIC - Uses SymPy for exact symbolic computation (high accuracy, slower)
2. NUMERIC - Uses NumPy/SciPy for numerical computation (fast, floating-point precision)
3. HYBRID - Uses both methods with cross-verification (highest reliability, slower)

Consider:
- Accuracy requirements
- Performance needs
- Problem complexity
- Variable types and values

Recommend ONE method and explain why it's the best choice for this specific problem."""
        
        return prompt
    
    def _create_interpretation_prompt(self, result: Dict[str, Any], expression: str, 
                                    context: Dict[str, Any]) -> str:
        """Create a prompt for GROK to interpret calculation results"""
        prompt = f"""Interpret this calculation result:

Original Expression: {expression}
Result: {result}
Context: {context or 'None'}

Please provide:
1. A clear explanation of what the result means
2. Whether the result appears reasonable and correct
3. Any potential issues or concerns with the result
4. Practical interpretation in plain language
5. Confidence level in the result (high/medium/low)

Focus on making the mathematical result understandable and actionable."""
        
        return prompt
    
    def _parse_grok_response(self, response: str) -> Dict[str, Any]:
        """Parse GROK's analysis response into structured data"""
        try:
            # Try to extract structured information from GROK's response
            analysis = {
                "raw_response": response,
                "problem_type": "unknown",
                "complexity": "moderate",
                "recommended_method": "numeric",
                "reasoning": response[:200] + "..." if len(response) > 200 else response,
                "confidence": 0.7
            }
            
            response_lower = response.lower()
            
            # Extract problem type
            if "symbolic" in response_lower or "algebraic" in response_lower or "calculus" in response_lower:
                analysis["problem_type"] = "symbolic"
                analysis["recommended_method"] = "symbolic"
            elif "statistical" in response_lower or "numerical" in response_lower:
                analysis["problem_type"] = "statistical"
                analysis["recommended_method"] = "numeric"
            elif "arithmetic" in response_lower or "simple" in response_lower:
                analysis["problem_type"] = "arithmetic"
                analysis["recommended_method"] = "numeric"
            
            # Extract complexity
            if "complex" in response_lower or "difficult" in response_lower:
                analysis["complexity"] = "complex"
            elif "simple" in response_lower or "basic" in response_lower:
                analysis["complexity"] = "simple"
            
            # Extract method recommendation
            if "hybrid" in response_lower and "recommend" in response_lower:
                analysis["recommended_method"] = "hybrid"
            elif "symbolic" in response_lower and "recommend" in response_lower:
                analysis["recommended_method"] = "symbolic"
            elif "numeric" in response_lower and "recommend" in response_lower:
                analysis["recommended_method"] = "numeric"
            
            # Extract confidence indicators
            if "confident" in response_lower or "certain" in response_lower:
                analysis["confidence"] = 0.9
            elif "uncertain" in response_lower or "unclear" in response_lower:
                analysis["confidence"] = 0.5
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse GROK response: {e}")
            return {
                "raw_response": response,
                "problem_type": "unknown",
                "complexity": "moderate", 
                "recommended_method": "numeric",
                "reasoning": "Failed to parse GROK response",
                "confidence": 0.5,
                "parse_error": str(e)
            }
    
    def _parse_method_selection(self, response: str) -> Dict[str, Any]:
        """Parse GROK's method selection response"""
        response_lower = response.lower()
        
        # Default selection
        selection = {
            "method": "numeric",
            "reasoning": response[:300] if len(response) > 300 else response,
            "confidence": 0.7
        }
        
        # Parse method selection
        if "symbolic" in response_lower and ("recommend" in response_lower or "choose" in response_lower):
            selection["method"] = "symbolic"
        elif "hybrid" in response_lower and ("recommend" in response_lower or "choose" in response_lower):
            selection["method"] = "hybrid"
        elif "numeric" in response_lower and ("recommend" in response_lower or "choose" in response_lower):
            selection["method"] = "numeric"
        
        # Extract confidence
        if "highly recommend" in response_lower or "definitely" in response_lower:
            selection["confidence"] = 0.95
        elif "recommend" in response_lower or "suggest" in response_lower:
            selection["confidence"] = 0.8
        elif "might" in response_lower or "could" in response_lower:
            selection["confidence"] = 0.6
        
        return selection
    
    def _parse_interpretation(self, response: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GROK's result interpretation"""
        interpretation = {
            "explanation": response,
            "reasonableness": "unknown",
            "confidence_assessment": "medium",
            "concerns": [],
            "practical_meaning": "",
            "grok_confidence": 0.7
        }
        
        response_lower = response.lower()
        
        # Extract reasonableness assessment
        if "reasonable" in response_lower or "correct" in response_lower:
            interpretation["reasonableness"] = "reasonable"
        elif "unreasonable" in response_lower or "incorrect" in response_lower:
            interpretation["reasonableness"] = "unreasonable"
        
        # Extract confidence assessment
        if "high confidence" in response_lower or "very confident" in response_lower:
            interpretation["confidence_assessment"] = "high"
            interpretation["grok_confidence"] = 0.9
        elif "low confidence" in response_lower or "uncertain" in response_lower:
            interpretation["confidence_assessment"] = "low"
            interpretation["grok_confidence"] = 0.4
        
        # Extract concerns
        if "concern" in response_lower or "issue" in response_lower:
            concerns_section = response_lower.split("concern")[-1][:200]
            interpretation["concerns"].append(concerns_section)
        
        return interpretation
    
    def _validate_method_selection(self, selection: Dict[str, Any], expression: str) -> Dict[str, Any]:
        """Validate and potentially override GROK's method selection"""
        method = selection.get("method", "numeric")
        
        # Validate method exists
        valid_methods = ["symbolic", "numeric", "hybrid"]
        if method not in valid_methods:
            logger.warning(f"Invalid method selected by GROK: {method}, defaulting to numeric")
            selection["method"] = "numeric"
            selection["validation_override"] = True
        
        return selection
    
    def _fallback_analysis(self, expression: str, variables: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when GROK is unavailable"""
        calc_type = classify_calculation_type(expression)
        complexity = assess_expression_complexity(expression)
        
        # Rule-based method selection
        if calc_type in [CalculationType.ALGEBRAIC, CalculationType.CALCULUS]:
            method = "symbolic"
        elif complexity > 10:
            method = "numeric"
        else:
            method = "hybrid"
        
        return {
            "problem_type": calc_type.value,
            "complexity": "moderate" if complexity < 10 else "complex",
            "recommended_method": method,
            "reasoning": "Fallback rule-based analysis (GROK unavailable)",
            "confidence": 0.6,
            "fallback": True
        }
    
    def _fallback_method_selection(self, expression: str, variables: Dict[str, Any],
                                 requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback method selection when GROK is unavailable"""
        calc_type = classify_calculation_type(expression)
        
        if calc_type == CalculationType.STATISTICAL:
            method = "numeric"
        elif calc_type in [CalculationType.ALGEBRAIC, CalculationType.CALCULUS]:
            method = "symbolic"
        else:
            method = "numeric"
        
        return {
            "method": method,
            "reasoning": f"Rule-based selection for {calc_type.value} problems",
            "confidence": 0.6,
            "fallback": True
        }
    
    def _fallback_interpretation(self, result: Dict[str, Any], expression: str) -> Dict[str, Any]:
        """Fallback interpretation when GROK is unavailable"""
        return {
            "explanation": f"Calculation of '{expression}' completed",
            "reasonableness": "unknown",
            "confidence_assessment": "medium",
            "concerns": [],
            "practical_meaning": f"Result: {result.get('result', 'N/A')}",
            "grok_confidence": 0.5,
            "fallback": True
        }
    
    # FINANCIAL ANALYSIS HELPER METHODS
    
    def _create_financial_analysis_prompt(self, problem_type: str, data_description: Dict[str, Any],
                                        requirements: Dict[str, Any]) -> str:
        """Create GROK prompt for financial analysis problems"""
        prompt = f"""Analyze this financial analysis problem for FX-crypto correlation trading:

Problem Type: {problem_type}
Available Data: {data_description}
Requirements: {requirements or 'None specified'}

Financial Analysis Types Available:
1. CORRELATION_ANALYSIS - Correlation matrices, rolling correlations, regime detection
2. RISK_METRICS - Sharpe ratio, maximum drawdown, volatility analysis
3. LEAD_LAG_ANALYSIS - Identify which asset leads/follows in price movements
4. BACKTESTING - Test correlation-based trading strategies
5. SIGNAL_GENERATION - Generate early warning signals based on correlations

Please recommend:
1. Which financial analysis method(s) would be most appropriate
2. What parameters or settings should be used
3. What insights we can expect from the analysis
4. Any limitations or considerations for this approach
5. How this fits into a broader FX-crypto correlation trading strategy

Focus on practical trading applications rather than theoretical analysis."""
        
        return prompt
    
    def _create_strategy_prompt(self, market_conditions: Dict[str, Any], available_tools: List[str],
                              risk_tolerance: str) -> str:
        """Create GROK prompt for trading strategy recommendations"""
        prompt = f"""Recommend a FX-crypto correlation trading strategy:

Current Market Conditions: {market_conditions}
Available Analysis Tools: {available_tools}  
Risk Tolerance: {risk_tolerance}

Context:
- We're looking for early signals in FX markets that predict crypto movements
- Strategy should use correlation analysis to identify opportunities
- Need practical entry/exit rules and risk management

Please recommend:
1. Which FX-crypto pairs to monitor based on current correlations
2. What correlation thresholds to use for signal generation
3. Optimal time horizons and analysis windows
4. Risk management parameters (stop loss, position sizing)
5. Expected performance characteristics and limitations
6. Market conditions where this strategy works best/worst

Focus on actionable, implementable trading rules."""
        
        return prompt
    
    def _parse_financial_grok_response(self, response: str) -> Dict[str, Any]:
        """Parse GROK's financial analysis response"""
        try:
            response_lower = response.lower()
            
            analysis = {
                "raw_response": response,
                "recommended_methods": [],
                "parameters": {},
                "expected_insights": [],
                "limitations": [],
                "confidence": 0.7
            }
            
            # Extract recommended methods
            if "correlation" in response_lower:
                analysis["recommended_methods"].append("correlation_analysis")
            if "risk" in response_lower or "sharpe" in response_lower or "drawdown" in response_lower:
                analysis["recommended_methods"].append("risk_metrics")
            if "lead" in response_lower or "lag" in response_lower:
                analysis["recommended_methods"].append("lead_lag_analysis")
            if "backtest" in response_lower:
                analysis["recommended_methods"].append("backtesting")
            if "signal" in response_lower or "warning" in response_lower:
                analysis["recommended_methods"].append("signal_generation")
            
            # Extract confidence indicators
            if "highly recommend" in response_lower or "strongly suggest" in response_lower:
                analysis["confidence"] = 0.9
            elif "might consider" in response_lower or "could try" in response_lower:
                analysis["confidence"] = 0.6
            
            # Extract practical insights
            analysis["practical_summary"] = response[:500] if len(response) > 500 else response
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse financial GROK response: {e}")
            return {
                "raw_response": response,
                "recommended_methods": ["correlation_analysis"],
                "confidence": 0.5,
                "parse_error": str(e),
                "fallback": True
            }
    
    def _parse_strategy_grok_response(self, response: str) -> Dict[str, Any]:
        """Parse GROK's strategy recommendation response"""
        try:
            response_lower = response.lower()
            
            strategy = {
                "raw_response": response,
                "recommended_pairs": [],
                "signal_parameters": {},
                "risk_management": {},
                "expected_performance": {},
                "market_conditions": [],
                "confidence": 0.7
            }
            
            # Extract pair recommendations
            common_pairs = ["eurusd", "gbpusd", "usdjpy", "btcusd", "ethusd", "btceur"]
            for pair in common_pairs:
                if pair.replace("usd", "/usd").replace("eur", "/eur") in response_lower or pair in response_lower:
                    strategy["recommended_pairs"].append(pair.upper())
            
            # Extract signal parameters
            import re
            correlation_thresholds = re.findall(r'correlation.*?(\d+\.?\d*)', response_lower)
            if correlation_thresholds:
                try:
                    strategy["signal_parameters"]["correlation_threshold"] = float(correlation_thresholds[0])
                except:
                    pass
            
            # Extract confidence
            if "high confidence" in response_lower or "strong recommendation" in response_lower:
                strategy["confidence"] = 0.9
            elif "moderate" in response_lower or "reasonable" in response_lower:
                strategy["confidence"] = 0.7
            elif "low confidence" in response_lower or "risky" in response_lower:
                strategy["confidence"] = 0.5
            
            # Strategy summary
            strategy["strategy_summary"] = response[:600] if len(response) > 600 else response
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to parse strategy GROK response: {e}")
            return {
                "raw_response": response,
                "recommended_pairs": ["EURUSD", "BTCUSD"],
                "confidence": 0.5,
                "parse_error": str(e),
                "fallback": True
            }
    
    def _fallback_financial_analysis(self, problem_type: str, data_description: Dict[str, Any],
                                   requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback financial analysis when GROK is unavailable"""
        method_mapping = {
            "correlation": ["correlation_analysis"],
            "risk": ["risk_metrics", "volatility_metrics"],
            "backtest": ["backtesting", "risk_metrics"],
            "signal": ["signal_generation", "correlation_analysis"],
            "lead_lag": ["lead_lag_analysis"]
        }
        
        recommended_methods = method_mapping.get(problem_type, ["correlation_analysis"])
        
        return {
            "recommended_methods": recommended_methods,
            "parameters": {"window_size": 20, "threshold": 0.6},
            "confidence": 0.6,
            "reasoning": "Rule-based recommendation (GROK unavailable)",
            "fallback": True
        }
    
    def _fallback_strategy_recommendation(self, market_conditions: Dict[str, Any],
                                        available_tools: List[str], risk_tolerance: str) -> Dict[str, Any]:
        """Fallback strategy recommendation when GROK is unavailable"""
        risk_params = {
            "low": {"correlation_threshold": 0.8, "position_size": 0.005},
            "medium": {"correlation_threshold": 0.6, "position_size": 0.01},
            "high": {"correlation_threshold": 0.4, "position_size": 0.02}
        }
        
        params = risk_params.get(risk_tolerance, risk_params["medium"])
        
        return {
            "recommended_pairs": ["EURUSD", "BTCUSD", "GBPUSD", "ETHUSD"],
            "signal_parameters": params,
            "risk_management": {"stop_loss": 0.02, "take_profit": 0.04},
            "confidence": 0.6,
            "strategy_summary": "Conservative correlation-based strategy with standard parameters",
            "fallback": True
        }