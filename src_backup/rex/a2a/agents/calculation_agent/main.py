"""
Main Calculation Agent - A2A Orchestrator

This is the main A2A agent that orchestrates all calculation sub-skills:
- Integrates with A2A protocol and Strands MCP
- Coordinates between symbolic, numeric, and verification sub-skills
- Handles A2A message routing and responses
- Provides unified interface for calculation services

Architecture:
- A2A Agent: Handles inter-agent communication
- Strands Integration: Provides tools through MCP
- Sub-skill Orchestration: Coordinates different calculation methods
- Performance Optimization: Auto-selects best methods
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import time

from ..memory_strands_agent import MemoryStrandsAgent
from ...protocols.a2a_protocol import A2AMessage, MessageType
from src.strands.tools import tool

# Import sub-skills
from .symbolic_skill import SymbolicComputationSkill
from .numeric_skill import NumericComputationSkill
from .verification_skill import VerificationSkill
from .financial_skill import FinancialAnalysisSkill
from .grok_intelligence import GrokIntelligence
from .types import ComputationMethod, CalculationType, CalculationResult
from .utils import (
    auto_select_computation_method, classify_calculation_type,
    estimate_computation_cost, format_result_for_display
)


logger = logging.getLogger(__name__)


class CalculationAgent(MemoryStrandsAgent):
    """
    Main A2A Calculation Agent with Hybrid Computation Sub-skills
    
    This agent orchestrates multiple calculation sub-skills through Strands MCP
    while maintaining A2A protocol compatibility for inter-agent communication.
    
    Sub-skills managed:
    1. Symbolic Computation (SymPy-based)
    2. Numeric Computation (NumPy/SciPy-based)  
    3. Verification & Cross-checking
    4. Step-by-step Reasoning
    5. A2A Coordination
    """
    
    def __init__(self):
        super().__init__(
            agent_id="calculation-agent-001",
            agent_type="hybrid_calculation",
            capabilities=[
                "symbolic_computation",
                "numeric_computation", 
                "calculation_verification",
                "hybrid_computation",
                "step_by_step_reasoning",
                "a2a_coordination",
                "financial_calculations",
                "statistical_analysis",
                "matrix_operations",
                "optimization",
                "precision_arithmetic"
            ],
            model_provider="grok4"
        )
        
        # Initialize sub-skills
        self.symbolic_skill = SymbolicComputationSkill()
        self.numeric_skill = NumericComputationSkill()
        self.verification_skill = VerificationSkill()
        self.financial_skill = FinancialAnalysisSkill()
        
        # Initialize GROK intelligence for AI decision making
        self.grok_intelligence = GrokIntelligence(self)
        
        # Agent state
        self.active_calculations: Dict[str, Dict] = {}
        self.coordination_sessions: Dict[str, Dict] = {}
        
        # Performance metrics
        self.agent_metrics = {
            "total_requests": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "average_response_time": 0.0,
            "sub_skill_usage": {
                "symbolic": 0,
                "numeric": 0,
                "hybrid": 0,
                "verification": 0
            },
            "method_success_rates": {
                "symbolic": 0.0,
                "numeric": 0.0,
                "hybrid": 0.0
            }
        }
        
        logger.info("Calculation Agent initialized with hybrid computation sub-skills")
    
    def _create_tools(self) -> List:
        """Create all calculation tools using sub-skills"""
        return [
            # Main orchestration tools
            self.calculate_with_auto_method,
            self.calculate_with_hybrid_verification,
            
            # Direct sub-skill access
            self.calculate_symbolic,
            self.calculate_numeric,
            self.verify_calculation_result,
            
            # Specialized calculation tools
            self.solve_equation,
            self.perform_calculus_operation,
            self.analyze_statistical_data,
            self.perform_matrix_calculation,
            
            # Financial analysis tools
            self.analyze_correlation,
            self.calculate_rolling_correlations,
            self.analyze_lead_lag_relationships,
            self.calculate_sharpe_ratio,
            self.calculate_maximum_drawdown,
            self.calculate_volatility_metrics,
            self.calculate_information_ratio,
            self.calculate_calmar_ratio,
            self.generate_early_warning_signals,
            self.backtest_correlation_strategy,
            
            # Agent coordination tools
            self.coordinate_distributed_calculation,
            self.request_peer_verification,
            
            # Utility and reporting tools
            self.optimize_calculation_method,
            self.get_calculation_report,
            self.explain_calculation_steps
        ]
    
    # MAIN ORCHESTRATION TOOLS
    
    @tool
    def calculate_with_auto_method(self, expression: str, variables: Dict[str, Any] = None,
                                  preferred_method: str = "auto") -> Dict[str, Any]:
        """
        Main calculation interface with automatic method selection
        
        Args:
            expression: Mathematical expression to calculate
            variables: Variable substitutions
            preferred_method: Preferred computation method (auto, symbolic, numeric, hybrid)
            
        Returns:
            Dict with calculation result and metadata
        """
        calculation_id = f"calc_{datetime.now().isoformat()}"
        start_time = time.time()
        
        try:
            logger.info(f"Starting calculation {calculation_id}: {expression}")
            
            # Store active calculation
            self.active_calculations[calculation_id] = {
                "expression": expression,
                "variables": variables,
                "preferred_method": preferred_method,
                "start_time": start_time,
                "status": "processing"
            }
            
            # Use GROK intelligence to select method
            if preferred_method == "auto":
                # Get GROK's analysis and recommendation
                grok_analysis = self.grok_intelligence.analyze_calculation_problem(
                    expression, variables, {"request_type": "auto_calculation"}
                )
                method_selection = self.grok_intelligence.select_computation_method(
                    expression, variables, {"accuracy_priority": "medium", "speed_priority": "medium"}
                )
                
                try:
                    selected_method = ComputationMethod(method_selection.get("method", "numeric"))
                except ValueError:
                    selected_method = ComputationMethod.NUMERIC
                    
                # Store GROK's reasoning in calculation record
                self.active_calculations[calculation_id]["grok_analysis"] = grok_analysis
                self.active_calculations[calculation_id]["grok_method_selection"] = method_selection
            else:
                try:
                    selected_method = ComputationMethod(preferred_method)
                except ValueError:
                    selected_method = ComputationMethod.NUMERIC
            
            # Estimate computation cost
            cost_estimate = estimate_computation_cost(expression, variables)
            
            # Execute calculation based on selected method
            if selected_method == ComputationMethod.SYMBOLIC:
                result = self.symbolic_skill.calculate_symbolic(expression, variables)
                self.agent_metrics["sub_skill_usage"]["symbolic"] += 1
                
            elif selected_method == ComputationMethod.NUMERIC:
                # Convert variables to numeric only
                numeric_vars = {k: v for k, v in (variables or {}).items() 
                              if isinstance(v, (int, float, list))}
                result = self.numeric_skill.calculate_numeric(expression, numeric_vars)
                self.agent_metrics["sub_skill_usage"]["numeric"] += 1
                
            elif selected_method == ComputationMethod.HYBRID:
                result = self._execute_hybrid_calculation(expression, variables)
                self.agent_metrics["sub_skill_usage"]["hybrid"] += 1
                
            else:  # AUTO - choose best method
                result = self._execute_auto_calculation(expression, variables, cost_estimate)
            
            # Update calculation record
            computation_time = time.time() - start_time
            self.active_calculations[calculation_id].update({
                "status": "completed" if result.get("success") else "failed",
                "result": result,
                "computation_time": computation_time,
                "method_used": result.get("method", "unknown"),
                "selected_method": selected_method.value
            })
            
            # Update agent metrics
            self._update_agent_metrics(result.get("success", False), computation_time, selected_method)
            
            # Let GROK learn from this calculation outcome
            self.grok_intelligence.learn_from_calculation(
                expression=expression,
                method_used=selected_method.value,
                result=result,
                success=result.get("success", False),
                performance_metrics={
                    "computation_time": computation_time,
                    "method_selected": selected_method.value,
                    "cost_estimate": cost_estimate
                }
            )
            
            # Get GROK's interpretation of the result
            if result.get("success"):
                grok_interpretation = self.grok_intelligence.interpret_calculation_result(
                    result, expression, {"calculation_id": calculation_id}
                )
                result["grok_interpretation"] = grok_interpretation
            
            # Enhance result with metadata
            result["calculation_id"] = calculation_id
            result["method_selection"] = selected_method.value
            result["cost_estimate"] = cost_estimate
            result["total_computation_time"] = computation_time
            
            logger.info(f"Calculation {calculation_id} completed: {result.get('success')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Calculation {calculation_id} failed: {e}")
            
            # Update failed calculation record
            if calculation_id in self.active_calculations:
                self.active_calculations[calculation_id].update({
                    "status": "failed",
                    "error": str(e),
                    "computation_time": time.time() - start_time
                })
            
            self._update_agent_metrics(False, time.time() - start_time, ComputationMethod.AUTO)
            
            return {
                "success": False,
                "calculation_id": calculation_id,
                "error": str(e),
                "expression": expression,
                "variables": variables
            }
    
    @tool
    def calculate_with_hybrid_verification(self, expression: str, variables: Dict[str, Any] = None,
                                         tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Calculate using both symbolic and numeric methods with cross-verification
        
        Args:
            expression: Mathematical expression
            variables: Variable substitutions
            tolerance: Tolerance for verification
            
        Returns:
            Dict with calculation results and verification
        """
        start_time = time.time()
        
        try:
            logger.info(f"Hybrid calculation with verification: {expression}")
            
            # Perform both symbolic and numeric calculations
            symbolic_result = self.symbolic_skill.calculate_symbolic(expression, variables)
            
            # Prepare numeric variables
            numeric_vars = {k: v for k, v in (variables or {}).items() 
                          if isinstance(v, (int, float, list))}
            numeric_result = self.numeric_skill.calculate_numeric(expression, numeric_vars)
            
            # Cross-verify results
            verification_result = None
            if symbolic_result.get("success") and numeric_result.get("success"):
                verification_result = self.verification_skill.verify_calculation(
                    symbolic_result, numeric_result, tolerance
                )
            
            # Determine best result
            final_result = self._select_best_result(symbolic_result, numeric_result, verification_result)
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "hybrid_verification",
                "computation_time": computation_time,
                "symbolic_result": symbolic_result,
                "numeric_result": numeric_result,
                "verification": verification_result,
                "final_result": final_result,
                "recommended_answer": final_result.get("result"),
                "confidence": verification_result.get("confidence", 0.5) if verification_result else 0.5
            }
            
        except Exception as e:
            logger.error(f"Hybrid verification calculation failed: {e}")
            return {
                "success": False,
                "method": "hybrid_verification",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    # DIRECT SUB-SKILL ACCESS
    
    @tool
    def calculate_symbolic(self, expression: str, variables: Dict[str, Any] = None, 
                          operation: str = "evaluate") -> Dict[str, Any]:
        """Direct access to symbolic computation sub-skill"""
        return self.symbolic_skill.calculate_symbolic(expression, variables, operation)
    
    @tool
    def calculate_numeric(self, expression: str, variables: Dict[str, Union[float, List]] = None,
                         method: str = "numpy") -> Dict[str, Any]:
        """Direct access to numeric computation sub-skill"""
        return self.numeric_skill.calculate_numeric(expression, variables, method)
    
    @tool
    def verify_calculation_result(self, result1: Dict[str, Any], result2: Dict[str, Any],
                                tolerance: float = 1e-10) -> Dict[str, Any]:
        """Direct access to verification sub-skill"""
        return self.verification_skill.verify_calculation(result1, result2, tolerance)
    
    # SPECIALIZED CALCULATION TOOLS
    
    @tool
    def solve_equation(self, equation: str, solve_for: str = None, domain: str = "complex") -> Dict[str, Any]:
        """Solve algebraic equations"""
        return self.symbolic_skill.solve_equation(equation, solve_for, domain)
    
    @tool
    def perform_calculus_operation(self, expression: str, operation: str, variable: str = None,
                                  limits: List = None) -> Dict[str, Any]:
        """Perform calculus operations (derivatives, integrals)"""
        return self.symbolic_skill.perform_calculus(expression, operation, variable, limits)
    
    @tool
    def analyze_statistical_data(self, data: List[float], analysis_type: str = "descriptive",
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """Perform statistical analysis"""
        return self.numeric_skill.perform_statistical_analysis(data, analysis_type, confidence_level)
    
    @tool
    def perform_matrix_calculation(self, operation: str, matrices: Dict[str, List[List[float]]],
                                 **kwargs) -> Dict[str, Any]:
        """Perform matrix operations"""
        return self.numeric_skill.perform_matrix_operations(operation, matrices, **kwargs)
    
    # FINANCIAL ANALYSIS TOOLS
    
    @tool
    def analyze_correlation(self, asset_data: Dict[str, List[float]], method: str = "pearson") -> Dict[str, Any]:
        """Calculate correlation matrix between multiple FX and crypto pairs"""
        return self.financial_skill.calculate_correlation_matrix(asset_data, method)
    
    @tool
    def calculate_rolling_correlations(self, asset1_data: List[float], asset2_data: List[float],
                                     asset1_name: str, asset2_name: str, window_size: int = 20,
                                     step_size: int = 1) -> Dict[str, Any]:
        """Calculate rolling correlations between two assets"""
        return self.financial_skill.calculate_rolling_correlations(asset1_data, asset2_data, asset1_name, asset2_name, window_size, step_size)
    
    @tool
    def analyze_lead_lag_relationships(self, series1_data: List[float], series2_data: List[float],
                                     series1_name: str, series2_name: str, max_lag: int = 8) -> Dict[str, Any]:
        """Analyze lead-lag relationships between FX and crypto pairs"""
        return self.financial_skill.analyze_lead_lag_relationships(series1_data, series2_data, series1_name, series2_name, max_lag)
    
    @tool
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """Calculate Sharpe ratio for performance evaluation"""
        return self.financial_skill.calculate_sharpe_ratio(returns, risk_free_rate)
    
    @tool
    def calculate_maximum_drawdown(self, returns: List[float]) -> Dict[str, Any]:
        """Calculate maximum drawdown risk metric"""
        return self.financial_skill.calculate_maximum_drawdown(returns)
    
    @tool
    def calculate_volatility_metrics(self, returns: List[float], window: int = 20) -> Dict[str, Any]:
        """Calculate volatility metrics for trading systems"""
        return self.financial_skill.calculate_volatility_metrics(returns, window)
    
    @tool
    def calculate_information_ratio(self, portfolio_returns: List[float], benchmark_returns: List[float]) -> Dict[str, Any]:
        """Calculate Information Ratio for active trading performance"""
        return self.financial_skill.calculate_information_ratio(portfolio_returns, benchmark_returns)
    
    @tool
    def calculate_calmar_ratio(self, returns: List[float]) -> Dict[str, Any]:
        """Calculate Calmar Ratio (return/drawdown risk-adjusted)"""
        return self.financial_skill.calculate_calmar_ratio(returns)
    
    @tool
    def generate_early_warning_signals(self, correlation_data: Dict[str, Any], price_data: Dict[str, List[float]],
                                     signal_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate early warning signals for trading"""
        return self.financial_skill.generate_early_warning_signals(correlation_data, price_data, signal_thresholds)
    
    @tool
    def backtest_correlation_strategy(self, fx_data: List[float], crypto_data: List[float],
                                    strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest FX-crypto correlation trading strategy"""
        return self.financial_skill.backtest_correlation_strategy(fx_data, crypto_data, strategy_config)
    
    # AGENT COORDINATION TOOLS
    
    @tool
    def coordinate_distributed_calculation(self, expression: str, agent_ids: List[str],
                                         strategy: str = "parallel") -> Dict[str, Any]:
        """
        Coordinate calculation across multiple A2A agents
        
        Args:
            expression: Complex expression to distribute
            agent_ids: List of agent IDs to coordinate with
            strategy: Coordination strategy (parallel, sequential, redundant)
            
        Returns:
            Dict with coordination session details
        """
        session_id = f"coord_{datetime.now().isoformat()}"
        
        try:
            coordination_plan = {
                "session_id": session_id,
                "expression": expression,
                "participating_agents": agent_ids,
                "strategy": strategy,
                "status": "initiated",
                "sub_tasks": [],
                "created_at": datetime.now().isoformat()
            }
            
            # For complex expressions, break them down into sub-tasks
            # This is a simplified implementation - full version would parse and decompose
            if strategy == "parallel":
                # Assign same calculation to multiple agents for verification
                for agent_id in agent_ids:
                    coordination_plan["sub_tasks"].append({
                        "task_id": f"task_{agent_id}_{session_id}",
                        "agent_id": agent_id,
                        "expression": expression,
                        "task_type": "full_calculation",
                        "status": "pending"
                    })
            
            elif strategy == "sequential":
                # Break down into sequential steps
                coordination_plan["sub_tasks"].append({
                    "task_id": f"task_step1_{session_id}",
                    "agent_id": agent_ids[0] if agent_ids else "self",
                    "expression": expression,
                    "task_type": "sequential_step",
                    "status": "pending"
                })
            
            self.coordination_sessions[session_id] = coordination_plan
            
            logger.info(f"Initiated coordination session {session_id} with {len(agent_ids)} agents")
            
            return {
                "success": True,
                "coordination_session": session_id,
                "strategy": strategy,
                "participating_agents": agent_ids,
                "sub_tasks_count": len(coordination_plan["sub_tasks"]),
                "status": "coordination_initiated"
            }
            
        except Exception as e:
            logger.error(f"Coordination setup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    @tool
    def request_peer_verification(self, calculation_result: Dict[str, Any], 
                                peer_agents: List[str]) -> Dict[str, Any]:
        """Request peer agents to verify calculation results"""
        return self.verification_skill.validate_agent_result(
            calculation_result, 
            f"peer_group_{len(peer_agents)}", 
            independent_verification=True
        )
    
    # UTILITY AND REPORTING TOOLS
    
    @tool
    def optimize_calculation_method(self, expression: str, variables: Dict[str, Any] = None,
                                  requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recommend optimal calculation method based on requirements
        
        Args:
            expression: Mathematical expression
            variables: Variable values
            requirements: Performance/accuracy requirements
            
        Returns:
            Dict with method recommendation and reasoning
        """
        try:
            # Analyze expression characteristics
            calc_type = classify_calculation_type(expression)
            cost_estimate = estimate_computation_cost(expression, variables)
            auto_method = auto_select_computation_method(expression, variables)
            
            # Consider requirements
            requirements = requirements or {}
            accuracy_priority = requirements.get("accuracy_priority", "medium")
            speed_priority = requirements.get("speed_priority", "medium")
            memory_limit = requirements.get("memory_limit_mb", 1000)
            
            # Method recommendation logic
            if accuracy_priority == "high":
                recommended_method = ComputationMethod.SYMBOLIC
                reasoning = "High accuracy requirement favors symbolic computation for exact results"
            elif speed_priority == "high":
                recommended_method = ComputationMethod.NUMERIC
                reasoning = "High speed requirement favors numeric computation for performance"
            elif calc_type in [CalculationType.STATISTICAL, CalculationType.MATRIX]:
                recommended_method = ComputationMethod.NUMERIC
                reasoning = f"Calculation type {calc_type.value} is optimized for numeric computation"
            else:
                recommended_method = auto_method
                reasoning = "Auto-selection based on expression characteristics"
            
            return {
                "success": True,
                "recommended_method": recommended_method.value,
                "auto_selection": auto_method.value,
                "calculation_type": calc_type.value,
                "cost_estimate": cost_estimate,
                "reasoning": reasoning,
                "alternatives": {
                    "for_accuracy": ComputationMethod.SYMBOLIC.value,
                    "for_speed": ComputationMethod.NUMERIC.value,
                    "for_verification": ComputationMethod.HYBRID.value
                }
            }
            
        except Exception as e:
            logger.error(f"Method optimization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    def get_calculation_report(self, include_history: bool = False) -> Dict[str, Any]:
        """Generate comprehensive calculation performance report"""
        try:
            report = {
                "agent_id": self.agent_id,
                "report_timestamp": datetime.now().isoformat(),
                "agent_metrics": self.agent_metrics.copy(),
                "active_calculations": len(self.active_calculations),
                "coordination_sessions": len(self.coordination_sessions)
            }
            
            # Add sub-skill performance data
            report["sub_skill_performance"] = {
                "symbolic": self.symbolic_skill.get_operation_statistics(),
                "numeric": self.numeric_skill.get_performance_statistics(),
                "verification": self.verification_skill.get_verification_report(include_history=False),
                "financial": self.financial_skill.get_performance_statistics()
            }
            
            # Add recent calculation history if requested
            if include_history:
                recent_calculations = []
                for calc_id, calc_data in list(self.active_calculations.items())[-10:]:
                    recent_calculations.append({
                        "calculation_id": calc_id,
                        "expression": calc_data.get("expression", ""),
                        "status": calc_data.get("status", "unknown"),
                        "method_used": calc_data.get("method_used", "unknown"),
                        "computation_time": calc_data.get("computation_time", 0),
                        "success": calc_data.get("status") == "completed"
                    })
                report["recent_calculations"] = recent_calculations
            
            return {
                "success": True,
                **report
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    def explain_calculation_steps(self, calculation_id: str = None, 
                                expression: str = None, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Provide step-by-step explanation of calculation process
        
        Args:
            calculation_id: ID of previous calculation to explain
            expression: New expression to explain (if no calculation_id)
            variables: Variables for new expression
            
        Returns:
            Dict with detailed step explanations
        """
        try:
            if calculation_id and calculation_id in self.active_calculations:
                # Explain existing calculation
                calc_data = self.active_calculations[calculation_id]
                expression = calc_data["expression"]
                variables = calc_data.get("variables", {})
                result = calc_data.get("result", {})
                
                explanation = {
                    "calculation_id": calculation_id,
                    "expression": expression,
                    "variables": variables,
                    "method_used": result.get("method", "unknown"),
                    "steps": result.get("steps", []),
                    "result": result.get("result"),
                    "computation_time": calc_data.get("computation_time", 0)
                }
                
            elif expression:
                # Generate explanation for new expression
                # Use symbolic computation for detailed steps
                symbolic_result = self.symbolic_skill.calculate_symbolic(expression, variables)
                
                explanation = {
                    "expression": expression,
                    "variables": variables or {},
                    "method_used": "symbolic_explanation",
                    "steps": symbolic_result.get("steps", []),
                    "result": symbolic_result.get("result"),
                    "detailed_explanation": self._generate_detailed_explanation(expression, variables, symbolic_result)
                }
                
            else:
                return {
                    "success": False,
                    "error": "Either calculation_id or expression must be provided"
                }
            
            return {
                "success": True,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Step explanation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # A2A MESSAGE HANDLING
    
    async def handle_a2a_message(self, message: A2AMessage) -> A2AMessage:
        """Handle incoming A2A messages for calculation requests"""
        
        try:
            if message.message_type == MessageType.CALCULATION_REQUEST:
                # Extract calculation request
                request_data = message.data
                expression = request_data.get("expression", "")
                variables = request_data.get("variables", {})
                method = request_data.get("method", "auto")
                requirements = request_data.get("requirements", {})
                
                logger.info(f"Received A2A calculation request from {message.sender}: {expression}")
                
                # Use GROK to analyze the request and determine best approach
                grok_analysis = self.grok_intelligence.analyze_calculation_problem(
                    expression, variables, {
                        "requesting_agent": message.sender,
                        "workflow_context": message.workflow_context,
                        "a2a_request": True
                    }
                )
                
                # Process calculation based on GROK's analysis and requested method
                if method == "hybrid_verification" or grok_analysis.get("recommended_method") == "hybrid":
                    result = self.calculate_with_hybrid_verification(expression, variables)
                else:
                    result = self.calculate_with_auto_method(expression, variables, method)
                
                # Add A2A specific metadata
                result["a2a_metadata"] = {
                    "requesting_agent": message.sender,
                    "processing_agent": self.agent_id,
                    "grok_analysis": grok_analysis,
                    "workflow_context": message.workflow_context
                }
                
                # Return result
                response = A2AMessage(
                    sender=self.agent_id,
                    recipient=message.sender,
                    message_type=MessageType.CALCULATION_RESPONSE,
                    data=result,
                    workflow_context=message.workflow_context
                )
                
                logger.info(f"Sent A2A calculation response to {message.sender}: success={result.get('success')}")
                return response
            
            elif message.message_type == MessageType.VERIFICATION_REQUEST:
                # Handle verification request from peer agent
                result_to_verify = message.data.get("result")
                original_expression = message.data.get("expression", "")
                
                logger.info(f"Received A2A verification request from {message.sender}")
                
                # Perform independent calculation for verification
                independent_result = self.calculate_with_auto_method(original_expression)
                
                # Verify against provided result
                verification = self.verification_skill.validate_agent_result(result_to_verify, message.sender)
                verification["independent_result"] = independent_result
                
                response = A2AMessage(
                    sender=self.agent_id,
                    recipient=message.sender,
                    message_type=MessageType.VERIFICATION_RESPONSE,
                    data=verification,
                    workflow_context=message.workflow_context
                )
                
                return response
            
            # Delegate to parent for other message types
            return await super().handle_a2a_message(message)
            
        except Exception as e:
            logger.error(f"A2A message handling failed: {e}")
            
            # Return error response
            error_response = A2AMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type=MessageType.ERROR,
                data={"error": str(e), "original_message_type": message.message_type.value},
                workflow_context=message.workflow_context
            )
            
            return error_response
    
    # PRIVATE HELPER METHODS
    
    def _execute_hybrid_calculation(self, expression: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute hybrid calculation using both symbolic and numeric methods"""
        try:
            # Run both methods
            symbolic_result = self.symbolic_skill.calculate_symbolic(expression, variables)
            
            numeric_vars = {k: v for k, v in (variables or {}).items() 
                          if isinstance(v, (int, float, list))}
            numeric_result = self.numeric_skill.calculate_numeric(expression, numeric_vars)
            
            # Verify results
            verification = None
            if symbolic_result.get("success") and numeric_result.get("success"):
                verification = self.verification_skill.verify_calculation(symbolic_result, numeric_result)
            
            # Select best result
            best_result = self._select_best_result(symbolic_result, numeric_result, verification)
            
            return {
                "success": True,
                "method": "hybrid",
                "symbolic_result": symbolic_result,
                "numeric_result": numeric_result,
                "verification": verification,
                "recommended_result": best_result,
                "result": best_result.get("result") if best_result else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "method": "hybrid",
                "error": str(e)
            }
    
    def _execute_auto_calculation(self, expression: str, variables: Dict[str, Any] = None,
                                cost_estimate: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute calculation with automatic method selection"""
        cost_estimate = cost_estimate or estimate_computation_cost(expression, variables)
        
        # Choose method based on cost estimate
        if cost_estimate["recommended_method"] == "symbolic":
            return self.symbolic_skill.calculate_symbolic(expression, variables)
        else:
            numeric_vars = {k: v for k, v in (variables or {}).items() 
                          if isinstance(v, (int, float, list))}
            return self.numeric_skill.calculate_numeric(expression, numeric_vars)
    
    def _select_best_result(self, symbolic_result: Dict, numeric_result: Dict,
                          verification_result: Dict = None) -> Dict[str, Any]:
        """Select the best result from symbolic and numeric computations"""
        
        # If verification passed, prefer symbolic for accuracy
        if verification_result and verification_result.get("verification_passed"):
            return symbolic_result if symbolic_result.get("success") else numeric_result
        
        # If only one succeeded, use that one
        if symbolic_result.get("success") and not numeric_result.get("success"):
            return symbolic_result
        elif numeric_result.get("success") and not symbolic_result.get("success"):
            return numeric_result
        
        # If both succeeded but verification failed/missing, prefer symbolic for accuracy
        if symbolic_result.get("success") and numeric_result.get("success"):
            return symbolic_result
        
        # If both failed, return symbolic with error info
        return {
            "success": False,
            "error": "Both symbolic and numeric calculations failed",
            "symbolic_error": symbolic_result.get("error"),
            "numeric_error": numeric_result.get("error")
        }
    
    def _update_agent_metrics(self, success: bool, computation_time: float, method: ComputationMethod):
        """Update agent performance metrics"""
        self.agent_metrics["total_requests"] += 1
        
        if success:
            self.agent_metrics["successful_calculations"] += 1
        else:
            self.agent_metrics["failed_calculations"] += 1
        
        # Update average response time
        total_requests = self.agent_metrics["total_requests"]
        current_avg = self.agent_metrics["average_response_time"]
        new_avg = (current_avg * (total_requests - 1) + computation_time) / total_requests
        self.agent_metrics["average_response_time"] = new_avg
        
        # Update method success rates
        method_key = method.value
        if method_key in self.agent_metrics["method_success_rates"]:
            # Simple success rate tracking (could be enhanced with more sophisticated metrics)
            current_rate = self.agent_metrics["method_success_rates"][method_key]
            self.agent_metrics["method_success_rates"][method_key] = (current_rate + (1.0 if success else 0.0)) / 2
    
    def _generate_detailed_explanation(self, expression: str, variables: Dict[str, Any],
                                     symbolic_result: Dict[str, Any]) -> List[str]:
        """Generate detailed step-by-step explanation"""
        explanation = []
        
        explanation.append(f"Expression to evaluate: {expression}")
        
        if variables:
            explanation.append(f"Variable substitutions: {variables}")
        
        # Add symbolic computation steps
        steps = symbolic_result.get("steps", [])
        for i, step in enumerate(steps, 1):
            explanation.append(f"Step {i}: {step}")
        
        # Add result interpretation
        result = symbolic_result.get("result")
        if result:
            formatted_result = format_result_for_display(result)
            explanation.append(f"Final result: {formatted_result}")
        
        return explanation


# Global instance factory
_calculation_agent_instance = None

def get_calculation_agent() -> CalculationAgent:
    """Get or create the calculation agent instance"""
    global _calculation_agent_instance
    if _calculation_agent_instance is None:
        _calculation_agent_instance = CalculationAgent()
    return _calculation_agent_instance