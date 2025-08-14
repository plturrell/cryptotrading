"""
Verification & Cross-checking Sub-skill for Calculation Agent

Cross-verifies calculation results between different methods:
- Compares symbolic vs numeric results
- Validates A2A agent calculations
- Identifies discrepancies and potential errors
- Provides confidence assessment

Advantages for A2A:
- Ensures calculation accuracy across agent interactions
- Builds trust between collaborating agents
- Identifies method-specific errors or limitations
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
from datetime import datetime

from src.strands.tools import tool
from .types import (
    CalculationResult, ComputationMethod, VerificationResult, 
    VerificationStatus, CalculationType
)
from .utils import compare_numeric_results, format_result_for_display


logger = logging.getLogger(__name__)


class VerificationSkill:
    """Verification and cross-checking sub-skill"""
    
    def __init__(self):
        self.verification_history: List[VerificationResult] = []
        self.trust_scores: Dict[str, float] = {}  # Trust scores for different agents/methods
        self.verification_stats = {
            "total_verifications": 0,
            "passed_verifications": 0,
            "failed_verifications": 0,
            "success_rate": 0.0,
            "average_discrepancy": 0.0
        }
    
    @tool
    def verify_calculation(self, result1: Dict[str, Any], result2: Dict[str, Any], 
                          tolerance: float = 1e-10, verification_method: str = "numeric_comparison") -> Dict[str, Any]:
        """
        Cross-verify calculation results between different methods
        
        Args:
            result1: First calculation result
            result2: Second calculation result  
            tolerance: Tolerance for numeric comparison
            verification_method: Type of verification to perform
            
        Returns:
            Dict with verification results and confidence assessment
        """
        start_time = time.time()
        
        try:
            verification_id = f"verify_{datetime.now().isoformat()}"
            
            # Extract methods used
            method1 = result1.get("method", "unknown")
            method2 = result2.get("method", "unknown")
            
            verification_data = {
                "verification_id": verification_id,
                "method1": method1,
                "method2": method2,
                "tolerance": tolerance,
                "verification_method": verification_method,
                "timestamp": datetime.now().isoformat()
            }
            
            # Perform verification based on method
            if verification_method == "numeric_comparison":
                verification_result = self._numeric_comparison(result1, result2, tolerance)
            elif verification_method == "symbolic_equivalence":
                verification_result = self._symbolic_equivalence(result1, result2)
            elif verification_method == "statistical_consistency":
                verification_result = self._statistical_consistency(result1, result2, tolerance)
            else:
                return {
                    "success": False,
                    "error": f"Unknown verification method: {verification_method}"
                }
            
            verification_data.update(verification_result)
            
            # Calculate confidence based on verification outcome
            confidence = self._calculate_verification_confidence(verification_result, method1, method2)
            verification_data["confidence"] = confidence
            
            # Store verification result
            ver_result = VerificationResult(
                methods_compared=[ComputationMethod(method1), ComputationMethod(method2)],
                results_match=verification_result.get("results_match", False),
                tolerance=tolerance,
                discrepancy=verification_result.get("discrepancy"),
                recommendation=verification_result.get("recommendation", ""),
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            self.verification_history.append(ver_result)
            self._update_verification_stats(ver_result)
            
            computation_time = time.time() - start_time
            
            logger.info(f"Verification completed: {method1} vs {method2}, match: {verification_result.get('results_match')}")
            
            return {
                "success": True,
                "verification_passed": verification_result.get("results_match", False),
                "computation_time": computation_time,
                **verification_data
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method1": result1.get("method", "unknown"),
                "method2": result2.get("method", "unknown")
            }
    
    @tool
    def validate_agent_result(self, agent_result: Dict[str, Any], agent_id: str,
                             independent_verification: bool = True) -> Dict[str, Any]:
        """
        Validate results from other A2A agents
        
        Args:
            agent_result: Result received from another agent
            agent_id: ID of the agent that provided the result
            independent_verification: Whether to perform independent calculation
            
        Returns:
            Dict with validation results and trust assessment
        """
        start_time = time.time()
        
        try:
            validation_data = {
                "agent_id": agent_id,
                "validation_timestamp": datetime.now().isoformat(),
                "independent_verification": independent_verification
            }
            
            # Check if result has required fields
            required_fields = ["result", "method", "expression"]
            missing_fields = [field for field in required_fields if field not in agent_result]
            
            if missing_fields:
                validation_data["validation_passed"] = False
                validation_data["issues"] = [f"Missing required fields: {missing_fields}"]
                validation_data["trust_impact"] = "negative"
                
                return {
                    "success": True,
                    "computation_time": time.time() - start_time,
                    **validation_data
                }
            
            # Basic validation checks
            validation_issues = []
            
            # Check for reasonable computation time
            comp_time = agent_result.get("computation_time", 0)
            if comp_time > 60:  # More than 60 seconds seems excessive
                validation_issues.append(f"Excessive computation time: {comp_time}s")
            
            # Check for error indicators
            if agent_result.get("error"):
                validation_issues.append(f"Agent reported error: {agent_result['error']}")
            
            # Check result validity
            result = agent_result.get("result")
            if result is None:
                validation_issues.append("Null result returned")
            elif isinstance(result, str) and "error" in result.lower():
                validation_issues.append("Result contains error message")
            
            # Perform independent verification if requested
            if independent_verification and not validation_issues:
                # This would require implementing the same calculation
                # For now, we'll simulate independent verification
                validation_data["independent_result"] = "Independent verification not fully implemented"
                validation_data["independent_match"] = None
            
            # Calculate validation outcome
            validation_passed = len(validation_issues) == 0
            validation_data["validation_passed"] = validation_passed
            validation_data["issues"] = validation_issues
            
            # Update trust score for agent
            self._update_agent_trust_score(agent_id, validation_passed, validation_issues)
            validation_data["agent_trust_score"] = self.trust_scores.get(agent_id, 0.5)
            
            # Provide recommendation
            if validation_passed:
                validation_data["recommendation"] = "Result appears valid and can be trusted"
                validation_data["trust_impact"] = "positive"
            else:
                validation_data["recommendation"] = f"Result has issues: {'; '.join(validation_issues)}"
                validation_data["trust_impact"] = "negative"
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "computation_time": computation_time,
                **validation_data
            }
            
        except Exception as e:
            logger.error(f"Agent result validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    @tool
    def assess_result_confidence(self, calculation_result: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess confidence level of a calculation result
        
        Args:
            calculation_result: The calculation result to assess
            context: Additional context for confidence assessment
            
        Returns:
            Dict with confidence assessment
        """
        start_time = time.time()
        
        try:
            confidence_factors = []
            confidence_score = 1.0
            
            # Factor 1: Method reliability
            method = calculation_result.get("method", "unknown")
            if method == "symbolic":
                method_reliability = 0.95  # High reliability for exact symbolic computation
                confidence_factors.append("Symbolic computation provides exact results")
            elif method == "numeric":
                method_reliability = 0.85  # Good reliability for numeric computation
                confidence_factors.append("Numeric computation with potential floating-point precision limits")
            else:
                method_reliability = 0.7
                confidence_factors.append(f"Unknown or mixed method: {method}")
            
            confidence_score *= method_reliability
            
            # Factor 2: Computation time (very fast or very slow might indicate issues)
            comp_time = calculation_result.get("computation_time", 0)
            if comp_time < 0.001:
                time_factor = 0.9  # Suspiciously fast
                confidence_factors.append("Very fast computation - possible cached or trivial result")
            elif comp_time > 30:
                time_factor = 0.8  # Very slow
                confidence_factors.append("Very slow computation - possible complexity or performance issues")
            else:
                time_factor = 1.0
                confidence_factors.append("Reasonable computation time")
            
            confidence_score *= time_factor
            
            # Factor 3: Result validation
            result = calculation_result.get("result")
            if result is None:
                result_factor = 0.0
                confidence_factors.append("Null result indicates failure")
            elif isinstance(result, str) and ("error" in result.lower() or "failed" in result.lower()):
                result_factor = 0.1
                confidence_factors.append("Result contains error indicators")
            elif isinstance(result, (int, float)) and (np.isnan(result) or np.isinf(result)):
                result_factor = 0.0
                confidence_factors.append("Result is NaN or infinite")
            else:
                result_factor = 1.0
                confidence_factors.append("Result appears valid")
            
            confidence_score *= result_factor
            
            # Factor 4: Error presence
            if calculation_result.get("error"):
                error_factor = 0.1
                confidence_factors.append(f"Error reported: {calculation_result['error']}")
                confidence_score *= error_factor
            
            # Factor 5: Steps/explanation availability
            steps = calculation_result.get("steps", [])
            if steps and len(steps) > 0:
                explanation_factor = 1.0
                confidence_factors.append("Detailed steps provided for transparency")
            else:
                explanation_factor = 0.9
                confidence_factors.append("Limited explanation of calculation process")
            
            confidence_score *= explanation_factor
            
            # Ensure confidence is within [0, 1]
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            # Determine confidence level category
            if confidence_score >= 0.9:
                confidence_level = "very_high"
            elif confidence_score >= 0.7:
                confidence_level = "high"
            elif confidence_score >= 0.5:
                confidence_level = "medium"
            elif confidence_score >= 0.3:
                confidence_level = "low"
            else:
                confidence_level = "very_low"
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "computation_time": computation_time,
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "confidence_factors": confidence_factors,
                "recommendation": self._get_confidence_recommendation(confidence_score, confidence_level)
            }
            
        except Exception as e:
            logger.error(f"Confidence assessment failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    def get_verification_report(self, include_history: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive verification report
        
        Args:
            include_history: Whether to include verification history
            
        Returns:
            Dict with verification statistics and insights
        """
        try:
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "verification_statistics": self.verification_stats.copy(),
                "agent_trust_scores": self.trust_scores.copy(),
                "insights": []
            }
            
            # Add insights based on statistics
            if self.verification_stats["total_verifications"] > 0:
                success_rate = self.verification_stats["success_rate"]
                if success_rate > 0.9:
                    report["insights"].append("Excellent verification success rate - calculations are highly reliable")
                elif success_rate > 0.7:
                    report["insights"].append("Good verification success rate - most calculations are reliable")
                elif success_rate > 0.5:
                    report["insights"].append("Moderate verification success rate - some calculation issues detected")
                else:
                    report["insights"].append("Low verification success rate - significant calculation reliability concerns")
                
                # Average discrepancy insight
                avg_discrepancy = self.verification_stats["average_discrepancy"]
                if avg_discrepancy < 1e-12:
                    report["insights"].append("Very low average discrepancy - excellent precision")
                elif avg_discrepancy < 1e-8:
                    report["insights"].append("Low average discrepancy - good precision")
                else:
                    report["insights"].append(f"Higher average discrepancy ({avg_discrepancy:.2e}) - precision concerns")
            
            # Agent trust insights
            if self.trust_scores:
                most_trusted = max(self.trust_scores.items(), key=lambda x: x[1])
                least_trusted = min(self.trust_scores.items(), key=lambda x: x[1])
                
                report["most_trusted_agent"] = {"agent_id": most_trusted[0], "trust_score": most_trusted[1]}
                report["least_trusted_agent"] = {"agent_id": least_trusted[0], "trust_score": least_trusted[1]}
            
            # Include verification history if requested
            if include_history and self.verification_history:
                report["recent_verifications"] = []
                for ver in self.verification_history[-10:]:  # Last 10 verifications
                    report["recent_verifications"].append({
                        "methods": [m.value for m in ver.methods_compared],
                        "results_match": ver.results_match,
                        "discrepancy": ver.discrepancy,
                        "confidence": ver.confidence,
                        "timestamp": ver.timestamp.isoformat() if ver.timestamp else None
                    })
            
            return {
                "success": True,
                **report
            }
            
        except Exception as e:
            logger.error(f"Verification report generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _numeric_comparison(self, result1: Dict[str, Any], result2: Dict[str, Any], 
                           tolerance: float) -> Dict[str, Any]:
        """Compare numeric results within tolerance"""
        try:
            # Extract numeric values
            val1 = self._extract_numeric_value(result1)
            val2 = self._extract_numeric_value(result2)
            
            if val1 is None or val2 is None:
                return {
                    "results_match": False,
                    "reason": "Could not extract numeric values for comparison",
                    "discrepancy": None,
                    "recommendation": "Use symbolic equivalence verification instead"
                }
            
            # Compare values
            comparison = compare_numeric_results(val1, val2, tolerance)
            
            return {
                "results_match": comparison["match"],
                "discrepancy": comparison.get("max_discrepancy"),
                "reason": comparison.get("reason", ""),
                "tolerance_used": tolerance,
                "recommendation": "Results verified" if comparison["match"] else "Investigate discrepancy"
            }
            
        except Exception as e:
            return {
                "results_match": False,
                "reason": f"Numeric comparison failed: {str(e)}",
                "discrepancy": None,
                "recommendation": "Manual verification required"
            }
    
    def _symbolic_equivalence(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
        """Check symbolic equivalence of results"""
        try:
            # Extract symbolic forms
            sym1 = result1.get("symbolic_form") or result1.get("result")
            sym2 = result2.get("symbolic_form") or result2.get("result")
            
            if not sym1 or not sym2:
                return {
                    "results_match": False,
                    "reason": "Symbolic forms not available",
                    "recommendation": "Use numeric comparison instead"
                }
            
            # Simple string comparison (could be enhanced with SymPy equivalence)
            symbolic_match = str(sym1).strip() == str(sym2).strip()
            
            return {
                "results_match": symbolic_match,
                "symbolic_form_1": str(sym1),
                "symbolic_form_2": str(sym2),
                "reason": "Symbolic forms match" if symbolic_match else "Symbolic forms differ",
                "recommendation": "Symbolic equivalence verified" if symbolic_match else "Check mathematical equivalence"
            }
            
        except Exception as e:
            return {
                "results_match": False,
                "reason": f"Symbolic equivalence check failed: {str(e)}",
                "recommendation": "Manual verification required"
            }
    
    def _statistical_consistency(self, result1: Dict[str, Any], result2: Dict[str, Any], 
                               tolerance: float) -> Dict[str, Any]:
        """Check statistical consistency for array/statistical results"""
        try:
            # This would be used for statistical analysis results
            stats1 = result1.get("statistics", {})
            stats2 = result2.get("statistics", {})
            
            if not stats1 or not stats2:
                return self._numeric_comparison(result1, result2, tolerance)
            
            # Compare statistical measures
            consistency_checks = []
            inconsistencies = []
            
            for stat_name in ["mean", "std", "median"]:
                if stat_name in stats1 and stat_name in stats2:
                    val1, val2 = stats1[stat_name], stats2[stat_name]
                    diff = abs(val1 - val2)
                    if diff <= tolerance:
                        consistency_checks.append(f"{stat_name}: consistent")
                    else:
                        inconsistencies.append(f"{stat_name}: {diff:.2e} difference")
            
            results_match = len(inconsistencies) == 0
            
            return {
                "results_match": results_match,
                "consistency_checks": consistency_checks,
                "inconsistencies": inconsistencies,
                "reason": "Statistical consistency verified" if results_match else "Statistical inconsistencies found",
                "recommendation": "Results verified" if results_match else "Review statistical differences"
            }
            
        except Exception as e:
            return {
                "results_match": False,
                "reason": f"Statistical consistency check failed: {str(e)}",
                "recommendation": "Manual verification required"
            }
    
    def _extract_numeric_value(self, result: Dict[str, Any]) -> Optional[Union[float, List, Dict]]:
        """Extract numeric value from calculation result"""
        # Try different fields where numeric values might be stored
        value = result.get("numeric_value")
        if value is not None:
            return value
        
        value = result.get("result")
        if isinstance(value, (int, float, list)):
            return value
        
        # Try to convert string result to float
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        
        return None
    
    def _calculate_verification_confidence(self, verification_result: Dict[str, Any], 
                                         method1: str, method2: str) -> float:
        """Calculate confidence in verification result"""
        base_confidence = 0.5
        
        if verification_result.get("results_match", False):
            base_confidence = 0.9
            
            # Higher confidence for symbolic vs numeric comparison
            if "symbolic" in [method1, method2] and "numeric" in [method1, method2]:
                base_confidence = 0.95
        else:
            # Lower confidence if results don't match
            discrepancy = verification_result.get("discrepancy")
            if discrepancy and discrepancy < 1e-8:
                base_confidence = 0.6  # Small discrepancy might be acceptable
            else:
                base_confidence = 0.2  # Large discrepancy is concerning
        
        return base_confidence
    
    def _update_agent_trust_score(self, agent_id: str, validation_passed: bool, 
                                issues: List[str]):
        """Update trust score for an agent based on validation result"""
        current_score = self.trust_scores.get(agent_id, 0.5)  # Start with neutral trust
        
        if validation_passed:
            # Increase trust (but slowly to prevent manipulation)
            new_score = min(1.0, current_score + 0.05)
        else:
            # Decrease trust based on severity of issues
            severity_penalty = 0.1 * len(issues)
            new_score = max(0.0, current_score - severity_penalty)
        
        self.trust_scores[agent_id] = new_score
    
    def _update_verification_stats(self, verification_result: VerificationResult):
        """Update verification statistics"""
        self.verification_stats["total_verifications"] += 1
        
        if verification_result.results_match:
            self.verification_stats["passed_verifications"] += 1
        else:
            self.verification_stats["failed_verifications"] += 1
        
        # Update success rate
        total = self.verification_stats["total_verifications"]
        passed = self.verification_stats["passed_verifications"]
        self.verification_stats["success_rate"] = passed / total
        
        # Update average discrepancy
        if verification_result.discrepancy is not None:
            current_avg = self.verification_stats["average_discrepancy"]
            new_avg = (current_avg * (total - 1) + verification_result.discrepancy) / total
            self.verification_stats["average_discrepancy"] = new_avg
    
    def _get_confidence_recommendation(self, confidence_score: float, confidence_level: str) -> str:
        """Get recommendation based on confidence assessment"""
        if confidence_level == "very_high":
            return "Result can be used with high confidence"
        elif confidence_level == "high":
            return "Result appears reliable for most applications"
        elif confidence_level == "medium":
            return "Result should be used with caution - consider verification"
        elif confidence_level == "low":
            return "Result has significant reliability concerns - verification recommended"
        else:
            return "Result has serious reliability issues - avoid using without manual verification"