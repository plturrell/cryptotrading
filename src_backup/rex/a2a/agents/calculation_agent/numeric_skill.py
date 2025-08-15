"""
Numeric Computation Sub-skill for Calculation Agent

Uses NumPy/SciPy for high-performance numerical calculations:
- Large-scale computations
- Statistical analysis
- Matrix operations
- Performance-critical calculations

Advantages for A2A:
- Fast computation for time-sensitive agent interactions
- Efficient handling of large datasets
- Vectorized operations for batch processing
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import scipy.linalg as linalg
from decimal import Decimal

from src.strands.tools import tool
from .types import CalculationResult, ComputationMethod, CalculationType, VerificationStatus, StatisticalData
from .utils import (
    classify_calculation_type, sanitize_expression_for_eval, 
    create_safe_math_context, validate_numeric_result
)


logger = logging.getLogger(__name__)


class NumericComputationSkill:
    """Numeric computation sub-skill using NumPy/SciPy"""
    
    def __init__(self):
        self.computation_cache = {}
        self.performance_stats = {
            "total_operations": 0,
            "average_computation_time": 0.0,
            "operations_by_type": {},
            "cache_hits": 0
        }
    
    @tool
    def calculate_numeric(self, expression: str, variables: Dict[str, Union[float, List]] = None, 
                         method: str = "numpy", precision: int = 15) -> Dict[str, Any]:
        """
        High-performance numeric computation using NumPy
        
        Args:
            expression: Mathematical expression as string
            variables: Dictionary of numeric variables
            method: Computation method (numpy, scipy)
            precision: Decimal precision for results
            
        Returns:
            Dict with numeric computation result and metadata
        """
        start_time = time.time()
        
        try:
            # Create safe computation context
            computation_context = create_safe_math_context()
            
            # Add variables to context
            if variables:
                for var, val in variables.items():
                    if isinstance(val, (list, tuple)):
                        computation_context[var] = np.array(val)
                    elif isinstance(val, Decimal):
                        computation_context[var] = float(val)
                    else:
                        computation_context[var] = val
            
            # Sanitize and prepare expression
            sanitized_expr = sanitize_expression_for_eval(expression, computation_context)
            
            # Evaluate expression
            result = eval(sanitized_expr, computation_context)
            
            # Validate result
            if not validate_numeric_result(result):
                return {
                    "success": False,
                    "method": "numeric",
                    "error": "Result contains NaN or infinite values",
                    "expression": expression
                }
            
            # Process result based on type
            result_data = self._process_numeric_result(result, expression, sanitized_expr, variables)
            
            computation_time = time.time() - start_time
            
            # Update performance statistics
            self._update_performance_stats(computation_time, classify_calculation_type(expression))
            
            logger.info(f"Numeric calculation completed: {expression}")
            
            return {
                "success": True,
                "method": "numeric",
                "computation_time": computation_time,
                **result_data
            }
            
        except Exception as e:
            logger.error(f"Numeric calculation failed: {e}")
            return {
                "success": False,
                "method": "numeric",
                "error": str(e),
                "expression": expression,
                "variables": variables
            }
    
    @tool
    def perform_matrix_operations(self, operation: str, matrices: Dict[str, List[List[float]]], 
                                 **kwargs) -> Dict[str, Any]:
        """
        Perform matrix operations using NumPy/SciPy
        
        Args:
            operation: Matrix operation (multiply, inverse, determinant, eigenvalues, etc.)
            matrices: Dictionary of matrices as nested lists
            **kwargs: Additional parameters for specific operations
            
        Returns:
            Dict with matrix operation result
        """
        start_time = time.time()
        
        try:
            # Convert matrices to NumPy arrays
            np_matrices = {}
            for name, matrix in matrices.items():
                np_matrices[name] = np.array(matrix)
            
            result_data = {
                "operation": operation,
                "input_matrices": {name: mat.shape for name, mat in np_matrices.items()},
                "steps": [f"Operation: {operation}"]
            }
            
            if operation == "multiply":
                if len(np_matrices) < 2:
                    return {"success": False, "error": "Matrix multiplication requires at least 2 matrices"}
                
                matrix_list = list(np_matrices.values())
                result = matrix_list[0]
                for i in range(1, len(matrix_list)):
                    result = np.dot(result, matrix_list[i])
                    result_data["steps"].append(f"Step {i}: Shape {result.shape}")
                
                result_data["result"] = result.tolist()
                result_data["result_shape"] = result.shape
                
            elif operation == "inverse":
                if len(np_matrices) != 1:
                    return {"success": False, "error": "Matrix inverse requires exactly 1 matrix"}
                
                matrix = list(np_matrices.values())[0]
                if matrix.shape[0] != matrix.shape[1]:
                    return {"success": False, "error": "Matrix must be square for inverse"}
                
                result = np.linalg.inv(matrix)
                result_data["result"] = result.tolist()
                result_data["determinant"] = float(np.linalg.det(matrix))
                result_data["steps"].append(f"Determinant: {result_data['determinant']}")
                
            elif operation == "determinant":
                if len(np_matrices) != 1:
                    return {"success": False, "error": "Determinant requires exactly 1 matrix"}
                
                matrix = list(np_matrices.values())[0]
                if matrix.shape[0] != matrix.shape[1]:
                    return {"success": False, "error": "Matrix must be square for determinant"}
                
                result = np.linalg.det(matrix)
                result_data["result"] = float(result)
                
            elif operation == "eigenvalues":
                if len(np_matrices) != 1:
                    return {"success": False, "error": "Eigenvalue computation requires exactly 1 matrix"}
                
                matrix = list(np_matrices.values())[0]
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                
                result_data["eigenvalues"] = eigenvalues.tolist()
                result_data["eigenvectors"] = eigenvectors.tolist()
                result_data["result"] = eigenvalues.tolist()
                
            elif operation == "transpose":
                if len(np_matrices) != 1:
                    return {"success": False, "error": "Transpose requires exactly 1 matrix"}
                
                matrix = list(np_matrices.values())[0]
                result = matrix.T
                result_data["result"] = result.tolist()
                result_data["result_shape"] = result.shape
                
            elif operation == "svd":  # Singular Value Decomposition
                if len(np_matrices) != 1:
                    return {"success": False, "error": "SVD requires exactly 1 matrix"}
                
                matrix = list(np_matrices.values())[0]
                U, s, Vt = np.linalg.svd(matrix)
                
                result_data["U"] = U.tolist()
                result_data["singular_values"] = s.tolist()
                result_data["Vt"] = Vt.tolist()
                result_data["result"] = s.tolist()  # Return singular values as main result
                
            else:
                return {"success": False, "error": f"Unknown matrix operation: {operation}"}
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "matrix_operations",
                "computation_time": computation_time,
                **result_data
            }
            
        except Exception as e:
            logger.error(f"Matrix operation failed: {e}")
            return {
                "success": False,
                "method": "matrix_operations",
                "error": str(e),
                "operation": operation
            }
    
    @tool
    def perform_statistical_analysis(self, data: List[float], analysis_type: str = "descriptive",
                                   confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Perform statistical analysis on numerical data
        
        Args:
            data: List of numerical data points
            analysis_type: Type of analysis (descriptive, hypothesis_test, distribution_fit)
            confidence_level: Confidence level for intervals
            
        Returns:
            Dict with statistical analysis results
        """
        start_time = time.time()
        
        try:
            data_array = np.array(data)
            n = len(data_array)
            
            if n == 0:
                return {"success": False, "error": "Empty data provided"}
            
            result_data = {
                "data_size": n,
                "analysis_type": analysis_type,
                "confidence_level": confidence_level
            }
            
            if analysis_type == "descriptive":
                stats_result = self._compute_descriptive_statistics(data_array, confidence_level)
                result_data.update(stats_result)
                
            elif analysis_type == "normality_test":
                # Shapiro-Wilk test for normality
                if n >= 3:
                    stat, p_value = stats.shapiro(data_array)
                    result_data["shapiro_wilk_statistic"] = float(stat)
                    result_data["p_value"] = float(p_value)
                    result_data["is_normal"] = p_value > (1 - confidence_level)
                    result_data["test"] = "Shapiro-Wilk"
                else:
                    result_data["error"] = "Insufficient data for normality test (need >= 3 points)"
                
            elif analysis_type == "distribution_fit":
                # Fit common distributions and find best fit
                distributions = ['norm', 'lognorm', 'expon', 'gamma', 'beta']
                best_fit = self._fit_distributions(data_array, distributions)
                result_data.update(best_fit)
                
            elif analysis_type == "outliers":
                # Detect outliers using IQR method
                outliers = self._detect_outliers(data_array)
                result_data.update(outliers)
                
            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "statistical_analysis",
                "computation_time": computation_time,
                **result_data
            }
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {
                "success": False,
                "method": "statistical_analysis",
                "error": str(e),
                "analysis_type": analysis_type
            }
    
    @tool
    def solve_optimization_problem(self, objective: str, variables: List[str],
                                  bounds: Dict[str, List[float]] = None,
                                  constraints: List[str] = None,
                                  method: str = "minimize") -> Dict[str, Any]:
        """
        Solve optimization problems using SciPy
        
        Args:
            objective: Objective function as string
            variables: List of optimization variables
            bounds: Variable bounds as {var: [min, max]}
            constraints: List of constraint expressions
            method: Optimization method (minimize, maximize)
            
        Returns:
            Dict with optimization results
        """
        start_time = time.time()
        
        try:
            # This is a simplified implementation
            # A full implementation would parse the objective function and constraints
            # and use scipy.optimize methods
            
            result_data = {
                "objective": objective,
                "variables": variables,
                "bounds": bounds or {},
                "constraints": constraints or [],
                "method": method,
                "status": "simplified_implementation"
            }
            
            # Placeholder for actual optimization
            # In practice, this would use scipy.optimize.minimize or similar
            result_data["message"] = "Optimization solving requires full implementation"
            result_data["success"] = False
            result_data["note"] = "This is a placeholder - full optimization implementation needed"
            
            computation_time = time.time() - start_time
            
            return {
                "success": False,
                "method": "optimization",
                "computation_time": computation_time,
                **result_data
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                "success": False,
                "method": "optimization",
                "error": str(e)
            }
    
    @tool
    def compute_numerical_derivatives(self, function_values: List[float], 
                                    x_values: List[float] = None,
                                    method: str = "central") -> Dict[str, Any]:
        """
        Compute numerical derivatives from function values
        
        Args:
            function_values: Y values of the function
            x_values: X values (optional, uses indices if None)
            method: Differentiation method (forward, backward, central)
            
        Returns:
            Dict with numerical derivative results
        """
        start_time = time.time()
        
        try:
            y = np.array(function_values)
            x = np.array(x_values) if x_values else np.arange(len(y))
            
            if len(x) != len(y):
                return {"success": False, "error": "x and y arrays must have same length"}
            
            if len(y) < 2:
                return {"success": False, "error": "Need at least 2 points for derivative"}
            
            if method == "forward":
                # Forward difference
                derivative = np.diff(y) / np.diff(x)
                x_deriv = x[:-1]
                
            elif method == "backward":
                # Backward difference  
                derivative = np.diff(y) / np.diff(x)
                x_deriv = x[1:]
                
            elif method == "central":
                # Central difference
                if len(y) < 3:
                    return {"success": False, "error": "Need at least 3 points for central difference"}
                
                derivative = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
                x_deriv = x[1:-1]
                
            else:
                return {"success": False, "error": f"Unknown method: {method}"}
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "numerical_derivative",
                "computation_time": computation_time,
                "derivative_method": method,
                "x_derivative": x_deriv.tolist(),
                "derivative": derivative.tolist(),
                "derivative_points": len(derivative)
            }
            
        except Exception as e:
            logger.error(f"Numerical derivative computation failed: {e}")
            return {
                "success": False,
                "method": "numerical_derivative",
                "error": str(e)
            }
    
    def _process_numeric_result(self, result: Any, original_expr: str, 
                              sanitized_expr: str, variables: Dict) -> Dict[str, Any]:
        """Process and format numeric computation result"""
        result_data = {
            "original_expression": original_expr,
            "numpy_expression": sanitized_expr,
            "variables": variables or {},
            "steps": [f"Converted to NumPy: {sanitized_expr}"]
        }
        
        if isinstance(result, np.ndarray):
            result_data["result"] = result.tolist()
            result_data["shape"] = result.shape
            result_data["dtype"] = str(result.dtype)
            
            if result.size <= 10:  # Show statistics for small arrays
                result_data["statistics"] = {
                    "mean": float(np.mean(result)),
                    "std": float(np.std(result)),
                    "min": float(np.min(result)),
                    "max": float(np.max(result))
                }
            result_data["steps"].append(f"Array result with shape {result.shape}")
            
        elif isinstance(result, (np.integer, np.floating)):
            result_data["result"] = float(result)
            result_data["steps"].append(f"Scalar result: {result}")
            
        elif isinstance(result, complex):
            result_data["result"] = {"real": result.real, "imag": result.imag}
            result_data["steps"].append(f"Complex result: {result}")
            
        else:
            result_data["result"] = result
            result_data["steps"].append(f"Result: {result}")
        
        return result_data
    
    def _compute_descriptive_statistics(self, data: np.ndarray, 
                                      confidence_level: float) -> Dict[str, Any]:
        """Compute descriptive statistics for data"""
        n = len(data)
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1) if n > 1 else 0
        
        # Confidence interval for mean
        if n > 1:
            t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin_error = t_critical * std_val / np.sqrt(n)
            ci_lower = mean_val - margin_error
            ci_upper = mean_val + margin_error
        else:
            ci_lower = ci_upper = mean_val
        
        return {
            "mean": float(mean_val),
            "median": float(np.median(data)),
            "std": float(std_val),
            "var": float(np.var(data, ddof=1)) if n > 1 else 0,
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "count": n,
            "percentiles": {
                "25th": float(np.percentile(data, 25)),
                "75th": float(np.percentile(data, 75)),
                "95th": float(np.percentile(data, 95)),
                "99th": float(np.percentile(data, 99))
            },
            "confidence_interval": {
                "level": confidence_level,
                "lower": float(ci_lower),
                "upper": float(ci_upper)
            },
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data))
        }
    
    def _fit_distributions(self, data: np.ndarray, distributions: List[str]) -> Dict[str, Any]:
        """Fit multiple distributions and find best fit"""
        best_fit = None
        best_aic = float('inf')
        fit_results = {}
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                
                # Calculate AIC (Akaike Information Criterion)
                log_likelihood = np.sum(dist.logpdf(data, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                
                fit_results[dist_name] = {
                    "parameters": params,
                    "aic": float(aic),
                    "log_likelihood": float(log_likelihood)
                }
                
                if aic < best_aic:
                    best_aic = aic
                    best_fit = dist_name
                    
            except Exception as e:
                fit_results[dist_name] = {"error": str(e)}
        
        return {
            "distributions_tested": distributions,
            "fit_results": fit_results,
            "best_fit": best_fit,
            "best_aic": float(best_aic) if best_fit else None
        }
    
    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        
        return {
            "method": "IQR",
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outliers": outliers.tolist(),
            "outlier_indices": outlier_indices.tolist(),
            "outlier_count": len(outliers),
            "outlier_percentage": float(len(outliers) / len(data) * 100)
        }
    
    def _update_performance_stats(self, computation_time: float, calc_type: CalculationType):
        """Update performance statistics"""
        self.performance_stats["total_operations"] += 1
        
        # Update average computation time
        total_ops = self.performance_stats["total_operations"]
        current_avg = self.performance_stats["average_computation_time"]
        new_avg = (current_avg * (total_ops - 1) + computation_time) / total_ops
        self.performance_stats["average_computation_time"] = new_avg
        
        # Update operations by type
        calc_type_str = calc_type.value
        if calc_type_str not in self.performance_stats["operations_by_type"]:
            self.performance_stats["operations_by_type"][calc_type_str] = 0
        self.performance_stats["operations_by_type"][calc_type_str] += 1
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for numeric computations"""
        return self.performance_stats.copy()