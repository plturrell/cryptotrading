"""
Shared Utility Functions for Calculation Agent

Common helper functions used across sub-skills
"""

import re
import numpy as np
from typing import Dict, Any, List, Union, Optional
from .types import CalculationType, ComputationMethod


def classify_calculation_type(expression: str) -> CalculationType:
    """Classify the type of calculation based on expression"""
    expression_lower = expression.lower()
    
    # Check for calculus operations
    if any(op in expression_lower for op in ["sin", "cos", "tan", "log", "exp", "sqrt", "diff", "integrate"]):
        return CalculationType.CALCULUS
    
    # Check for matrix operations  
    if any(op in expression_lower for op in ["dot", "matmul", "@", "inv", "det", "transpose"]):
        return CalculationType.MATRIX
        
    # Check for statistical operations
    if any(op in expression_lower for op in ["mean", "std", "var", "median", "percentile", "normal", "distribution"]):
        return CalculationType.STATISTICAL
        
    # Check for algebraic operations (variables)
    if any(var in expression_lower for var in ["x", "y", "z", "solve", "factor", "expand"]):
        return CalculationType.ALGEBRAIC
        
    # Check for optimization
    if any(op in expression_lower for op in ["minimize", "maximize", "optimize", "constraint"]):
        return CalculationType.OPTIMIZATION
        
    # Default to arithmetic
    return CalculationType.ARITHMETIC


def auto_select_computation_method(expression: str, variables: Dict[str, Any] = None, 
                                 symbolic_threshold: float = 1e6, 
                                 complexity_threshold: int = 10) -> ComputationMethod:
    """Auto-select optimal computation method based on problem characteristics"""
    
    # Check for large numbers that might cause symbolic computation issues
    numbers = re.findall(r'\d+\.?\d*', expression)
    if numbers:
        max_number = max(float(n) for n in numbers if float(n) != 0)
        if max_number > symbolic_threshold:
            return ComputationMethod.NUMERIC
    
    # Check expression complexity
    complexity = assess_expression_complexity(expression)
    if complexity > complexity_threshold:
        return ComputationMethod.NUMERIC
    
    # Check for variables that require exact symbolic manipulation
    if variables and any(isinstance(v, str) for v in variables.values()):
        return ComputationMethod.SYMBOLIC
    
    # Check for calculus operations that benefit from symbolic computation
    if classify_calculation_type(expression) in [CalculationType.CALCULUS, CalculationType.ALGEBRAIC]:
        return ComputationMethod.SYMBOLIC
    
    # Default to hybrid for verification
    return ComputationMethod.HYBRID


def assess_expression_complexity(expression: str) -> int:
    """Assess complexity score of mathematical expression"""
    complexity = 0
    
    # Parentheses increase complexity
    complexity += expression.count("(") * 2
    complexity += expression.count("[") * 2
    
    # Trigonometric functions
    complexity += len(re.findall(r'\b(sin|cos|tan|asin|acos|atan)\b', expression, re.IGNORECASE))
    
    # Logarithmic and exponential functions
    complexity += len(re.findall(r'\b(log|ln|exp|sqrt)\b', expression, re.IGNORECASE))
    
    # Power operations
    complexity += expression.count("**") * 2
    complexity += expression.count("^") * 2
    
    # Derivatives and integrals
    complexity += len(re.findall(r'\b(diff|integrate|derivative|integral)\b', expression, re.IGNORECASE)) * 5
    
    # Matrix operations
    complexity += len(re.findall(r'\b(dot|matmul|inv|det|transpose)\b', expression, re.IGNORECASE)) * 3
    
    return complexity


def extract_variables_from_expression(expression: str) -> List[str]:
    """Extract variable names from mathematical expression"""
    # Find single letter variables (common in math)
    variables = re.findall(r'\b[a-zA-Z]\b', expression)
    
    # Find multi-letter variables (like theta, alpha, etc.)
    multi_letter = re.findall(r'\b[a-zA-Z]{2,}\b', expression)
    
    # Filter out known functions
    known_functions = {
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'log', 'ln', 'exp', 'sqrt', 'abs',
        'min', 'max', 'sum', 'mean', 'std',
        'diff', 'integrate', 'solve', 'factor', 'expand'
    }
    
    variables.extend([v for v in multi_letter if v.lower() not in known_functions])
    
    # Remove duplicates and constants
    constants = {'pi', 'e'}
    return list(set(v for v in variables if v.lower() not in constants))


def format_calculation_steps(steps: List[str], title: str = "Calculation Steps") -> str:
    """Format calculation steps for display"""
    if not steps:
        return f"{title}: No steps recorded"
    
    formatted = f"{title}:\n"
    for i, step in enumerate(steps, 1):
        formatted += f"  {i}. {step}\n"
    
    return formatted


def validate_numeric_result(result: Union[float, complex, np.ndarray]) -> bool:
    """Validate that numeric result is valid (not NaN, not infinite)"""
    try:
        if isinstance(result, np.ndarray):
            return not (np.isnan(result).any() or np.isinf(result).any())
        elif isinstance(result, complex):
            return not (np.isnan(result.real) or np.isnan(result.imag) or 
                       np.isinf(result.real) or np.isinf(result.imag))
        elif isinstance(result, (int, float)):
            return not (np.isnan(result) or np.isinf(result))
        else:
            return True  # Assume valid for other types
    except:
        return False


def sanitize_expression_for_eval(expression: str, allowed_names: Dict[str, Any]) -> str:
    """Sanitize mathematical expression for safe evaluation"""
    # Replace common mathematical notation
    sanitized = expression.replace("^", "**")  # Power notation
    sanitized = re.sub(r'\b(\d+)([a-zA-Z])\b', r'\1*\2', sanitized)  # Implicit multiplication like 2x -> 2*x
    
    # Ensure only allowed names are used
    for name in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sanitized):
        if name not in allowed_names:
            raise ValueError(f"Unauthorized name '{name}' in expression")
    
    return sanitized


def compare_numeric_results(result1: Union[float, complex, List], 
                          result2: Union[float, complex, List],
                          tolerance: float = 1e-10) -> Dict[str, Any]:
    """Compare two numeric results within tolerance"""
    try:
        # Convert to numpy arrays for easier comparison
        arr1 = np.atleast_1d(np.array(result1))
        arr2 = np.atleast_1d(np.array(result2))
        
        if arr1.shape != arr2.shape:
            return {
                "match": False,
                "reason": "Shape mismatch",
                "discrepancy": None
            }
        
        # Calculate absolute difference
        diff = np.abs(arr1 - arr2)
        max_diff = np.max(diff)
        
        # Check if within tolerance
        within_tolerance = max_diff <= tolerance
        
        return {
            "match": bool(within_tolerance),
            "max_discrepancy": float(max_diff),
            "mean_discrepancy": float(np.mean(diff)),
            "tolerance": tolerance,
            "reason": "Within tolerance" if within_tolerance else "Exceeds tolerance"
        }
        
    except Exception as e:
        return {
            "match": False,
            "reason": f"Comparison failed: {str(e)}",
            "discrepancy": None
        }


def create_safe_math_context():
    """Create safe mathematical computation context for eval()"""
    return {
        "__builtins__": {},
        # NumPy functions
        "np": np,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "log": np.log,
        "log10": np.log10,
        "exp": np.exp,
        "abs": np.abs,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
        "mean": np.mean,
        "std": np.std,
        "var": np.var,
        "median": np.median,
        # Constants
        "pi": np.pi,
        "e": np.e,
        # Basic operations
        "pow": np.power,
        "round": np.round,
        "floor": np.floor,
        "ceil": np.ceil,
    }


def generate_calculation_id(expression: str, variables: Dict[str, Any] = None) -> str:
    """Generate unique ID for calculation"""
    import hashlib
    
    content = expression
    if variables:
        content += str(sorted(variables.items()))
    
    return hashlib.md5(content.encode()).hexdigest()[:12]


def format_result_for_display(result: Any, precision: int = 6) -> str:
    """Format calculation result for human-readable display"""
    if isinstance(result, float):
        if abs(result) < 1e-10:
            return "0"
        elif abs(result) > 1e6 or abs(result) < 1e-4:
            return f"{result:.{precision}e}"
        else:
            return f"{result:.{precision}f}".rstrip('0').rstrip('.')
    
    elif isinstance(result, complex):
        real_str = format_result_for_display(result.real, precision)
        imag_str = format_result_for_display(abs(result.imag), precision)
        sign = "+" if result.imag >= 0 else "-"
        return f"{real_str} {sign} {imag_str}i"
    
    elif isinstance(result, list):
        if len(result) <= 5:
            return "[" + ", ".join(format_result_for_display(r, precision) for r in result) + "]"
        else:
            first_few = ", ".join(format_result_for_display(r, precision) for r in result[:3])
            return f"[{first_few}, ... ({len(result)} elements)]"
    
    else:
        return str(result)


def estimate_computation_cost(expression: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
    """Estimate computational cost for different methods"""
    complexity = assess_expression_complexity(expression)
    calc_type = classify_calculation_type(expression)
    
    # Rough cost estimates (relative scale 1-100)
    symbolic_cost = complexity * 5
    numeric_cost = complexity * 2
    
    # Adjust based on calculation type
    if calc_type == CalculationType.CALCULUS:
        symbolic_cost *= 2
    elif calc_type == CalculationType.MATRIX:
        numeric_cost *= 3
    elif calc_type == CalculationType.STATISTICAL:
        numeric_cost *= 1.5
    
    # Factor in data size
    if variables:
        for var, val in variables.items():
            if isinstance(val, (list, tuple)):
                data_size_factor = len(val) / 1000.0
                numeric_cost *= (1 + data_size_factor)
                symbolic_cost *= (1 + data_size_factor * 2)  # Symbolic is more affected by data size
    
    return {
        "symbolic_cost": min(symbolic_cost, 100),
        "numeric_cost": min(numeric_cost, 100),
        "recommended_method": "numeric" if numeric_cost < symbolic_cost else "symbolic",
        "complexity_score": complexity
    }