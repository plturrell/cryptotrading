"""
Symbolic Computation Sub-skill for Calculation Agent

Uses SymPy for exact symbolic computation:
- Algebraic manipulation
- Calculus operations (derivatives, integrals)
- Equation solving
- Expression simplification

Advantages for A2A:
- Exact results without floating-point errors
- Readable symbolic expressions for agent communication
- Mathematical verification and proof capabilities
"""

import time
import logging
from typing import Dict, Any, List, Optional
import sympy as sp
from decimal import Decimal

from src.strands.tools import tool
from .types import CalculationResult, ComputationMethod, CalculationType, VerificationStatus
from .utils import classify_calculation_type, extract_variables_from_expression


logger = logging.getLogger(__name__)


class SymbolicComputationSkill:
    """Symbolic computation sub-skill using SymPy"""
    
    def __init__(self):
        self.computation_cache = {}
        self.operation_stats = {
            "evaluate": 0,
            "expand": 0, 
            "factor": 0,
            "solve": 0,
            "differentiate": 0,
            "integrate": 0,
            "simplify": 0
        }
    
    @tool
    def calculate_symbolic(self, expression: str, variables: Dict[str, Any] = None, 
                          operation: str = "evaluate") -> Dict[str, Any]:
        """
        Symbolic computation using SymPy
        
        Args:
            expression: Mathematical expression as string
            variables: Dictionary of variable substitutions
            operation: Type of operation (evaluate, expand, factor, solve, differentiate, integrate)
            
        Returns:
            Dict with symbolic computation result and metadata
        """
        start_time = time.time()
        
        try:
            # Parse symbolic expression
            expr = sp.sympify(expression)
            
            # Substitute variables if provided
            if variables:
                sym_vars = self._prepare_symbolic_variables(variables)
                expr = expr.subs(sym_vars)
            
            # Initialize result data
            result_data = {
                "original_expression": expression,
                "parsed_expression": str(expr),
                "variables": variables or {},
                "operation": operation,
                "steps": []
            }
            
            # Perform the requested operation
            result_data.update(self._perform_symbolic_operation(expr, operation))
            
            # Try to get numeric approximation
            numeric_value = self._get_numeric_approximation(expr)
            if numeric_value is not None:
                result_data["numeric_value"] = numeric_value
            
            # Calculate computation time
            computation_time = time.time() - start_time
            
            # Create structured result
            calc_result = CalculationResult(
                result=result_data["result"],
                method_used=ComputationMethod.SYMBOLIC,
                calculation_type=classify_calculation_type(expression),
                symbolic_form=str(expr),
                numeric_value=numeric_value,
                verification_status=VerificationStatus.PASSED,
                steps=result_data["steps"],
                confidence=1.0,  # Symbolic computation is exact
                computation_time=computation_time
            )
            
            # Update statistics
            self.operation_stats[operation] = self.operation_stats.get(operation, 0) + 1
            
            logger.info(f"Symbolic calculation completed: {operation} on {expression}")
            
            return {
                "success": True,
                "method": "symbolic",
                "computation_time": computation_time,
                **result_data
            }
            
        except Exception as e:
            logger.error(f"Symbolic calculation failed: {e}")
            return {
                "success": False,
                "method": "symbolic",
                "error": str(e),
                "expression": expression,
                "variables": variables,
                "operation": operation
            }
    
    @tool
    def solve_equation(self, equation: str, solve_for: str = None, 
                      domain: str = "complex") -> Dict[str, Any]:
        """
        Solve algebraic equations symbolically
        
        Args:
            equation: Equation to solve (can use = or assume = 0)
            solve_for: Variable to solve for (auto-detected if None)
            domain: Domain for solutions (real, complex, positive, etc.)
            
        Returns:
            Dict with solutions and metadata
        """
        start_time = time.time()
        
        try:
            # Handle equations with = sign
            if "=" in equation:
                lhs, rhs = equation.split("=", 1)
                expr = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                expr = sp.sympify(equation)
            
            # Determine variable to solve for
            free_vars = list(expr.free_symbols)
            if solve_for:
                var = sp.Symbol(solve_for)
                if var not in free_vars:
                    return {
                        "success": False,
                        "error": f"Variable '{solve_for}' not found in equation"
                    }
            elif len(free_vars) == 1:
                var = free_vars[0]
            elif len(free_vars) == 0:
                return {
                    "success": False,
                    "error": "No variables found in equation"
                }
            else:
                return {
                    "success": False,
                    "error": f"Multiple variables found: {[str(v) for v in free_vars]}. Please specify solve_for."
                }
            
            # Solve the equation
            solutions = sp.solve(expr, var, domain=domain if domain != "complex" else sp.S.Complexes)
            
            # Format solutions
            solution_data = {
                "equation": equation,
                "variable": str(var),
                "domain": domain,
                "solutions": [str(sol) for sol in solutions],
                "solution_count": len(solutions),
                "steps": [
                    f"Equation: {equation}",
                    f"Reformulated as: {expr} = 0",
                    f"Solving for: {var}",
                    f"Solutions: {solutions}"
                ]
            }
            
            # Get numeric values if possible
            numeric_solutions = []
            for sol in solutions:
                try:
                    numeric_val = complex(sol.evalf())
                    if numeric_val.imag == 0:
                        numeric_solutions.append(float(numeric_val.real))
                    else:
                        numeric_solutions.append({"real": float(numeric_val.real), "imag": float(numeric_val.imag)})
                except:
                    numeric_solutions.append(str(sol))
            
            solution_data["numeric_solutions"] = numeric_solutions
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "symbolic_solve",
                "computation_time": computation_time,
                **solution_data
            }
            
        except Exception as e:
            logger.error(f"Equation solving failed: {e}")
            return {
                "success": False,
                "method": "symbolic_solve",
                "error": str(e),
                "equation": equation
            }
    
    @tool 
    def perform_calculus(self, expression: str, operation: str, variable: str = None,
                        limits: List = None) -> Dict[str, Any]:
        """
        Perform calculus operations (derivatives, integrals)
        
        Args:
            expression: Mathematical expression
            operation: 'derivative' or 'integral'
            variable: Variable for operation (auto-detected if None)
            limits: Integration limits [lower, upper] for definite integrals
            
        Returns:
            Dict with calculus result
        """
        start_time = time.time()
        
        try:
            expr = sp.sympify(expression)
            
            # Determine variable
            free_vars = list(expr.free_symbols)
            if variable:
                var = sp.Symbol(variable)
            elif len(free_vars) == 1:
                var = free_vars[0]
            else:
                # Default to 'x' if multiple or no variables
                var = sp.Symbol('x')
            
            steps = [f"Expression: {expr}", f"Variable: {var}"]
            
            if operation == "derivative":
                result = sp.diff(expr, var)
                steps.append(f"d/d{var}({expr}) = {result}")
                
            elif operation == "integral":
                if limits:
                    # Definite integral
                    lower, upper = limits
                    result = sp.integrate(expr, (var, lower, upper))
                    steps.append(f"∫[{lower} to {upper}] {expr} d{var} = {result}")
                else:
                    # Indefinite integral
                    result = sp.integrate(expr, var)
                    steps.append(f"∫ {expr} d{var} = {result} + C")
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown calculus operation: {operation}"
                }
            
            # Get numeric value if possible
            numeric_value = self._get_numeric_approximation(result)
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": f"symbolic_{operation}",
                "computation_time": computation_time,
                "original_expression": expression,
                "result": str(result),
                "variable": str(var),
                "operation": operation,
                "limits": limits,
                "steps": steps,
                "numeric_value": numeric_value
            }
            
        except Exception as e:
            logger.error(f"Calculus operation failed: {e}")
            return {
                "success": False,
                "method": f"symbolic_{operation}",
                "error": str(e),
                "expression": expression,
                "operation": operation
            }
    
    @tool
    def manipulate_expression(self, expression: str, operation: str) -> Dict[str, Any]:
        """
        Perform algebraic manipulations on expressions
        
        Args:
            expression: Mathematical expression
            operation: expand, factor, simplify, collect, cancel
            
        Returns:
            Dict with manipulated expression
        """
        start_time = time.time()
        
        try:
            expr = sp.sympify(expression)
            original_expr = str(expr)
            
            steps = [f"Original: {original_expr}"]
            
            if operation == "expand":
                result = sp.expand(expr)
                steps.append(f"Expanded: {result}")
                
            elif operation == "factor":
                result = sp.factor(expr)
                steps.append(f"Factored: {result}")
                
            elif operation == "simplify":
                result = sp.simplify(expr)
                steps.append(f"Simplified: {result}")
                
            elif operation == "collect":
                # Collect terms with respect to first variable found
                free_vars = list(expr.free_symbols)
                if free_vars:
                    var = free_vars[0]
                    result = sp.collect(expr, var)
                    steps.append(f"Collected terms of {var}: {result}")
                else:
                    result = expr
                    steps.append("No variables to collect")
            
            elif operation == "cancel":
                result = sp.cancel(expr)
                steps.append(f"Cancelled: {result}")
                
            elif operation == "apart":
                result = sp.apart(expr)
                steps.append(f"Partial fractions: {result}")
                
            elif operation == "together":
                result = sp.together(expr)
                steps.append(f"Combined fractions: {result}")
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown manipulation operation: {operation}"
                }
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": f"symbolic_{operation}",
                "computation_time": computation_time,
                "original_expression": expression,
                "result": str(result),
                "operation": operation,
                "steps": steps,
                "numeric_value": self._get_numeric_approximation(result)
            }
            
        except Exception as e:
            logger.error(f"Expression manipulation failed: {e}")
            return {
                "success": False,
                "method": f"symbolic_{operation}",
                "error": str(e),
                "expression": expression,
                "operation": operation
            }
    
    def _prepare_symbolic_variables(self, variables: Dict[str, Any]) -> Dict[sp.Symbol, Any]:
        """Prepare variables for SymPy substitution"""
        sym_vars = {}
        for var, val in variables.items():
            symbol = sp.Symbol(var)
            if isinstance(val, (int, float)):
                sym_vars[symbol] = val
            elif isinstance(val, str):
                sym_vars[symbol] = sp.sympify(val)
            elif isinstance(val, Decimal):
                sym_vars[symbol] = float(val)
            else:
                sym_vars[symbol] = val
        return sym_vars
    
    def _perform_symbolic_operation(self, expr: sp.Expr, operation: str) -> Dict[str, Any]:
        """Perform the specified symbolic operation"""
        result_data = {"steps": []}
        
        if operation == "evaluate":
            result_data["result"] = str(expr)
            simplified = sp.simplify(expr)
            result_data["simplified"] = str(simplified)
            result_data["steps"].extend([
                f"Parsed: {expr}",
                f"Simplified: {simplified}"
            ])
            
        elif operation == "expand":
            expanded = sp.expand(expr)
            result_data["result"] = str(expanded)
            result_data["steps"].extend([
                f"Original: {expr}",
                f"Expanded: {expanded}"
            ])
            
        elif operation == "factor":
            factored = sp.factor(expr)
            result_data["result"] = str(factored)
            result_data["steps"].extend([
                f"Original: {expr}",
                f"Factored: {factored}"
            ])
            
        elif operation == "solve":
            free_vars = list(expr.free_symbols)
            if free_vars:
                var = free_vars[0]
                solutions = sp.solve(expr, var)
                result_data["result"] = [str(sol) for sol in solutions]
                result_data["solved_for"] = str(var)
                result_data["steps"].extend([
                    f"Solving {expr} = 0 for {var}",
                    f"Solutions: {solutions}"
                ])
            else:
                result_data["result"] = "No variables to solve for"
                result_data["steps"].append("No variables found in expression")
                
        elif operation == "differentiate":
            free_vars = list(expr.free_symbols)
            if free_vars:
                var = free_vars[0]
                derivative = sp.diff(expr, var)
                result_data["result"] = str(derivative)
                result_data["differentiated_with_respect_to"] = str(var)
                result_data["steps"].extend([
                    f"d/d{var}({expr})",
                    f"= {derivative}"
                ])
            else:
                result_data["result"] = "0"
                result_data["steps"].append("Derivative of constant is 0")
                
        elif operation == "integrate":
            free_vars = list(expr.free_symbols)
            if free_vars:
                var = free_vars[0]
                integral = sp.integrate(expr, var)
                result_data["result"] = str(integral)
                result_data["integrated_with_respect_to"] = str(var)
                result_data["steps"].extend([
                    f"∫({expr})d{var}",
                    f"= {integral} + C"
                ])
            else:
                # Integral of constant
                result_data["result"] = str(expr * sp.Symbol('x'))
                result_data["steps"].append(f"∫{expr}dx = {expr}x + C")
        
        elif operation == "simplify":
            simplified = sp.simplify(expr)
            result_data["result"] = str(simplified)
            result_data["steps"].extend([
                f"Original: {expr}",
                f"Simplified: {simplified}"
            ])
        
        else:
            result_data["result"] = str(expr)
            result_data["steps"].append(f"Unknown operation '{operation}', returning expression as-is")
        
        return result_data
    
    def _get_numeric_approximation(self, expr: sp.Expr) -> Optional[Any]:
        """Get numeric approximation of symbolic expression"""
        try:
            numeric_val = complex(expr.evalf())
            if abs(numeric_val.imag) < 1e-15:
                return float(numeric_val.real)
            else:
                return {
                    "real": float(numeric_val.real), 
                    "imag": float(numeric_val.imag)
                }
        except:
            try:
                # Try evaluating as float directly
                return float(expr.evalf())
            except:
                return None
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about symbolic operations performed"""
        total_operations = sum(self.operation_stats.values())
        return {
            "total_operations": total_operations,
            "operation_breakdown": self.operation_stats.copy(),
            "most_used_operation": max(self.operation_stats.items(), key=lambda x: x[1])[0] if total_operations > 0 else None
        }