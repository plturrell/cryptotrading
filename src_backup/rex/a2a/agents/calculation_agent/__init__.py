"""
Calculation Agent Package

A2A Hybrid Calculation Agent with modular sub-skills for:
- Symbolic computation (SymPy)
- Numeric computation (NumPy/SciPy) 
- Cross-verification between methods
- Step-by-step reasoning
- A2A coordination
- Financial domain calculations

File Organization:
- main.py - Main A2A agent orchestrator
- symbolic_skill.py - Symbolic computation sub-skill
- numeric_skill.py - Numeric computation sub-skill  
- verification_skill.py - Cross-verification sub-skill
- reasoning_skill.py - Step-by-step reasoning sub-skill
- coordination_skill.py - A2A coordination sub-skill
- financial_skill.py - Financial domain calculations
- types.py - Shared data types and enums
- utils.py - Shared utility functions
"""

from .main import CalculationAgent, get_calculation_agent

__all__ = ['CalculationAgent', 'get_calculation_agent']