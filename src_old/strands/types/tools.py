"""
Tool types for Strands
"""
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None