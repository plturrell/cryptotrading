"""
Tool Specifications and Utilities
Defines tool interfaces and specifications for agents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    """Specification for agent tools"""

    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    returns: Dict[str, Any]

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate tool parameters"""
        for required_param in self.required_params:
            if required_param not in params:
                return False
        return True


class BaseTool(ABC):
    """Base class for agent tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        pass

    def get_spec(self) -> ToolSpec:
        """Get tool specification"""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={},
            required_params=[],
            returns={},
        )


class ToolRegistry:
    """Registry for managing tools"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())
