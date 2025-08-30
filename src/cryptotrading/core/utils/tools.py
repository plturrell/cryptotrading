"""
Tool utilities for crypto trading agents
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for agent tools"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Register a tool in the registry"""
        self.tools[name] = {
            "name": name,
            "function": func,
            "description": description,
            "parameters": parameters or {},
            **kwargs
        }
        logger.debug(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        try:
            return tool["function"](**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            raise


def create_tool_schema(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a tool schema for MCP or other tool protocols"""
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": parameters,
            "required": required or []
        }
    }


def validate_tool_parameters(parameters: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate tool parameters against schema"""
    # Simple validation - in production this would be more comprehensive
    required = schema.get("required", [])
    
    for req_param in required:
        if req_param not in parameters:
            return False
    
    return True


def format_tool_result(result: Any, success: bool = True, error: Optional[str] = None) -> Dict[str, Any]:
    """Format tool execution result"""
    return {
        "success": success,
        "result": result,
        "error": error,
        "timestamp": logger.time() if hasattr(logger, 'time') else None
    }


class ToolSpec:
    """Tool specification class for STRANDS agents"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }