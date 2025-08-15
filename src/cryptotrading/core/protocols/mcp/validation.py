"""
Parameter validation for MCP tools
Validates tool inputs against JSON schemas
"""
import jsonschema
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ParameterValidator:
    """Validates tool parameters against JSON schemas"""
    
    def __init__(self):
        self.draft = jsonschema.Draft7Validator
    
    def validate_parameters(self, parameters: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate parameters against schema
        
        Args:
            parameters: Parameter values to validate
            schema: JSON schema to validate against
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Create validator with schema
            validator = self.draft(schema)
            
            # Validate parameters
            validation_errors = sorted(validator.iter_errors(parameters), key=lambda e: e.path)
            
            for error in validation_errors:
                path = ".".join(str(p) for p in error.path)
                if path:
                    errors.append(f"Parameter '{path}': {error.message}")
                else:
                    errors.append(f"Root: {error.message}")
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_tool_call(self, tool_schema: Dict[str, Any], arguments: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate tool call arguments against tool schema
        
        Args:
            tool_schema: Tool's input schema
            arguments: Arguments passed to tool
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if "inputSchema" in tool_schema:
            return self.validate_parameters(arguments, tool_schema["inputSchema"])
        elif "parameters" in tool_schema:
            # Convert simple parameters to JSON schema
            schema = {
                "type": "object",
                "properties": tool_schema["parameters"],
                "required": list(tool_schema["parameters"].keys())
            }
            return self.validate_parameters(arguments, schema)
        else:
            # No schema to validate against
            return True, []
    
    def validate_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate that value matches expected type
        
        Args:
            value: Value to validate
            expected_type: Expected JSON schema type
            
        Returns:
            True if value matches type
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return False
        
        return isinstance(value, expected_python_type)
    
    def coerce_type(self, value: Any, expected_type: str) -> Any:
        """
        Attempt to coerce value to expected type
        
        Args:
            value: Value to coerce
            expected_type: Target type
            
        Returns:
            Coerced value or original value if coercion fails
        """
        try:
            if expected_type == "string":
                return str(value)
            elif expected_type == "integer":
                if isinstance(value, str):
                    return int(float(value))  # Handle "1.0" -> 1
                return int(value)
            elif expected_type == "number":
                return float(value)
            elif expected_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif expected_type == "array":
                if isinstance(value, str):
                    # Try to parse as JSON array
                    import json
                    return json.loads(value)
                return list(value)
            elif expected_type == "object":
                if isinstance(value, str):
                    # Try to parse as JSON object
                    import json
                    return json.loads(value)
                return dict(value)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
        
        return value
    
    def validate_and_coerce_parameters(self, parameters: Dict[str, Any], 
                                     tool_schema: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """
        Validate and coerce parameters, returning cleaned parameters and errors
        
        Args:
            parameters: Input parameters
            tool_schema: Tool schema
            
        Returns:
            Tuple of (cleaned_parameters, error_messages)
        """
        errors = []
        cleaned = {}
        
        # Get parameter definitions
        param_defs = {}
        if "inputSchema" in tool_schema and "properties" in tool_schema["inputSchema"]:
            param_defs = tool_schema["inputSchema"]["properties"]
        elif "parameters" in tool_schema:
            param_defs = tool_schema["parameters"]
        
        # Process each parameter
        for param_name, param_value in parameters.items():
            if param_name in param_defs:
                param_def = param_defs[param_name]
                expected_type = param_def.get("type")
                
                if expected_type:
                    # Try to coerce type
                    coerced_value = self.coerce_type(param_value, expected_type)
                    
                    # Validate coerced value
                    if self.validate_type(coerced_value, expected_type):
                        cleaned[param_name] = coerced_value
                    else:
                        errors.append(f"Parameter '{param_name}': expected {expected_type}, got {type(param_value).__name__}")
                        cleaned[param_name] = param_value  # Keep original
                else:
                    cleaned[param_name] = param_value
            else:
                # Unknown parameter
                errors.append(f"Unknown parameter: '{param_name}'")
                cleaned[param_name] = param_value
        
        # Check for required parameters
        required = []
        if "inputSchema" in tool_schema:
            required = tool_schema["inputSchema"].get("required", [])
        elif "parameters" in tool_schema:
            # Assume all parameters are required unless they have defaults
            required = [name for name, def_ in param_defs.items() 
                       if "default" not in def_]
        
        for req_param in required:
            if req_param not in parameters:
                errors.append(f"Missing required parameter: '{req_param}'")
        
        return cleaned, errors


def create_validator() -> ParameterValidator:
    """Create a parameter validator instance"""
    return ParameterValidator()


# Enhanced MCPTool with validation
class ValidatedMCPTool:
    """MCP Tool with parameter validation"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], 
                 function: Optional[callable] = None, validate_params: bool = True):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.validate_params = validate_params
        self.validator = create_validator() if validate_params else None
    
    async def execute(self, arguments: Dict[str, Any]):
        """Execute tool with validation"""
        from .tools import ToolResult
        
        if self.validate_params and self.validator:
            # Validate and coerce parameters
            cleaned_args, errors = self.validator.validate_and_coerce_parameters(
                arguments, {"parameters": self.parameters}
            )
            
            if errors:
                error_msg = "Parameter validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
                return ToolResult.error_result(error_msg)
            
            arguments = cleaned_args
        
        # Execute function
        if not self.function:
            return ToolResult.error_result(f"No function defined for tool '{self.name}'")
        
        try:
            import asyncio
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(**arguments)
            else:
                result = self.function(**arguments)
            
            # Convert result to ToolResult if needed
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, dict):
                return ToolResult.json_result(result)
            elif isinstance(result, str):
                return ToolResult.text_result(result)
            else:
                return ToolResult.text_result(str(result))
                
        except Exception as e:
            return ToolResult.error_result(str(e))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to MCP format"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()) if self.parameters else []
            }
        }