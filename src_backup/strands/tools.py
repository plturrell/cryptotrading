"""
Tool utilities for Strands framework with automatic schema generation
"""
import inspect
import functools
from typing import Callable, Dict, Any, Optional, Union, get_type_hints, get_origin, get_args
from .types.tools import ToolSpec


def _extract_parameter_schema(func: Callable) -> Dict[str, Any]:
    """
    Extract parameter schema from function signature using type hints
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    schema = {}
    
    for param_name, param in sig.parameters.items():
        if param_name in ['self', 'cls']:
            continue
            
        param_schema = {}
        
        # Get type information
        param_type = type_hints.get(param_name, type(None))
        
        # Convert Python types to JSON schema types
        if param_type == str:
            param_schema['type'] = 'string'
        elif param_type == int:
            param_schema['type'] = 'integer'
        elif param_type == float:
            param_schema['type'] = 'number'
        elif param_type == bool:
            param_schema['type'] = 'boolean'
        elif param_type == list or get_origin(param_type) == list:
            param_schema['type'] = 'array'
            # Try to get array item type
            args = get_args(param_type)
            if args:
                if args[0] == str:
                    param_schema['items'] = {'type': 'string'}
                elif args[0] == int:
                    param_schema['items'] = {'type': 'integer'}
                elif args[0] == float:
                    param_schema['items'] = {'type': 'number'}
        elif param_type == dict or get_origin(param_type) == dict:
            param_schema['type'] = 'object'
        elif get_origin(param_type) == Union:
            # Handle Optional types (Union[X, None])
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                # This is Optional[X]
                non_none_type = args[0] if args[1] == type(None) else args[1]
                if non_none_type == str:
                    param_schema['type'] = 'string'
                elif non_none_type == int:
                    param_schema['type'] = 'integer'
                elif non_none_type == float:
                    param_schema['type'] = 'number'
                elif non_none_type == bool:
                    param_schema['type'] = 'boolean'
                else:
                    param_schema['type'] = 'string'  # fallback
            else:
                param_schema['type'] = 'string'  # fallback for complex unions
        else:
            param_schema['type'] = 'string'  # fallback
        
        # Add description from parameter annotation
        param_schema['description'] = f"Parameter of type {param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)}"
        
        # Handle default values
        if param.default != inspect.Parameter.empty:
            param_schema['default'] = param.default
        
        schema[param_name] = param_schema
    
    return schema


def _extract_description_from_docstring(func: Callable) -> tuple[str, Dict[str, str]]:
    """
    Extract tool description and parameter descriptions from docstring
    """
    docstring = func.__doc__ or ""
    lines = docstring.strip().split('\n')
    
    if not lines:
        return func.__name__, {}
    
    # First line is usually the main description
    description = lines[0].strip()
    
    # Look for Args: section for parameter descriptions
    param_descriptions = {}
    in_args_section = False
    
    for line in lines[1:]:
        line = line.strip()
        if line.lower().startswith('args:'):
            in_args_section = True
            continue
        elif line.lower().startswith(('returns:', 'return:', 'raises:', 'examples:')):
            in_args_section = False
            continue
        
        if in_args_section and ':' in line:
            # Parse parameter description: "param_name: description"
            parts = line.split(':', 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                param_desc = parts[1].strip()
                param_descriptions[param_name] = param_desc
    
    return description, param_descriptions


def tool(func: Callable = None, *, name: Optional[str] = None, description: Optional[str] = None, 
         parameters: Optional[Dict[str, Any]] = None):
    """
    Decorator to mark functions as tools for strands agents with automatic schema generation
    
    Can be used as @tool or @tool(name="...", description="...")
    
    Args:
        func: The function to decorate (when used as @tool)
        name: Tool name (defaults to function name)
        description: Tool description (auto-extracted from docstring)
        parameters: Tool parameters schema (auto-generated from function signature)
    """
    def decorator(f: Callable) -> Callable:
        tool_name = name or f.__name__
        
        # Auto-extract description and parameter descriptions
        auto_description, param_descriptions = _extract_description_from_docstring(f)
        tool_description = description or auto_description
        
        # Auto-generate parameter schema if not provided
        if parameters is None:
            tool_parameters = _extract_parameter_schema(f)
            
            # Enhance with descriptions from docstring
            for param_name, param_desc in param_descriptions.items():
                if param_name in tool_parameters:
                    tool_parameters[param_name]['description'] = param_desc
        else:
            tool_parameters = parameters
        
        # Create ToolSpec and attach it to the function
        f._tool_spec = ToolSpec(
            name=tool_name,
            description=tool_description,
            parameters=tool_parameters,
            function=f
        )
        
        # Legacy attributes for backward compatibility
        f.__tool_name__ = tool_name
        f.__tool_description__ = tool_description
        f._is_tool = True
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # Preserve function attributes
        wrapper.__name__ = tool_name
        wrapper.__doc__ = tool_description
        wrapper.__wrapped__ = f
        wrapper.__tool_name__ = tool_name
        wrapper.__tool_description__ = tool_description
        wrapper._tool_spec = f._tool_spec
        wrapper._is_tool = True
        
        return wrapper
    
    # Handle both @tool and @tool() usage
    if func is None:
        # Called as @tool() or @tool(name=..., description=...)
        return decorator
    else:
        # Called as @tool
        return decorator(func)


def get_tool_spec(func: Callable) -> Optional[ToolSpec]:
    """Get ToolSpec from a decorated function"""
    return getattr(func, '_tool_spec', None)


def is_tool(func: Callable) -> bool:
    """Check if function is decorated as a tool"""
    return getattr(func, '_is_tool', False)