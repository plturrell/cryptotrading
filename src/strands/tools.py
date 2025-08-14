"""
Tool decorator for Strands
"""
from functools import wraps
from typing import Callable, Dict, Any, Optional

def tool(func: Callable = None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to mark a function as a tool for agents
    
    Can be used as @tool or @tool(name="...", description="...")
    
    Args:
        func: The function to decorate (when used as @tool)
        name: Optional tool name (defaults to function name)
        description: Optional tool description (defaults to docstring)
    """
    def decorator(f: Callable) -> Callable:
        tool_name = name or f.__name__
        tool_desc = description or f.__doc__ or ""
        
        # Add metadata to function
        f.__tool_name__ = tool_name
        f.__tool_description__ = tool_desc
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # Preserve function attributes
        wrapper.__name__ = tool_name
        wrapper.__doc__ = tool_desc
        wrapper.__wrapped__ = f
        wrapper.__tool_name__ = tool_name
        wrapper.__tool_description__ = tool_desc
        
        return wrapper
    
    # Handle both @tool and @tool() usage
    if func is None:
        # Called as @tool() or @tool(name=..., description=...)
        return decorator
    else:
        # Called as @tool
        return decorator(func)