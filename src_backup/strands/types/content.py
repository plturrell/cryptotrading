"""
Content types for Strands
"""
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: Union[str, List[Dict[str, Any]]]

@dataclass
class ToolUse:
    toolUseId: str
    name: str
    input: Dict[str, Any]

@dataclass
class ToolResult:
    toolUseId: str
    output: Any
    isError: bool = False

@dataclass
class ContentBlockStartToolUse:
    toolUseId: str
    name: str