"""
Streaming event types for Strands
"""
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# Base event type
@dataclass
class StreamEvent:
    pass

@dataclass
class MessageStartEvent(StreamEvent):
    role: str

@dataclass
class MessageStopEvent(StreamEvent):
    stopReason: str
    additionalModelResponseFields: Dict[str, Any]

@dataclass
class ContentBlockStart(StreamEvent):
    toolUse: Optional['ContentBlockStartToolUse'] = None
    index: Optional[int] = None

@dataclass
class ContentBlockDelta(StreamEvent):
    text: Optional[str] = None
    toolUse: Optional['ContentBlockDeltaToolUse'] = None
    reasoningContent: Optional[str] = None

@dataclass
class ContentBlockStopEvent(StreamEvent):
    contentBlockIndex: int

@dataclass
class ContentBlockDeltaToolUse:
    input: str

# Import to avoid circular dependency
from .content import ContentBlockStartToolUse