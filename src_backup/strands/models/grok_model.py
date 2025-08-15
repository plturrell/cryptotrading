"""
Grok Model implementation for Strands framework
"""
import os
import json
import uuid
from typing import List, Dict, Any, Optional, AsyncIterable, Type
from .model import Model
from ..types.content import Message
from ..types.streaming import (
    MessageStartEvent, ContentBlockStart, ContentBlockDelta, 
    ContentBlockStopEvent, MessageStopEvent, ContentBlockDeltaToolUse
)
from ..types.tools import ToolSpec
from ...rex.a2a.grok4_client import get_grok4_client


class GrokModel(Model):
    """Grok-4 model implementation for strands framework"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.client = get_grok4_client()
        self.config = {
            "model": "grok-4-latest",
            "temperature": 0.3,
            "max_tokens": 4096
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.copy()
    
    def update_config(self, **model_config: Any) -> None:
        """Update model configuration"""
        self.config.update(model_config)
    
    async def stream(
        self,
        messages: List[Message],
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterable[Any]:
        """Stream model response with proper strands events"""
        try:
            # Prepare messages for Grok API
            api_messages = []
            
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            
            for msg in messages:
                if isinstance(msg, Message):
                    # Handle Message objects with content list
                    if isinstance(msg.content, list):
                        content_text = ""
                        for content_block in msg.content:
                            if isinstance(content_block, dict):
                                if 'text' in content_block:
                                    content_text += content_block['text']
                                elif 'toolResult' in content_block:
                                    # Handle tool results
                                    tool_result = content_block['toolResult']
                                    content_text += f"Tool result: {tool_result.get('output', '')}"
                            else:
                                content_text += str(content_block)
                        api_messages.append({"role": msg.role, "content": content_text})
                    else:
                        api_messages.append({"role": msg.role, "content": str(msg.content)})
                elif isinstance(msg, dict):
                    api_messages.append(msg)
                else:
                    api_messages.append({"role": "user", "content": str(msg)})
            
            # Convert tool specs to Grok format if provided
            tools = None
            if tool_specs:
                tools = []
                for spec in tool_specs:
                    if isinstance(spec, ToolSpec):
                        tool_def = {
                            "type": "function",
                            "function": {
                                "name": spec.name,
                                "description": spec.description,
                                "parameters": {
                                    "type": "object",
                                    "properties": spec.parameters,
                                    "required": list(spec.parameters.keys()) if spec.parameters else []
                                }
                            }
                        }
                        tools.append(tool_def)
            
            # Emit message start event
            yield MessageStartEvent(
                messageId=str(uuid.uuid4()),
                role="assistant"
            )
            
            # Make API call to Grok
            response = await self.client.complete(
                messages=api_messages,
                tools=tools,
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 4096),
                **kwargs
            )
            
            if response.get("success"):
                content = response["content"]
                tool_calls = response.get("tool_calls", [])
                
                if tool_calls:
                    # Handle tool calls
                    for tool_call in tool_calls:
                        tool_use_id = str(uuid.uuid4())
                        
                        # Start tool use block
                        yield ContentBlockStart(
                            index=0,
                            text=None,
                            toolUse={
                                "toolUseId": tool_use_id,
                                "name": tool_call.get("name", "unknown"),
                                "input": {}
                            }
                        )
                        
                        # Tool use delta with input
                        if "arguments" in tool_call:
                            yield ContentBlockDelta(
                                index=0,
                                text=None,
                                toolUse=ContentBlockDeltaToolUse(
                                    input=json.dumps(tool_call["arguments"])
                                )
                            )
                        
                        # Stop tool use block
                        yield ContentBlockStopEvent(index=0)
                else:
                    # Handle text content
                    if content:
                        # Start text block
                        yield ContentBlockStart(
                            index=0,
                            text="",
                            toolUse=None
                        )
                        
                        # Stream text content in chunks
                        chunk_size = 50
                        for i in range(0, len(content), chunk_size):
                            chunk = content[i:i + chunk_size]
                            yield ContentBlockDelta(
                                index=0,
                                text=chunk,
                                toolUse=None
                            )
                        
                        # Stop text block
                        yield ContentBlockStopEvent(index=0)
                
                # Emit message stop event
                yield MessageStopEvent(
                    stopReason="end_turn",
                    usage={
                        "inputTokens": response.get("usage", {}).get("prompt_tokens", 0),
                        "outputTokens": response.get("usage", {}).get("completion_tokens", 0),
                        "totalTokens": response.get("usage", {}).get("total_tokens", 0)
                    }
                )
            else:
                # Handle error case
                error_msg = response.get("error", "Unknown error")
                
                yield ContentBlockStart(
                    index=0,
                    text="",
                    toolUse=None
                )
                
                yield ContentBlockDelta(
                    index=0,
                    text=f"Error: {error_msg}",
                    toolUse=None
                )
                
                yield ContentBlockStopEvent(index=0)
                
                yield MessageStopEvent(
                    stopReason="error",
                    usage={"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
                )
                
        except Exception as e:
            # Handle exception case
            error_msg = str(e)
            
            yield MessageStartEvent(
                messageId=str(uuid.uuid4()),
                role="assistant"
            )
            
            yield ContentBlockStart(
                index=0,
                text="",
                toolUse=None
            )
            
            yield ContentBlockDelta(
                index=0,
                text=f"Error: {error_msg}",
                toolUse=None
            )
            
            yield ContentBlockStopEvent(index=0)
            
            yield MessageStopEvent(
                stopReason="error",
                usage={"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
            )
    
    async def structured_output(
        self,
        output_model: Type,
        prompt: List[Any],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterable[Dict[str, Any]]:
        """Generate structured output"""
        # For now, delegate to stream method
        async for response in self.stream(prompt, system_prompt=system_prompt, **kwargs):
            yield response
