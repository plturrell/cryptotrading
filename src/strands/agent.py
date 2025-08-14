"""
Strands Agent Implementation
"""
import asyncio
import json
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from .models.model import Model
from .types.content import Message, ToolUse, ToolResult
from .types.tools import ToolSpec
from .types.streaming import (
    ContentBlockStart, ContentBlockDelta, ContentBlockStopEvent,
    MessageStartEvent, MessageStopEvent, ContentBlockDeltaToolUse
)

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return super().default(obj)

@dataclass
class AgentResult:
    """Result from agent execution"""
    message: Dict[str, Any]
    stop_reason: str = "end_turn"
    metrics: Optional[Any] = None
    
    def __str__(self):
        content = self.message.get('content', [])
        if content:
            # Extract text from content blocks
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if 'text' in block:
                        texts.append(block['text'])
                    elif 'toolUse' in block:
                        tool_use = block['toolUse']
                        texts.append(f"Tool call: {tool_use.get('name', 'unknown')}")
            return ' '.join(texts) if texts else "No content"
        return "No content"

class Agent:
    """Strands Agent that can use tools and models"""
    
    def __init__(self, tools: List[Union[Callable, ToolSpec]] = None, model: Model = None):
        self.tools = tools or []
        self.model = model
        self._tool_map = self._build_tool_map()
        
    def _build_tool_map(self) -> Dict[str, Callable]:
        """Build a map of tool names to functions"""
        tool_map = {}
        for tool in self.tools:
            if isinstance(tool, ToolSpec):
                tool_map[tool.name] = tool.function
            elif hasattr(tool, '__wrapped__') and hasattr(tool.__wrapped__, '__name__'):
                # Handle decorated functions - use the wrapped function name
                tool_map[tool.__wrapped__.__name__] = tool
            elif hasattr(tool, '__name__'):
                tool_map[tool.__name__] = tool
        return tool_map
    
    def __call__(self, prompt: Union[str, List[Message]]) -> AgentResult:
        """Process a prompt and return result"""
        # Convert string prompt to message format
        if isinstance(prompt, str):
            messages = [Message(role="user", content=prompt)]
        else:
            messages = prompt
        
        # Simple approach: always create new event loop for sync call
        # This matches how strands-agents likely works
        result = asyncio.run(self._process_messages(messages))
        return result
    
    async def process_async(self, prompt: Union[str, List[Message]]) -> AgentResult:
        """Process a prompt asynchronously - for use in async contexts"""
        # Convert string prompt to message format
        if isinstance(prompt, str):
            messages = [Message(role="user", content=prompt)]
        else:
            messages = prompt
        
        # Directly await the async method
        return await self._process_messages(messages)
    
    async def _process_messages(self, messages: List[Message]) -> AgentResult:
        """Process messages and handle tool calls"""
        if not self.model:
            return AgentResult(
                message={'role': 'assistant', 'content': [{'text': 'No model configured'}]},
                stop_reason='error'
            )
        
        # Convert tools to ToolSpec format for model
        tool_specs = []
        for tool in self.tools:
            if isinstance(tool, ToolSpec):
                tool_specs.append(tool)
            else:
                # Extract name from function (handle decorated functions)
                if hasattr(tool, '__wrapped__') and hasattr(tool.__wrapped__, '__name__'):
                    name = tool.__wrapped__.__name__
                    doc = tool.__wrapped__.__doc__ or ''
                elif hasattr(tool, '__name__'):
                    name = tool.__name__
                    doc = tool.__doc__ or ''
                else:
                    continue
                
                # Basic tool spec from function
                tool_specs.append(ToolSpec(
                    name=name,
                    description=doc,
                    parameters={}
                ))
        
        # Stream response from model
        content_blocks = []
        stop_reason = "end_turn"
        current_block = None
        
        try:
            async for event in self.model.stream(messages, tool_specs=tool_specs):
                logger.debug(f"Agent received event: {type(event).__name__}")
                
                if isinstance(event, MessageStartEvent):
                    # Start of message
                    pass
                
                elif isinstance(event, ContentBlockStart):
                    # Start new content block
                    if event.toolUse:
                        current_block = {
                            'toolUse': {
                                'toolUseId': event.toolUse.toolUseId,
                                'name': event.toolUse.name,
                                'input': {}
                            }
                        }
                    else:
                        current_block = {'text': ''}
                
                elif isinstance(event, ContentBlockDelta):
                    # Add to current block
                    if current_block:
                        if event.text and 'text' in current_block:
                            current_block['text'] += event.text
                        elif event.toolUse and 'toolUse' in current_block:
                            # Parse tool input
                            try:
                                input_data = json.loads(event.toolUse.input)
                                current_block['toolUse']['input'] = input_data
                            except:
                                current_block['toolUse']['input'] = event.toolUse.input
                
                elif isinstance(event, ContentBlockStopEvent):
                    # Finish current block
                    if current_block:
                        content_blocks.append(current_block)
                        
                        # Execute tool if it's a tool use block
                        if 'toolUse' in current_block:
                            tool_result = await self._execute_tool(current_block['toolUse'])
                            if tool_result:
                                # Add tool result as new message and continue
                                messages.append(Message(
                                    role="assistant",
                                    content=[current_block]
                                ))
                                messages.append(Message(
                                    role="user", 
                                    content=[{
                                        'toolResult': {
                                            'toolUseId': tool_result.toolUseId,
                                            'output': tool_result.output,
                                            'isError': tool_result.isError
                                        }
                                    }]
                                ))
                                # Get model's response to tool result
                                return await self._process_messages(messages)
                        
                        current_block = None
                
                elif isinstance(event, MessageStopEvent):
                    # End of message
                    stop_reason = event.stopReason
                    break
        
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
            content_blocks = [{'text': f'Error: {str(e)}'}]
            stop_reason = 'error'
        
        return AgentResult(
            message={
                'role': 'assistant',
                'content': content_blocks
            },
            stop_reason=stop_reason
        )
    
    async def _execute_tool(self, tool_use: Dict[str, Any]) -> Optional[ToolResult]:
        """Execute a tool and return result"""
        tool_name = tool_use.get('name')
        tool_input = tool_use.get('input', {})
        tool_use_id = tool_use.get('toolUseId')
        
        if tool_name not in self._tool_map:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(
                toolUseId=tool_use_id,
                output=f"Tool '{tool_name}' not found",
                isError=True
            )
        
        try:
            tool_func = self._tool_map[tool_name]
            logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
            
            # Execute tool
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_input)
            else:
                result = tool_func(**tool_input)
            
            # Ensure result is JSON serializable
            if isinstance(result, dict) or isinstance(result, list):
                # Convert to JSON and back to ensure all types are serializable
                try:
                    result_json = json.dumps(result, cls=NumpyEncoder)
                    result = json.loads(result_json)
                except Exception as e:
                    logger.warning(f"Failed to serialize result: {e}")
            
            return ToolResult(
                toolUseId=tool_use_id,
                output=result,
                isError=False
            )
        
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult(
                toolUseId=tool_use_id,
                output=str(e),
                isError=True
            )