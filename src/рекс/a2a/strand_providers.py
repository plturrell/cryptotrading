"""
Custom Strand Agent Model Providers for рекс Trading Platform
Integrates our DeepSeek R1 and Anthropic (via Vercel) with Strand Agents
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from strands.agent.models.base import ModelProvider, ModelResponse
import asyncio
import json

from ..ml.deepseek import DeepSeekR1
from ..ml.perplexity import PerplexityClient

class DeepSeekModelProvider(ModelProvider):
    """DeepSeek R1 model provider for Strand Agents"""
    
    def __init__(self):
        self.deepseek = DeepSeekR1()
        self.model_name = "deepseek-r1"
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ModelResponse:
        """Complete a conversation using DeepSeek R1"""
        try:
            # Convert messages to DeepSeek format
            # Get the last user message
            last_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_message = msg.get("content", "")
                    break
            
            # If there are tools, we need to analyze which tool to use
            if tools and last_message:
                # Simple tool selection based on keywords
                tool_to_use = self._select_tool(last_message, tools)
                
                if tool_to_use:
                    # Return a tool use response
                    return ModelResponse(
                        content="",
                        tool_calls=[{
                            "id": f"call_{hash(last_message)}",
                            "function": {
                                "name": tool_to_use["name"],
                                "arguments": self._extract_arguments(last_message, tool_to_use)
                            }
                        }],
                        finish_reason="tool_calls"
                    )
            
            # Otherwise, use DeepSeek for analysis
            market_data = self._extract_market_data(last_message)
            response = self.deepseek.analyze_market(market_data)
            
            return ModelResponse(
                content=response,
                finish_reason="stop"
            )
            
        except Exception as e:
            return ModelResponse(
                content=f"DeepSeek error: {str(e)}",
                finish_reason="error"
            )
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[ModelResponse]:
        """Stream responses (not implemented for DeepSeek)"""
        # DeepSeek doesn't support streaming, so return complete response
        response = await self.complete(messages, tools, **kwargs)
        yield response
    
    def _select_tool(self, message: str, tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Simple tool selection based on message content"""
        message_lower = message.lower()
        
        for tool in tools:
            tool_name = tool.get("name", "").lower()
            
            # Match tool based on keywords
            if "discover" in message_lower and "structure" in message_lower:
                if "discover_data_structure" in tool_name:
                    return tool
            elif "store" in message_lower and "schema" in message_lower:
                if "store_schema" in tool_name:
                    return tool
            elif "get" in message_lower and "schema" in message_lower:
                if "get_schema" in tool_name:
                    return tool
            elif "list" in message_lower and "schema" in message_lower:
                if "list_schemas" in tool_name:
                    return tool
        
        return None
    
    def _extract_arguments(self, message: str, tool: Dict[str, Any]) -> str:
        """Extract arguments for tool call from message"""
        # This is a simplified extraction - in production, use better parsing
        args = {}
        
        # Extract common patterns
        if "source:" in message:
            parts = message.split("source:")
            if len(parts) > 1:
                source = parts[1].split()[0].strip()
                args["source_name"] = source
        
        if "config:" in message:
            # Try to extract JSON config
            try:
                start = message.find("{")
                end = message.rfind("}") + 1
                if start >= 0 and end > start:
                    config = json.loads(message[start:end])
                    args["source_config"] = config
            except:
                pass
        
        return json.dumps(args)
    
    def _extract_market_data(self, message: str) -> Dict[str, Any]:
        """Extract market data from message for DeepSeek analysis"""
        # Simple extraction logic
        data = {
            "symbol": "BTC",
            "price": 50000,
            "volume": 1000000,
            "rsi": 50
        }
        
        # Extract symbol if mentioned
        for symbol in ["BTC", "ETH", "BTCUSDT", "ETHUSDT"]:
            if symbol in message.upper():
                data["symbol"] = symbol
                break
        
        return data


class AnthropicVercelModelProvider(ModelProvider):
    """Anthropic model provider via Vercel AI SDK"""
    
    def __init__(self):
        self.model_name = "claude-3-sonnet"
        # This would integrate with Vercel's AI SDK
        # For now, it's a placeholder
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ModelResponse:
        """Complete using Anthropic via Vercel"""
        # This would call Vercel's AI SDK
        # For now, return a placeholder
        return ModelResponse(
            content="Anthropic via Vercel not yet implemented",
            finish_reason="stop"
        )
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[ModelResponse]:
        """Stream responses from Anthropic"""
        response = await self.complete(messages, tools, **kwargs)
        yield response


def get_model_provider(provider_name: str = "deepseek") -> ModelProvider:
    """Get the appropriate model provider"""
    if provider_name == "deepseek":
        return DeepSeekModelProvider()
    elif provider_name == "anthropic":
        return AnthropicVercelModelProvider()
    else:
        # Default to DeepSeek
        return DeepSeekModelProvider()


# Create singleton instances
deepseek_provider = DeepSeekModelProvider()
anthropic_provider = AnthropicVercelModelProvider()