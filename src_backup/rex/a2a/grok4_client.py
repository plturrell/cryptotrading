"""
Grok-4 Client for XAI API Integration
Provides Grok-4 model access for A2A agents
"""

import aiohttp
import json
import logging
import ssl
import os
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime

logger = logging.getLogger(__name__)

class Grok4Client:
    """Client for Grok-4 via XAI API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        if not self.api_key:
            logger.warning("XAI_API_KEY not provided - API calls will fail until key is set")
            self.api_key = None
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.model_name = "grok-4-latest"
        
    async def complete(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete a chat conversation using Grok-4"""
        
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable is required for Grok-4 operations")
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if tools:
            payload["tools"] = tools
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Create secure SSL context - always verify certificates in production
            ssl_context = ssl.create_default_context()
            # Only disable SSL verification if explicitly requested for development
            if os.getenv('DISABLE_SSL_VERIFY', '').lower() == 'true':
                logger.warning("SSL verification disabled - development mode only")
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "content": result["choices"][0]["message"]["content"],
                            "finish_reason": result["choices"][0]["finish_reason"],
                            "usage": result.get("usage", {}),
                            "model": result["model"]
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Grok-4 API error {response.status}: {error_text}")
                        return {
                            "success": False,
                            "error": f"API error {response.status}: {error_text}"
                        }
        except Exception as e:
            logger.error(f"Grok-4 client error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a chat conversation using Grok-4"""
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if tools:
            payload["tools"] = tools
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Create secure SSL context - always verify certificates in production
            ssl_context = ssl.create_default_context()
            # Only disable SSL verification if explicitly requested for development
            if os.getenv('DISABLE_SSL_VERIFY', '').lower() == 'true':
                logger.warning("SSL verification disabled - development mode only")
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data = line[6:]  # Remove 'data: ' prefix
                                if data == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if chunk["choices"]:
                                        delta = chunk["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            yield {
                                                "success": True,
                                                "content": delta["content"],
                                                "finish_reason": chunk["choices"][0].get("finish_reason"),
                                                "model": chunk["model"]
                                            }
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        yield {
                            "success": False,
                            "error": f"Stream error {response.status}: {error_text}"
                        }
        except Exception as e:
            yield {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_crypto_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cryptocurrency data using Grok-4"""
        
        system_prompt = """You are an expert cryptocurrency analyst. Analyze the provided data and give actionable insights. 
        Focus on technical indicators, market trends, and trading recommendations.
        Be concise and specific in your analysis."""
        
        user_prompt = f"""Analyze this cryptocurrency data:
        Symbol: {data.get('symbol', 'Unknown')}
        Price: ${data.get('price', 'N/A')}
        Volume: {data.get('volume', 'N/A')}
        RSI: {data.get('rsi', 'N/A')}
        Date Range: {data.get('date_range', 'N/A')}
        
        Additional data: {json.dumps(data, indent=2)}
        
        Provide analysis including:
        1. Market sentiment
        2. Technical analysis
        3. Trading recommendation
        4. Risk assessment"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return await self.complete(messages, temperature=0.3)
    
    async def process_agent_request(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a general agent request using Grok-4"""
        
        system_prompt = """You are an AI agent in a cryptocurrency trading system. You help with data analysis, 
        decision making, and providing insights. Be precise, actionable, and focus on the specific request."""
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{request}{context_str}"}
        ]
        
        return await self.complete(messages, temperature=0.5)
    
    def test_connection(self) -> bool:
        """Test if Grok-4 API is accessible"""
        import asyncio
        
        async def _test():
            messages = [
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Testing. Just say hi and hello world and nothing else."}
            ]
            result = await self.complete(messages, temperature=0)
            return result.get("success", False)
        
        try:
            return asyncio.run(_test())
        except Exception as e:
            logger.error(f"Grok-4 connection test failed: {e}")
            return False

# Global client instance
grok4_client = Grok4Client()

def get_grok4_client() -> Grok4Client:
    """Get the global Grok-4 client instance"""
    return grok4_client

# Test function moved to tests/integration/test_grok4.py