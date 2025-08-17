"""
Vercel Serverless Function for MCP Server
Handles MCP requests in Vercel's serverless environment
"""

import json
import os
import asyncio
from typing import Dict, Any
from datetime import datetime

# Import MCP components
from cryptotrading.core.protocols.mcp.quick_start import MCPQuickStart
from cryptotrading.core.protocols.mcp.security.middleware import VercelSecurityMiddleware


class VercelMCPServer:
    """MCP Server optimized for Vercel serverless functions"""
    
    def __init__(self):
        self.server = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize server (cached between requests)"""
        if self._initialized and self.server:
            return self.server
        
        # Create MCP server with Vercel-optimized config
        self.server = (MCPQuickStart("vercel-mcp-server")
                      .with_transport("http")
                      .with_security(
                          enabled=os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true",
                          require_auth=os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true"
                      )
                      .with_features(
                          tools=True,
                          resources=True,
                          prompts=True,
                          sampling=True,
                          subscriptions=False,  # Disabled for serverless
                          progress=False        # Disabled for serverless
                      ))
        
        # Add built-in tools
        await self._add_builtin_tools()
        
        # Add custom tools from environment
        await self._add_custom_tools()
        
        # Initialize server
        await self.server.start()
        
        self._initialized = True
        return self.server
    
    async def _add_builtin_tools(self):
        """Add built-in utility tools"""
        
        # Echo tool
        def echo(message: str) -> str:
            return f"Echo: {message}"
        
        # Current time tool
        def current_time() -> str:
            return datetime.utcnow().isoformat() + "Z"
        
        # Environment info tool
        def env_info() -> dict:
            return {
                "platform": "vercel",
                "python_version": os.sys.version,
                "environment": os.getenv("VERCEL_ENV", "unknown"),
                "region": os.getenv("VERCEL_REGION", "unknown"),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Calculator tool (safe eval)
        def calculator(expression: str) -> str:
            try:
                # Safe mathematical evaluation
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Only basic mathematical operations allowed"
                
                # Simple eval with limited scope
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Add tools to server
        self.server.add_simple_tool("echo", "Echo a message", echo, 
                                   {"message": {"type": "string"}})
        self.server.add_simple_tool("current_time", "Get current UTC time", current_time)
        self.server.add_simple_tool("env_info", "Get environment information", env_info)
        self.server.add_simple_tool("calculator", "Calculate mathematical expressions", calculator,
                                   {"expression": {"type": "string"}})
    
    async def _add_custom_tools(self):
        """Add custom tools from environment configuration"""
        # This could load tools from environment variables or database
        # For now, we'll add some crypto-trading specific tools
        
        def get_portfolio_summary() -> dict:
            """Mock portfolio summary"""
            return {
                "total_value": 10000.0,
                "positions": [
                    {"symbol": "BTC", "amount": 0.5, "value": 5000.0},
                    {"symbol": "ETH", "amount": 3.0, "value": 5000.0}
                ],
                "last_updated": datetime.utcnow().isoformat()
            }
        
        def get_market_data(symbol: str) -> dict:
            """Mock market data"""
            # In real implementation, this would fetch from an API
            mock_prices = {
                "BTC": 50000.0,
                "ETH": 3000.0,
                "ADA": 1.0,
                "SOL": 100.0
            }
            
            price = mock_prices.get(symbol.upper(), 0.0)
            return {
                "symbol": symbol.upper(),
                "price": price,
                "change_24h": 0.05,  # 5% mock change
                "volume": 1000000.0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Add crypto tools
        self.server.add_simple_tool("get_portfolio", "Get portfolio summary", get_portfolio_summary)
        self.server.add_simple_tool("get_market_data", "Get market data for symbol", get_market_data,
                                   {"symbol": {"type": "string", "description": "Trading symbol (BTC, ETH, etc.)"}})


# Global server instance (reused across requests)
vercel_server = VercelMCPServer()


async def handle_mcp_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP JSON-RPC request"""
    try:
        # Initialize server if needed
        server = await vercel_server.initialize()
        
        # Handle the request
        response = await server.server.handle_request(request_data)
        return response
        
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_data.get("id"),
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }


def handler(request):
    """Vercel serverless function handler"""
    from urllib.parse import parse_qs
    
    # Handle different HTTP methods
    method = request.method
    
    if method == "OPTIONS":
        # CORS preflight
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key"
            },
            "body": ""
        }
    
    elif method == "GET":
        # Health check / status endpoint
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "status": "healthy",
                "service": "MCP Server",
                "version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "endpoints": {
                    "mcp": "/api/mcp",
                    "tools": "/api/mcp-tools"
                }
            })
        }
    
    elif method == "POST":
        try:
            # Parse request body
            if hasattr(request, 'body'):
                body = request.body
            else:
                body = request.get('body', '{}')
            
            if isinstance(body, bytes):
                body = body.decode('utf-8')
            
            request_data = json.loads(body)
            
            # Handle MCP request
            response = asyncio.run(handle_mcp_request(request_data))
            
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps(response)
            }
            
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                })
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    }
                })
            }
    
    else:
        return {
            "statusCode": 405,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "Method not allowed"})
        }


# Export for Vercel
def app(request):
    """Main Vercel handler"""
    return handler(request)