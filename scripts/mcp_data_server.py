#!/usr/bin/env python3
"""
Simple MCP Data Analysis Server
Provides market data analysis tools via MCP protocol
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import aiohttp_cors
from aiohttp import web

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MCPDataServer:
    """Simple MCP server for data analysis"""

    def __init__(self, host: str = "localhost", port: int = 3002):
        self.host = host
        self.port = port
        self.market_data_cache = []
        self.analysis_results = []

    async def handle_mcp_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP JSON-RPC requests"""
        method = request_data.get("method", "")
        params = request_data.get("params", {})
        request_id = request_data.get("id", "unknown")

        try:
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "store_market_data",
                                "description": "Store market data for analysis",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"market_data": {"type": "object"}},
                                },
                            },
                            {
                                "name": "analyze_market_trends",
                                "description": "Analyze market trends from stored data",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "timeframe": {"type": "string", "default": "1h"}
                                    },
                                },
                            },
                            {
                                "name": "get_market_summary",
                                "description": "Get market data summary",
                                "inputSchema": {"type": "object", "properties": {}},
                            },
                        ]
                    },
                }

            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})

                result = await self.call_tool(tool_name, arguments)

                return {"jsonrpc": "2.0", "id": request_id, "result": result}

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool"""
        try:
            if tool_name == "store_market_data":
                return await self.store_market_data(arguments.get("market_data", {}))

            elif tool_name == "analyze_market_trends":
                return await self.analyze_market_trends(arguments.get("timeframe", "1h"))

            elif tool_name == "get_market_summary":
                return await self.get_market_summary()

            else:
                return {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"content": [{"type": "text", "text": f"Tool error: {str(e)}"}], "isError": True}

    async def store_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store market data for analysis"""
        try:
            market_data["stored_at"] = datetime.now().isoformat()
            self.market_data_cache.append(market_data)

            # Keep only last 100 entries
            if len(self.market_data_cache) > 100:
                self.market_data_cache = self.market_data_cache[-100:]

            response_data = {
                "status": "success",
                "stored_count": len(self.market_data_cache),
                "latest_data": market_data,
            }

            return {
                "content": [{"type": "text", "text": json.dumps(response_data, indent=2)}],
                "isError": False,
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Storage error: {str(e)}"}],
                "isError": True,
            }

    async def analyze_market_trends(self, timeframe: str) -> Dict[str, Any]:
        """Analyze market trends"""
        try:
            if not self.market_data_cache:
                return {
                    "content": [{"type": "text", "text": "No market data available for analysis"}],
                    "isError": False,
                }

            # Simple trend analysis
            recent_data = self.market_data_cache[-10:]  # Last 10 entries

            btc_prices = [d.get("btc_price", 0) for d in recent_data if "btc_price" in d]
            eth_prices = [d.get("eth_price", 0) for d in recent_data if "eth_price" in d]

            analysis = {
                "timeframe": timeframe,
                "data_points": len(recent_data),
                "btc_analysis": self._analyze_price_trend(btc_prices, "BTC"),
                "eth_analysis": self._analyze_price_trend(eth_prices, "ETH"),
                "timestamp": datetime.now().isoformat(),
            }

            self.analysis_results.append(analysis)

            return {
                "content": [{"type": "text", "text": json.dumps(analysis, indent=2)}],
                "isError": False,
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Analysis error: {str(e)}"}],
                "isError": True,
            }

    def _analyze_price_trend(self, prices: List[float], symbol: str) -> Dict[str, Any]:
        """Analyze price trend for a symbol"""
        if len(prices) < 2:
            return {"symbol": symbol, "trend": "insufficient_data"}

        first_price = prices[0]
        last_price = prices[-1]
        change_percent = ((last_price - first_price) / first_price) * 100

        trend = (
            "bullish" if change_percent > 1 else "bearish" if change_percent < -1 else "sideways"
        )

        return {
            "symbol": symbol,
            "first_price": first_price,
            "last_price": last_price,
            "change_percent": round(change_percent, 2),
            "trend": trend,
            "volatility": round(max(prices) - min(prices), 2) if prices else 0,
        }

    async def get_market_summary(self) -> Dict[str, Any]:
        """Get market data summary"""
        try:
            summary = {
                "total_data_points": len(self.market_data_cache),
                "total_analyses": len(self.analysis_results),
                "latest_data": self.market_data_cache[-1] if self.market_data_cache else None,
                "latest_analysis": self.analysis_results[-1] if self.analysis_results else None,
                "server_uptime": datetime.now().isoformat(),
            }

            return {
                "content": [{"type": "text", "text": json.dumps(summary, indent=2)}],
                "isError": False,
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Summary error: {str(e)}"}],
                "isError": True,
            }

    async def start_server(self):
        """Start the HTTP server"""
        try:
            app = web.Application()

            # Setup CORS
            cors = aiohttp_cors.setup(
                app,
                defaults={
                    "*": aiohttp_cors.ResourceOptions(
                        allow_credentials=True,
                        expose_headers="*",
                        allow_headers="*",
                        allow_methods="*",
                    )
                },
            )

            async def handle_mcp(request):
                """Handle MCP requests"""
                try:
                    data = await request.json()
                    response = await self.handle_mcp_request(data)
                    return web.json_response(response)
                except Exception as e:
                    logger.error(f"Error handling MCP request: {e}")
                    return web.json_response({"error": str(e)}, status=500)

            async def handle_status(request):
                """Handle status requests"""
                return web.json_response(
                    {
                        "status": "running",
                        "server": "mcp-data-analysis",
                        "transport": "http",
                        "version": "1.0.0",
                        "data_points": len(self.market_data_cache),
                        "analyses": len(self.analysis_results),
                    }
                )

            async def handle_tools(request):
                """Handle tools list"""
                tools_request = {"jsonrpc": "2.0", "id": "list_tools", "method": "tools/list"}
                response = await self.handle_mcp_request(tools_request)
                return web.json_response(response)

            # Setup routes
            app.router.add_post("/mcp", handle_mcp)
            app.router.add_get("/mcp/status", handle_status)
            app.router.add_get("/mcp/tools", handle_tools)

            # Add CORS to all routes
            for route in list(app.router.routes()):
                cors.add(route)

            # Start server
            runner = web.AppRunner(app)
            await runner.setup()

            site = web.TCPSite(runner, self.host, self.port)
            await site.start()

            logger.info(f"🚀 MCP Data Server running on http://{self.host}:{self.port}")
            logger.info(f"   • Status: http://{self.host}:{self.port}/mcp/status")
            logger.info(f"   • Tools: http://{self.host}:{self.port}/mcp/tools")
            logger.info(f"   • MCP Endpoint: http://{self.host}:{self.port}/mcp")

            return runner, site

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise


async def main():
    """Main server function"""
    import argparse

    parser = argparse.ArgumentParser(description="Launch MCP Data Analysis Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=3002, help="Server port")

    args = parser.parse_args()

    server = MCPDataServer(args.host, args.port)

    try:
        runner, site = await server.start_server()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if "runner" in locals():
            await site.stop()
            await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
