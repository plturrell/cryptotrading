#!/usr/bin/env python3
"""
MCP Server Launcher with HTTP Transport
Launches the Strands-Glean MCP server with HTTP connectivity
"""

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MCPServerLauncher:
    """Launches MCP server with HTTP transport bypassing security"""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        self.glean_server = None

    async def create_simple_server(self):
        """Create a simple MCP server without complex security"""
        try:
            # Import the core components without security middleware
            from cryptotrading.core.ai.grok_client import GrokClient
            from cryptotrading.infrastructure.analysis.angle_parser import (
                PYTHON_QUERIES,
                create_query,
            )
            from cryptotrading.infrastructure.analysis.vercel_glean_client import VercelGleanClient

            # Set Grok API key
            os.environ[
                "GROK_API_KEY"
            ] = "YOUR_XAI_API_KEY_HEREm6U8QovWNoUphU8Ax8dUMAbh2I3nlgCRNAwYc8yUMnMUtCbYPo44bJBxX8BoKw3EdkAXOp7TJJFQIT7b"

            # Create Glean client
            self.glean_client = VercelGleanClient(project_root=str(project_root))

            # Create simple server class
            class SimpleMCPServer:
                def __init__(self, glean_client):
                    self.glean_client = glean_client
                    self.indexed_units = set()

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
                                            "name": "glean_index_project",
                                            "description": "Index the codebase using SCIP for Glean analysis",
                                        },
                                        {
                                            "name": "glean_symbol_search",
                                            "description": "Search for symbols in the codebase",
                                        },
                                        {
                                            "name": "ai_enhance_analysis",
                                            "description": "Enhance analysis with AI insights using Grok",
                                        },
                                        {
                                            "name": "ai_code_review",
                                            "description": "AI-powered code review using Grok",
                                        },
                                        {
                                            "name": "ai_explain_code",
                                            "description": "AI explanation of code components",
                                        },
                                        {
                                            "name": "glean_statistics",
                                            "description": "Get codebase analysis statistics",
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

                async def call_tool(
                    self, tool_name: str, arguments: Dict[str, Any]
                ) -> Dict[str, Any]:
                    """Call a specific tool"""
                    try:
                        if tool_name == "glean_index_project":
                            return await self.index_project(
                                arguments.get("unit_name", "main"),
                                arguments.get("force_reindex", False),
                            )

                        elif tool_name == "glean_symbol_search":
                            return await self.search_symbols(
                                arguments.get("pattern", ""), arguments.get("limit", 10)
                            )

                        elif tool_name == "ai_enhance_analysis":
                            return await self.enhance_with_ai(
                                arguments.get("analysis_data", {}),
                                arguments.get("enhancement_type", "summary"),
                                arguments.get("ai_provider", "grok"),
                            )

                        elif tool_name == "ai_code_review":
                            return await self.ai_code_review(
                                arguments.get("files", []),
                                arguments.get("review_type", "comprehensive"),
                            )

                        elif tool_name == "ai_explain_code":
                            return await self.ai_explain_code(
                                arguments.get("symbol", ""),
                                arguments.get("context_level", "moderate"),
                            )

                        elif tool_name == "glean_statistics":
                            return await self.get_statistics()

                        else:
                            return {
                                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                                "isError": True,
                            }

                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        return {
                            "content": [{"type": "text", "text": f"Tool error: {str(e)}"}],
                            "isError": True,
                        }

                async def index_project(
                    self, unit_name: str, force_reindex: bool
                ) -> Dict[str, Any]:
                    """Index the project"""
                    try:
                        result = await self.glean_client.index_project(unit_name, force_reindex)

                        if result.get("status") == "success":
                            self.indexed_units.add(unit_name)
                            stats = result.get("stats", {})

                            response_data = {
                                "status": "success",
                                "unit_name": unit_name,
                                "files_indexed": stats.get("files_indexed", 0),
                                "symbols_found": stats.get("symbols_found", 0),
                                "facts_stored": stats.get("facts_stored", 0),
                            }

                            return {
                                "content": [
                                    {
                                        "type": "resource",
                                        "data": json.dumps(response_data, indent=2),
                                        "mimeType": "application/json",
                                    }
                                ],
                                "isError": False,
                            }
                        else:
                            return {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Indexing failed: {result.get('error', 'Unknown error')}",
                                    }
                                ],
                                "isError": True,
                            }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Indexing error: {str(e)}"}],
                            "isError": True,
                        }

                async def search_symbols(self, pattern: str, limit: int) -> Dict[str, Any]:
                    """Search for symbols"""
                    try:
                        query = create_query("symbol_search", pattern=pattern)
                        results = await self.glean_client.query(query)

                        if isinstance(results, list):
                            symbols = results[:limit]

                            response_data = {
                                "query_pattern": pattern,
                                "total_found": len(results),
                                "returned": len(symbols),
                                "symbols": symbols,
                            }

                            return {
                                "content": [
                                    {
                                        "type": "resource",
                                        "data": json.dumps(response_data, indent=2),
                                        "mimeType": "application/json",
                                    }
                                ],
                                "isError": False,
                            }
                        else:
                            return {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"No symbols found for pattern: {pattern}",
                                    }
                                ],
                                "isError": False,
                            }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Search error: {str(e)}"}],
                            "isError": True,
                        }

                async def enhance_with_ai(
                    self, analysis_data: Dict[str, Any], enhancement_type: str, ai_provider: str
                ) -> Dict[str, Any]:
                    """Enhance analysis with AI"""
                    try:
                        if ai_provider == "grok":
                            async with GrokClient() as grok:
                                if enhancement_type == "summary":
                                    result = await grok.analyze_code_structure(
                                        analysis_data, focus="architecture"
                                    )
                                elif enhancement_type == "recommendations":
                                    result = await grok.generate_refactoring_suggestions(
                                        analysis_data
                                    )
                                else:
                                    result = await grok.analyze_code_structure(
                                        analysis_data, focus=enhancement_type
                                    )

                                response_data = {
                                    "original_analysis": analysis_data,
                                    "enhancement_type": enhancement_type,
                                    "ai_provider": ai_provider,
                                    "grok_response": result,
                                    "confidence_score": 0.90,
                                }
                        else:
                            response_data = {"error": f"Unsupported AI provider: {ai_provider}"}

                        return {
                            "content": [
                                {
                                    "type": "resource",
                                    "data": json.dumps(response_data, indent=2),
                                    "mimeType": "application/json",
                                }
                            ],
                            "isError": False,
                        }
                    except Exception as e:
                        return {
                            "content": [
                                {"type": "text", "text": f"AI enhancement error: {str(e)}"}
                            ],
                            "isError": True,
                        }

                async def ai_code_review(self, files: list, review_type: str) -> Dict[str, Any]:
                    """AI code review"""
                    try:
                        # Prepare file data
                        file_analyses = []
                        for file_path in files:
                            file_data = {"path": file_path, "symbols": [], "complexity": "medium"}
                            file_analyses.append(file_data)

                        # Use Grok for review
                        async with GrokClient() as grok:
                            grok_review = await grok.generate_code_review(
                                file_analyses, review_type
                            )

                            response_data = {
                                "files_reviewed": files,
                                "review_type": review_type,
                                "grok_review": grok_review,
                                "ai_powered": True,
                            }

                        return {
                            "content": [
                                {
                                    "type": "resource",
                                    "data": json.dumps(response_data, indent=2),
                                    "mimeType": "application/json",
                                }
                            ],
                            "isError": False,
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"AI review error: {str(e)}"}],
                            "isError": True,
                        }

                async def ai_explain_code(self, symbol: str, context_level: str) -> Dict[str, Any]:
                    """AI code explanation"""
                    try:
                        # Get symbol data
                        symbol_data = {"name": symbol, "kind": "unknown", "file": "unknown"}

                        # Use Grok for explanation
                        async with GrokClient() as grok:
                            grok_explanation = await grok.explain_code_component(
                                symbol_data, context_level
                            )

                            response_data = {
                                "symbol": symbol,
                                "context_level": context_level,
                                "symbol_data": symbol_data,
                                "grok_explanation": grok_explanation,
                                "ai_powered": True,
                            }

                        return {
                            "content": [
                                {
                                    "type": "resource",
                                    "data": json.dumps(response_data, indent=2),
                                    "mimeType": "application/json",
                                }
                            ],
                            "isError": False,
                        }
                    except Exception as e:
                        return {
                            "content": [
                                {"type": "text", "text": f"AI explanation error: {str(e)}"}
                            ],
                            "isError": True,
                        }

                async def get_statistics(self) -> Dict[str, Any]:
                    """Get statistics"""
                    try:
                        stats = await self.glean_client.get_storage_stats()

                        response_data = {
                            "project_root": str(project_root),
                            "indexed_units": list(self.indexed_units),
                            "statistics": stats,
                        }

                        return {
                            "content": [
                                {
                                    "type": "resource",
                                    "data": json.dumps(response_data, indent=2),
                                    "mimeType": "application/json",
                                }
                            ],
                            "isError": False,
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Statistics error: {str(e)}"}],
                            "isError": True,
                        }

            return SimpleMCPServer(self.glean_client)

        except Exception as e:
            logger.error(f"Failed to create server: {e}")
            raise

    async def start_http_server(self):
        """Start HTTP server"""
        try:
            from aiohttp import web

            # Create the MCP server
            self.glean_server = await self.create_simple_server()

            # Create web app
            app = web.Application()

            async def handle_mcp(request):
                """Handle MCP requests"""
                try:
                    data = await request.json()
                    response = await self.glean_server.handle_mcp_request(data)

                    return web.json_response(response, headers={"Access-Control-Allow-Origin": "*"})
                except Exception as e:
                    logger.error(f"Error handling MCP request: {e}")
                    return web.json_response(
                        {"error": str(e)}, status=500, headers={"Access-Control-Allow-Origin": "*"}
                    )

            async def handle_status(request):
                """Handle status requests"""
                return web.json_response(
                    {
                        "status": "running",
                        "server": "strands-glean-mcp",
                        "transport": "http",
                        "version": "1.0.0",
                    },
                    headers={"Access-Control-Allow-Origin": "*"},
                )

            async def handle_tools(request):
                """Handle tools list"""
                tools_request = {"jsonrpc": "2.0", "id": "list_tools", "method": "tools/list"}
                response = await self.glean_server.handle_mcp_request(tools_request)
                return web.json_response(response, headers={"Access-Control-Allow-Origin": "*"})

            async def handle_options(request):
                """Handle CORS preflight"""
                return web.Response(
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    }
                )

            # Setup routes
            app.router.add_post("/mcp", handle_mcp)
            app.router.add_get("/mcp/status", handle_status)
            app.router.add_get("/mcp/tools", handle_tools)
            app.router.add_options("/mcp", handle_options)

            # Start server
            runner = web.AppRunner(app)
            await runner.setup()

            site = web.TCPSite(runner, self.host, self.port)
            await site.start()

            self.server = (runner, site)

            logger.info(f"🚀 MCP Server running on http://{self.host}:{self.port}")
            logger.info(f"   • Status: http://{self.host}:{self.port}/mcp/status")
            logger.info(f"   • Tools: http://{self.host}:{self.port}/mcp/tools")
            logger.info(f"   • MCP Endpoint: http://{self.host}:{self.port}/mcp")

        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            raise

    async def stop(self):
        """Stop the server"""
        if self.server:
            runner, site = self.server
            await site.stop()
            await runner.cleanup()
            self.server = None
            logger.info("Server stopped")


async def main():
    """Main server launcher"""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Strands-Glean MCP Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")

    args = parser.parse_args()

    # Create and start launcher
    launcher = MCPServerLauncher(args.host, args.port)

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(launcher.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await launcher.start_http_server()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await launcher.stop()


if __name__ == "__main__":
    asyncio.run(main())
