#!/usr/bin/env python3
"""
Real-time MCP Server with File Watching
Combines MCP server with real-time file watching and AI analysis
"""

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RealtimeMCPServer:
    """MCP Server with integrated real-time file watching"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        self.glean_server = None
        self.file_watcher = None
        
        # Real-time analysis data
        self.recent_changes: List[Dict[str, Any]] = []
        self.recent_analyses: List[Dict[str, Any]] = []
        self.max_recent_items = 100
        
        # Set Grok API key
        os.environ['GROK_API_KEY'] = 'YOUR_XAI_API_KEY_HEREm6U8QovWNoUphU8Ax8dUMAbh2I3nlgCRNAwYc8yUMnMUtCbYPo44bJBxX8BoKw3EdkAXOp7TJJFQIT7b'
    
    async def create_enhanced_server(self):
        """Create enhanced MCP server with real-time capabilities"""
        try:
            # Import core components
            from cryptotrading.infrastructure.analysis.vercel_glean_client import VercelGleanClient
            from cryptotrading.infrastructure.analysis.angle_parser import create_query, PYTHON_QUERIES
            from cryptotrading.core.ai.grok_client import GrokClient
            from cryptotrading.core.monitoring.file_watcher import create_file_watcher
            
            # Create Glean client
            self.glean_client = VercelGleanClient(project_root=str(project_root))
            
            # Create file watcher
            watch_dirs = [str(project_root / "src" / "cryptotrading")]
            self.file_watcher = await create_file_watcher(
                watch_directories=watch_dirs,
                project_root=str(project_root),
                enable_ai=True
            )
            
            # Add callbacks for real-time updates
            self.file_watcher.add_change_callback(self._on_file_change)
            self.file_watcher.add_analysis_callback(self._on_analysis_complete)
            
            # Enhanced MCP server class
            class EnhancedMCPServer:
                def __init__(self, glean_client, file_watcher, realtime_server):
                    self.glean_client = glean_client
                    self.file_watcher = file_watcher
                    self.realtime_server = realtime_server
                    self.indexed_units = set()
                
                async def handle_mcp_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
                    """Handle MCP JSON-RPC requests with real-time features"""
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
                                            "description": "Index the codebase using SCIP for Glean analysis"
                                        },
                                        {
                                            "name": "glean_symbol_search", 
                                            "description": "Search for symbols in the codebase"
                                        },
                                        {
                                            "name": "ai_enhance_analysis",
                                            "description": "Enhance analysis with AI insights using Grok"
                                        },
                                        {
                                            "name": "ai_code_review",
                                            "description": "AI-powered code review using Grok"
                                        },
                                        {
                                            "name": "ai_explain_code",
                                            "description": "AI explanation of code components"
                                        },
                                        {
                                            "name": "glean_statistics",
                                            "description": "Get codebase analysis statistics"
                                        },
                                        {
                                            "name": "realtime_start_watching",
                                            "description": "Start real-time file watching and analysis"
                                        },
                                        {
                                            "name": "realtime_stop_watching",
                                            "description": "Stop real-time file watching"
                                        },
                                        {
                                            "name": "realtime_get_changes",
                                            "description": "Get recent file changes"
                                        },
                                        {
                                            "name": "realtime_get_analyses",
                                            "description": "Get recent analysis results"
                                        },
                                        {
                                            "name": "realtime_get_status",
                                            "description": "Get real-time monitoring status"
                                        }
                                    ]
                                }
                            }
                        
                        elif method == "tools/call":
                            tool_name = params.get("name", "")
                            arguments = params.get("arguments", {})
                            
                            result = await self.call_tool(tool_name, arguments)
                            
                            return {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": result
                            }
                        
                        else:
                            return {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32601,
                                    "message": f"Method not found: {method}"
                                }
                            }
                    
                    except Exception as e:
                        logger.error(f"Error handling request: {e}")
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}"
                            }
                        }
                
                async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
                    """Call a specific tool including real-time features"""
                    try:
                        # Original tools
                        if tool_name == "glean_index_project":
                            return await self.index_project(
                                arguments.get("unit_name", "main"),
                                arguments.get("force_reindex", False)
                            )
                        
                        elif tool_name == "glean_symbol_search":
                            return await self.search_symbols(
                                arguments.get("pattern", ""),
                                arguments.get("limit", 10)
                            )
                        
                        elif tool_name == "ai_enhance_analysis":
                            return await self.enhance_with_ai(
                                arguments.get("analysis_data", {}),
                                arguments.get("enhancement_type", "summary"),
                                arguments.get("ai_provider", "grok")
                            )
                        
                        elif tool_name == "ai_code_review":
                            return await self.ai_code_review(
                                arguments.get("files", []),
                                arguments.get("review_type", "comprehensive")
                            )
                        
                        elif tool_name == "ai_explain_code":
                            return await self.ai_explain_code(
                                arguments.get("symbol", ""),
                                arguments.get("context_level", "moderate")
                            )
                        
                        elif tool_name == "glean_statistics":
                            return await self.get_statistics()
                        
                        # Real-time tools
                        elif tool_name == "realtime_start_watching":
                            return await self.start_watching()
                        
                        elif tool_name == "realtime_stop_watching":
                            return await self.stop_watching()
                        
                        elif tool_name == "realtime_get_changes":
                            return await self.get_recent_changes(
                                arguments.get("limit", 20)
                            )
                        
                        elif tool_name == "realtime_get_analyses":
                            return await self.get_recent_analyses(
                                arguments.get("limit", 20)
                            )
                        
                        elif tool_name == "realtime_get_status":
                            return await self.get_realtime_status()
                        
                        else:
                            return {
                                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                                "isError": True
                            }
                    
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        return {
                            "content": [{"type": "text", "text": f"Tool error: {str(e)}"}],
                            "isError": True
                        }
                
                # Original tools (same as before)
                async def index_project(self, unit_name: str, force_reindex: bool) -> Dict[str, Any]:
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
                                "facts_stored": stats.get("facts_stored", 0)
                            }
                            
                            return {
                                "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                                "isError": False
                            }
                        else:
                            return {
                                "content": [{"type": "text", "text": f"Indexing failed: {result.get('error', 'Unknown error')}"}],
                                "isError": True
                            }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Indexing error: {str(e)}"}],
                            "isError": True
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
                                "symbols": symbols
                            }
                            
                            return {
                                "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                                "isError": False
                            }
                        else:
                            return {
                                "content": [{"type": "text", "text": f"No symbols found for pattern: {pattern}"}],
                                "isError": False
                            }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Search error: {str(e)}"}],
                            "isError": True
                        }
                
                async def enhance_with_ai(self, analysis_data: Dict[str, Any], enhancement_type: str, ai_provider: str) -> Dict[str, Any]:
                    """Enhance analysis with AI"""
                    try:
                        if ai_provider == "grok":
                            async with GrokClient() as grok:
                                if enhancement_type == "summary":
                                    result = await grok.analyze_code_structure(analysis_data, focus="architecture")
                                elif enhancement_type == "recommendations":
                                    result = await grok.generate_refactoring_suggestions(analysis_data)
                                else:
                                    result = await grok.analyze_code_structure(analysis_data, focus=enhancement_type)
                                
                                response_data = {
                                    "original_analysis": analysis_data,
                                    "enhancement_type": enhancement_type,
                                    "ai_provider": ai_provider,
                                    "grok_response": result,
                                    "confidence_score": 0.90
                                }
                        else:
                            response_data = {
                                "error": f"Unsupported AI provider: {ai_provider}"
                            }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"AI enhancement error: {str(e)}"}],
                            "isError": True
                        }
                
                async def ai_code_review(self, files: list, review_type: str) -> Dict[str, Any]:
                    """AI code review"""
                    try:
                        file_analyses = []
                        for file_path in files:
                            file_data = {"path": file_path, "symbols": [], "complexity": "medium"}
                            file_analyses.append(file_data)
                        
                        async with GrokClient() as grok:
                            grok_review = await grok.generate_code_review(file_analyses, review_type)
                            
                            response_data = {
                                "files_reviewed": files,
                                "review_type": review_type,
                                "grok_review": grok_review,
                                "ai_powered": True
                            }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"AI review error: {str(e)}"}],
                            "isError": True
                        }
                
                async def ai_explain_code(self, symbol: str, context_level: str) -> Dict[str, Any]:
                    """AI code explanation"""
                    try:
                        symbol_data = {"name": symbol, "kind": "unknown", "file": "unknown"}
                        
                        async with GrokClient() as grok:
                            grok_explanation = await grok.explain_code_component(symbol_data, context_level)
                            
                            response_data = {
                                "symbol": symbol,
                                "context_level": context_level,
                                "symbol_data": symbol_data,
                                "grok_explanation": grok_explanation,
                                "ai_powered": True
                            }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"AI explanation error: {str(e)}"}],
                            "isError": True
                        }
                
                async def get_statistics(self) -> Dict[str, Any]:
                    """Get statistics including real-time data"""
                    try:
                        stats = await self.glean_client.get_storage_stats()
                        
                        # Add real-time statistics
                        realtime_stats = self.file_watcher.get_statistics() if self.file_watcher else {}
                        
                        response_data = {
                            "project_root": str(project_root),
                            "indexed_units": list(self.indexed_units),
                            "glean_statistics": stats,
                            "realtime_statistics": realtime_stats,
                            "recent_changes_count": len(self.realtime_server.recent_changes),
                            "recent_analyses_count": len(self.realtime_server.recent_analyses)
                        }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Statistics error: {str(e)}"}],
                            "isError": True
                        }
                
                # Real-time tools
                async def start_watching(self) -> Dict[str, Any]:
                    """Start real-time file watching"""
                    try:
                        if self.file_watcher:
                            success = await self.file_watcher.start_watching()
                            
                            response_data = {
                                "status": "watching_started" if success else "start_failed",
                                "directories_watched": len(self.file_watcher.watch_directories),
                                "files_being_watched": self.file_watcher.stats.get("files_watched", 0)
                            }
                        else:
                            response_data = {
                                "status": "file_watcher_unavailable"
                            }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Start watching error: {str(e)}"}],
                            "isError": True
                        }
                
                async def stop_watching(self) -> Dict[str, Any]:
                    """Stop real-time file watching"""
                    try:
                        if self.file_watcher:
                            await self.file_watcher.stop_watching()
                            
                            response_data = {
                                "status": "watching_stopped",
                                "final_stats": self.file_watcher.get_statistics()
                            }
                        else:
                            response_data = {
                                "status": "file_watcher_unavailable"
                            }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Stop watching error: {str(e)}"}],
                            "isError": True
                        }
                
                async def get_recent_changes(self, limit: int) -> Dict[str, Any]:
                    """Get recent file changes"""
                    try:
                        recent = self.realtime_server.recent_changes[-limit:]
                        
                        response_data = {
                            "recent_changes": recent,
                            "total_changes": len(self.realtime_server.recent_changes),
                            "limit": limit
                        }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Get changes error: {str(e)}"}],
                            "isError": True
                        }
                
                async def get_recent_analyses(self, limit: int) -> Dict[str, Any]:
                    """Get recent analysis results"""
                    try:
                        recent = self.realtime_server.recent_analyses[-limit:]
                        
                        response_data = {
                            "recent_analyses": recent,
                            "total_analyses": len(self.realtime_server.recent_analyses),
                            "limit": limit
                        }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Get analyses error: {str(e)}"}],
                            "isError": True
                        }
                
                async def get_realtime_status(self) -> Dict[str, Any]:
                    """Get real-time monitoring status"""
                    try:
                        if self.file_watcher:
                            stats = self.file_watcher.get_statistics()
                            
                            response_data = {
                                "realtime_status": "active" if self.file_watcher.is_watching else "inactive",
                                "watcher_statistics": stats,
                                "recent_activity": {
                                    "changes_last_hour": len([c for c in self.realtime_server.recent_changes 
                                                             if (datetime.now() - datetime.fromisoformat(c['timestamp'])).seconds < 3600]),
                                    "analyses_last_hour": len([a for a in self.realtime_server.recent_analyses 
                                                              if (datetime.now() - datetime.fromisoformat(a['timestamp'])).seconds < 3600])
                                }
                            }
                        else:
                            response_data = {
                                "realtime_status": "unavailable",
                                "error": "File watcher not initialized"
                            }
                        
                        return {
                            "content": [{"type": "resource", "data": json.dumps(response_data, indent=2), "mimeType": "application/json"}],
                            "isError": False
                        }
                    except Exception as e:
                        return {
                            "content": [{"type": "text", "text": f"Status error: {str(e)}"}],
                            "isError": True
                        }
            
            return EnhancedMCPServer(self.glean_client, self.file_watcher, self)
            
        except Exception as e:
            logger.error(f"Failed to create enhanced server: {e}")
            raise
    
    def _on_file_change(self, change):
        """Callback for file changes"""
        change_data = change.to_dict()
        self.recent_changes.append(change_data)
        
        # Keep only recent items
        if len(self.recent_changes) > self.max_recent_items:
            self.recent_changes = self.recent_changes[-self.max_recent_items:]
        
        logger.info(f"File change: {Path(change.file_path).name} ({change.change_type.value})")
    
    def _on_analysis_complete(self, result):
        """Callback for analysis completion"""
        analysis_data = result.to_dict()
        self.recent_analyses.append(analysis_data)
        
        # Keep only recent items
        if len(self.recent_analyses) > self.max_recent_items:
            self.recent_analyses = self.recent_analyses[-self.max_recent_items:]
        
        if result.ai_insights:
            logger.info(f"AI analysis: {Path(result.file_path).name} - {len(result.symbols_found)} symbols, {result.processing_time:.2f}s")
        else:
            logger.info(f"Analysis: {Path(result.file_path).name} - {len(result.symbols_found)} symbols, {result.processing_time:.2f}s")
    
    async def start_http_server(self):
        """Start enhanced HTTP server with real-time capabilities"""
        try:
            from aiohttp import web
            
            # Create the enhanced MCP server
            self.glean_server = await self.create_enhanced_server()
            
            # Create web app
            app = web.Application()
            
            async def handle_mcp(request):
                """Handle MCP requests"""
                try:
                    data = await request.json()
                    response = await self.glean_server.handle_mcp_request(data)
                    
                    return web.json_response(
                        response,
                        headers={'Access-Control-Allow-Origin': '*'}
                    )
                except Exception as e:
                    logger.error(f"Error handling MCP request: {e}")
                    return web.json_response(
                        {"error": str(e)},
                        status=500,
                        headers={'Access-Control-Allow-Origin': '*'}
                    )
            
            async def handle_status(request):
                """Handle status requests"""
                realtime_active = self.file_watcher.is_watching if self.file_watcher else False
                
                return web.json_response({
                    "status": "running",
                    "server": "realtime-strands-glean-mcp",
                    "transport": "http",
                    "version": "2.0.0",
                    "realtime_monitoring": realtime_active,
                    "recent_changes": len(self.recent_changes),
                    "recent_analyses": len(self.recent_analyses)
                }, headers={'Access-Control-Allow-Origin': '*'})
            
            async def handle_tools(request):
                """Handle tools list"""
                tools_request = {"jsonrpc": "2.0", "id": "list_tools", "method": "tools/list"}
                response = await self.glean_server.handle_mcp_request(tools_request)
                return web.json_response(
                    response,
                    headers={'Access-Control-Allow-Origin': '*'}
                )
            
            async def handle_options(request):
                """Handle CORS preflight"""
                return web.Response(
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                    }
                )
            
            # Setup routes
            app.router.add_post('/mcp', handle_mcp)
            app.router.add_get('/mcp/status', handle_status)
            app.router.add_get('/mcp/tools', handle_tools)
            app.router.add_options('/mcp', handle_options)
            
            # Start server
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            self.server = (runner, site)
            
            logger.info(f"🚀 Real-time MCP Server running on http://{self.host}:{self.port}")
            logger.info(f"   • Status: http://{self.host}:{self.port}/mcp/status")
            logger.info(f"   • Tools: http://{self.host}:{self.port}/mcp/tools")
            logger.info(f"   • MCP Endpoint: http://{self.host}:{self.port}/mcp")
            logger.info(f"   • Real-time File Watching: Available")
            
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            raise
    
    async def stop(self):
        """Stop the server and file watcher"""
        if self.file_watcher:
            await self.file_watcher.stop_watching()
        
        if self.server:
            runner, site = self.server
            await site.stop()
            await runner.cleanup()
            self.server = None
            
        logger.info("Real-time server stopped")


async def main():
    """Main server launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Real-time Strands-Glean MCP Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8082, help="Server port")
    parser.add_argument("--start-watching", action="store_true", help="Start file watching immediately")
    
    args = parser.parse_args()
    
    # Create and start launcher
    launcher = RealtimeMCPServer(args.host, args.port)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(launcher.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await launcher.start_http_server()
        
        # Auto-start file watching if requested
        if args.start_watching:
            if launcher.file_watcher:
                logger.info("🔍 Starting file watching...")
                await launcher.file_watcher.start_watching()
                logger.info("✅ File watching started")
            else:
                logger.warning("⚠️ File watcher not available")
        
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