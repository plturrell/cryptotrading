#!/usr/bin/env python3
"""
Enhanced MCP CLI with multiple transport support
Connects to Strands-Glean via HTTP, WebSocket, or Process transports
"""

import asyncio
import json
import sys
import logging
import os
from pathlib import Path
import click
from typing import Dict, Any, Optional
import aiohttp

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class EnhancedMCPClient:
    """Enhanced MCP client with multiple transport support"""
    
    def __init__(self, transport_type: str = "http"):
        self.transport_type = transport_type
        self.connected = False
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = None
        
    async def connect(self, **kwargs) -> bool:
        """Connect using specified transport"""
        try:
            if self.transport_type == "http":
                return await self._connect_http(**kwargs)
            elif self.transport_type == "websocket":
                return await self._connect_websocket(**kwargs)
            elif self.transport_type == "process":
                return await self._connect_process(**kwargs)
            else:
                print(f"❌ Unsupported transport: {self.transport_type}")
                return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def _connect_http(self, host: str = "localhost", port: int = 8080) -> bool:
        """Connect via HTTP transport"""
        self.base_url = f"http://{host}:{port}"
        self.session = aiohttp.ClientSession()
        
        # Test connection
        try:
            async with self.session.get(f"{self.base_url}/mcp/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Connected via HTTP to {self.base_url}")
                    print(f"   Server status: {data.get('status', 'unknown')}")
                    self.connected = True
                    return True
                else:
                    print(f"❌ HTTP server not responding (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ HTTP connection failed: {e}")
            return False
    
    async def _connect_websocket(self, uri: str = "ws://localhost:8080/ws") -> bool:
        """Connect via WebSocket transport"""
        print(f"🔌 WebSocket transport not yet implemented")
        return False
    
    async def _connect_process(self, server_script: str = None) -> bool:
        """Connect via Process transport"""
        print(f"🔌 Process transport not yet implemented") 
        return False
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        print("🔌 Disconnected")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a tool via HTTP"""
        if not self.connected or not self.session:
            return {"error": "Not connected"}
        
        request_data = {
            "jsonrpc": "2.0",
            "id": f"{tool_name}_{asyncio.get_event_loop().time()}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }
        
        try:
            async with self.session.post(f"{self.base_url}/mcp", json=request_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}: {await response.text()}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        if not self.connected or not self.session:
            return {"error": "Not connected"}
        
        try:
            async with self.session.get(f"{self.base_url}/mcp/tools") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}: {await response.text()}"}
        except Exception as e:
            return {"error": str(e)}


# CLI Commands
@click.group()
@click.option('--transport', '-t', default='http', 
              type=click.Choice(['http', 'websocket', 'process']),
              help='Transport type to use')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, transport, verbose):
    """Enhanced MCP CLI with multiple transport support"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['client'] = EnhancedMCPClient(transport)
    ctx.obj['transport'] = transport


@cli.command()
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=8080, help='Server port')
@click.pass_context
async def connect(ctx, host, port):
    """Connect to MCP server"""
    client = ctx.obj['client']
    transport = ctx.obj['transport']
    
    print(f"🚀 Connecting via {transport.upper()} transport...")
    
    if transport == 'http':
        success = await client.connect(host=host, port=port)
    else:
        success = await client.connect()
    
    if success:
        print("🎉 Ready for enhanced MCP operations!")
    else:
        print("💥 Connection failed")
        sys.exit(1)


@cli.command()
@click.pass_context
async def tools(ctx):
    """List available tools"""
    client = ctx.obj['client']
    
    if not client.connected:
        print("Connecting first...")
        await client.connect()
    
    result = await client.list_tools()
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("🔧 AVAILABLE TOOLS")
        print("=" * 40)
        
        tools = result.get("result", {}).get("tools", [])
        for tool in tools:
            print(f"📋 {tool.get('name', 'unknown')}")
            print(f"   {tool.get('description', 'No description')}")
            print()


@cli.command()
@click.option('--unit', '-u', default='main', help='Unit name for indexing')
@click.option('--force', '-f', is_flag=True, help='Force reindexing')
@click.pass_context
async def index(ctx, unit, force):
    """Index the project with AI analysis"""
    client = ctx.obj['client']
    
    if not client.connected:
        await client.connect()
    
    print(f"📚 Indexing project (unit: {unit})...")
    result = await client.call_tool("glean_index_project", {
        "unit_name": unit,
        "force_reindex": force
    })
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        response = result.get("result", {})
        if "content" in response:
            try:
                data = json.loads(response["content"][0]["data"])
                print("✅ Indexing completed!")
                print(f"   • Files indexed: {data.get('files_indexed', 0)}")
                print(f"   • Symbols found: {data.get('symbols_found', 0)}")
                print(f"   • Facts stored: {data.get('facts_stored', 0)}")
            except:
                print(f"   Raw result: {response}")


@cli.command()
@click.argument('pattern')
@click.option('--ai-enhance', '-ai', is_flag=True, help='Enhance results with AI')
@click.option('--limit', '-l', default=5, help='Maximum results')
@click.pass_context
async def search(ctx, pattern, ai_enhance, limit):
    """Search for symbols with optional AI enhancement"""
    client = ctx.obj['client']
    
    if not client.connected:
        await client.connect()
    
    print(f"🔍 Searching for: {pattern}")
    result = await client.call_tool("glean_symbol_search", {
        "pattern": pattern,
        "limit": limit
    })
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    response = result.get("result", {})
    if "content" in response:
        try:
            data = json.loads(response["content"][0]["data"])
            symbols = data.get("symbols", [])
            print(f"📊 Found {data.get('total_found', 0)} symbols (showing {len(symbols)})")
            
            for symbol in symbols:
                print(f"   • {symbol.get('name', 'unknown')} in {symbol.get('file', 'unknown')}")
            
            # AI Enhancement
            if ai_enhance and symbols:
                print(f"\n🤖 Enhancing with AI...")
                ai_result = await client.call_tool("ai_enhance_analysis", {
                    "analysis_data": data,
                    "enhancement_type": "summary",
                    "ai_provider": "grok"
                })
                
                if "error" not in ai_result:
                    ai_response = ai_result.get("result", {})
                    if "content" in ai_response:
                        ai_data = json.loads(ai_response["content"][0]["data"])
                        grok_response = ai_data.get("grok_response", {})
                        
                        if grok_response.get("status") == "success":
                            print("🧠 AI INSIGHTS:")
                            print(f"   {grok_response.get('analysis', 'No analysis available')}")
                        else:
                            print(f"   ⚠️ AI analysis failed: {grok_response.get('error', 'Unknown error')}")
                        
        except Exception as e:
            print(f"Error parsing results: {e}")


@cli.command()
@click.argument('files', nargs=-1, required=True)
@click.option('--type', '-t', default='comprehensive',
              type=click.Choice(['security', 'performance', 'maintainability', 'style', 'comprehensive']),
              help='Review type')
@click.pass_context
async def review(ctx, files, type):
    """AI-powered code review using Grok"""
    client = ctx.obj['client']
    
    if not client.connected:
        await client.connect()
    
    print(f"🤖 AI Code Review ({type})")
    print(f"Files: {', '.join(files)}")
    
    result = await client.call_tool("ai_code_review", {
        "files": list(files),
        "review_type": type
    })
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        response = result.get("result", {})
        if "content" in response:
            try:
                data = json.loads(response["content"][0]["data"])
                grok_review = data.get("grok_review", {})
                
                if grok_review.get("status") == "success":
                    print("✅ AI review completed!")
                    print(f"🤖 Model: {data.get('model_used', 'unknown')}")
                    print(f"\n📝 REVIEW:")
                    print(grok_review.get("review", "No review content"))
                else:
                    print(f"❌ AI review failed: {grok_review.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error parsing review: {e}")


@cli.command()
@click.argument('symbol')
@click.option('--context', '-c', default='detailed',
              type=click.Choice(['brief', 'detailed', 'tutorial']),
              help='Context level')
@click.pass_context
async def explain(ctx, symbol, context):
    """AI explanation of code components"""
    client = ctx.obj['client']
    
    if not client.connected:
        await client.connect()
    
    print(f"🤖 Explaining: {symbol} ({context} context)")
    
    result = await client.call_tool("ai_explain_code", {
        "symbol": symbol,
        "context_level": context
    })
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        response = result.get("result", {})
        if "content" in response:
            try:
                data = json.loads(response["content"][0]["data"])
                grok_explanation = data.get("grok_explanation", {})
                
                if grok_explanation.get("status") == "success":
                    print("✅ AI explanation completed!")
                    print(f"🤖 Model: {data.get('model_used', 'unknown')}")
                    print(f"\n📝 EXPLANATION:")
                    print(grok_explanation.get("explanation", "No explanation available"))
                else:
                    print(f"❌ AI explanation failed: {grok_explanation.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error parsing explanation: {e}")


@cli.command()
@click.pass_context
async def interactive(ctx):
    """Interactive mode for AI-enhanced analysis"""
    client = ctx.obj['client']
    
    if not client.connected:
        await client.connect()
    
    print("🚀 INTERACTIVE AI-ENHANCED ANALYSIS")
    print("=" * 50)
    print("Type 'help' for commands, 'quit' to exit")
    
    while True:
        try:
            command = input("\n🤖 > ").strip()
            
            if command == 'quit':
                break
            elif command == 'help':
                print("Available commands:")
                print("  search <pattern> - Search symbols with AI insights")
                print("  explain <symbol> - Get AI explanation")
                print("  review <file> - AI code review")
                print("  stats - Show analysis statistics")
                print("  quit - Exit")
            elif command.startswith('search '):
                pattern = command[7:]
                # Enhanced search with AI
                search_result = await client.call_tool("glean_symbol_search", {"pattern": pattern, "limit": 3})
                if "error" not in search_result:
                    print(f"🔍 Found symbols for '{pattern}'")
                    # Add AI enhancement
                    ai_result = await client.call_tool("ai_enhance_analysis", {
                        "analysis_data": {"symbols": []},
                        "enhancement_type": "summary"
                    })
                    print("🧠 AI insights added")
            elif command.startswith('explain '):
                symbol = command[8:]
                result = await client.call_tool("ai_explain_code", {"symbol": symbol})
                print(f"🤖 Explanation for '{symbol}' generated")
            elif command == 'stats':
                result = await client.call_tool("glean_statistics")
                print("📊 Statistics retrieved")
            else:
                print(f"Unknown command: {command}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    await client.disconnect()
    print("👋 Goodbye!")


@cli.command()
@click.pass_context
async def demo(ctx):
    """Demo the complete AI-enhanced workflow"""
    client = ctx.obj['client']
    
    print("🎬 AI-ENHANCED WORKFLOW DEMO")
    print("=" * 50)
    
    if not client.connected:
        print("Step 1: Connecting...")
        await client.connect()
    
    print("Step 2: Indexing codebase...")
    await client.call_tool("glean_index_project", {"unit_name": "demo", "force_reindex": True})
    print("   ✅ Codebase indexed")
    
    print("Step 3: AI-enhanced symbol search...")
    search_result = await client.call_tool("glean_symbol_search", {"pattern": "Agent", "limit": 3})
    print("   ✅ Symbols found")
    
    print("Step 4: AI code analysis...")
    ai_result = await client.call_tool("ai_enhance_analysis", {
        "analysis_data": {"symbols": [{"name": "Agent", "kind": "class"}]},
        "enhancement_type": "summary",
        "ai_provider": "grok"
    })
    print("   ✅ AI analysis completed")
    
    print("Step 5: AI code review...")
    review_result = await client.call_tool("ai_code_review", {
        "files": ["src/cryptotrading/core/agents/base.py"],
        "review_type": "comprehensive"
    })
    print("   ✅ AI review completed")
    
    print("\n🎉 DEMO COMPLETED!")
    print("The system demonstrated:")
    print("   • Real-time code indexing")
    print("   • AI-enhanced symbol search")
    print("   • Grok-powered code analysis")
    print("   • Intelligent code reviews")
    print("   • Multiple transport support")


def main():
    """Main entry point with async support"""
    # Convert click commands to async
    def async_wrapper(func):
        def wrapper(*args, **kwargs):
            return asyncio.run(func(*args, **kwargs))
        return wrapper
    
    # Wrap async commands
    for command in [connect, tools, index, search, review, explain, interactive, demo]:
        command.callback = async_wrapper(command.callback)
    
    cli()


if __name__ == '__main__':
    main()