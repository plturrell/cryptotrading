#!/usr/bin/env python3
"""
MCP Client for Strands-Glean Integration
Connects to the Strands-Glean MCP server for enhanced code analysis
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from cryptotrading.core.protocols.mcp.client import MCPClient
    from cryptotrading.core.protocols.mcp.transport import StdioTransport

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP components not available")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class StrandsGleanMCPClient:
    """Client for interacting with Strands-Glean MCP server"""

    def __init__(self):
        self.client: Optional[MCPClient] = None
        self.connected = False

    async def connect(self, server_command: Optional[str] = None) -> bool:
        """Connect to the MCP server"""
        if not MCP_AVAILABLE:
            print("❌ MCP not available")
            return False

        try:
            # Default server command
            if not server_command:
                server_command = [
                    sys.executable,
                    str(
                        project_root
                        / "src"
                        / "cryptotrading"
                        / "core"
                        / "protocols"
                        / "mcp"
                        / "strands_glean_server.py"
                    ),
                    "stdio",
                ]

            # Create transport and client
            transport = StdioTransport(server_command)
            self.client = MCPClient("strands-glean-cli", "1.0.0", transport)

            # Connect and initialize
            await self.client.connect()
            await self.client.initialize(
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "roots": [{"uri": f"file://{project_root}"}]},
                    "clientInfo": {"name": "strands-glean-cli", "version": "1.0.0"},
                }
            )

            self.connected = True
            print("✅ Connected to Strands-Glean MCP server")
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the server"""
        if self.client and self.connected:
            await self.client.disconnect()
            self.connected = False
            print("🔌 Disconnected from server")

    async def list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        if not self.connected:
            return {"error": "Not connected"}

        try:
            result = await self.client.list_tools()
            return result
        except Exception as e:
            return {"error": str(e)}

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a tool on the server"""
        if not self.connected:
            return {"error": "Not connected"}

        try:
            result = await self.client.call_tool(tool_name, arguments or {})
            return result
        except Exception as e:
            return {"error": str(e)}

    async def index_project(self, unit_name: str = "main", force: bool = False) -> Dict[str, Any]:
        """Index the project"""
        return await self.call_tool(
            "glean_index_project", {"unit_name": unit_name, "force_reindex": force}
        )

    async def search_symbols(self, pattern: str, limit: int = 10) -> Dict[str, Any]:
        """Search for symbols"""
        return await self.call_tool("glean_symbol_search", {"pattern": pattern, "limit": limit})

    async def analyze_dependencies(self, symbol: str, depth: int = 3) -> Dict[str, Any]:
        """Analyze dependencies"""
        return await self.call_tool("glean_dependency_analysis", {"symbol": symbol, "depth": depth})

    async def review_architecture(self, component: str, rules: list = None) -> Dict[str, Any]:
        """Review architecture"""
        return await self.call_tool(
            "glean_architecture_review",
            {
                "component": component,
                "rules": rules or ["layer_separation", "dependency_direction", "circular_deps"],
            },
        )

    async def enhance_with_ai(
        self, analysis_data: Dict[str, Any], enhancement_type: str = "summary"
    ) -> Dict[str, Any]:
        """Enhance analysis with AI"""
        return await self.call_tool(
            "ai_enhance_analysis",
            {
                "analysis_data": analysis_data,
                "enhancement_type": enhancement_type,
                "ai_provider": "grok",
            },
        )

    async def ai_code_review(
        self, files: list, review_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """AI-powered code review"""
        return await self.call_tool("ai_code_review", {"files": files, "review_type": review_type})

    async def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return await self.call_tool("glean_statistics")


# CLI Commands
@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """Strands-Glean MCP Client CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ctx.ensure_object(dict)
    ctx.obj["client"] = StrandsGleanMCPClient()


@cli.command()
@click.pass_context
async def connect(ctx):
    """Connect to the MCP server"""
    client = ctx.obj["client"]
    success = await client.connect()
    if success:
        print("🎉 Ready for enhanced code analysis!")
    else:
        print("💥 Connection failed")
        sys.exit(1)


@cli.command()
@click.pass_context
async def tools(ctx):
    """List available tools"""
    client = ctx.obj["client"]

    if not client.connected:
        await client.connect()

    result = await client.list_tools()
    print("🔧 AVAILABLE TOOLS")
    print("=" * 40)

    if "tools" in result:
        for tool in result["tools"]:
            print(f"📋 {tool.get('name', 'unknown')}")
            print(f"   {tool.get('description', 'No description')}")
            print()
    else:
        print(f"Error: {result}")


@cli.command()
@click.option("--unit", "-u", default="main", help="Unit name for indexing")
@click.option("--force", "-f", is_flag=True, help="Force reindexing")
@click.pass_context
async def index(ctx, unit, force):
    """Index the project"""
    client = ctx.obj["client"]

    if not client.connected:
        await client.connect()

    print(f"📚 Indexing project (unit: {unit})...")
    result = await client.index_project(unit, force)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Indexing completed!")
        if "content" in result and result["content"]:
            try:
                data = json.loads(result["content"][0]["data"])
                print(f"   • Files indexed: {data.get('files_indexed', 0)}")
                print(f"   • Symbols found: {data.get('symbols_found', 0)}")
                print(f"   • Facts stored: {data.get('facts_stored', 0)}")
            except:
                print(f"   Raw result: {result}")


@cli.command()
@click.argument("pattern")
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.pass_context
async def search(ctx, pattern, limit):
    """Search for symbols"""
    client = ctx.obj["client"]

    if not client.connected:
        await client.connect()

    print(f"🔍 Searching for: {pattern}")
    result = await client.search_symbols(pattern, limit)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        if "content" in result and result["content"]:
            try:
                data = json.loads(result["content"][0]["data"])
                symbols = data.get("symbols", [])
                print(f"📊 Found {data.get('total_found', 0)} symbols (showing {len(symbols)})")

                for symbol in symbols:
                    print(
                        f"   • {symbol.get('name', 'unknown')} in {symbol.get('file', 'unknown')}"
                    )
            except:
                print(f"Raw result: {result}")


@cli.command()
@click.argument("symbol")
@click.option("--depth", "-d", default=3, help="Analysis depth")
@click.pass_context
async def deps(ctx, symbol, depth):
    """Analyze dependencies"""
    client = ctx.obj["client"]

    if not client.connected:
        await client.connect()

    print(f"🔗 Analyzing dependencies for: {symbol}")
    result = await client.analyze_dependencies(symbol, depth)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Dependency analysis completed!")
        if "content" in result and result["content"]:
            try:
                data = json.loads(result["content"][0]["data"])
                print(json.dumps(data, indent=2))
            except:
                print(f"Raw result: {result}")


@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.option(
    "--type",
    "-t",
    default="comprehensive",
    type=click.Choice(["security", "performance", "maintainability", "style", "comprehensive"]),
    help="Review type",
)
@click.pass_context
async def review(ctx, files, type):
    """AI-powered code review"""
    client = ctx.obj["client"]

    if not client.connected:
        await client.connect()

    print(f"🤖 AI Code Review ({type})")
    print(f"Files: {', '.join(files)}")

    result = await client.ai_code_review(list(files), type)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ AI review completed!")
        if "content" in result and result["content"]:
            try:
                data = json.loads(result["content"][0]["data"])
                print(f"📊 Overall Score: {data.get('overall_score', 'N/A')}")

                issues = data.get("issues_found", [])
                if issues:
                    print(f"\n⚠️  Issues Found ({len(issues)}):")
                    for issue in issues:
                        print(
                            f"   • {issue.get('type', 'unknown')}: {issue.get('message', 'no message')}"
                        )

                recommendations = data.get("recommendations", [])
                if recommendations:
                    print(f"\n💡 Recommendations:")
                    for rec in recommendations:
                        print(f"   • {rec}")
            except:
                print(f"Raw result: {result}")


@cli.command()
@click.pass_context
async def stats(ctx):
    """Get analysis statistics"""
    client = ctx.obj["client"]

    if not client.connected:
        await client.connect()

    print("📈 Getting analysis statistics...")
    result = await client.get_statistics()

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        if "content" in result and result["content"]:
            try:
                data = json.loads(result["content"][0]["data"])
                print("📊 ANALYSIS STATISTICS")
                print("=" * 40)
                print(f"Project Root: {data.get('project_root', 'unknown')}")
                print(f"Indexed Units: {data.get('indexed_units', [])}")
                print(f"Glean Available: {data.get('glean_available', False)}")

                stats = data.get("statistics", {})
                if stats:
                    print(f"Total Facts: {stats.get('total_facts', 0):,}")
                    print(f"Files Indexed: {stats.get('files_indexed', 0)}")
                    print(f"Total Symbols: {stats.get('total_symbols', 0):,}")
            except:
                print(f"Raw result: {result}")


@cli.command()
@click.pass_context
async def interactive(ctx):
    """Interactive mode for exploration"""
    client = ctx.obj["client"]

    if not client.connected:
        await client.connect()

    print("🚀 INTERACTIVE STRANDS-GLEAN ANALYSIS")
    print("=" * 50)
    print("Type 'help' for commands, 'quit' to exit")

    while True:
        try:
            command = input("\n> ").strip()

            if command == "quit":
                break
            elif command == "help":
                print("Available commands:")
                print("  search <pattern> - Search for symbols")
                print("  deps <symbol> - Analyze dependencies")
                print("  stats - Show statistics")
                print("  tools - List available tools")
                print("  quit - Exit")
            elif command.startswith("search "):
                pattern = command[7:]
                result = await client.search_symbols(pattern, 5)
                print(f"Search results: {result}")
            elif command.startswith("deps "):
                symbol = command[5:]
                result = await client.analyze_dependencies(symbol)
                print(f"Dependency analysis: {result}")
            elif command == "stats":
                result = await client.get_statistics()
                print(f"Statistics: {result}")
            elif command == "tools":
                result = await client.list_tools()
                print(f"Tools: {result}")
            else:
                print(f"Unknown command: {command}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    await client.disconnect()
    print("👋 Goodbye!")


def main():
    """Main entry point with async support"""

    # Convert click commands to async
    def async_wrapper(func):
        def wrapper(*args, **kwargs):
            return asyncio.run(func(*args, **kwargs))

        return wrapper

    # Wrap async commands
    for command in [connect, tools, index, search, deps, review, stats, interactive]:
        command.callback = async_wrapper(command.callback)

    cli()


if __name__ == "__main__":
    main()
