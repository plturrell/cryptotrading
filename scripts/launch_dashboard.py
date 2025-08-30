#!/usr/bin/env python3
"""
Launch Real-time Code Analysis Dashboard
Standalone launcher for the dashboard with automatic MCP server detection
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import aiohttp

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def check_mcp_server(url: str, timeout: int = 5) -> bool:
    """Check if MCP server is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/mcp/status", timeout=timeout) as response:
                return response.status == 200
    except:
        return False


async def wait_for_mcp_server(url: str, max_wait: int = 60) -> bool:
    """Wait for MCP server to become available"""
    logger.info(f"Waiting for MCP server at {url}...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        if await check_mcp_server(url):
            logger.info("✅ MCP server is running")
            return True

        logger.info("⏳ MCP server not ready, waiting 2 seconds...")
        await asyncio.sleep(2)

    logger.warning(f"⚠️ MCP server not available after {max_wait}s")
    return False


async def launch_dashboard(
    host: str = "localhost",
    port: int = 8090,
    mcp_url: str = "http://localhost:8082",
    wait_for_mcp: bool = True,
):
    """Launch the dashboard with MCP integration"""

    # Import dashboard after path setup
    try:
        from cryptotrading.core.dashboard.realtime_dashboard import create_dashboard
    except ImportError as e:
        logger.error(f"Failed to import dashboard: {e}")
        return False

    # Check MCP server availability
    if wait_for_mcp:
        if not await wait_for_mcp_server(mcp_url, max_wait=60):
            logger.error("❌ Dashboard requires MCP server to be running")
            logger.info("Start the MCP server first with:")
            logger.info("  python3 scripts/realtime_mcp_server.py --port 8082 --start-watching")
            return False
    else:
        logger.info("⚠️ Skipping MCP server check")

    # Create and start dashboard
    try:
        logger.info(f"🚀 Starting Real-time Dashboard...")
        logger.info(f"   • Host: {host}")
        logger.info(f"   • Port: {port}")
        logger.info(f"   • MCP Server: {mcp_url}")

        dashboard = await create_dashboard(mcp_url)

        if await dashboard.start_server(host, port):
            logger.info("✅ Dashboard started successfully!")
            logger.info("=" * 60)
            logger.info(f"🌐 Dashboard URL: http://{host}:{port}")
            logger.info(f"📊 API Metrics: http://{host}:{port}/api/metrics")
            logger.info(f"🔌 WebSocket: ws://{host}:{port}/ws")
            logger.info("=" * 60)
            logger.info("🎯 Features available:")
            logger.info("   • Real-time file change monitoring")
            logger.info("   • Live code analysis results")
            logger.info("   • AI-enhanced insights via Grok")
            logger.info("   • System performance metrics")
            logger.info("   • WebSocket real-time updates")
            logger.info("=" * 60)
            logger.info("Press Ctrl+C to stop the dashboard")

            try:
                # Keep dashboard running
                while True:
                    await asyncio.sleep(1)

            except KeyboardInterrupt:
                logger.info("\n🛑 Stopping dashboard...")

        else:
            logger.error("❌ Failed to start dashboard")
            return False

    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        return False

    finally:
        try:
            await dashboard.stop_server()
            logger.info("👋 Dashboard stopped")
        except:
            pass

    return True


async def show_dashboard_info(mcp_url: str = "http://localhost:8082"):
    """Show dashboard information and status"""

    print("🔍 REAL-TIME CODE ANALYSIS DASHBOARD")
    print("=" * 60)

    # Check MCP server
    mcp_available = await check_mcp_server(mcp_url)
    print(f"MCP Server Status: {'✅ Running' if mcp_available else '❌ Not available'}")
    print(f"MCP Server URL: {mcp_url}")

    if mcp_available:
        try:
            async with aiohttp.ClientSession() as session:
                # Get tools
                async with session.get(f"{mcp_url}/mcp/tools") as response:
                    if response.status == 200:
                        data = await response.json()
                        tools = data.get("result", {}).get("tools", [])
                        print(f"Available Tools: {len(tools)}")
                        for tool in tools[:5]:  # Show first 5
                            print(f"   • {tool.get('name', 'unknown')}")
                        if len(tools) > 5:
                            print(f"   • ... and {len(tools) - 5} more")

                # Get status
                status_request = {
                    "jsonrpc": "2.0",
                    "id": "dashboard_check",
                    "method": "tools/call",
                    "params": {"name": "realtime_get_status", "arguments": {}},
                }

                async with session.post(f"{mcp_url}/mcp", json=status_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", {})
                        if not result.get("isError", False):
                            content = result.get("content", [{}])[0]
                            if content.get("type") == "resource":
                                import json

                                status_data = json.loads(content.get("data", "{}"))
                                watcher_stats = status_data.get("watcher_statistics", {})

                                print(
                                    f"Real-time Monitoring: {'✅ Active' if status_data.get('realtime_status') == 'active' else '❌ Inactive'}"
                                )
                                print(f"Files Watched: {watcher_stats.get('files_watched', 0)}")
                                print(
                                    f"Changes Detected: {watcher_stats.get('changes_detected', 0)}"
                                )
                                print(
                                    f"Analyses Completed: {watcher_stats.get('analyses_completed', 0)}"
                                )
                                print(
                                    f"AI Insights Generated: {watcher_stats.get('ai_insights_generated', 0)}"
                                )

        except Exception as e:
            print(f"Error getting detailed status: {e}")

    print("\n📋 Dashboard Features:")
    print("   • Real-time file change monitoring")
    print("   • Incremental code analysis with Glean")
    print("   • AI-enhanced insights via Grok")
    print("   • Live system metrics and statistics")
    print("   • WebSocket-based real-time updates")
    print("   • Interactive web interface")

    print("\n🚀 How to start:")
    if not mcp_available:
        print("1. Start the MCP server:")
        print("   python3 scripts/realtime_mcp_server.py --port 8082 --start-watching")
        print("")
    print("2. Start the dashboard:")
    print("   python3 scripts/launch_dashboard.py --port 8090")
    print("")
    print("3. Open your browser:")
    print("   http://localhost:8090")


async def main():
    """Main dashboard launcher"""
    parser = argparse.ArgumentParser(description="Launch Real-time Code Analysis Dashboard")
    parser.add_argument("--host", default="localhost", help="Dashboard host (default: localhost)")
    parser.add_argument("--port", type=int, default=8090, help="Dashboard port (default: 8090)")
    parser.add_argument(
        "--mcp-url",
        default="http://localhost:8082",
        help="MCP server URL (default: http://localhost:8082)",
    )
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for MCP server")
    parser.add_argument("--info", action="store_true", help="Show dashboard info and status")

    args = parser.parse_args()

    if args.info:
        await show_dashboard_info(args.mcp_url)
        return

    # Launch dashboard
    success = await launch_dashboard(
        host=args.host, port=args.port, mcp_url=args.mcp_url, wait_for_mcp=not args.no_wait
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Dashboard launcher stopped")
    except Exception as e:
        print(f"❌ Launcher error: {e}")
        sys.exit(1)
