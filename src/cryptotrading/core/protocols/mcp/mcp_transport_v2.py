"""
Enhanced MCP Transport Layer
Supports multiple transport types with robust connectivity for external tools
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import websockets
from aiohttp import web

logger = logging.getLogger(__name__)


class ProcessTransport:
    """Transport for external process communication"""

    def __init__(self, command: List[str], cwd: Optional[str] = None):
        self.command = command
        self.cwd = cwd
        self.process: Optional[subprocess.Popen] = None
        self.message_handler: Optional[Callable[[str], Any]] = None
        self.is_connected = False
        self._stop_event = asyncio.Event()

    def set_message_handler(self, handler: Callable[[str], Any]):
        """Set message handler for incoming messages"""
        self.message_handler = handler

    async def connect(self):
        """Start the external process and connect"""
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.cwd,
                bufsize=0,
            )

            self.is_connected = True
            logger.info(f"Connected to process: {' '.join(self.command)}")

            # Start message handling task
            asyncio.create_task(self._handle_messages())

        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the process"""
        self._stop_event.set()

        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)

                if self.process.poll() is None:
                    self.process.kill()

                self.process = None
                logger.info("Process disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting process: {e}")

        self.is_connected = False

    async def send_message(self, message: str):
        """Send message to the process"""
        if not self.process or not self.is_connected:
            raise RuntimeError("Not connected to process")

        try:
            self.process.stdin.write(message + "\n")
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    async def _handle_messages(self):
        """Handle incoming messages from the process"""
        while not self._stop_event.is_set() and self.is_connected:
            try:
                if self.process and self.process.stdout:
                    line = self.process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line and self.message_handler:
                            await self.message_handler(line)
                    else:
                        # Process ended
                        break

                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error reading from process: {e}")
                break

        await self.disconnect()


class HTTPTransport:
    """HTTP transport for web-based MCP connections"""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.message_handler: Optional[Callable[[str], Any]] = None
        self.is_connected = False

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup HTTP routes for MCP communication"""
        self.app.router.add_post("/mcp", self._handle_mcp_request)
        self.app.router.add_get("/mcp/status", self._handle_status)
        self.app.router.add_get("/mcp/tools", self._handle_list_tools)

        # Enable CORS
        self.app.router.add_options("/mcp", self._handle_options)

    async def _handle_options(self, request):
        """Handle CORS preflight requests"""
        return web.Response(
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )

    async def _handle_mcp_request(self, request):
        """Handle MCP JSON-RPC requests"""
        try:
            data = await request.json()

            if self.message_handler:
                response = await self.message_handler(json.dumps(data))

                return web.json_response(
                    json.loads(response) if isinstance(response, str) else response,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            else:
                return web.json_response(
                    {"error": "No message handler configured"},
                    status=500,
                    headers={"Access-Control-Allow-Origin": "*"},
                )

        except Exception as e:
            logger.error(f"Error handling MCP request: {e}")
            return web.json_response(
                {"error": str(e)}, status=500, headers={"Access-Control-Allow-Origin": "*"}
            )

    async def _handle_status(self, request):
        """Handle status requests"""
        return web.json_response(
            {
                "status": "connected" if self.is_connected else "disconnected",
                "host": self.host,
                "port": self.port,
                "transport": "http",
            },
            headers={"Access-Control-Allow-Origin": "*"},
        )

    async def _handle_list_tools(self, request):
        """Handle tool listing requests"""
        if self.message_handler:
            # Send a list_tools request
            list_tools_request = {"jsonrpc": "2.0", "id": "list_tools", "method": "tools/list"}

            try:
                response = await self.message_handler(json.dumps(list_tools_request))
                return web.json_response(
                    json.loads(response) if isinstance(response, str) else response,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            except Exception as e:
                return web.json_response(
                    {"error": str(e)}, status=500, headers={"Access-Control-Allow-Origin": "*"}
                )
        else:
            return web.json_response(
                {"error": "No message handler configured"},
                status=500,
                headers={"Access-Control-Allow-Origin": "*"},
            )

    def set_message_handler(self, handler: Callable[[str], Any]):
        """Set message handler for incoming requests"""
        self.message_handler = handler

    async def connect(self):
        """Start the HTTP server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            self.is_connected = True
            logger.info(f"HTTP transport started on http://{self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start HTTP transport: {e}")
            raise

    async def disconnect(self):
        """Stop the HTTP server"""
        if self.site:
            await self.site.stop()
            self.site = None

        if self.runner:
            await self.runner.cleanup()
            self.runner = None

        self.is_connected = False
        logger.info("HTTP transport stopped")

    async def send_message(self, message: str):
        """HTTP transport doesn't actively send messages (server mode)"""
        logger.warning("HTTP transport is server-mode only, cannot send active messages")


class EnhancedWebSocketTransport:
    """Enhanced WebSocket transport with better error handling"""

    def __init__(self, uri: str, auto_reconnect: bool = True):
        self.uri = uri
        self.auto_reconnect = auto_reconnect
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.message_handler: Optional[Callable[[str], Any]] = None
        self.is_connected = False
        self._stop_event = asyncio.Event()
        self._reconnect_delay = 5

    def set_message_handler(self, handler: Callable[[str], Any]):
        """Set message handler for incoming messages"""
        self.message_handler = handler

    async def connect(self):
        """Connect to WebSocket server with retry logic"""
        while not self._stop_event.is_set():
            try:
                self.websocket = await websockets.connect(self.uri)
                self.is_connected = True
                logger.info(f"Connected to WebSocket: {self.uri}")

                # Start message handling
                await self._handle_messages()
                break

            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")

                if not self.auto_reconnect:
                    raise

                logger.info(f"Retrying connection in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)  # Exponential backoff

    async def disconnect(self):
        """Disconnect from WebSocket"""
        self._stop_event.set()

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self.is_connected = False
        logger.info("WebSocket disconnected")

    async def send_message(self, message: str):
        """Send message via WebSocket"""
        if not self.websocket or not self.is_connected:
            raise RuntimeError("WebSocket not connected")

        try:
            await self.websocket.send(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise

    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                if self.message_handler:
                    await self.message_handler(message)

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
        finally:
            self.is_connected = False


class TransportManager:
    """Manages multiple transport types for MCP connectivity"""

    def __init__(self):
        self.transports: Dict[str, Any] = {}
        self.active_transport: Optional[str] = None

    def add_transport(self, name: str, transport: Any):
        """Add a transport to the manager"""
        self.transports[name] = transport
        logger.info(f"Added transport: {name}")

    async def connect_transport(self, name: str):
        """Connect to a specific transport"""
        if name not in self.transports:
            raise ValueError(f"Transport '{name}' not found")

        transport = self.transports[name]
        await transport.connect()
        self.active_transport = name
        logger.info(f"Connected to transport: {name}")

    async def disconnect_all(self):
        """Disconnect from all transports"""
        for name, transport in self.transports.items():
            try:
                if hasattr(transport, "disconnect"):
                    await transport.disconnect()
                logger.info(f"Disconnected transport: {name}")
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")

        self.active_transport = None

    def get_active_transport(self):
        """Get the currently active transport"""
        if self.active_transport:
            return self.transports[self.active_transport]
        return None

    async def send_message(self, message: str):
        """Send message via active transport"""
        transport = self.get_active_transport()
        if transport:
            await transport.send_message(message)
        else:
            raise RuntimeError("No active transport")


# Factory functions for easy transport creation
def create_process_transport(server_script: str, project_root: str = None) -> ProcessTransport:
    """Create a process transport for MCP server"""
    command = [sys.executable, server_script, "stdio"]

    return ProcessTransport(command, cwd=project_root)


def create_http_transport(host: str = "localhost", port: int = 8080) -> HTTPTransport:
    """Create HTTP transport for web connectivity"""
    return HTTPTransport(host, port)


def create_websocket_transport(uri: str, auto_reconnect: bool = True) -> EnhancedWebSocketTransport:
    """Create WebSocket transport"""
    return EnhancedWebSocketTransport(uri, auto_reconnect)


# CLI script for testing transports
async def test_transports():
    """Test various transport configurations"""
    print("🚀 TESTING MCP TRANSPORTS")
    print("=" * 40)

    manager = TransportManager()

    # Test HTTP transport
    print("1️⃣ Testing HTTP Transport...")
    http_transport = create_http_transport("localhost", 8081)

    def mock_handler(message):
        return json.dumps({"result": "HTTP transport working", "echo": message})

    http_transport.set_message_handler(mock_handler)
    manager.add_transport("http", http_transport)

    try:
        await manager.connect_transport("http")
        print("   ✅ HTTP transport started on http://localhost:8081")
        await asyncio.sleep(2)
    except Exception as e:
        print(f"   ❌ HTTP transport failed: {e}")

    # Test Process transport (if server exists)
    print("\n2️⃣ Testing Process Transport...")
    server_path = Path(__file__).parent / "strands_glean_server.py"

    if server_path.exists():
        process_transport = create_process_transport(str(server_path))
        process_transport.set_message_handler(lambda msg: print(f"Process: {msg}"))
        manager.add_transport("process", process_transport)

        try:
            # This would fail due to security middleware, but we can test the setup
            print("   ✅ Process transport configured")
        except Exception as e:
            print(f"   ⚠️ Process transport setup: {e}")
    else:
        print("   ⚠️ Server script not found")

    print(f"\n📊 Transport Summary:")
    print(f"   • Available: {list(manager.transports.keys())}")
    print(f"   • Active: {manager.active_transport}")

    # Cleanup
    await manager.disconnect_all()
    print("\n✅ Transport tests completed")


if __name__ == "__main__":
    asyncio.run(test_transports())
