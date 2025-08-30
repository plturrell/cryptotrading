"""
MCP-SAP Fiori Integration
Real-time data streaming and tool integration for SAP Fiori Launchpad
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .auth import AuthContext
from .client import MCPClient
from .events import EventType, MCPEvent, create_event_publisher, get_event_streamer
from .metrics import mcp_metrics
from .server import MCPServer
from .tools import MCPTool, ToolResult
from .transport import WebSocketTransport

logger = logging.getLogger(__name__)


@dataclass
class FioriTileConfig:
    """Configuration for SAP Fiori tile integration"""

    tile_id: str
    title: str
    subtitle: str
    mcp_tool: str
    mcp_resource: Optional[str] = None
    refresh_interval: int = 30  # seconds
    auto_refresh: bool = True
    real_time: bool = False
    icon: str = "sap-icon://business-card"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "mcp_tool": self.mcp_tool,
            "mcp_resource": self.mcp_resource,
            "refresh_interval": self.refresh_interval,
            "auto_refresh": self.auto_refresh,
            "real_time": self.real_time,
            "icon": self.icon,
        }


@dataclass
class FioriAppConfig:
    """Configuration for SAP Fiori application integration"""

    app_id: str
    app_name: str
    mcp_tools: List[str]
    mcp_resources: List[str]
    navigation_intent: str
    semantic_object: str
    action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_id": self.app_id,
            "app_name": self.app_name,
            "mcp_tools": self.mcp_tools,
            "mcp_resources": self.mcp_resources,
            "navigation_intent": self.navigation_intent,
            "semantic_object": self.semantic_object,
            "action": self.action,
        }


class FioriWebSocketHandler:
    """WebSocket handler for real-time Fiori updates"""

    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.event_streamer = get_event_streamer()
        self.event_publisher = create_event_publisher()

    async def handle_client_connection(self, websocket, path: str):
        """Handle new WebSocket client connection"""
        client_id = f"fiori_client_{int(time.time())}_{id(websocket)}"

        self.connected_clients[client_id] = {
            "websocket": websocket,
            "connected_at": time.time(),
            "subscriptions": set(),
            "path": path,
        }

        logger.info(f"Fiori WebSocket client connected: {client_id}")

        try:
            # Send initial connection message
            await self._send_to_client(
                client_id,
                {
                    "type": "connection_established",
                    "client_id": client_id,
                    "server_capabilities": self._get_server_capabilities(),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client_id, message)

        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
        finally:
            # Clean up client
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
            logger.info(f"Fiori WebSocket client disconnected: {client_id}")

    async def _handle_client_message(self, client_id: str, message: str):
        """Handle message from Fiori client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "subscribe_tile":
                await self._handle_tile_subscription(client_id, data)
            elif message_type == "execute_tool":
                await self._handle_tool_execution(client_id, data)
            elif message_type == "get_resource":
                await self._handle_resource_request(client_id, data)
            elif message_type == "ping":
                await self._send_to_client(
                    client_id, {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                )
            else:
                await self._send_error_to_client(client_id, f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            await self._send_error_to_client(client_id, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await self._send_error_to_client(client_id, str(e))

    async def _handle_tile_subscription(self, client_id: str, data: Dict[str, Any]):
        """Handle tile subscription request"""
        tile_id = data.get("tile_id")
        if not tile_id:
            await self._send_error_to_client(client_id, "Missing tile_id")
            return

        client = self.connected_clients.get(client_id)
        if client:
            client["subscriptions"].add(tile_id)

            # Send initial tile data
            await self._send_tile_update(client_id, tile_id)

            await self._send_to_client(
                client_id,
                {
                    "type": "subscription_confirmed",
                    "tile_id": tile_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    async def _handle_tool_execution(self, client_id: str, data: Dict[str, Any]):
        """Handle tool execution request from Fiori"""
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})
        request_id = data.get("request_id")

        if not tool_name:
            await self._send_error_to_client(client_id, "Missing tool_name", request_id)
            return

        try:
            # Get tool from MCP server
            mcp_tool = self.mcp_server.tools.get(tool_name)
            if not mcp_tool:
                await self._send_error_to_client(
                    client_id, f"Tool '{tool_name}' not found", request_id
                )
                return

            # Execute tool
            start_time = mcp_metrics.tool_execution_start(tool_name, client_id)
            result = await mcp_tool.execute(arguments)
            mcp_metrics.tool_execution_end(tool_name, start_time, result.is_success)

            # Send result back to client
            await self._send_to_client(
                client_id,
                {
                    "type": "tool_result",
                    "request_id": request_id,
                    "tool_name": tool_name,
                    "success": result.is_success,
                    "result": result.to_dict(),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Publish event
            await self.event_publisher.publish_tool_execution(
                tool_name, arguments, result.to_dict(), result.is_success
            )

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            await self._send_error_to_client(client_id, str(e), request_id)

    async def _handle_resource_request(self, client_id: str, data: Dict[str, Any]):
        """Handle resource request from Fiori"""
        resource_uri = data.get("resource_uri")
        request_id = data.get("request_id")

        if not resource_uri:
            await self._send_error_to_client(client_id, "Missing resource_uri", request_id)
            return

        try:
            # Get resource from MCP server
            resource = self.mcp_server.resources.get(resource_uri)
            if not resource:
                await self._send_error_to_client(
                    client_id, f"Resource '{resource_uri}' not found", request_id
                )
                return

            # Send resource data
            await self._send_to_client(
                client_id,
                {
                    "type": "resource_data",
                    "request_id": request_id,
                    "resource_uri": resource_uri,
                    "data": resource.to_dict(),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Resource request error: {e}")
            await self._send_error_to_client(client_id, str(e), request_id)

    async def _send_tile_update(self, client_id: str, tile_id: str):
        """Send tile update to client"""
        # This would be configured based on tile configuration
        # For now, send a sample update
        await self._send_to_client(
            client_id,
            {
                "type": "tile_update",
                "tile_id": tile_id,
                "data": {
                    "value": "Loading...",
                    "trend": "neutral",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            },
        )

    async def _send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        client = self.connected_clients.get(client_id)
        if client and "websocket" in client:
            try:
                await client["websocket"].send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to client {client_id}: {e}")

    async def _send_error_to_client(self, client_id: str, error: str, request_id: str = None):
        """Send error message to client"""
        message = {"type": "error", "error": error, "timestamp": datetime.utcnow().isoformat()}
        if request_id:
            message["request_id"] = request_id

        await self._send_to_client(client_id, message)

    async def broadcast_tile_update(self, tile_id: str, data: Dict[str, Any]):
        """Broadcast tile update to all subscribed clients"""
        message = {
            "type": "tile_update",
            "tile_id": tile_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        for client_id, client in self.connected_clients.items():
            if tile_id in client.get("subscriptions", set()):
                await self._send_to_client(client_id, message)

    def _get_server_capabilities(self) -> Dict[str, Any]:
        """Get MCP server capabilities for Fiori"""
        return {
            "tools": [tool.to_dict() for tool in self.mcp_server.tools.values()],
            "resources": list(self.mcp_server.resources.keys()),
            "real_time_updates": True,
            "authentication": True,
        }


class MCPFioriIntegration:
    """Main MCP-Fiori integration class"""

    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
        self.tile_configs: Dict[str, FioriTileConfig] = {}
        self.app_configs: Dict[str, FioriAppConfig] = {}
        self.websocket_handler = FioriWebSocketHandler(mcp_server)
        self.event_publisher = create_event_publisher()

        # Setup default crypto trading tiles
        self._setup_default_tiles()

        logger.info("MCP-Fiori integration initialized")

    def _setup_default_tiles(self):
        """Setup default crypto trading tiles"""
        # Bitcoin tile
        self.register_tile(
            FioriTileConfig(
                tile_id="bitcoin_tile",
                title="Bitcoin",
                subtitle="BTC Price & Volume",
                mcp_tool="get_market_data",
                refresh_interval=30,
                real_time=True,
                icon="sap-icon://currency",
            )
        )

        # Ethereum tile
        self.register_tile(
            FioriTileConfig(
                tile_id="ethereum_tile",
                title="Ethereum",
                subtitle="ETH Price & Volume",
                mcp_tool="get_market_data",
                refresh_interval=30,
                real_time=True,
                icon="sap-icon://currency",
            )
        )

        # Portfolio tile
        self.register_tile(
            FioriTileConfig(
                tile_id="portfolio_tile",
                title="Portfolio",
                subtitle="Total Value & P&L",
                mcp_tool="get_portfolio",
                refresh_interval=60,
                real_time=False,
                icon="sap-icon://pie-chart",
            )
        )

        # Trading signals tile
        self.register_tile(
            FioriTileConfig(
                tile_id="signals_tile",
                title="Trading Signals",
                subtitle="AI Recommendations",
                mcp_tool="get_trading_signals",
                refresh_interval=120,
                real_time=True,
                icon="sap-icon://trend-up",
            )
        )

    def register_tile(self, config: FioriTileConfig):
        """Register a Fiori tile configuration"""
        self.tile_configs[config.tile_id] = config
        logger.info(f"Registered Fiori tile: {config.tile_id}")

    def register_app(self, config: FioriAppConfig):
        """Register a Fiori application configuration"""
        self.app_configs[config.app_id] = config
        logger.info(f"Registered Fiori app: {config.app_id}")

    async def start_real_time_updates(self):
        """Start real-time tile updates"""
        for tile_id, config in self.tile_configs.items():
            if config.auto_refresh:
                asyncio.create_task(self._tile_update_loop(tile_id, config))

        logger.info("Started real-time Fiori tile updates")

    async def _tile_update_loop(self, tile_id: str, config: FioriTileConfig):
        """Update loop for a specific tile"""
        while True:
            try:
                # Get fresh data using MCP tool
                tile_data = await self._get_tile_data(config)

                # Broadcast update to connected clients
                await self.websocket_handler.broadcast_tile_update(tile_id, tile_data)

                # Wait for next update
                await asyncio.sleep(config.refresh_interval)

            except Exception as e:
                logger.error(f"Error updating tile {tile_id}: {e}")
                await asyncio.sleep(config.refresh_interval)

    async def _get_tile_data(self, config: FioriTileConfig) -> Dict[str, Any]:
        """Get data for a tile using MCP tools"""
        try:
            # Get MCP tool
            mcp_tool = self.mcp_server.tools.get(config.mcp_tool)
            if not mcp_tool:
                return {"error": f"Tool '{config.mcp_tool}' not found"}

            # Determine arguments based on tile type
            arguments = self._get_tile_arguments(config)

            # Execute tool
            result = await mcp_tool.execute(arguments)

            if result.is_success:
                # Format data for Fiori tile
                return self._format_tile_data(config, result.to_dict())
            else:
                return {"error": result.content}

        except Exception as e:
            logger.error(f"Error getting tile data for {config.tile_id}: {e}")
            return {"error": str(e)}

    def _get_tile_arguments(self, config: FioriTileConfig) -> Dict[str, Any]:
        """Get arguments for tile MCP tool execution"""
        if config.tile_id == "bitcoin_tile":
            return {"symbol": "BTC"}
        elif config.tile_id == "ethereum_tile":
            return {"symbol": "ETH"}
        elif config.tile_id == "portfolio_tile":
            return {"user_id": "default"}
        elif config.tile_id == "signals_tile":
            return {"limit": 5}
        else:
            return {}

    def _format_tile_data(
        self, config: FioriTileConfig, raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format raw MCP data for Fiori tile display"""
        if config.tile_id in ["bitcoin_tile", "ethereum_tile"]:
            # Market data formatting
            price = raw_data.get("price", 0)
            change_24h = raw_data.get("change_24h", 0)

            return {
                "value": f"${price:,.2f}",
                "subtitle": f"24h: {change_24h:+.2f}%",
                "trend": "up" if change_24h > 0 else "down" if change_24h < 0 else "neutral",
                "color": "Good" if change_24h > 0 else "Error" if change_24h < 0 else "Neutral",
            }

        elif config.tile_id == "portfolio_tile":
            # Portfolio formatting
            total_value = raw_data.get("total_value", 0)
            pnl = raw_data.get("total_pnl", 0)

            return {
                "value": f"${total_value:,.2f}",
                "subtitle": f"P&L: {pnl:+.2f}%",
                "trend": "up" if pnl > 0 else "down" if pnl < 0 else "neutral",
                "color": "Good" if pnl > 0 else "Error" if pnl < 0 else "Neutral",
            }

        elif config.tile_id == "signals_tile":
            # Trading signals formatting
            signals = raw_data.get("signals", [])
            active_signals = len([s for s in signals if s.get("active", False)])

            return {
                "value": str(active_signals),
                "subtitle": f"{len(signals)} total signals",
                "trend": "up" if active_signals > 0 else "neutral",
                "color": "Good" if active_signals > 0 else "Neutral",
            }

        else:
            # Default formatting
            return {
                "value": str(raw_data.get("value", "N/A")),
                "subtitle": raw_data.get("subtitle", ""),
                "trend": "neutral",
                "color": "Neutral",
            }

    def get_fiori_manifest(self) -> Dict[str, Any]:
        """Generate SAP Fiori manifest.json with MCP integration"""
        return {
            "sap.app": {
                "id": "com.cryptotrading.crypto.trading",
                "type": "application",
                "title": "cryptotrading.com Crypto Trading",
                "description": "Enterprise crypto trading platform with MCP integration",
            },
            "sap.ui": {
                "technology": "UI5",
                "deviceTypes": {"desktop": True, "tablet": True, "phone": True},
            },
            "sap.ui5": {
                "dependencies": {
                    "minUI5Version": "1.108.0",
                    "libs": {"sap.m": {}, "sap.ui.core": {}, "sap.f": {}, "sap.ushell": {}},
                }
            },
            "sap.fiori": {"registrationIds": ["F1234"], "archeType": "analytical"},
            "mcp": {
                "integration": {
                    "enabled": True,
                    "websocket_endpoint": "/mcp/websocket",
                    "tools": [tool.to_dict() for tool in self.mcp_server.tools.values()],
                    "tiles": [config.to_dict() for config in self.tile_configs.values()],
                    "apps": [config.to_dict() for config in self.app_configs.values()],
                }
            },
        }

    def get_fiori_launchpad_config(self) -> Dict[str, Any]:
        """Generate Fiori Launchpad configuration with MCP tiles"""
        tiles = []

        for config in self.tile_configs.values():
            tiles.append(
                {
                    "id": config.tile_id,
                    "title": config.title,
                    "subtitle": config.subtitle,
                    "icon": config.icon,
                    "target": {"semanticObject": "CryptoTrading", "action": "display"},
                    "mcp": {
                        "tool": config.mcp_tool,
                        "resource": config.mcp_resource,
                        "real_time": config.real_time,
                        "refresh_interval": config.refresh_interval,
                    },
                }
            )

        return {
            "launchpad": {
                "tiles": tiles,
                "groups": [
                    {
                        "id": "crypto_trading",
                        "title": "Crypto Trading",
                        "tiles": [config.tile_id for config in self.tile_configs.values()],
                    }
                ],
            },
            "mcp": {
                "websocket_url": "ws://localhost:8080/mcp/websocket",
                "authentication_required": True,
                "real_time_updates": True,
            },
        }

    async def handle_fiori_navigation(
        self, intent: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle navigation from Fiori to MCP-powered applications"""
        try:
            # Parse navigation intent
            parts = intent.split("-")
            if len(parts) >= 2:
                semantic_object = parts[0]
                action = parts[1]

                # Find matching app configuration
                for app_config in self.app_configs.values():
                    if (
                        app_config.semantic_object == semantic_object
                        and app_config.action == action
                    ):
                        # Execute app initialization tools
                        app_data = {}
                        for tool_name in app_config.mcp_tools:
                            tool = self.mcp_server.tools.get(tool_name)
                            if tool:
                                result = await tool.execute(parameters)
                                if result.is_success:
                                    app_data[tool_name] = result.to_dict()

                        return {
                            "success": True,
                            "app_id": app_config.app_id,
                            "app_name": app_config.app_name,
                            "data": app_data,
                        }

            return {"success": False, "error": f"No app found for intent: {intent}"}

        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return {"success": False, "error": str(e)}

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get Fiori integration statistics"""
        return {
            "registered_tiles": len(self.tile_configs),
            "registered_apps": len(self.app_configs),
            "connected_clients": len(self.websocket_handler.connected_clients),
            "mcp_tools_available": len(self.mcp_server.tools),
            "real_time_tiles": len([c for c in self.tile_configs.values() if c.real_time]),
        }


# Global integration instance
global_fiori_integration: Optional[MCPFioriIntegration] = None


def setup_fiori_integration(mcp_server: MCPServer) -> MCPFioriIntegration:
    """Setup MCP-Fiori integration"""
    global global_fiori_integration
    global_fiori_integration = MCPFioriIntegration(mcp_server)
    logger.info("MCP-Fiori integration setup completed")
    return global_fiori_integration


def get_fiori_integration() -> Optional[MCPFioriIntegration]:
    """Get global Fiori integration"""
    return global_fiori_integration
