"""
MCP Serverless Connection Management
Lightweight connection handling for Vercel/serverless environments
"""
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

from .transport import MCPTransport, StdioTransport, WebSocketTransport
from .client import MCPClient
from .auth import AuthContext

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for MCP connections"""
    transport_type: str = 'stdio'
    transport_config: Dict[str, Any] = None
    timeout: float = 10.0
    retry_attempts: int = 3
    
    def __post_init__(self):
        if self.transport_config is None:
            self.transport_config = {}


class ServerlessConnectionManager:
    """Lightweight connection manager for serverless environments"""
    
    def __init__(self):
        self.connection_timeout = 10.0
        self.max_retry_attempts = 3


    async def create_connection(self, config: ConnectionConfig, 
                              auth_context: AuthContext = None) -> MCPClient:
        """Create a single MCP connection (serverless-friendly)"""
        for attempt in range(config.retry_attempts):
            try:
                # Create transport
                if config.transport_type == 'stdio':
                    transport = StdioTransport()
                elif config.transport_type == 'websocket':
                    uri = config.transport_config.get('uri', 'ws://localhost:8080/mcp')
                    transport = WebSocketTransport(uri)
                else:
                    raise ValueError(f"Unsupported transport type: {config.transport_type}")
                
                # Create client
                client_name = f"mcp_client_{int(time.time())}"
                client = MCPClient(client_name, "1.0.0", transport)
                
                # Connect with timeout
                connected = await asyncio.wait_for(
                    client.connect(), 
                    timeout=config.timeout
                )
                
                if not connected:
                    raise RuntimeError("Failed to connect client")
                
                # Initialize
                await asyncio.wait_for(
                    client.initialize(), 
                    timeout=config.timeout
                )
                
                logger.info(f"Created MCP connection ({config.transport_type})")
                return client
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == config.retry_attempts - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    async def execute_with_connection(self, 
                                    config: ConnectionConfig,
                                    operation: callable,
                                    auth_context: AuthContext = None) -> Any:
        """Execute operation with a temporary connection"""
        client = None
        try:
            client = await self.create_connection(config, auth_context)
            return await operation(client)
        finally:
            if client:
                try:
                    await client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting client: {e}")
    

    

# Global serverless connection manager instance
serverless_connection_manager = ServerlessConnectionManager()
