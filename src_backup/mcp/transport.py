"""
MCP Transport Layer Implementation
Supports stdio, WebSocket, and Server-Sent Events transports
"""
import asyncio
import json
import sys
import websockets
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict
import logging

logger = logging.getLogger(__name__)


class MCPTransport(ABC):
    """Abstract base class for MCP transports"""
    
    def __init__(self):
        self.message_handler: Optional[Callable[[str], Any]] = None
        self.is_connected = False
    
    def set_message_handler(self, handler: Callable[[str], Any]):
        """Set the message handler for incoming messages"""
        self.message_handler = handler
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the transport"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the transport"""
        pass
    
    @abstractmethod
    async def send_message(self, message: str):
        """Send a message through the transport"""
        pass
    
    @abstractmethod
    async def receive_messages(self):
        """Start receiving messages (should run in background)"""
        pass


class StdioTransport(MCPTransport):
    """Standard input/output transport for MCP"""
    
    def __init__(self):
        super().__init__()
        self.reader = None
        self.writer = None
        self._stop_event = asyncio.Event()
    
    async def connect(self) -> bool:
        """Connect to stdio streams"""
        try:
            # Set up async readers/writers for stdin/stdout
            loop = asyncio.get_event_loop()
            self.reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(self.reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)
            
            transport, protocol = await loop.connect_write_pipe(
                asyncio.streams.FlowControlMixin, sys.stdout
            )
            self.writer = asyncio.StreamWriter(transport, protocol, self.reader, loop)
            
            self.is_connected = True
            logger.info("Connected to stdio transport")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to stdio: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from stdio"""
        self._stop_event.set()
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.is_connected = False
        logger.info("Disconnected from stdio transport")
    
    async def send_message(self, message: str):
        """Send message to stdout"""
        if not self.is_connected or not self.writer:
            raise RuntimeError("Transport not connected")
        
        # MCP messages are sent as JSON lines
        line = message + '\n'
        self.writer.write(line.encode())
        await self.writer.drain()
        logger.debug(f"Sent message: {message}")
    
    async def receive_messages(self):
        """Receive messages from stdin"""
        if not self.is_connected or not self.reader:
            raise RuntimeError("Transport not connected")
        
        try:
            while not self._stop_event.is_set():
                # Read line from stdin
                line = await self.reader.readline()
                if not line:
                    break
                
                message = line.decode().strip()
                if message and self.message_handler:
                    try:
                        await self.message_handler(message)
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
        finally:
            self.is_connected = False


class WebSocketTransport(MCPTransport):
    """WebSocket transport for MCP"""
    
    def __init__(self, uri: str):
        super().__init__()
        self.uri = uri
        self.websocket = None
        self._stop_event = asyncio.Event()
    
    async def connect(self) -> bool:
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            logger.info(f"Connected to WebSocket: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket {self.uri}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self._stop_event.set()
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("Disconnected from WebSocket transport")
    
    async def send_message(self, message: str):
        """Send message through WebSocket"""
        if not self.is_connected or not self.websocket:
            raise RuntimeError("Transport not connected")
        
        await self.websocket.send(message)
        logger.debug(f"Sent WebSocket message: {message}")
    
    async def receive_messages(self):
        """Receive messages from WebSocket"""
        if not self.is_connected or not self.websocket:
            raise RuntimeError("Transport not connected")
        
        try:
            while not self._stop_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=1.0
                    )
                    if self.message_handler:
                        await self.message_handler(message)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    break
        except Exception as e:
            logger.error(f"Error receiving WebSocket messages: {e}")
        finally:
            self.is_connected = False


class SSETransport(MCPTransport):
    """Server-Sent Events transport for MCP"""
    
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.session = None
        self._stop_event = asyncio.Event()
    
    async def connect(self) -> bool:
        """Connect to SSE endpoint"""
        try:
            import aiohttp
            self.session = aiohttp.ClientSession()
            self.is_connected = True
            logger.info(f"Connected to SSE: {self.url}")
            return True
        except ImportError:
            logger.error("aiohttp is required for SSE transport")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to SSE {self.url}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from SSE"""
        self._stop_event.set()
        if self.session:
            await self.session.close()
        self.is_connected = False
        logger.info("Disconnected from SSE transport")
    
    async def send_message(self, message: str):
        """Send message via POST (SSE is typically unidirectional)"""
        if not self.is_connected or not self.session:
            raise RuntimeError("Transport not connected")
        
        # For SSE, we typically send messages via POST to a different endpoint
        post_url = self.url.replace('/events', '/send')
        async with self.session.post(post_url, data=message) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to send message: {response.status}")
        
        logger.debug(f"Sent SSE message: {message}")
    
    async def receive_messages(self):
        """Receive messages from SSE stream"""
        if not self.is_connected or not self.session:
            raise RuntimeError("Transport not connected")
        
        try:
            async with self.session.get(self.url) as response:
                if response.status != 200:
                    raise RuntimeError(f"SSE connection failed: {response.status}")
                
                async for line in response.content:
                    if self._stop_event.is_set():
                        break
                    
                    line = line.decode().strip()
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data and self.message_handler:
                            try:
                                await self.message_handler(data)
                            except Exception as e:
                                logger.error(f"Error handling SSE message: {e}")
        except Exception as e:
            logger.error(f"Error receiving SSE messages: {e}")
        finally:
            self.is_connected = False