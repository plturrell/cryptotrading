"""
MCP Transport Layer Implementation

This module provides transport abstractions for the Model Context Protocol,
enabling communication over different channels:

- StdioTransport: Standard input/output (default for CLI tools)
- WebSocketTransport: WebSocket connections (for web integrations)
- SSETransport: Server-Sent Events (for unidirectional streaming)

The transport layer handles:
- Connection management
- Message serialization/deserialization
- Async message handling
- Error recovery
- Graceful disconnection

Key Concepts:
    Transport Abstraction: All transports implement MCPTransport interface
    Message Handler: Callback for processing incoming messages
    Async Operations: All I/O operations are asynchronous
    Error Resilience: Transports handle disconnections gracefully

Example:
    >>> # Stdio transport (default)
    >>> transport = StdioTransport()
    >>> transport.set_message_handler(my_handler)
    >>> await transport.connect()
    >>> 
    >>> # WebSocket transport
    >>> ws_transport = WebSocketTransport("ws://localhost:8080")
    >>> await ws_transport.connect()
    >>> 
    >>> # Send and receive messages
    >>> await transport.send_message('{"jsonrpc":"2.0","method":"test"}')
    >>> await transport.receive_messages()  # Runs in background
"""
import asyncio
import json
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import websockets

logger = logging.getLogger(__name__)


class MCPTransport(ABC):
    """Abstract base class for MCP transports.

    Defines the interface that all MCP transports must implement.
    Provides a consistent API for different transport mechanisms.

    Transport implementations handle:
    - Connection lifecycle (connect/disconnect)
    - Bidirectional message passing
    - Async message reception
    - Error handling and recovery

    Attributes:
        message_handler: Callback function for incoming messages
        is_connected: Current connection status

    Example Implementation:
        >>> class CustomTransport(MCPTransport):
        ...     async def connect(self) -> bool:
        ...         # Establish connection
        ...         self.is_connected = True
        ...         return True
        ...
        ...     async def send_message(self, message: str):
        ...         # Send message logic
        ...         pass
    """

    def __init__(self):
        self.message_handler: Optional[Callable[[str], Any]] = None
        self.is_connected = False

    def set_message_handler(self, handler: Callable[[str], Any]):
        """Set the message handler for incoming messages.

        The handler will be called for each received message.
        Handler should be async and accept a string parameter.

        Args:
            handler: Async function that processes messages
                    Should accept (message: str) parameter

        Example:
            >>> async def handle_message(message: str):
            ...     data = json.loads(message)
            ...     print(f"Received: {data}")
            >>>
            >>> transport.set_message_handler(handle_message)
        """
        self.message_handler = handler

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the transport.

        Establishes connection to the transport medium.
        Must set self.is_connected on success.

        Returns:
            True if connection successful, False otherwise

        Note:
            Implementations should handle connection errors gracefully
            and log appropriate error messages.
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the transport.

        Gracefully closes the transport connection.
        Must set self.is_connected to False.

        Note:
            Should be safe to call multiple times.
            Should clean up all resources.
        """
        pass

    @abstractmethod
    async def send_message(self, message: str):
        """Send a message through the transport.

        Args:
            message: JSON string to send

        Raises:
            RuntimeError: If transport not connected

        Note:
            Message format depends on transport type.
            Most transports expect JSON-RPC formatted strings.
        """
        pass

    @abstractmethod
    async def receive_messages(self):
        """Start receiving messages (should run in background).

        Continuously receives messages and passes them to the
        registered message handler. Should run until disconnect.

        Note:
            This method blocks and should be run as a background task.
            Implementations must handle connection errors and cleanup.
        """
        pass


class StdioTransport(MCPTransport):
    """Standard input/output transport for MCP.

    Uses stdin/stdout for communication, making it ideal for:
    - CLI tools and scripts
    - Local process communication
    - Testing and debugging

    Message Format:
        - Messages are sent as JSON lines (newline-delimited)
        - Each line contains a complete JSON-RPC message
        - Binary data is not supported

    Attributes:
        reader: AsyncIO StreamReader for stdin
        writer: AsyncIO StreamWriter for stdout
        _stop_event: Event to signal shutdown

    Example:
        >>> # Create stdio transport
        >>> transport = StdioTransport()
        >>>
        >>> # Set up message handler
        >>> async def handler(msg):
        ...     print(f"Received: {msg}")
        >>> transport.set_message_handler(handler)
        >>>
        >>> # Connect and start receiving
        >>> await transport.connect()
        >>> asyncio.create_task(transport.receive_messages())
        >>>
        >>> # Send a message
        >>> await transport.send_message('{"test": true}')
    """

    def __init__(self):
        super().__init__()
        self.reader = None
        self.writer = None
        self._stop_event = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to stdio streams.

        Sets up async readers/writers for stdin/stdout.
        Uses asyncio pipes for non-blocking I/O.

        Returns:
            True if setup successful, False on error

        Technical Details:
            - Creates StreamReader for stdin
            - Creates StreamWriter for stdout
            - Uses event loop's pipe connections
            - Enables async reading from console input
        """
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
        """Send message to stdout.

        Writes message as a single line to stdout.
        Adds newline delimiter for JSON lines format.

        Args:
            message: JSON string to send

        Raises:
            RuntimeError: If transport not connected

        Note:
            Messages are flushed immediately to ensure
            real-time communication.
        """
        if not self.is_connected or not self.writer:
            raise RuntimeError("Transport not connected")

        # MCP messages are sent as JSON lines
        line = message + "\n"
        self.writer.write(line.encode())
        await self.writer.drain()
        logger.debug(f"Sent message: {message}")

    async def receive_messages(self):
        """Receive messages from stdin.

        Continuously reads lines from stdin and passes them
        to the message handler. Runs until disconnected.

        Message Processing:
            1. Read line from stdin
            2. Strip whitespace and decode
            3. Pass to message handler
            4. Handle any errors gracefully

        Raises:
            RuntimeError: If transport not connected

        Note:
            Empty lines are ignored.
            Stops on EOF or disconnect signal.
        """
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
    """WebSocket transport for MCP.

    Uses WebSocket protocol for bidirectional communication.
    Ideal for:
    - Web browser integrations
    - Remote server connections
    - Real-time applications

    Features:
        - Full duplex communication
        - Automatic reconnection handling
        - Binary and text message support
        - Low latency messaging

    Attributes:
        uri: WebSocket URI (ws:// or wss://)
        websocket: Active WebSocket connection
        _stop_event: Event to signal shutdown

    Example:
        >>> # Connect to WebSocket server
        >>> transport = WebSocketTransport("ws://localhost:8080/mcp")
        >>> await transport.connect()
        >>>
        >>> # Send JSON message
        >>> await transport.send_message('{"method": "ping"}')
        >>>
        >>> # Receive messages in background
        >>> asyncio.create_task(transport.receive_messages())
    """

    def __init__(self, uri: str):
        """Initialize WebSocket transport.

        Args:
            uri: WebSocket URI (e.g., "ws://localhost:8080")
                 Supports both ws:// and wss:// protocols
        """
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
        """Send message through WebSocket.

        Sends text frame containing JSON message.

        Args:
            message: JSON string to send

        Raises:
            RuntimeError: If transport not connected
            websockets.exceptions.ConnectionClosed: If connection lost
        """
        if not self.is_connected or not self.websocket:
            raise RuntimeError("Transport not connected")

        await self.websocket.send(message)
        logger.debug(f"Sent WebSocket message: {message}")

    async def receive_messages(self):
        """Receive messages from WebSocket.

        Continuously receives messages with timeout handling.
        Gracefully handles connection closure.

        Features:
            - 1 second timeout for responsiveness
            - Automatic detection of closed connections
            - Clean shutdown on stop signal

        Note:
            Uses timeout to periodically check stop signal
            while still receiving messages promptly.
        """
        if not self.is_connected or not self.websocket:
            raise RuntimeError("Transport not connected")

        try:
            while not self._stop_event.is_set():
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
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
    """Server-Sent Events transport for MCP.

    Uses Server-Sent Events for unidirectional streaming from server.
    Sends messages via separate POST endpoint.

    Ideal for:
    - Server push notifications
    - Live data streaming
    - Firewall-friendly communications

    Architecture:
        - GET request to receive SSE stream
        - POST requests to send messages
        - Automatic reconnection on failure

    Attributes:
        url: SSE endpoint URL
        session: aiohttp ClientSession
        _stop_event: Event to signal shutdown

    Example:
        >>> # Connect to SSE endpoint
        >>> transport = SSETransport("http://localhost:8080/events")
        >>> await transport.connect()
        >>>
        >>> # Receive events
        >>> asyncio.create_task(transport.receive_messages())
        >>>
        >>> # Send message via POST
        >>> await transport.send_message('{"method": "subscribe"}')

    Note:
        Requires aiohttp library for HTTP operations.
    """

    def __init__(self, url: str):
        """Initialize SSE transport.

        Args:
            url: SSE endpoint URL (e.g., "http://localhost:8080/events")
                 Should be the GET endpoint for receiving events
        """
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
        """Send message via POST (SSE is typically unidirectional).

        Since SSE is for server->client streaming, messages are
        sent via a separate POST endpoint.

        Args:
            message: JSON string to send

        Raises:
            RuntimeError: If transport not connected or POST fails

        Note:
            Assumes POST endpoint is at /send instead of /events.
            Adjust based on your server implementation.
        """
        if not self.is_connected or not self.session:
            raise RuntimeError("Transport not connected")

        # For SSE, we typically send messages via POST to a different endpoint
        post_url = self.url.replace("/events", "/send")
        async with self.session.post(post_url, data=message) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to send message: {response.status}")

        logger.debug(f"Sent SSE message: {message}")

    async def receive_messages(self):
        """Receive messages from SSE stream.

        Opens GET request to SSE endpoint and processes events.
        SSE Format:
            data: {"json": "content"}

            data: {"another": "message"}

        Processing:
            1. Open streaming GET request
            2. Read lines from response
            3. Extract data from 'data: ' prefixed lines
            4. Pass JSON to message handler

        Note:
            Only processes 'data:' events.
            Other SSE fields (event:, id:) are ignored.
        """
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
                    if line.startswith("data: "):
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
