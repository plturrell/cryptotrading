"""
A2A Transport Implementation
Implements networking layer for the cryptotrading.com A2A protocol
"""
import asyncio
import hashlib
import json
import logging
import socket
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from .a2a_protocol import A2AMessage, AgentStatus, MessageType

try:
    from ...core.di_container import resolve
    from ...core.interfaces import ILogger, IMetricsCollector, IServiceRegistry
except ImportError:
    # Fallback for missing interfaces
    ILogger = None
    IMetricsCollector = None
    IServiceRegistry = None

    def resolve(service_type):
        return None


from ...blockchain.anvil_client import AnvilA2AClient
from .service_discovery import ServiceDiscovery


class A2ATransportManager:
    """
    A2A Transport Manager
    Handles networking for the cryptotrading.com A2A protocol
    """

    def __init__(
        self,
        agent_id: str,
        host: str = "localhost",
        port: int = None,
        use_blockchain: bool = True,
        anvil_url: str = "http://localhost:8545",
    ):
        self.agent_id = agent_id
        self.host = host
        self.port = port or self._find_available_port()
        self.use_blockchain = use_blockchain

        # Network components
        self.server: Optional[asyncio.Server] = None
        self.connections: Dict[str, asyncio.StreamWriter] = {}  # agent_id -> writer
        self.agent_registry: Dict[str, Dict[str, Any]] = {}  # agent_id -> endpoint info

        # Blockchain components
        self.blockchain_client: Optional[AnvilA2AClient] = None
        if use_blockchain:
            self.blockchain_client = AnvilA2AClient(anvil_url)

        # Service discovery
        self.service_discovery = ServiceDiscovery(agent_id, self.blockchain_client)

        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_messages: Dict[str, A2AMessage] = {}
        self.message_queue = asyncio.Queue()

        # Agent status
        self.status = AgentStatus.INACTIVE
        self.capabilities: Set[str] = set()
        self.last_heartbeat = datetime.utcnow()

        # Performance tracking
        self.messages_sent = 0
        self.messages_received = 0
        self.connection_count = 0

        # DI services
        self.logger: Optional[ILogger] = None
        self.metrics: Optional[IMetricsCollector] = None
        self.service_registry: Optional[IServiceRegistry] = None

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown = False

        self._setup_logger()

    def _setup_logger(self):
        """Setup logging"""
        self.local_logger = logging.getLogger(f"A2ATransport-{self.agent_id}")

    def _find_available_port(self) -> int:
        """Find an available port for the A2A server"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    async def initialize(self) -> bool:
        """Initialize the A2A transport"""
        try:
            self.local_logger.info(f"Initializing A2A transport for {self.agent_id}")

            # Resolve DI services
            await self._resolve_dependencies()

            # Initialize blockchain client if enabled
            if self.blockchain_client:
                blockchain_success = await self.blockchain_client.initialize()
                if blockchain_success:
                    # Register agent on blockchain
                    await self.blockchain_client.register_agent(
                        self.agent_id, list(self.capabilities), f"{self.host}:{self.port}"
                    )

                    # Setup blockchain message handlers
                    self.blockchain_client.register_message_handler(
                        "heartbeat", self._handle_blockchain_heartbeat
                    )
                    self.blockchain_client.register_message_handler(
                        "workflow_request", self._handle_blockchain_workflow_request
                    )

                    # Start blockchain listener
                    await self.blockchain_client.start_listening()

                    self.local_logger.info("Blockchain A2A integration initialized")
                else:
                    self.local_logger.warning(
                        "Blockchain initialization failed, falling back to TCP only"
                    )

            # Start TCP server (as backup or primary if no blockchain)
            await self._start_server()

            # Start service discovery
            await self.service_discovery.start()

            # Register with service discovery
            await self._register_service()

            # Setup default handlers
            self._setup_default_handlers()

            # Start background tasks
            self._start_background_tasks()

            self.status = AgentStatus.ACTIVE

            if self.logger:
                self.logger.info(f"A2A transport initialized on {self.host}:{self.port}")

            return True

        except Exception as e:
            self.local_logger.error(f"Failed to initialize A2A transport: {e}", exc_info=True)
            return False

    async def _resolve_dependencies(self):
        """Resolve DI services"""
        try:
            self.logger = await resolve(ILogger)
        except Exception:
            pass

        try:
            self.metrics = await resolve(IMetricsCollector)
        except Exception:
            pass

        try:
            self.service_registry = await resolve(IServiceRegistry)
        except Exception:
            pass

    async def _start_server(self):
        """Start TCP server for A2A communication"""
        self.server = await asyncio.start_server(self._handle_connection, self.host, self.port)

        self.local_logger.info(f"A2A server started on {self.host}:{self.port}")

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming A2A connection"""
        peer_addr = writer.get_extra_info("peername")
        connection_id = f"{peer_addr[0]}:{peer_addr[1]}"

        self.local_logger.debug(f"New A2A connection from {connection_id}")
        self.connection_count += 1

        try:
            # Handle A2A handshake
            agent_id = await self._handle_handshake(reader, writer)

            if agent_id:
                self.connections[agent_id] = writer
                self.local_logger.info(f"Established A2A connection with agent {agent_id}")

                # Process messages from this agent
                await self._process_connection_messages(agent_id, reader, writer)

        except Exception as e:
            self.local_logger.error(f"A2A connection error: {e}")

        finally:
            writer.close()
            await writer.wait_closed()
            self.connection_count -= 1

            # Remove from connections
            agent_to_remove = None
            for aid, conn_writer in self.connections.items():
                if conn_writer == writer:
                    agent_to_remove = aid
                    break

            if agent_to_remove:
                del self.connections[agent_to_remove]
                self.local_logger.info(f"A2A connection closed for {agent_to_remove}")

    async def _handle_handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> Optional[str]:
        """Handle A2A protocol handshake"""
        try:
            # Read handshake message
            message_data = await self._read_message(reader)
            handshake = json.loads(message_data.decode("utf-8"))

            if handshake.get("type") == "a2a_handshake":
                remote_agent_id = handshake.get("agent_id")
                remote_capabilities = handshake.get("capabilities", [])
                protocol_version = handshake.get("protocol_version", "1.0")

                # Store agent info
                self.agent_registry[remote_agent_id] = {
                    "agent_id": remote_agent_id,
                    "capabilities": remote_capabilities,
                    "protocol_version": protocol_version,
                    "connected_at": datetime.utcnow().isoformat(),
                    "last_seen": datetime.utcnow().isoformat(),
                }

                # Send handshake response
                handshake_response = {
                    "type": "a2a_handshake_ack",
                    "agent_id": self.agent_id,
                    "capabilities": list(self.capabilities),
                    "protocol_version": "1.0",
                    "status": self.status.value,
                }

                await self._send_raw_message(writer, json.dumps(handshake_response).encode("utf-8"))
                return remote_agent_id

            return None

        except Exception as e:
            self.local_logger.error(f"Handshake error: {e}")
            return None

    async def _process_connection_messages(
        self, agent_id: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Process messages from connected agent"""
        while not self._shutdown:
            try:
                message_data = await self._read_message(reader)
                if not message_data:
                    break

                # Parse A2A message
                message_dict = json.loads(message_data.decode("utf-8"))

                # Handle keep-alive messages
                if message_dict.get("type") == "keepalive":
                    await self._send_raw_message(
                        writer, json.dumps({"type": "keepalive_ack"}).encode("utf-8")
                    )
                    continue

                # Convert to A2A message
                message = A2AMessage(**message_dict)

                # Update agent registry
                if agent_id in self.agent_registry:
                    self.agent_registry[agent_id]["last_seen"] = datetime.utcnow().isoformat()

                # Process message
                await self._process_message(message, writer)

                self.messages_received += 1
                if self.metrics:
                    await self.metrics.increment_counter("a2a_messages_received")

            except asyncio.IncompleteReadError:
                break
            except Exception as e:
                self.local_logger.error(f"Message processing error: {e}")
                break

    async def _read_message(self, reader: asyncio.StreamReader) -> bytes:
        """Read message with length prefix"""
        # Read message length (4 bytes)
        length_data = await reader.read(4)
        if not length_data:
            return b""

        message_length = int.from_bytes(length_data, byteorder="big")

        # Read message data
        message_data = await reader.read(message_length)
        return message_data

    async def _send_raw_message(self, writer: asyncio.StreamWriter, message_data: bytes):
        """Send raw message with length prefix"""
        message_length = len(message_data)

        # Send length prefix + message
        writer.write(message_length.to_bytes(4, byteorder="big"))
        writer.write(message_data)
        await writer.drain()

    async def _process_message(
        self, message: A2AMessage, response_writer: asyncio.StreamWriter = None
    ):
        """Process incoming A2A message"""
        try:
            # Route to appropriate handler
            if message.message_type in self.message_handlers:
                handler = self.message_handlers[message.message_type]
                response = await handler(message)

                # Send response if handler returns one
                if response and response_writer:
                    response_message = A2AMessage(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        message_type=self._get_response_type(message.message_type),
                        payload=response,
                        message_id=str(uuid.uuid4()),
                        timestamp=datetime.utcnow().isoformat(),
                        correlation_id=message.message_id,
                    )

                    response_data = json.dumps(asdict(response_message)).encode("utf-8")
                    await self._send_raw_message(response_writer, response_data)
            else:
                self.local_logger.warning(f"No handler for message type: {message.message_type}")

        except Exception as e:
            self.local_logger.error(f"Message processing failed: {e}")

    def _get_response_type(self, request_type: MessageType) -> MessageType:
        """Get corresponding response type for request"""
        response_map = {
            MessageType.DATA_LOAD_REQUEST: MessageType.DATA_LOAD_RESPONSE,
            MessageType.ANALYSIS_REQUEST: MessageType.ANALYSIS_RESPONSE,
            MessageType.DATA_QUERY: MessageType.DATA_QUERY_RESPONSE,
            MessageType.TRADE_EXECUTION: MessageType.TRADE_RESPONSE,
            MessageType.WORKFLOW_REQUEST: MessageType.WORKFLOW_RESPONSE,
            MessageType.MEMORY_REQUEST: MessageType.MEMORY_RESPONSE,
        }
        return response_map.get(request_type, MessageType.ERROR)

    async def _register_service(self):
        """Register this agent with service discovery"""
        try:
            # Register with our service discovery system
            await self.service_discovery.register_service(
                self.agent_id,
                {
                    "agent_id": self.agent_id,
                    "host": self.host,
                    "port": self.port,
                    "protocol": "a2a",
                    "capabilities": list(self.capabilities),
                    "metadata": {
                        "protocol_version": "1.0",
                        "status": self.status.value,
                        "blockchain_enabled": self.blockchain_client is not None,
                    },
                },
            )

            # Also register with DI service registry if available
            if self.service_registry:
                try:
                    await self.service_registry.register_service(
                        self.agent_id,
                        {
                            "agent_id": self.agent_id,
                            "host": self.host,
                            "port": self.port,
                            "protocol": "a2a",
                            "status": self.status.value,
                            "capabilities": list(self.capabilities),
                            "protocol_version": "1.0",
                        },
                    )
                except Exception as e:
                    self.local_logger.error(f"DI service registration failed: {e}")

        except Exception as e:
            self.local_logger.error(f"Service registration failed: {e}")

    def _setup_default_handlers(self):
        """Setup default A2A message handlers"""
        self.message_handlers.update(
            {
                MessageType.HEARTBEAT: self._handle_heartbeat,
                MessageType.WORKFLOW_STATUS: self._handle_workflow_status,
                MessageType.MEMORY_SHARE: self._handle_memory_share,
                MessageType.ERROR: self._handle_error,
            }
        )

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        tasks = [self._heartbeat_task(), self._cleanup_task(), self._discovery_task()]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    # Public API Methods
    async def send_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 0,
        correlation_id: str = None,
        workflow_context: Dict[str, Any] = None,
    ) -> str:
        """Send A2A message to another agent"""
        message_id = str(uuid.uuid4())

        message = A2AMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            message_id=message_id,
            timestamp=datetime.utcnow().isoformat(),
            priority=priority,
            correlation_id=correlation_id,
            workflow_context=workflow_context,
        )

        # Try blockchain first if available, fallback to TCP
        success = False

        if self.blockchain_client:
            try:
                tx_hash = await self.blockchain_client.send_message(
                    receiver_id, message_type.value, payload, message_id
                )
                if tx_hash:
                    success = True
                    self.local_logger.debug(f"Sent message via blockchain: {tx_hash}")
            except Exception as e:
                self.local_logger.error(f"Blockchain send failed, trying TCP: {e}")

        # Fallback to TCP if blockchain failed or not available
        if not success:
            success = await self._deliver_message(message)

        if success:
            self.messages_sent += 1
            if self.metrics:
                await self.metrics.increment_counter("a2a_messages_sent")

        return message_id

    async def _deliver_message(self, message: A2AMessage) -> bool:
        """Deliver message to target agent"""
        receiver_id = message.receiver_id

        # Check if we have direct connection
        if receiver_id in self.connections:
            try:
                writer = self.connections[receiver_id]
                message_data = json.dumps(asdict(message)).encode("utf-8")
                await self._send_raw_message(writer, message_data)
                return True
            except Exception as e:
                self.local_logger.error(f"Direct delivery failed to {receiver_id}: {e}")
                # Remove broken connection
                del self.connections[receiver_id]

        # Try to establish new connection
        if receiver_id in self.agent_registry:
            agent_info = self.agent_registry[receiver_id]
            return await self._connect_and_send(agent_info, message)

        # Try our service discovery
        try:
            service_endpoint = await self.service_discovery.get_service(receiver_id)
            if service_endpoint:
                if service_endpoint.protocol == "blockchain":
                    # Send via blockchain
                    if self.blockchain_client:
                        tx_hash = await self.blockchain_client.send_message(
                            receiver_id,
                            message.message_type.value,
                            message.payload,
                            message.message_id,
                        )
                        return tx_hash is not None
                else:
                    # Send via TCP
                    service_info = {
                        "host": service_endpoint.host,
                        "port": service_endpoint.port,
                        "agent_id": service_endpoint.agent_id,
                    }
                    return await self._connect_and_send(service_info, message)
        except Exception as e:
            self.local_logger.error(f"Service discovery failed: {e}")

        # Try legacy service registry
        if self.service_registry:
            try:
                services = await self.service_registry.list_services()
                for service in services:
                    if service.get("agent_id") == receiver_id:
                        return await self._connect_and_send(service, message)
            except Exception as e:
                self.local_logger.error(f"Legacy service discovery failed: {e}")

        self.local_logger.error(f"Cannot deliver message to {receiver_id}: agent not found")
        return False

    async def _connect_and_send(self, agent_info: Dict[str, Any], message: A2AMessage) -> bool:
        """Connect to agent and send message"""
        try:
            host = agent_info.get("host", "localhost")
            port = agent_info.get("port")

            if not port:
                return False

            # Establish connection
            reader, writer = await asyncio.open_connection(host, port)

            # Send handshake
            handshake = {
                "type": "a2a_handshake",
                "agent_id": self.agent_id,
                "capabilities": list(self.capabilities),
                "protocol_version": "1.0",
            }

            await self._send_raw_message(writer, json.dumps(handshake).encode("utf-8"))

            # Read handshake response
            response_data = await self._read_message(reader)
            response = json.loads(response_data.decode("utf-8"))

            if response.get("type") == "a2a_handshake_ack":
                # Send message
                message_data = json.dumps(asdict(message)).encode("utf-8")
                await self._send_raw_message(writer, message_data)

                # Store connection for future use
                self.connections[message.receiver_id] = writer

                self.local_logger.info(f"Successfully delivered message to {message.receiver_id}")
                return True

            writer.close()
            await writer.wait_closed()
            return False

        except Exception as e:
            self.local_logger.error(f"Connection failed to {agent_info}: {e}")
            return False

    async def connect_to_agent(self, host: str, port: int) -> Optional[str]:
        """Manually connect to an agent"""
        try:
            reader, writer = await asyncio.open_connection(host, port)

            # Send handshake
            handshake = {
                "type": "a2a_handshake",
                "agent_id": self.agent_id,
                "capabilities": list(self.capabilities),
                "protocol_version": "1.0",
            }

            await self._send_raw_message(writer, json.dumps(handshake).encode("utf-8"))

            # Read response
            response_data = await self._read_message(reader)
            response = json.loads(response_data.decode("utf-8"))

            if response.get("type") == "a2a_handshake_ack":
                remote_agent_id = response.get("agent_id")

                # Store connection and agent info
                self.connections[remote_agent_id] = writer
                self.agent_registry[remote_agent_id] = {
                    "agent_id": remote_agent_id,
                    "host": host,
                    "port": port,
                    "capabilities": response.get("capabilities", []),
                    "protocol_version": response.get("protocol_version", "1.0"),
                    "connected_at": datetime.utcnow().isoformat(),
                }

                self.local_logger.info(f"Connected to agent {remote_agent_id} at {host}:{port}")

                # Start processing messages from this connection
                asyncio.create_task(
                    self._process_connection_messages(remote_agent_id, reader, writer)
                )

                return remote_agent_id

            writer.close()
            await writer.wait_closed()
            return None

        except Exception as e:
            self.local_logger.error(f"Failed to connect to {host}:{port}: {e}")
            return None

    # Message Handlers
    async def _handle_heartbeat(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle heartbeat message"""
        return {
            "status": self.status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
        }

    async def _handle_workflow_status(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle workflow status request"""
        # Get actual workflow status if workflow engine is available
        active_workflows = []

        try:
            # Check if we have a workflow engine via DI
            if hasattr(self, "workflow_engine") and self.workflow_engine:
                active_workflows = await self.workflow_engine.get_active_workflows()
            else:
                self.logger.debug("No workflow engine available for status query")
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")

        return {
            "agent_id": self.agent_id,
            "active_workflows": active_workflows,
            "status": self.status.value,
            "workflow_engine_available": hasattr(self, "workflow_engine")
            and self.workflow_engine is not None,
        }

    async def _handle_memory_share(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle memory sharing request"""
        memory_shared = False
        shared_memory_id = None
        error = None

        try:
            # Check if we have a memory manager via DI
            if hasattr(self, "memory_manager") and self.memory_manager:
                # Extract memory request details
                memory_type = message.payload.get("memory_type", "general")
                memory_keys = message.payload.get("keys", [])

                # Share the requested memory
                shared_data = await self.memory_manager.get_memory(memory_type, memory_keys)
                if shared_data:
                    # Store in shared memory space
                    shared_memory_id = f"shared_{message.message_id}_{int(time.time())}"
                    await self.memory_manager.store_shared_memory(shared_memory_id, shared_data)
                    memory_shared = True
            else:
                error = "No memory manager available"
                self.logger.warning("Memory sharing requested but no memory manager available")
        except Exception as e:
            error = str(e)
            self.logger.error(f"Failed to share memory: {e}")

        return {
            "agent_id": self.agent_id,
            "memory_shared": memory_shared,
            "shared_memory_id": shared_memory_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _handle_error(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle error message"""
        self.local_logger.error(f"Received error from {message.sender_id}: {message.payload}")
        return {"error_acknowledged": True, "timestamp": datetime.utcnow().isoformat()}

    # Background Tasks
    async def _heartbeat_task(self):
        """Send periodic heartbeats"""
        while not self._shutdown:
            try:
                # Send heartbeat to all connected agents
                for agent_id in list(self.connections.keys()):
                    await self.send_message(
                        receiver_id=agent_id,
                        message_type=MessageType.HEARTBEAT,
                        payload={"timestamp": datetime.utcnow().isoformat()},
                    )

                self.last_heartbeat = datetime.utcnow()
                await asyncio.sleep(30)  # Heartbeat every 30 seconds

            except Exception as e:
                self.local_logger.error(f"Heartbeat task error: {e}")
                await asyncio.sleep(5)

    async def _cleanup_task(self):
        """Clean up stale connections and data"""
        while not self._shutdown:
            try:
                now = datetime.utcnow()

                # Clean up stale agent registry entries
                stale_agents = []
                for agent_id, info in self.agent_registry.items():
                    last_seen = datetime.fromisoformat(info["last_seen"])
                    if (now - last_seen).total_seconds() > 300:  # 5 minutes
                        stale_agents.append(agent_id)

                for agent_id in stale_agents:
                    del self.agent_registry[agent_id]
                    if agent_id in self.connections:
                        try:
                            self.connections[agent_id].close()
                            await self.connections[agent_id].wait_closed()
                        except (ConnectionError, OSError) as e:
                            self.local_logger.debug(
                                f"Connection already closed for {agent_id}: {e}"
                            )
                        del self.connections[agent_id]

                    self.local_logger.info(f"Cleaned up stale agent: {agent_id}")

                await asyncio.sleep(120)  # Cleanup every 2 minutes

            except Exception as e:
                self.local_logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(30)

    async def _discovery_task(self):
        """Discover new agents periodically"""
        while not self._shutdown:
            try:
                if self.service_registry:
                    services = await self.service_registry.list_services()

                    for service in services:
                        agent_id = service.get("agent_id")
                        if (
                            agent_id
                            and agent_id != self.agent_id
                            and agent_id not in self.agent_registry
                        ):
                            # Try to connect to new agent
                            host = service.get("host")
                            port = service.get("port")

                            if host and port:
                                connected_agent = await self.connect_to_agent(host, port)
                                if connected_agent:
                                    self.local_logger.info(
                                        f"Discovered and connected to agent: {connected_agent}"
                                    )

                await asyncio.sleep(180)  # Discovery every 3 minutes

            except Exception as e:
                self.local_logger.error(f"Discovery task error: {e}")
                await asyncio.sleep(60)

    # Management Methods
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler

    def add_capability(self, capability: str):
        """Add agent capability"""
        self.capabilities.add(capability)

    def get_connected_agents(self) -> List[str]:
        """Get list of connected agents"""
        return list(self.connections.keys())

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent"""
        return self.agent_registry.get(agent_id)

    def get_status(self) -> Dict[str, Any]:
        """Get transport status"""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "host": self.host,
            "port": self.port,
            "connected_agents": len(self.connections),
            "known_agents": len(self.agent_registry),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "capabilities": list(self.capabilities),
            "last_heartbeat": self.last_heartbeat.isoformat(),
        }

    async def shutdown(self):
        """Shutdown the A2A transport"""
        self.local_logger.info("Shutting down A2A transport")
        self._shutdown = True
        self.status = AgentStatus.INACTIVE

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close all connections
        for writer in self.connections.values():
            writer.close()
            await writer.wait_closed()

        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Unregister from service discovery
        if self.service_registry:
            try:
                await self.service_registry.unregister_service(self.agent_id)
            except Exception as e:
                self.local_logger.warning(f"Failed to unregister service: {e}")

        self.local_logger.info("A2A transport shutdown complete")

    # Blockchain Message Handlers
    async def _handle_blockchain_heartbeat(self, message):
        """Handle heartbeat message from blockchain"""
        from ...blockchain.anvil_client import A2AMessage as BlockchainMessage

        if isinstance(message, BlockchainMessage):
            self.local_logger.debug(f"Received blockchain heartbeat from {message.sender_id}")

            # Update agent registry
            if message.sender_id not in self.agent_registry:
                self.agent_registry[message.sender_id] = {
                    "agent_id": message.sender_id,
                    "last_seen": datetime.utcnow().isoformat(),
                    "via_blockchain": True,
                }
            else:
                self.agent_registry[message.sender_id]["last_seen"] = datetime.utcnow().isoformat()

    async def _handle_blockchain_workflow_request(self, message):
        """Handle workflow request from blockchain"""
        from ...blockchain.anvil_client import A2AMessage as BlockchainMessage

        if isinstance(message, BlockchainMessage):
            self.local_logger.info(f"Received blockchain workflow request from {message.sender_id}")

            # Convert blockchain message to A2A protocol message
            a2a_message = A2AMessage(
                sender_id=message.sender_id,
                receiver_id=message.recipient_id,
                message_type=MessageType.WORKFLOW_REQUEST,
                payload=message.payload,
                message_id=message.message_id,
                timestamp=message.timestamp,
                protocol_version="1.0",
            )

            # Process through normal A2A handler
            if MessageType.WORKFLOW_REQUEST in self.message_handlers:
                try:
                    response = await self.message_handlers[MessageType.WORKFLOW_REQUEST](
                        a2a_message
                    )

                    # Send response back via blockchain
                    if response and self.blockchain_client:
                        await self.blockchain_client.send_message(
                            message.sender_id,
                            "workflow_response",
                            response,
                            f"response_{message.message_id}",
                        )
                except Exception as e:
                    self.local_logger.error(f"Error processing blockchain workflow request: {e}")

    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get blockchain integration status"""
        if not self.blockchain_client:
            return {"enabled": False}

        return {"enabled": True, "status": self.blockchain_client.get_status()}
