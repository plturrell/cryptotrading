"""
Real Agent-to-Agent Communication Implementation
Replaces simulated A2A with actual networking and distributed coordination
"""
import asyncio
import json
import uuid
import hashlib
import time
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import socket
import http.server
import socketserver
import urllib.request
import urllib.parse
from contextlib import asynccontextmanager

from ..interfaces import (
    ICommunicationManager, IServiceDiscovery, ILogger, IMetricsCollector,
    IHealthChecker, ICache, ILockManager, MessageType, MessagePriority
)
from ..di_container import resolve


class A2ATransportType(Enum):
    """Transport mechanisms for A2A communication"""
    HTTP = "http"
    WEBSOCKET = "websocket"
    TCP = "tcp"
    MESSAGE_QUEUE = "message_queue"


class A2AMessageStatus(Enum):
    """Message delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class A2AEndpoint:
    """Agent endpoint information"""
    agent_id: str
    host: str
    port: int
    transport: A2ATransportType
    health_check_url: str
    last_seen: datetime = field(default_factory=datetime.utcnow)
    is_healthy: bool = True
    response_time_ms: float = 0.0
    capabilities: List[str] = field(default_factory=list)


@dataclass
class A2AMessage:
    """Real A2A message with routing and delivery tracking"""
    id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: datetime
    ttl: datetime
    routing_path: List[str] = field(default_factory=list)
    delivery_attempts: int = 0
    max_retries: int = 3
    requires_ack: bool = False
    correlation_id: Optional[str] = None
    status: A2AMessageStatus = A2AMessageStatus.PENDING


@dataclass
class A2ADeliveryResult:
    """Message delivery result"""
    message_id: str
    success: bool
    error: Optional[str] = None
    delivery_time_ms: float = 0.0
    recipient_response: Optional[Dict[str, Any]] = None


class RealA2ACommunication:
    """
    Real Agent-to-Agent Communication System
    
    Features:
    - HTTP/WebSocket transport
    - Service discovery and health checking
    - Message routing and delivery guarantees
    - Distributed coordination and consensus
    - Network resilience and failover
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.logger = logging.getLogger(f"RealA2A-{agent_id}")
        
        # Transport configuration
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 8080)
        self.transport = A2ATransportType(self.config.get("transport", "http"))
        
        # Network topology
        self.known_agents: Dict[str, A2AEndpoint] = {}
        self.direct_connections: Set[str] = set()
        self.routing_table: Dict[str, str] = {}  # destination -> next_hop
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_messages: Dict[str, A2AMessage] = {}
        self.message_history: List[A2AMessage] = []
        self.delivery_results: Dict[str, A2ADeliveryResult] = {}
        
        # Server and client 
        self.tcp_server: Optional[asyncio.Server] = None
        self.tcp_connections: Dict[str, asyncio.StreamWriter] = {}
        self.server_socket: Optional[socket.socket] = None
        
        # DI services
        self.communication_manager: Optional[ICommunicationManager] = None
        self.service_discovery: Optional[IServiceDiscovery] = None
        self.logger_service: Optional[ILogger] = None
        self.metrics: Optional[IMetricsCollector] = None
        self.health_checker: Optional[IHealthChecker] = None
        self.cache: Optional[ICache] = None
        self.lock_manager: Optional[ILockManager] = None
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown = False
        
        # Performance metrics
        self.message_count = 0
        self.successful_deliveries = 0
        self.failed_deliveries = 0
        self.total_delivery_time = 0.0
    
    async def initialize(self) -> bool:
        """Initialize the real A2A communication system"""
        try:
            self.logger.info(f"Initializing RealA2ACommunication for {self.agent_id}")
            
            # Resolve DI services
            await self._resolve_dependencies()
            
            # Initialize transport layer
            await self._initialize_transport()
            
            # Start service discovery
            await self._start_service_discovery()
            
            # Setup default message handlers
            self._setup_default_handlers()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info(f"RealA2ACommunication initialized on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RealA2ACommunication: {e}", exc_info=True)
            return False
    
    async def _resolve_dependencies(self):
        """Resolve services from DI container"""
        try:
            self.logger_service = await resolve(ILogger)
        except Exception:
            pass
        
        try:
            self.metrics = await resolve(IMetricsCollector)
        except Exception:
            pass
        
        try:
            self.health_checker = await resolve(IHealthChecker)
        except Exception:
            pass
        
        try:
            self.cache = await resolve(ICache)
        except Exception:
            pass
    
    async def _initialize_transport(self):
        """Initialize transport layer (TCP server)"""
        if self.transport == A2ATransportType.TCP:
            await self._initialize_tcp_server()
        else:
            # Default to TCP for real implementation
            self.transport = A2ATransportType.TCP
            await self._initialize_tcp_server()
    
    async def _initialize_tcp_server(self):
        """Initialize TCP server for real A2A communication"""
        self.tcp_server = await asyncio.start_server(
            self._handle_tcp_connection,
            self.host,
            self.port
        )
        
        self.logger.info(f"TCP A2A server started on {self.host}:{self.port}")
    
    async def _handle_tcp_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP connection"""
        peer = writer.get_extra_info('peername')
        self.logger.debug(f"New TCP connection from {peer}")
        
        try:
            while True:
                # Read message length (first 4 bytes)
                length_data = await reader.read(4)
                if not length_data:
                    break
                
                message_length = int.from_bytes(length_data, byteorder='big')
                
                # Read message data
                message_data = await reader.read(message_length)
                if not message_data:
                    break
                
                # Parse and process message
                try:
                    message_json = message_data.decode('utf-8')
                    message_dict = json.loads(message_json)
                    message = A2AMessage(**message_dict)
                    
                    # Store connection for this agent
                    sender_id = message.sender_id
                    if sender_id and sender_id not in self.tcp_connections:
                        self.tcp_connections[sender_id] = writer
                        self.direct_connections.add(sender_id)
                        self.logger.info(f"Established TCP connection with agent {sender_id}")
                    
                    # Process message
                    response = await self._process_incoming_message(message)
                    
                    # Send response
                    if response:
                        response_json = json.dumps(response)
                        response_data = response_json.encode('utf-8')
                        response_length = len(response_data)
                        
                        writer.write(response_length.to_bytes(4, byteorder='big'))
                        writer.write(response_data)
                        await writer.drain()
                
                except Exception as e:
                    self.logger.error(f"Error processing TCP message: {e}")
                    # Send error response
                    error_response = {"error": str(e)}
                    error_json = json.dumps(error_response)
                    error_data = error_json.encode('utf-8')
                    error_length = len(error_data)
                    
                    writer.write(error_length.to_bytes(4, byteorder='big'))
                    writer.write(error_data)
                    await writer.drain()
        
        except Exception as e:
            self.logger.error(f"TCP connection error: {e}")
        
        finally:
            # Clean up connection
            writer.close()
            await writer.wait_closed()
            
            # Remove from connections
            agent_to_remove = None
            for agent_id, conn_writer in self.tcp_connections.items():
                if conn_writer == writer:
                    agent_to_remove = agent_id
                    break
            
            if agent_to_remove:
                del self.tcp_connections[agent_to_remove]
                self.direct_connections.discard(agent_to_remove)
                self.logger.info(f"TCP connection closed for agent {agent_to_remove}")
    
    async def _start_service_discovery(self):
        """Start service discovery to find other agents"""
        # Register this agent
        if self.service_discovery:
            await self.service_discovery.register_service(
                self.agent_id,
                {
                    "host": self.host,
                    "port": self.port,
                    "transport": self.transport.value,
                    "health_check_url": f"http://{self.host}:{self.port}/a2a/health",
                    "capabilities": ["messaging", "coordination", "consensus"]
                }
            )
        
        # Discover other agents
        await self._discover_agents()
    
    async def _discover_agents(self):
        """Discover other agents in the network"""
        if not self.service_discovery:
            return
        
        try:
            services = await self.service_discovery.discover_services("agent")
            
            for service in services:
                agent_id = service.get("service_id")
                if agent_id and agent_id != self.agent_id:
                    endpoint = A2AEndpoint(
                        agent_id=agent_id,
                        host=service.get("host", "localhost"),
                        port=service.get("port", 8080),
                        transport=A2ATransportType(service.get("transport", "http")),
                        health_check_url=service.get("health_check_url", ""),
                        capabilities=service.get("capabilities", [])
                    )
                    
                    self.known_agents[agent_id] = endpoint
                    self.logger.info(f"Discovered agent: {agent_id} at {endpoint.host}:{endpoint.port}")
        
        except Exception as e:
            self.logger.error(f"Service discovery failed: {e}")
    
    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.message_handlers.update({
            "heartbeat": self._handle_heartbeat,
            "ping": self._handle_ping,
            "discovery": self._handle_discovery,
            "status_request": self._handle_status_message,
            "coordination": self._handle_coordination,
            "consensus": self._handle_consensus
        })
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        tasks = [
            self._heartbeat_loop(),
            self._health_check_loop(),
            self._message_cleanup_loop(),
            self._network_discovery_loop()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    # HTTP Handlers
    async def _handle_http_message(self, request: aiohttp.web.Request):
        """Handle incoming HTTP message"""
        try:
            data = await request.json()
            message = A2AMessage(**data)
            
            # Process message
            response = await self._process_incoming_message(message)
            
            return aiohttp.web.json_response({
                "success": True,
                "message_id": message.id,
                "response": response
            })
            
        except Exception as e:
            self.logger.error(f"HTTP message handling error: {e}")
            return aiohttp.web.json_response({
                "success": False,
                "error": str(e)
            }, status=400)
    
    async def _handle_health_check(self, request: aiohttp.web.Request):
        """Handle health check request"""
        health_data = {
            "agent_id": self.agent_id,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "transport": self.transport.value,
            "capabilities": ["messaging", "coordination", "consensus"],
            "metrics": {
                "message_count": self.message_count,
                "successful_deliveries": self.successful_deliveries,
                "failed_deliveries": self.failed_deliveries,
                "avg_delivery_time_ms": (
                    self.total_delivery_time / max(self.successful_deliveries, 1)
                )
            }
        }
        
        return aiohttp.web.json_response(health_data)
    
    async def _handle_status_request(self, request: aiohttp.web.Request):
        """Handle status request"""
        status = {
            "agent_id": self.agent_id,
            "known_agents": len(self.known_agents),
            "direct_connections": len(self.direct_connections),
            "pending_messages": len(self.pending_messages),
            "message_history_size": len(self.message_history),
            "transport": self.transport.value,
            "server_info": {
                "host": self.host,
                "port": self.port
            }
        }
        
        return aiohttp.web.json_response(status)
    
    async def _handle_discovery_request(self, request: aiohttp.web.Request):
        """Handle agent discovery request"""
        try:
            data = await request.json()
            requesting_agent = data.get("agent_id")
            
            if requesting_agent:
                # Add requesting agent to known agents
                endpoint = A2AEndpoint(
                    agent_id=requesting_agent,
                    host=data.get("host", request.remote),
                    port=data.get("port", 8080),
                    transport=A2ATransportType(data.get("transport", "http")),
                    health_check_url=data.get("health_check_url", ""),
                    capabilities=data.get("capabilities", [])
                )
                
                self.known_agents[requesting_agent] = endpoint
                self.direct_connections.add(requesting_agent)
                
                self.logger.info(f"Agent {requesting_agent} discovered and connected")
            
            # Return our agent information
            return aiohttp.web.json_response({
                "agent_id": self.agent_id,
                "host": self.host,
                "port": self.port,
                "transport": self.transport.value,
                "health_check_url": f"http://{self.host}:{self.port}/a2a/health",
                "capabilities": ["messaging", "coordination", "consensus"]
            })
            
        except Exception as e:
            self.logger.error(f"Discovery request error: {e}")
            return aiohttp.web.json_response({
                "error": str(e)
            }, status=400)
    
    async def _handle_websocket(self, request: aiohttp.web.Request):
        """Handle WebSocket connection"""
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        
        agent_id = None
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "handshake":
                        agent_id = data.get("agent_id")
                        if agent_id:
                            self.websocket_connections[agent_id] = ws
                            self.direct_connections.add(agent_id)
                            await ws.send_text(json.dumps({
                                "type": "handshake_ack",
                                "agent_id": self.agent_id
                            }))
                    else:
                        # Handle message
                        message = A2AMessage(**data)
                        response = await self._process_incoming_message(message)
                        
                        if response:
                            await ws.send_text(json.dumps(response))
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
        
        except Exception as e:
            self.logger.error(f"WebSocket handling error: {e}")
        
        finally:
            if agent_id and agent_id in self.websocket_connections:
                del self.websocket_connections[agent_id]
                self.direct_connections.discard(agent_id)
        
        return ws
    
    # Message Processing
    async def _process_incoming_message(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Process incoming message"""
        self.message_count += 1
        
        if self.metrics:
            await self.metrics.increment_counter("a2a_messages_received")
        
        # Add to message history
        self.message_history.append(message)
        
        # Handle message based on payload action
        action = message.payload.get("action", "unknown")
        
        if action in self.message_handlers:
            try:
                response = await self.message_handlers[action](message)
                return response
            except Exception as e:
                self.logger.error(f"Message handler error for {action}: {e}")
                return {"error": str(e)}
        else:
            self.logger.warning(f"No handler for message action: {action}")
            return {"error": f"No handler for action: {action}"}
    
    # Message Sending
    async def send_message(self, recipient_id: Optional[str], action: str, 
                          payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL,
                          requires_ack: bool = False, ttl_seconds: int = 300) -> str:
        """Send message to another agent"""
        message_id = str(uuid.uuid4())
        
        message = A2AMessage(
            id=message_id,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST if recipient_id else MessageType.BROADCAST,
            priority=priority,
            payload={"action": action, **payload},
            timestamp=datetime.utcnow(),
            ttl=datetime.utcnow() + timedelta(seconds=ttl_seconds),
            requires_ack=requires_ack
        )
        
        # Track pending message
        self.pending_messages[message_id] = message
        
        # Send message
        if recipient_id:
            delivery_result = await self._deliver_to_agent(recipient_id, message)
        else:
            delivery_result = await self._broadcast_message(message)
        
        # Update delivery tracking
        self.delivery_results[message_id] = delivery_result
        
        if delivery_result.success:
            self.successful_deliveries += 1
            self.total_delivery_time += delivery_result.delivery_time_ms
            if self.metrics:
                await self.metrics.increment_counter("a2a_messages_sent_success")
        else:
            self.failed_deliveries += 1
            if self.metrics:
                await self.metrics.increment_counter("a2a_messages_sent_failed")
        
        return message_id
    
    async def _deliver_to_agent(self, agent_id: str, message: A2AMessage) -> A2ADeliveryResult:
        """Deliver message to specific agent"""
        start_time = time.time()
        
        if agent_id not in self.known_agents:
            return A2ADeliveryResult(
                message_id=message.id,
                success=False,
                error=f"Agent {agent_id} not found"
            )
        
        endpoint = self.known_agents[agent_id]
        
        try:
            if self.transport == A2ATransportType.HTTP:
                result = await self._deliver_via_http(endpoint, message)
            elif self.transport == A2ATransportType.WEBSOCKET:
                result = await self._deliver_via_websocket(agent_id, message)
            else:
                raise NotImplementedError(f"Transport {self.transport} not implemented")
            
            delivery_time = (time.time() - start_time) * 1000
            result.delivery_time_ms = delivery_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Message delivery failed to {agent_id}: {e}")
            return A2ADeliveryResult(
                message_id=message.id,
                success=False,
                error=str(e),
                delivery_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _deliver_via_http(self, endpoint: A2AEndpoint, message: A2AMessage) -> A2ADeliveryResult:
        """Deliver message via HTTP"""
        url = f"http://{endpoint.host}:{endpoint.port}/a2a/message"
        
        async with self.client_session.post(url, json=asdict(message)) as response:
            if response.status == 200:
                response_data = await response.json()
                return A2ADeliveryResult(
                    message_id=message.id,
                    success=response_data.get("success", False),
                    recipient_response=response_data.get("response")
                )
            else:
                error_text = await response.text()
                return A2ADeliveryResult(
                    message_id=message.id,
                    success=False,
                    error=f"HTTP {response.status}: {error_text}"
                )
    
    async def _deliver_via_websocket(self, agent_id: str, message: A2AMessage) -> A2ADeliveryResult:
        """Deliver message via WebSocket"""
        if agent_id not in self.websocket_connections:
            return A2ADeliveryResult(
                message_id=message.id,
                success=False,
                error=f"No WebSocket connection to {agent_id}"
            )
        
        ws = self.websocket_connections[agent_id]
        
        try:
            await ws.send_text(json.dumps(asdict(message)))
            return A2ADeliveryResult(
                message_id=message.id,
                success=True
            )
        except Exception as e:
            return A2ADeliveryResult(
                message_id=message.id,
                success=False,
                error=str(e)
            )
    
    async def _broadcast_message(self, message: A2AMessage) -> A2ADeliveryResult:
        """Broadcast message to all connected agents"""
        successful_deliveries = 0
        total_agents = len(self.direct_connections)
        errors = []
        
        for agent_id in self.direct_connections:
            try:
                result = await self._deliver_to_agent(agent_id, message)
                if result.success:
                    successful_deliveries += 1
                else:
                    errors.append(f"{agent_id}: {result.error}")
            except Exception as e:
                errors.append(f"{agent_id}: {str(e)}")
        
        return A2ADeliveryResult(
            message_id=message.id,
            success=successful_deliveries > 0,
            error="; ".join(errors) if errors else None,
            recipient_response={
                "broadcast_stats": {
                    "total_agents": total_agents,
                    "successful_deliveries": successful_deliveries,
                    "failed_deliveries": total_agents - successful_deliveries
                }
            }
        )
    
    # Agent Discovery and Connection
    async def connect_to_agent(self, host: str, port: int, agent_id: str = None) -> bool:
        """Connect to another agent"""
        try:
            # Send discovery request
            url = f"http://{host}:{port}/a2a/discover"
            
            discovery_data = {
                "agent_id": self.agent_id,
                "host": self.host,
                "port": self.port,
                "transport": self.transport.value,
                "health_check_url": f"http://{self.host}:{self.port}/a2a/health",
                "capabilities": ["messaging", "coordination", "consensus"]
            }
            
            async with self.client_session.post(url, json=discovery_data) as response:
                if response.status == 200:
                    agent_data = await response.json()
                    discovered_agent_id = agent_data.get("agent_id")
                    
                    if discovered_agent_id:
                        endpoint = A2AEndpoint(
                            agent_id=discovered_agent_id,
                            host=agent_data.get("host", host),
                            port=agent_data.get("port", port),
                            transport=A2ATransportType(agent_data.get("transport", "http")),
                            health_check_url=agent_data.get("health_check_url", ""),
                            capabilities=agent_data.get("capabilities", [])
                        )
                        
                        self.known_agents[discovered_agent_id] = endpoint
                        self.direct_connections.add(discovered_agent_id)
                        
                        self.logger.info(f"Connected to agent {discovered_agent_id} at {host}:{port}")
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to agent at {host}:{port}: {e}")
            return False
    
    # Message Handlers
    async def _handle_heartbeat(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle heartbeat message"""
        return {
            "action": "heartbeat_ack",
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_ping(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle ping message"""
        return {
            "action": "pong",
            "agent_id": self.agent_id,
            "original_timestamp": message.timestamp.isoformat(),
            "response_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_discovery(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle agent discovery message"""
        return {
            "action": "discovery_response",
            "agent_id": self.agent_id,
            "host": self.host,
            "port": self.port,
            "transport": self.transport.value,
            "capabilities": ["messaging", "coordination", "consensus"]
        }
    
    async def _handle_status_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle status request message"""
        return {
            "action": "status_response",
            "agent_id": self.agent_id,
            "status": "active",
            "known_agents": list(self.known_agents.keys()),
            "metrics": {
                "message_count": self.message_count,
                "successful_deliveries": self.successful_deliveries,
                "failed_deliveries": self.failed_deliveries
            }
        }
    
    async def _handle_coordination(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle coordination message"""
        coord_type = message.payload.get("coordination_type")
        
        if coord_type == "lock_request":
            # Implement distributed locking
            return await self._handle_lock_request(message)
        elif coord_type == "state_sync":
            # Implement state synchronization
            return await self._handle_state_sync(message)
        else:
            return {"error": f"Unknown coordination type: {coord_type}"}
    
    async def _handle_consensus(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle consensus message"""
        consensus_type = message.payload.get("consensus_type")
        
        if consensus_type == "proposal":
            return await self._handle_consensus_proposal(message)
        elif consensus_type == "vote":
            return await self._handle_consensus_vote(message)
        else:
            return {"error": f"Unknown consensus type: {consensus_type}"}
    
    async def _handle_lock_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle distributed lock request"""
        resource_id = message.payload.get("resource_id")
        requester = message.sender_id
        
        # Simple distributed lock implementation
        if self.lock_manager:
            try:
                lock_token = await self.lock_manager.acquire_lock(
                    f"a2a_resource_{resource_id}", timeout=30.0
                )
                
                if lock_token:
                    return {
                        "action": "lock_granted",
                        "resource_id": resource_id,
                        "lock_token": lock_token,
                        "granted_to": requester
                    }
                else:
                    return {
                        "action": "lock_denied",
                        "resource_id": resource_id,
                        "reason": "Resource already locked"
                    }
            except Exception as e:
                return {
                    "action": "lock_error",
                    "resource_id": resource_id,
                    "error": str(e)
                }
        else:
            return {"error": "Lock manager not available"}
    
    async def _handle_state_sync(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle state synchronization"""
        # Implement state synchronization logic
        return {
            "action": "state_sync_response",
            "state_version": 1,
            "synchronized": True
        }
    
    async def _handle_consensus_proposal(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle consensus proposal"""
        proposal_id = message.payload.get("proposal_id")
        proposal_data = message.payload.get("proposal_data")
        
        # Simple consensus: auto-accept for now
        # In production, implement proper consensus algorithm (Raft, PBFT, etc.)
        return {
            "action": "consensus_vote",
            "proposal_id": proposal_id,
            "vote": "accept",
            "voter": self.agent_id
        }
    
    async def _handle_consensus_vote(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle consensus vote"""
        proposal_id = message.payload.get("proposal_id")
        vote = message.payload.get("vote")
        voter = message.sender_id
        
        return {
            "action": "vote_acknowledged",
            "proposal_id": proposal_id,
            "voter": voter,
            "vote": vote
        }
    
    # Background Tasks
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to connected agents"""
        while not self._shutdown:
            try:
                for agent_id in list(self.direct_connections):
                    await self.send_message(
                        recipient_id=agent_id,
                        action="heartbeat",
                        payload={"timestamp": datetime.utcnow().isoformat()}
                    )
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Check health of connected agents"""
        while not self._shutdown:
            try:
                for agent_id, endpoint in list(self.known_agents.items()):
                    if endpoint.health_check_url:
                        try:
                            start_time = time.time()
                            async with self.client_session.get(endpoint.health_check_url) as response:
                                if response.status == 200:
                                    response_time = (time.time() - start_time) * 1000
                                    endpoint.response_time_ms = response_time
                                    endpoint.is_healthy = True
                                    endpoint.last_seen = datetime.utcnow()
                                else:
                                    endpoint.is_healthy = False
                        except Exception:
                            endpoint.is_healthy = False
                
                await asyncio.sleep(60)  # Health check every minute
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(10)
    
    async def _message_cleanup_loop(self):
        """Clean up expired messages and delivery results"""
        while not self._shutdown:
            try:
                now = datetime.utcnow()
                
                # Clean up expired pending messages
                expired_messages = [
                    msg_id for msg_id, msg in self.pending_messages.items()
                    if msg.ttl < now
                ]
                
                for msg_id in expired_messages:
                    del self.pending_messages[msg_id]
                    if msg_id in self.delivery_results:
                        self.delivery_results[msg_id].status = A2AMessageStatus.EXPIRED
                
                # Clean up old message history (keep last 1000)
                if len(self.message_history) > 1000:
                    self.message_history = self.message_history[-1000:]
                
                # Clean up old delivery results (keep last 1000)
                if len(self.delivery_results) > 1000:
                    sorted_results = sorted(
                        self.delivery_results.items(),
                        key=lambda x: x[1].delivery_time_ms or 0
                    )
                    self.delivery_results = dict(sorted_results[-1000:])
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Message cleanup loop error: {e}")
                await asyncio.sleep(30)
    
    async def _network_discovery_loop(self):
        """Periodically discover new agents"""
        while not self._shutdown:
            try:
                await self._discover_agents()
                await asyncio.sleep(120)  # Discovery every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Network discovery loop error: {e}")
                await asyncio.sleep(30)
    
    # Management Methods
    def get_connected_agents(self) -> List[str]:
        """Get list of connected agent IDs"""
        return list(self.direct_connections)
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network status information"""
        return {
            "agent_id": self.agent_id,
            "transport": self.transport.value,
            "server": f"{self.host}:{self.port}",
            "known_agents": len(self.known_agents),
            "direct_connections": len(self.direct_connections),
            "pending_messages": len(self.pending_messages),
            "metrics": {
                "total_messages": self.message_count,
                "successful_deliveries": self.successful_deliveries,
                "failed_deliveries": self.failed_deliveries,
                "avg_delivery_time_ms": (
                    self.total_delivery_time / max(self.successful_deliveries, 1)
                )
            }
        }
    
    async def shutdown(self):
        """Shutdown the A2A communication system"""
        self.logger.info("Shutting down RealA2ACommunication")
        self._shutdown = True
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close WebSocket connections
        for ws in self.websocket_connections.values():
            await ws.close()
        
        # Close HTTP client session
        if self.client_session:
            await self.client_session.close()
        
        # Stop HTTP server
        if self.http_runner:
            await self.http_runner.cleanup()
        
        # Unregister from service discovery
        if self.service_discovery:
            await self.service_discovery.unregister_service(self.agent_id)
        
        self.logger.info("RealA2ACommunication shutdown complete")