"""
Advanced Agent-to-Agent Communication System for Strands Framework
Enterprise-grade A2A protocols with mesh networking, consensus mechanisms, and distributed coordination.
"""
import asyncio
import hashlib
import json
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .distributed_coordination import DistributedCoordinator


class MessageType(Enum):
    """Types of inter-agent messages"""

    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    REQUEST = "request"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    CONSENSUS = "consensus"
    COORDINATION = "coordination"


class MessagePriority(Enum):
    """Message priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class ConsensusType(Enum):
    """Types of consensus mechanisms"""

    SIMPLE_MAJORITY = "simple_majority"
    UNANIMOUS = "unanimous"
    WEIGHTED = "weighted"
    RAFT = "raft"


@dataclass
class A2AMessage:
    """Agent-to-agent message structure"""

    id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: datetime
    ttl: Optional[datetime] = None
    requires_ack: bool = False
    correlation_id: Optional[str] = None
    routing_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """Agent capability definition"""

    name: str
    version: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class NetworkNode:
    """Network node representation"""

    agent_id: str
    capabilities: List[AgentCapability]
    last_seen: datetime
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusProposal:
    """Consensus proposal structure"""

    id: str
    proposer_id: str
    proposal_type: str
    data: Dict[str, Any]
    consensus_type: ConsensusType
    required_votes: int
    votes: Dict[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    status: str = "pending"  # pending, accepted, rejected, expired


class StrandsA2ACommunication:
    """Advanced Agent-to-Agent Communication System"""

    def __init__(self, agent: "EnhancedStrandsAgent"):
        self.agent = agent
        self.logger = logging.getLogger(f"StrandsA2A-{agent.agent_id}")

        # Network topology
        self.known_agents: Dict[str, NetworkNode] = {}
        self.direct_connections: Set[str] = set()
        self.routing_table: Dict[str, str] = {}  # destination -> next_hop

        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.pending_acks: Dict[str, A2AMessage] = {}
        self.message_history: List[A2AMessage] = []

        # Consensus system
        self.active_proposals: Dict[str, ConsensusProposal] = {}
        self.consensus_handlers: Dict[str, Callable] = {}

        # Coordination and synchronization - DEPRECATED (use distributed_coordinator)
        self.coordination_locks: Dict[str, asyncio.Lock] = {}
        self.shared_state: Dict[str, Any] = {}
        self.state_version: int = 0

        # Distributed coordination system
        self.distributed_coordinator: Optional[DistributedCoordinator] = None

        # Performance and reliability
        self.heartbeat_interval = 30  # seconds
        self.message_timeout = 60  # seconds
        self.max_retries = 3
        self.network_health: Dict[str, float] = {}

        self._setup_default_handlers()
        self._background_tasks_started = False

    async def initialize_distributed_coordination(self):
        """Initialize distributed coordination system"""
        if not self.distributed_coordinator:
            self.distributed_coordinator = DistributedCoordinator(
                agent_id=self.agent.agent_id, a2a_communication=self
            )

            # Add known agents to coordinator
            for agent_id in self.direct_connections:
                self.distributed_coordinator.add_known_agent(agent_id)

            self.logger.info("Distributed coordination system initialized")

    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.message_handlers.update(
            {
                "heartbeat": self._handle_heartbeat,
                "capability_discovery": self._handle_capability_discovery,
                "routing_update": self._handle_routing_update,
                "consensus_proposal": self._handle_consensus_proposal,
                "consensus_vote": self._handle_consensus_vote,
                "state_sync": self._handle_state_sync,
                "coordination_request": self._handle_coordination_request_distributed,
            }
        )

    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self._background_tasks_started:
            return

        try:
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._cleanup_expired())
            self._background_tasks_started = True
        except RuntimeError:
            # No event loop - tasks will be started when needed
            pass

    async def connect_to_agent(self, agent_id: str, capabilities: List[AgentCapability] = None):
        """Establish connection to another agent"""
        # Ensure background tasks are running
        await self._start_background_tasks()

        if agent_id == self.agent.agent_id:
            return  # Don't connect to self

        capabilities = capabilities or []

        # Add to known agents
        self.known_agents[agent_id] = NetworkNode(
            agent_id=agent_id, capabilities=capabilities, last_seen=datetime.utcnow()
        )

        # Add direct connection
        self.direct_connections.add(agent_id)

        # Update routing table
        self.routing_table[agent_id] = agent_id  # Direct route

        # Add to distributed coordinator if available
        if self.distributed_coordinator:
            self.distributed_coordinator.add_known_agent(agent_id)

        # Send capability discovery
        await self.send_message(
            recipient_id=agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "action": "capability_discovery",
                "capabilities": [
                    {"name": cap.name, "version": cap.version, "description": cap.description}
                    for cap in self.agent.capabilities
                ],
            },
        )

        self.logger.info(f"Connected to agent {agent_id}")

    async def disconnect_from_agent(self, agent_id: str):
        """Disconnect from an agent"""
        self.direct_connections.discard(agent_id)

        if agent_id in self.known_agents:
            self.known_agents[agent_id].status = "disconnected"

        # Update routing table
        self.routing_table = {
            dest: next_hop for dest, next_hop in self.routing_table.items() if next_hop != agent_id
        }

        # Remove from distributed coordinator if available
        if self.distributed_coordinator:
            self.distributed_coordinator.remove_known_agent(agent_id)

        self.logger.info(f"Disconnected from agent {agent_id}")

    async def send_message(
        self,
        recipient_id: Optional[str],
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_ack: bool = False,
        ttl_seconds: int = 300,
    ) -> str:
        """Send message to another agent or broadcast"""
        message_id = str(uuid.uuid4())

        message = A2AMessage(
            id=message_id,
            sender_id=self.agent.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            payload=payload,
            timestamp=datetime.utcnow(),
            ttl=datetime.utcnow() + timedelta(seconds=ttl_seconds),
            requires_ack=requires_ack,
        )

        # Add to message history
        self.message_history.append(message)

        # Handle different message types
        if message_type == MessageType.BROADCAST:
            await self._broadcast_message(message)
        elif recipient_id:
            await self._route_message(message)
        else:
            raise ValueError("Recipient required for non-broadcast messages")

        # Track pending acknowledgments
        if requires_ack:
            self.pending_acks[message_id] = message

        return message_id

    async def _broadcast_message(self, message: A2AMessage):
        """Broadcast message to all connected agents"""
        for agent_id in self.direct_connections:
            try:
                await self._deliver_message(agent_id, message)
            except Exception as e:
                self.logger.error(f"Failed to broadcast to {agent_id}: {e}")

    async def _route_message(self, message: A2AMessage):
        """Route message to specific recipient"""
        recipient_id = message.recipient_id

        if recipient_id in self.direct_connections:
            # Direct delivery
            await self._deliver_message(recipient_id, message)
        elif recipient_id in self.routing_table:
            # Route through intermediate agent
            next_hop = self.routing_table[recipient_id]
            message.routing_path.append(self.agent.agent_id)
            await self._deliver_message(next_hop, message)
        else:
            # No route found - try discovery
            await self._discover_route(recipient_id)
            raise ValueError(f"No route to agent {recipient_id}")

    async def _deliver_message(self, agent_id: str, message: A2AMessage):
        """Deliver message to specific agent using real A2A transport"""
        if not hasattr(self, "_a2a_transport") or not self._a2a_transport:
            # Initialize A2A transport if not available
            await self._initialize_a2a_transport()

        try:
            # Use real A2A transport for delivery
            from ...protocols.a2a.a2a_protocol import MessageType

            # Convert message to A2A protocol format
            # Map Strands message types to A2A protocol types
            message_type = self._map_to_a2a_message_type(message)
            payload = {
                "original_message": message.payload,
                "message_id": message.id,
                "timestamp": message.timestamp.isoformat(),
            }

            # Send via A2A transport
            result_id = await self._a2a_transport.send_message(
                receiver_id=agent_id,
                message_type=message_type,
                payload=payload,
                priority=message.priority.value,
                correlation_id=message.id,
            )

            # Update network health on successful delivery
            self.network_health[agent_id] = min(1.0, self.network_health.get(agent_id, 1.0) + 0.1)

            if agent_id in self.known_agents:
                self.known_agents[agent_id].last_seen = datetime.utcnow()

            self.logger.debug(
                f"Successfully delivered message {message.id} to {agent_id} (transport_id: {result_id})"
            )

        except Exception as e:
            # Handle delivery failure
            self.network_health[agent_id] = max(0.0, self.network_health.get(agent_id, 1.0) - 0.2)
            self.logger.error(f"A2A delivery failed to {agent_id}: {e}")
            raise Exception(f"A2A delivery failed to {agent_id}: {e}")

    async def _initialize_a2a_transport(self):
        """Initialize A2A transport for real networking"""
        try:
            from ...protocols.a2a.a2a_transport import A2ATransportManager

            # Create A2A transport manager
            self._a2a_transport = A2ATransportManager(
                agent_id=self.agent.agent_id,
                host=getattr(self.agent, "host", "localhost"),
                port=getattr(self.agent, "a2a_port", None),
            )

            # Initialize the transport
            success = await self._a2a_transport.initialize()
            if success:
                # Add A2A capabilities
                self._a2a_transport.add_capability("strands_communication")
                self._a2a_transport.add_capability("consensus")
                self._a2a_transport.add_capability("coordination")

                self.logger.info(f"A2A transport initialized for agent {self.agent.agent_id}")
            else:
                self.logger.error("Failed to initialize A2A transport")
                self._a2a_transport = None

        except Exception as e:
            self.logger.error(f"A2A transport initialization failed: {e}")
            self._a2a_transport = None

    def _map_to_a2a_message_type(self, message: A2AMessage):
        """Map Strands message types to A2A protocol message types"""
        from ...protocols.a2a.a2a_protocol import MessageType as A2AMessageType

        # Extract action from message payload
        action = message.payload.get("action", "unknown")

        # Map common Strands actions to A2A message types
        action_mapping = {
            "workflow_request": A2AMessageType.WORKFLOW_REQUEST,
            "workflow_response": A2AMessageType.WORKFLOW_RESPONSE,
            "workflow_status": A2AMessageType.WORKFLOW_STATUS,
            "heartbeat": A2AMessageType.HEARTBEAT,
            "capability_discovery": A2AMessageType.DATA_QUERY,
            "routing_update": A2AMessageType.DATA_QUERY,
            "consensus_proposal": A2AMessageType.WORKFLOW_REQUEST,
            "consensus_vote": A2AMessageType.WORKFLOW_RESPONSE,
            "state_sync": A2AMessageType.MEMORY_SHARE,
            "coordination_request": A2AMessageType.WORKFLOW_REQUEST,
            "memory_share": A2AMessageType.MEMORY_SHARE,
            "memory_request": A2AMessageType.MEMORY_REQUEST,
            "analysis_request": A2AMessageType.ANALYSIS_REQUEST,
            "data_request": A2AMessageType.DATA_QUERY,
        }

        # Return mapped type or default to WORKFLOW_REQUEST for unknown actions
        return action_mapping.get(action, A2AMessageType.WORKFLOW_REQUEST)

    async def _discover_route(self, target_agent_id: str):
        """Discover route to target agent"""
        # Send route discovery requests
        discovery_payload = {
            "action": "route_discovery",
            "target": target_agent_id,
            "path": [self.agent.agent_id],
        }

        for agent_id in self.direct_connections:
            await self.send_message(
                recipient_id=agent_id, message_type=MessageType.REQUEST, payload=discovery_payload
            )

    async def receive_message(self, message: A2AMessage):
        """Receive and process incoming message"""
        await self.message_queue.put(message)

    async def _message_processor(self):
        """Background message processor"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._process_message(message)
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")

    async def _process_message(self, message: A2AMessage):
        """Process incoming message"""
        # Check TTL
        if message.ttl and datetime.utcnow() > message.ttl:
            self.logger.warning(f"Message {message.id} expired")
            return

        # Handle acknowledgment
        if message.requires_ack:
            await self._send_acknowledgment(message)

        # Route message if not for us
        if message.recipient_id and message.recipient_id != self.agent.agent_id:
            await self._route_message(message)
            return

        # Process message payload
        action = message.payload.get("action")
        if action in self.message_handlers:
            try:
                await self.message_handlers[action](message)
            except Exception as e:
                self.logger.error(f"Handler error for {action}: {e}")
        else:
            self.logger.warning(f"No handler for action: {action}")

    async def _send_acknowledgment(self, original_message: A2AMessage):
        """Send acknowledgment for received message"""
        ack_payload = {
            "action": "acknowledgment",
            "original_message_id": original_message.id,
            "status": "received",
        }

        await self.send_message(
            recipient_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            payload=ack_payload,
        )

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                heartbeat_payload = {
                    "action": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "active",
                    "capabilities": len(self.agent.tool_registry),
                }

                await self.send_message(
                    recipient_id=None,  # Broadcast
                    message_type=MessageType.HEARTBEAT,
                    payload=heartbeat_payload,
                )

            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")

    async def _cleanup_expired(self):
        """Clean up expired messages and proposals"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()

                # Clean expired pending acks
                expired_acks = [
                    msg_id for msg_id, msg in self.pending_acks.items() if msg.ttl and now > msg.ttl
                ]
                for msg_id in expired_acks:
                    del self.pending_acks[msg_id]

                # Clean expired proposals
                expired_proposals = [
                    prop_id
                    for prop_id, prop in self.active_proposals.items()
                    if prop.expires_at and now > prop.expires_at
                ]
                for prop_id in expired_proposals:
                    self.active_proposals[prop_id].status = "expired"

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    # Message Handlers
    async def _handle_heartbeat(self, message: A2AMessage):
        """Handle heartbeat message"""
        sender_id = message.sender_id

        if sender_id in self.known_agents:
            self.known_agents[sender_id].last_seen = datetime.utcnow()
            self.known_agents[sender_id].status = "active"

        self.logger.debug(f"Heartbeat received from {sender_id}")

    async def _handle_capability_discovery(self, message: A2AMessage):
        """Handle capability discovery request"""
        sender_id = message.sender_id
        sender_capabilities = message.payload.get("capabilities", [])

        # Update known agent capabilities
        if sender_id in self.known_agents:
            self.known_agents[sender_id].capabilities = [
                AgentCapability(**cap) for cap in sender_capabilities
            ]

        # Respond with our capabilities
        response_payload = {
            "action": "capability_response",
            "capabilities": [
                {"name": cap, "version": "1.0", "description": f"Agent capability: {cap}"}
                for cap in self.agent.capabilities
            ],
        }

        await self.send_message(
            recipient_id=sender_id, message_type=MessageType.RESPONSE, payload=response_payload
        )

    async def _handle_routing_update(self, message: A2AMessage):
        """Handle routing table update"""
        updates = message.payload.get("routes", {})

        for destination, next_hop in updates.items():
            if destination != self.agent.agent_id:  # Don't route to self
                self.routing_table[destination] = next_hop

        self.logger.debug(f"Routing table updated with {len(updates)} routes")

    async def _handle_consensus_proposal(self, message: A2AMessage):
        """Handle consensus proposal"""
        proposal_data = message.payload

        proposal = ConsensusProposal(
            id=proposal_data["id"],
            proposer_id=message.sender_id,
            proposal_type=proposal_data["type"],
            data=proposal_data["data"],
            consensus_type=ConsensusType(proposal_data["consensus_type"]),
            required_votes=proposal_data["required_votes"],
        )

        self.active_proposals[proposal.id] = proposal

        # Auto-vote if we have a handler
        proposal_type = proposal.proposal_type
        if proposal_type in self.consensus_handlers:
            try:
                vote = await self.consensus_handlers[proposal_type](proposal)
                await self._cast_vote(proposal.id, vote)
            except Exception as e:
                self.logger.error(f"Consensus handler error: {e}")

    async def _handle_consensus_vote(self, message: A2AMessage):
        """Handle consensus vote"""
        proposal_id = message.payload["proposal_id"]
        vote = message.payload["vote"]
        voter_id = message.sender_id

        if proposal_id in self.active_proposals:
            proposal = self.active_proposals[proposal_id]
            proposal.votes[voter_id] = vote

            # Check if consensus reached
            await self._check_consensus(proposal)

    async def _handle_state_sync(self, message: A2AMessage):
        """Handle state synchronization"""
        remote_state = message.payload.get("state", {})
        remote_version = message.payload.get("version", 0)

        if remote_version > self.state_version:
            # Remote state is newer
            self.shared_state.update(remote_state)
            self.state_version = remote_version
            self.logger.info(f"State synchronized to version {remote_version}")

    async def _handle_coordination_request_distributed(self, message: A2AMessage):
        """Handle coordination request using distributed coordinator"""
        if not self.distributed_coordinator:
            await self.initialize_distributed_coordination()

        try:
            # Delegate to distributed coordinator
            response = await self.distributed_coordinator.handle_coordination_message(message)

            # Send response if needed
            if response and "error" not in response:
                await self.send_message(
                    recipient_id=message.sender_id,
                    message_type=MessageType.RESPONSE,
                    payload={"action": "coordination_response", "response": response},
                )
        except Exception as e:
            self.logger.error(f"Distributed coordination error: {e}")

            # Fallback to legacy coordination for basic lock requests
            await self._handle_coordination_request_legacy(message)

    async def _handle_coordination_request_legacy(self, message: A2AMessage):
        """Legacy coordination handler for backward compatibility"""
        coordination_type = message.payload.get("type")
        coordination_data = message.payload.get("data", {})

        if coordination_type == "lock_request":
            resource_id = coordination_data["resource_id"]
            await self._handle_lock_request(message.sender_id, resource_id)
        elif coordination_type == "sync_request":
            await self._handle_sync_request(message.sender_id, coordination_data)

    async def _cast_vote(self, proposal_id: str, vote: bool):
        """Cast vote on consensus proposal"""
        vote_payload = {"action": "consensus_vote", "proposal_id": proposal_id, "vote": vote}

        if proposal_id in self.active_proposals:
            proposal = self.active_proposals[proposal_id]
            await self.send_message(
                recipient_id=proposal.proposer_id,
                message_type=MessageType.RESPONSE,
                payload=vote_payload,
            )

    async def _check_consensus(self, proposal: ConsensusProposal):
        """Check if consensus has been reached"""
        total_votes = len(proposal.votes)
        yes_votes = sum(1 for vote in proposal.votes.values() if vote)

        consensus_reached = False

        if proposal.consensus_type == ConsensusType.SIMPLE_MAJORITY:
            consensus_reached = yes_votes > total_votes / 2
        elif proposal.consensus_type == ConsensusType.UNANIMOUS:
            consensus_reached = yes_votes == total_votes

        if consensus_reached and total_votes >= proposal.required_votes:
            proposal.status = "accepted"
            self.logger.info(f"Consensus reached for proposal {proposal.id}")

            # Execute proposal if we're the proposer
            if proposal.proposer_id == self.agent.agent_id:
                await self._execute_consensus_decision(proposal)

    async def _execute_consensus_decision(self, proposal: ConsensusProposal):
        """Execute consensus decision"""
        self.logger.info(f"Executing consensus decision: {proposal.proposal_type}")

        # Implementation depends on proposal type
        if proposal.proposal_type == "state_update":
            self.shared_state.update(proposal.data)
            self.state_version += 1

    async def _handle_lock_request(self, requester_id: str, resource_id: str):
        """Handle distributed lock request"""
        if resource_id not in self.coordination_locks:
            self.coordination_locks[resource_id] = asyncio.Lock()

        # Try to acquire lock
        lock = self.coordination_locks[resource_id]
        if lock.locked():
            # Lock is busy
            response_payload = {
                "action": "lock_response",
                "resource_id": resource_id,
                "granted": False,
                "reason": "resource_busy",
            }
        else:
            # Grant lock
            response_payload = {
                "action": "lock_response",
                "resource_id": resource_id,
                "granted": True,
            }

        await self.send_message(
            recipient_id=requester_id, message_type=MessageType.RESPONSE, payload=response_payload
        )

    async def _handle_sync_request(self, requester_id: str, sync_data: Dict[str, Any]):
        """Handle synchronization request"""
        sync_type = sync_data.get("type")

        if sync_type == "state_sync":
            # Send current state
            response_payload = {
                "action": "sync_response",
                "state": self.shared_state,
                "version": self.state_version,
            }

            await self.send_message(
                recipient_id=requester_id,
                message_type=MessageType.RESPONSE,
                payload=response_payload,
            )

    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        status = {
            "agent_id": self.agent.agent_id,
            "connected_agents": len(self.direct_connections),
            "known_agents": len(self.known_agents),
            "active_proposals": len(
                [p for p in self.active_proposals.values() if p.status == "pending"]
            ),
            "message_queue_size": self.message_queue.qsize(),
            "pending_acks": len(self.pending_acks),
            "shared_state_version": self.state_version,
            "network_health": dict(self.network_health),
        }

        # Add distributed coordination status if available
        if self.distributed_coordinator:
            status[
                "distributed_coordination"
            ] = self.distributed_coordinator.get_coordination_status()

        return status

    # Distributed Coordination Public API
    async def acquire_distributed_lock(
        self, resource_id: str, timeout_seconds: float = 30.0
    ) -> bool:
        """Acquire a distributed lock across the agent network"""
        if not self.distributed_coordinator:
            await self.initialize_distributed_coordination()

        return await self.distributed_coordinator.acquire_distributed_lock(
            resource_id, timeout_seconds
        )

    async def release_distributed_lock(self, resource_id: str) -> bool:
        """Release a distributed lock"""
        if not self.distributed_coordinator:
            return False

        return await self.distributed_coordinator.release_distributed_lock(resource_id)

    async def start_distributed_transaction(
        self, participants: List[str], operations: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Start a distributed transaction using Two-Phase Commit"""
        if not self.distributed_coordinator:
            await self.initialize_distributed_coordination()

        return await self.distributed_coordinator.start_distributed_transaction(
            participants, operations
        )

    async def start_leader_election(self) -> bool:
        """Start leader election process"""
        if not self.distributed_coordinator:
            await self.initialize_distributed_coordination()

        return await self.distributed_coordinator.start_leader_election()

    def is_coordination_leader(self) -> bool:
        """Check if this agent is the coordination leader"""
        if not self.distributed_coordinator:
            return False

        return self.distributed_coordinator.is_leader()

    async def shutdown_coordination(self):
        """Shutdown distributed coordination system"""
        if self.distributed_coordinator:
            await self.distributed_coordinator.shutdown()
            self.distributed_coordinator = None
