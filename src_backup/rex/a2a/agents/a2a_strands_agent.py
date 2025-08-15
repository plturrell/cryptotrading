"""
A2A Protocol Compliant Strands Agent
Provides A2A protocol compliance layer on top of BaseStrandsAgent
"""

import logging
import asyncio
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime

from .base_strands_agent import BaseStrandsAgent
from ..protocols.a2a_protocol import (
    A2AMessage, A2AResponse, A2AProtocol, MessageType, AgentStatus, 
    A2A_CAPABILITIES, A2A_ROUTING
)
from ..registry.registry import agent_registry

logger = logging.getLogger(__name__)

# Import MessageTransport from base transport to avoid circular imports  
from ..transports.base_transport import MessageTransport

class DefaultTransport(MessageTransport):
    """Default in-memory transport for testing"""
    
    def __init__(self):
        self.message_queues: Dict[str, List[A2AMessage]] = {}
    
    async def send_message(self, message: A2AMessage) -> bool:
        """Send message to receiver's queue"""
        if message.receiver_id not in self.message_queues:
            self.message_queues[message.receiver_id] = []
        self.message_queues[message.receiver_id].append(message)
        logger.debug(f"Message sent to {message.receiver_id}: {message.message_id}")
        return True
    
    async def receive_messages(self, agent_id: str) -> List[A2AMessage]:
        """Get all pending messages for agent"""
        messages = self.message_queues.get(agent_id, [])
        self.message_queues[agent_id] = []  # Clear after reading
        return messages
    
    async def acknowledge_message(self, agent_id: str, message_id: str) -> bool:
        """Acknowledge message (no-op for default transport)"""
        return True

class A2AStrandsAgent(BaseStrandsAgent):
    """A2A Protocol Compliant Strands Agent"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        model_provider: str = "grok4",
        version: str = "1.0",
        transport: Optional[MessageTransport] = None
    ):
        """
        Initialize A2A compliant Strands agent
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., 'historical_loader', 'database')
            capabilities: List of agent capabilities
            model_provider: AI model provider (only 'grok4' supported)
            version: Agent version
            transport: Message transport implementation (defaults to in-memory)
        """
        # Initialize base Strands agent
        super().__init__(agent_id, agent_type, capabilities, model_provider, version)
        
        # A2A Protocol state
        self.agent_status = AgentStatus.ACTIVE
        self.transport = transport or DefaultTransport()
        self.message_processing = False
        self.message_check_interval = 2.0  # seconds
        
        # A2A Protocol validation
        self._validate_a2a_compliance()
        
        # Start message monitoring (lazy - only when event loop is available)
        self._message_monitor_task = None
        self._monitoring_started = False
        self._start_monitoring()
        
        logger.info(f"A2A Strands agent initialized: {self.agent_id}")
    
    def _create_tools(self):
        """Create base A2A tools for all Strands agents"""
        from strands import tool
        
        @tool
        def send_a2a_message(receiver_id: str, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            """Send A2A message to another agent"""
            try:
                message = A2AProtocol.create_message(
                    sender_id=self.agent_id,
                    receiver_id=receiver_id,
                    message_type=MessageType(message_type),
                    payload=payload
                )
                
                # Send via transport
                result = asyncio.run(self.transport.send(message))
                return {"success": True, "message_id": message.message_id, "result": result}
                
            except Exception as e:
                logger.error(f"Failed to send A2A message: {e}")
                return {"success": False, "error": str(e)}
        
        @tool
        def receive_a2a_messages() -> Dict[str, Any]:
            """Receive pending A2A messages for this agent"""
            try:
                messages = asyncio.run(self.transport.receive(self.agent_id))
                return {
                    "success": True,
                    "message_count": len(messages),
                    "messages": [msg.to_dict() for msg in messages]
                }
                
            except Exception as e:
                logger.error(f"Failed to receive A2A messages: {e}")
                return {"success": False, "error": str(e)}
        
        @tool
        def get_agent_info() -> Dict[str, Any]:
            """Get this agent's information and capabilities"""
            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "capabilities": self.capabilities,
                "model_provider": self.model_provider,
                "version": self.version,
                "transport_type": type(self.transport).__name__
            }
            
        return [send_a2a_message, receive_a2a_messages, get_agent_info]
    
    def _start_monitoring(self):
        """Start message monitoring if event loop is available"""
        try:
            if asyncio.get_running_loop():
                self._message_monitor_task = asyncio.create_task(self._monitor_a2a_messages())
        except RuntimeError:
            # No event loop running, monitoring will start when needed
            logger.debug(f"No event loop running for {self.agent_id}, monitoring will start later")
    
    def _validate_a2a_compliance(self):
        """Validate agent meets A2A protocol requirements"""
        # Check if agent type has defined capabilities
        if self.agent_id in A2A_CAPABILITIES:
            expected_capabilities = A2A_CAPABILITIES[self.agent_id]
            missing_capabilities = [cap for cap in expected_capabilities if cap not in self.capabilities]
            if missing_capabilities:
                logger.warning(f"Agent {self.agent_id} missing A2A capabilities: {missing_capabilities}")
        
        # Validate agent can handle required message types
        required_handlers = self._get_required_message_handlers()
        for msg_type in required_handlers:
            if not hasattr(self, f'_handle_{msg_type.value.lower()}'):
                logger.warning(f"Agent {self.agent_id} missing handler for {msg_type.value}")
    
    def _get_required_message_handlers(self) -> List[MessageType]:
        """Get message types this agent should handle based on A2A routing"""
        handled_types = []
        for msg_type, target_agents in A2A_ROUTING.items():
            if self.agent_id in target_agents or self.agent_type in [a.split('-')[0] for a in target_agents]:
                handled_types.append(msg_type)
        return handled_types
    
    async def _monitor_a2a_messages(self):
        """Monitor for incoming A2A messages"""
        while True:
            try:
                if not self.message_processing and self.agent_status == AgentStatus.ACTIVE:
                    await self._process_pending_a2a_messages()
                
                await asyncio.sleep(self.message_check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring A2A messages for {self.agent_id}: {e}")
                await asyncio.sleep(self.message_check_interval)
    
    async def _process_pending_a2a_messages(self):
        """Process pending A2A messages"""
        self.message_processing = True
        
        try:
            messages = await self.transport.receive_messages(self.agent_id)
            
            for message in messages:
                logger.info(f"Processing A2A message: {message.message_id} from {message.sender_id}")
                
                # Validate message
                if not self._validate_a2a_message(message):
                    logger.error(f"Invalid A2A message: {message.message_id}")
                    continue
                
                # Process message based on type
                response = await self._route_a2a_message(message)
                
                # Send response if needed
                if response and self._requires_response(message.message_type):
                    await self._send_a2a_response(message, response)
                
                # Acknowledge message
                await self.transport.acknowledge_message(self.agent_id, message.message_id)
                
        except Exception as e:
            logger.error(f"Error processing A2A messages: {e}")
        
        finally:
            self.message_processing = False
    
    def _validate_a2a_message(self, message: A2AMessage) -> bool:
        """Validate A2A message compliance"""
        # Check required fields
        if not all([message.sender_id, message.receiver_id, message.message_type, message.message_id]):
            return False
        
        # Check if agent should handle this message type
        if message.message_type not in self._get_required_message_handlers():
            logger.warning(f"Agent {self.agent_id} received unexpected message type: {message.message_type.value}")
            return False
        
        # Check protocol version compatibility
        if message.protocol_version and message.protocol_version != "1.0":
            logger.warning(f"Unsupported protocol version: {message.protocol_version}")
        
        return True
    
    async def _route_a2a_message(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Route A2A message to appropriate handler"""
        message_type = message.message_type
        
        # Try specific handler first
        handler_name = f'_handle_{message_type.value.lower()}'
        if hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            return await handler(message)
        
        # Fall back to generic handler
        return await self.handle_a2a_message(message)
    
    def _requires_response(self, message_type: MessageType) -> bool:
        """Check if message type requires a response"""
        response_required = [
            MessageType.DATA_LOAD_REQUEST,
            MessageType.ANALYSIS_REQUEST,
            MessageType.DATA_QUERY,
            MessageType.WORKFLOW_REQUEST,
            MessageType.TRADE_EXECUTION
        ]
        return message_type in response_required
    
    async def _send_a2a_response(self, original_message: A2AMessage, response_data: Dict[str, Any]):
        """Send A2A response message"""
        response_type = self._get_response_message_type(original_message.message_type)
        
        response_message = A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id=original_message.sender_id,
            message_type=response_type,
            payload=response_data,
            priority=original_message.priority
        )
        
        # Copy workflow context if present
        if original_message.workflow_context:
            response_message.workflow_context = original_message.workflow_context
        
        response_message.correlation_id = original_message.message_id
        
        success = await self.transport.send_message(response_message)
        if success:
            logger.info(f"A2A response sent: {response_message.message_id}")
        else:
            logger.error(f"Failed to send A2A response: {response_message.message_id}")
    
    def _get_response_message_type(self, request_type: MessageType) -> MessageType:
        """Get appropriate response message type for request"""
        response_map = {
            MessageType.DATA_LOAD_REQUEST: MessageType.DATA_LOAD_RESPONSE,
            MessageType.ANALYSIS_REQUEST: MessageType.ANALYSIS_RESPONSE,
            MessageType.DATA_QUERY: MessageType.DATA_QUERY_RESPONSE,
            MessageType.WORKFLOW_REQUEST: MessageType.WORKFLOW_RESPONSE,
            MessageType.TRADE_EXECUTION: MessageType.TRADE_RESPONSE
        }
        return response_map.get(request_type, MessageType.ERROR)
    
    async def send_a2a_message(
        self,
        target_agent_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 0,
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Send A2A compliant message to another agent"""
        message = A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id=target_agent_id,
            message_type=message_type,
            payload=payload,
            priority=priority
        )
        
        if workflow_context:
            message.workflow_context = workflow_context
        
        success = await self.transport.send_message(message)
        if success:
            logger.info(f"A2A message sent to {target_agent_id}: {message.message_id}")
            return message.message_id
        else:
            logger.error(f"Failed to send A2A message to {target_agent_id}")
            return None
    
    def set_transport(self, transport: MessageTransport):
        """Set message transport implementation"""
        self.transport = transport
        logger.info(f"Transport updated for agent {self.agent_id}")
    
    def update_agent_status(self, status: AgentStatus):
        """Update agent status"""
        old_status = self.agent_status
        self.agent_status = status
        logger.info(f"Agent {self.agent_id} status changed: {old_status.value} -> {status.value}")
    
    def get_a2a_info(self) -> Dict[str, Any]:
        """Get A2A protocol compliance information"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "status": self.agent_status.value,
            "protocol_version": "1.0",
            "a2a_compliant": True,
            "supported_message_types": [mt.value for mt in self._get_required_message_handlers()],
            "transport_type": type(self.transport).__name__,
            "message_processing": self.message_processing
        }
    
    async def shutdown(self):
        """Gracefully shutdown A2A agent"""
        logger.info(f"Shutting down A2A agent {self.agent_id}")
        
        # Update status
        self.update_agent_status(AgentStatus.INACTIVE)
        
        # Cancel message monitoring
        if hasattr(self, '_message_monitor_task'):
            self._message_monitor_task.cancel()
        
        # Wait for current message processing to complete
        while self.message_processing:
            await asyncio.sleep(0.1)
        
        logger.info(f"A2A agent {self.agent_id} shutdown complete")
    
    # Default A2A message handlers (can be overridden by subclasses)
    
    async def _handle_heartbeat(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle heartbeat message"""
        return {
            "agent_id": self.agent_id,
            "status": self.agent_status.value,
            "timestamp": datetime.now().isoformat(),
            "capabilities": self.capabilities
        }
    
    async def _handle_workflow_status(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle workflow status request"""
        workflow_id = message.payload.get('workflow_id')
        return {
            "agent_id": self.agent_id,
            "workflow_id": workflow_id,
            "status": "ready",
            "capabilities": self.capabilities
        }