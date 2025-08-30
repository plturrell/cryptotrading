"""
A2A Messaging Client for Agent-to-Agent Communication
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from .a2a_protocol import A2AMessage, AgentStatus, MessageType

logger = logging.getLogger(__name__)


class A2AMessagingClient:
    """A2A Messaging Client for agent-to-agent communication"""

    def __init__(self, agent_id: str):
        """Initialize messaging client for an agent"""
        self.agent_id = agent_id
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.connected = False
        
        logger.info(f"A2A Messaging Client initialized for agent: {agent_id}")

    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for specific message type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type}")

    async def send_message(
        self, 
        receiver_id: str, 
        message_type: MessageType, 
        payload: Dict[str, Any],
        wait_for_response: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Send a message to another agent"""
        message_id = str(uuid.uuid4())
        
        message = A2AMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            message_id=message_id,
            timestamp=None,  # Will be set by protocol
            priority=0,
            correlation_id=None
        )
        
        logger.info(f"Sending {message_type} message from {self.agent_id} to {receiver_id}")
        
        # In a real implementation, this would send via network transport
        # For now, we'll simulate the message sending
        if wait_for_response:
            # Create future for response
            future = asyncio.Future()
            self.pending_responses[message_id] = future
            
            # Simulate async response (in real implementation, this would come from network)
            asyncio.create_task(self._simulate_response(message_id, message_type))
            
            try:
                response = await asyncio.wait_for(future, timeout=30.0)
                return response
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for response to message {message_id}")
                self.pending_responses.pop(message_id, None)
                return None
        
        return {"status": "sent", "message_id": message_id}

    async def _simulate_response(self, message_id: str, original_type: MessageType):
        """Simulate a response message (for testing)"""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        if message_id in self.pending_responses:
            # Create a mock response
            response = {
                "status": "success",
                "message_id": message_id,
                "original_type": original_type.value,
                "response_data": {"mock": True}
            }
            
            future = self.pending_responses.pop(message_id)
            future.set_result(response)

    async def handle_incoming_message(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle an incoming message"""
        logger.info(f"Received {message.message_type} message from {message.sender_id}")
        
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                return await handler(message)
            except Exception as e:
                logger.error(f"Error handling message {message.message_id}: {e}")
                return {"error": str(e)}
        else:
            logger.warning(f"No handler registered for message type: {message.message_type}")
            return {"error": f"No handler for {message.message_type}"}

    async def start(self):
        """Start the messaging client"""
        self.connected = True
        logger.info(f"A2A Messaging Client started for {self.agent_id}")

    async def stop(self):
        """Stop the messaging client"""
        self.connected = False
        
        # Cancel any pending responses
        for future in self.pending_responses.values():
            if not future.done():
                future.cancel()
        
        self.pending_responses.clear()
        logger.info(f"A2A Messaging Client stopped for {self.agent_id}")

    def get_status(self) -> Dict[str, Any]:
        """Get messaging client status"""
        return {
            "agent_id": self.agent_id,
            "connected": self.connected,
            "handlers_registered": len(self.message_handlers),
            "pending_responses": len(self.pending_responses)
        }


class A2AMessageBroker:
    """Central message broker for A2A communication"""
    
    def __init__(self):
        self.clients: Dict[str, A2AMessagingClient] = {}
        self.message_queue: List[A2AMessage] = []
        
    def register_client(self, client: A2AMessagingClient):
        """Register a messaging client"""
        self.clients[client.agent_id] = client
        logger.info(f"Registered A2A client: {client.agent_id}")
    
    def unregister_client(self, agent_id: str):
        """Unregister a messaging client"""
        if agent_id in self.clients:
            del self.clients[agent_id]
            logger.info(f"Unregistered A2A client: {agent_id}")
    
    async def route_message(self, message: A2AMessage):
        """Route a message to the appropriate client"""
        target_client = self.clients.get(message.receiver_id)
        if target_client:
            await target_client.handle_incoming_message(message)
        else:
            logger.warning(f"No client found for agent: {message.receiver_id}")


# Global message broker instance
_message_broker = A2AMessageBroker()


def get_message_broker() -> A2AMessageBroker:
    """Get the global message broker instance"""
    return _message_broker