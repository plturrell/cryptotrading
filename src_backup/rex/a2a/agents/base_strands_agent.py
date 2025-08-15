"""
Base Strands Agent Class for A2A Protocol Compliant Agents
Provides common functionality for all A2A agents using Strands SDK
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from strands import Agent
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime

# Use Strands native model support instead of custom client
from ..registry.registry import agent_registry
from ..protocols import A2AMessage, A2AProtocol, MessageType

logger = logging.getLogger(__name__)

class BaseStrandsAgent(ABC):
    """Base class for all A2A protocol compliant Strands agents"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        model_provider: str = "grok4",
        version: str = "1.0"
    ):
        """
        Initialize base Strands agent with common functionality
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., 'historical_loader', 'database')
            capabilities: List of agent capabilities
            model_provider: AI model provider ('grok4' only)
            version: Agent version
        """
        # A2A Protocol compliance
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.model_provider = model_provider
        self.version = version
        
        # Initialize model based on provider
        self.model = self._initialize_model(model_provider)
        
        # Get agent tools (to be implemented by subclasses)
        self.tools = self._create_tools()
        
        # Create Strands agent
        self.agent = Agent(
            tools=self.tools,
            model=self.model
        )
        
        # Register with A2A registry
        self._register_agent()
        
        logger.info(f"Initialized {self.agent_type} agent: {self.agent_id}")
    
    def _initialize_model(self, model_provider: str):
        """Initialize AI model based on provider - Use Strands native support"""
        if model_provider == "grok4":
            # Strands will handle Grok-4 natively - no custom client needed
            return None  # Let Strands use its native Grok-4 integration
        else:
            logger.warning(f"Unknown model provider: {model_provider}, using Grok-4 as default")
            return None  # Let Strands handle Grok-4 natively
    
    def _register_agent(self):
        """Register agent with A2A registry"""
        agent_registry.register_agent(
            self.agent_id,
            self.agent_type,
            self.capabilities,
            {
                'version': self.version,
                'model_provider': self.model_provider,
                'a2a_compliant': True,
                'registered_at': datetime.now().isoformat()
            }
        )
    
    @abstractmethod
    def _create_tools(self) -> List[Callable]:
        """
        Create and return agent-specific tools
        Must be implemented by subclasses
        
        Returns:
            List of tool functions decorated with @tool
        """
        pass
    
    async def process_request(self, request: str) -> str:
        """Process natural language requests using Strands agent"""
        try:
            # Use async method when in async context
            response = await self.agent.process_async(request)
            return str(response)
        except Exception as e:
            logger.error(f"Error processing request in {self.agent_id}: {e}")
            return f"Error: {str(e)}"
    
    async def handle_a2a_message(self, message: A2AMessage) -> Dict[str, Any]:
        """
        Handle incoming A2A protocol messages
        Base implementation - can be overridden by subclasses
        """
        try:
            logger.info(f"{self.agent_id} handling A2A message from {message.sender_id}: {message.message_type.value}")
            
            # Convert message to natural language for agent processing
            prompt = self._message_to_prompt(message)
            response = await self.process_request(prompt)
            
            return {
                "success": True,
                "agent_id": self.agent_id,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling A2A message in {self.agent_id}: {e}")
            return {
                "success": False,
                "agent_id": self.agent_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _message_to_prompt(self, message: A2AMessage) -> str:
        """
        Convert A2A message to natural language prompt
        Can be overridden by subclasses for custom handling
        """
        message_type = message.message_type.value
        payload = message.payload
        
        # Default conversion - subclasses can customize
        return f"Process {message_type} with data: {payload}"
    
    async def send_a2a_message(
        self,
        target_agent_id: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> A2AMessage:
        """Send A2A compliant message to another agent"""
        message = A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id=target_agent_id,
            message_type=message_type,
            payload=payload
        )
        
        logger.info(f"{self.agent_id} sending A2A message to {target_agent_id}: {message_type.value}")
        return message
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "model_provider": self.model_provider,
            "version": self.version,
            "status": "active"
        }
    
    def validate_payload(self, payload: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        Validate that payload contains required fields
        Utility method for subclasses
        """
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            logger.error(f"Missing required fields in payload: {missing_fields}")
            return False
        return True
    
    def format_error_response(self, error: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format standardized error response
        Utility method for subclasses
        """
        response = {
            "success": False,
            "agent_id": self.agent_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        if context:
            response["context"] = context
        return response
    
    def format_success_response(self, data: Any, message: str = None) -> Dict[str, Any]:
        """
        Format standardized success response
        Utility method for subclasses
        """
        response = {
            "success": True,
            "agent_id": self.agent_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        if message:
            response["message"] = message
        return response