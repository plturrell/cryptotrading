"""
Base Transport Interface for A2A Messages
Defines abstract transport layer to avoid circular imports
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..protocols.a2a_protocol import A2AMessage

class MessageTransport(ABC):
    """Abstract transport layer for A2A messages"""
    
    @abstractmethod
    async def send_message(self, message: 'A2AMessage') -> bool:
        """Send A2A message via transport"""
        pass
    
    @abstractmethod
    async def receive_messages(self, agent_id: str) -> List['A2AMessage']:
        """Receive pending messages for agent"""
        pass
    
    @abstractmethod
    async def acknowledge_message(self, agent_id: str, message_id: str) -> bool:
        """Acknowledge message processing"""
        pass