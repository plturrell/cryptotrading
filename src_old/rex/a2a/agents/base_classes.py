"""
Base classes for A2A agents
Provides proper class hierarchy for agent implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """Metaclass for singleton pattern implementation"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseModelProvider(ABC):
    """Base class for model providers with singleton support"""
    __metaclass__ = SingletonMeta
    
    @abstractmethod
    async def complete(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Complete a chat interaction"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to the model provider"""
        pass


class BaseAgent(ABC):
    """Base class for all agents with common functionality"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": "active"
        }


class BaseKeyManager(ABC):
    """Base class for key management with proper instance methods"""
    
    def __init__(self):
        self.keys_cache = {}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def generate_keys(self, identifier: str, password: Optional[str] = None) -> Tuple[str, str]:
        """Generate new keys for an identifier"""
        pass
    
    @abstractmethod
    def load_keys(self, identifier: str, password: Optional[str] = None) -> Tuple[str, str]:
        """Load existing keys for an identifier"""
        pass
    
    def get_or_create_keys(self, identifier: str, password: Optional[str] = None) -> Tuple[str, str]:
        """Get existing keys or create new ones"""
        try:
            return self.load_keys(identifier, password)
        except FileNotFoundError:
            self._logger.info(f"Creating new keys for {identifier}")
            return self.generate_keys(identifier, password)