"""
Unified base agent class for the cryptotrading platform.
Consolidates functionality from multiple previous agent base classes.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime

class BaseAgent(ABC):
    """
    Unified base agent class that consolidates functionality from:
    - rex.a2a.agents.base_classes.BaseAgent
    - rex.a2a.agents.a2a_agent_base.A2AAgentBase
    - strands.agent.Agent
    """
    
    def __init__(self, agent_id: str, agent_type: str, **kwargs):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.created_at = datetime.utcnow()
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{agent_id}]")
        self._initialize(**kwargs)
    
    def _initialize(self, **kwargs):
        """Initialize agent-specific configuration"""
        pass
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.logger.info(f"Starting agent {self.agent_id}")
    
    async def stop(self):
        """Stop the agent"""
        self.logger.info(f"Stopping agent {self.agent_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "created_at": self.created_at.isoformat(),
            "status": "active"
        }
