"""
Agent Registry for rex.com A2A System
Now powered by persistent database storage with Redis caching
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging

from ...registry.persistent_registry import persistent_agent_registry
from ...database.cache import cache_manager

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Central registry for all A2A agents - now with persistent storage"""
    
    def __init__(self):
        # Use persistent registry as backend
        self.persistent_registry = persistent_agent_registry
        self.cache = cache_manager
        
        # Legacy in-memory storage for backward compatibility during migration
        self.agents = {}
        self.capabilities = {}
        self.connections = {}
        
        # Load existing agents from database
        self._load_from_database()
        
    def _load_from_database(self):
        """Load agents from database into memory for backward compatibility"""
        try:
            db_agents = self.persistent_registry.get_all_agents()
            for agent_id, agent_data in db_agents.items():
                self.agents[agent_id] = agent_data
                
                # Index capabilities
                for capability in agent_data.get('capabilities', []):
                    if capability not in self.capabilities:
                        self.capabilities[capability] = []
                    if agent_id not in self.capabilities[capability]:
                        self.capabilities[capability].append(agent_id)
                        
            logger.info(f"Loaded {len(db_agents)} agents from database")
        except Exception as e:
            logger.error(f"Failed to load agents from database: {e}")
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str], config: Dict[str, Any] = None):
        """Register an agent in the system with persistent storage"""
        try:
            # Register in persistent storage first
            success = self.persistent_registry.register_agent(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                config=config
            )
            
            if success:
                # Update in-memory cache for backward compatibility
                self.agents[agent_id] = {
                    'id': agent_id,
                    'type': agent_type,
                    'capabilities': capabilities,
                    'config': config or {},
                    'status': 'active',
                    'registered_at': datetime.now().isoformat()
                }
                
                # Index capabilities
                for capability in capabilities:
                    if capability not in self.capabilities:
                        self.capabilities[capability] = []
                    if agent_id not in self.capabilities[capability]:
                        self.capabilities[capability].append(agent_id)
                
                logger.info(f"Agent {agent_id} registered successfully")
            else:
                logger.error(f"Failed to register agent {agent_id} in persistent storage")
                
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
            # Fall back to in-memory only
            self.agents[agent_id] = {
                'id': agent_id,
                'type': agent_type,
                'capabilities': capabilities,
                'config': config or {},
                'status': 'active',
                'registered_at': datetime.now().isoformat()
            }
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information by ID"""
        # Try persistent storage first (with caching)
        try:
            agent_data = self.persistent_registry.get_agent(agent_id)
            if agent_data:
                return agent_data
        except Exception as e:
            logger.warning(f"Failed to get agent from persistent storage: {e}")
        
        # Fall back to in-memory
        return self.agents.get(agent_id)
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find all agents with a specific capability"""
        # Try persistent storage first (with caching)
        try:
            agent_ids = self.persistent_registry.find_agents_by_capability(capability)
            if agent_ids:
                return agent_ids
        except Exception as e:
            logger.warning(f"Failed to find agents by capability from persistent storage: {e}")
        
        # Fall back to in-memory
        return self.capabilities.get(capability, [])
    
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered agents"""
        # Try persistent storage first
        try:
            db_agents = self.persistent_registry.get_all_agents()
            if db_agents:
                return db_agents
        except Exception as e:
            logger.warning(f"Failed to get all agents from persistent storage: {e}")
        
        # Fall back to in-memory
        return self.agents
    
    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        try:
            # Update in persistent storage
            success = self.persistent_registry.update_agent_status(agent_id, status)
            
            if success:
                # Update in-memory cache
                if agent_id in self.agents:
                    self.agents[agent_id]['status'] = status
                    self.agents[agent_id]['last_updated'] = datetime.now().isoformat()
            else:
                logger.warning(f"Failed to update agent status in persistent storage")
                
        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
            # Fall back to in-memory only
            if agent_id in self.agents:
                self.agents[agent_id]['status'] = status
                self.agents[agent_id]['last_updated'] = datetime.now().isoformat()
    
    def establish_connection(self, agent1_id: str, agent2_id: str, protocol: str):
        """Establish connection between two agents"""
        try:
            # Store in persistent storage
            success = self.persistent_registry.establish_connection(agent1_id, agent2_id, protocol)
            
            if success:
                # Update in-memory cache
                connection_id = f"{agent1_id}-{agent2_id}"
                self.connections[connection_id] = {
                    'agents': [agent1_id, agent2_id],
                    'protocol': protocol,
                    'established_at': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Failed to establish connection in persistent storage")
                
        except Exception as e:
            logger.error(f"Error establishing connection: {e}")
            # Fall back to in-memory only
            connection_id = f"{agent1_id}-{agent2_id}"
            self.connections[connection_id] = {
                'agents': [agent1_id, agent2_id],
                'protocol': protocol,
                'established_at': datetime.now().isoformat()
            }
    
    def get_agent_connections(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all connections for an agent"""
        try:
            # Try persistent storage first
            connections = self.persistent_registry.get_agent_connections(agent_id)
            if connections:
                return connections
        except Exception as e:
            logger.warning(f"Failed to get connections from persistent storage: {e}")
        
        # Fall back to in-memory
        connections = []
        for conn_id, conn_info in self.connections.items():
            if agent_id in conn_info['agents']:
                connections.append(conn_info)
        return connections

# Global registry instance
agent_registry = AgentRegistry()

# Pre-register core agents
agent_registry.register_agent(
    'transform-001',
    'transform',
    ['data_preprocessing', 'format_conversion', 'data_validation'],
    {'version': '1.0'}
)

agent_registry.register_agent(
    'illuminate-001',
    'illuminate',
    ['market_analysis', 'pattern_recognition', 'insight_generation'],
    {'version': '1.0'}
)

agent_registry.register_agent(
    'execute-001',
    'execute',
    ['trade_execution', 'order_management', 'risk_control'],
    {'version': '1.0'}
)