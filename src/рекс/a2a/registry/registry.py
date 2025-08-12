"""
Agent Registry for рекс.com A2A System
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json

class AgentRegistry:
    """Central registry for all A2A agents"""
    
    def __init__(self):
        self.agents = {}
        self.capabilities = {}
        self.connections = {}
        
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str], config: Dict[str, Any] = None):
        """Register an agent in the system"""
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
            self.capabilities[capability].append(agent_id)
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information by ID"""
        return self.agents.get(agent_id)
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find all agents with a specific capability"""
        return self.capabilities.get(capability, [])
    
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered agents"""
        return self.agents
    
    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id]['status'] = status
            self.agents[agent_id]['last_updated'] = datetime.now().isoformat()
    
    def establish_connection(self, agent1_id: str, agent2_id: str, protocol: str):
        """Establish connection between two agents"""
        connection_id = f"{agent1_id}-{agent2_id}"
        self.connections[connection_id] = {
            'agents': [agent1_id, agent2_id],
            'protocol': protocol,
            'established_at': datetime.now().isoformat()
        }
    
    def get_agent_connections(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all connections for an agent"""
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