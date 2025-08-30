"""
Agent Discovery Service using Blockchain
Enables agents to discover each other through on-chain registry
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from .blockchain_registration import BlockchainRegistrationService, BlockchainAgentRegistration
from .a2a_protocol import AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredAgent:
    """Information about a discovered agent"""
    
    agent_id: str
    agent_type: str
    capabilities: List[str]
    mcp_tools: List[str]
    wallet_address: str
    status: str
    last_seen: datetime
    compliance_score: int
    endpoint: Optional[str] = None


class AgentDiscoveryService:
    """Service for discovering A2A agents on the blockchain"""
    
    def __init__(self, blockchain_service: BlockchainRegistrationService):
        """Initialize agent discovery service"""
        self.blockchain_service = blockchain_service
        self.discovered_agents: Dict[str, DiscoveredAgent] = {}
        self.discovery_cache_ttl = timedelta(minutes=5)  # Cache TTL
        self.last_discovery_scan = datetime.min
        
        logger.info("Agent Discovery Service initialized")
    
    async def discover_all_agents(self, force_refresh: bool = False) -> Dict[str, DiscoveredAgent]:
        """Discover all agents registered on blockchain"""
        
        # Check cache
        if not force_refresh and datetime.now() - self.last_discovery_scan < self.discovery_cache_ttl:
            logger.debug("Returning cached agent discovery results")
            return self.discovered_agents
        
        try:
            # Get all registered agent IDs from blockchain
            agent_ids = await self.blockchain_service.get_all_registered_agents()
            logger.info(f"Found {len(agent_ids)} agents on blockchain")
            
            # Fetch details for each agent
            discovered = {}
            
            for agent_id in agent_ids:
                try:
                    registration = await self.blockchain_service.get_agent_registration(agent_id)
                    if registration:
                        discovered_agent = DiscoveredAgent(
                            agent_id=registration.agent_id,
                            agent_type=registration.agent_type,
                            capabilities=registration.capabilities,
                            mcp_tools=registration.mcp_tools,
                            wallet_address=registration.wallet_address,
                            status=registration.status,
                            last_seen=datetime.now(),
                            compliance_score=registration.compliance_score
                        )
                        
                        discovered[agent_id] = discovered_agent
                        logger.debug(f"Discovered agent: {agent_id} ({registration.agent_type})")
                    
                except Exception as e:
                    logger.warning(f"Failed to get details for agent {agent_id}: {e}")
                    continue
            
            # Update cache
            self.discovered_agents = discovered
            self.last_discovery_scan = datetime.now()
            
            logger.info(f"✅ Agent discovery complete: {len(discovered)} agents discovered")
            return discovered
            
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
            return self.discovered_agents  # Return cached results on error
    
    async def find_agents_by_capability(self, capability: str) -> List[DiscoveredAgent]:
        """Find agents that have a specific capability"""
        agents = await self.discover_all_agents()
        
        matching_agents = []
        for agent in agents.values():
            if capability in agent.capabilities:
                matching_agents.append(agent)
        
        logger.info(f"Found {len(matching_agents)} agents with capability '{capability}'")
        return matching_agents
    
    async def find_agents_by_type(self, agent_type: str) -> List[DiscoveredAgent]:
        """Find agents of a specific type"""
        agents = await self.discover_all_agents()
        
        matching_agents = []
        for agent in agents.values():
            if agent.agent_type == agent_type:
                matching_agents.append(agent)
        
        logger.info(f"Found {len(matching_agents)} agents of type '{agent_type}'")
        return matching_agents
    
    async def find_agents_with_mcp_tool(self, tool_name: str) -> List[DiscoveredAgent]:
        """Find agents that support a specific MCP tool"""
        agents = await self.discover_all_agents()
        
        matching_agents = []
        for agent in agents.values():
            if tool_name in agent.mcp_tools:
                matching_agents.append(agent)
        
        logger.info(f"Found {len(matching_agents)} agents with MCP tool '{tool_name}'")
        return matching_agents
    
    async def get_agent_details(self, agent_id: str) -> Optional[DiscoveredAgent]:
        """Get detailed information about a specific agent"""
        agents = await self.discover_all_agents()
        return agents.get(agent_id)
    
    async def is_agent_available(self, agent_id: str) -> bool:
        """Check if an agent is available for communication"""
        try:
            return await self.blockchain_service.is_agent_active(agent_id)
        except Exception as e:
            logger.error(f"Failed to check availability for {agent_id}: {e}")
            return False
    
    async def get_compatible_agents(
        self, 
        required_capabilities: List[str], 
        required_tools: List[str] = None,
        agent_type: Optional[str] = None
    ) -> List[DiscoveredAgent]:
        """Find agents that match specified requirements"""
        agents = await self.discover_all_agents()
        required_tools = required_tools or []
        
        compatible_agents = []
        
        for agent in agents.values():
            # Check agent type if specified
            if agent_type and agent.agent_type != agent_type:
                continue
            
            # Check capabilities
            has_all_capabilities = all(cap in agent.capabilities for cap in required_capabilities)
            if not has_all_capabilities:
                continue
            
            # Check MCP tools
            has_all_tools = all(tool in agent.mcp_tools for tool in required_tools)
            if not has_all_tools:
                continue
            
            # Check if agent is active
            if agent.status != "ACTIVE":
                continue
            
            compatible_agents.append(agent)
        
        logger.info(f"Found {len(compatible_agents)} compatible agents for requirements")
        return compatible_agents
    
    async def get_agent_network_statistics(self) -> Dict[str, any]:
        """Get statistics about the agent network"""
        agents = await self.discover_all_agents()
        
        if not agents:
            return {"total_agents": 0}
        
        # Count by type
        type_counts = {}
        for agent in agents.values():
            type_counts[agent.agent_type] = type_counts.get(agent.agent_type, 0) + 1
        
        # Count by status
        status_counts = {}
        for agent in agents.values():
            status_counts[agent.status] = status_counts.get(agent.status, 0) + 1
        
        # Collect all capabilities
        all_capabilities = set()
        for agent in agents.values():
            all_capabilities.update(agent.capabilities)
        
        # Collect all MCP tools
        all_mcp_tools = set()
        for agent in agents.values():
            all_mcp_tools.update(agent.mcp_tools)
        
        # Average compliance score
        avg_compliance = sum(agent.compliance_score for agent in agents.values()) / len(agents)
        
        statistics = {
            "total_agents": len(agents),
            "agent_types": type_counts,
            "agent_statuses": status_counts,
            "unique_capabilities": len(all_capabilities),
            "all_capabilities": list(all_capabilities),
            "unique_mcp_tools": len(all_mcp_tools),
            "all_mcp_tools": list(all_mcp_tools),
            "average_compliance_score": round(avg_compliance, 2),
            "last_discovery_scan": self.last_discovery_scan.isoformat(),
            "cache_valid_until": (self.last_discovery_scan + self.discovery_cache_ttl).isoformat()
        }
        
        return statistics


class AgentRecommendationEngine:
    """Engine for recommending agents based on task requirements"""
    
    def __init__(self, discovery_service: AgentDiscoveryService):
        """Initialize recommendation engine"""
        self.discovery_service = discovery_service
        logger.info("Agent Recommendation Engine initialized")
    
    async def recommend_agents_for_task(
        self,
        task_description: str,
        required_capabilities: List[str],
        preferred_tools: List[str] = None,
        max_agents: int = 5
    ) -> List[tuple[DiscoveredAgent, float]]:
        """Recommend agents for a specific task with scoring"""
        
        preferred_tools = preferred_tools or []
        
        # Get all compatible agents
        compatible_agents = await self.discovery_service.get_compatible_agents(
            required_capabilities=required_capabilities,
            required_tools=preferred_tools
        )
        
        if not compatible_agents:
            logger.warning("No compatible agents found for task")
            return []
        
        # Score agents
        scored_agents = []
        for agent in compatible_agents:
            score = self._calculate_agent_score(
                agent, 
                required_capabilities, 
                preferred_tools,
                task_description
            )
            scored_agents.append((agent, score))
        
        # Sort by score (descending)
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Return top agents
        top_agents = scored_agents[:max_agents]
        
        logger.info(f"Recommended {len(top_agents)} agents for task")
        return top_agents
    
    def _calculate_agent_score(
        self,
        agent: DiscoveredAgent,
        required_capabilities: List[str],
        preferred_tools: List[str],
        task_description: str
    ) -> float:
        """Calculate a score for how well an agent matches task requirements"""
        
        score = 0.0
        
        # Base score from compliance
        score += agent.compliance_score / 100.0 * 0.3
        
        # Capability match bonus (higher for more capabilities)
        capability_bonus = len(agent.capabilities) / 10.0 * 0.2
        score += min(capability_bonus, 0.2)  # Cap at 0.2
        
        # MCP tools bonus
        tools_bonus = len(agent.mcp_tools) / 10.0 * 0.2  
        score += min(tools_bonus, 0.2)  # Cap at 0.2
        
        # Preferred tools match
        if preferred_tools:
            matching_tools = sum(1 for tool in preferred_tools if tool in agent.mcp_tools)
            tools_match_ratio = matching_tools / len(preferred_tools)
            score += tools_match_ratio * 0.3
        
        # Agent type bonus (simple keyword matching with task description)
        if agent.agent_type.lower() in task_description.lower():
            score += 0.1
        
        return min(score, 1.0)  # Cap score at 1.0


# Global discovery service instance
_discovery_service: Optional[AgentDiscoveryService] = None
_recommendation_engine: Optional[AgentRecommendationEngine] = None


def get_discovery_service() -> Optional[AgentDiscoveryService]:
    """Get the global discovery service"""
    return _discovery_service


def get_recommendation_engine() -> Optional[AgentRecommendationEngine]:
    """Get the global recommendation engine"""
    return _recommendation_engine


async def initialize_discovery_services(blockchain_service: BlockchainRegistrationService):
    """Initialize discovery services"""
    global _discovery_service, _recommendation_engine
    
    try:
        _discovery_service = AgentDiscoveryService(blockchain_service)
        _recommendation_engine = AgentRecommendationEngine(_discovery_service)
        
        # Perform initial discovery
        agents = await _discovery_service.discover_all_agents()
        logger.info(f"🔍 Agent discovery services initialized with {len(agents)} agents")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize discovery services: {e}")
        return False