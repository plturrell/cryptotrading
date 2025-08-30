"""
Agent Registry - STRANDS Integration
Central registry for all STRANDS agents including new MCP-backed agents
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from ..base_agent import BaseAgent
from ..specialized.data_analysis_agent import data_analysis_agent
from ..specialized.feature_store_agent import feature_store_agent
from ..specialized.glean_agent import GleanAgent
from ..specialized.mcts_calculation_agent import MCTSCalculationAgent
from ..specialized.ml_agent import MLAgent
from ..specialized.technical_analysis.technical_analysis_agent import TechnicalAnalysisAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for all STRANDS agents"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_status: Dict[str, str] = {}

        # Register built-in agent types
        self._register_builtin_agents()

        logger.info("Agent Registry initialized")

    def _register_builtin_agents(self):
        """Register built-in agent types"""
        self.agent_types.update(
            {
                "technical_analysis": TechnicalAnalysisAgent,
                "ml_agent": MLAgent,
                "glean_agent": GleanAgent,
                "mcts_calculation": MCTSCalculationAgent,
                "feature_store": type(feature_store_agent),
                "data_analysis": type(data_analysis_agent),
            }
        )

        # Register singleton instances
        self.register_agent(feature_store_agent)
        self.register_agent(data_analysis_agent)

        logger.info(f"Registered {len(self.agent_types)} built-in agent types")

    def register_agent_type(self, agent_type: str, agent_class: Type[BaseAgent]):
        """Register a new agent type"""
        self.agent_types[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")

    def register_agent(self, agent: BaseAgent):
        """Register an agent instance"""
        self.agents[agent.agent_id] = agent
        self.agent_capabilities[agent.agent_id] = agent.config.capabilities
        self.agent_status[agent.agent_id] = "registered"

        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")

    def create_agent(self, agent_type: str, agent_id: str, **kwargs) -> Optional[BaseAgent]:
        """Create a new agent instance"""
        try:
            if agent_type not in self.agent_types:
                logger.error(f"Unknown agent type: {agent_type}")
                return None

            agent_class = self.agent_types[agent_type]
            agent = agent_class(agent_id=agent_id, **kwargs)

            self.register_agent(agent)
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent {agent_id}: {e}")
            return None

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        return [agent for agent in self.agents.values() if agent.agent_type == agent_type]

    def get_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Get all agents with a specific capability"""
        matching_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if capability in capabilities:
                agent = self.agents.get(agent_id)
                if agent:
                    matching_agents.append(agent)
        return matching_agents

    async def initialize_all_agents(self) -> Dict[str, bool]:
        """Initialize all registered agents"""
        results = {}

        for agent_id, agent in self.agents.items():
            try:
                logger.info(f"Initializing agent: {agent_id}")
                success = await agent.initialize()
                results[agent_id] = success
                self.agent_status[agent_id] = "initialized" if success else "failed"

            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_id}: {e}")
                results[agent_id] = False
                self.agent_status[agent_id] = "failed"

        successful = sum(1 for success in results.values() if success)
        logger.info(f"Initialized {successful}/{len(results)} agents successfully")

        return results

    async def start_all_agents(self) -> Dict[str, bool]:
        """Start all initialized agents"""
        results = {}

        for agent_id, agent in self.agents.items():
            if self.agent_status.get(agent_id) != "initialized":
                logger.warning(f"Skipping start for non-initialized agent: {agent_id}")
                results[agent_id] = False
                continue

            try:
                logger.info(f"Starting agent: {agent_id}")
                success = await agent.start()
                results[agent_id] = success
                self.agent_status[agent_id] = "running" if success else "failed"

            except Exception as e:
                logger.error(f"Failed to start agent {agent_id}: {e}")
                results[agent_id] = False
                self.agent_status[agent_id] = "failed"

        successful = sum(1 for success in results.values() if success)
        logger.info(f"Started {successful}/{len(results)} agents successfully")

        return results

    async def stop_all_agents(self) -> Dict[str, bool]:
        """Stop all running agents"""
        results = {}

        for agent_id, agent in self.agents.items():
            try:
                if hasattr(agent, "stop"):
                    logger.info(f"Stopping agent: {agent_id}")
                    success = await agent.stop()
                    results[agent_id] = success
                    self.agent_status[agent_id] = "stopped" if success else "failed"
                else:
                    results[agent_id] = True
                    self.agent_status[agent_id] = "stopped"

            except Exception as e:
                logger.error(f"Failed to stop agent {agent_id}: {e}")
                results[agent_id] = False
                self.agent_status[agent_id] = "failed"

        return results

    def get_agent_status(self, agent_id: str = None) -> Dict[str, Any]:
        """Get status of agents"""
        if agent_id:
            agent = self.agents.get(agent_id)
            if not agent:
                return {"error": f"Agent {agent_id} not found"}

            return {
                "agent_id": agent_id,
                "agent_type": agent.agent_type,
                "status": self.agent_status.get(agent_id, "unknown"),
                "capabilities": self.agent_capabilities.get(agent_id, []),
                "config": agent.config.__dict__ if hasattr(agent, "config") else {},
            }

        # Return all agents status
        return {
            "agents": {
                agent_id: {
                    "agent_type": agent.agent_type,
                    "status": self.agent_status.get(agent_id, "unknown"),
                    "capabilities": self.agent_capabilities.get(agent_id, []),
                }
                for agent_id, agent in self.agents.items()
            },
            "summary": {
                "total_agents": len(self.agents),
                "running": sum(1 for status in self.agent_status.values() if status == "running"),
                "initialized": sum(
                    1 for status in self.agent_status.values() if status == "initialized"
                ),
                "failed": sum(1 for status in self.agent_status.values() if status == "failed"),
            },
        }

    def get_available_capabilities(self) -> List[str]:
        """Get all available capabilities across all agents"""
        all_capabilities = set()
        for capabilities in self.agent_capabilities.values():
            all_capabilities.update(capabilities)
        return sorted(list(all_capabilities))

    async def route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route message to appropriate agent based on capabilities"""
        try:
            required_capability = message.get("capability")
            target_agent_id = message.get("agent_id")

            # Direct routing to specific agent
            if target_agent_id:
                agent = self.get_agent(target_agent_id)
                if agent:
                    return await agent.process_message(message)
                else:
                    return {"error": f"Agent {target_agent_id} not found"}

            # Route by capability
            if required_capability:
                capable_agents = self.get_agents_by_capability(required_capability)
                if capable_agents:
                    # Use first available agent (could implement load balancing)
                    agent = capable_agents[0]
                    return await agent.process_message(message)
                else:
                    return {"error": f"No agents available with capability: {required_capability}"}

            return {"error": "No routing criteria specified (agent_id or capability)"}

        except Exception as e:
            logger.error(f"Error routing message: {e}")
            return {"error": str(e)}


# Global registry instance
agent_registry = AgentRegistry()


# Convenience functions
def get_agent(agent_id: str) -> Optional[BaseAgent]:
    """Get agent by ID"""
    return agent_registry.get_agent(agent_id)


def get_agents_by_capability(capability: str) -> List[BaseAgent]:
    """Get agents by capability"""
    return agent_registry.get_agents_by_capability(capability)


def create_agent(agent_type: str, agent_id: str, **kwargs) -> Optional[BaseAgent]:
    """Create new agent"""
    return agent_registry.create_agent(agent_type, agent_id, **kwargs)


async def initialize_all_agents() -> Dict[str, bool]:
    """Initialize all agents"""
    return await agent_registry.initialize_all_agents()


async def start_all_agents() -> Dict[str, bool]:
    """Start all agents"""
    return await agent_registry.start_all_agents()


def get_agent_status(agent_id: str = None) -> Dict[str, Any]:
    """Get agent status"""
    return agent_registry.get_agent_status(agent_id)
