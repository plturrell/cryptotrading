"""
A2A Agent Registry Integration - Manages registration of MCP agents with A2A protocol
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from ...infrastructure.mcp.clrs_algorithms_mcp_agent import CLRSAlgorithmsMCPAgent
from ...infrastructure.mcp.code_quality_mcp_agent import CodeQualityMCPAgent
from ...infrastructure.mcp.data_analysis_mcp_agent import DataAnalysisMCPAgent
from ...infrastructure.mcp.feature_store_mcp_agent import FeatureStoreMCPAgent
from ...infrastructure.mcp.ml_models_mcp_agent import MLModelsMCPAgent
from ...infrastructure.mcp.technical_analysis_skills_mcp_agent import (
    TechnicalAnalysisSkillsMCPAgent,
)
from .a2a_protocol import A2A_CAPABILITIES, A2AAgentRegistry, MessageType

logger = logging.getLogger(__name__)


class A2AAgentRegistryIntegration:
    """Manages registration and lifecycle of MCP agents with A2A protocol"""

    def __init__(self):
        self.registered_agents: Dict[str, Any] = {}
        self.agent_instances: Dict[str, Any] = {}

    async def initialize_and_register_agents(self) -> bool:
        """Initialize and register all MCP agents with A2A protocol"""
        try:
            # Initialize all MCP agents
            agents_to_initialize = [
                ("feature_store_agent", FeatureStoreMCPAgent),
                ("data_analysis_agent", DataAnalysisMCPAgent),
                ("clrs_algorithms_agent", CLRSAlgorithmsMCPAgent),
                ("technical_analysis_skills_agent", TechnicalAnalysisSkillsMCPAgent),
                ("ml_models_agent", MLModelsMCPAgent),
                ("code_quality_agent", CodeQualityMCPAgent),
            ]

            for agent_id, agent_class in agents_to_initialize:
                agent_instance = agent_class(agent_id)
                await agent_instance.initialize()

                # Store agent instance
                self.agent_instances[agent_id] = agent_instance

                # Register with A2A protocol
                self._register_agent_with_a2a(agent_id, agent_instance)

            logger.info("All MCP agents initialized and registered with A2A protocol")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize and register agents: {str(e)}")
            return False

    def _register_agent_with_a2a(self, agent_id: str, agent_instance: Any) -> bool:
        """Register individual agent with A2A protocol"""
        try:
            # Get capabilities for this agent
            capabilities = A2A_CAPABILITIES.get(agent_id, [])

            if not capabilities:
                logger.warning(f"No capabilities defined for agent {agent_id}")
                return False

            # Register with A2A registry
            A2AAgentRegistry.register_agent(agent_id, capabilities, agent_instance)

            # Store registration info
            self.registered_agents[agent_id] = {
                "capabilities": capabilities,
                "status": "registered",
                "instance": agent_instance,
            }

            logger.info(
                f"Agent {agent_id} registered with A2A protocol with capabilities: {capabilities}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id} with A2A protocol: {str(e)}")
            return False

    async def start_all_agents(self) -> bool:
        """Start all registered agents"""
        try:
            for agent_id, agent_instance in self.agent_instances.items():
                if hasattr(agent_instance, "start"):
                    success = await agent_instance.start()
                    if success:
                        self.registered_agents[agent_id]["status"] = "running"
                        logger.info(f"Agent {agent_id} started successfully")
                    else:
                        logger.error(f"Failed to start agent {agent_id}")
                        return False

            logger.info("All agents started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start agents: {str(e)}")
            return False

    async def stop_all_agents(self) -> bool:
        """Stop all registered agents"""
        try:
            for agent_id, agent_instance in self.agent_instances.items():
                if hasattr(agent_instance, "stop"):
                    success = await agent_instance.stop()
                    if success:
                        self.registered_agents[agent_id]["status"] = "stopped"
                        logger.info(f"Agent {agent_id} stopped successfully")
                    else:
                        logger.error(f"Failed to stop agent {agent_id}")

            logger.info("All agents stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop agents: {str(e)}")
            return False

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific agent"""
        if agent_id in self.registered_agents:
            agent_info = self.registered_agents[agent_id].copy()
            if agent_id in self.agent_instances:
                instance = self.agent_instances[agent_id]
                if hasattr(instance, "get_status"):
                    agent_info.update(instance.get_status())
            return agent_info
        return None

    def get_all_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        status = {}
        for agent_id in self.registered_agents:
            status[agent_id] = self.get_agent_status(agent_id)
        return status

    def get_agents_by_capability(self, capability: str) -> List[str]:
        """Get list of agent IDs that have specified capability"""
        matching_agents = []
        for agent_id, agent_info in self.registered_agents.items():
            if capability in agent_info.get("capabilities", []):
                matching_agents.append(agent_id)
        return matching_agents

    async def route_message_to_agents(
        self, message_type: MessageType, message: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Route message to appropriate agents based on A2A routing"""
        try:
            from .a2a_protocol import A2A_ROUTING

            # Get target agents for this message type
            target_agents = A2A_ROUTING.get(message_type, [])

            results = []
            for agent_id in target_agents:
                if agent_id in self.agent_instances:
                    agent_instance = self.agent_instances[agent_id]
                    if hasattr(agent_instance, "process_message"):
                        result = await agent_instance.process_message(message)
                        results.append({"agent_id": agent_id, "result": result})
                    else:
                        logger.warning(f"Agent {agent_id} does not support message processing")
                else:
                    logger.warning(f"Agent {agent_id} not found in registered instances")

            return results

        except Exception as e:
            logger.error(f"Failed to route message to agents: {str(e)}")
            return []


# Global registry instance
agent_registry_integration = A2AAgentRegistryIntegration()


async def initialize_mcp_a2a_integration() -> bool:
    """Initialize MCP agents and register with A2A protocol"""
    return await agent_registry_integration.initialize_and_register_agents()


async def start_mcp_agents() -> bool:
    """Start all MCP agents"""
    return await agent_registry_integration.start_all_agents()


async def stop_mcp_agents() -> bool:
    """Stop all MCP agents"""
    return await agent_registry_integration.stop_all_agents()


def get_mcp_agent_status() -> Dict[str, Any]:
    """Get status of all MCP agents"""
    return agent_registry_integration.get_all_agent_status()
