"""
Enhanced A2A Agent Registry and Discovery System
Works with Agent Manager to provide comprehensive agent management
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ...core.protocols.a2a.a2a_protocol import A2A_CAPABILITIES, AgentStatus, MessageType
from ...data.database.cache import cache_manager
from ...data.database.client import get_db
from ...data.database.models import A2AAgent, A2AConnection

logger = logging.getLogger(__name__)


class AgentDiscoveryFilter(Enum):
    """Agent discovery filter types"""

    BY_CAPABILITY = "capability"
    BY_AGENT_TYPE = "agent_type"
    BY_MCP_TOOL = "mcp_tool"
    BY_SKILL_CARD = "skill_card"
    BY_STATUS = "status"


@dataclass
class AgentHealthStatus:
    """Agent health monitoring status"""

    agent_id: str
    status: AgentStatus
    last_heartbeat: datetime
    response_time_ms: float
    error_count: int = 0
    uptime_percentage: float = 100.0


class EnhancedA2ARegistry:
    """Enhanced A2A Agent Registry with comprehensive discovery and management"""

    def __init__(self):
        self.db = get_db()
        self.cache = cache_manager
        self.health_status: Dict[str, AgentHealthStatus] = {}
        self._initialize_core_agents()

    def _initialize_core_agents(self):
        """Initialize core A2A agents from protocol definition"""
        logger.info("Initializing core A2A agents from protocol")

        # Register agents from A2A_CAPABILITIES
        for agent_id, capabilities in A2A_CAPABILITIES.items():
            self._register_core_agent(agent_id, capabilities)

    def _register_core_agent(self, agent_id: str, capabilities: List[str]):
        """Register a core agent in the database"""
        try:
            with self.db.get_session() as session:
                existing = session.query(A2AAgent).filter(A2AAgent.agent_id == agent_id).first()

                if not existing:
                    agent = A2AAgent(
                        agent_id=agent_id,
                        agent_type=self._infer_agent_type(agent_id),
                        capabilities=json.dumps(capabilities),
                        metadata=json.dumps(
                            {
                                "core_agent": True,
                                "auto_registered": True,
                                "mcp_tools": self._get_default_mcp_tools(agent_id),
                            }
                        ),
                        status="active",
                        created_at=datetime.utcnow(),
                        last_updated=datetime.utcnow(),
                    )
                    session.add(agent)
                    session.commit()
                    logger.info(f"Registered core agent: {agent_id}")

        except Exception as e:
            logger.error(f"Failed to register core agent {agent_id}: {e}")

    def _infer_agent_type(self, agent_id: str) -> str:
        """Infer agent type from agent ID"""
        if "historical-loader" in agent_id:
            return "historical_loader"
        elif "database" in agent_id:
            return "database_manager"
        elif "transform" in agent_id:
            return "data_transformer"
        elif "illuminate" in agent_id:
            return "market_analyzer"
        elif "execute" in agent_id:
            return "trade_executor"
        elif "strands-glean" in agent_id:
            return "strands_agent"
        elif "mcts" in agent_id:
            return "mcts_calculation"
        else:
            return "unknown"

    def _get_default_mcp_tools(self, agent_id: str) -> List[str]:
        """Get default MCP tools for agent"""
        tool_mappings = {
            "historical-loader-001": [
                "get_market_data",
                "get_historical_prices",
                "calculate_technical_indicators",
            ],
            "database-001": ["get_portfolio", "get_wallet_balance", "store_transaction_history"],
            "transform-001": ["preprocess_data", "validate_data", "calculate_features"],
            "illuminate-001": ["analyze_market_sentiment", "get_price_alerts", "generate_insights"],
            "execute-001": ["execute_trade", "manage_orders", "assess_risk"],
        }
        return tool_mappings.get(agent_id, [])

    async def discover_agents(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Discover agents based on filters"""
        try:
            with self.db.get_session() as session:
                query = session.query(A2AAgent).filter(A2AAgent.is_active == True)

                # Apply filters
                if filters:
                    if "agent_type" in filters:
                        query = query.filter(A2AAgent.agent_type == filters["agent_type"])

                    if "capability" in filters:
                        query = query.filter(
                            A2AAgent.capabilities.contains(f'"{filters["capability"]}"')
                        )

                    if "status" in filters:
                        query = query.filter(A2AAgent.status == filters["status"])

                agents = query.all()

                result = []
                for agent in agents:
                    agent_data = {
                        "agent_id": agent.agent_id,
                        "agent_type": agent.agent_type,
                        "capabilities": json.loads(agent.capabilities)
                        if agent.capabilities
                        else [],
                        "metadata": json.loads(agent.metadata) if agent.metadata else {},
                        "status": agent.status,
                        "created_at": agent.created_at.isoformat(),
                        "last_updated": agent.last_updated.isoformat(),
                        "health_status": self._get_agent_health(agent.agent_id),
                    }
                    result.append(agent_data)

                return {
                    "success": True,
                    "agents": result,
                    "total_count": len(result),
                    "filters_applied": filters or {},
                }

        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Get agent health status"""
        if agent_id in self.health_status:
            health = self.health_status[agent_id]
            return {
                "status": health.status.value,
                "last_heartbeat": health.last_heartbeat.isoformat(),
                "response_time_ms": health.response_time_ms,
                "error_count": health.error_count,
                "uptime_percentage": health.uptime_percentage,
            }
        else:
            return {
                "status": "unknown",
                "last_heartbeat": None,
                "response_time_ms": 0,
                "error_count": 0,
                "uptime_percentage": 0,
            }

    async def get_agent_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Find agents by specific capability"""
        return await self.discover_agents({"capability": capability})

    async def get_agents_by_mcp_tool(self, tool_name: str) -> List[Dict[str, Any]]:
        """Find agents that own specific MCP tool"""
        try:
            with self.db.get_session() as session:
                agents = session.query(A2AAgent).filter(A2AAgent.is_active == True).all()

                result = []
                for agent in agents:
                    metadata = json.loads(agent.metadata) if agent.metadata else {}
                    mcp_tools = metadata.get("mcp_tools", [])

                    if tool_name in mcp_tools:
                        result.append(
                            {
                                "agent_id": agent.agent_id,
                                "agent_type": agent.agent_type,
                                "mcp_tools": mcp_tools,
                            }
                        )

                return result

        except Exception as e:
            logger.error(f"MCP tool discovery failed: {e}")
            return []

    async def validate_agent_boundaries(self) -> Dict[str, Any]:
        """Validate agent boundaries and detect conflicts"""
        violations = []
        tool_ownership = {}

        try:
            with self.db.get_session() as session:
                agents = session.query(A2AAgent).filter(A2AAgent.is_active == True).all()

                # Check for MCP tool conflicts
                for agent in agents:
                    metadata = json.loads(agent.metadata) if agent.metadata else {}
                    mcp_tools = metadata.get("mcp_tools", [])

                    for tool in mcp_tools:
                        if tool in tool_ownership:
                            violations.append(
                                {
                                    "type": "mcp_tool_conflict",
                                    "tool": tool,
                                    "agents": [tool_ownership[tool], agent.agent_id],
                                    "severity": "high",
                                }
                            )
                        else:
                            tool_ownership[tool] = agent.agent_id

                # Check for capability overlaps
                capability_map = {}
                for agent in agents:
                    capabilities = json.loads(agent.capabilities) if agent.capabilities else []
                    for cap in capabilities:
                        if cap not in capability_map:
                            capability_map[cap] = []
                        capability_map[cap].append(agent.agent_id)

                # Flag suspicious overlaps
                for cap, agent_list in capability_map.items():
                    if len(agent_list) > 2:  # More than 2 agents with same capability
                        violations.append(
                            {
                                "type": "capability_overlap",
                                "capability": cap,
                                "agents": agent_list,
                                "severity": "medium",
                            }
                        )

                return {
                    "success": True,
                    "validation_results": {
                        "total_agents": len(agents),
                        "violations": violations,
                        "boundaries_healthy": len(violations) == 0,
                        "tool_ownership_map": tool_ownership,
                    },
                }

        except Exception as e:
            logger.error(f"Boundary validation failed: {e}")
            return {"success": False, "error": str(e)}

    async def update_agent_health(self, agent_id: str, health_data: Dict[str, Any]):
        """Update agent health status"""
        try:
            health_status = AgentHealthStatus(
                agent_id=agent_id,
                status=AgentStatus(health_data.get("status", "active")),
                last_heartbeat=datetime.utcnow(),
                response_time_ms=health_data.get("response_time_ms", 0),
                error_count=health_data.get("error_count", 0),
                uptime_percentage=health_data.get("uptime_percentage", 100.0),
            )

            self.health_status[agent_id] = health_status

            # Update database
            with self.db.get_session() as session:
                agent = session.query(A2AAgent).filter(A2AAgent.agent_id == agent_id).first()

                if agent:
                    agent.status = health_data.get("status", "active")
                    agent.last_updated = datetime.utcnow()
                    session.commit()

            logger.debug(f"Updated health for agent {agent_id}")

        except Exception as e:
            logger.error(f"Health update failed for {agent_id}: {e}")

    async def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        try:
            with self.db.get_session() as session:
                total_agents = session.query(A2AAgent).count()
                active_agents = session.query(A2AAgent).filter(A2AAgent.is_active == True).count()

                # Agent type distribution
                type_distribution = {}
                agents = session.query(A2AAgent).all()
                for agent in agents:
                    agent_type = agent.agent_type
                    type_distribution[agent_type] = type_distribution.get(agent_type, 0) + 1

                # Capability coverage
                all_capabilities = set()
                for agent in agents:
                    caps = json.loads(agent.capabilities) if agent.capabilities else []
                    all_capabilities.update(caps)

                return {
                    "total_agents": total_agents,
                    "active_agents": active_agents,
                    "inactive_agents": total_agents - active_agents,
                    "agent_type_distribution": type_distribution,
                    "total_capabilities": len(all_capabilities),
                    "capability_list": list(all_capabilities),
                    "healthy_agents": len(
                        [h for h in self.health_status.values() if h.status == AgentStatus.ACTIVE]
                    ),
                    "last_updated": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(f"Statistics generation failed: {e}")
            return {"error": str(e)}


# Global registry instance
_registry_instance = None


def get_enhanced_a2a_registry() -> EnhancedA2ARegistry:
    """Get singleton registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = EnhancedA2ARegistry()
    return _registry_instance
