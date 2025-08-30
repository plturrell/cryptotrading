"""
API routes for Strands Agent management
Provides REST endpoints for the Strands Agent UI integration
Connected to real agent registry system
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
import logging
import sys
import os
from datetime import datetime

# Add src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

try:
    from cryptotrading.infrastructure.registry.registry import agent_registry

    REAL_REGISTRY_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Connected to real agent registry")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import real registry, using mock data: {e}")
    REAL_REGISTRY_AVAILABLE = False

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/list")
async def list_agents() -> List[Dict[str, Any]]:
    """Get list of all registered agents"""
    logger.info("Fetching agent list")

    if not REAL_REGISTRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent registry service unavailable")

    try:
        # Get real agents from registry
        real_agents = agent_registry.get_all_agents()

        # Convert to UI format
        agent_list = []
        for agent_id, agent_data in real_agents.items():
            formatted_agent = {
                "agent_id": agent_id,
                "agent_type": agent_data.get("type", "unknown"),
                "status": agent_data.get("status", "active"),
                "capabilities": agent_data.get("capabilities", []),
                "model_provider": agent_data.get("config", {}).get("model_provider", "deepseek"),
                "created_at": agent_data.get("registered_at", datetime.now().isoformat()),
                "last_activity": agent_data.get("last_updated", datetime.now().isoformat()),
            }
            agent_list.append(formatted_agent)

        logger.info(f"Retrieved {len(agent_list)} real agents from registry")
        return agent_list

    except Exception as e:
        logger.error(f"Error fetching real agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch agents: {str(e)}")


@router.get("/status")
async def get_agent_status() -> Dict[str, Any]:
    """Get overall agent status summary"""

    if not REAL_REGISTRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent registry service unavailable")

    try:
        real_agents = agent_registry.get_all_agents()
        agent_statuses = [agent_data.get("status", "active") for agent_data in real_agents.values()]

        active_count = len([s for s in agent_statuses if s == "active"])
        inactive_count = len([s for s in agent_statuses if s == "inactive"])

        return {
            "total_agents": len(real_agents),
            "active_count": active_count,
            "inactive_count": inactive_count,
            "last_updated": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@router.get("/{agent_id}")
async def get_agent_details(agent_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific agent"""

    if not REAL_REGISTRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent registry service unavailable")

    try:
        agent_data = agent_registry.get_agent(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Format for UI
        formatted_agent = {
            "agent_id": agent_id,
            "agent_type": agent_data.get("type", "unknown"),
            "status": agent_data.get("status", "active"),
            "capabilities": agent_data.get("capabilities", []),
            "model_provider": agent_data.get("config", {}).get("model_provider", "deepseek"),
            "created_at": agent_data.get("registered_at", datetime.now().isoformat()),
            "last_activity": agent_data.get("last_updated", datetime.now().isoformat()),
            "config": agent_data.get("config", {}),
        }

        return formatted_agent

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent details: {str(e)}")


@router.post("/{agent_id}/start")
async def start_agent(agent_id: str) -> Dict[str, Any]:
    """Start an inactive agent"""

    if not REAL_REGISTRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent registry service unavailable")

    try:
        agent_data = agent_registry.get_agent(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        current_status = agent_data.get("status", "active")
        if current_status == "active":
            return {"message": f"Agent {agent_id} is already active"}

        # Update agent status to active
        agent_registry.update_agent_status(agent_id, "active")
        logger.info(f"Agent {agent_id} started")

        return {"message": f"Agent {agent_id} started successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start agent: {str(e)}")


@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: str) -> Dict[str, Any]:
    """Stop an active agent"""

    if not REAL_REGISTRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent registry service unavailable")

    try:
        agent_data = agent_registry.get_agent(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        current_status = agent_data.get("status", "active")
        if current_status == "inactive":
            return {"message": f"Agent {agent_id} is already inactive"}

        # Update agent status to inactive
        agent_registry.update_agent_status(agent_id, "inactive")
        logger.info(f"Agent {agent_id} stopped")

        return {"message": f"Agent {agent_id} stopped successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop agent: {str(e)}")


@router.post("/create")
async def create_agent(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new agent"""

    if not REAL_REGISTRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent registry service unavailable")

    required_fields = ["agent_id", "agent_type"]

    for field in required_fields:
        if field not in agent_data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    agent_id = agent_data["agent_id"]

    try:
        # Check if agent already exists
        existing_agent = agent_registry.get_agent(agent_id)
        if existing_agent:
            raise HTTPException(status_code=409, detail=f"Agent {agent_id} already exists")

        # Get capabilities and config
        capabilities = _get_default_capabilities(agent_data["agent_type"])
        config = {"model_provider": agent_data.get("model_provider", "deepseek"), "version": "1.0"}

        # Register new agent
        agent_registry.register_agent(
            agent_id=agent_id,
            agent_type=agent_data["agent_type"],
            capabilities=capabilities,
            config=config,
        )

        logger.info(f"Agent {agent_id} created successfully")

        return {
            "message": f"Agent {agent_id} created successfully",
            "agent": {
                "agent_id": agent_id,
                "agent_type": agent_data["agent_type"],
                "capabilities": capabilities,
                "model_provider": config["model_provider"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    """Delete an agent"""

    if not REAL_REGISTRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent registry service unavailable")

    try:
        agent_data = agent_registry.get_agent(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Note: The current registry doesn't have a delete method
        # For now, we'll mark it as inactive
        agent_registry.update_agent_status(agent_id, "inactive")
        logger.info(f"Agent {agent_id} deactivated (delete not implemented)")

        return {"message": f"Agent {agent_id} deactivated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")


def _get_default_capabilities(agent_type: str) -> List[str]:
    """Get default capabilities for agent type"""
    capability_map = {
        "historical_loader": ["data_loading", "historical_analysis"],
        "database": ["data_storage", "query_processing"],
        "blockchain_strands": ["blockchain_analysis", "transaction_monitoring"],
        "data_management": ["data_validation", "cache_management"],
        "a2a_coordinator": ["agent_coordination", "workflow_management"],
    }

    return capability_map.get(agent_type, ["basic_processing"])
