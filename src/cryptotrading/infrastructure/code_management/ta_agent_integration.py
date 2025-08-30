"""
Technical Analysis Agent Integration
Integration module for adding TA agent to the enterprise orchestrator
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.agents.specialized.technical_analysis import (
    TechnicalAnalysisAgent,
    create_technical_analysis_agent,
)
from .thread_safe_orchestrator import ThreadSafeEnterpriseOrchestrator

logger = logging.getLogger(__name__)


class TAAgentIntegration:
    """Integration class for Technical Analysis agent in enterprise orchestrator"""

    def __init__(self, orchestrator: ThreadSafeEnterpriseOrchestrator):
        self.orchestrator = orchestrator
        self.ta_agent: Optional[TechnicalAnalysisAgent] = None
        self.integration_status = "not_initialized"

    async def initialize_ta_agent(self) -> bool:
        """Initialize and register TA agent with orchestrator"""
        try:
            logger.info("Initializing Technical Analysis agent for enterprise orchestrator")

            # Create TA agent
            self.ta_agent = create_technical_analysis_agent("enterprise_ta_agent")

            # Register agent with orchestrator's agent registry if available
            if hasattr(self.orchestrator, "agent_registry"):
                self.orchestrator.agent_registry["technical_analysis"] = self.ta_agent

            # Add TA-specific monitoring tasks
            await self._register_ta_monitoring_tasks()

            self.integration_status = "initialized"
            logger.info("Technical Analysis agent successfully integrated")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize TA agent: {e}")
            self.integration_status = "failed"
            return False

    async def _register_ta_monitoring_tasks(self):
        """Register TA-specific monitoring and health check tasks"""
        try:
            # Add TA agent health monitoring
            if hasattr(self.orchestrator, "add_monitoring_task"):
                await self.orchestrator.add_monitoring_task(
                    "ta_agent_health", self._check_ta_agent_health, interval=300  # 5 minutes
                )

            # Add TA skill status monitoring
            if hasattr(self.orchestrator, "add_monitoring_task"):
                await self.orchestrator.add_monitoring_task(
                    "ta_skills_status", self._monitor_ta_skills, interval=600  # 10 minutes
                )

        except Exception as e:
            logger.error(f"Failed to register TA monitoring tasks: {e}")

    async def _check_ta_agent_health(self) -> Dict[str, Any]:
        """Health check for TA agent"""
        if not self.ta_agent:
            return {
                "status": "unhealthy",
                "reason": "TA agent not initialized",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # Check agent status
            skill_status = await self.ta_agent.get_skill_status()

            # Check if all skills are functional
            failed_skills = [
                skill
                for skill, state in skill_status["skill_states"].items()
                if not state["enabled"]
            ]

            # Check circuit breaker status
            open_breakers = [
                name for name, is_open in skill_status["circuit_breaker_status"].items() if is_open
            ]

            health_status = "healthy"
            issues = []

            if failed_skills:
                health_status = "degraded"
                issues.append(f"Disabled skills: {', '.join(failed_skills)}")

            if open_breakers:
                health_status = "degraded"
                issues.append(f"Open circuit breakers: {', '.join(open_breakers)}")

            return {
                "status": health_status,
                "total_tools": skill_status["total_tools"],
                "failed_skills": failed_skills,
                "open_breakers": open_breakers,
                "issues": issues,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }

    async def _monitor_ta_skills(self) -> Dict[str, Any]:
        """Monitor TA skills performance and usage"""
        if not self.ta_agent:
            return {"status": "no_agent"}

        try:
            skill_status = await self.ta_agent.get_skill_status()

            # Calculate skill usage metrics
            now = datetime.now()
            skill_metrics = {}

            for skill, state in skill_status["skill_states"].items():
                last_used = state.get("last_used")
                if last_used:
                    time_since_use = (now - last_used).total_seconds()
                    skill_metrics[skill] = {
                        "last_used_seconds_ago": time_since_use,
                        "enabled": state["enabled"],
                        "status": "active" if time_since_use < 3600 else "idle",
                    }
                else:
                    skill_metrics[skill] = {
                        "last_used_seconds_ago": None,
                        "enabled": state["enabled"],
                        "status": "unused",
                    }

            return {
                "status": "monitored",
                "skill_metrics": skill_metrics,
                "tools_by_skill": skill_status["tools_by_skill"],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"TA skills monitoring failed: {e}")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    async def get_ta_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive TA agent status"""
        return {
            "integration_status": self.integration_status,
            "agent_initialized": self.ta_agent is not None,
            "agent_id": self.ta_agent.agent_id if self.ta_agent else None,
            "health_check": await self._check_ta_agent_health(),
            "skills_status": await self._monitor_ta_skills(),
        }

    async def execute_ta_analysis(
        self, market_data, analysis_type: str = "comprehensive", risk_tolerance: str = "medium"
    ) -> Dict[str, Any]:
        """Execute TA analysis through the integrated agent"""
        if not self.ta_agent:
            return {"success": False, "error": "TA agent not initialized"}

        try:
            result = await self.ta_agent.analyze_market_data(
                data=market_data, analysis_type=analysis_type, risk_tolerance=risk_tolerance
            )

            # Log analysis execution for monitoring
            logger.info(
                f"TA analysis executed: {analysis_type}, success: {result.get('success', False)}"
            )

            return result

        except Exception as e:
            logger.error(f"TA analysis execution failed: {e}")
            return {"success": False, "error": f"Analysis execution failed: {str(e)}"}


def integrate_ta_agent_with_orchestrator(
    orchestrator: ThreadSafeEnterpriseOrchestrator,
) -> TAAgentIntegration:
    """
    Factory function to integrate TA agent with enterprise orchestrator

    Args:
        orchestrator: The enterprise orchestrator instance

    Returns:
        TAAgentIntegration instance
    """
    integration = TAAgentIntegration(orchestrator)

    # Add integration to orchestrator's extensions if supported
    if hasattr(orchestrator, "extensions"):
        orchestrator.extensions["technical_analysis"] = integration

    return integration


async def setup_ta_agent_in_orchestrator(orchestrator: ThreadSafeEnterpriseOrchestrator) -> bool:
    """
    Setup and initialize TA agent in the enterprise orchestrator

    Args:
        orchestrator: The enterprise orchestrator instance

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create integration
        integration = integrate_ta_agent_with_orchestrator(orchestrator)

        # Initialize the agent
        success = await integration.initialize_ta_agent()

        if success:
            logger.info(
                "Technical Analysis agent successfully integrated with enterprise orchestrator"
            )
        else:
            logger.error("Failed to integrate Technical Analysis agent")

        return success

    except Exception as e:
        logger.error(f"TA agent orchestrator setup failed: {e}")
        return False
