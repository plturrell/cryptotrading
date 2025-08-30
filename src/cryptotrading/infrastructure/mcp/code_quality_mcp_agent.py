"""
Code Quality MCP Agent - STRANDS agent wrapper for Code Quality MCP tools
"""
import asyncio
import logging
from typing import Any, Dict, Optional

from .code_quality_mcp_tools import CodeQualityMCPTools

logger = logging.getLogger(__name__)


class CodeQualityMCPAgent:
    """STRANDS agent that wraps Code Quality MCP tools"""

    def __init__(
        self, agent_id: str = "code_quality_agent", config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.agent_type = "code_quality"
        self.config = config or {}
        self.status = "initialized"

        # Initialize MCP tools
        self.mcp_tools = CodeQualityMCPTools()

        # Register capabilities
        self.capabilities = [
            "analyze_code_quality",
            "calculate_complexity_metrics",
            "detect_code_smells",
            "analyze_dependencies",
            "calculate_impact_analysis",
            "generate_quality_report",
            "code_analysis",
            "quality_metrics",
            "code_smells",
            "dependency_analysis",
            "impact_analysis",
            "quality_reporting",
        ]

        logger.info(f"Code Quality MCP Agent {agent_id} initialized")

    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.status = "running"
            logger.info(f"Code Quality Agent {self.agent_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Code Quality Agent {self.agent_id}: {str(e)}")
            self.status = "error"
            return False

    async def start(self) -> bool:
        """Start the agent"""
        try:
            if self.status != "running":
                await self.initialize()
            logger.info(f"Code Quality Agent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Code Quality Agent {self.agent_id}: {str(e)}")
            return False

    async def stop(self) -> bool:
        """Stop the agent"""
        try:
            self.status = "stopped"
            logger.info(f"Code Quality Agent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Code Quality Agent {self.agent_id}: {str(e)}")
            return False

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages"""
        try:
            message_type = message.get("type")
            tool_name = message.get("tool_name")
            arguments = message.get("arguments", {})

            if tool_name in [
                "analyze_code_quality",
                "calculate_complexity_metrics",
                "detect_code_smells",
                "analyze_dependencies",
                "calculate_impact_analysis",
                "generate_quality_report",
            ]:
                return await self._invoke_mcp_tool(tool_name, arguments)

            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error processing message in Code Quality Agent: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _invoke_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke MCP tool handler"""
        try:
            return await self.mcp_tools.handle_tool_call(tool_name, arguments)
        except Exception as e:
            logger.error(f"Error invoking MCP tool {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "capabilities": self.capabilities,
            "tools_count": len(self.mcp_tools.tools),
        }

    def get_capabilities(self) -> list:
        """Get agent capabilities"""
        return self.capabilities.copy()
