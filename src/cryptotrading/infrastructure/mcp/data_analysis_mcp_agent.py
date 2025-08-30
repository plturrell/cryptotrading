"""
Data Analysis MCP Agent - STRANDS agent wrapper for Data Analysis MCP tools
"""
import asyncio
import logging
from typing import Any, Dict, Optional

from ...core.data.data_ingestion import DataIngestion
from .data_analysis_mcp_tools import DataAnalysisMCPTools

logger = logging.getLogger(__name__)


class DataAnalysisMCPAgent:
    """STRANDS agent that wraps Data Analysis MCP tools"""

    def __init__(
        self, agent_id: str = "data_analysis_agent", config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.agent_type = "data_analysis"
        self.config = config or {}
        self.status = "initialized"

        # Initialize data ingestion and MCP tools
        self.data_ingestion = DataIngestion()
        self.mcp_tools = DataAnalysisMCPTools(self.data_ingestion)

        # Register capabilities
        self.capabilities = [
            "validate_data_quality",
            "analyze_data_distribution",
            "compute_correlation_matrix",
            "detect_outliers",
            "compute_rolling_statistics",
            "statistical_analysis",
            "data_validation",
            "quality_assessment",
        ]

        logger.info(f"Data Analysis MCP Agent {agent_id} initialized")

    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.status = "running"
            logger.info(f"Data Analysis Agent {self.agent_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Data Analysis Agent {self.agent_id}: {str(e)}")
            self.status = "error"
            return False

    async def start(self) -> bool:
        """Start the agent"""
        try:
            if self.status != "running":
                await self.initialize()
            logger.info(f"Data Analysis Agent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Data Analysis Agent {self.agent_id}: {str(e)}")
            return False

    async def stop(self) -> bool:
        """Stop the agent"""
        try:
            self.status = "stopped"
            logger.info(f"Data Analysis Agent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Data Analysis Agent {self.agent_id}: {str(e)}")
            return False

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages"""
        try:
            message_type = message.get("type")
            tool_name = message.get("tool_name")
            arguments = message.get("arguments", {})

            if tool_name in [
                "validate_data_quality",
                "analyze_data_distribution",
                "compute_correlation_matrix",
                "detect_outliers",
                "compute_rolling_statistics",
            ]:
                return await self._invoke_mcp_tool(tool_name, arguments)

            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error processing message in Data Analysis Agent: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _invoke_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke MCP tool handler"""
        try:
            handler_map = {
                "validate_data_quality": self.mcp_tools._handle_validate_data_quality,
                "analyze_data_distribution": self.mcp_tools._handle_analyze_data_distribution,
                "compute_correlation_matrix": self.mcp_tools._handle_compute_correlation_matrix,
                "detect_outliers": self.mcp_tools._handle_detect_outliers,
                "compute_rolling_statistics": self.mcp_tools._handle_compute_rolling_statistics,
            }

            handler = handler_map.get(tool_name)
            if handler:
                return await handler(arguments)
            else:
                return {"success": False, "error": f"No handler for tool: {tool_name}"}

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
            "tools_count": len(self.mcp_tools.tools) if hasattr(self.mcp_tools, "tools") else 5,
        }

    def get_capabilities(self) -> list:
        """Get agent capabilities"""
        return self.capabilities.copy()
