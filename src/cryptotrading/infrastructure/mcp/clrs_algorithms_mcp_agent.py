"""
CLRS Algorithms MCP Agent - STRANDS agent wrapper for CLRS Algorithms MCP tools
"""
import asyncio
import logging
from typing import Any, Dict, Optional

from .clrs_algorithms_mcp_tools import CLRSAlgorithmsMCPTools

logger = logging.getLogger(__name__)


class CLRSAlgorithmsMCPAgent:
    """STRANDS agent that wraps CLRS Algorithms MCP tools"""

    def __init__(
        self, agent_id: str = "clrs_algorithms_agent", config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.agent_type = "clrs_algorithms"
        self.config = config or {}
        self.status = "initialized"

        # Initialize MCP tools
        self.mcp_tools = CLRSAlgorithmsMCPTools()

        # Register capabilities
        self.capabilities = [
            "binary_search",
            "linear_search",
            "quick_select",
            "find_minimum",
            "find_maximum",
            "insertion_sort",
            "merge_sort",
            "quick_sort",
            "algorithmic_calculations",
            "search_algorithms",
            "sorting_algorithms",
            "clrs_algorithms",
        ]

        logger.info(f"CLRS Algorithms MCP Agent {agent_id} initialized")

    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.status = "running"
            logger.info(f"CLRS Algorithms Agent {self.agent_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CLRS Algorithms Agent {self.agent_id}: {str(e)}")
            self.status = "error"
            return False

    async def start(self) -> bool:
        """Start the agent"""
        try:
            if self.status != "running":
                await self.initialize()
            logger.info(f"CLRS Algorithms Agent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start CLRS Algorithms Agent {self.agent_id}: {str(e)}")
            return False

    async def stop(self) -> bool:
        """Stop the agent"""
        try:
            self.status = "stopped"
            logger.info(f"CLRS Algorithms Agent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop CLRS Algorithms Agent {self.agent_id}: {str(e)}")
            return False

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages"""
        try:
            message_type = message.get("type")
            tool_name = message.get("tool_name")
            arguments = message.get("arguments", {})

            if tool_name in [
                "binary_search",
                "linear_search",
                "quick_select",
                "insertion_sort",
                "merge_sort",
                "quick_sort",
            ]:
                return await self._invoke_mcp_tool(tool_name, arguments)

            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error processing message in CLRS Algorithms Agent: {str(e)}")
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
