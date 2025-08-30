"""
AWS Data Exchange MCP Agent
Exposes AWS Data Exchange agent capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.agents.specialized.aws_data_exchange_agent import AWSDataExchangeAgent
from .aws_data_exchange_mcp_tools import AWSDataExchangeMCPTools

logger = logging.getLogger(__name__)


class AWSDataExchangeMCPAgent:
    """MCP wrapper for AWS Data Exchange agent"""

    def __init__(self):
        """Initialize AWS Data Exchange MCP agent"""
        self.agent = AWSDataExchangeAgent()
        self.mcp_tools = AWSDataExchangeMCPTools()
        self.server_name = "aws_data_exchange"
        self.version = "1.0.0"

        logger.info("AWS Data Exchange MCP Agent initialized")

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools"""
        # Add high-level agent methods as additional tools
        agent_tools = [
            {
                "name": "discover_datasets_with_recommendations",
                "description": "Discover datasets with AI-powered recommendations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_type": {
                            "type": "string",
                            "enum": ["all", "crypto", "economic"],
                            "default": "all",
                        },
                        "keywords": {"type": "array", "items": {"type": "string"}, "default": []},
                        "force_refresh": {"type": "boolean", "default": False},
                    },
                },
            },
            {
                "name": "create_and_monitor_export_pipeline",
                "description": "Complete pipeline: create export job, monitor completion, and process data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "The dataset ID"},
                        "asset_id": {"type": "string", "description": "The asset ID"},
                        "auto_process": {"type": "boolean", "default": True},
                        "timeout_minutes": {"type": "number", "default": 30},
                    },
                    "required": ["dataset_id", "asset_id"],
                },
            },
            {
                "name": "get_comprehensive_agent_status",
                "description": "Get comprehensive AWS Data Exchange agent status",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "cleanup_old_jobs",
                "description": "Clean up completed jobs older than specified hours",
                "inputSchema": {
                    "type": "object",
                    "properties": {"older_than_hours": {"type": "number", "default": 24}},
                },
            },
        ]

        # Combine with base MCP tools
        return self.mcp_tools.tools + agent_tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via the agent or MCP tools"""
        try:
            # Check if it's a high-level agent method
            if tool_name == "discover_datasets_with_recommendations":
                return await self.agent.discover_datasets(**arguments)
            elif tool_name == "create_and_monitor_export_pipeline":
                return await self.agent.create_and_monitor_export(**arguments)
            elif tool_name == "get_comprehensive_agent_status":
                return await self.agent.get_agent_status()
            elif tool_name == "cleanup_old_jobs":
                jobs_cleaned = await self.agent.cleanup_completed_jobs(
                    arguments.get("older_than_hours", 24)
                )
                return {
                    "status": "success",
                    "jobs_cleaned": jobs_cleaned,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Execute via MCP tools
                return await self.mcp_tools.execute_tool(tool_name, arguments)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server information"""
        return {
            "name": self.server_name,
            "version": self.version,
            "description": "AWS Data Exchange A2A Agent with MCP interface",
            "capabilities": self.agent.capabilities,
            "tools_count": len(self.get_tools()),
            "agent_id": self.agent.agent_id,
            "initialized_at": datetime.utcnow().isoformat(),
        }

    async def handle_mcp_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        try:
            if method == "tools/list":
                return {"tools": self.get_tools()}
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if not tool_name:
                    return {"error": "Tool name is required", "code": -32602}

                result = await self.execute_tool(tool_name, arguments)
                return {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
                }
            elif method == "server/info":
                return self.get_server_info()
            else:
                return {"error": f"Unknown method: {method}", "code": -32601}

        except Exception as e:
            logger.error(f"Error handling MCP request {method}: {e}")
            return {"error": str(e), "code": -32603}


# Global instance for MCP server
mcp_agent = None


def get_mcp_agent() -> AWSDataExchangeMCPAgent:
    """Get or create global MCP agent instance"""
    global mcp_agent
    if mcp_agent is None:
        mcp_agent = AWSDataExchangeMCPAgent()
    return mcp_agent


# MCP server entry point functions
async def list_tools() -> List[Dict[str, Any]]:
    """MCP tools list endpoint"""
    agent = get_mcp_agent()
    return agent.get_tools()


async def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP tool call endpoint"""
    agent = get_mcp_agent()
    return await agent.execute_tool(name, arguments)
