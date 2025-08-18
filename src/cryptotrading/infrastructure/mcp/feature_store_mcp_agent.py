"""
Feature Store MCP Agent - STRANDS agent wrapper for Feature Store MCP tools
"""
import logging
from typing import Dict, Any, Optional
import asyncio

from ...core.ml.feature_store import FeatureStore
from .feature_store_mcp_tools import FeatureStoreMCPTools

logger = logging.getLogger(__name__)


class FeatureStoreMCPAgent:
    """STRANDS agent that wraps Feature Store MCP tools"""
    
    def __init__(self, agent_id: str = "feature_store_agent", config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.agent_type = "feature_store"
        self.config = config or {}
        self.status = "initialized"
        
        # Initialize feature store and MCP tools
        self.feature_store = FeatureStore()
        self.mcp_tools = FeatureStoreMCPTools(self.feature_store)
        
        # Register capabilities
        self.capabilities = [
            'compute_features', 'get_feature_vector', 'get_training_features',
            'get_feature_definitions', 'get_feature_importance', 'feature_engineering',
            'ml_features', 'technical_indicators'
        ]
        
        logger.info(f"Feature Store MCP Agent {agent_id} initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.status = "running"
            logger.info(f"Feature Store Agent {self.agent_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Feature Store Agent {self.agent_id}: {str(e)}")
            self.status = "error"
            return False
    
    async def start(self) -> bool:
        """Start the agent"""
        try:
            if self.status != "running":
                await self.initialize()
            logger.info(f"Feature Store Agent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Feature Store Agent {self.agent_id}: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """Stop the agent"""
        try:
            self.status = "stopped"
            logger.info(f"Feature Store Agent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Feature Store Agent {self.agent_id}: {str(e)}")
            return False
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages"""
        try:
            message_type = message.get("type")
            tool_name = message.get("tool_name")
            arguments = message.get("arguments", {})
            
            if tool_name in ["compute_features", "get_feature_vector", "get_training_features", 
                           "get_feature_definitions", "get_feature_importance"]:
                return await self._invoke_mcp_tool(tool_name, arguments)
            
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
            
        except Exception as e:
            logger.error(f"Error processing message in Feature Store Agent: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _invoke_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke MCP tool handler"""
        try:
            handler_map = {
                "compute_features": self.mcp_tools._handle_compute_features,
                "get_feature_vector": self.mcp_tools._handle_get_feature_vector,
                "get_training_features": self.mcp_tools._handle_get_training_features,
                "get_feature_definitions": self.mcp_tools._handle_get_feature_definitions,
                "get_feature_importance": self.mcp_tools._handle_get_feature_importance
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
            "tools_count": len(self.mcp_tools.tools) if hasattr(self.mcp_tools, 'tools') else 5
        }
    
    def get_capabilities(self) -> list:
        """Get agent capabilities"""
        return self.capabilities.copy()
