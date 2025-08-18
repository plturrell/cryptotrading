"""
ML Models MCP Agent - STRANDS agent wrapper for ML Models MCP tools
"""
import logging
from typing import Dict, Any, Optional
import asyncio

from .ml_models_mcp_tools import MLModelsMCPTools

logger = logging.getLogger(__name__)


class MLModelsMCPAgent:
    """STRANDS agent that wraps ML Models MCP tools"""
    
    def __init__(self, agent_id: str = "ml_models_agent", config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.agent_type = "ml_models"
        self.config = config or {}
        self.status = "initialized"
        
        # Initialize MCP tools
        self.mcp_tools = MLModelsMCPTools()
        
        # Register capabilities
        self.capabilities = [
            'train_model', 'predict_prices', 'evaluate_model', 'optimize_hyperparameters',
            'ensemble_predict', 'feature_importance', 'ml_training', 'model_evaluation',
            'hyperparameter_optimization', 'ensemble_methods', 'ml_calculations'
        ]
        
        logger.info(f"ML Models MCP Agent {agent_id} initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.status = "running"
            logger.info(f"ML Models Agent {self.agent_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ML Models Agent {self.agent_id}: {str(e)}")
            self.status = "error"
            return False
    
    async def start(self) -> bool:
        """Start the agent"""
        try:
            if self.status != "running":
                await self.initialize()
            logger.info(f"ML Models Agent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start ML Models Agent {self.agent_id}: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """Stop the agent"""
        try:
            self.status = "stopped"
            logger.info(f"ML Models Agent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop ML Models Agent {self.agent_id}: {str(e)}")
            return False
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages"""
        try:
            message_type = message.get("type")
            tool_name = message.get("tool_name")
            arguments = message.get("arguments", {})
            
            if tool_name in ["train_model", "predict_prices", "evaluate_model", 
                           "optimize_hyperparameters", "ensemble_predict", "feature_importance"]:
                return await self._invoke_mcp_tool(tool_name, arguments)
            
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
            
        except Exception as e:
            logger.error(f"Error processing message in ML Models Agent: {str(e)}")
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
            "tools_count": len(self.mcp_tools.tools)
        }
    
    def get_capabilities(self) -> list:
        """Get agent capabilities"""
        return self.capabilities.copy()
