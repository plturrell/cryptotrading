"""
STRANDS Agent wrapper for MCTS Calculation MCP Tools
Provides agent lifecycle management and message processing for MCTS calculations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from ...core.agents.base import BaseAgent
from .mcts_calculation_mcp_tools import MCTSCalculationMCPTools

logger = logging.getLogger(__name__)

class MCTSCalculationMCPAgent(BaseAgent):
    """STRANDS agent wrapper for MCTS Calculation MCP tools"""
    
    def __init__(self, agent_id: str = "mcts_calculation_mcp_agent"):
        super().__init__(
            agent_id=agent_id,
            agent_type="mcts_calculation_mcp",
            capabilities=[
                "mcts_calculation",
                "data_analysis", 
                "feature_optimization",
                "pattern_recognition",
                "statistical_analysis",
                "performance_metrics"
            ]
        )
        
        # Initialize MCP tools
        self.mcp_tools = MCTSCalculationMCPTools()
        
        # Register all MCP tools as STRANDS tools
        self._register_mcp_tools_as_strands_tools()
        
        logger.info(f"MCTS Calculation MCP Agent {agent_id} initialized with {len(self.mcp_tools.tools)} tools")
    
    def _register_mcp_tools_as_strands_tools(self):
        """Register all MCP tools as STRANDS tools"""
        for tool_def in self.mcp_tools.tools:
            tool_name = tool_def["name"]
            
            # Create STRANDS tool wrapper
            async def tool_wrapper(*args, tool_name=tool_name, **kwargs):
                # Convert args/kwargs to MCP tool arguments format
                if args and isinstance(args[0], dict):
                    arguments = args[0]
                else:
                    arguments = kwargs
                
                return await self.mcp_tools.handle_tool_call(tool_name, arguments)
            
            # Register with STRANDS framework
            self.register_tool(tool_name, tool_wrapper)
    
    async def initialize(self) -> bool:
        """Initialize the MCTS Calculation MCP agent"""
        try:
            logger.info(f"Initializing MCTS Calculation MCP Agent {self.agent_id}")
            
            # Test MCP tools functionality
            test_result = await self.mcp_tools.handle_tool_call(
                "mcts_get_performance_metrics", 
                {"calculation_id": "test"}
            )
            
            if test_result.get("success", False):
                logger.info("MCTS Calculation MCP tools validation successful")
            else:
                logger.warning("MCTS Calculation MCP tools validation returned no results")
            
            logger.info(f"MCTS Calculation MCP Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCTS Calculation MCP Agent {self.agent_id}: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the MCTS Calculation MCP agent"""
        try:
            logger.info(f"Starting MCTS Calculation MCP Agent {self.agent_id}")
            
            # Agent is ready to process requests
            self.status = "running"
            
            logger.info(f"MCTS Calculation MCP Agent {self.agent_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCTS Calculation MCP Agent {self.agent_id}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the MCTS Calculation MCP agent"""
        try:
            logger.info(f"Stopping MCTS Calculation MCP Agent {self.agent_id}")
            
            self.status = "stopped"
            
            logger.info(f"MCTS Calculation MCP Agent {self.agent_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCTS Calculation MCP Agent {self.agent_id}: {e}")
            return False
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages for MCTS calculations"""
        try:
            message_type = message.get("type", "unknown")
            
            if message_type == "mcts_calculation_request":
                return await self._handle_calculation_request(message)
            elif message_type == "performance_metrics_request":
                return await self._handle_metrics_request(message)
            elif message_type == "tool_call":
                return await self._handle_tool_call_message(message)
            else:
                return {
                    "success": False,
                    "error": f"Unknown message type: {message_type}",
                    "agent_id": self.agent_id
                }
                
        except Exception as e:
            logger.error(f"Error processing message in MCTS Calculation MCP Agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    async def _handle_calculation_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCTS calculation requests"""
        try:
            problem_type = message.get("problem_type", "data_analysis")
            parameters = message.get("parameters", {})
            constraints = message.get("constraints", {})
            max_iterations = message.get("max_iterations", 1000)
            
            # Call MCTS calculation tool
            result = await self.mcp_tools.handle_tool_call(
                "mcts_calculate",
                {
                    "problem_type": problem_type,
                    "parameters": parameters,
                    "constraints": constraints,
                    "max_iterations": max_iterations
                }
            )
            
            return {
                "success": True,
                "result": result,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    async def _handle_metrics_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance metrics requests"""
        try:
            calculation_id = message.get("calculation_id")
            include_convergence = message.get("include_convergence", True)
            
            result = await self.mcp_tools.handle_tool_call(
                "mcts_get_performance_metrics",
                {
                    "calculation_id": calculation_id,
                    "include_convergence": include_convergence
                }
            )
            
            return {
                "success": True,
                "result": result,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    async def _handle_tool_call_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle direct tool call messages"""
        try:
            tool_name = message.get("tool_name")
            arguments = message.get("arguments", {})
            
            if not tool_name:
                return {
                    "success": False,
                    "error": "Missing tool_name in message",
                    "agent_id": self.agent_id
                }
            
            result = await self.mcp_tools.handle_tool_call(tool_name, arguments)
            
            return {
                "success": True,
                "result": result,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return self.capabilities
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        return [tool["name"] for tool in self.mcp_tools.tools]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test basic functionality
            test_result = await self.mcp_tools.handle_tool_call(
                "mcts_get_performance_metrics",
                {}
            )
            
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "tools_available": len(self.mcp_tools.tools),
                "test_result": test_result.get("success", False),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent_id": self.agent_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global instance for registration
mcts_calculation_mcp_agent = MCTSCalculationMCPAgent()
