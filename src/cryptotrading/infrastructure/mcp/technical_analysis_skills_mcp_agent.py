"""
Technical Analysis Skills MCP Agent - STRANDS agent wrapper for TA Skills MCP tools
"""
import logging
from typing import Dict, Any, Optional
import asyncio

from .technical_analysis_skills_mcp_tools import TechnicalAnalysisSkillsMCPTools

logger = logging.getLogger(__name__)


class TechnicalAnalysisSkillsMCPAgent:
    """STRANDS agent that wraps Technical Analysis Skills MCP tools"""
    
    def __init__(self, agent_id: str = "technical_analysis_skills_agent", config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.agent_type = "technical_analysis_skills"
        self.config = config or {}
        self.status = "initialized"
        
        # Initialize MCP tools
        self.mcp_tools = TechnicalAnalysisSkillsMCPTools()
        
        # Register capabilities
        self.capabilities = [
            'calculate_momentum_indicators', 'calculate_momentum_volatility', 'analyze_volume_patterns',
            'identify_support_resistance', 'detect_chart_patterns', 'comprehensive_analysis',
            'technical_indicators', 'momentum_analysis', 'volume_analysis', 'pattern_recognition',
            'support_resistance', 'chart_patterns', 'ta_calculations'
        ]
        
        logger.info(f"Technical Analysis Skills MCP Agent {agent_id} initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.status = "running"
            logger.info(f"Technical Analysis Skills Agent {self.agent_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Technical Analysis Skills Agent {self.agent_id}: {str(e)}")
            self.status = "error"
            return False
    
    async def start(self) -> bool:
        """Start the agent"""
        try:
            if self.status != "running":
                await self.initialize()
            logger.info(f"Technical Analysis Skills Agent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Technical Analysis Skills Agent {self.agent_id}: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """Stop the agent"""
        try:
            self.status = "stopped"
            logger.info(f"Technical Analysis Skills Agent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Technical Analysis Skills Agent {self.agent_id}: {str(e)}")
            return False
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages"""
        try:
            message_type = message.get("type")
            tool_name = message.get("tool_name")
            arguments = message.get("arguments", {})
            
            if tool_name in ["calculate_momentum_indicators", "calculate_momentum_volatility", 
                           "analyze_volume_patterns", "identify_support_resistance", 
                           "detect_chart_patterns", "comprehensive_analysis"]:
                return await self._invoke_mcp_tool(tool_name, arguments)
            
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
            
        except Exception as e:
            logger.error(f"Error processing message in Technical Analysis Skills Agent: {str(e)}")
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
