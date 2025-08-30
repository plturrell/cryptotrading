"""
A2A Strands Framework Integration
Connects all A2A agents with the Strands orchestrator and MCP tools
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .strands_orchestrator import EnhancedStrandsAgent
from ..protocols.a2a.a2a_protocol import A2A_CAPABILITIES, A2A_ROUTING, MessageType
from ...infrastructure.registry.a2a_registry_v2 import EnhancedA2ARegistry
from ...infrastructure.analysis.mcp_agent_segregation import get_segregation_manager

logger = logging.getLogger(__name__)

class A2AStrandsIntegration:
    """Integrates A2A agents with Strands framework and MCP tools"""
    
    def __init__(self):
        self.strands_agent = None
        self.a2a_registry = None
        self.segregation_manager = None
        self.agent_instances = {}
        self.mcp_tools = {}
        
    async def initialize(self):
        """Initialize the integration system"""
        try:
            # Initialize Strands orchestrator
            self.strands_agent = EnhancedStrandsAgent("a2a_strands_orchestrator")
            await self.strands_agent.initialize()
            
            # Initialize A2A registry
            self.a2a_registry = EnhancedA2ARegistry()
            await self.a2a_registry.initialize()
            
            # Initialize MCP segregation manager
            self.segregation_manager = get_segregation_manager()
            
            # Register A2A agents with Strands
            await self._register_a2a_agents_with_strands()
            
            # Setup MCP tool integration
            await self._setup_mcp_integration()
            
            logger.info("A2A Strands integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize A2A Strands integration: {e}")
            return False
    
    async def _register_a2a_agents_with_strands(self):
        """Register all A2A agents as Strands tools"""
        for agent_id, capabilities in A2A_CAPABILITIES.items():
            try:
                # Create Strands tool for each A2A agent
                tool_spec = {
                    "name": f"a2a_{agent_id}",
                    "description": f"A2A {agent_id.replace('_', ' ').title()}",
                    "capabilities": capabilities,
                    "agent_id": agent_id,
                    "type": "a2a_agent"
                }
                
                # Register with Strands orchestrator
                await self.strands_agent.register_tool(tool_spec)
                
                logger.info(f"Registered A2A agent {agent_id} with Strands")
                
            except Exception as e:
                logger.error(f"Failed to register A2A agent {agent_id}: {e}")
    
    async def _setup_mcp_integration(self):
        """Setup MCP tool integration for A2A agents"""
        try:
            # Map A2A capabilities to MCP tools
            mcp_mappings = {
                'technical_analysis_agent': ['CLRSAnalysisTool', 'DependencyGraphTool'],
                'ml_agent': ['OptimizationRecommendationTool', 'ConfigurationMergeTool'],
                'strands_glean_agent': ['CodeSimilarityTool', 'HierarchicalIndexingTool'],
                'code_quality_agent': ['CLRSAnalysisTool', 'CodeSimilarityTool'],
                'data_analysis_agent': ['OptimizationRecommendationTool', 'DependencyGraphTool']
            }
            
            for agent_id, mcp_tools in mcp_mappings.items():
                if agent_id in A2A_CAPABILITIES:
                    self.mcp_tools[agent_id] = mcp_tools
                    logger.info(f"Mapped MCP tools for {agent_id}: {mcp_tools}")
            
        except Exception as e:
            logger.error(f"Failed to setup MCP integration: {e}")
    
    async def execute_a2a_workflow(self, workflow_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow across A2A agents using Strands orchestration"""
        try:
            workflow_id = workflow_request.get('workflow_id', f"a2a_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            agents_required = workflow_request.get('agents', [])
            
            logger.info(f"Executing A2A workflow {workflow_id} with agents: {agents_required}")
            
            # Create Strands workflow
            strands_workflow = {
                "workflow_id": workflow_id,
                "steps": [],
                "context": workflow_request.get('context', {}),
                "metadata": {
                    "type": "a2a_workflow",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Add steps for each required agent
            for agent_id in agents_required:
                if agent_id in A2A_CAPABILITIES:
                    step = {
                        "tool": f"a2a_{agent_id}",
                        "action": workflow_request.get('action', 'execute'),
                        "parameters": workflow_request.get('parameters', {}),
                        "agent_id": agent_id
                    }
                    strands_workflow["steps"].append(step)
            
            # Execute through Strands orchestrator
            result = await self.strands_agent.execute_workflow(strands_workflow)
            
            # Enhance result with A2A metadata
            result["a2a_metadata"] = {
                "agents_used": agents_required,
                "capabilities_invoked": [A2A_CAPABILITIES.get(agent, []) for agent in agents_required],
                "workflow_type": "a2a_coordinated"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute A2A workflow: {e}")
            return {"error": str(e), "workflow_id": workflow_id}
    
    async def route_message_to_agents(self, message_type: MessageType, message_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Route messages to appropriate A2A agents based on routing table"""
        try:
            # Get target agents from routing table
            target_agents = A2A_ROUTING.get(message_type, [])
            
            if not target_agents:
                logger.warning(f"No agents found for message type: {message_type}")
                return []
            
            results = []
            
            # Send message to each target agent through Strands
            for agent_id in target_agents:
                try:
                    # Create Strands tool execution request
                    tool_request = {
                        "tool": f"a2a_{agent_id}",
                        "action": "process_message",
                        "parameters": {
                            "message_type": message_type.value,
                            "message_data": message_data,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    # Execute through Strands with MCP integration
                    if agent_id in self.mcp_tools:
                        # Use segregated MCP tools for this agent
                        mcp_context = {
                            "agent_id": agent_id,
                            "tools": self.mcp_tools[agent_id],
                            "segregation_enabled": True
                        }
                        tool_request["mcp_context"] = mcp_context
                    
                    result = await self.strands_agent.execute_tool(tool_request)
                    
                    results.append({
                        "agent_id": agent_id,
                        "result": result,
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to route message to agent {agent_id}: {e}")
                    results.append({
                        "agent_id": agent_id,
                        "error": str(e),
                        "status": "error"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to route message: {e}")
            return []
    
    async def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get capabilities for a specific A2A agent"""
        return A2A_CAPABILITIES.get(agent_id, [])
    
    async def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all available A2A agents with their status"""
        agents = {}
        
        for agent_id, capabilities in A2A_CAPABILITIES.items():
            # Check if agent is registered with registry
            agent_info = await self.a2a_registry.get_agent_info(agent_id)
            
            agents[agent_id] = {
                "capabilities": capabilities,
                "status": agent_info.get("status", "unknown") if agent_info else "not_registered",
                "strands_integrated": f"a2a_{agent_id}" in self.strands_agent.tools if self.strands_agent else False,
                "mcp_tools": self.mcp_tools.get(agent_id, []),
                "last_seen": agent_info.get("last_heartbeat") if agent_info else None
            }
        
        return agents
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the integration system"""
        health = {
            "status": "healthy",
            "components": {},
            "agents": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check Strands orchestrator
            if self.strands_agent:
                strands_health = await self.strands_agent.get_health()
                health["components"]["strands"] = strands_health
            else:
                health["components"]["strands"] = {"status": "not_initialized"}
            
            # Check A2A registry
            if self.a2a_registry:
                registry_health = await self.a2a_registry.health_check()
                health["components"]["registry"] = registry_health
            else:
                health["components"]["registry"] = {"status": "not_initialized"}
            
            # Check MCP segregation
            if self.segregation_manager:
                health["components"]["mcp_segregation"] = {"status": "active"}
            else:
                health["components"]["mcp_segregation"] = {"status": "not_available"}
            
            # Check individual agents
            agents_status = await self.get_available_agents()
            health["agents"] = agents_status
            
            # Determine overall health
            component_issues = sum(1 for comp in health["components"].values() 
                                 if comp.get("status") not in ["healthy", "active"])
            agent_issues = sum(1 for agent in health["agents"].values() 
                             if agent.get("status") not in ["active", "healthy"])
            
            if component_issues > 0 or agent_issues > len(health["agents"]) // 2:
                health["status"] = "degraded"
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["status"] = "unhealthy"
            health["error"] = str(e)
            return health

# Global integration instance
_integration_instance = None

async def get_a2a_strands_integration():
    """Get or create the global A2A Strands integration instance"""
    global _integration_instance
    
    if _integration_instance is None:
        _integration_instance = A2AStrandsIntegration()
        await _integration_instance.initialize()
    
    return _integration_instance
