"""
A2A MCP Bridge
Connects A2A agents with MCP tools for enhanced functionality
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..protocols.a2a.a2a_protocol import A2A_CAPABILITIES, MessageType
from ...infrastructure.analysis.mcp_agent_segregation import get_segregation_manager
from ...infrastructure.analysis.segregated_mcp_tools import (
    CLRSAnalysisTool, DependencyGraphTool, CodeSimilarityTool,
    HierarchicalIndexingTool, ConfigurationMergeTool, OptimizationRecommendationTool
)

logger = logging.getLogger(__name__)

class A2AMCPBridge:
    """Bridge between A2A agents and MCP tools with segregation support"""
    
    def __init__(self):
        self.segregation_manager = None
        self.mcp_tools = {}
        self.agent_tool_mappings = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the MCP bridge"""
        try:
            # Get segregation manager
            self.segregation_manager = get_segregation_manager()
            
            # Initialize MCP tools
            await self._initialize_mcp_tools()
            
            # Setup agent-tool mappings
            self._setup_agent_mappings()
            
            self.initialized = True
            logger.info("A2A MCP Bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize A2A MCP Bridge: {e}")
            return False
    
    async def _initialize_mcp_tools(self):
        """Initialize all MCP tools with segregation support"""
        try:
            # Initialize segregated MCP tools
            tools = {
                'clrs_analysis': CLRSAnalysisTool(),
                'dependency_graph': DependencyGraphTool(),
                'code_similarity': CodeSimilarityTool(),
                'hierarchical_indexing': HierarchicalIndexingTool(),
                'configuration_merge': ConfigurationMergeTool(),
                'optimization_recommendation': OptimizationRecommendationTool()
            }
            
            # Wrap tools with segregation
            for tool_name, tool_instance in tools.items():
                if self.segregation_manager:
                    wrapped_tool = self.segregation_manager.wrap_tool(tool_instance)
                    self.mcp_tools[tool_name] = wrapped_tool
                else:
                    self.mcp_tools[tool_name] = tool_instance
            
            logger.info(f"Initialized {len(self.mcp_tools)} MCP tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
    
    def _setup_agent_mappings(self):
        """Setup mappings between A2A agents and MCP tools"""
        self.agent_tool_mappings = {
            'technical_analysis_agent': {
                'primary_tools': ['clrs_analysis', 'optimization_recommendation'],
                'secondary_tools': ['dependency_graph'],
                'capabilities_mapping': {
                    'technical_indicators': ['clrs_analysis'],
                    'momentum_analysis': ['optimization_recommendation'],
                    'pattern_recognition': ['clrs_analysis', 'code_similarity']
                }
            },
            'ml_agent': {
                'primary_tools': ['optimization_recommendation', 'configuration_merge'],
                'secondary_tools': ['clrs_analysis'],
                'capabilities_mapping': {
                    'model_training': ['optimization_recommendation'],
                    'hyperparameter_optimization': ['optimization_recommendation'],
                    'feature_engineering': ['configuration_merge']
                }
            },
            'strands_glean_agent': {
                'primary_tools': ['code_similarity', 'hierarchical_indexing'],
                'secondary_tools': ['dependency_graph'],
                'capabilities_mapping': {
                    'code_analysis': ['code_similarity', 'hierarchical_indexing'],
                    'dependency_mapping': ['dependency_graph'],
                    'symbol_search': ['hierarchical_indexing']
                }
            },
            'code_quality_agent': {
                'primary_tools': ['code_similarity', 'clrs_analysis'],
                'secondary_tools': ['dependency_graph'],
                'capabilities_mapping': {
                    'code_analysis': ['code_similarity'],
                    'quality_metrics': ['clrs_analysis'],
                    'dependency_analysis': ['dependency_graph']
                }
            },
            'data_analysis_agent': {
                'primary_tools': ['optimization_recommendation', 'clrs_analysis'],
                'secondary_tools': ['configuration_merge'],
                'capabilities_mapping': {
                    'statistical_analysis': ['clrs_analysis'],
                    'data_validation': ['optimization_recommendation'],
                    'quality_assessment': ['configuration_merge']
                }
            },
            'clrs_algorithms_agent': {
                'primary_tools': ['clrs_analysis'],
                'secondary_tools': ['optimization_recommendation'],
                'capabilities_mapping': {
                    'algorithmic_calculations': ['clrs_analysis'],
                    'search_algorithms': ['clrs_analysis'],
                    'sorting_algorithms': ['clrs_analysis']
                }
            }
        }
    
    async def execute_agent_with_mcp(self, agent_id: str, capability: str, 
                                   parameters: Dict[str, Any], 
                                   agent_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute an A2A agent capability using appropriate MCP tools"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get agent mapping
            agent_mapping = self.agent_tool_mappings.get(agent_id)
            if not agent_mapping:
                return {"error": f"No MCP mapping found for agent {agent_id}"}
            
            # Get tools for this capability
            capability_tools = agent_mapping.get('capabilities_mapping', {}).get(capability, [])
            if not capability_tools:
                # Fallback to primary tools
                capability_tools = agent_mapping.get('primary_tools', [])
            
            if not capability_tools:
                return {"error": f"No MCP tools available for capability {capability}"}
            
            # Execute with segregation context
            segregation_context = {
                "agent_id": agent_id,
                "capability": capability,
                "timestamp": datetime.now().isoformat(),
                "session_token": f"a2a_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            if agent_context:
                segregation_context.update(agent_context)
            
            results = {}
            
            # Execute each tool
            for tool_name in capability_tools:
                if tool_name in self.mcp_tools:
                    try:
                        tool = self.mcp_tools[tool_name]
                        
                        # Prepare tool parameters
                        tool_params = {
                            **parameters,
                            "agent_context": segregation_context
                        }
                        
                        # Execute tool with segregation
                        if self.segregation_manager:
                            result = await self.segregation_manager.execute_tool(
                                tool, tool_params, segregation_context
                            )
                        else:
                            result = await tool.execute(tool_params)
                        
                        results[tool_name] = result
                        
                    except Exception as e:
                        logger.error(f"Error executing MCP tool {tool_name}: {e}")
                        results[tool_name] = {"error": str(e)}
            
            # Combine results
            combined_result = {
                "agent_id": agent_id,
                "capability": capability,
                "tools_used": list(results.keys()),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add aggregated insights if multiple tools were used
            if len(results) > 1:
                combined_result["aggregated_insights"] = self._aggregate_tool_results(results)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Failed to execute agent {agent_id} with MCP: {e}")
            return {"error": str(e), "agent_id": agent_id, "capability": capability}
    
    def _aggregate_tool_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple MCP tools"""
        aggregated = {
            "tools_count": len(results),
            "successful_tools": len([r for r in results.values() if "error" not in r]),
            "common_patterns": [],
            "confidence_scores": [],
            "recommendations": []
        }
        
        # Extract common patterns and insights
        for tool_name, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                # Extract confidence scores
                if "confidence" in result:
                    aggregated["confidence_scores"].append(result["confidence"])
                
                # Extract recommendations
                if "recommendations" in result:
                    if isinstance(result["recommendations"], list):
                        aggregated["recommendations"].extend(result["recommendations"])
                    else:
                        aggregated["recommendations"].append(result["recommendations"])
                
                # Extract patterns
                if "patterns" in result:
                    if isinstance(result["patterns"], list):
                        aggregated["common_patterns"].extend(result["patterns"])
                    else:
                        aggregated["common_patterns"].append(result["patterns"])
        
        # Calculate average confidence
        if aggregated["confidence_scores"]:
            aggregated["average_confidence"] = sum(aggregated["confidence_scores"]) / len(aggregated["confidence_scores"])
        
        return aggregated
    
    async def get_agent_mcp_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Get MCP capabilities available for an A2A agent"""
        agent_mapping = self.agent_tool_mappings.get(agent_id, {})
        
        capabilities = {
            "agent_id": agent_id,
            "primary_tools": agent_mapping.get('primary_tools', []),
            "secondary_tools": agent_mapping.get('secondary_tools', []),
            "capability_mappings": agent_mapping.get('capabilities_mapping', {}),
            "total_tools_available": len(set(
                agent_mapping.get('primary_tools', []) + 
                agent_mapping.get('secondary_tools', [])
            ))
        }
        
        # Add tool descriptions
        tool_descriptions = {}
        for tool_name in capabilities["primary_tools"] + capabilities["secondary_tools"]:
            if tool_name in self.mcp_tools:
                tool_descriptions[tool_name] = {
                    "type": type(self.mcp_tools[tool_name]).__name__,
                    "available": True
                }
            else:
                tool_descriptions[tool_name] = {
                    "type": "unknown",
                    "available": False
                }
        
        capabilities["tool_descriptions"] = tool_descriptions
        
        return capabilities
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the MCP bridge"""
        health = {
            "status": "healthy",
            "initialized": self.initialized,
            "mcp_tools_count": len(self.mcp_tools),
            "agent_mappings_count": len(self.agent_tool_mappings),
            "segregation_manager_available": self.segregation_manager is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check individual tools
        tool_health = {}
        for tool_name, tool in self.mcp_tools.items():
            try:
                # Basic health check - try to access tool properties
                tool_health[tool_name] = {
                    "available": True,
                    "type": type(tool).__name__
                }
            except Exception as e:
                tool_health[tool_name] = {
                    "available": False,
                    "error": str(e)
                }
        
        health["tools"] = tool_health
        
        # Determine overall health
        unavailable_tools = sum(1 for t in tool_health.values() if not t.get("available", False))
        if unavailable_tools > len(tool_health) // 2:
            health["status"] = "degraded"
        elif unavailable_tools > 0:
            health["status"] = "partial"
        
        return health

# Global bridge instance
_bridge_instance = None

async def get_a2a_mcp_bridge():
    """Get or create the global A2A MCP bridge instance"""
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = A2AMCPBridge()
        await _bridge_instance.initialize()
    
    return _bridge_instance
