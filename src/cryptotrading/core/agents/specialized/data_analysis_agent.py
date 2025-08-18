"""
Data Analysis Agent - STRANDS Integration
Specialized agent for statistical analysis and data quality validation
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from ..base_agent import BaseAgent, AgentConfig
from ...data_ingestion.quality_validator import FactorQualityValidator
from ...processing.parallel_executor import ParallelExecutor
from ...infrastructure.mcp.data_analysis_mcp_tools import data_analysis_mcp_tools

logger = logging.getLogger(__name__)

class DataAnalysisAgent(BaseAgent):
    """STRANDS agent for data analysis operations"""
    
    def __init__(self, agent_id: str = "data_analysis_agent", **kwargs):
        """Initialize Data Analysis Agent"""
        config = AgentConfig(
            agent_id=agent_id,
            agent_type="data_analysis",
            description="Statistical analysis and data quality validation agent",
            capabilities=[
                "validate_data_quality", "analyze_data_distribution", "compute_correlation_matrix",
                "detect_outliers", "compute_rolling_statistics"
            ],
            max_concurrent_tools=4,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=30
        )
        
        super().__init__(
            agent_id=agent_id,
            agent_type="data_analysis",
            config=config,
            **kwargs
        )
        
        self.quality_validator = FactorQualityValidator()
        self.parallel_executor = ParallelExecutor()
        self.mcp_tools = data_analysis_mcp_tools
        
        # Register MCP tools as STRANDS tools
        self._register_strands_tools()
        
        logger.info(f"Data Analysis Agent {agent_id} initialized")
    
    def _register_strands_tools(self):
        """Register MCP tools as STRANDS tools"""
        for tool_def in self.mcp_tools.tools:
            tool_name = tool_def["name"]
            
            # Create STRANDS tool wrapper
            async def tool_wrapper(tool_name=tool_name, **kwargs):
                return await self.mcp_tools.handle_tool_call(tool_name, kwargs)
            
            # Register with STRANDS
            self.register_tool(
                name=tool_name,
                description=tool_def["description"],
                func=tool_wrapper,
                input_schema=tool_def["inputSchema"]
            )
    
    async def initialize(self) -> bool:
        """Initialize the Data Analysis Agent"""
        try:
            logger.info(f"Initializing Data Analysis Agent {self.agent_id}")
            
            # Test quality validator
            test_data = pd.Series([1, 2, 3, 4, 5])
            test_result = await self.quality_validator.validate_factor("test", test_data, {})
            logger.info(f"Quality validator test: {test_result.get('score', 0)}")
            
            logger.info(f"Data Analysis Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Data Analysis Agent {self.agent_id}: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the Data Analysis Agent"""
        try:
            logger.info(f"Starting Data Analysis Agent {self.agent_id}")
            
            # Data analysis is primarily request-driven
            # No background processes needed
            
            logger.info(f"Data Analysis Agent {self.agent_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Data Analysis Agent {self.agent_id}: {e}")
            return False
    
    async def validate_factor_quality(self, data: Dict[str, Any], factor_names: List[str], 
                                    validation_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate quality of factors in data"""
        try:
            result = await self.execute_tool("validate_data_quality", {
                "data": data,
                "factor_names": factor_names,
                "validation_rules": validation_rules or {}
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating factor quality: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_statistical_distribution(self, data: Dict[str, Any], 
                                             columns: List[str] = None) -> Dict[str, Any]:
        """Analyze statistical distribution of data"""
        try:
            result = await self.execute_tool("analyze_data_distribution", {
                "data": data,
                "columns": columns
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing distribution: {e}")
            return {"success": False, "error": str(e)}
    
    async def compute_correlations(self, data: Dict[str, Any], 
                                 method: str = "pearson") -> Dict[str, Any]:
        """Compute correlation matrix"""
        try:
            result = await self.execute_tool("compute_correlation_matrix", {
                "data": data,
                "method": method
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing correlations: {e}")
            return {"success": False, "error": str(e)}
    
    async def find_outliers(self, data: Dict[str, Any], method: str = "iqr", 
                          threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers in data"""
        try:
            result = await self.execute_tool("detect_outliers", {
                "data": data,
                "method": method,
                "threshold": threshold
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return {"success": False, "error": str(e)}
    
    async def compute_time_series_stats(self, data: Dict[str, Any], window: int = 20, 
                                      statistics: List[str] = None) -> Dict[str, Any]:
        """Compute rolling statistics for time series"""
        try:
            result = await self.execute_tool("compute_rolling_statistics", {
                "data": data,
                "window": window,
                "statistics": statistics or ["mean", "std"]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing rolling statistics: {e}")
            return {"success": False, "error": str(e)}
    
    async def comprehensive_data_analysis(self, data: Dict[str, Any], 
                                        factor_names: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "comprehensive"
            }
            
            # Distribution analysis
            distribution_result = await self.analyze_statistical_distribution(data)
            results["distribution_analysis"] = distribution_result
            
            # Correlation analysis
            correlation_result = await self.compute_correlations(data)
            results["correlation_analysis"] = correlation_result
            
            # Outlier detection
            outlier_result = await self.find_outliers(data)
            results["outlier_analysis"] = outlier_result
            
            # Quality validation if factor names provided
            if factor_names:
                quality_result = await self.validate_factor_quality(data, factor_names)
                results["quality_validation"] = quality_result
            
            # Rolling statistics for time series data
            rolling_result = await self.compute_time_series_stats(data)
            results["rolling_statistics"] = rolling_result
            
            return {
                "success": True,
                "comprehensive_analysis": results
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages with data analysis operations"""
        try:
            message_type = message.get("type", "unknown")
            
            if message_type == "validate_quality":
                data = message.get("data", {})
                factor_names = message.get("factor_names", [])
                validation_rules = message.get("validation_rules")
                return await self.validate_factor_quality(data, factor_names, validation_rules)
                
            elif message_type == "analyze_distribution":
                data = message.get("data", {})
                columns = message.get("columns")
                return await self.analyze_statistical_distribution(data, columns)
                
            elif message_type == "compute_correlations":
                data = message.get("data", {})
                method = message.get("method", "pearson")
                return await self.compute_correlations(data, method)
                
            elif message_type == "detect_outliers":
                data = message.get("data", {})
                method = message.get("method", "iqr")
                threshold = message.get("threshold", 3.0)
                return await self.find_outliers(data, method, threshold)
                
            elif message_type == "rolling_statistics":
                data = message.get("data", {})
                window = message.get("window", 20)
                statistics = message.get("statistics")
                return await self.compute_time_series_stats(data, window, statistics)
                
            elif message_type == "comprehensive_analysis":
                data = message.get("data", {})
                factor_names = message.get("factor_names")
                return await self.comprehensive_data_analysis(data, factor_names)
                
            else:
                return await super().process_message(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"success": False, "error": str(e)}

# Global agent instance
data_analysis_agent = DataAnalysisAgent()
