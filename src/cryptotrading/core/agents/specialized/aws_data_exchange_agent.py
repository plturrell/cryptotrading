"""
AWS Data Exchange Strands Agent
A2A agent for accessing premium financial datasets through AWS Data Exchange
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from ...strands import StrandsAgent
from ....infrastructure.mcp.aws_data_exchange_mcp_tools import AWSDataExchangeMCPTools

logger = logging.getLogger(__name__)

class AWSDataExchangeAgent(StrandsAgent):
    """
    Specialized A2A Strands agent for AWS Data Exchange operations
    
    Capabilities:
    - Dataset discovery and analysis
    - Financial and economic data sourcing
    - Automated data export and processing
    - Quality analysis and validation
    - Database integration
    """
    
    def __init__(self, agent_id: str = None, **kwargs):
        """Initialize AWS Data Exchange agent"""
        super().__init__(
            agent_id=agent_id or f"aws-data-exchange-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            agent_type="aws_data_exchange",
            capabilities=[
                "dataset_discovery",
                "financial_data_sourcing", 
                "economic_data_sourcing",
                "data_export",
                "data_processing",
                "quality_analysis",
                "database_integration",
                "premium_data_access"
            ],
            **kwargs
        )
        
        # Initialize MCP tools
        self.mcp_tools = AWSDataExchangeMCPTools()
        self._register_mcp_tools()
        
        # Agent state
        self.active_jobs = {}  # Track active export jobs
        self.discovered_datasets = {}  # Cache discovered datasets
        self.last_discovery_time = None
        
        logger.info(f"AWS Data Exchange Agent initialized: {self.agent_id}")
    
    def _register_mcp_tools(self):
        """Register MCP tools with the agent"""
        for tool_def in self.mcp_tools.tools:
            self.tool_registry.register_tool(
                name=tool_def["name"],
                description=tool_def["description"],
                func=self._execute_mcp_tool,
                schema=tool_def["inputSchema"]
            )
        
        logger.info(f"Registered {len(self.mcp_tools.tools)} MCP tools")
    
    async def _execute_mcp_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute an MCP tool"""
        try:
            result = await self.mcp_tools.execute_tool(tool_name, kwargs)
            
            # Update agent state based on tool execution
            await self._update_agent_state(tool_name, kwargs, result)
            
            return result
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tool": tool_name
            }
    
    async def _update_agent_state(self, tool_name: str, arguments: Dict[str, Any], 
                                result: Dict[str, Any]):
        """Update agent state based on tool execution"""
        try:
            if tool_name == "discover_financial_datasets" and result.get("status") == "success":
                self.discovered_datasets = {
                    ds["dataset_id"]: ds for ds in result.get("datasets", [])
                }
                self.last_discovery_time = datetime.utcnow()
                
            elif tool_name == "create_data_export_job" and result.get("status") == "success":
                job_id = result.get("job_id")
                if job_id:
                    self.active_jobs[job_id] = {
                        "dataset_id": arguments.get("dataset_id"),
                        "asset_id": arguments.get("asset_id"),
                        "created_at": datetime.utcnow(),
                        "status": "running"
                    }
                    
            elif tool_name == "monitor_export_job" and result.get("status") == "success":
                job_id = arguments.get("job_id")
                if job_id in self.active_jobs:
                    job_completed = result.get("job_completed", False)
                    self.active_jobs[job_id]["status"] = "completed" if job_completed else "running"
                    self.active_jobs[job_id]["last_checked"] = datetime.utcnow()
                    
        except Exception as e:
            logger.warning(f"Error updating agent state: {e}")
    
    async def discover_datasets(self, dataset_type: str = "all", 
                              keywords: List[str] = None,
                              force_refresh: bool = False) -> Dict[str, Any]:
        """
        High-level method to discover financial datasets
        
        Args:
            dataset_type: Type of datasets ('all', 'crypto', 'economic')
            keywords: Additional filter keywords
            force_refresh: Force refresh of cached data
            
        Returns:
            Discovery results with recommendations
        """
        # Check cache unless force refresh
        if not force_refresh and self.discovered_datasets and self.last_discovery_time:
            cache_age = datetime.utcnow() - self.last_discovery_time
            if cache_age.total_seconds() < 3600:  # 1 hour cache
                return {
                    "status": "success",
                    "source": "cache",
                    "datasets": list(self.discovered_datasets.values()),
                    "cached_at": self.last_discovery_time.isoformat()
                }
        
        # Execute discovery
        result = await self._execute_mcp_tool(
            "discover_financial_datasets",
            dataset_type=dataset_type,
            keywords=keywords or []
        )
        
        if result.get("status") == "success":
            # Add recommendations based on dataset characteristics
            datasets = result.get("datasets", [])
            recommended_datasets = self._analyze_and_recommend_datasets(datasets, dataset_type)
            
            result["recommendations"] = recommended_datasets
            result["analysis"] = {
                "total_datasets": len(datasets),
                "providers": list(set(ds.get("provider", "unknown") for ds in datasets)),
                "avg_freshness_days": self._calculate_avg_freshness(datasets)
            }
        
        return result
    
    def _analyze_and_recommend_datasets(self, datasets: List[Dict], 
                                      dataset_type: str) -> List[Dict]:
        """Analyze datasets and provide recommendations"""
        recommendations = []
        
        # Sort by last updated (most recent first)
        sorted_datasets = sorted(
            datasets, 
            key=lambda x: x.get("last_updated", "1900-01-01T00:00:00"), 
            reverse=True
        )
        
        # Recommend top 3 most recent datasets
        for i, dataset in enumerate(sorted_datasets[:3]):
            recommendation = {
                "dataset_id": dataset.get("dataset_id"),
                "name": dataset.get("name"),
                "reason": f"Rank #{i+1} - Most recently updated dataset",
                "last_updated": dataset.get("last_updated"),
                "provider": dataset.get("provider")
            }
            
            # Add specific reasons based on dataset type
            if dataset_type == "crypto" and any(keyword in dataset.get("name", "").lower() 
                                              for keyword in ["bitcoin", "crypto", "ethereum"]):
                recommendation["reason"] += " with crypto focus"
            elif dataset_type == "economic" and any(keyword in dataset.get("name", "").lower()
                                                   for keyword in ["gdp", "inflation", "fed"]):
                recommendation["reason"] += " with economic indicators"
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_avg_freshness(self, datasets: List[Dict]) -> float:
        """Calculate average dataset freshness in days"""
        if not datasets:
            return 0.0
            
        total_days = 0
        valid_dates = 0
        current_time = datetime.utcnow()
        
        for dataset in datasets:
            try:
                last_updated = datetime.fromisoformat(
                    dataset.get("last_updated", "").replace("Z", "+00:00")
                )
                days_old = (current_time - last_updated).days
                total_days += days_old
                valid_dates += 1
            except:
                continue
        
        return total_days / valid_dates if valid_dates > 0 else 0.0
    
    async def create_and_monitor_export(self, dataset_id: str, asset_id: str,
                                      auto_process: bool = True,
                                      timeout_minutes: int = 30) -> Dict[str, Any]:
        """
        High-level method to create export job and monitor completion
        
        Args:
            dataset_id: The dataset ID
            asset_id: The asset ID
            auto_process: Automatically process data when job completes
            timeout_minutes: Job timeout
            
        Returns:
            Complete export and processing results
        """
        # Create export job
        job_result = await self._execute_mcp_tool(
            "create_data_export_job",
            dataset_id=dataset_id,
            asset_id=asset_id
        )
        
        if job_result.get("status") != "success":
            return job_result
        
        job_id = job_result.get("job_id")
        
        # Monitor job completion
        monitor_result = await self._execute_mcp_tool(
            "monitor_export_job",
            job_id=job_id,
            timeout_minutes=timeout_minutes
        )
        
        if monitor_result.get("status") != "success":
            return monitor_result
        
        job_completed = monitor_result.get("job_completed", False)
        
        # If auto-process is enabled and job completed successfully
        if auto_process and job_completed:
            process_result = await self._execute_mcp_tool(
                "download_and_process_data",
                dataset_id=dataset_id,
                asset_id=asset_id,
                processing_options={"clean_data": True, "sample_rows": 1000}
            )
            
            return {
                "status": "success",
                "export_job": job_result,
                "monitoring": monitor_result,
                "processing": process_result,
                "completed_pipeline": True
            }
        
        return {
            "status": "success",
            "export_job": job_result,
            "monitoring": monitor_result,
            "completed_pipeline": job_completed
        }
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        # Get service status
        service_status = await self._execute_mcp_tool("get_service_status")
        
        # Compile agent status
        status = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "service_status": service_status,
            "active_jobs": len(self.active_jobs),
            "discovered_datasets": len(self.discovered_datasets),
            "last_discovery": self.last_discovery_time.isoformat() if self.last_discovery_time else None,
            "tools_available": len(self.mcp_tools.tools),
            "uptime": datetime.utcnow().isoformat()
        }
        
        # Add job details
        if self.active_jobs:
            status["job_details"] = {
                job_id: {
                    "dataset_id": job_info["dataset_id"],
                    "status": job_info["status"],
                    "created_at": job_info["created_at"].isoformat()
                }
                for job_id, job_info in self.active_jobs.items()
            }
        
        return status
    
    async def cleanup_completed_jobs(self, older_than_hours: int = 24):
        """Clean up completed jobs older than specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        jobs_to_remove = []
        
        for job_id, job_info in self.active_jobs.items():
            if (job_info["status"] == "completed" and 
                job_info.get("last_checked", job_info["created_at"]) < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} completed jobs")
        return len(jobs_to_remove)

# Factory function for easy agent creation
def create_aws_data_exchange_agent(**kwargs) -> AWSDataExchangeAgent:
    """Factory function to create AWS Data Exchange agent"""
    return AWSDataExchangeAgent(**kwargs)