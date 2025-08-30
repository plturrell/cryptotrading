"""
AWS Data Exchange Strands Agent
A2A agent for accessing premium financial datasets through AWS Data Exchange
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ....infrastructure.mcp.aws_data_exchange_mcp_tools import AWSDataExchangeMCPTools
from ...strands import StrandsAgent
from ...protocols.a2a.a2a_messaging import A2AMessagingClient
from ...protocols.a2a.a2a_protocol import A2A_CAPABILITIES, A2AAgentRegistry
from ...protocols.a2a.blockchain_registration import EnhancedA2AAgentRegistry

logger = logging.getLogger(__name__)


class AWSDataExchangeAgent(StrandsAgent):
    """
    MCP-Compliant AWS Data Exchange Strands Agent

    This agent follows the Model Context Protocol (MCP) specification.
    ALL functionality is accessed exclusively through MCP tools via the process_mcp_request() method.
    Direct method calls are not supported - all operations must go through MCP tools.

    Capabilities:
    - Dataset discovery and analysis
    - Financial and economic data sourcing
    - Automated data export and processing
    - Quality analysis and validation
    - Database integration

    MCP Tools Available:
    - discover_datasets: Dataset discovery with caching and recommendations
    - create_and_monitor_export: Complete export workflow with auto-processing
    - get_agent_status: Comprehensive agent status and metrics
    - cleanup_completed_jobs: Maintenance operations for job cleanup
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
                "premium_data_access",
            ],
            **kwargs,
        )

        # Initialize MCP tools
        self.mcp_tools = AWSDataExchangeMCPTools()
        self._register_mcp_tools()

        # Initialize A2A messaging for cross-agent communication
        self.a2a_messaging = A2AMessagingClient(agent_id=self.agent_id)

        # MCP handler registry - maps tool names to handler methods
        self.mcp_handlers = {
            "discover_datasets": self._mcp_discover_datasets,
            "create_and_monitor_export": self._mcp_create_and_monitor_export,
            "get_agent_status": self._mcp_get_agent_status,
            "cleanup_completed_jobs": self._mcp_cleanup_completed_jobs,
            # A2A data integration tools
            "a2a_provide_dataset": self._mcp_a2a_provide_dataset,
            "a2a_stream_financial_data": self._mcp_a2a_stream_financial_data,
            "a2a_validate_data_quality": self._mcp_a2a_validate_data_quality,
        }

        # Agent state
        self.active_jobs = {}  # Track active export jobs
        self.discovered_datasets = {}  # Cache discovered datasets
        self.last_discovery_time = None

        # Register with A2A Agent Registry (including blockchain)
        capabilities = A2A_CAPABILITIES.get(self.agent_id, [])
        mcp_tools = list(self.mcp_handlers.keys())
        
        # Try blockchain registration, fallback to local only
        try:
            import asyncio
            asyncio.create_task(
                EnhancedA2AAgentRegistry.register_agent_with_blockchain(
                    agent_id=self.agent_id,
                    capabilities=capabilities,
                    agent_instance=self,
                    agent_type="aws_data_exchange",
                    mcp_tools=mcp_tools
                )
            )
            logger.info(f"AWS Data Exchange Agent {self.agent_id} blockchain registration initiated")
        except Exception as e:
            # Fallback to local registration only
            A2AAgentRegistry.register_agent(self.agent_id, capabilities, self)
            logger.warning(f"AWS Data Exchange Agent {self.agent_id} registered locally only (blockchain failed: {e})")

        logger.info(f"AWS Data Exchange Agent initialized: {self.agent_id}")
        logger.info(f"MCP handlers registered: {list(self.mcp_handlers.keys())}")

    def _register_mcp_tools(self):
        """Register MCP tools with the agent"""
        for tool_def in self.mcp_tools.tools:
            self.tool_registry.register_tool(
                name=tool_def["name"],
                description=tool_def["description"],
                func=self._execute_mcp_tool,
                schema=tool_def["inputSchema"],
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
            return {"status": "error", "error": str(e), "tool": tool_name}

    async def process_mcp_request(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        MAIN MCP ENTRY POINT - All functionality must go through this method

        This is the ONLY public interface for accessing agent functionality.
        All operations are performed via MCP tools registered in mcp_handlers.

        Args:
            tool_name: Name of the MCP tool to execute
            arguments: Tool-specific arguments

        Returns:
            Tool execution results with status and data
        """
        try:
            # Check if handler exists
            if tool_name not in self.mcp_handlers:
                return {
                    "status": "error",
                    "error": f"Unknown MCP tool: {tool_name}",
                    "available_tools": list(self.mcp_handlers.keys()),
                }

            # Execute the MCP handler
            handler = self.mcp_handlers[tool_name]
            result = await handler(arguments)

            # Update agent state based on tool execution
            await self._update_agent_state(tool_name, arguments, result)

            # Add MCP metadata
            result["mcp_metadata"] = {
                "tool_name": tool_name,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_type": self.agent_type,
            }

            return result

        except Exception as e:
            logger.error(f"Error in MCP request {tool_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tool_name": tool_name,
                "agent_id": self.agent_id,
            }

    async def _update_agent_state(
        self, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any]
    ):
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
                        "status": "running",
                    }

            elif tool_name == "monitor_export_job" and result.get("status") == "success":
                job_id = arguments.get("job_id")
                if job_id in self.active_jobs:
                    job_completed = result.get("job_completed", False)
                    self.active_jobs[job_id]["status"] = "completed" if job_completed else "running"
                    self.active_jobs[job_id]["last_checked"] = datetime.utcnow()

        except Exception as e:
            logger.warning(f"Error updating agent state: {e}")

    async def _mcp_discover_datasets(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Handler: Discover financial datasets with caching and analysis

        Arguments:
            dataset_type: Type of datasets ('all', 'crypto', 'economic') - default: 'all'
            keywords: Additional filter keywords - default: []
            force_refresh: Force refresh of cached data - default: False

        Returns:
            Discovery results with recommendations and analysis
        """
        # Extract arguments with defaults
        dataset_type = arguments.get("dataset_type", "all")
        keywords = arguments.get("keywords", [])
        force_refresh = arguments.get("force_refresh", False)
        # Check cache unless force refresh
        if not force_refresh and self.discovered_datasets and self.last_discovery_time:
            cache_age = datetime.utcnow() - self.last_discovery_time
            if cache_age.total_seconds() < 3600:  # 1 hour cache
                return {
                    "status": "success",
                    "source": "cache",
                    "datasets": list(self.discovered_datasets.values()),
                    "cached_at": self.last_discovery_time.isoformat(),
                }

        # Execute discovery
        result = await self._execute_mcp_tool(
            "discover_financial_datasets", dataset_type=dataset_type, keywords=keywords or []
        )

        if result.get("status") == "success":
            # Add recommendations based on dataset characteristics
            datasets = result.get("datasets", [])
            recommended_datasets = self._analyze_and_recommend_datasets(datasets, dataset_type)

            result["recommendations"] = recommended_datasets
            result["analysis"] = {
                "total_datasets": len(datasets),
                "providers": list(set(ds.get("provider", "unknown") for ds in datasets)),
                "avg_freshness_days": self._calculate_avg_freshness(datasets),
            }

        return result

    def _analyze_and_recommend_datasets(
        self, datasets: List[Dict], dataset_type: str
    ) -> List[Dict]:
        """Analyze datasets and provide recommendations"""
        recommendations = []

        # Sort by last updated (most recent first)
        sorted_datasets = sorted(
            datasets, key=lambda x: x.get("last_updated", "1900-01-01T00:00:00"), reverse=True
        )

        # Recommend top 3 most recent datasets
        for i, dataset in enumerate(sorted_datasets[:3]):
            recommendation = {
                "dataset_id": dataset.get("dataset_id"),
                "name": dataset.get("name"),
                "reason": f"Rank #{i+1} - Most recently updated dataset",
                "last_updated": dataset.get("last_updated"),
                "provider": dataset.get("provider"),
            }

            # Add specific reasons based on dataset type
            if dataset_type == "crypto" and any(
                keyword in dataset.get("name", "").lower()
                for keyword in ["bitcoin", "crypto", "ethereum"]
            ):
                recommendation["reason"] += " with crypto focus"
            elif dataset_type == "economic" and any(
                keyword in dataset.get("name", "").lower()
                for keyword in ["gdp", "inflation", "fed"]
            ):
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

    async def _mcp_create_and_monitor_export(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Handler: Create export job and monitor completion with auto-processing

        Arguments:
            dataset_id: The dataset ID (required)
            asset_id: The asset ID (required)
            auto_process: Automatically process data when job completes - default: True
            timeout_minutes: Job timeout in minutes - default: 30

        Returns:
            Complete export and processing results
        """
        # Extract and validate required arguments
        dataset_id = arguments.get("dataset_id")
        asset_id = arguments.get("asset_id")

        if not dataset_id or not asset_id:
            return {
                "status": "error",
                "error": "Both dataset_id and asset_id are required",
                "missing_params": [p for p in ["dataset_id", "asset_id"] if not arguments.get(p)],
            }

        # Extract optional arguments with defaults
        auto_process = arguments.get("auto_process", True)
        timeout_minutes = arguments.get("timeout_minutes", 30)
        # Create export job
        job_result = await self._execute_mcp_tool(
            "create_data_export_job", dataset_id=dataset_id, asset_id=asset_id
        )

        if job_result.get("status") != "success":
            return job_result

        job_id = job_result.get("job_id")

        # Monitor job completion
        monitor_result = await self._execute_mcp_tool(
            "monitor_export_job", job_id=job_id, timeout_minutes=timeout_minutes
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
                processing_options={"clean_data": True, "sample_rows": 1000},
            )

            return {
                "status": "success",
                "export_job": job_result,
                "monitoring": monitor_result,
                "processing": process_result,
                "completed_pipeline": True,
            }

        return {
            "status": "success",
            "export_job": job_result,
            "monitoring": monitor_result,
            "completed_pipeline": job_completed,
        }

    async def _mcp_get_agent_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Handler: Get comprehensive agent status and metrics

        Arguments:
            include_job_details: Include detailed job information - default: True

        Returns:
            Comprehensive agent status with service info, jobs, and metrics
        """
        include_job_details = arguments.get("include_job_details", True)
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
            "last_discovery": self.last_discovery_time.isoformat()
            if self.last_discovery_time
            else None,
            "tools_available": len(self.mcp_tools.tools),
            "uptime": datetime.utcnow().isoformat(),
        }

        # Add job details if requested
        if include_job_details and self.active_jobs:
            status["job_details"] = {
                job_id: {
                    "dataset_id": job_info["dataset_id"],
                    "status": job_info["status"],
                    "created_at": job_info["created_at"].isoformat(),
                }
                for job_id, job_info in self.active_jobs.items()
            }

        return status

    async def _mcp_cleanup_completed_jobs(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Handler: Clean up completed jobs older than specified hours

        Arguments:
            older_than_hours: Remove jobs completed more than this many hours ago - default: 24

        Returns:
            Cleanup results with count of jobs removed
        """
        older_than_hours = arguments.get("older_than_hours", 24)
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        jobs_to_remove = []

        for job_id, job_info in self.active_jobs.items():
            if (
                job_info["status"] == "completed"
                and job_info.get("last_checked", job_info["created_at"]) < cutoff_time
            ):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]

        logger.info(f"Cleaned up {len(jobs_to_remove)} completed jobs")
        return {
            "status": "success",
            "jobs_cleaned": len(jobs_to_remove),
            "removed_job_ids": jobs_to_remove,
            "older_than_hours": older_than_hours,
            "remaining_active_jobs": len(self.active_jobs),
        }

    # ============= A2A Data Integration MCP Tools =============

    async def _mcp_a2a_provide_dataset(self, dataset_request: Dict[str, Any]) -> Dict[str, Any]:
        """Provide dataset to requesting calculation/strategy agents via A2A messaging"""
        try:
            # Extract request details
            requesting_agent = dataset_request.get("requesting_agent")
            dataset_type = dataset_request.get("dataset_type", "financial")
            symbols = dataset_request.get("symbols", [])
            timeframe = dataset_request.get("timeframe", "1d")
            start_date = dataset_request.get("start_date")
            end_date = dataset_request.get("end_date")

            logger.info(f"Processing A2A dataset request from {requesting_agent}")

            # Discover and prepare relevant datasets
            discovery_result = await self._mcp_discover_datasets({
                "dataset_type": dataset_type,
                "keywords": symbols,
                "force_refresh": False
            })

            if discovery_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": f"Dataset discovery failed: {discovery_result.get('error')}"
                }

            # Filter datasets for requested symbols and timeframe
            suitable_datasets = []
            for dataset in discovery_result.get("datasets", []):
                if self._dataset_matches_request(dataset, symbols, timeframe):
                    suitable_datasets.append(dataset)

            if not suitable_datasets:
                return {
                    "status": "error", 
                    "error": f"No suitable datasets found for symbols {symbols} with {timeframe} timeframe"
                }

            # Create export for the most suitable dataset
            best_dataset = suitable_datasets[0]  # Most recent/relevant
            export_result = await self._mcp_create_and_monitor_export({
                "dataset_id": best_dataset.get("dataset_id"),
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date
            })

            return {
                "status": "success",
                "dataset_provided": best_dataset.get("name", "unknown"),
                "dataset_id": best_dataset.get("dataset_id"),
                "export_job_id": export_result.get("job_id"),
                "data_location": export_result.get("export_location"),
                "symbols_covered": symbols,
                "timeframe": timeframe,
                "requesting_agent": requesting_agent
            }

        except Exception as e:
            logger.error(f"A2A dataset provision failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _mcp_a2a_stream_financial_data(self, stream_request: Dict[str, Any]) -> Dict[str, Any]:
        """Stream real-time financial data to calculation agents via A2A messaging"""
        try:
            symbols = stream_request.get("symbols", [])
            data_types = stream_request.get("data_types", ["price", "volume"])
            frequency = stream_request.get("frequency", "1min")
            requesting_agent = stream_request.get("requesting_agent")

            logger.info(f"Starting A2A data stream for {requesting_agent}: {symbols}")

            # Find real-time data sources
            realtime_datasets = await self._find_realtime_datasets(symbols, data_types)

            if not realtime_datasets:
                return {
                    "status": "error",
                    "error": f"No real-time data sources available for {symbols}"
                }

            # Set up streaming configuration
            stream_config = {
                "stream_id": f"stream_{requesting_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "datasets": realtime_datasets,
                "symbols": symbols,
                "data_types": data_types,
                "frequency": frequency,
                "target_agent": requesting_agent,
                "started_at": datetime.now().isoformat()
            }

            # In a real implementation, would set up actual streaming pipeline
            # For now, return configuration that requesting agent can use

            return {
                "status": "success",
                "stream_id": stream_config["stream_id"],
                "stream_config": stream_config,
                "data_sources": len(realtime_datasets),
                "symbols_covered": symbols,
                "estimated_latency_ms": 50,  # Mock latency
                "requesting_agent": requesting_agent
            }

        except Exception as e:
            logger.error(f"A2A data streaming setup failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _mcp_a2a_validate_data_quality(self, quality_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality for calculation agents via A2A messaging"""
        try:
            dataset_info = quality_request.get("dataset_info", {})
            quality_metrics = quality_request.get("required_metrics", ["completeness", "accuracy", "freshness"])
            requesting_agent = quality_request.get("requesting_agent")

            logger.info(f"Validating data quality for {requesting_agent}")

            # Perform quality validation
            validation_results = {
                "dataset_id": dataset_info.get("dataset_id"),
                "validation_timestamp": datetime.now().isoformat(),
                "quality_scores": {},
                "issues_found": [],
                "recommendations": []
            }

            # Completeness check
            if "completeness" in quality_metrics:
                completeness_score = self._check_data_completeness(dataset_info)
                validation_results["quality_scores"]["completeness"] = completeness_score
                if completeness_score < 0.95:
                    validation_results["issues_found"].append(f"Data completeness is {completeness_score:.2%}")
                    validation_results["recommendations"].append("Consider alternative data sources")

            # Accuracy check
            if "accuracy" in quality_metrics:
                accuracy_score = self._check_data_accuracy(dataset_info)
                validation_results["quality_scores"]["accuracy"] = accuracy_score
                if accuracy_score < 0.98:
                    validation_results["issues_found"].append(f"Data accuracy is {accuracy_score:.2%}")

            # Freshness check
            if "freshness" in quality_metrics:
                freshness_score = self._check_data_freshness(dataset_info)
                validation_results["quality_scores"]["freshness"] = freshness_score
                if freshness_score < 0.9:
                    validation_results["issues_found"].append("Data may not be current enough")

            # Overall quality assessment
            avg_score = sum(validation_results["quality_scores"].values()) / len(validation_results["quality_scores"])
            validation_results["overall_quality"] = avg_score
            validation_results["quality_grade"] = "A" if avg_score >= 0.95 else "B" if avg_score >= 0.85 else "C"

            return {
                "status": "success",
                "validation_results": validation_results,
                "requesting_agent": requesting_agent,
                "quality_approved": avg_score >= 0.85
            }

        except Exception as e:
            logger.error(f"A2A data quality validation failed: {e}")
            return {"status": "error", "error": str(e)}

    # Helper methods for A2A data integration

    def _dataset_matches_request(self, dataset: Dict, symbols: List[str], timeframe: str) -> bool:
        """Check if dataset matches the request criteria"""
        try:
            # Check symbol coverage
            dataset_symbols = dataset.get("symbols", [])
            if symbols and not any(symbol in str(dataset_symbols).upper() for symbol in symbols):
                return False

            # Check timeframe compatibility
            dataset_frequency = dataset.get("frequency", "").lower()
            if timeframe == "1min" and "minute" not in dataset_frequency:
                return False
            if timeframe == "1h" and "hour" not in dataset_frequency and "minute" not in dataset_frequency:
                return False

            return True
        except Exception:
            return False

    async def _find_realtime_datasets(self, symbols: List[str], data_types: List[str]) -> List[Dict]:
        """Find datasets suitable for real-time streaming"""
        try:
            # Mock implementation - in reality would query AWS Data Exchange
            realtime_sources = []
            for symbol in symbols[:3]:  # Limit to first 3 symbols
                realtime_sources.append({
                    "dataset_id": f"realtime_{symbol.lower()}_feed",
                    "provider": "Premium Financial Data Co.",
                    "symbol": symbol,
                    "data_types": data_types,
                    "latency_ms": 50,
                    "update_frequency": "real-time"
                })
            return realtime_sources
        except Exception:
            return []

    def _check_data_completeness(self, dataset_info: Dict) -> float:
        """Check data completeness score"""
        # Mock implementation - in reality would analyze actual dataset
        return 0.96  # 96% complete

    def _check_data_accuracy(self, dataset_info: Dict) -> float:
        """Check data accuracy score"""
        # Mock implementation - would validate against trusted sources
        return 0.99  # 99% accurate

    def _check_data_freshness(self, dataset_info: Dict) -> float:
        """Check data freshness score"""
        # Mock implementation - would check last update time
        return 0.92  # 92% fresh


# Factory function for easy agent creation
def create_aws_data_exchange_agent(**kwargs) -> AWSDataExchangeAgent:
    """Factory function to create AWS Data Exchange agent"""
    return AWSDataExchangeAgent(**kwargs)
