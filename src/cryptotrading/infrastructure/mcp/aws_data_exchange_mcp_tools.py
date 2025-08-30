"""
MCP Tools for AWS Data Exchange Agent
Exposes AWS Data Exchange capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ...infrastructure.aws.data_exchange_service import AWSDataExchangeService

logger = logging.getLogger(__name__)


class AWSDataExchangeMCPTools:
    """MCP tools for AWS Data Exchange operations"""

    def __init__(self):
        """Initialize AWS Data Exchange MCP tools"""
        try:
            self.data_exchange_service = AWSDataExchangeService()
            self.service_available = True
        except Exception as e:
            logger.error(f"Failed to initialize AWS Data Exchange service: {e}")
            self.data_exchange_service = None
            self.service_available = False

        self.tools = self._create_tools()

    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions for AWS Data Exchange"""
        return [
            {
                "name": "discover_financial_datasets",
                "description": "Discover available financial datasets from AWS Data Exchange",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_type": {
                            "type": "string",
                            "enum": ["all", "crypto", "economic"],
                            "description": "Type of datasets to discover",
                            "default": "all",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional keywords to filter datasets",
                            "default": [],
                        },
                    },
                },
            },
            {
                "name": "get_dataset_details",
                "description": "Get detailed information about a specific dataset",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "The ID of the dataset to analyze",
                        }
                    },
                    "required": ["dataset_id"],
                },
            },
            {
                "name": "list_dataset_assets",
                "description": "List all assets (files) within a specific dataset",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "The ID of the dataset"}
                    },
                    "required": ["dataset_id"],
                },
            },
            {
                "name": "create_data_export_job",
                "description": "Create a job to export dataset data to S3",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "The ID of the dataset"},
                        "asset_id": {
                            "type": "string",
                            "description": "The ID of the specific asset to export",
                        },
                        "export_destination": {
                            "type": "string",
                            "description": "S3 destination path (optional)",
                            "default": "",
                        },
                    },
                    "required": ["dataset_id", "asset_id"],
                },
            },
            {
                "name": "monitor_export_job",
                "description": "Monitor the status of a data export job",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The ID of the export job to monitor",
                        },
                        "timeout_minutes": {
                            "type": "number",
                            "description": "Timeout in minutes",
                            "default": 30,
                        },
                    },
                    "required": ["job_id"],
                },
            },
            {
                "name": "download_and_process_data",
                "description": "Download and process data from a completed export job",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "The ID of the dataset"},
                        "asset_id": {"type": "string", "description": "The ID of the asset"},
                        "processing_options": {
                            "type": "object",
                            "description": "Data processing options",
                            "properties": {
                                "format": {
                                    "type": "string",
                                    "enum": ["csv", "json", "excel", "parquet"],
                                    "default": "csv",
                                },
                                "clean_data": {"type": "boolean", "default": true},
                                "sample_rows": {
                                    "type": "number",
                                    "description": "Number of sample rows to return (0 for all)",
                                    "default": 0,
                                },
                            },
                        },
                    },
                    "required": ["dataset_id", "asset_id"],
                },
            },
            {
                "name": "load_data_to_database",
                "description": "Load AWS Data Exchange data directly to the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "The ID of the dataset"},
                        "asset_id": {"type": "string", "description": "The ID of the asset"},
                        "table_name": {
                            "type": "string",
                            "description": "Target table name (optional, auto-generated if not provided)",
                        },
                        "load_options": {
                            "type": "object",
                            "description": "Data loading options",
                            "properties": {
                                "if_exists": {
                                    "type": "string",
                                    "enum": ["fail", "replace", "append"],
                                    "default": "append",
                                },
                                "create_index": {"type": "boolean", "default": true},
                            },
                        },
                    },
                    "required": ["dataset_id", "asset_id"],
                },
            },
            {
                "name": "analyze_dataset_quality",
                "description": "Analyze the quality and structure of a dataset",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "The ID of the dataset"},
                        "asset_id": {"type": "string", "description": "The ID of the asset"},
                        "analysis_type": {
                            "type": "string",
                            "enum": ["basic", "comprehensive", "financial_metrics"],
                            "default": "basic",
                        },
                    },
                    "required": ["dataset_id", "asset_id"],
                },
            },
            {
                "name": "get_service_status",
                "description": "Get the current status of the AWS Data Exchange service",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific MCP tool"""
        if not self.service_available:
            return {
                "status": "error",
                "error": "AWS Data Exchange service not available. Check AWS credentials and permissions.",
            }

        try:
            if tool_name == "discover_financial_datasets":
                return await self._discover_financial_datasets(**arguments)
            elif tool_name == "get_dataset_details":
                return await self._get_dataset_details(**arguments)
            elif tool_name == "list_dataset_assets":
                return await self._list_dataset_assets(**arguments)
            elif tool_name == "create_data_export_job":
                return await self._create_data_export_job(**arguments)
            elif tool_name == "monitor_export_job":
                return await self._monitor_export_job(**arguments)
            elif tool_name == "download_and_process_data":
                return await self._download_and_process_data(**arguments)
            elif tool_name == "load_data_to_database":
                return await self._load_data_to_database(**arguments)
            elif tool_name == "analyze_dataset_quality":
                return await self._analyze_dataset_quality(**arguments)
            elif tool_name == "get_service_status":
                return await self._get_service_status(**arguments)
            else:
                return {"status": "error", "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"status": "error", "error": str(e), "tool": tool_name}

    async def _discover_financial_datasets(
        self, dataset_type: str = "all", keywords: List[str] = None
    ) -> Dict[str, Any]:
        """Discover financial datasets"""
        try:
            if dataset_type == "crypto":
                datasets = self.data_exchange_service.get_available_crypto_datasets()
            elif dataset_type == "economic":
                datasets = self.data_exchange_service.get_available_economic_datasets()
            else:
                # Get all financial datasets
                all_datasets = self.data_exchange_service.discover_financial_datasets()
                datasets = [
                    {
                        "dataset_id": ds.dataset_id,
                        "name": ds.name,
                        "description": ds.description,
                        "provider": ds.provider,
                        "last_updated": ds.last_updated.isoformat(),
                        "asset_count": ds.asset_count,
                    }
                    for ds in all_datasets
                ]

            # Additional keyword filtering if provided
            if keywords:
                filtered_datasets = []
                for dataset in datasets:
                    text_content = f"{dataset['name']} {dataset['description']}".lower()
                    if any(keyword.lower() in text_content for keyword in keywords):
                        filtered_datasets.append(dataset)
                datasets = filtered_datasets

            return {
                "status": "success",
                "dataset_type": dataset_type,
                "dataset_count": len(datasets),
                "datasets": datasets,
                "retrieved_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed dataset information"""
        try:
            # Get dataset assets to provide comprehensive details
            assets = self.data_exchange_service.get_dataset_assets(dataset_id)

            asset_details = [
                {
                    "asset_id": asset.asset_id,
                    "name": asset.name,
                    "file_format": asset.file_format,
                    "size_bytes": asset.size_bytes,
                    "size_mb": round(asset.size_bytes / (1024 * 1024), 2),
                    "created_at": asset.created_at.isoformat(),
                }
                for asset in assets
            ]

            total_size_mb = sum(asset["size_mb"] for asset in asset_details)

            return {
                "status": "success",
                "dataset_id": dataset_id,
                "asset_count": len(asset_details),
                "total_size_mb": round(total_size_mb, 2),
                "assets": asset_details,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _list_dataset_assets(self, dataset_id: str) -> Dict[str, Any]:
        """List dataset assets"""
        try:
            assets = self.data_exchange_service.get_dataset_assets(dataset_id)

            asset_list = [
                {
                    "asset_id": asset.asset_id,
                    "name": asset.name,
                    "file_format": asset.file_format,
                    "size_bytes": asset.size_bytes,
                    "created_at": asset.created_at.isoformat(),
                }
                for asset in assets
            ]

            return {"status": "success", "dataset_id": dataset_id, "assets": asset_list}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _create_data_export_job(
        self, dataset_id: str, asset_id: str, export_destination: str = ""
    ) -> Dict[str, Any]:
        """Create data export job"""
        try:
            # Get the latest revision
            revisions = self.data_exchange_service.dataexchange.list_data_set_revisions(
                DataSetId=dataset_id
            )
            if not revisions["Revisions"]:
                return {"status": "error", "error": "No revisions found for dataset"}

            revision_id = revisions["Revisions"][0]["Id"]

            # Create export job
            job_id = self.data_exchange_service.create_data_job(asset_id, dataset_id, revision_id)

            # Start the job
            job_started = self.data_exchange_service.start_data_job(job_id)

            return {
                "status": "success",
                "job_id": job_id,
                "job_started": job_started,
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "revision_id": revision_id,
                "created_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _monitor_export_job(self, job_id: str, timeout_minutes: int = 30) -> Dict[str, Any]:
        """Monitor export job status"""
        try:
            # Wait for job completion
            success = self.data_exchange_service.wait_for_job_completion(job_id, timeout_minutes)

            # Get final status
            status = self.data_exchange_service.get_job_status(job_id)

            return {
                "status": "success",
                "job_completed": success,
                "job_status": status,
                "monitored_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _download_and_process_data(
        self, dataset_id: str, asset_id: str, processing_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Download and process data"""
        try:
            # Download data
            df = self.data_exchange_service.download_and_process_data(dataset_id, asset_id)

            options = processing_options or {}

            # Apply processing options
            if options.get("clean_data", True):
                # Basic data cleaning
                df = df.dropna()

            # Sample data if requested
            sample_rows = options.get("sample_rows", 0)
            if sample_rows > 0:
                df = df.head(sample_rows)

            # Generate data summary
            data_summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
            }

            return {
                "status": "success",
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "data_summary": data_summary,
                "processed_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _load_data_to_database(
        self,
        dataset_id: str,
        asset_id: str,
        table_name: str = None,
        load_options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Load data to database"""
        try:
            # Generate table name if not provided
            if not table_name:
                table_name = f"aws_data_{dataset_id}_{asset_id}"[:50]
                # Sanitize table name
                table_name = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in table_name.lower()
                )

            # Load data to database
            result = self.data_exchange_service.load_dataset_to_database(
                dataset_id, asset_id, table_name
            )

            return {
                "status": result["status"],
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "table_name": table_name,
                "result": result,
                "loaded_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _analyze_dataset_quality(
        self, dataset_id: str, asset_id: str, analysis_type: str = "basic"
    ) -> Dict[str, Any]:
        """Analyze dataset quality"""
        try:
            # Download sample data for analysis
            df = self.data_exchange_service.download_and_process_data(dataset_id, asset_id)

            # Basic quality analysis
            analysis = {
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "analysis_type": analysis_type,
                "shape": df.shape,
                "completeness": {
                    "total_cells": df.size,
                    "missing_cells": df.isnull().sum().sum(),
                    "completeness_rate": 1 - (df.isnull().sum().sum() / df.size),
                },
                "column_analysis": {},
            }

            # Per-column analysis
            for column in df.columns:
                col_data = df[column]
                analysis["column_analysis"][column] = {
                    "dtype": str(col_data.dtype),
                    "null_count": int(col_data.isnull().sum()),
                    "null_percentage": float(col_data.isnull().sum() / len(col_data)),
                    "unique_values": int(col_data.nunique()),
                    "uniqueness_rate": float(col_data.nunique() / len(col_data)),
                }

                # Add numeric statistics if applicable
                if col_data.dtype in ["int64", "float64"]:
                    analysis["column_analysis"][column].update(
                        {
                            "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                            "std": float(col_data.std()) if not col_data.isnull().all() else None,
                            "min": float(col_data.min()) if not col_data.isnull().all() else None,
                            "max": float(col_data.max()) if not col_data.isnull().all() else None,
                        }
                    )

            return {
                "status": "success",
                "analysis": analysis,
                "analyzed_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        try:
            import boto3

            # Check AWS credentials
            try:
                sts = boto3.client("sts")
                identity = sts.get_caller_identity()
                aws_available = True
                aws_account = identity.get("Account")
                aws_user_arn = identity.get("Arn")
                aws_error_msg = None
            except Exception as aws_error:
                aws_available = False
                aws_account = None
                aws_user_arn = None
                aws_error_msg = str(aws_error)

            return {
                "status": "success",
                "service_available": self.service_available,
                "aws_credentials_valid": aws_available,
                "aws_account": aws_account,
                "aws_user_arn": aws_user_arn,
                "error": aws_error_msg,
                "checked_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
