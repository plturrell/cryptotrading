"""
MCP Protocol Extensions for Historical Data Ingestion

Provides standardized tools and resources for coordinating distributed 
historical data collection with validation and quality checks.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..resources import Resource
from ..tools import MCPTool, ToolResult


class DataSource(Enum):
    """Supported data sources"""

    YAHOO = "yahoo"
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    POLYGON = "polygon"
    ALPHAVANTAGE = "alphavantage"
    CRYPTODATADOWNLOAD = "cryptodatadownload"


@dataclass
class DataIngestionTool(MCPTool):
    """Tool for coordinating distributed historical data ingestion"""

    name = "data_ingestion_coordinator"
    description = "Coordinate distributed historical data ingestion with validation"
    parameters = {
        "sources": {
            "type": "array",
            "items": {"type": "string", "enum": [s.value for s in DataSource]},
            "description": "Data sources to ingest from",
        },
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Trading symbols (e.g., BTC-USD, ETH-USD)",
        },
        "date_range": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "format": "date"},
                "end": {"type": "string", "format": "date"},
            },
            "required": ["start", "end"],
        },
        "interval": {
            "type": "string",
            "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "default": "1h",
        },
        "parallel_workers": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
        "quality_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.95},
        "validation_rules": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["completeness", "price_validity", "volume_consistency", "time_gaps"],
            },
            "default": ["completeness", "price_validity"],
        },
    }


@dataclass
class DataValidationTool(MCPTool):
    """Tool for validating ingested data quality"""

    name = "data_quality_validator"
    description = "Validate data quality and generate quality reports"
    parameters = {
        "dataset_id": {"type": "string", "description": "Unique identifier for the dataset"},
        "validation_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_type": {
                        "type": "string",
                        "enum": ["completeness", "accuracy", "consistency", "timeliness"],
                    },
                    "parameters": {"type": "object"},
                    "severity": {
                        "type": "string",
                        "enum": ["info", "warning", "error", "critical"],
                        "default": "error",
                    },
                },
            },
        },
        "sample_size": {
            "type": "number",
            "minimum": 0.01,
            "maximum": 1.0,
            "default": 0.1,
            "description": "Percentage of data to sample for validation",
        },
    }


@dataclass
class DataStreamingTool(MCPTool):
    """Tool for streaming validated data to consumers"""

    name = "data_streamer"
    description = "Stream validated historical data with flow control"
    parameters = {
        "dataset_id": {"type": "string"},
        "consumer_id": {"type": "string"},
        "filters": {
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}},
                "quality_score_min": {"type": "number", "minimum": 0, "maximum": 1},
                "date_range": {"type": "object"},
            },
        },
        "streaming_config": {
            "type": "object",
            "properties": {
                "batch_size": {"type": "integer", "default": 1000},
                "max_rate_per_second": {"type": "integer", "default": 10000},
                "compression": {
                    "type": "string",
                    "enum": ["none", "gzip", "zstd"],
                    "default": "gzip",
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "msgpack", "parquet"],
                    "default": "msgpack",
                },
            },
        },
    }


class DataIngestionStatusResource(Resource):
    """Resource for monitoring data ingestion progress"""

    def __init__(self):
        super().__init__(
            uri="ingestion://status",
            name="Data Ingestion Status",
            description="Current status of all data ingestion jobs",
            mime_type="application/json",
        )
        self.jobs = {}

    async def read(self) -> str:
        """Return current ingestion status"""
        import json

        status = {
            "active_jobs": len([j for j in self.jobs.values() if j["status"] == "running"]),
            "completed_jobs": len([j for j in self.jobs.values() if j["status"] == "completed"]),
            "failed_jobs": len([j for j in self.jobs.values() if j["status"] == "failed"]),
            "jobs": self.jobs,
        }
        return json.dumps(status, indent=2)


class DataQualityReportResource(Resource):
    """Resource for accessing data quality reports"""

    def __init__(self):
        super().__init__(
            uri="quality://reports",
            name="Data Quality Reports",
            description="Historical data quality validation reports",
            mime_type="application/json",
        )
        self.reports = {}

    async def read(self) -> str:
        """Return quality reports"""
        import json

        return json.dumps(self.reports, indent=2, default=str)
