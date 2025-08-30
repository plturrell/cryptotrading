"""
MCP Tools for Feature Store
Exposes ML feature engineering and computation capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import Feature Store components
from ...core.ml.feature_store import FeatureDefinition, feature_store

logger = logging.getLogger(__name__)


class FeatureStoreMCPTools:
    """MCP tools for Feature Store operations"""

    def __init__(self):
        self.feature_store = feature_store
        self.tools = self._create_tools()

    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions"""
        return [
            {
                "name": "compute_features",
                "description": "Compute technical features for a symbol",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol (e.g., BTC-USD)",
                        },
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of features to compute (optional, computes all if not specified)",
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_feature_vector",
                "description": "Get feature vector for a specific point in time",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Trading symbol"},
                        "timestamp": {
                            "type": "string",
                            "description": "ISO timestamp (optional, uses latest if not specified)",
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_training_features",
                "description": "Get features for multiple symbols for training purposes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of trading symbols",
                        },
                        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                    },
                    "required": ["symbols", "start_date", "end_date"],
                },
            },
            {
                "name": "get_feature_definitions",
                "description": "Get available feature definitions and metadata",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "feature_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific features to get info for (optional, returns all if not specified)",
                        }
                    },
                },
            },
            {
                "name": "get_feature_importance",
                "description": "Get feature importance scores from latest models",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "compute_features":
                return await self._compute_features(arguments)
            elif tool_name == "get_feature_vector":
                return await self._get_feature_vector(arguments)
            elif tool_name == "get_training_features":
                return await self._get_training_features(arguments)
            elif tool_name == "get_feature_definitions":
                return await self._get_feature_definitions(arguments)
            elif tool_name == "get_feature_importance":
                return await self._get_feature_importance(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Feature Store MCP tool error: {e}")
            return {"error": str(e)}

    async def _compute_features(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features for a symbol"""
        try:
            symbol = args["symbol"]
            features = args.get("features")

            result_df = await self.feature_store.compute_features(symbol, features)

            if result_df.empty:
                return {
                    "success": False,
                    "error": f"No features computed for {symbol}",
                    "symbol": symbol,
                }

            return {
                "success": True,
                "symbol": symbol,
                "features": {
                    "data": result_df.to_dict("records"),
                    "columns": result_df.columns.tolist(),
                    "shape": result_df.shape,
                    "latest_values": result_df.iloc[-1].to_dict() if not result_df.empty else {},
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_feature_vector(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get feature vector for a specific point in time"""
        try:
            symbol = args["symbol"]
            timestamp_str = args.get("timestamp")

            timestamp = None
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            feature_vector = await self.feature_store.get_feature_vector(symbol, timestamp)

            return {
                "success": True,
                "symbol": symbol,
                "timestamp": timestamp.isoformat() if timestamp else "latest",
                "feature_vector": feature_vector,
                "feature_count": len(feature_vector),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_training_features(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get training features for multiple symbols"""
        try:
            symbols = args["symbols"]
            start_date = args["start_date"]
            end_date = args["end_date"]

            training_df = await self.feature_store.get_training_features(
                symbols, start_date, end_date
            )

            if training_df.empty:
                return {
                    "success": False,
                    "error": "No training features available for the specified parameters",
                }

            return {
                "success": True,
                "symbols": symbols,
                "date_range": {"start": start_date, "end": end_date},
                "training_data": {
                    "shape": training_df.shape,
                    "columns": training_df.columns.tolist(),
                    "sample_data": training_df.head(10).to_dict("records"),
                    "statistics": training_df.describe().to_dict(),
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_feature_definitions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get feature definitions and metadata"""
        try:
            feature_names = args.get("feature_names")

            if feature_names:
                # Get specific features
                definitions = {}
                for name in feature_names:
                    if name in self.feature_store.features:
                        feature_def = self.feature_store.features[name]
                        definitions[name] = {
                            "name": feature_def.name,
                            "dtype": feature_def.dtype,
                            "description": feature_def.description,
                            "dependencies": feature_def.dependencies,
                            "metadata": feature_def.metadata,
                        }
            else:
                # Get all features
                definitions = {}
                for name, feature_def in self.feature_store.features.items():
                    definitions[name] = {
                        "name": feature_def.name,
                        "dtype": feature_def.dtype,
                        "description": feature_def.description,
                        "dependencies": feature_def.dependencies,
                        "metadata": feature_def.metadata,
                    }

            return {
                "success": True,
                "feature_definitions": definitions,
                "total_features": len(definitions),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_feature_importance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get feature importance scores"""
        try:
            importance_scores = self.feature_store.get_feature_importance()

            # Sort by importance
            sorted_importance = dict(
                sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            )

            return {
                "success": True,
                "feature_importance": sorted_importance,
                "top_features": list(sorted_importance.keys())[:10],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Global instance for MCP server registration
feature_store_mcp_tools = FeatureStoreMCPTools()
