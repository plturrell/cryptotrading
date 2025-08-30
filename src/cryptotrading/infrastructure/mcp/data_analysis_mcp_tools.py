"""
MCP Tools for Data Analysis
Exposes data analysis and computation capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Import data analysis components
from ...core.agents.specialized.mcts_calculation_agent import DataAnalysisEnvironment
from ...core.data_ingestion.quality_validator import FactorQualityValidator
from ...core.processing.parallel_executor import ParallelExecutor

logger = logging.getLogger(__name__)


class DataAnalysisMCPTools:
    """MCP tools for data analysis operations"""

    def __init__(self):
        self.quality_validator = FactorQualityValidator()
        self.parallel_executor = ParallelExecutor()
        self.tools = self._create_tools()

    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions"""
        return [
            {
                "name": "validate_data_quality",
                "description": "Validate data quality for factors and indicators",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to validate (JSON format)"},
                        "factor_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of factor names to validate",
                        },
                        "validation_rules": {
                            "type": "object",
                            "description": "Custom validation rules (optional)",
                        },
                    },
                    "required": ["data", "factor_names"],
                },
            },
            {
                "name": "analyze_data_distribution",
                "description": "Analyze statistical distribution of data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to analyze (JSON format)"},
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to analyze (optional, analyzes all if not specified)",
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "compute_correlation_matrix",
                "description": "Compute correlation matrix for numerical data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Data for correlation analysis (JSON format)",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman", "kendall"],
                            "default": "pearson",
                            "description": "Correlation method",
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "detect_outliers",
                "description": "Detect outliers in numerical data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Data to analyze for outliers (JSON format)",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["iqr", "zscore", "isolation_forest"],
                            "default": "iqr",
                            "description": "Outlier detection method",
                        },
                        "threshold": {
                            "type": "number",
                            "default": 3.0,
                            "description": "Threshold for outlier detection",
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "compute_rolling_statistics",
                "description": "Compute rolling statistics for time series data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Time series data (JSON format)"},
                        "window": {
                            "type": "integer",
                            "default": 20,
                            "description": "Rolling window size",
                        },
                        "statistics": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["mean", "std", "min", "max", "median", "skew", "kurt"],
                            },
                            "default": ["mean", "std"],
                            "description": "Statistics to compute",
                        },
                    },
                    "required": ["data"],
                },
            },
        ]

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "validate_data_quality":
                return await self._validate_data_quality(arguments)
            elif tool_name == "analyze_data_distribution":
                return await self._analyze_data_distribution(arguments)
            elif tool_name == "compute_correlation_matrix":
                return await self._compute_correlation_matrix(arguments)
            elif tool_name == "detect_outliers":
                return await self._detect_outliers(arguments)
            elif tool_name == "compute_rolling_statistics":
                return await self._compute_rolling_statistics(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Data Analysis MCP tool error: {e}")
            return {"error": str(e)}

    async def _validate_data_quality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality for factors"""
        try:
            data_dict = args["data"]
            factor_names = args["factor_names"]
            validation_rules = args.get("validation_rules", {})

            # Convert to DataFrame
            df = pd.DataFrame(data_dict)

            validation_results = {}
            overall_score = 0.0

            for factor_name in factor_names:
                if factor_name not in df.columns:
                    validation_results[factor_name] = {
                        "error": f"Factor {factor_name} not found in data"
                    }
                    continue

                # Validate factor
                result = await asyncio.to_thread(
                    self.quality_validator.validate_factor,
                    factor_name,
                    df[factor_name],
                    validation_rules.get(factor_name, {}),
                )

                validation_results[factor_name] = result
                overall_score += result.get("score", 0.0)

            overall_score = overall_score / len(factor_names) if factor_names else 0.0

            return {
                "success": True,
                "validation_results": validation_results,
                "overall_score": overall_score,
                "quality_grade": self._get_quality_grade(overall_score),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _analyze_data_distribution(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statistical distribution of data"""
        try:
            data_dict = args["data"]
            columns = args.get("columns")

            # Convert to DataFrame
            df = pd.DataFrame(data_dict)

            if columns:
                df = df[columns]

            # Select only numerical columns
            numerical_df = df.select_dtypes(include=[np.number])

            if numerical_df.empty:
                return {
                    "success": False,
                    "error": "No numerical columns found for distribution analysis",
                }

            distribution_analysis = {}

            for column in numerical_df.columns:
                series = numerical_df[column].dropna()

                if len(series) == 0:
                    continue

                distribution_analysis[column] = {
                    "count": len(series),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "missing_values": int(df[column].isna().sum()),
                    "missing_percentage": float(df[column].isna().sum() / len(df) * 100),
                }

            return {
                "success": True,
                "distribution_analysis": distribution_analysis,
                "total_columns_analyzed": len(distribution_analysis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _compute_correlation_matrix(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compute correlation matrix"""
        try:
            data_dict = args["data"]
            method = args.get("method", "pearson")

            # Convert to DataFrame
            df = pd.DataFrame(data_dict)

            # Select only numerical columns
            numerical_df = df.select_dtypes(include=[np.number])

            if numerical_df.empty:
                return {
                    "success": False,
                    "error": "No numerical columns found for correlation analysis",
                }

            # Compute correlation matrix
            correlation_matrix = numerical_df.corr(method=method)

            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_correlations.append(
                            {
                                "variable1": correlation_matrix.columns[i],
                                "variable2": correlation_matrix.columns[j],
                                "correlation": float(corr_value),
                            }
                        )

            return {
                "success": True,
                "correlation_matrix": correlation_matrix.to_dict(),
                "high_correlations": high_correlations,
                "method": method,
                "variables_analyzed": list(numerical_df.columns),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _detect_outliers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in data"""
        try:
            data_dict = args["data"]
            method = args.get("method", "iqr")
            threshold = args.get("threshold", 3.0)

            # Convert to DataFrame
            df = pd.DataFrame(data_dict)

            # Select only numerical columns
            numerical_df = df.select_dtypes(include=[np.number])

            if numerical_df.empty:
                return {
                    "success": False,
                    "error": "No numerical columns found for outlier detection",
                }

            outlier_results = {}

            for column in numerical_df.columns:
                series = numerical_df[column].dropna()

                if len(series) == 0:
                    continue

                outliers = []

                if method == "iqr":
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (series < lower_bound) | (series > upper_bound)

                elif method == "zscore":
                    z_scores = np.abs((series - series.mean()) / series.std())
                    outlier_mask = z_scores > threshold

                elif method == "isolation_forest":
                    from sklearn.ensemble import IsolationForest

                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    outlier_mask = outlier_labels == -1

                outlier_indices = series[outlier_mask].index.tolist()
                outlier_values = series[outlier_mask].tolist()

                outlier_results[column] = {
                    "outlier_count": len(outlier_indices),
                    "outlier_percentage": float(len(outlier_indices) / len(series) * 100),
                    "outlier_indices": outlier_indices,
                    "outlier_values": outlier_values,
                    "method": method,
                }

            return {
                "success": True,
                "outlier_results": outlier_results,
                "method": method,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _compute_rolling_statistics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compute rolling statistics for time series"""
        try:
            data_dict = args["data"]
            window = args.get("window", 20)
            statistics = args.get("statistics", ["mean", "std"])

            # Convert to DataFrame
            df = pd.DataFrame(data_dict)

            # Select only numerical columns
            numerical_df = df.select_dtypes(include=[np.number])

            if numerical_df.empty:
                return {
                    "success": False,
                    "error": "No numerical columns found for rolling statistics",
                }

            rolling_results = {}

            for column in numerical_df.columns:
                series = numerical_df[column]
                rolling_window = series.rolling(window=window)

                column_stats = {}

                for stat in statistics:
                    if stat == "mean":
                        column_stats[f"rolling_{stat}"] = rolling_window.mean().tolist()
                    elif stat == "std":
                        column_stats[f"rolling_{stat}"] = rolling_window.std().tolist()
                    elif stat == "min":
                        column_stats[f"rolling_{stat}"] = rolling_window.min().tolist()
                    elif stat == "max":
                        column_stats[f"rolling_{stat}"] = rolling_window.max().tolist()
                    elif stat == "median":
                        column_stats[f"rolling_{stat}"] = rolling_window.median().tolist()
                    elif stat == "skew":
                        column_stats[f"rolling_{stat}"] = rolling_window.skew().tolist()
                    elif stat == "kurt":
                        column_stats[f"rolling_{stat}"] = rolling_window.kurt().tolist()

                rolling_results[column] = column_stats

            return {
                "success": True,
                "rolling_statistics": rolling_results,
                "window_size": window,
                "statistics_computed": statistics,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"


# Global instance for MCP server registration
data_analysis_mcp_tools = DataAnalysisMCPTools()
