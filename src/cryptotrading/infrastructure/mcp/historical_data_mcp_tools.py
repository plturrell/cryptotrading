"""
MCP Tools for A2A Historical Data Loader Agent
Exposes comprehensive historical data loading capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Import historical data components
from ...data.historical.a2a_data_loader import A2AHistoricalDataLoader, DataLoadRequest
from ...data.historical.fred_client import FREDClient
from ...data.historical.yahoo_finance import YahooFinanceClient

logger = logging.getLogger(__name__)


class HistoricalDataMCPTools:
    """MCP tools for A2A Historical Data Loader Agent"""

    def __init__(self):
        self.data_loader = A2AHistoricalDataLoader()
        self.tools = self._create_tools()

    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions"""
        return [
            {
                "name": "load_crypto_historical_data",
                "description": "Load historical cryptocurrency data from Yahoo Finance",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Cryptocurrency symbols (e.g., BTC-USD, ETH-USD)",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format",
                        },
                        "save_data": {
                            "type": "boolean",
                            "description": "Whether to save data to disk",
                            "default": True,
                        },
                    },
                    "required": ["symbols"],
                },
            },
            {
                "name": "load_economic_data",
                "description": "Load economic indicators from FRED (Federal Reserve Economic Data)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "series_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "FRED series IDs (e.g., DGS10, T10Y2Y, WALCL)",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format",
                        },
                        "save_data": {
                            "type": "boolean",
                            "description": "Whether to save data to disk",
                            "default": True,
                        },
                    },
                    "required": ["series_ids"],
                },
            },
            {
                "name": "load_aligned_multi_source_data",
                "description": "Load and temporally align data from multiple sources (Yahoo Finance + FRED)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "crypto_symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Cryptocurrency symbols for Yahoo Finance",
                        },
                        "fred_series": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "FRED economic series IDs",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format",
                        },
                        "frequency": {
                            "type": "string",
                            "description": "Alignment frequency (D=daily, W=weekly, M=monthly)",
                            "default": "D",
                        },
                        "save_data": {
                            "type": "boolean",
                            "description": "Whether to save aligned data to disk",
                            "default": True,
                        },
                    },
                    "required": ["crypto_symbols", "fred_series"],
                },
            },
            {
                "name": "get_data_catalog",
                "description": "Get catalog of available historical data files and sources",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_filter": {
                            "type": "string",
                            "description": "Filter by data source (yahoo, fred, aligned)",
                            "enum": ["yahoo", "fred", "aligned", "all"],
                        },
                        "date_range": {
                            "type": "object",
                            "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
                            "description": "Filter by date range",
                        },
                    },
                },
            },
            {
                "name": "validate_data_quality",
                "description": "Validate quality and completeness of historical data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data_source": {
                            "type": "string",
                            "description": "Data source to validate",
                            "enum": ["yahoo", "fred", "aligned"],
                        },
                        "symbols_or_series": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific symbols or series to validate",
                        },
                        "quality_checks": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "completeness",
                                    "consistency",
                                    "outliers",
                                    "gaps",
                                    "duplicates",
                                ],
                            },
                            "description": "Types of quality checks to perform",
                            "default": ["completeness", "consistency", "gaps"],
                        },
                    },
                    "required": ["data_source"],
                },
            },
            {
                "name": "get_data_statistics",
                "description": "Get comprehensive statistics for historical data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data_source": {
                            "type": "string",
                            "description": "Data source for statistics",
                            "enum": ["yahoo", "fred", "aligned"],
                        },
                        "symbols_or_series": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific symbols or series for statistics",
                        },
                        "statistics_type": {
                            "type": "string",
                            "description": "Type of statistics to compute",
                            "enum": ["basic", "detailed", "correlation", "volatility"],
                            "default": "basic",
                        },
                    },
                    "required": ["data_source"],
                },
            },
        ]

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "load_crypto_historical_data":
                return await self._load_crypto_data(arguments)
            elif tool_name == "load_economic_data":
                return await self._load_economic_data(arguments)
            elif tool_name == "load_aligned_multi_source_data":
                return await self._load_aligned_data(arguments)
            elif tool_name == "get_data_catalog":
                return await self._get_data_catalog(arguments)
            elif tool_name == "validate_data_quality":
                return await self._validate_data_quality(arguments)
            elif tool_name == "get_data_statistics":
                return await self._get_data_statistics(arguments)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}")
            return {"success": False, "error": str(e), "tool": tool_name}

    async def _load_crypto_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Load cryptocurrency historical data"""
        symbols = args["symbols"]
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        save_data = args.get("save_data", True)

        try:
            # Use the data loader's Yahoo client
            results = self.data_loader.yahoo_client.download_multiple(
                symbols, start_date=start_date, end_date=end_date
            )

            # Process results
            data_summary = {}
            total_records = 0

            for symbol, df in results.items():
                if not df.empty:
                    data_summary[symbol] = {
                        "records": len(df),
                        "date_range": {
                            "start": df.index.min().isoformat(),
                            "end": df.index.max().isoformat(),
                        },
                        "columns": list(df.columns),
                        "latest_price": float(df["Close"].iloc[-1])
                        if "Close" in df.columns
                        else None,
                    }
                    total_records += len(df)

            return {
                "success": True,
                "source": "yahoo_finance",
                "symbols_loaded": len(results),
                "total_records": total_records,
                "data_summary": data_summary,
                "saved_to_disk": save_data,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load crypto data: {str(e)}",
                "source": "yahoo_finance",
            }

    async def _load_economic_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Load economic data from FRED"""
        series_ids = args["series_ids"]
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        save_data = args.get("save_data", True)

        try:
            # Use the data loader's FRED client
            results = self.data_loader.fred_client.get_multiple_series(
                series_ids, start_date=start_date, end_date=end_date
            )

            # Process results
            data_summary = {}
            total_records = 0

            for series_id, df in results.items():
                if not df.empty:
                    data_summary[series_id] = {
                        "records": len(df),
                        "date_range": {
                            "start": df.index.min().isoformat(),
                            "end": df.index.max().isoformat(),
                        },
                        "latest_value": float(df.iloc[-1, 0]) if len(df.columns) > 0 else None,
                        "series_info": self.data_loader.fred_client.get_series_info(series_id),
                    }
                    total_records += len(df)

            return {
                "success": True,
                "source": "fred",
                "series_loaded": len(results),
                "total_records": total_records,
                "data_summary": data_summary,
                "saved_to_disk": save_data,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load economic data: {str(e)}",
                "source": "fred",
            }

    async def _load_aligned_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Load and align multi-source data"""
        crypto_symbols = args["crypto_symbols"]
        fred_series = args["fred_series"]
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        frequency = args.get("frequency", "D")
        save_data = args.get("save_data", True)

        try:
            # Create data load request
            request = DataLoadRequest(
                sources=["yahoo", "fred"],
                symbols=crypto_symbols,
                fred_series=fred_series,
                start_date=start_date,
                end_date=end_date,
                align_data=True,
                save_data=save_data,
            )

            # Load aligned data
            result = await self.data_loader.load_data_async(request)

            if result.get("status") == "success":
                aligned_data = result.get("aligned_data", {})

                return {
                    "success": True,
                    "sources": ["yahoo_finance", "fred"],
                    "crypto_symbols": crypto_symbols,
                    "fred_series": fred_series,
                    "alignment_frequency": frequency,
                    "aligned_datasets": len(aligned_data),
                    "common_date_range": result.get("common_date_range"),
                    "data_summary": result.get("data_summary", {}),
                    "saved_to_disk": save_data,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error in data alignment"),
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to load aligned data: {str(e)}"}

    async def _get_data_catalog(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get catalog of available data"""
        source_filter = args.get("source_filter", "all")
        date_range = args.get("date_range")

        try:
            catalog = {
                "yahoo_finance": [],
                "fred": [],
                "aligned": [],
                "summary": {"total_files": 0, "total_size_mb": 0, "date_range": None},
            }

            # Scan data directories
            import os
            from pathlib import Path

            data_dir = self.data_loader.data_dir

            # Yahoo Finance data
            if source_filter in ["yahoo", "all"]:
                yahoo_dir = data_dir / "yahoo"
                if yahoo_dir.exists():
                    for file_path in yahoo_dir.glob("*.csv"):
                        file_stat = file_path.stat()
                        catalog["yahoo_finance"].append(
                            {
                                "filename": file_path.name,
                                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                                "symbol": file_path.stem,
                            }
                        )

            # FRED data
            if source_filter in ["fred", "all"]:
                fred_dir = data_dir / "fred"
                if fred_dir.exists():
                    for file_path in fred_dir.glob("*.csv"):
                        file_stat = file_path.stat()
                        catalog["fred"].append(
                            {
                                "filename": file_path.name,
                                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                                "series_id": file_path.stem,
                            }
                        )

            # Aligned data
            if source_filter in ["aligned", "all"]:
                aligned_dir = data_dir / "aligned"
                if aligned_dir.exists():
                    for file_path in aligned_dir.glob("*.csv"):
                        file_stat = file_path.stat()
                        catalog["aligned"].append(
                            {
                                "filename": file_path.name,
                                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                            }
                        )

            # Calculate summary
            all_files = catalog["yahoo_finance"] + catalog["fred"] + catalog["aligned"]
            catalog["summary"]["total_files"] = len(all_files)
            catalog["summary"]["total_size_mb"] = sum(f["size_mb"] for f in all_files)

            return {"success": True, "catalog": catalog, "filter_applied": source_filter}

        except Exception as e:
            return {"success": False, "error": f"Failed to get data catalog: {str(e)}"}

    async def _validate_data_quality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality"""
        data_source = args["data_source"]
        symbols_or_series = args.get("symbols_or_series", [])
        quality_checks = args.get("quality_checks", ["completeness", "consistency", "gaps"])

        try:
            validation_results = {
                "source": data_source,
                "checks_performed": quality_checks,
                "results": {},
                "overall_score": 0.0,
                "issues_found": [],
            }

            # REAL validation logic - analyze actual data files
            for check in quality_checks:
                try:
                    if check == "completeness":
                        # Actually check data completeness
                        completeness_result = await self._check_data_completeness(
                            data_source, symbols_or_series
                        )
                        validation_results["results"]["completeness"] = completeness_result
                    elif check == "consistency":
                        # Actually check data consistency
                        consistency_result = await self._check_data_consistency(
                            data_source, symbols_or_series
                        )
                        validation_results["results"]["consistency"] = consistency_result
                    elif check == "gaps":
                        # Actually check for data gaps
                        gaps_result = await self._check_data_gaps(data_source, symbols_or_series)
                        validation_results["results"]["gaps"] = gaps_result
                    elif check == "outliers":
                        # Actually detect outliers
                        outliers_result = await self._detect_outliers(
                            data_source, symbols_or_series
                        )
                        validation_results["results"]["outliers"] = outliers_result
                    elif check == "duplicates":
                        # Actually check for duplicates
                        duplicates_result = await self._check_duplicates(
                            data_source, symbols_or_series
                        )
                        validation_results["results"]["duplicates"] = duplicates_result
                except Exception as e:
                    # Log error but continue with other checks
                    logger.error(f"Failed to perform {check} validation: {e}")
                    validation_results["results"][check] = {
                        "score": 0.0,
                        "error": str(e),
                        "details": f"Validation check {check} failed",
                    }

            # Calculate overall score
            scores = [result["score"] for result in validation_results["results"].values()]
            validation_results["overall_score"] = sum(scores) / len(scores) if scores else 0.0

            # Generate issues
            for check, result in validation_results["results"].items():
                if result["score"] < 0.9:
                    validation_results["issues_found"].append(
                        {
                            "check": check,
                            "severity": "medium" if result["score"] > 0.8 else "high",
                            "description": result["details"],
                        }
                    )

            return {"success": True, "validation": validation_results}

        except Exception as e:
            return {"success": False, "error": f"Failed to validate data quality: {str(e)}"}

    async def _get_data_statistics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get data statistics"""
        data_source = args["data_source"]
        symbols_or_series = args.get("symbols_or_series", [])
        statistics_type = args.get("statistics_type", "basic")

        try:
            stats_result = {
                "source": data_source,
                "statistics_type": statistics_type,
                "statistics": {},
            }

            # REAL statistics - compute from actual data files
            try:
                if statistics_type == "basic":
                    stats_result["statistics"] = await self._compute_basic_statistics(
                        data_source, symbols_or_series
                    )
                elif statistics_type == "detailed":
                    stats_result["statistics"] = await self._compute_detailed_statistics(
                        data_source, symbols_or_series
                    )
                elif statistics_type == "correlation":
                    stats_result["statistics"] = await self._compute_correlation_statistics(
                        data_source, symbols_or_series
                    )
                elif statistics_type == "volatility":
                    stats_result["statistics"] = await self._compute_volatility_statistics(
                        data_source, symbols_or_series
                    )
                else:
                    return {
                        "success": False,
                        "error": f"Unknown statistics type: {statistics_type}",
                    }
            except Exception as e:
                logger.error(f"Failed to compute {statistics_type} statistics: {e}")
                return {"success": False, "error": f"Statistics computation failed: {str(e)}"}

            return {"success": True, "statistics": stats_result}

        except Exception as e:
            return {"success": False, "error": f"Failed to get data statistics: {str(e)}"}

    async def _check_data_completeness(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Check actual data completeness"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            if not data_dir.exists():
                return {
                    "score": 0.0,
                    "missing_data_percentage": 100.0,
                    "details": f"No {data_source} data directory found",
                }

            total_files = len(list(data_dir.glob("*.csv")))
            if total_files == 0:
                return {
                    "score": 0.0,
                    "missing_data_percentage": 100.0,
                    "details": "No data files found",
                }

            missing_count = 0
            for symbol in symbols_or_series:
                file_path = data_dir / f"{symbol}.csv"
                if not file_path.exists():
                    missing_count += 1

            missing_percentage = (
                (missing_count / len(symbols_or_series)) * 100 if symbols_or_series else 0
            )
            score = max(0.0, 1.0 - (missing_percentage / 100))

            return {
                "score": score,
                "missing_data_percentage": missing_percentage,
                "details": f"{missing_count}/{len(symbols_or_series)} files missing"
                if symbols_or_series
                else f"{total_files} files available",
            }
        except Exception as e:
            return {"score": 0.0, "error": str(e), "details": "Completeness check failed"}

    async def _check_data_consistency(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Check actual data consistency"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            inconsistencies = 0
            files_checked = 0

            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    files_checked += 1

                    # Check for data type inconsistencies
                    if "Close" in df.columns and not pd.api.types.is_numeric_dtype(df["Close"]):
                        inconsistencies += 1
                    if "Volume" in df.columns and not pd.api.types.is_numeric_dtype(df["Volume"]):
                        inconsistencies += 1

                except Exception:
                    inconsistencies += 1

            score = max(0.0, 1.0 - (inconsistencies / max(1, files_checked)))
            return {
                "score": score,
                "inconsistencies": inconsistencies,
                "details": f"{inconsistencies} inconsistencies found in {files_checked} files",
            }
        except Exception as e:
            return {"score": 0.0, "error": str(e), "details": "Consistency check failed"}

    async def _check_data_gaps(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Check for actual data gaps"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            total_gaps = 0
            files_checked = 0

            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
                    files_checked += 1

                    # Check for date gaps (more than 1 day for daily data)
                    date_diffs = df.index.to_series().diff()
                    gaps = (date_diffs > pd.Timedelta(days=2)).sum()
                    total_gaps += gaps

                except Exception:
                    total_gaps += 1

            score = max(
                0.0, 1.0 - (total_gaps / max(1, files_checked * 10))
            )  # Allow up to 10 gaps per file
            return {
                "score": score,
                "gaps_found": total_gaps,
                "details": f"{total_gaps} date gaps found across {files_checked} files",
            }
        except Exception as e:
            return {"score": 0.0, "error": str(e), "details": "Gap check failed"}

    async def _detect_outliers(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Detect actual outliers in data"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            total_outliers = 0
            total_records = 0

            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    if "Close" in df.columns:
                        close_prices = df["Close"].dropna()
                        total_records += len(close_prices)

                        # Use IQR method for outlier detection
                        Q1 = close_prices.quantile(0.25)
                        Q3 = close_prices.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = (
                            (close_prices < (Q1 - 1.5 * IQR)) | (close_prices > (Q3 + 1.5 * IQR))
                        ).sum()
                        total_outliers += outliers

                except Exception:
                    continue

            outlier_percentage = (total_outliers / max(1, total_records)) * 100
            score = max(0.0, 1.0 - (outlier_percentage / 10))  # Penalize if >10% outliers

            return {
                "score": score,
                "outliers_detected": total_outliers,
                "details": f"{total_outliers} outliers detected in {total_records} records ({outlier_percentage:.2f}%)",
            }
        except Exception as e:
            return {"score": 0.0, "error": str(e), "details": "Outlier detection failed"}

    async def _check_duplicates(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Check for actual duplicate records"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            total_duplicates = 0
            files_checked = 0

            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    files_checked += 1
                    duplicates = df.duplicated().sum()
                    total_duplicates += duplicates
                except Exception:
                    continue

            score = (
                1.0
                if total_duplicates == 0
                else max(0.0, 1.0 - (total_duplicates / max(1, files_checked * 100)))
            )
            return {
                "score": score,
                "duplicates_found": total_duplicates,
                "details": f"{total_duplicates} duplicate records found across {files_checked} files",
            }
        except Exception as e:
            return {"score": 0.0, "error": str(e), "details": "Duplicate check failed"}

    async def _compute_basic_statistics(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Compute actual basic statistics"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            total_records = 0
            date_ranges = []
            all_columns = set()

            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
                    total_records += len(df)
                    all_columns.update(df.columns)
                    if not df.empty:
                        date_ranges.extend([df.index.min(), df.index.max()])
                except Exception:
                    continue

            return {
                "record_count": total_records,
                "date_range": {
                    "start": min(date_ranges).isoformat() if date_ranges else None,
                    "end": max(date_ranges).isoformat() if date_ranges else None,
                },
                "columns": list(all_columns),
                "data_types": {
                    "numeric": len(
                        [
                            col
                            for col in all_columns
                            if col in ["Open", "High", "Low", "Close", "Volume"]
                        ]
                    ),
                    "datetime": 1,
                    "categorical": len(all_columns)
                    - len(
                        [
                            col
                            for col in all_columns
                            if col in ["Open", "High", "Low", "Close", "Volume"]
                        ]
                    )
                    - 1,
                },
            }
        except Exception as e:
            return {"error": str(e), "details": "Basic statistics computation failed"}

    async def _compute_detailed_statistics(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Compute actual detailed statistics"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            all_close_prices = []

            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    if "Close" in df.columns:
                        all_close_prices.extend(df["Close"].dropna().tolist())
                except Exception:
                    continue

            if not all_close_prices:
                return {
                    "error": "No price data found",
                    "details": "Cannot compute detailed statistics",
                }

            prices = pd.Series(all_close_prices)
            return {
                "descriptive": {
                    "mean": float(prices.mean()),
                    "median": float(prices.median()),
                    "std": float(prices.std()),
                    "min": float(prices.min()),
                    "max": float(prices.max()),
                    "skewness": float(prices.skew()),
                    "kurtosis": float(prices.kurtosis()),
                },
                "distribution": {
                    "normal_test_available": False,  # Would need scipy for proper test
                    "is_normal": abs(prices.skew()) < 0.5 and abs(prices.kurtosis()) < 3,
                },
            }
        except Exception as e:
            return {"error": str(e), "details": "Detailed statistics computation failed"}

    async def _compute_correlation_statistics(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Compute actual correlation statistics"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            price_data = {}

            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
                    if "Close" in df.columns:
                        symbol = csv_file.stem
                        price_data[symbol] = df["Close"]
                except Exception:
                    continue

            if len(price_data) < 2:
                return {
                    "error": "Insufficient data for correlation",
                    "details": "Need at least 2 symbols",
                }

            # Align data and compute correlations
            combined_df = pd.DataFrame(price_data).dropna()
            corr_matrix = combined_df.corr()

            # Extract correlation pairs
            correlations = {}
            strongest_correlations = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    pair = f"{corr_matrix.columns[i]}-{corr_matrix.columns[j]}"
                    corr_value = corr_matrix.iloc[i, j]
                    correlations[pair] = float(corr_value)
                    strongest_correlations.append({"pair": pair, "correlation": float(corr_value)})

            # Sort by absolute correlation
            strongest_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            return {
                "correlation_matrix": correlations,
                "strongest_correlations": strongest_correlations[:5],  # Top 5
            }
        except Exception as e:
            return {"error": str(e), "details": "Correlation computation failed"}

    async def _compute_volatility_statistics(
        self, data_source: str, symbols_or_series: List[str]
    ) -> Dict[str, Any]:
        """Compute actual volatility statistics"""
        try:
            data_dir = self.data_loader.data_dir / data_source
            all_returns = []

            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
                    if "Close" in df.columns and len(df) > 1:
                        returns = df["Close"].pct_change().dropna()
                        all_returns.extend(returns.tolist())
                except Exception:
                    continue

            if not all_returns:
                return {"error": "No return data found", "details": "Cannot compute volatility"}

            returns_series = pd.Series(all_returns)
            daily_vol = float(returns_series.std())
            annualized_vol = daily_vol * (252**0.5)  # Assuming 252 trading days

            # Simple volatility regime classification
            if annualized_vol > 0.5:
                regime = "high"
            elif annualized_vol > 0.2:
                regime = "medium"
            else:
                regime = "low"

            return {
                "daily_volatility": daily_vol,
                "annualized_volatility": annualized_vol,
                "volatility_regime": regime,
                "garch_parameters": {
                    "note": "GARCH parameters require specialized estimation",
                    "available": False,
                },
            }
        except Exception as e:
            return {"error": str(e), "details": "Volatility computation failed"}


# Export for MCP server registration
historical_data_mcp_tools = HistoricalDataMCPTools()
