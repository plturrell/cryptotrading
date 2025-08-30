"""
Performance Optimization Module for Technical Analysis
Provides batch processing, memory management, and performance monitoring
"""

import asyncio
import cProfile
import gc
import logging
import pstats
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor performance metrics for TA calculations"""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.memory_usage = []

    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()

    def end_timer(self, operation: str) -> float:
        """End timing and record duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0

    def record_memory_usage(self):
        """Record current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(
            {"timestamp": time.time(), "memory_mb": memory_mb, "cpu_percent": process.cpu_percent()}
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            "operation_times": {},
            "memory_stats": {},
            "cpu_stats": {},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Operation timing statistics
        for operation, times in self.metrics.items():
            report["operation_times"][operation] = {
                "count": len(times),
                "total_time": sum(times),
                "avg_time": np.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "std_time": np.std(times),
            }

        # Memory statistics
        if self.memory_usage:
            memory_values = [m["memory_mb"] for m in self.memory_usage]
            cpu_values = [m["cpu_percent"] for m in self.memory_usage if m["cpu_percent"] > 0]

            report["memory_stats"] = {
                "peak_memory_mb": max(memory_values),
                "avg_memory_mb": np.mean(memory_values),
                "min_memory_mb": min(memory_values),
                "memory_samples": len(memory_values),
            }

            if cpu_values:
                report["cpu_stats"] = {
                    "peak_cpu_percent": max(cpu_values),
                    "avg_cpu_percent": np.mean(cpu_values),
                    "cpu_samples": len(cpu_values),
                }

        return report


# Global performance monitor
perf_monitor = PerformanceMonitor()


def performance_monitor(func):
    """Decorator to monitor function performance"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        operation_name = f"{func.__module__}.{func.__name__}"
        perf_monitor.start_timer(operation_name)
        perf_monitor.record_memory_usage()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            perf_monitor.end_timer(operation_name)
            perf_monitor.record_memory_usage()

    return wrapper


def async_performance_monitor(func):
    """Decorator to monitor async function performance"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        operation_name = f"{func.__module__}.{func.__name__}"
        perf_monitor.start_timer(operation_name)
        perf_monitor.record_memory_usage()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            perf_monitor.end_timer(operation_name)
            perf_monitor.record_memory_usage()

    return wrapper


class BatchProcessor:
    """Efficient batch processing for large datasets"""

    def __init__(self, batch_size: int = 10000, max_workers: int = None):
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)

    @performance_monitor
    def process_large_dataset(
        self, data: pd.DataFrame, processing_func: callable, **kwargs
    ) -> Dict[str, Any]:
        """
        Process large datasets in batches to manage memory

        Args:
            data: Large OHLCV DataFrame
            processing_func: Function to apply to each batch
            **kwargs: Additional arguments for processing function

        Returns:
            Combined results from all batches
        """
        try:
            if len(data) <= self.batch_size:
                return processing_func(data, **kwargs)

            logger.info(f"Processing {len(data)} rows in batches of {self.batch_size}")

            # Split data into batches
            batches = []
            for i in range(0, len(data), self.batch_size):
                batch = data.iloc[i : i + self.batch_size].copy()
                batches.append(batch)

            # Process batches in parallel
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(processing_func, batch, **kwargs): i
                    for i, batch in enumerate(batches)
                }

                for future in future_to_batch:
                    try:
                        batch_result = future.result()
                        if batch_result.get("success", False):
                            results.append(batch_result)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")

            # Combine results
            return self._combine_batch_results(results)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {"success": False, "error": str(e)}

    def _combine_batch_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple batches"""
        if not results:
            return {"success": False, "error": "No successful batches"}

        combined = {
            "success": True,
            "indicators": {},
            "signals": [],
            "batches_processed": len(results),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Combine indicators
        for result in results:
            if "indicators" in result:
                for indicator_name, values in result["indicators"].items():
                    if indicator_name not in combined["indicators"]:
                        combined["indicators"][indicator_name] = []

                    if isinstance(values, list):
                        combined["indicators"][indicator_name].extend(values)

        # Combine signals
        for result in results:
            if "signals" in result and isinstance(result["signals"], list):
                combined["signals"].extend(result["signals"])

        return combined


class MemoryManager:
    """Manage memory usage during TA calculations"""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = max_memory_mb * 0.8

    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        return {
            "current_memory_mb": memory_mb,
            "max_memory_mb": self.max_memory_mb,
            "usage_percent": (memory_mb / self.max_memory_mb) * 100,
            "needs_cleanup": memory_mb > self.cleanup_threshold,
        }

    def cleanup_memory(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()

        # Additional cleanup for pandas
        try:
            import pandas as pd

            pd.core.common._maybe_box_datetimelike = lambda x: x
        except:
            pass

    def memory_efficient_calculation(
        self, data: pd.DataFrame, calculation_func: callable, **kwargs
    ) -> Dict[str, Any]:
        """
        Perform memory-efficient calculations with automatic cleanup

        Args:
            data: Input DataFrame
            calculation_func: Function to perform calculations
            **kwargs: Additional arguments

        Returns:
            Calculation results
        """
        memory_before = self.check_memory_usage()

        try:
            # Check if we need to use chunked processing
            if memory_before["needs_cleanup"]:
                self.cleanup_memory()

            # Perform calculation
            result = calculation_func(data, **kwargs)

            # Check memory after calculation
            memory_after = self.check_memory_usage()

            if memory_after["needs_cleanup"]:
                self.cleanup_memory()

            # Add memory usage info to result
            if isinstance(result, dict):
                result["memory_usage"] = {
                    "before_mb": memory_before["current_memory_mb"],
                    "after_mb": memory_after["current_memory_mb"],
                    "peak_usage_percent": max(
                        memory_before["usage_percent"], memory_after["usage_percent"]
                    ),
                }

            return result

        except MemoryError as e:
            logger.error(f"Memory error during calculation: {e}")
            self.cleanup_memory()
            return {"success": False, "error": "Memory limit exceeded"}
        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return {"success": False, "error": str(e)}


class DataValidator:
    """Validate data quality and accuracy"""

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance  # 1% tolerance for accuracy checks

    @performance_monitor
    def validate_ohlcv_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLCV data quality

        Args:
            data: OHLCV DataFrame

        Returns:
            Validation results
        """
        try:
            validation_results = {
                "success": True,
                "data_quality": {},
                "issues": [],
                "recommendations": [],
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            # Check required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                validation_results["issues"].append(f"Missing columns: {missing_columns}")
                validation_results["success"] = False

            # Check data types
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    validation_results["issues"].append(f"Column {col} is not numeric")

            # Check for missing values
            missing_data = data[required_columns].isnull().sum()
            if missing_data.sum() > 0:
                validation_results["issues"].append(
                    f"Missing values detected: {missing_data.to_dict()}"
                )

            # Check OHLC relationships
            invalid_ohlc = (
                (data["high"] < data["low"])
                | (data["high"] < data["open"])
                | (data["high"] < data["close"])
                | (data["low"] > data["open"])
                | (data["low"] > data["close"])
            ).sum()

            if invalid_ohlc > 0:
                validation_results["issues"].append(
                    f"Invalid OHLC relationships: {invalid_ohlc} rows"
                )

            # Check for zero or negative volumes
            invalid_volume = (data["volume"] <= 0).sum()
            if invalid_volume > 0:
                validation_results["issues"].append(f"Zero/negative volume: {invalid_volume} rows")

            # Data quality metrics
            validation_results["data_quality"] = {
                "total_rows": len(data),
                "date_range": f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else "Empty",
                "missing_data_percent": (missing_data.sum() / len(data)) * 100
                if len(data) > 0
                else 0,
                "invalid_ohlc_percent": (invalid_ohlc / len(data)) * 100 if len(data) > 0 else 0,
                "price_range": {
                    "min": float(data["low"].min()) if "low" in data.columns else 0,
                    "max": float(data["high"].max()) if "high" in data.columns else 0,
                },
                "volume_stats": {
                    "mean": float(data["volume"].mean()) if "volume" in data.columns else 0,
                    "std": float(data["volume"].std()) if "volume" in data.columns else 0,
                },
            }

            # Generate recommendations
            if len(validation_results["issues"]) == 0:
                validation_results["recommendations"].append("Data quality is good")
            else:
                validation_results["recommendations"].extend(
                    [
                        "Clean missing data before analysis",
                        "Validate data source quality",
                        "Consider filtering invalid rows",
                    ]
                )

            return validation_results

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {"success": False, "error": str(e)}

    def validate_indicator_accuracy(
        self, calculated_values: List[float], reference_values: List[float], indicator_name: str
    ) -> Dict[str, Any]:
        """
        Validate indicator calculation accuracy against reference

        Args:
            calculated_values: Our calculated values
            reference_values: Reference values (e.g., from TA-Lib)
            indicator_name: Name of the indicator

        Returns:
            Accuracy validation results
        """
        try:
            if len(calculated_values) != len(reference_values):
                return {
                    "success": False,
                    "error": "Length mismatch between calculated and reference values",
                }

            # Calculate accuracy metrics
            calculated = np.array(calculated_values)
            reference = np.array(reference_values)

            # Remove NaN values for comparison
            valid_mask = ~(np.isnan(calculated) | np.isnan(reference))
            if not np.any(valid_mask):
                return {"success": False, "error": "No valid values for comparison"}

            calc_valid = calculated[valid_mask]
            ref_valid = reference[valid_mask]

            # Calculate error metrics
            absolute_error = np.abs(calc_valid - ref_valid)
            relative_error = np.abs(absolute_error / ref_valid) * 100

            # Remove infinite values
            finite_mask = np.isfinite(relative_error)
            if np.any(finite_mask):
                relative_error = relative_error[finite_mask]

            accuracy_results = {
                "success": True,
                "indicator": indicator_name,
                "validation_metrics": {
                    "total_points": len(calculated_values),
                    "valid_points": len(calc_valid),
                    "mean_absolute_error": float(np.mean(absolute_error)),
                    "max_absolute_error": float(np.max(absolute_error)),
                    "mean_relative_error_percent": float(np.mean(relative_error))
                    if len(relative_error) > 0
                    else 0,
                    "max_relative_error_percent": float(np.max(relative_error))
                    if len(relative_error) > 0
                    else 0,
                    "accuracy_within_tolerance": float(
                        np.mean(relative_error <= self.tolerance * 100)
                    )
                    * 100
                    if len(relative_error) > 0
                    else 0,
                },
                "passed_validation": float(np.mean(relative_error <= self.tolerance * 100)) * 100
                > 95
                if len(relative_error) > 0
                else False,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            return accuracy_results

        except Exception as e:
            logger.error(f"Accuracy validation failed for {indicator_name}: {e}")
            return {"success": False, "error": str(e)}


# Global instances
batch_processor = BatchProcessor()
memory_manager = MemoryManager()
data_validator = DataValidator()


@performance_monitor
def optimize_dataframe_memory(data: pd.DataFrame) -> pd.DataFrame:
    """
    STRAND Tool: Optimize DataFrame memory usage

    Args:
        data: Input DataFrame

    Returns:
        Memory-optimized DataFrame
    """
    try:
        optimized_data = data.copy()

        # Optimize numeric columns
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            col_min = optimized_data[col].min()
            col_max = optimized_data[col].max()

            if optimized_data[col].dtype == "int64":
                if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    optimized_data[col] = optimized_data[col].astype(np.int32)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    optimized_data[col] = optimized_data[col].astype(np.int16)

            elif optimized_data[col].dtype == "float64":
                if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                    optimized_data[col] = optimized_data[col].astype(np.float32)

        return optimized_data

    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        return data


def batch_process_indicators(
    data: pd.DataFrame, processing_func: callable, **kwargs
) -> Dict[str, Any]:
    """
    STRAND Tool: Process indicators in batches for large datasets

    Args:
        data: Large OHLCV DataFrame
        processing_func: Indicator calculation function
        **kwargs: Additional arguments

    Returns:
        Batch processing results
    """
    return batch_processor.process_large_dataset(data, processing_func, **kwargs)


def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    STRAND Tool: Validate OHLCV data quality

    Args:
        data: OHLCV DataFrame to validate

    Returns:
        Data quality validation results
    """
    return data_validator.validate_ohlcv_data(data)


def validate_indicator_accuracy(
    calculated_values: List[float], reference_values: List[float], indicator_name: str
) -> Dict[str, Any]:
    """
    STRAND Tool: Validate indicator calculation accuracy

    Args:
        calculated_values: Our calculated values
        reference_values: Reference values
        indicator_name: Name of the indicator

    Returns:
        Accuracy validation results
    """
    return data_validator.validate_indicator_accuracy(
        calculated_values, reference_values, indicator_name
    )


def get_performance_report() -> Dict[str, Any]:
    """
    STRAND Tool: Get comprehensive performance report

    Returns:
        Performance monitoring report
    """
    return perf_monitor.get_performance_report()


def check_memory_usage() -> Dict[str, Any]:
    """
    STRAND Tool: Check current memory usage

    Returns:
        Memory usage statistics
    """
    return memory_manager.check_memory_usage()


def create_performance_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for performance optimization

    Returns:
        List of tool specifications for STRAND framework
    """
    return [
        {
            "name": "optimize_dataframe_memory",
            "function": optimize_dataframe_memory,
            "description": "Optimize DataFrame memory usage by downcasting numeric types",
            "parameters": {"data": "Input DataFrame to optimize"},
            "category": "performance",
            "skill": "optimization",
        },
        {
            "name": "batch_process_indicators",
            "function": batch_process_indicators,
            "description": "Process indicators in batches for large datasets",
            "parameters": {
                "data": "Large OHLCV DataFrame",
                "processing_func": "Indicator calculation function",
                "**kwargs": "Additional arguments",
            },
            "category": "performance",
            "skill": "batch_processing",
        },
        {
            "name": "validate_data_quality",
            "function": validate_data_quality,
            "description": "Validate OHLCV data quality and detect issues",
            "parameters": {"data": "OHLCV DataFrame to validate"},
            "category": "validation",
            "skill": "data_quality",
        },
        {
            "name": "validate_indicator_accuracy",
            "function": validate_indicator_accuracy,
            "description": "Validate indicator calculation accuracy against reference",
            "parameters": {
                "calculated_values": "Our calculated values",
                "reference_values": "Reference values",
                "indicator_name": "Name of the indicator",
            },
            "category": "validation",
            "skill": "accuracy",
        },
        {
            "name": "get_performance_report",
            "function": get_performance_report,
            "description": "Get comprehensive performance monitoring report",
            "parameters": {},
            "category": "monitoring",
            "skill": "performance",
        },
        {
            "name": "check_memory_usage",
            "function": check_memory_usage,
            "description": "Check current memory usage statistics",
            "parameters": {},
            "category": "monitoring",
            "skill": "memory",
        },
    ]
