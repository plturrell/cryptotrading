"""
Parallel processing executor with environment-aware optimizations
Handles both local development and Vercel serverless environments
"""

import asyncio
import concurrent.futures
import functools
import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from ..config.environment import get_feature_flags, is_serverless, is_vercel

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of parallel processing operation"""

    success: bool
    data: Any
    execution_time: float
    worker_id: Optional[str] = None
    error: Optional[str] = None


class MemoryOptimizedExecutor:
    """Memory-optimized executor for serverless environments"""

    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.current_memory_usage = 0
        self._lock = threading.Lock()

    def check_memory_usage(self) -> bool:
        """Check if we're within memory limits"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < self.max_memory_mb * 0.8  # 80% threshold
        except ImportError:
            # Fallback if psutil not available
            return True

    def cleanup_memory(self):
        """Force garbage collection to free memory"""
        import gc

        gc.collect()


class ParallelExecutor:
    """Environment-aware parallel execution handler"""

    def __init__(self):
        self.flags = get_feature_flags()
        self.memory_optimizer = MemoryOptimizedExecutor(self.flags.memory_limit_mb)
        self._thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None

        logger.info(
            f"ParallelExecutor initialized: "
            f"parallel_processing={self.flags.enable_parallel_processing}, "
            f"max_workers={self.flags.max_workers}, "
            f"serverless={is_serverless()}"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    async def execute_async_batch(
        self, tasks: List[Coroutine], batch_size: Optional[int] = None
    ) -> List[ProcessingResult]:
        """Execute async tasks in batches with memory management"""
        if not tasks:
            return []

        batch_size = batch_size or self.flags.batch_size
        results = []

        # Process in batches to manage memory
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            start_time = time.time()

            try:
                # Check memory before processing batch
                if not self.memory_optimizer.check_memory_usage():
                    logger.warning("Memory usage high, forcing cleanup")
                    self.memory_optimizer.cleanup_memory()

                if self.flags.enable_parallel_processing and len(batch) > 1:
                    # Parallel execution with semaphore for resource control
                    semaphore = asyncio.Semaphore(self.flags.max_workers)

                    async def limited_task(task):
                        async with semaphore:
                            return await task

                    batch_results = await asyncio.gather(
                        *[limited_task(task) for task in batch], return_exceptions=True
                    )
                else:
                    # Sequential execution for serverless or small batches
                    batch_results = []
                    for task in batch:
                        try:
                            result = await task
                            batch_results.append(result)
                        except Exception as e:
                            batch_results.append(e)

                # Process results
                execution_time = time.time() - start_time
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        results.append(
                            ProcessingResult(
                                success=False,
                                data=None,
                                execution_time=execution_time / len(batch),
                                worker_id=f"async_worker_{i+j}",
                                error=str(result),
                            )
                        )
                    else:
                        results.append(
                            ProcessingResult(
                                success=True,
                                data=result,
                                execution_time=execution_time / len(batch),
                                worker_id=f"async_worker_{i+j}",
                            )
                        )

                # Memory cleanup after each batch
                if is_serverless():
                    self.memory_optimizer.cleanup_memory()

            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                # Add error results for the entire batch
                for j in range(len(batch)):
                    results.append(
                        ProcessingResult(
                            success=False,
                            data=None,
                            execution_time=0,
                            worker_id=f"async_worker_{i+j}",
                            error=str(e),
                        )
                    )

        return results

    def execute_sync_batch(
        self, func: Callable, items: List[Any], use_threads: bool = True
    ) -> List[ProcessingResult]:
        """Execute synchronous function on batch of items"""
        if not items:
            return []

        if not self.flags.enable_parallel_processing or len(items) == 1:
            # Sequential execution
            return self._execute_sequential(func, items)

        if use_threads and not is_serverless():
            return self._execute_with_threads(func, items)
        else:
            return self._execute_sequential(func, items)

    def _execute_sequential(self, func: Callable, items: List[Any]) -> List[ProcessingResult]:
        """Sequential execution fallback"""
        results = []

        for i, item in enumerate(items):
            start_time = time.time()
            try:
                result = func(item)
                results.append(
                    ProcessingResult(
                        success=True,
                        data=result,
                        execution_time=time.time() - start_time,
                        worker_id=f"sequential_{i}",
                    )
                )
            except Exception as e:
                logger.error(f"Sequential execution failed for item {i}: {e}")
                results.append(
                    ProcessingResult(
                        success=False,
                        data=None,
                        execution_time=time.time() - start_time,
                        worker_id=f"sequential_{i}",
                        error=str(e),
                    )
                )

        return results

    def _execute_with_threads(self, func: Callable, items: List[Any]) -> List[ProcessingResult]:
        """Thread-based parallel execution"""
        if self._thread_pool is None:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.flags.max_workers
            )

        results = []
        batch_size = self.flags.batch_size

        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            start_time = time.time()

            try:
                future_to_item = {
                    self._thread_pool.submit(func, item): (j, item) for j, item in enumerate(batch)
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    j, item = future_to_item[future]
                    try:
                        result = future.result()
                        results.append(
                            ProcessingResult(
                                success=True,
                                data=result,
                                execution_time=time.time() - start_time,
                                worker_id=f"thread_{i+j}",
                            )
                        )
                    except Exception as e:
                        logger.error(f"Thread execution failed for item {i+j}: {e}")
                        results.append(
                            ProcessingResult(
                                success=False,
                                data=None,
                                execution_time=time.time() - start_time,
                                worker_id=f"thread_{i+j}",
                                error=str(e),
                            )
                        )

            except Exception as e:
                logger.error(f"Thread batch execution failed: {e}")
                for j in range(len(batch)):
                    results.append(
                        ProcessingResult(
                            success=False,
                            data=None,
                            execution_time=0,
                            worker_id=f"thread_{i+j}",
                            error=str(e),
                        )
                    )

        return results

    def shutdown(self):
        """Cleanup resources"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None


class OptimizedFactorComputer:
    """Optimized factor computation with parallel processing"""

    def __init__(self):
        self.executor = ParallelExecutor()

    async def compute_factors_parallel(
        self, symbols: List[str], factor_functions: Dict[str, Callable], market_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute factors in parallel for multiple symbols"""

        # Create tasks for each symbol-factor combination
        tasks = []
        symbol_factor_map = []

        for symbol in symbols:
            if symbol not in market_data:
                continue

            symbol_data = market_data[symbol]

            for factor_name, factor_func in factor_functions.items():
                # Create async wrapper for factor computation
                task = self._compute_factor_async(factor_func, symbol, symbol_data)
                tasks.append(task)
                symbol_factor_map.append((symbol, factor_name))

        # Execute all factor computations
        results = await self.executor.execute_async_batch(tasks)

        # Organize results by symbol
        factor_results = {symbol: {} for symbol in symbols}

        for i, result in enumerate(results):
            if i < len(symbol_factor_map):
                symbol, factor_name = symbol_factor_map[i]

                if result.success:
                    factor_results[symbol][factor_name] = result.data
                else:
                    logger.error(
                        f"Factor computation failed for {symbol}.{factor_name}: {result.error}"
                    )
                    factor_results[symbol][factor_name] = None

        return factor_results

    async def _compute_factor_async(self, factor_func: Callable, symbol: str, data: Any) -> Any:
        """Async wrapper for factor computation"""
        try:
            # Run factor computation in thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, factor_func, data)
            return result
        except Exception as e:
            logger.error(f"Factor computation error for {symbol}: {e}")
            raise


# Global executor instance
_global_executor: Optional[ParallelExecutor] = None


def get_parallel_executor() -> ParallelExecutor:
    """Get global parallel executor instance"""
    global _global_executor
    if _global_executor is None:
        _global_executor = ParallelExecutor()
    return _global_executor


def cleanup_parallel_executor():
    """Cleanup global executor"""
    global _global_executor
    if _global_executor:
        _global_executor.shutdown()
        _global_executor = None
