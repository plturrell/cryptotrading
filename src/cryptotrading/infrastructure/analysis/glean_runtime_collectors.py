"""
Runtime data collection decorators for Glean
Captures data inputs, outputs, parameters, and factors during execution
"""
import functools
import inspect
import logging
import asyncio
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from .glean_data_schemas import (
    DataInputFact, DataOutputFact, ParameterFact,
    FactorFact, DataLineageFact
)
# Conditional import for GleanStorage
try:
    from .glean_storage import GleanStorage
except ImportError:
    # Mock for when storage is not available
    class GleanStorage:
        def __init__(self, *args, **kwargs):
            pass
        async def store_facts(self, facts, unit):
            return {"stored": 0}

logger = logging.getLogger(__name__)

# Global storage instance
_glean_storage: Optional[GleanStorage] = None


def init_glean_collectors(storage_path: Optional[str] = None):
    """Initialize Glean collectors with storage backend"""
    global _glean_storage
    if not _glean_storage:
        _glean_storage = GleanStorage(storage_path)
        logger.info("Glean runtime collectors initialized")


def get_glean_storage() -> Optional[GleanStorage]:
    """Get the global Glean storage instance"""
    return _glean_storage


class GleanCollector:
    """Base class for Glean data collectors"""

    def __init__(self):
        self.storage = get_glean_storage()
        if not self.storage:
            logger.warning("Glean storage not initialized. Call init_glean_collectors() first.")

    async def store_fact(self, fact: Dict[str, Any], unit_name: str = "runtime") -> bool:
        """Store a fact to Glean storage"""
        if not self.storage:
            return False

        try:
            # Store fact in the appropriate unit
            result = await self.storage.store_facts([fact], unit_name)
            return result.get('stored', 0) > 0
        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            return False

    def _serialize_value(self, value: Any) -> str:
        """Serialize a value for storage"""
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return f"DataFrame(shape={value.shape})"
        elif isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        else:
            return str(value)

    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the calling function"""
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_frame = frame.f_back.f_back
            return {
                'file': caller_frame.f_code.co_filename,
                'function': caller_frame.f_code.co_name,
                'line': caller_frame.f_lineno
            }
        return {'file': 'unknown', 'function': 'unknown', 'line': 0}


def track_data_input(
    source: str,
    data_type: str,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None
):
    """
    Decorator to track data inputs to functions

    @track_data_input(source="yahoo_finance", data_type="market_data", symbol="BTC-USD")
    def fetch_bitcoin_data():
        ...
    """
    def decorator(func: Callable) -> Callable:
        collector = GleanCollector()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create input fact
            caller_info = collector._get_caller_info()
            input_fact = DataInputFact(
                function=func.__name__,
                file=func.__code__.co_filename,
                line=func.__code__.co_firstlineno,
                input_id=str(uuid.uuid4()),
                data_type=data_type,
                source=source,
                symbol=symbol or kwargs.get('symbol', ''),
                timeframe=timeframe or kwargs.get('timeframe', ''),
                parameters={**kwargs}
            )

            # Store fact
            await collector.store_fact(input_fact.to_fact())

            # Execute function
            result = await func(*args, **kwargs)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create input fact
            caller_info = collector._get_caller_info()
            input_fact = DataInputFact(
                function=func.__name__,
                file=func.__code__.co_filename,
                line=func.__code__.co_firstlineno,
                input_id=str(uuid.uuid4()),
                data_type=data_type,
                source=source,
                symbol=symbol or kwargs.get('symbol', ''),
                timeframe=timeframe or kwargs.get('timeframe', ''),
                parameters={**kwargs}
            )

            # Store fact synchronously
            if collector.storage:
                asyncio.create_task(collector.store_fact(input_fact.to_fact()))

            # Execute function
            result = func(*args, **kwargs)

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def track_data_output(output_type: str):
    """
    Decorator to track data outputs from functions

    @track_data_output(output_type="prediction")
    def predict_price():
        ...
    """
    def decorator(func: Callable) -> Callable:
        collector = GleanCollector()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute function
            result = await func(*args, **kwargs)

            # Create output fact
            output_fact = DataOutputFact(
                function=func.__name__,
                file=func.__code__.co_filename,
                line=func.__code__.co_firstlineno,
                output_id=str(uuid.uuid4()),
                output_type=output_type,
                data_shape=_analyze_data_shape(result),
                symbol=kwargs.get('symbol', ''),
                metadata={'args': str(args)[:100], 'kwargs': str(kwargs)[:100]}
            )

            # Store fact
            await collector.store_fact(output_fact.to_fact())

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)

            # Create output fact
            output_fact = DataOutputFact(
                function=func.__name__,
                file=func.__code__.co_filename,
                line=func.__code__.co_firstlineno,
                output_id=str(uuid.uuid4()),
                output_type=output_type,
                data_shape=_analyze_data_shape(result),
                symbol=kwargs.get('symbol', ''),
                metadata={'args': str(args)[:100], 'kwargs': str(kwargs)[:100]}
            )

            # Store fact synchronously
            if collector.storage:
                asyncio.create_task(collector.store_fact(output_fact.to_fact()))

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def track_parameters(**param_specs):
    """
    Decorator to track function parameters

    @track_parameters(
        window={"type": "int", "range_min": 1, "range_max": 100, "default": 14},
        threshold={"type": "float", "range_min": 0, "range_max": 1, "default": 0.5}
    )
    def calculate_signal(window=14, threshold=0.5):
        ...
    """
    def decorator(func: Callable) -> Callable:
        collector = GleanCollector()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Track parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name, param_value in bound_args.arguments.items():
                if param_name in param_specs:
                    spec = param_specs[param_name]
                    param_fact = ParameterFact(
                        name=param_name,
                        context=func.__name__,
                        file=func.__code__.co_filename,
                        param_type=spec.get('type', type(param_value).__name__),
                        value=param_value,
                        range_min=spec.get('range_min'),
                        range_max=spec.get('range_max'),
                        default=spec.get('default'),
                        description=spec.get('description', ''),
                        category=spec.get('category', 'runtime')
                    )
                    await collector.store_fact(param_fact.to_fact())

            # Execute function
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Track parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name, param_value in bound_args.arguments.items():
                if param_name in param_specs:
                    spec = param_specs[param_name]
                    param_fact = ParameterFact(
                        name=param_name,
                        context=func.__name__,
                        file=func.__code__.co_filename,
                        param_type=spec.get('type', type(param_value).__name__),
                        value=param_value,
                        range_min=spec.get('range_min'),
                        range_max=spec.get('range_max'),
                        default=spec.get('default'),
                        description=spec.get('description', ''),
                        category=spec.get('category', 'runtime')
                    )
                    if collector.storage:
                        asyncio.create_task(collector.store_fact(param_fact.to_fact()))

            # Execute function
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def track_factor(
    factor_name: str,
    category: str,
    formula: Optional[str] = None
):
    """
    Decorator to track factor calculations

    @track_factor(factor_name="rsi", category="momentum", formula="RSI(close, period)")
    def calculate_rsi(prices, period=14):
        ...
    """
    def decorator(func: Callable) -> Callable:
        collector = GleanCollector()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()

            # Execute function
            result = await func(*args, **kwargs)

            # Create factor fact
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            factor_fact = FactorFact(
                name=factor_name,
                symbol=kwargs.get('symbol', ''),
                timeframe=kwargs.get('timeframe', '1d'),
                category=category,
                calculation={
                    'function': func.__name__,
                    'execution_time_ms': execution_time,
                    'parameters': {k: collector._serialize_value(v) for k, v in kwargs.items()}
                },
                dependencies=_extract_dependencies(args, kwargs),
                formula=formula or func.__name__,
                result_type=type(result).__name__
            )

            # Store fact
            await collector.store_fact(factor_fact.to_fact())

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()

            # Execute function
            result = func(*args, **kwargs)

            # Create factor fact
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            factor_fact = FactorFact(
                name=factor_name,
                symbol=kwargs.get('symbol', ''),
                timeframe=kwargs.get('timeframe', '1d'),
                category=category,
                calculation={
                    'function': func.__name__,
                    'execution_time_ms': execution_time,
                    'parameters': {k: collector._serialize_value(v) for k, v in kwargs.items()}
                },
                dependencies=_extract_dependencies(args, kwargs),
                formula=formula or func.__name__,
                result_type=type(result).__name__
            )

            # Store fact synchronously
            if collector.storage:
                asyncio.create_task(collector.store_fact(factor_fact.to_fact()))

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def track_lineage(source_type: str, target_type: str, edge_type: str = "transforms"):
    """
    Decorator to track data lineage between functions

    @track_lineage(source_type="market_data", target_type="technical_indicator")
    def calculate_indicator(market_data):
        ...
    """
    def decorator(func: Callable) -> Callable:
        collector = GleanCollector()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate IDs for lineage tracking
            source_id = f"{source_type}_{uuid.uuid4().hex[:8]}"
            target_id = f"{target_type}_{uuid.uuid4().hex[:8]}"

            # Execute function
            result = await func(*args, **kwargs)

            # Create lineage fact
            lineage_fact = DataLineageFact(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                source_type=source_type,
                target_type=target_type,
                transformation=func.__name__,
                file=func.__code__.co_filename,
                line=func.__code__.co_firstlineno
            )

            # Store fact
            await collector.store_fact(lineage_fact.to_fact())

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate IDs for lineage tracking
            source_id = f"{source_type}_{uuid.uuid4().hex[:8]}"
            target_id = f"{target_type}_{uuid.uuid4().hex[:8]}"

            # Execute function
            result = func(*args, **kwargs)

            # Create lineage fact
            lineage_fact = DataLineageFact(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                source_type=source_type,
                target_type=target_type,
                transformation=func.__name__,
                file=func.__code__.co_filename,
                line=func.__code__.co_firstlineno
            )

            # Store fact synchronously
            if collector.storage:
                asyncio.create_task(collector.store_fact(lineage_fact.to_fact()))

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Helper functions
def _analyze_data_shape(data: Any) -> Dict[str, Any]:
    """Analyze the shape and structure of data"""
    if isinstance(data, pd.DataFrame):
        return {
            'type': 'DataFrame',
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
    elif isinstance(data, pd.Series):
        return {
            'type': 'Series',
            'shape': data.shape,
            'dtype': str(data.dtype)
        }
    elif isinstance(data, np.ndarray):
        return {
            'type': 'ndarray',
            'shape': data.shape,
            'dtype': str(data.dtype)
        }
    elif isinstance(data, list):
        return {
            'type': 'list',
            'length': len(data),
            'sample': str(data[:3]) if data else []
        }
    elif isinstance(data, dict):
        return {
            'type': 'dict',
            'keys': list(data.keys()),
            'size': len(data)
        }
    else:
        return {
            'type': type(data).__name__,
            'value': str(data)[:100]
        }


def _extract_dependencies(args: tuple, kwargs: dict) -> List[str]:
    """Extract dependencies from function arguments"""
    dependencies = []

    # Check for common data dependencies
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            dependencies.append("DataFrame")
        elif isinstance(arg, pd.Series):
            dependencies.append("Series")
        elif hasattr(arg, '__name__'):
            dependencies.append(arg.__name__)

    # Check kwargs for named dependencies
    for key, value in kwargs.items():
        if key in ['data', 'prices', 'market_data', 'indicators']:
            dependencies.append(f"{key}:{type(value).__name__}")

    return dependencies


# Example usage
if __name__ == "__main__":
    # Initialize collectors
    init_glean_collectors()

    # Example decorated function
    @track_data_input(source="yahoo_finance", data_type="market_data")
    @track_data_output(output_type="technical_indicator")
    @track_parameters(period={"type": "int", "range_min": 1, "range_max": 100, "default": 14})
    @track_factor(factor_name="rsi", category="momentum")
    async def calculate_rsi(symbol: str, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        # Simplified RSI calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Test the decorated function
    async def test():
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        result = await calculate_rsi(symbol="BTC-USD", prices=prices, period=14)
        print(f"RSI calculated: {result}")

    asyncio.run(test())