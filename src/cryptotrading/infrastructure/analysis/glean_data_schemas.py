"""
Glean predicate schemas for data tracking in cryptotrading
Extends the core Glean schemas to track data inputs, outputs, parameters, and factors
"""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

# Data tracking predicate schemas
DATA_TRACKING_SCHEMAS = {
    # Track data inputs to functions/calculations
    "crypto.DataInput": {
        "key": {
            "function": "string",  # Function consuming the data
            "file": "string",  # File location
            "line": "nat",  # Line number
            "input_id": "string",  # Unique identifier for this input
        },
        "value": {
            "data_type": "string",  # Type of input (market_data, price, volume, etc.)
            "source": "string",  # Data source (yahoo_finance, binance, etc.)
            "symbol": "string",  # Crypto symbol if applicable
            "timeframe": "string",  # Data timeframe (1m, 1h, 1d, etc.)
            "parameters": "string",  # JSON-encoded parameters
            "timestamp": "string",  # When data was accessed
        },
    },
    # Track data outputs from functions/calculations
    "crypto.DataOutput": {
        "key": {
            "function": "string",  # Function producing the output
            "file": "string",  # File location
            "line": "nat",  # Line number
            "output_id": "string",  # Unique identifier for this output
        },
        "value": {
            "output_type": "string",  # prediction, signal, metric, indicator, etc.
            "data_shape": "string",  # JSON-encoded shape/structure of output
            "symbol": "string",  # Crypto symbol if applicable
            "confidence": "maybe float",  # Confidence score if applicable
            "metadata": "string",  # JSON-encoded additional metadata
            "timestamp": "string",  # When output was produced
        },
    },
    # Track configuration and runtime parameters
    "crypto.Parameter": {
        "key": {
            "name": "string",  # Parameter name
            "context": "string",  # Where it's used (function/class/module)
            "file": "string",  # File location
        },
        "value": {
            "param_type": "string",  # int, float, string, bool, etc.
            "value": "string",  # String representation of value
            "range_min": "maybe string",  # Minimum valid value if applicable
            "range_max": "maybe string",  # Maximum valid value if applicable
            "default": "string",  # Default value
            "description": "string",  # Parameter description
            "category": "string",  # model, strategy, risk, etc.
        },
    },
    # Track crypto factors (CRD - Crypto Related Data)
    "crypto.Factor": {
        "key": {
            "name": "string",  # Factor name (e.g., rsi_14, macd_signal)
            "symbol": "string",  # Crypto symbol
            "timeframe": "string",  # Calculation timeframe
        },
        "value": {
            "category": "string",  # price, volume, technical, volatility, sentiment
            "calculation": "string",  # JSON-encoded calculation details
            "dependencies": "string",  # JSON-encoded list of required inputs
            "formula": "string",  # Human-readable formula/description
            "result_type": "string",  # float, bool, signal, etc.
            "timestamp": "string",  # When factor was calculated
        },
    },
    # Track factor calculations and results
    "crypto.FactorCalculation": {
        "key": {
            "factor_name": "string",  # Factor being calculated
            "symbol": "string",  # Crypto symbol
            "timestamp": "string",  # Calculation timestamp
            "calc_id": "string",  # Unique calculation ID
        },
        "value": {
            "result": "string",  # String representation of result
            "execution_time_ms": "nat",  # Calculation time in milliseconds
            "input_values": "string",  # JSON-encoded input values used
            "cache_hit": "bool",  # Whether result was cached
            "error": "maybe string",  # Error message if calculation failed
        },
    },
    # Track data lineage - how data flows through the system
    "crypto.DataLineage": {
        "key": {
            "source_id": "string",  # ID of source data/calculation
            "target_id": "string",  # ID of target data/calculation
            "edge_type": "string",  # input, output, derives_from, etc.
        },
        "value": {
            "source_type": "string",  # Type of source (input, factor, parameter)
            "target_type": "string",  # Type of target
            "transformation": "string",  # Description of transformation applied
            "file": "string",  # Where transformation occurs
            "line": "nat",  # Line number
        },
    },
    # Track data quality metrics
    "crypto.DataQuality": {
        "key": {
            "data_id": "string",  # ID of data being assessed
            "metric": "string",  # Quality metric name
        },
        "value": {
            "score": "float",  # Quality score (0-1)
            "issues": "string",  # JSON-encoded list of issues
            "timestamp": "string",  # When quality was assessed
            "assessor": "string",  # Function/module that assessed quality
        },
    },
}


# Helper classes for type-safe fact creation
@dataclass
class DataInputFact:
    """Helper for creating data input facts"""

    function: str
    file: str
    line: int
    input_id: str
    data_type: str
    source: str
    symbol: str = ""
    timeframe: str = ""
    parameters: Dict[str, Any] = None
    timestamp: str = None

    def to_fact(self) -> Dict[str, Any]:
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        return {
            "predicate": "crypto.DataInput",
            "key": {
                "function": self.function,
                "file": self.file,
                "line": self.line,
                "input_id": self.input_id,
            },
            "value": {
                "data_type": self.data_type,
                "source": self.source,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "parameters": json.dumps(self.parameters or {}),
                "timestamp": self.timestamp,
            },
        }


@dataclass
class DataOutputFact:
    """Helper for creating data output facts"""

    function: str
    file: str
    line: int
    output_id: str
    output_type: str
    data_shape: Dict[str, Any]
    symbol: str = ""
    confidence: float = None
    metadata: Dict[str, Any] = None
    timestamp: str = None

    def to_fact(self) -> Dict[str, Any]:
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        return {
            "predicate": "crypto.DataOutput",
            "key": {
                "function": self.function,
                "file": self.file,
                "line": self.line,
                "output_id": self.output_id,
            },
            "value": {
                "output_type": self.output_type,
                "data_shape": json.dumps(self.data_shape),
                "symbol": self.symbol,
                "confidence": self.confidence,
                "metadata": json.dumps(self.metadata or {}),
                "timestamp": self.timestamp,
            },
        }


@dataclass
class ParameterFact:
    """Helper for creating parameter facts"""

    name: str
    context: str
    file: str
    param_type: str
    value: Any
    range_min: Any = None
    range_max: Any = None
    default: Any = None
    description: str = ""
    category: str = ""

    def to_fact(self) -> Dict[str, Any]:
        return {
            "predicate": "crypto.Parameter",
            "key": {"name": self.name, "context": self.context, "file": self.file},
            "value": {
                "param_type": self.param_type,
                "value": str(self.value),
                "range_min": str(self.range_min) if self.range_min is not None else None,
                "range_max": str(self.range_max) if self.range_max is not None else None,
                "default": str(self.default) if self.default is not None else "",
                "description": self.description,
                "category": self.category,
            },
        }


@dataclass
class FactorFact:
    """Helper for creating factor facts"""

    name: str
    symbol: str
    timeframe: str
    category: str
    calculation: Dict[str, Any]
    dependencies: List[str]
    formula: str
    result_type: str
    timestamp: str = None

    def to_fact(self) -> Dict[str, Any]:
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        return {
            "predicate": "crypto.Factor",
            "key": {"name": self.name, "symbol": self.symbol, "timeframe": self.timeframe},
            "value": {
                "category": self.category,
                "calculation": json.dumps(self.calculation),
                "dependencies": json.dumps(self.dependencies),
                "formula": self.formula,
                "result_type": self.result_type,
                "timestamp": self.timestamp,
            },
        }


@dataclass
class DataLineageFact:
    """Helper for creating data lineage facts"""

    source_id: str
    target_id: str
    edge_type: str
    source_type: str
    target_type: str
    transformation: str
    file: str
    line: int

    def to_fact(self) -> Dict[str, Any]:
        return {
            "predicate": "crypto.DataLineage",
            "key": {
                "source_id": self.source_id,
                "target_id": self.target_id,
                "edge_type": self.edge_type,
            },
            "value": {
                "source_type": self.source_type,
                "target_type": self.target_type,
                "transformation": self.transformation,
                "file": self.file,
                "line": self.line,
            },
        }


# Export all schemas for registration
ALL_DATA_TRACKING_SCHEMAS = DATA_TRACKING_SCHEMAS
