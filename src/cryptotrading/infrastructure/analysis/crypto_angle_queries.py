"""
Angle queries for crypto data tracking
Extends the basic Angle query system with crypto-specific queries
"""
from typing import Any, Dict, List, Optional

# Crypto-specific Angle query templates
CRYPTO_ANGLE_QUERIES = {
    # Data flow queries
    "data_inputs_by_symbol": """
        crypto.DataInput {symbol: {symbol}} -> {
            function: string,
            data_type: string,
            source: string,
            timestamp: string
        }
    """,
    "data_outputs_by_function": """
        crypto.DataOutput {function: {function}} -> {
            output_type: string,
            symbol: string,
            confidence: float,
            timestamp: string
        }
    """,
    "data_lineage_trace": """
        crypto.DataLineage {source_id: {source_id}} -> {
            target_id: string,
            edge_type: string,
            transformation: string,
            file: string
        }
    """,
    # Parameter queries
    "parameters_by_category": """
        crypto.Parameter {category: {category}} -> {
            name: string,
            param_type: string,
            value: string,
            context: string
        }
    """,
    "parameters_by_range": """
        crypto.Parameter {param_type: "float"} where value >= {min_value} && value <= {max_value} -> {
            name: string,
            value: string,
            context: string,
            category: string
        }
    """,
    # Factor queries
    "factors_by_symbol": """
        crypto.Factor {symbol: {symbol}} -> {
            name: string,
            category: string,
            timeframe: string,
            formula: string,
            dependencies: string
        }
    """,
    "factors_by_category": """
        crypto.Factor {category: {category}} -> {
            name: string,
            symbol: string,
            calculation: string,
            result_type: string
        }
    """,
    "factor_calculations_recent": """
        crypto.FactorCalculation {factor_name: {factor_name}}
        where timestamp >= {since_timestamp} -> {
            symbol: string,
            result: string,
            execution_time_ms: nat,
            cache_hit: bool,
            error: string
        }
    """,
    # Factor dependency analysis
    "factor_dependency_chain": """
        crypto.Factor {name: {factor_name}} -> {
            dependencies: string
        } |
        crypto.DataLineage {target_type: "factor"} where target_id contains {factor_name} -> {
            source_id: string,
            source_type: string,
            transformation: string
        }
    """,
    # Data quality queries
    "data_quality_by_score": """
        crypto.DataQuality where score >= {min_score} -> {
            data_id: string,
            metric: string,
            score: float,
            issues: string,
            timestamp: string
        }
    """,
    "data_quality_issues": """
        crypto.DataQuality where score < {threshold} -> {
            data_id: string,
            metric: string,
            score: float,
            issues: string,
            assessor: string
        }
    """,
    # Complex analysis queries
    "complete_data_flow": """
        crypto.DataInput {symbol: {symbol}} -> input {
            function: string,
            data_type: string,
            source: string
        } |
        crypto.DataOutput {symbol: {symbol}} -> output {
            function: string,
            output_type: string
        } |
        crypto.Factor {symbol: {symbol}} -> factor {
            name: string,
            category: string,
            formula: string
        }
    """,
    "parameter_impact_analysis": """
        crypto.Parameter {name: {param_name}} -> param {
            context: string,
            value: string
        } |
        crypto.FactorCalculation where input_values contains {param_name} -> calc {
            factor_name: string,
            result: string,
            execution_time_ms: nat
        }
    """,
    "performance_bottlenecks": """
        crypto.FactorCalculation where execution_time_ms > {threshold_ms} -> {
            factor_name: string,
            symbol: string,
            execution_time_ms: nat,
            input_values: string,
            timestamp: string
        }
    """,
    # Cross-factor analysis
    "factor_correlation_candidates": """
        crypto.Factor {category: {category}} -> factor1 {
            name: string,
            symbol: string
        } |
        crypto.Factor {category: {category}} -> factor2 {
            name: string,
            symbol: string
        }
        where factor1.symbol == factor2.symbol && factor1.name != factor2.name
    """,
    # Error analysis
    "calculation_errors": """
        crypto.FactorCalculation where error != null -> {
            factor_name: string,
            symbol: string,
            error: string,
            input_values: string,
            timestamp: string
        }
    """,
    # Data freshness analysis
    "stale_data_detection": """
        crypto.DataInput where timestamp < {stale_threshold} -> {
            function: string,
            symbol: string,
            source: string,
            timestamp: string,
            data_type: string
        }
    """,
}


def create_crypto_query(query_type: str, parameters: Dict[str, Any]) -> str:
    """
    Create a crypto-specific Angle query

    Args:
        query_type: Type of query from CRYPTO_ANGLE_QUERIES
        parameters: Parameters to substitute in the query template

    Returns:
        Formatted Angle query string
    """
    if query_type not in CRYPTO_ANGLE_QUERIES:
        raise ValueError(f"Unknown crypto query type: {query_type}")

    template = CRYPTO_ANGLE_QUERIES[query_type]

    # Simple template substitution
    try:
        query = template
        for key, value in parameters.items():
            placeholder = f"{{{key}}}"
            if isinstance(value, str):
                # Quote string values
                query = query.replace(placeholder, f'"{value}"')
            else:
                # Use numeric/boolean values directly
                query = query.replace(placeholder, str(value))

        return query
    except Exception as e:
        raise ValueError(f"Failed to format query template: {e}")


def build_data_lineage_query(
    symbol: str, include_factors: bool = True, include_parameters: bool = True, max_depth: int = 3
) -> str:
    """Build a comprehensive data lineage query for a symbol"""

    base_query = f"""
        // Get all data inputs for {symbol}
        crypto.DataInput {{symbol: "{symbol}"}} -> input {{
            input_id: string,
            function: string,
            data_type: string,
            source: string,
            timestamp: string
        }}

        // Get all data outputs for {symbol}
        crypto.DataOutput {{symbol: "{symbol}"}} -> output {{
            output_id: string,
            function: string,
            output_type: string,
            timestamp: string
        }}

        // Get lineage connections
        crypto.DataLineage -> lineage {{
            source_id: string,
            target_id: string,
            edge_type: string,
            transformation: string,
            source_type: string,
            target_type: string
        }}
    """

    if include_factors:
        base_query += f"""

        // Get factors for {symbol}
        crypto.Factor {{symbol: "{symbol}"}} -> factor {{
            name: string,
            category: string,
            dependencies: string,
            formula: string
        }}

        // Get factor calculations
        crypto.FactorCalculation {{symbol: "{symbol}"}} -> calc {{
            factor_name: string,
            result: string,
            execution_time_ms: nat,
            cache_hit: bool
        }}
        """

    if include_parameters:
        base_query += """

        // Get related parameters
        crypto.Parameter -> param {
            name: string,
            context: string,
            value: string,
            category: string
        }
        """

    return base_query


def build_factor_dependency_query(factor_name: str) -> str:
    """Build a query to trace factor dependencies"""

    return f"""
        // Get the factor definition
        crypto.Factor {{name: "{factor_name}"}} -> factor {{
            symbol: string,
            category: string,
            dependencies: string,
            calculation: string,
            formula: string
        }}

        // Get lineage for this factor
        crypto.DataLineage {{target_type: "factor"}}
        where target_id contains "{factor_name}" -> dependency {{
            source_id: string,
            source_type: string,
            transformation: string,
            file: string,
            line: nat
        }}

        // Get input data for functions that calculate this factor
        crypto.DataInput -> input {{
            function: string,
            data_type: string,
            source: string,
            symbol: string
        }}

        // Get parameters used in factor calculations
        crypto.Parameter -> param {{
            name: string,
            context: string,
            value: string,
            param_type: string
        }}
    """


def build_performance_analysis_query(
    min_execution_time: int = 1000,  # milliseconds
    error_threshold: float = 0.05,  # 5% error rate
    time_window: str = "24h",
) -> str:
    """Build a performance analysis query"""

    return f"""
        // Find slow factor calculations
        crypto.FactorCalculation where execution_time_ms >= {min_execution_time} -> slow {{
            factor_name: string,
            symbol: string,
            execution_time_ms: nat,
            timestamp: string,
            cache_hit: bool
        }}

        // Find calculations with errors
        crypto.FactorCalculation where error != null -> error {{
            factor_name: string,
            symbol: string,
            error: string,
            input_values: string,
            timestamp: string
        }}

        // Get data quality issues
        crypto.DataQuality where score < {1.0 - error_threshold} -> quality {{
            data_id: string,
            metric: string,
            score: float,
            issues: string,
            assessor: string
        }}

        // Get parameter variations that might affect performance
        crypto.Parameter where category == "performance" -> perf_param {{
            name: string,
            context: string,
            value: string,
            param_type: string
        }}
    """


def validate_crypto_query(query: str) -> Dict[str, Any]:
    """Validate a crypto Angle query"""

    validation_result = {"valid": True, "errors": [], "warnings": [], "predicates_used": []}

    # Check for required crypto predicates
    crypto_predicates = [
        "crypto.DataInput",
        "crypto.DataOutput",
        "crypto.Parameter",
        "crypto.Factor",
        "crypto.FactorCalculation",
        "crypto.DataLineage",
        "crypto.DataQuality",
    ]

    for predicate in crypto_predicates:
        if predicate in query:
            validation_result["predicates_used"].append(predicate)

    # Basic syntax checks
    if not validation_result["predicates_used"]:
        validation_result["valid"] = False
        validation_result["errors"].append("No crypto predicates found in query")

    # Check for balanced braces
    open_braces = query.count("{")
    close_braces = query.count("}")
    if open_braces != close_braces:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"Unbalanced braces: {open_braces} open, {close_braces} close"
        )

    # Check for required arrows
    if " -> " not in query:
        validation_result["warnings"].append("No result projections found (missing '->')")

    return validation_result


# Export all query functions
__all__ = [
    "CRYPTO_ANGLE_QUERIES",
    "create_crypto_query",
    "build_data_lineage_query",
    "build_factor_dependency_query",
    "build_performance_analysis_query",
    "validate_crypto_query",
]
