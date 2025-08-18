"""
Extended SCIP indexer that captures data flow information
Tracks data inputs, outputs, parameters, and factor calculations
"""
import ast
import logging
from typing import Dict, List, Any, Set, Optional, Tuple
from pathlib import Path
import uuid
from datetime import datetime

from .scip_indexer import PythonSCIPIndexer, SCIPSymbol, SCIPDocument
from .glean_data_schemas import (
    DataInputFact, DataOutputFact, ParameterFact,
    FactorFact, DataLineageFact
)

logger = logging.getLogger(__name__)


class DataFlowVisitor(ast.NodeVisitor):
    """AST visitor that tracks data flow through code"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data_inputs = []
        self.data_outputs = []
        self.parameters = []
        self.factors = []
        self.lineage = []
        self.current_function = None
        self.current_class = None
        self.import_map = {}  # Track imports for resolving function calls

    def visit_Import(self, node: ast.Import) -> None:
        """Track imports to resolve data source calls"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.import_map[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports"""
        module = node.module or ""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.import_map[name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function definitions and their parameters"""
        old_function = self.current_function
        self.current_function = node.name

        # Extract function parameters as configuration parameters
        for arg in node.args.args:
            if arg.arg not in ['self', 'cls']:
                # Try to infer parameter type from defaults or annotations
                param_type = "any"
                default_value = None

                if arg.annotation:
                    param_type = self._get_annotation_string(arg.annotation)

                # Check for default values
                defaults_offset = len(node.args.args) - len(node.args.defaults)
                if node.args.args.index(arg) >= defaults_offset:
                    default_idx = node.args.args.index(arg) - defaults_offset
                    default_value = self._get_literal_value(node.args.defaults[default_idx])

                param_fact = ParameterFact(
                    name=arg.arg,
                    context=f"{self.current_class}.{node.name}" if self.current_class else node.name,
                    file=self.file_path,
                    param_type=param_type,
                    value=default_value or "undefined",
                    default=default_value,
                    category="function_parameter"
                )
                self.parameters.append(param_fact)

        # Visit function body
        self.generic_visit(node)
        self.current_function = old_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class definitions"""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Call(self, node: ast.Call) -> None:
        """Detect data source calls and factor calculations"""
        if not self.current_function:
            self.generic_visit(node)
            return

        # Check for data source calls
        data_source = self._identify_data_source_call(node)
        if data_source:
            self._track_data_input(node, data_source)

        # Check for factor calculations
        if self._is_factor_calculation(node):
            self._track_factor_calculation(node)

        # Check for data outputs (predictions, signals, etc.)
        if self._is_data_output(node):
            self._track_data_output(node)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variable assignments that might be parameters or configurations"""
        # Check for configuration-like assignments
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # Common parameter patterns
                if any(pattern in var_name.lower() for pattern in
                       ['threshold', 'window', 'period', 'factor', 'weight',
                        'alpha', 'beta', 'gamma', 'epsilon', 'rate', 'size']):

                    value = self._get_literal_value(node.value)
                    if value is not None:
                        context_name = (
                            f"{self.current_class}.{self.current_function}"
                            if self.current_class
                            else self.current_function or "module"
                        )
                        param_fact = ParameterFact(
                            name=var_name,
                            context=context_name,
                            file=self.file_path,
                            param_type=type(value).__name__,
                            value=value,
                            category="configuration"
                        )
                        self.parameters.append(param_fact)

        self.generic_visit(node)

    def _identify_data_source_call(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """Identify if this is a call to a data source"""
        func_name = self._get_call_name(node)

        # Common data source patterns
        data_sources = {
            'yfinance': ['download', 'Ticker'],
            'yahoo_finance': ['get_data', 'fetch_prices'],
            'binance': ['get_historical_klines', 'get_ticker'],
            'ccxt': ['fetch_ohlcv', 'fetch_ticker'],
            'fred': ['get_series', 'get_data'],
            'pandas_datareader': ['DataReader', 'get_data_yahoo'],
            'requests': ['get', 'post'],  # API calls
        }

        for source, methods in data_sources.items():
            if any(method in func_name for method in methods):
                return {
                    'source': source,
                    'method': func_name,
                    'line': node.lineno
                }

        # Check for DataFrame read operations
        if any(pattern in func_name for pattern in ['read_csv', 'read_json', 'read_sql', 'from_dict']):
            return {
                'source': 'file_system',
                'method': func_name,
                'line': node.lineno
            }

        return None

    def _is_factor_calculation(self, node: ast.Call) -> bool:
        """Check if this call is calculating a technical factor"""
        func_name = self._get_call_name(node).lower()

        # Common technical indicator patterns
        indicators = [
            'rsi', 'macd', 'bollinger', 'ema', 'sma', 'adx', 'atr',
            'stochastic', 'williams', 'obv', 'vwap', 'fibonacci',
            'ichimoku', 'parabolic', 'cci', 'mfi', 'roc', 'momentum'
        ]

        return any(indicator in func_name for indicator in indicators)

    def _is_data_output(self, node: ast.Call) -> bool:
        """Check if this call produces a data output"""
        func_name = self._get_call_name(node).lower()

        # Common output patterns
        outputs = [
            'predict', 'forecast', 'signal', 'score', 'classify',
            'recommend', 'alert', 'trigger', 'calculate_return',
            'generate_signal', 'evaluate', 'backtest'
        ]

        return any(output in func_name for output in outputs)

    def _track_data_input(self, node: ast.Call, data_source: Dict[str, Any]) -> None:
        """Track a data input"""
        # Extract parameters from the call
        params = self._extract_call_parameters(node)

        # Try to identify symbol from parameters
        symbol = params.get('symbol', params.get('ticker', ''))
        if not symbol and node.args:
            # First argument might be symbol
            symbol = self._get_literal_value(node.args[0]) or ''

        input_fact = DataInputFact(
            function=self.current_function or "module",
            file=self.file_path,
            line=node.lineno,
            input_id=str(uuid.uuid4()),
            data_type=self._infer_data_type(data_source['method']),
            source=data_source['source'],
            symbol=str(symbol),
            parameters=params
        )
        self.data_inputs.append(input_fact)

    def _track_factor_calculation(self, node: ast.Call) -> None:
        """Track a factor calculation"""
        func_name = self._get_call_name(node)
        params = self._extract_call_parameters(node)

        # Extract common factor parameters
        period = params.get('period', params.get('window', params.get('length', 14)))

        factor_fact = FactorFact(
            name=func_name.lower(),
            symbol=params.get('symbol', ''),
            timeframe=params.get('timeframe', '1d'),
            category=self._categorize_factor(func_name),
            calculation={'function': func_name, 'parameters': params},
            dependencies=self._extract_dependencies(node),
            formula=f"{func_name}({period})" if period else func_name,
            result_type="float"
        )
        self.factors.append(factor_fact)

    def _track_data_output(self, node: ast.Call) -> None:
        """Track a data output"""
        func_name = self._get_call_name(node)
        params = self._extract_call_parameters(node)

        output_fact = DataOutputFact(
            function=self.current_function or "module",
            file=self.file_path,
            line=node.lineno,
            output_id=str(uuid.uuid4()),
            output_type=self._infer_output_type(func_name),
            data_shape={'dimensions': 'unknown'},  # Would need more analysis
            symbol=params.get('symbol', ''),
            metadata={'function_called': func_name, 'parameters': params}
        )
        self.data_outputs.append(output_fact)

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the full name of a function call"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return "unknown"

    def _extract_call_parameters(self, node: ast.Call) -> Dict[str, Any]:
        """Extract parameters from a function call"""
        params = {}

        # Extract keyword arguments
        for keyword in node.keywords:
            if keyword.arg:
                value = self._get_literal_value(keyword.value)
                if value is not None:
                    params[keyword.arg] = value

        return params

    def _get_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return [self._get_literal_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._get_literal_value(k): self._get_literal_value(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        return None

    def _get_annotation_string(self, node: ast.AST) -> str:
        """Convert type annotation to string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_annotation_string(node.value)}[...]"
        return "any"

    def _infer_data_type(self, method: str) -> str:
        """Infer data type from method name"""
        method_lower = method.lower()
        if any(x in method_lower for x in ['price', 'ohlcv', 'ticker']):
            return "market_data"
        elif any(x in method_lower for x in ['volume']):
            return "volume"
        elif any(x in method_lower for x in ['series', 'indicator']):
            return "economic_indicator"
        return "generic"

    def _categorize_factor(self, func_name: str) -> str:
        """Categorize a factor based on its name"""
        func_lower = func_name.lower()
        if any(x in func_lower for x in ['rsi', 'stoch', 'williams', 'cci']):
            return "momentum"
        elif any(x in func_lower for x in ['macd', 'ema', 'sma']):
            return "trend"
        elif any(x in func_lower for x in ['bollinger', 'atr', 'std']):
            return "volatility"
        elif any(x in func_lower for x in ['obv', 'vwap', 'mfi']):
            return "volume"
        return "technical"

    def _infer_output_type(self, func_name: str) -> str:
        """Infer output type from function name"""
        func_lower = func_name.lower()
        if 'predict' in func_lower:
            return "prediction"
        elif 'signal' in func_lower:
            return "signal"
        elif 'score' in func_lower:
            return "score"
        elif 'forecast' in func_lower:
            return "forecast"
        return "calculation"

    def _extract_dependencies(self, node: ast.Call) -> List[str]:
        """Extract dependencies from a function call"""
        deps = []
        # This would need more sophisticated analysis
        # For now, return empty list
        return deps


class DataFlowSCIPIndexer(PythonSCIPIndexer):
    """Extended SCIP indexer that captures data flow"""

    def __init__(self, project_root: str):
        super().__init__(project_root)
        self.data_flow_facts = []

    def index_file(self, file_path: str) -> Dict[str, Any]:
        """Index a file and extract data flow information"""
        # First do normal SCIP indexing if parent has the method
        result = {"status": "success", "file": file_path}

        try:
            if hasattr(super(), 'index_file'):
                parent_result = super().index_file(file_path)
                if parent_result and parent_result.get('status') == 'success':
                    result = parent_result
                elif parent_result and parent_result.get('status') != 'success':
                    return parent_result
        except Exception as e:
            # Continue with data flow indexing even if base indexing fails
            logger.warning(f"Base indexing failed: {e}")

        # Ensure result is a dictionary
        if not isinstance(result, dict):
            result = {"status": "success", "file": file_path}

        try:
            # Now extract data flow
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=file_path)
            visitor = DataFlowVisitor(file_path)
            visitor.visit(tree)

            # Convert to facts
            facts = []

            # Add data input facts
            for input_fact in visitor.data_inputs:
                facts.append(input_fact.to_fact())

            # Add data output facts
            for output_fact in visitor.data_outputs:
                facts.append(output_fact.to_fact())

            # Add parameter facts
            for param_fact in visitor.parameters:
                facts.append(param_fact.to_fact())

            # Add factor facts
            for factor_fact in visitor.factors:
                facts.append(factor_fact.to_fact())

            # Store data flow facts
            self.data_flow_facts.extend(facts)

            # Add to result
            result['data_flow'] = {
                'inputs': len(visitor.data_inputs),
                'outputs': len(visitor.data_outputs),
                'parameters': len(visitor.parameters),
                'factors': len(visitor.factors)
            }

            logger.info(f"Extracted data flow from {file_path}: {result['data_flow']}")

        except Exception as e:
            logger.error(f"Failed to extract data flow from {file_path}: {e}")
            result['data_flow_error'] = str(e)

        return result

    def get_data_flow_facts(self) -> List[Dict[str, Any]]:
        """Get all extracted data flow facts"""
        return self.data_flow_facts

    def clear_data_flow_facts(self) -> None:
        """Clear stored data flow facts"""
        self.data_flow_facts = []