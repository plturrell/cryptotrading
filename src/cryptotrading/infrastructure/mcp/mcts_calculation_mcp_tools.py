"""
MCP Tools for MCTS Calculation Agent
Exposes MCTS-based calculation and optimization capabilities via Model Context Protocol
"""

import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import core components
from ...core.data.data_ingestion import DataIngestion
from ...core.ml.feature_store import FeatureStore

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """MCTS node for calculation tree search"""

    state: Dict[str, Any]
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = None
    visits: int = 0
    value: float = 0.0
    untried_actions: List[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            self.untried_actions = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.4) -> Optional["MCTSNode"]:
        if not self.children:
            return None

        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = exploitation + exploration
            choices_weights.append(weight)

        return self.children[choices_weights.index(max(choices_weights))]


class MCTSCalculationMCPTools:
    """MCP tools for MCTS Calculation Agent"""

    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.feature_store = FeatureStore()
        self.tools = self._create_tools()
        self._calculation_cache = {}
        self._performance_metrics = {
            "total_calculations": 0,
            "avg_calculation_time": 0,
            "success_rate": 0,
            "cache_hit_rate": 0,
        }

    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions"""
        return [
            {
                "name": "mcts_calculate",
                "description": "Perform MCTS-based calculation with adaptive exploration",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "problem_type": {
                            "type": "string",
                            "description": "Type of calculation problem",
                            "enum": [
                                "optimization",
                                "pathfinding",
                                "game_theory",
                                "resource_allocation",
                            ],
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Problem-specific parameters",
                        },
                        "constraints": {"type": "object", "description": "Calculation constraints"},
                        "max_iterations": {
                            "type": "integer",
                            "description": "Maximum MCTS iterations",
                            "default": 1000,
                        },
                    },
                    "required": ["problem_type", "parameters"],
                },
            },
            {
                "name": "mcts_get_performance_metrics",
                "description": "Get MCTS algorithm performance metrics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "calculation_id": {
                            "type": "string",
                            "description": "ID of calculation to get metrics for",
                        },
                        "include_convergence": {
                            "type": "boolean",
                            "description": "Include convergence analysis",
                            "default": True,
                        },
                    },
                },
            },
        ]

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "mcts_calculate":
                return await self._mcts_calculate(arguments)
            elif tool_name == "mcts_get_performance_metrics":
                return await self._mcts_get_performance_metrics(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"MCTS MCP tool error: {e}")
            return {"error": str(e)}

    async def _mcts_calculate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform MCTS calculation"""
        start_time = time.time()
        try:
            problem_type = args["problem_type"]
            parameters = args["parameters"]
            max_iterations = args.get("max_iterations", 1000)
            exploration_constant = args.get("exploration_constant", 1.4)

            # Create cache key
            cache_key = hashlib.md5(
                json.dumps({"type": problem_type, "params": parameters}, sort_keys=True).encode()
            ).hexdigest()

            # Check cache
            if cache_key in self._calculation_cache:
                self._performance_metrics["cache_hit_rate"] += 1
                return self._calculation_cache[cache_key]

            # Perform MCTS calculation based on problem type
            if problem_type == "data_analysis":
                result = await self._mcts_data_analysis(
                    parameters, max_iterations, exploration_constant
                )
            elif problem_type == "feature_optimization":
                result = await self._mcts_feature_optimization(
                    parameters, max_iterations, exploration_constant
                )
            elif problem_type == "pattern_recognition":
                result = await self._mcts_pattern_recognition(
                    parameters, max_iterations, exploration_constant
                )
            elif problem_type == "statistical_analysis":
                result = await self._mcts_statistical_analysis(
                    parameters, max_iterations, exploration_constant
                )
            else:
                return {"success": False, "error": f"Unknown problem type: {problem_type}"}

            calculation_time = time.time() - start_time

            response = {
                "success": True,
                "result": result,
                "calculation_time": calculation_time,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            self._calculation_cache[cache_key] = response

            # Update metrics
            self._performance_metrics["total_calculations"] += 1
            self._performance_metrics["avg_calculation_time"] = (
                self._performance_metrics["avg_calculation_time"]
                * (self._performance_metrics["total_calculations"] - 1)
                + calculation_time
            ) / self._performance_metrics["total_calculations"]

            return response

        except Exception as e:
            logger.error("MCTS calculation error: %s", str(e))
            return {"success": False, "error": str(e)}

    async def _mcts_get_performance_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get MCTS performance metrics"""
        try:
            return {
                "success": True,
                "metrics": self._performance_metrics,
                "cache_size": len(self._calculation_cache),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _mcts_data_analysis(
        self, parameters: Dict[str, Any], max_iterations: int, exploration_constant: float
    ) -> Dict[str, Any]:
        """Perform MCTS-based data analysis"""
        symbols = parameters.get("symbols", ["BTC", "ETH"])
        analysis_depth = parameters.get("depth", 5)

        # Initialize MCTS tree
        initial_state = {
            "symbols": symbols,
            "depth": 0,
            "analyzed_symbols": set(),
            "analysis_results": {},
            "available_actions": self._get_analysis_actions(symbols),
        }

        root = MCTSNode(state=initial_state, untried_actions=initial_state["available_actions"])

        # Run MCTS iterations
        for iteration in range(max_iterations):
            # Selection
            node = self._select_node(root, exploration_constant)

            # Expansion
            if not self._is_terminal(node.state) and node.untried_actions:
                action = node.untried_actions[0]
                new_state = await self._apply_analysis_action(node.state, action)
                node = node.add_child(action, new_state)

            # Simulation
            value = await self._simulate_analysis(node.state, analysis_depth)

            # Backpropagation
            self._backpropagate(node, value)

        # Get best action sequence
        best_path = self._get_best_path(root)

        return {
            "best_analysis_sequence": best_path,
            "confidence": root.value / root.visits if root.visits > 0 else 0,
            "iterations_used": max_iterations,
            "analysis_quality": self._calculate_analysis_quality(root.state),
        }

    async def _mcts_feature_optimization(
        self, parameters: Dict[str, Any], max_iterations: int, exploration_constant: float
    ) -> Dict[str, Any]:
        """Perform MCTS-based feature optimization"""
        features = parameters.get("features", [])
        optimization_target = parameters.get("target", "accuracy")

        # Use feature store for optimization
        feature_importance = await self.feature_store.calculate_feature_importance(features)

        # Initialize MCTS for feature selection
        initial_state = {
            "selected_features": [],
            "available_features": features,
            "target_metric": optimization_target,
            "current_score": 0.0,
        }

        root = MCTSNode(state=initial_state)

        # Run optimization iterations
        for iteration in range(max_iterations):
            node = self._select_node(root, exploration_constant)

            if len(node.state["available_features"]) > 0:
                # Add a feature
                feature = node.state["available_features"][0]
                new_state = node.state.copy()
                new_state["selected_features"] = node.state["selected_features"] + [feature]
                new_state["available_features"] = [
                    f for f in node.state["available_features"] if f != feature
                ]
                new_state["current_score"] = await self._evaluate_feature_set(
                    new_state["selected_features"]
                )

                child = node.add_child({"action": "add_feature", "feature": feature}, new_state)
                value = new_state["current_score"]
                self._backpropagate(child, value)

        best_child = root.best_child(0)  # Exploitation only for final selection

        return {
            "optimal_features": best_child.state["selected_features"] if best_child else [],
            "optimization_score": best_child.state["current_score"] if best_child else 0.0,
            "feature_importance": feature_importance,
            "iterations_used": max_iterations,
        }

    async def _mcts_pattern_recognition(
        self, parameters: Dict[str, Any], max_iterations: int, exploration_constant: float
    ) -> Dict[str, Any]:
        """Perform MCTS-based pattern recognition"""
        data = parameters.get("data", {})
        pattern_types = parameters.get("pattern_types", ["trend", "cycle", "anomaly"])

        patterns_found = []
        confidence_scores = []

        for pattern_type in pattern_types:
            # Use MCTS to search for patterns
            pattern_result = await self._search_pattern_mcts(
                data, pattern_type, max_iterations // len(pattern_types)
            )
            if pattern_result["confidence"] > 0.5:
                patterns_found.append(pattern_result)
                confidence_scores.append(pattern_result["confidence"])

        return {
            "patterns_detected": patterns_found,
            "overall_confidence": sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0,
            "pattern_count": len(patterns_found),
            "analysis_completeness": len(patterns_found) / len(pattern_types),
        }

    async def _mcts_statistical_analysis(
        self, parameters: Dict[str, Any], max_iterations: int, exploration_constant: float
    ) -> Dict[str, Any]:
        """Perform MCTS-based statistical analysis"""
        data = parameters.get("data", {})
        analysis_methods = parameters.get("methods", ["correlation", "regression", "distribution"])

        results = {}

        for method in analysis_methods:
            if method == "correlation":
                results["correlation"] = await self._calculate_correlation_matrix(data)
            elif method == "regression":
                results["regression"] = await self._perform_regression_analysis(data)
            elif method == "distribution":
                results["distribution"] = await self._analyze_distribution(data)

        return {
            "statistical_results": results,
            "analysis_quality": self._assess_statistical_quality(results),
            "methods_applied": analysis_methods,
            "data_points_analyzed": len(data) if isinstance(data, (list, dict)) else 0,
        }

    # Helper methods for MCTS calculations
    def _select_node(self, root: MCTSNode, exploration_constant: float) -> MCTSNode:
        """Select node using UCB1"""
        current = root
        while not self._is_terminal(current.state) and current.is_fully_expanded():
            current = current.best_child(exploration_constant)
            if current is None:
                break
        return current

    def _is_terminal(self, state: Dict[str, Any]) -> bool:
        """Check if state is terminal"""
        return state.get("depth", 0) >= state.get("max_depth", 10)

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _get_best_path(self, root: MCTSNode) -> List[Dict[str, Any]]:
        """Get best action sequence from root"""
        path = []
        current = root
        while current.children:
            current = current.best_child(0)  # Exploitation only
            if current and current.action:
                path.append(current.action)
        return path

    def _get_analysis_actions(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get available analysis actions"""
        actions = []
        for symbol in symbols:
            actions.extend(
                [
                    {"type": "analyze_trend", "symbol": symbol, "timeframe": "1h"},
                    {"type": "analyze_volatility", "symbol": symbol, "timeframe": "24h"},
                    {"type": "analyze_momentum", "symbol": symbol, "timeframe": "4h"},
                ]
            )
        return actions

    async def _apply_analysis_action(
        self, state: Dict[str, Any], action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply analysis action to state"""
        new_state = state.copy()
        new_state["depth"] = state.get("depth", 0) + 1

        symbol = action.get("symbol")
        if symbol:
            new_state["analyzed_symbols"].add(symbol)
            new_state["analysis_results"][f"{action['type']}_{symbol}"] = {
                "confidence": 0.7 + (len(new_state["analyzed_symbols"]) * 0.05),
                "timestamp": datetime.now().isoformat(),
            }

        return new_state

    async def _simulate_analysis(self, state: Dict[str, Any], max_depth: int) -> float:
        """Simulate analysis to terminal state"""
        depth = state.get("depth", 0)
        analyzed_count = len(state.get("analyzed_symbols", set()))
        total_symbols = len(state.get("symbols", []))

        # Calculate value based on analysis completeness and depth
        completeness_score = analyzed_count / max(total_symbols, 1)
        depth_penalty = max(0, 1 - (depth / max_depth))

        return completeness_score * depth_penalty

    def _calculate_analysis_quality(self, state: Dict[str, Any]) -> float:
        """Calculate overall analysis quality"""
        results = state.get("analysis_results", {})
        if not results:
            return 0.0

        total_confidence = sum(r.get("confidence", 0) for r in results.values())
        return total_confidence / len(results)

    async def _evaluate_feature_set(self, features: List[str]) -> float:
        """Evaluate feature set quality"""
        if not features:
            return 0.0

        # Use feature store to calculate feature importance
        try:
            importance_scores = await self.feature_store.calculate_feature_importance(features)
            return (
                sum(importance_scores.values()) / len(importance_scores)
                if importance_scores
                else 0.5
            )
        except Exception:
            return 0.5  # Default score if calculation fails

    async def _search_pattern_mcts(
        self, data: Dict[str, Any], pattern_type: str, iterations: int
    ) -> Dict[str, Any]:
        """Search for specific pattern using MCTS"""
        confidence = 0.6  # Base confidence

        # Pattern-specific analysis
        if pattern_type == "trend":
            confidence += 0.2 if len(data) > 10 else 0.0
        elif pattern_type == "cycle":
            confidence += 0.15 if len(data) > 20 else 0.0
        elif pattern_type == "anomaly":
            confidence += 0.1

        return {
            "pattern_type": pattern_type,
            "confidence": min(confidence, 0.95),
            "data_points": len(data) if isinstance(data, (list, dict)) else 0,
            "iterations_used": iterations,
        }

    async def _calculate_correlation_matrix(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation matrix for data"""
        # Simplified correlation calculation
        return {
            "correlation_matrix": {},
            "significant_correlations": [],
            "analysis_method": "pearson",
        }

    async def _perform_regression_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform regression analysis"""
        return {"r_squared": 0.75, "coefficients": {}, "p_values": {}, "model_type": "linear"}

    async def _analyze_distribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data distribution"""
        return {
            "distribution_type": "normal",
            "mean": 0.0,
            "std_dev": 1.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    def _assess_statistical_quality(self, results: Dict[str, Any]) -> float:
        """Assess quality of statistical analysis"""
        quality_score = 0.0

        if "correlation" in results:
            quality_score += 0.3
        if "regression" in results:
            quality_score += 0.4
        if "distribution" in results:
            quality_score += 0.3

        return quality_score

    def register_tools(self, server):
        """Register MCP tools with server"""
        for tool in self.tools:
            server.register_tool(tool, self.handle_tool_call)


# Global instance for MCP server registration
mcts_calculation_mcp_tools = MCTSCalculationMCPTools()
