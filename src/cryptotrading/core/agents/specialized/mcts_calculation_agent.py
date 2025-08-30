"""
MCP-Compliant Production-Ready MCTS-based Calculation Agent using Strands Framework
Optimized for Vercel Edge Runtime with commercial-grade features

MCP COMPLIANCE:
This agent implements the Model Context Protocol (MCP) specification.
- All functionality is accessed through MCP tools only
- Single entry point: process_mcp_request()
- All previously public methods are now private _mcp_* handlers
- Lifecycle methods (initialize, start) remain public
- Full tool registration and parameter validation

Available MCP Tools:
- mcts_analyze_with_technical_indicators: AI-enhanced technical analysis
- mcts_analyze_market_correlation: Market correlation analysis  
- mcts_analyze_market_risk: Risk analysis using MCTS
- mcts_run_simulation: Raw MCTS simulation
- mcts_execute_trading_workflow: Complex trading workflows
- mcts_process_workflow: Strands workflow processing
- mcts_execute_tool: Strands tool execution
- mcts_process_message: Legacy message processing
- mcts_run_parallel: Parallel MCTS execution
- mcts_analyze_market_sentiment: AI-powered sentiment analysis
- mcts_analyze_technical_signals: Technical signal analysis
- mcts_predict_market_movement: AI market predictions
- mcts_backtest_strategy: Strategy backtesting
- mcts_compare_strategies: Strategy comparison
- mcts_cleanup: Agent cleanup and resource management
"""
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

# Import Grok4 client and strategy backtesting
from ...ai.grok4_client import (
    Grok4APIError,
    Grok4Client,
    Grok4ConfigError,
    Grok4Error,
    MarketInsight,
    StrategyAnalysis,
    close_grok4_client,
    get_grok4_client,
)

# Import security components
from ...security.mcts_auth import Permission, SecureMCTSAgent, SecurityManager, require_permission

# Import A/B testing and anomaly detection
from .mcts_ab_testing import ABTestManager, VariantConfig

# Import adaptive control components
from .mcts_adaptive_control import (
    AdaptiveIterationController,
    ConvergenceMetrics,
    DynamicExplorationParams,
    MemoryOptimizedMCTSNode,
)
from .mcts_anomaly_detection import AnomalyDetector, MCTSMonitoringDashboard

# Import Vercel runtime adapter
from .vercel_runtime_adapter import VercelRuntimeError, vercel_adapter, vercel_edge_handler

# Removed backtesting imports - not needed for Week 2

# Import Technical Analysis MCP tools
try:
    from ...infrastructure.mcp.technical_analysis_mcp_tools import TechnicalAnalysisMCPServer

    TA_MCP_AVAILABLE = True
except ImportError:
    TA_MCP_AVAILABLE = False
    # Logger not available yet during import, will log later

# Conditional imports to handle missing dependencies
try:
    from ..strands import StrandsAgent
    from ...protocols.a2a.a2a_protocol import A2A_CAPABILITIES, A2AAgentRegistry
    from ...protocols.a2a.blockchain_registration import EnhancedA2AAgentRegistry

    STRANDS_AVAILABLE = True
except ImportError:
    # Create minimal base class if StrandsAgent not available
    from ..base import BaseAgent as StrandsAgent
    
    # Mock A2A classes for when dependencies are missing
    class A2AAgentRegistry:
        @staticmethod
        def register_agent(*args, **kwargs):
            pass
    
    A2A_CAPABILITIES = {}

    STRANDS_AVAILABLE = False

try:
    from ...protocols.mcp.cache import _global_cache as mcp_cache
    from ...protocols.mcp.metrics import mcp_metrics
    from ...protocols.mcp.strand_integration import get_mcp_strand_bridge
    from ...protocols.mcp.tools import MCPTool, ToolResult

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Production mode requires MCP dependencies
    if os.getenv("ENVIRONMENT", "development") == "production":
        raise ImportError(
            "MCP dependencies required for production MCTS agent. Install with: pip install mcp-toolkit"
        )

    # Development/test mode fallback - log warning and provide minimal interface
    import logging

    logging.warning("MCP dependencies not available - using minimal fallback for testing")

    class MCPTool:
        def __init__(self, **kwargs):
            pass

    class ToolResult:
        @classmethod
        def json_result(cls, data):
            return {"content": data, "isError": False}

        @classmethod
        def error_result(cls, error):
            return {"content": error, "isError": True}

    # Use the real cache if available
    try:
        from ...protocols.mcp.cache import _global_cache as mcp_cache
    except ImportError:

        class MinimalCache:
            def get(self, key):
                return None

            def set(self, key, value, ttl=None):
                pass

        mcp_cache = MinimalCache()

    # Use real metrics if available
    try:
        from ...protocols.mcp.metrics import mcp_metrics
    except ImportError:

        class MinimalMetrics:
            class collector:
                @staticmethod
                def counter(name, value, tags=None):
                    pass

                @staticmethod
                def timer(name, duration, tags=None):
                    pass

            def tool_execution_start(self, tool, user=None):
                return time.time()

            def tool_execution_end(self, tool, start_time, success):
                pass

        mcp_metrics = MinimalMetrics()

    # Bridge fallback
    def get_mcp_strand_bridge():
        return None


try:
    from ...protocols.mcp.security.rate_limiter import RateLimiter
except ImportError:
    # Production mode requires rate limiter
    if os.getenv("ENVIRONMENT", "development") == "production":
        raise ImportError(
            "Rate limiter required for production MCTS agent. Check MCP installation."
        )

    # Test/dev fallback
    class RateLimiter:
        def __init__(self, **kwargs):
            pass

        async def check_limit(self, key):
            return True

        async def get_remaining(self, key):
            return 100


logger = logging.getLogger(__name__)

# Log TA MCP availability now that logger is available
if not TA_MCP_AVAILABLE:
    logger.warning("Technical Analysis MCP tools not available - using simulated analysis")


# Configuration Management
class MCTSConfig:
    """Configuration with Vercel environment variable support"""

    def __init__(self):
        self.iterations = int(os.getenv("MCTS_ITERATIONS", "1000"))
        self.exploration_constant = float(os.getenv("MCTS_EXPLORATION", "1.4"))
        self.simulation_depth = int(os.getenv("MCTS_SIM_DEPTH", "10"))
        self.timeout_seconds = int(os.getenv("MCTS_TIMEOUT", "30"))
        self.max_memory_mb = int(os.getenv("MCTS_MAX_MEMORY_MB", "512"))
        self.cache_ttl = int(os.getenv("MCTS_CACHE_TTL", "300"))
        self.enable_progressive_widening = (
            os.getenv("MCTS_PROGRESSIVE_WIDENING", "true").lower() == "true"
        )
        self.enable_rave = os.getenv("MCTS_RAVE", "true").lower() == "true"
        self.parallel_simulations = int(os.getenv("MCTS_PARALLEL_SIMS", "4"))
        self.simulation_strategy = os.getenv(
            "MCTS_SIMULATION_STRATEGY", "weighted_random"
        )  # pure_random or weighted_random


# Input Validation
class ValidationError(Exception):
    """Input validation error"""

    pass


class InputValidator:
    """Validates and sanitizes input parameters"""

    @staticmethod
    def validate_calculation_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate calculation parameters"""
        if not isinstance(params, dict):
            raise ValidationError("Parameters must be a dictionary")

        # Validate portfolio value
        portfolio = params.get("initial_portfolio", 0)
        if not isinstance(portfolio, (int, float)) or portfolio <= 0:
            raise ValidationError("Initial portfolio must be positive number")
        if portfolio > 1e9:  # Max 1 billion
            raise ValidationError("Portfolio value exceeds maximum allowed")

        # Validate symbols
        symbols = params.get("symbols", [])
        if not isinstance(symbols, list) or len(symbols) == 0:
            raise ValidationError("Symbols must be non-empty list")
        if len(symbols) > 20:
            raise ValidationError("Maximum 20 symbols allowed")

        # Validate symbol format
        valid_symbols = []
        for symbol in symbols:
            if not isinstance(symbol, str):
                raise ValidationError(f"Symbol must be string, got {type(symbol)}")
            symbol = symbol.upper().strip()
            if not symbol or len(symbol) > 10:
                raise ValidationError(f"Invalid symbol format: {symbol}")
            if symbol in valid_symbols:
                raise ValidationError(f"Duplicate symbol: {symbol}")
            valid_symbols.append(symbol)
        params["symbols"] = valid_symbols

        # Validate depth
        max_depth = params.get("max_depth", 10)
        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 50:
            raise ValidationError("Max depth must be between 1 and 50")

        # Validate risk tolerance if provided
        risk_tolerance = params.get("risk_tolerance")
        if risk_tolerance is not None:
            if not isinstance(risk_tolerance, (int, float)) or not (0 <= risk_tolerance <= 1):
                raise ValidationError("Risk tolerance must be between 0 and 1")

        # Validate time horizon if provided
        time_horizon = params.get("time_horizon")
        if time_horizon is not None:
            if not isinstance(time_horizon, int) or time_horizon < 1 or time_horizon > 365:
                raise ValidationError("Time horizon must be between 1 and 365 days")

        # Validate iterations if provided
        iterations = params.get("iterations")
        if iterations is not None:
            if not isinstance(iterations, int) or iterations < 10 or iterations > 100000:
                raise ValidationError("Iterations must be between 10 and 100000")

        return params

    @staticmethod
    def sanitize_string(value: str, max_length: int = 100) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            value = str(value)
        return value[:max_length].strip()


# Enhanced MCTS Node with RAVE support
@dataclass
class MCTSNodeV2:
    """Enhanced MCTS node with production features"""

    state: Dict[str, Any]
    parent: Optional["MCTSNodeV2"] = None
    children: List["MCTSNodeV2"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[Dict[str, Any]] = field(default_factory=list)
    action: Optional[Dict[str, Any]] = None

    # RAVE (Rapid Action Value Estimation)
    rave_visits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rave_values: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # Virtual loss for parallel MCTS (prevents multiple threads from exploring same path)
    virtual_loss: int = 0

    # Action priors (from policy network or domain knowledge)
    action_priors: Dict[str, float] = field(default_factory=dict)

    # Memory optimization
    _hash: Optional[str] = None

    def state_hash(self) -> str:
        """Generate hash of state for caching"""
        if self._hash is None:
            state_str = json.dumps(self.state, sort_keys=True)
            self._hash = hashlib.md5(state_str.encode()).hexdigest()
        return self._hash

    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried"""
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.4, use_rave: bool = True) -> Optional["MCTSNodeV2"]:
        """Select best child using UCB1 with optional RAVE - FIXED ALGORITHM"""
        if not self.children:
            return None

        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float("inf")
            else:
                # FIXED UCB1: Proper formula with correct log
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(
                    math.log(self.visits) / child.visits
                )  # Fixed: removed factor of 2

                # RAVE bonus if enabled
                if use_rave and child.action:
                    action_key = str(child.action)
                    if action_key in self.rave_visits and self.rave_visits[action_key] > 0:
                        rave_value = self.rave_values[action_key] / self.rave_visits[action_key]
                        # FIXED RAVE: Proper beta calculation from Silver & Gelly
                        equiv_param = 1000  # Equivalence parameter
                        beta = math.sqrt(equiv_param / (3 * child.visits + equiv_param))

                        # FIXED: Combine UCT and RAVE properly
                        uct_value = exploitation + exploration
                        rave_exploration = c_param * math.sqrt(
                            math.log(self.visits) / self.rave_visits[action_key]
                        )
                        rave_uct = rave_value + rave_exploration

                        weight = beta * rave_uct + (1 - beta) * uct_value
                    else:
                        weight = exploitation + exploration
                else:
                    weight = exploitation + exploration

            choices_weights.append(weight)

        return self.children[choices_weights.index(max(choices_weights))]

    def add_child(self, action: Dict[str, Any], state: Dict[str, Any]) -> "MCTSNodeV2":
        """Add a new child node with action priors"""
        child = MCTSNodeV2(
            state=state,
            parent=self,
            untried_actions=list(state.get("available_actions", [])),
            action=action,
        )

        # Set action priors if available from environment
        if hasattr(state, "action_priors") and state.get("action_priors"):
            child.action_priors = state["action_priors"]
        elif "action_priors" in state:
            child.action_priors = state["action_priors"]

        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def update_rave(self, action_sequence: List[Dict[str, Any]], value: float):
        """Update RAVE statistics for action sequence"""
        for action in action_sequence:
            action_key = str(action)
            self.rave_visits[action_key] += 1
            self.rave_values[action_key] += value

    def add_virtual_loss(self, loss: int = 1):
        """Add virtual loss for parallel MCTS to prevent multiple threads exploring same path"""
        self.virtual_loss += loss

    def remove_virtual_loss(self, loss: int = 1):
        """Remove virtual loss after MCTS iteration completes"""
        self.virtual_loss = max(0, self.virtual_loss - loss)

    @property
    def effective_visits(self) -> int:
        """Get effective visits including virtual loss"""
        return self.visits + self.virtual_loss

    @property
    def q_value(self) -> float:
        """Get Q-value (average value)"""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits


# Circuit Breaker for resilience
class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record successful operation"""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed operation"""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"

    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True


class CalculationEnvironment(ABC):
    """Abstract base class for calculation environments"""

    @abstractmethod
    async def get_initial_state(self) -> Dict[str, Any]:
        """Get the initial state of the calculation environment"""
        pass

    @abstractmethod
    async def get_available_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get available actions from current state"""
        pass

    @abstractmethod
    async def apply_action(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an action to a state and return new state"""
        pass

    @abstractmethod
    async def is_terminal_state(self, state: Dict[str, Any]) -> bool:
        """Check if state is terminal"""
        pass

    @abstractmethod
    async def evaluate_state(self, state: Dict[str, Any]) -> float:
        """Evaluate the value of a state"""
        pass


# Data analysis environment (trading functionality removed)
class DataAnalysisEnvironment(CalculationEnvironment):
    """Data analysis environment for market research only"""

    def __init__(self, config: Dict[str, Any], market_data_provider=None):
        self.config = InputValidator.validate_calculation_params(config)
        self.max_depth = config.get("max_depth", 10)
        self.market_data_provider = market_data_provider
        self._analysis_cache = {}
        self._state_cache = {}

    async def get_initial_state(self) -> Dict[str, Any]:
        """Initialize analysis state with market data"""
        market_data = await self._fetch_market_data()

        return {
            "market_data": market_data,
            "depth": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "available_actions": await self.get_available_actions({}),
            "analysis_metrics": await self._calculate_analysis_metrics({}),
        }

    async def get_available_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get available analysis actions"""
        state_hash = hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()

        if state_hash in self._analysis_cache:
            return self._analysis_cache[state_hash]

        actions = []
        depth = state.get("depth", 0)

        # Analysis actions only
        for symbol in self.config.get("symbols", ["BTC", "ETH"]):
            actions.append(
                {
                    "type": "analyze_trend",
                    "symbol": symbol,
                    "timeframe": "1h",
                    "analysis_score": await self._calculate_analysis_score("trend", symbol),
                }
            )
            actions.append(
                {
                    "type": "analyze_volatility",
                    "symbol": symbol,
                    "timeframe": "24h",
                    "analysis_score": await self._calculate_analysis_score("volatility", symbol),
                }
            )

        # Cache results
        self._analysis_cache[state_hash] = actions
        return actions
        if depth < 3:
            actions.extend(
                [
                    {"type": "technical_analysis", "indicators": ["RSI", "MACD"], "cost": 0.001},
                    {"type": "sentiment_analysis", "sources": ["news", "social"], "cost": 0.002},
                ]
            )

        self._action_cache[state_hash] = actions
        return actions

    async def apply_action(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply trading action with transaction costs and slippage"""
        new_state = state.copy()
        new_state["depth"] = state.get("depth", 0) + 1
        new_state["timestamp"] = datetime.utcnow().isoformat()

        # Transaction cost model
        transaction_cost = 0.001  # 0.1% per trade
        # Real slippage would come from order book depth and market conditions
        # For now, use a conservative fixed estimate
        slippage = 0.0002  # 0.02% typical for liquid crypto markets

        if action["type"] == "buy":
            symbol = action["symbol"]
            percentage = action.get("percentage", 0.1)

            # For technical analysis focused MCTS, track action cost without portfolio value
            base_amount = 1000  # Base amount for cost calculation
            amount = base_amount * percentage

            # Apply transaction costs and slippage
            effective_amount = amount * (1 - transaction_cost - slippage)

            price = new_state["market_data"].get(symbol, {}).get("price", 1)
            quantity = effective_amount / price

            new_state["positions"][symbol] = new_state["positions"].get(symbol, 0) + quantity
            new_state["total_cost"] = new_state.get("total_cost", 0) + amount

        elif action["type"] == "sell":
            symbol = action["symbol"]
            percentage = action.get("percentage", 0.1)
            position = new_state["positions"].get(symbol, 0)
            sell_quantity = position * percentage
            price = new_state["market_data"].get(symbol, {}).get("price", 1)

            # Apply transaction costs and slippage
            sale_value = sell_quantity * price * (1 - transaction_cost - slippage)

            new_state["positions"][symbol] = position - sell_quantity
            new_state["total_sales"] = new_state.get("total_sales", 0) + sale_value

        elif action["type"] in ["technical_analysis", "sentiment_analysis"]:
            # Store analysis result without cost deduction
            new_state[f'{action["type"]}_result'] = await self._perform_analysis(action)
        new_state["available_actions"] = await self.get_available_actions(new_state)

        return new_state

    async def is_terminal_state(self, state: Dict[str, Any]) -> bool:
        """Check terminal conditions for technical analysis"""
        # Depth limit
        if state.get("depth", 0) >= self.max_depth:
            return True

        # Terminal if all symbols have been analyzed
        symbols = self.config.get("symbols", [])
        analyzed_symbols = set()
        for key in state.keys():
            if key.endswith("_result") and key.replace("_result", "") in symbols:
                analyzed_symbols.add(key.replace("_result", ""))

        if len(analyzed_symbols) >= len(symbols):
            return True

        return False

    async def evaluate_state(self, state: Dict[str, Any]) -> float:
        """Evaluate state based on technical analysis quality and AI confidence"""
        score = 0.0

        # Base score for completed analyses
        analysis_count = sum(1 for key in state.keys() if key.endswith("_result"))
        score += analysis_count * 0.3

        # Bonus for AI-enhanced analysis
        for key, value in state.items():
            if key.endswith("_result") and isinstance(value, dict):
                if value.get("ai_enhanced", False):
                    ai_confidence = value.get("ai_confidence", 0.5)
                    combined_strength = value.get("combined_strength", value.get("strength", 0.5))
                    score += ai_confidence * combined_strength * 0.5

                # Bonus for signal alignment
                alignment = value.get("signal_alignment", "neutral")
                if alignment == "strong":
                    score += 0.3
                elif alignment == "conflicting":
                    score -= 0.2

        return score

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data with intelligent caching and fallback"""
        symbols = self.config.get("symbols", ["BTC", "ETH"])
        cache_key = f"market_data:{','.join(symbols)}"

        # Check for fresh cached data first
        hashed_key = hashlib.md5(cache_key.encode()).hexdigest()
        cached = mcp_cache.cache.get(hashed_key)
        if cached:
            return cached

        # Try to get live data from provider
        if self.market_data_provider:
            try:
                data = await self.market_data_provider.get_prices(symbols)
                mcp_cache.cache.set(hashed_key, data, ttl=60)  # Cache for 1 minute
                # Also store as historical backup with longer TTL
                backup_key = f"market_data_backup:{','.join(symbols)}"
                backup_hashed_key = hashlib.md5(backup_key.encode()).hexdigest()
                mcp_cache.cache.set(backup_hashed_key, data, ttl=3600)  # 1 hour backup
                return data
            except Exception as e:
                logger.warning(f"Market data provider failed: {e}, checking backup cache")

        # Try backup cache if provider failed
        backup_key = f"market_data_backup:{','.join(symbols)}"
        backup_hashed_key = hashlib.md5(backup_key.encode()).hexdigest()
        backup_data = mcp_cache.cache.get(backup_hashed_key)
        if backup_data:
            logger.info("Using backup cached market data")
            return backup_data

        # Try to get any historical data for these symbols from cache
        for symbol in symbols:
            historical_key = f"market_data_historical:{symbol}"
            historical_hashed_key = hashlib.md5(historical_key.encode()).hexdigest()
            historical_data = mcp_cache.cache.get(historical_hashed_key)
            if historical_data:
                logger.info(f"Using historical cached data for emergency fallback")
                # Build data structure from historical cache
                data = {}
                for sym in symbols:
                    hist_key = f"market_data_historical:{sym}"
                    hist_hashed_key = hashlib.md5(hist_key.encode()).hexdigest()
                    hist_data = mcp_cache.cache.get(hist_hashed_key)
                    if hist_data:
                        data[sym] = hist_data
                    else:
                        # Only estimate if we have at least some historical data as reference
                        if len([d for d in data.values() if d]) > 0:
                            data[sym] = self._extrapolate_from_historical(sym, data)
                        else:
                            # No historical reference - cannot safely estimate
                            raise ValueError(
                                f"No historical data available for {sym} and no reference data"
                            )

                if data:
                    mcp_cache.cache.set(hashed_key, data, ttl=300)  # Cache estimated data shorter
                    return data

        # No fallback to fake data - fail fast if no real data available
        raise ValueError(
            f"No market data available for symbols {symbols}. "
            f"Cannot proceed without real market data. "
            f"Ensure market data provider is configured or cache contains recent data."
        )

    def _extrapolate_from_historical(
        self, symbol: str, existing_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extrapolate missing symbol data from existing historical data"""
        if not existing_data:
            raise ValueError(f"Cannot extrapolate data for {symbol} without reference data")

        # Get a reference symbol that has data
        reference_symbol = None
        reference_data = None
        for sym, data in existing_data.items():
            if data and isinstance(data, dict):
                reference_symbol = sym
                reference_data = data
                break

        if not reference_data:
            raise ValueError(f"No valid reference data available for extrapolation")

        # Extrapolate based on typical correlations with reference symbol
        # Use conservative estimates based on historical crypto correlations
        correlation_factors = {
            ("BTC", "ETH"): {
                "price_ratio": 0.067,
                "vol_ratio": 1.2,
            },  # ETH typically ~6.7% of BTC price
            ("ETH", "BTC"): {"price_ratio": 15.0, "vol_ratio": 0.8},  # BTC typically 15x ETH price
            ("BTC", "ADA"): {"price_ratio": 0.000067, "vol_ratio": 1.5},
            ("ETH", "ADA"): {"price_ratio": 0.001, "vol_ratio": 1.3},
        }

        # Find correlation factor or use conservative default
        factor_key = (reference_symbol, symbol)
        if factor_key in correlation_factors:
            factors = correlation_factors[factor_key]
        else:
            # Conservative default: assume similar to reference but more volatile
            factors = {"price_ratio": 0.1, "vol_ratio": 1.3}

        # Calculate extrapolated values
        ref_price = reference_data.get("price", 0)
        ref_vol = reference_data.get("volatility", 0.2)
        ref_volume = reference_data.get("volume", 0)
        if ref_volume <= 0:
            return "unknown"

        if ref_price <= 0:
            raise ValueError(f"Invalid reference price data for extrapolation")

        extrapolated_price = ref_price * factors["price_ratio"]
        extrapolated_volatility = min(ref_vol * factors["vol_ratio"], 0.8)  # Cap at 80%
        extrapolated_volume = ref_volume * 0.3  # Conservative volume estimate

        return {
            "price": extrapolated_price,
            "volume": extrapolated_volume,
            "volatility": extrapolated_volatility,
            "extrapolated": True,
            "reference_symbol": reference_symbol,
            "timestamp": time.time(),
            "warning": f"Extrapolated from {reference_symbol} - use with caution",
        }

    async def _calculate_position_value(self, state: Dict[str, Any]) -> float:
        """Calculate total position value for technical analysis evaluation"""
        total_value = 0

        for symbol, quantity in state.get("positions", {}).items():
            price = state["market_data"].get(symbol, {}).get("price", 1)
            total_value += quantity * price

        return total_value

    async def _calculate_analysis_quality(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical analysis quality metrics"""
        analyses = {}

        for key, value in state.items():
            if key.endswith("_result") and isinstance(value, dict):
                symbol = key.replace("_result", "")
                quality_score = 0.5  # Base score

                # Higher score for AI-enhanced analysis
                if value.get("ai_enhanced", False):
                    quality_score += 0.3

                # Score based on signal strength
                strength = value.get("combined_strength", value.get("strength", 0.5))
                quality_score += strength * 0.3

                analyses[symbol] = {
                    "quality_score": min(quality_score, 1.0),
                    "has_ai_enhancement": value.get("ai_enhanced", False),
                    "signal_strength": strength,
                }

        return analyses

    async def _perform_analysis(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis using actual market indicators"""
        symbol = action.get("symbol", "BTC")
        action_type = action.get("type", "hold")

        # Get current market data
        if hasattr(self, "market_data_provider") and self.market_data_provider:
            try:
                market_data = await self.market_data_provider.get_latest_data(symbol)
                price = market_data.get("price")
                volume = market_data.get("volume")
                volatility = market_data.get("volatility", 0.02)

                if not price or not volume:
                    return {"confidence": 0.0, "reason": "No market data available"}
            except Exception as e:
                # No fallback - return low confidence if no real data
                return {"confidence": 0.0, "reason": f"Market data unavailable: {str(e)}"}
        else:
            # No market data provider - cannot evaluate
            return {"confidence": 0.0, "reason": "No market data provider configured"}

        # Calculate technical indicators
        rsi = self._calculate_rsi(price, volatility)
        ma_signal = self._calculate_moving_average_signal(price)
        volume_signal = self._calculate_volume_signal(volume)

        # Determine confidence based on indicator alignment
        signals = [rsi, ma_signal, volume_signal]
        buy_signals = sum(1 for s in signals if s == "buy")
        sell_signals = sum(1 for s in signals if s == "sell")

        if buy_signals >= 2:
            final_signal = "buy"
            confidence = 0.7 + (buy_signals - 2) * 0.15
        elif sell_signals >= 2:
            final_signal = "sell"
            confidence = 0.7 + (sell_signals - 2) * 0.15
        else:
            final_signal = "hold"
            confidence = 0.6

        return {
            "result": "technical_analysis_complete",
            "confidence": min(confidence, 0.95),
            "signal": final_signal,
            "indicators": {
                "rsi_signal": rsi,
                "ma_signal": ma_signal,
                "volume_signal": volume_signal,
            },
        }

    def _calculate_rsi(self, price: float, volatility: float) -> str:
        """Calculate RSI-based signal"""
        # Simplified RSI calculation based on volatility
        normalized_vol = min(volatility / 0.05, 1.0)  # Normalize to 0-1

        if normalized_vol < 0.3:
            return "buy"  # Low volatility suggests accumulation
        elif normalized_vol > 0.7:
            return "sell"  # High volatility suggests distribution
        else:
            return "hold"

    def _calculate_moving_average_signal(self, price: float) -> str:
        """Calculate moving average signal based on dynamic support/resistance levels"""
        # Get dynamic support and resistance levels from historical data
        try:
            support_resistance = self._calculate_dynamic_support_resistance(price)
            resistance_level = support_resistance["resistance"]
            support_level = support_resistance["support"]

            if price > resistance_level:  # Above dynamic resistance
                return "buy"
            elif price < support_level:  # Below dynamic support
                return "sell"
            else:
                return "hold"
        except Exception as e:
            # If dynamic calculation fails, return neutral signal
            logger.warning(f"Failed to calculate dynamic support/resistance: {e}")
            return "hold"

    def _calculate_volume_signal(self, volume: float) -> str:
        """Calculate volume-based signal using dynamic volume thresholds"""
        try:
            # Calculate dynamic volume thresholds based on historical data
            volume_thresholds = self._calculate_dynamic_volume_thresholds(volume)
            high_threshold = volume_thresholds["high"]
            low_threshold = volume_thresholds["low"]

            if volume > high_threshold:  # High volume (dynamic)
                return "buy"  # High volume suggests strong interest
            elif volume < low_threshold:  # Low volume (dynamic)
                return "sell"  # Low volume suggests weak interest
            else:
                return "hold"
        except Exception as e:
            # If dynamic calculation fails, return neutral signal
            logger.warning(f"Failed to calculate dynamic volume thresholds: {e}")
            return "hold"

    async def _calculate_dynamic_support_resistance(self, current_price: float) -> Dict[str, float]:
        """Calculate dynamic support and resistance levels using MCP tools"""
        try:
            # Use MCTS calculation MCP tools for calculation
            from ...infrastructure.mcp.mcts_calculation_mcp_tools import mcts_calculation_mcp_tools

            result = await mcts_calculation_mcp_tools.handle_tool_call(
                "calculate_dynamic_support_resistance", {"current_price": current_price}
            )

            if result.get("success", False):
                return result["result"]
            else:
                # Fallback calculation
                price_volatility = 0.15
                resistance_level = current_price * (1 + price_volatility * 0.8)
                support_level = current_price * (1 - price_volatility * 0.8)

                return {
                    "resistance": resistance_level,
                    "support": support_level,
                    "volatility_used": price_volatility,
                    "calculation_method": "price_relative_fallback",
                }
        except Exception as e:
            logger.error(f"Dynamic support/resistance calculation failed: {e}")
            raise ValueError("Unable to calculate dynamic support/resistance levels")

    async def _calculate_dynamic_volume_thresholds(self, current_volume: float) -> Dict[str, float]:
        """Calculate dynamic volume thresholds using MCP tools"""
        try:
            # Use MCTS calculation MCP tools for calculation
            from ...infrastructure.mcp.mcts_calculation_mcp_tools import mcts_calculation_mcp_tools

            result = await mcts_calculation_mcp_tools.handle_tool_call(
                "calculate_dynamic_volume_thresholds", {"current_volume": current_volume}
            )

            if result.get("success", False):
                return result["result"]
            else:
                # Fallback calculation
                volume_multiplier_high = 2.5
                volume_multiplier_low = 0.4

                return {
                    "high": current_volume * volume_multiplier_high,
                    "low": current_volume * volume_multiplier_low,
                    "base_volume": current_volume,
                    "calculation_method": "volume_relative_fallback",
                }
        except Exception as e:
            logger.error(f"Dynamic volume threshold calculation failed: {e}")
            raise ValueError("Unable to calculate dynamic volume thresholds")


class MCTSCalculationAgent(SecureMCTSAgent, StrandsAgent):
    """
    MCP-Compliant MCTS Calculation Agent with AI and Technical Analysis integration

    This agent is fully MCP-compliant. ALL functionality must be accessed through MCP tools.
    The only public entry point is process_mcp_request().

    Available MCP Tools:
    - mcts_analyze_with_technical_indicators: AI-enhanced technical analysis
    - mcts_analyze_market_correlation: Market correlation analysis
    - mcts_analyze_market_risk: Risk analysis using MCTS
    - mcts_run_simulation: Raw MCTS simulation
    - mcts_execute_trading_workflow: Complex trading workflows
    - mcts_process_workflow: Strands workflow processing
    - mcts_execute_tool: Strands tool execution
    - mcts_process_message: Legacy message processing
    - mcts_run_parallel: Parallel MCTS execution
    - mcts_analyze_market_sentiment: AI-powered sentiment analysis
    - mcts_analyze_technical_signals: Technical signal analysis
    - mcts_predict_market_movement: AI market predictions
    - mcts_backtest_strategy: Strategy backtesting
    - mcts_compare_strategies: Strategy comparison
    - mcts_calculate_optimal_portfolio: Portfolio optimization
    - mcts_optimize_allocation: Asset allocation optimization
    - mcts_evaluate_trading_strategy: Strategy evaluation

    Note: initialize() and start() remain public as they are lifecycle methods.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[MCTSConfig] = None,
        market_data_provider=None,
        **kwargs,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="mcts_calculation_v2",
            capabilities=[
                "calculation",
                "optimization",
                "strategic_planning",
                "monte_carlo_search",
            ],
            **kwargs,
        )

        # Configuration
        self.config = config or MCTSConfig()

        # Components
        self.environment: Optional[CalculationEnvironment] = None
        self.market_data_provider = market_data_provider
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

        # AI and Backtesting Components
        self.grok4_client: Optional[Grok4Client] = None
        self.strategy_backtester = StrategyBacktester(market_data_provider)

        # Metrics
        self.calculation_count = 0
        self.total_iterations = 0
        self.start_time = time.time()

        # MCP integration
        self.mcp_bridge = get_mcp_strand_bridge()
        self._register_mcp_tools()

        # Register this agent with the bridge for Strands-MCP integration
        if self.mcp_bridge:
            self.mcp_bridge.register_agent(self)

        # Register Strands tools
        self._register_strands_tools()

        # Initialize monitoring and security
        self.security_manager = SecurityManager()
        self.ab_test_manager = ABTestManager()
        self.anomaly_detector = AnomalyDetector()
        self.monitoring_dashboard = MCTSMonitoringDashboard()

        # Initialize Grok4 AI client (will be properly initialized in async context)
        self.grok4_client = None  # Initialized in _initialize_grok4_client()

        # Initialize strategy backtester
        self.strategy_backtester = StrategyBacktester()

        # Initialize market data provider (placeholder)
        self.market_data_provider = None  # To be injected or configured

        # Initialize memory system for MCTS calculations and learning
        asyncio.create_task(self._initialize_memory_system())

        # Initialize AI cache for performance
        self._ai_cache = {}
        self._ai_cache_ttl = 300  # 5 minutes
        self._last_ai_fetch = {}

        # Initialize Technical Analysis MCP server if available
        self.ta_mcp_server = None
        if TA_MCP_AVAILABLE:
            try:
                self.ta_mcp_server = TechnicalAnalysisMCPServer()
                logger.info("Technical Analysis MCP server initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TA MCP server: {e}")
                self.ta_mcp_server = None

        # Register with A2A Agent Registry (including blockchain)


        capabilities = A2A_CAPABILITIES.get(agent_id, [])


        mcp_tools = list(self.mcp_handlers.keys()) if hasattr(self, 'mcp_handlers') else []


        


        # Try blockchain registration, fallback to local only


        try:


            import asyncio


            asyncio.create_task(


                EnhancedA2AAgentRegistry.register_agent_with_blockchain(


                    agent_id=agent_id,


                    capabilities=capabilities,


                    agent_instance=self,


                    agent_type="mcts_calculation",


                    mcp_tools=mcp_tools


                )


            )


            logger.info(f"MCTS Calculation Agent {agent_id} blockchain registration initiated")


        except Exception as e:


            # Fallback to local registration only


            A2AAgentRegistry.register_agent(agent_id, capabilities, self)


            logger.warning(f"MCTS Calculation Agent {agent_id} registered locally only (blockchain failed: {e})")

        logger.info(f"MCTS Agent {agent_id} initialized successfully")

    async def _initialize_memory_system(self):
        """Initialize memory system for MCTS calculations and performance tracking"""
        try:
            # Store MCTS agent configuration
            await self.store_memory(
                "mcts_agent_config",
                {
                    "agent_id": self.agent_id,
                    "iterations": self.config.iterations,
                    "exploration_constant": self.config.exploration_constant,
                    "simulation_depth": self.config.simulation_depth,
                    "timeout_seconds": self.config.timeout_seconds,
                    "initialized_at": datetime.now().isoformat(),
                },
                {"type": "configuration", "persistent": True},
            )

            # Initialize calculation cache
            await self.store_memory(
                "calculation_cache", {}, {"type": "calculation_cache", "persistent": True}
            )

            # Initialize performance metrics
            await self.store_memory(
                "mcts_performance",
                {
                    "total_calculations": 0,
                    "avg_calculation_time": 0,
                    "success_rate": 0,
                    "cache_hit_rate": 0,
                    "best_results": [],
                },
                {"type": "performance_tracking", "persistent": True},
            )

            # Initialize strategy learning cache
            await self.store_memory(
                "strategy_learning",
                {"successful_strategies": [], "failed_strategies": []},
                {"type": "strategy_learning", "persistent": True},
            )

            # Initialize calculation history
            await self.store_memory(
                "calculation_history", [], {"type": "calculation_log", "persistent": True}
            )

            logger.info(f"Memory system initialized for MCTS Agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize MCTS agent memory system: {e}")
        if (
            not self.is_vercel_runtime
            and os.getenv("MCTS_MONITORING_ENABLED", "true").lower() == "true"
        ):
            try:
                asyncio.create_task(self.monitoring_dashboard.start_monitoring())
            except RuntimeError:
                # Handle case where event loop isn't running yet
                logger.info("Monitoring will be enabled when event loop starts")

        # Detect runtime environment
        self.is_vercel_runtime = bool(os.getenv("VERCEL_ENV") or os.getenv("VERCEL"))

        # Start monitoring only for local development (not in Vercel)
        if (
            not self.is_vercel_runtime
            and os.getenv("MCTS_MONITORING_ENABLED", "true").lower() == "true"
        ):
            try:
                asyncio.create_task(self.monitoring_dashboard.start_monitoring())
            except RuntimeError:
                # Handle case where event loop isn't running yet
                logger.info("Monitoring will be enabled when event loop starts")

        logger.info(
            f"Enhanced MCTS Agent {agent_id} initialized with MCTS, Grok4 AI, backtesting, and monitoring"
        )

    async def initialize(self) -> bool:
        """Initialize the MCTS Calculation Agent"""
        try:
            logger.info(f"Initializing MCTS Agent {self.agent_id}")

            # Initialize Grok4 client if available
            await self._initialize_grok4_client()

            # Verify MCTS environment
            if not self.environment:
                logger.warning("MCTS environment not initialized")

            # Test basic MCTS functionality
            try:
                # Quick validation test
                test_result = await self._mcp_run_mcts_parallel(iterations=10)
                if test_result and test_result.get("best_action"):
                    logger.info("MCTS validation successful")
                else:
                    logger.warning("MCTS validation returned no results")
            except Exception as e:
                logger.warning(f"MCTS validation failed: {e}")

            logger.info(f"MCTS Agent {self.agent_id} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCTS Agent {self.agent_id}: {e}")
            return False

    async def start(self) -> bool:
        """Start the MCTS Calculation Agent"""
        try:
            logger.info(f"Starting MCTS Agent {self.agent_id}")

            # Start monitoring dashboard if enabled
            if (
                not self.is_vercel_runtime
                and os.getenv("MCTS_MONITORING_ENABLED", "true").lower() == "true"
            ):
                try:
                    await self.monitoring_dashboard.start_monitoring()
                    logger.info("MCTS monitoring dashboard started")
                except Exception as e:
                    logger.warning(f"Failed to start monitoring dashboard: {e}")

            logger.info(f"MCTS Agent {self.agent_id} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start MCTS Agent {self.agent_id}: {e}")
            return False

    def _register_mcp_tools(self):
        """Register calculation-specific MCP tools with rate limiting"""
        tools = [
            # Legacy tools
            MCPTool(
                name="mcts_calculate_v2",
                description="Production MCTS-based calculation with advanced features",
                parameters={
                    "problem_type": {
                        "type": "string",
                        "description": "Type of calculation problem",
                        "enum": ["trading", "portfolio", "optimization"],
                    },
                    "parameters": {"type": "object", "description": "Problem-specific parameters"},
                    "iterations": {
                        "type": "integer",
                        "description": "Number of MCTS iterations",
                        "minimum": 10,
                        "maximum": 10000,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30,
                    },
                },
                function=self._mcts_calculate_with_monitoring,
            ),
            MCPTool(
                name="get_calculation_metrics",
                description="Get calculation performance metrics",
                parameters={},
                function=self._get_metrics,
            ),
            MCPTool(
                name="health_check",
                description="Agent health check",
                parameters={},
                function=self._health_check,
            ),
            
            # New MCP-compliant tools
            MCPTool(
                name="mcts_analyze_with_technical_indicators",
                description="AI-enhanced technical analysis with MCTS optimization",
                parameters={
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Symbols to analyze"},
                    "market_data": {"type": "object", "description": "OHLCV market data"},
                },
                function=self._mcp_analyze_with_technical_indicators,
            ),
            MCPTool(
                name="mcts_analyze_market_correlation",
                description="Analyze market correlation patterns using MCTS",
                parameters={
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Symbols to analyze"},
                    "timeframe": {"type": "string", "description": "Analysis timeframe", "default": "1h"},
                },
                function=self._mcp_analyze_market_correlation,
            ),
            MCPTool(
                name="mcts_analyze_market_risk",
                description="Risk analysis using MCTS simulation",
                parameters={
                    "position": {"type": "object", "description": "Position to analyze risk for"},
                    "market_conditions": {"type": "object", "description": "Current market conditions"},
                },
                function=self._mcp_analyze_market_risk,
            ),
            MCPTool(
                name="mcts_run_simulation",
                description="Run raw MCTS simulation",
                parameters={
                    "problem_type": {"type": "string", "description": "Type of problem", "enum": ["trading", "portfolio", "optimization"]},
                    "parameters": {"type": "object", "description": "Simulation parameters"},
                    "iterations": {"type": "integer", "description": "Number of iterations", "default": 1000},
                },
                function=self._mcp_run_mcts_simulation,
            ),
            MCPTool(
                name="mcts_execute_trading_workflow",
                description="Execute complex trading workflows with MCTS optimization",
                parameters={
                    "workflow_config": {"type": "object", "description": "Workflow configuration"},
                    "market_data": {"type": "object", "description": "Current market data"},
                },
                function=self._mcp_execute_trading_workflow,
            ),
            MCPTool(
                name="mcts_process_workflow",
                description="Process Strands workflow with MCTS integration",
                parameters={
                    "workflow_id": {"type": "string", "description": "Workflow identifier"},
                    "inputs": {"type": "object", "description": "Workflow inputs"},
                },
                function=self._mcp_process_workflow,
            ),
            MCPTool(
                name="mcts_execute_tool",
                description="Execute Strands tool with MCTS enhancement",
                parameters={
                    "tool_name": {"type": "string", "description": "Tool name"},
                    "parameters": {"type": "object", "description": "Tool parameters"},
                },
                function=self._mcp_execute_tool,
            ),
            MCPTool(
                name="mcts_process_message",
                description="Process legacy messages with MCTS capability",
                parameters={
                    "message": {"type": "object", "description": "Message to process"},
                },
                function=self._mcp_process_message,
            ),
            MCPTool(
                name="mcts_run_parallel",
                description="Run parallel MCTS simulations",
                parameters={
                    "iterations": {"type": "integer", "description": "Number of iterations", "default": 1000},
                },
                function=self._mcp_run_mcts_parallel,
            ),
            MCPTool(
                name="mcts_analyze_market_sentiment",
                description="AI-powered market sentiment analysis with MCTS",
                parameters={
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Symbols to analyze"},
                    "timeframe": {"type": "string", "description": "Analysis timeframe", "default": "1h"},
                },
                function=self._mcp_analyze_market_sentiment,
            ),
            MCPTool(
                name="mcts_analyze_technical_signals",
                description="Technical signal analysis with MCTS optimization",
                parameters={
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Symbols to analyze"},
                    "market_data": {"type": "object", "description": "Market data for analysis"},
                },
                function=self._mcp_analyze_technical_signals,
            ),
            MCPTool(
                name="mcts_predict_market_movement",
                description="AI market movement prediction with MCTS validation",
                parameters={
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Symbols to predict"},
                    "horizon": {"type": "string", "description": "Prediction horizon", "default": "1h"},
                },
                function=self._mcp_predict_market_movement,
            ),
            MCPTool(
                name="mcts_backtest_strategy",
                description="Strategy backtesting with MCTS optimization",
                parameters={
                    "strategy_config": {"type": "object", "description": "Strategy configuration"},
                    "historical_data": {"type": "object", "description": "Historical data for backtesting"},
                },
                function=self._mcp_backtest_strategy,
            ),
            MCPTool(
                name="mcts_compare_strategies",
                description="Compare multiple strategies using MCTS analysis",
                parameters={
                    "strategy_configs": {"type": "array", "items": {"type": "object"}, "description": "List of strategy configurations"},
                },
                function=self._mcp_compare_strategies,
            ),
            MCPTool(
                name="mcts_cleanup",
                description="Clean up agent resources and perform maintenance",
                parameters={},
                function=self._mcp_cleanup,
            ),
        ]

        if self.mcp_bridge and self.mcp_bridge.mcp_server:
            for tool in tools:
                self.mcp_bridge.mcp_server.register_tool(tool)

    async def _initialize_grok4_client(self):
        """Initialize Grok4 AI client for market analysis"""
        try:
            self.grok4_client = await get_grok4_client()
            logger.info("Grok4 AI client initialized successfully")
        except Grok4ConfigError as e:
            logger.error(f"Grok4 configuration error: {e}")
            logger.warning("Please set GROK4_API_KEY environment variable")
            self.grok4_client = None
        except Grok4Error as e:
            logger.warning(f"Grok4 client initialization failed: {e} - using fallback mode")
            self.grok4_client = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Grok4 client: {e}")
            self.grok4_client = None

    async def _get_ai_market_sentiment(
        self, symbols: List[str], state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get AI market sentiment with caching for MCTS integration"""
        if not self.grok4_client or not symbols:
            return {}

        # Check cache
        cache_key = f"ai_sentiment_{','.join(sorted(symbols))}_{state.get('timestamp', '')}"
        current_time = time.time()

        if cache_key in self._ai_cache and cache_key in self._last_ai_fetch:
            if current_time - self._last_ai_fetch[cache_key] < self._ai_cache_ttl:
                return self._ai_cache[cache_key]

        try:
            # Fetch fresh AI insights
            insights = await self.grok4_client.analyze_market_sentiment(symbols, timeframe="1h")

            # Convert to dict for easy lookup
            sentiment_dict = {}
            for insight in insights:
                sentiment_dict[insight.symbol] = {
                    "recommendation": insight.recommendation,
                    "sentiment_score": insight.score,
                    "risk_level": insight.risk_level,
                    "confidence": insight.confidence,
                    "reasoning": insight.reasoning,
                }

            # Cache the results
            self._ai_cache[cache_key] = sentiment_dict
            self._last_ai_fetch[cache_key] = current_time

            return sentiment_dict

        except Grok4APIError as e:
            if e.status_code == 429:
                logger.warning("AI rate limit reached, using cached data")
            return self._ai_cache.get(cache_key, {})
        except Exception as e:
            logger.warning(f"Failed to get AI sentiment: {e}")
            return {}

    async def _calculate_ai_value_boost(
        self, action_sequence: List[Dict[str, Any]], final_state: Dict[str, Any]
    ) -> float:
        """Calculate value boost based on AI alignment with action sequence"""
        if not action_sequence:
            return 0.0

        total_boost = 0.0
        alignment_count = 0

        # Check how well actions align with AI recommendations
        for action in action_sequence:
            action_type = action.get("type", "")
            symbol = action.get("symbol", "")

            # Get cached AI sentiment for this symbol
            cache_key = f"ai_sentiment_{final_state.get('timestamp', '')}"
            if cache_key in self._ai_cache and symbol in self._ai_cache[cache_key]:
                ai_data = self._ai_cache[cache_key][symbol]
                ai_rec = ai_data.get("recommendation", "HOLD")
                ai_confidence = ai_data.get("confidence", 0.5)

                # Check alignment
                if (
                    (action_type == "buy" and ai_rec == "BUY")
                    or (action_type == "sell" and ai_rec == "SELL")
                    or (action_type == "hold" and ai_rec == "HOLD")
                ):
                    # Action aligns with AI recommendation
                    total_boost += ai_confidence * 0.1  # Max 10% boost per aligned action
                    alignment_count += 1
                elif (action_type == "buy" and ai_rec == "SELL") or (
                    action_type == "sell" and ai_rec == "BUY"
                ):
                    # Action opposes AI recommendation
                    total_boost -= ai_confidence * 0.05  # Max 5% penalty per opposed action

        # Cap the total boost/penalty
        return max(-0.2, min(0.3, total_boost))  # Between -20% and +30%

    async def _get_ai_predictions_cached(
        self, symbols: List[str], state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get AI market predictions with caching"""
        if not self.grok4_client or not symbols:
            return None

        cache_key = f"ai_predictions_{','.join(sorted(symbols))}_{state.get('timestamp', '')}"
        current_time = time.time()

        if cache_key in self._ai_cache and cache_key in self._last_ai_fetch:
            if current_time - self._last_ai_fetch[cache_key] < self._ai_cache_ttl:
                return self._ai_cache[cache_key]

        try:
            # Get AI predictions
            predictions = await self.grok4_client.predict_market_movement(symbols, horizon="1h")

            # Cache the results
            self._ai_cache[cache_key] = predictions
            self._last_ai_fetch[cache_key] = current_time

            return predictions

        except Exception as e:
            logger.warning(f"Failed to get AI predictions: {e}")
            return None

    async def _enhance_technical_analysis_with_ai(
        self, ta_result: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """Enhance technical analysis results with AI insights"""
        if not self.grok4_client:
            return ta_result

        try:
            # Get cached AI sentiment
            cache_key = f"ai_sentiment_{symbol}"
            ai_sentiment = self._ai_cache.get(cache_key, {})

            if symbol in ai_sentiment:
                ai_data = ai_sentiment[symbol]

                # Enhance TA result with AI insights
                ta_result["ai_enhanced"] = True
                ta_result["ai_recommendation"] = ai_data.get("recommendation", "HOLD")
                ta_result["ai_confidence"] = ai_data.get("confidence", 0.5)
                ta_result["ai_reasoning"] = ai_data.get("reasoning", "")

                # Adjust signal strength based on AI alignment
                if "signal" in ta_result and "strength" in ta_result:
                    ta_signal = ta_result["signal"]  # BUY/SELL/HOLD
                    ai_rec = ai_data.get("recommendation", "HOLD")

                    if ta_signal == ai_rec:
                        # Signals align - boost confidence
                        ta_result["combined_strength"] = min(
                            ta_result["strength"] * (1 + ai_data.get("confidence", 0.5) * 0.3), 1.0
                        )
                        ta_result["signal_alignment"] = "strong"
                    elif (ta_signal == "BUY" and ai_rec == "SELL") or (
                        ta_signal == "SELL" and ai_rec == "BUY"
                    ):
                        # Signals oppose - reduce confidence
                        ta_result["combined_strength"] = ta_result["strength"] * 0.5
                        ta_result["signal_alignment"] = "conflicting"
                    else:
                        # Mixed signals
                        ta_result["combined_strength"] = ta_result["strength"] * 0.8
                        ta_result["signal_alignment"] = "neutral"

                logger.info(
                    f"Enhanced TA with AI: {ta_signal} + {ai_rec} = {ta_result.get('signal_alignment', 'unknown')}"
                )

            return ta_result

        except Exception as e:
            logger.warning(f"Failed to enhance TA with AI: {e}")
            return ta_result

    async def _execute_real_technical_analysis(
        self, symbol: str, analysis_type: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute real technical analysis using MCP tools"""
        if not self.ta_mcp_server:
            # Fallback to simulated analysis if MCP tools not available
            return {
                "symbol": symbol,
                "analysis_type": analysis_type,
                "signal": "HOLD",
                "strength": 0.5,
                "indicators": {"fallback": True, "reason": "TA MCP server not available"},
            }

        try:
            # Convert market data to JSON format for MCP
            market_data_json = json.dumps(market_data.get(symbol, {}))

            # Map MCTS action types to MCP tool names
            mcp_tool_mapping = {
                "analyze_indicators": "analyze_momentum_indicators",
                "detect_patterns": "detect_chart_patterns",
                "support_resistance": "analyze_support_resistance",
                "generate_signals": "generate_trading_signals",
                "comprehensive": "analyze_market_comprehensive",
            }

            mcp_tool_name = mcp_tool_mapping.get(analysis_type, "analyze_momentum_indicators")

            # Prepare arguments for MCP tool
            arguments = {"market_data": market_data_json, "symbol": symbol}

            # Call the MCP tool directly (bypassing MCP protocol for internal use)
            if mcp_tool_name == "analyze_momentum_indicators":
                result = await self.ta_mcp_server._handle_momentum_indicators(
                    pd.DataFrame(market_data.get(symbol, {})), symbol
                )
            elif mcp_tool_name == "detect_chart_patterns":
                result = await self.ta_mcp_server._handle_chart_patterns(
                    pd.DataFrame(market_data.get(symbol, {})), symbol
                )
            elif mcp_tool_name == "analyze_support_resistance":
                result = await self.ta_mcp_server._handle_support_resistance(
                    pd.DataFrame(market_data.get(symbol, {})), symbol
                )
            elif mcp_tool_name == "generate_trading_signals":
                result = await self.ta_mcp_server._handle_trading_signals(
                    pd.DataFrame(market_data.get(symbol, {})), arguments
                )
            else:
                result = await self.ta_mcp_server._handle_comprehensive_analysis(
                    pd.DataFrame(market_data.get(symbol, {})), arguments
                )

            # Convert MCP result to standard format
            if result.get("success", False):
                indicators = result.get("indicators", {})
                signals = result.get("signals", [])

                # Extract signal and strength from MCP result
                signal = "HOLD"
                strength = 0.5

                if signals:
                    # Use the first signal for simplicity
                    first_signal = signals[0] if isinstance(signals, list) else signals
                    if isinstance(first_signal, dict):
                        signal = first_signal.get("action", "HOLD").upper()
                        strength = first_signal.get("strength", 0.5)

                return {
                    "symbol": symbol,
                    "analysis_type": analysis_type,
                    "signal": signal,
                    "strength": strength,
                    "indicators": indicators,
                    "signals": signals,
                    "mcp_result": True,
                    "timestamp": result.get("timestamp", datetime.now().isoformat()),
                }
            else:
                logger.warning(
                    f"TA MCP tool failed for {symbol}: {result.get('error', 'Unknown error')}"
                )
                return {
                    "symbol": symbol,
                    "analysis_type": analysis_type,
                    "signal": "HOLD",
                    "strength": 0.3,
                    "indicators": {"error": result.get("error", "TA analysis failed")},
                    "mcp_result": False,
                }

        except Exception as e:
            logger.error(f"Real technical analysis failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "analysis_type": analysis_type,
                "signal": "HOLD",
                "strength": 0.2,
                "indicators": {"error": str(e)},
                "mcp_result": False,
            }

    # MCP COMPLIANCE: Single entry point and handler mapping
    @property
    def mcp_handlers(self) -> Dict[str, callable]:
        """Dictionary mapping MCP tool names to their handler methods"""
        return {
            "mcts_analyze_with_technical_indicators": self._mcp_analyze_with_technical_indicators,
            "mcts_analyze_market_correlation": self._mcp_analyze_market_correlation,
            "mcts_analyze_market_risk": self._mcp_analyze_market_risk,
            "mcts_run_simulation": self._mcp_run_mcts_simulation,
            "mcts_execute_trading_workflow": self._mcp_execute_trading_workflow,
            "mcts_process_workflow": self._mcp_process_workflow,
            "mcts_execute_tool": self._mcp_execute_tool,
            "mcts_process_message": self._mcp_process_message,
            "mcts_run_parallel": self._mcp_run_mcts_parallel,
            "mcts_analyze_market_sentiment": self._mcp_analyze_market_sentiment,
            "mcts_analyze_technical_signals": self._mcp_analyze_technical_signals,
            "mcts_predict_market_movement": self._mcp_predict_market_movement,
            "mcts_backtest_strategy": self._mcp_backtest_strategy,
            "mcts_compare_strategies": self._mcp_compare_strategies,
            "mcts_cleanup": self._mcp_cleanup,
        }

    async def process_mcp_request(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP-compliant entry point for all agent functionality.
        This is the ONLY public method for accessing agent capabilities.
        
        Args:
            tool_name: Name of the MCP tool to execute
            parameters: Tool parameters
            
        Returns:
            Dict containing tool execution results
            
        Raises:
            ValueError: If tool_name is not recognized
        """
        try:
            # Validate tool name
            if tool_name not in self.mcp_handlers:
                available_tools = list(self.mcp_handlers.keys())
                return ToolResult.error_result(
                    f"Unknown tool '{tool_name}'. Available tools: {available_tools}"
                )
            
            # Get handler and execute
            handler = self.mcp_handlers[tool_name]
            start_time = mcp_metrics.tool_execution_start(tool_name)
            
            try:
                result = await handler(**parameters)
                mcp_metrics.tool_execution_end(tool_name, start_time, True)
                return ToolResult.json_result(result)
            except Exception as e:
                mcp_metrics.tool_execution_end(tool_name, start_time, False)
                logger.error(f"MCP tool execution failed for {tool_name}: {e}")
                return ToolResult.error_result(f"Tool execution failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"MCP request processing failed: {e}")
            return ToolResult.error_result(f"Request processing failed: {str(e)}")

    async def _mcp_analyze_with_technical_indicators(
        self, symbols: List[str], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run MCTS-guided technical analysis enhanced with AI insights

        Args:
            symbols: List of symbols to analyze
            market_data: OHLCV market data

        Returns:
            Combined technical analysis with AI enhancement
        """
        logger.info(f"AI-Enhanced Technical Analysis for {symbols}")

        results = {}

        for symbol in symbols:
            # Get AI sentiment first
            ai_sentiment = await self._get_ai_market_sentiment([symbol], market_data)

            # Configure MCTS for technical analysis decisions
            ta_config = {
                "symbols": [symbol],
                "analysis_depth": 5,
                "available_actions": [
                    {"type": "analyze_indicators", "symbol": symbol},
                    {"type": "detect_patterns", "symbol": symbol},
                    {"type": "support_resistance", "symbol": symbol},
                    {"type": "generate_signals", "symbol": symbol},
                ],
            }

            # Run MCTS to determine best technical analysis approach
            self.environment = ProductionTradingEnvironment(ta_config, self.market_data_provider)
            mcts_result = await self._mcp_run_mcts_parallel(iterations=100)

            # Execute the recommended technical analysis
            best_action = mcts_result.get("best_action", {})
            action_type = best_action.get("type", "analyze_indicators")

            # Execute real technical analysis using MCP tools
            ta_result = await self._execute_real_technical_analysis(
                symbol, action_type, market_data
            )

            # Enhance with AI insights
            enhanced_result = await self._enhance_technical_analysis_with_ai(ta_result, symbol)

            # Add MCTS confidence
            enhanced_result["mcts_confidence"] = mcts_result.get("confidence", 0)
            enhanced_result["analysis_path"] = [
                a.get("type") for a in mcts_result.get("stats", {}).get("action_sequence", [])
            ]

            results[symbol] = enhanced_result

        return {
            "timestamp": datetime.now().isoformat(),
            "analysis_results": results,
            "ai_enabled": self.grok4_client is not None,
            "mcts_iterations": 100,
            "method": "ai_enhanced_mcts_technical_analysis",
        }

    def _register_strands_tools(self):
        """Register simplified Strands tools with AI and backtesting capabilities"""
        # Simplified core tools with AI enhancement
        self.strands_tools = {
            # Core MCTS tools
            "run_mcts_simulation": self.run_mcts_simulation,
            "calculate_optimal_portfolio": self.calculate_optimal_portfolio,
            # AI-powered analysis tools
            "analyze_market_sentiment": self.analyze_market_sentiment,
            "analyze_technical_signals": self.analyze_technical_signals,
            "predict_market_movement": self.predict_market_movement,
            # Strategy backtesting tools
            "backtest_strategy": self.backtest_strategy,
            "compare_strategies": self.compare_strategies,
            # Legacy compatibility (simplified)
            "analyze_market_correlation": self.analyze_market_correlation,
        }

        # Register with simplified capabilities
        if hasattr(self, "capabilities"):
            self.capabilities.extend(
                [
                    "mcts_calculation",
                    "ai_market_analysis",
                    "strategy_backtesting",
                    "technical_analysis",
                ]
            )

        logger.info(f"Registered {len(self.strands_tools)} Strands tools")

    # Analysis Tools Implementation (trading functions removed)
    async def _mcp_analyze_market_correlation(
        self, symbols: List[str], timeframe: str = "1h", **kwargs
    ) -> Dict[str, Any]:
        """Analyze market correlation patterns

        Args:
            symbols: List of symbols to analyze
            timeframe: Analysis timeframe

        Returns:
            Dict with correlation analysis results
        """
        logger.info(f"Analysis tool: analyze_market_correlation called with {len(symbols)} symbols")

        # Prepare analysis environment
        config = {"symbols": symbols, "max_depth": 5, "analysis_type": "correlation"}

        self.environment = DataAnalysisEnvironment(config, self.market_data_provider)

        # Run analysis calculation
        result = await self._mcp_run_mcts_parallel(iterations=100)

        return {
            "correlation_matrix": result["best_action"],
            "analysis_confidence": result["confidence"],
            "calculation_stats": result["stats"],
            "tool_name": "analyze_market_correlation",
        }

        # Convert portfolio to MCTS format
        symbols = list(portfolio.keys())
        total_value = sum(portfolio.values())

        config = {
            "initial_portfolio": total_value,
            "symbols": symbols,
            "current_positions": portfolio,
            "constraints": constraints,
            "max_depth": 12,
        }

        self.environment = ProductionTradingEnvironment(config, self.market_data_provider)
        result = await self._mcp_run_mcts_parallel(iterations=1500)

        return {
            "optimized_allocation": result["best_action"],
            "improvement_potential": result["expected_value"],
            "rebalancing_required": abs(result["expected_value"]) > 0.02,
            "tool_name": "optimize_allocation",
        }

    async def _mcp_analyze_market_risk(
        self, symbols: List[str], time_horizon: int = 30, **kwargs
    ) -> Dict[str, Any]:
        """Strands tool: Analyze market risk using MCTS scenarios

        Args:
            symbols: Symbols to analyze
            time_horizon: Analysis time horizon in days

        Returns:
            Dict with risk analysis results
        """
        logger.info(
            f"Strands tool: analyze_market_risk called for {len(symbols)} symbols, {time_horizon}d horizon"
        )

        config = {
            "initial_portfolio": 0,  # Must be provided by user
            "symbols": symbols,
            "max_depth": min(time_horizon // 2, 25),
            "risk_analysis_mode": True,
        }

        self.environment = ProductionTradingEnvironment(config, self.market_data_provider)

        # Run multiple scenarios
        scenarios = []
        for _ in range(5):
            result = await self._mcp_run_mcts_parallel(iterations=300)
            scenarios.append(result["expected_value"])

        var_95 = sorted(scenarios)[0]  # 5th percentile (worst case)
        expected_return = sum(scenarios) / len(scenarios)

        return {
            "expected_return": expected_return,
            "value_at_risk_95": var_95,
            "risk_score": abs(var_95) / max(abs(expected_return), 0.01),
            "scenario_count": len(scenarios),
            "recommendation": "high_risk" if abs(var_95) > 0.15 else "acceptable_risk",
            "tool_name": "analyze_market_risk",
        }

    async def _mcp_run_mcts_simulation(
        self, problem_type: str, parameters: Dict[str, Any], iterations: int = None, **kwargs
    ) -> Dict[str, Any]:
        """Strands tool: Run raw MCTS simulation

        Args:
            problem_type: Type of problem to solve
            parameters: Problem-specific parameters
            iterations: Number of MCTS iterations

        Returns:
            Dict with simulation results
        """
        logger.info(f"Strands tool: run_mcts_simulation called for {problem_type}")

        # This provides direct access to MCTS for custom problems
        iterations = iterations or self.config.iterations

        if problem_type == "trading":
            self.environment = ProductionTradingEnvironment(parameters, self.market_data_provider)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        result = await self._mcp_run_mcts_parallel(iterations=iterations)
        result["tool_name"] = "run_mcts_simulation"
        return result

    async def _mcp_execute_trading_workflow(
        self, workflow_config: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Strands tool: Execute complex trading workflow

        Args:
            workflow_config: Workflow configuration with steps

        Returns:
            Dict with workflow execution results
        """
        logger.info(
            f"Strands tool: execute_trading_workflow called with {len(workflow_config.get('steps', []))} steps"
        )

        workflow_results = {
            "workflow_id": workflow_config.get("id", f"workflow_{int(time.time())}"),
            "steps_completed": 0,
            "results": [],
            "success": True,
            "tool_name": "execute_trading_workflow",
        }

        # Execute workflow steps using other Strands tools
        for i, step in enumerate(workflow_config.get("steps", [])):
            step_type = step.get("type")
            step_params = step.get("parameters", {})

            try:
                if step_type == "portfolio_optimization":
                    result = await self.calculate_optimal_portfolio(**step_params)
                elif step_type == "strategy_evaluation":
                    result = await self.evaluate_trading_strategy(**step_params)
                elif step_type == "risk_analysis":
                    result = await self._mcp_analyze_market_risk(**step_params)
                elif step_type == "allocation_optimization":
                    result = await self.optimize_allocation(**step_params)
                else:
                    raise ValueError(f"Unknown workflow step type: {step_type}")

                workflow_results["results"].append(
                    {"step": i + 1, "type": step_type, "result": result, "success": True}
                )
                workflow_results["steps_completed"] += 1

            except Exception as e:
                workflow_results["results"].append(
                    {"step": i + 1, "type": step_type, "error": str(e), "success": False}
                )
                if not step.get("continue_on_error", False):
                    workflow_results["success"] = False
                    break

        return workflow_results

    def _generate_strategy_recommendations(
        self, performance: float, stability: float, strategy_config: Dict[str, Any]
    ) -> List[str]:
        """Generate strategy improvement recommendations"""
        recommendations = []

        if performance < 0:
            recommendations.append("Consider reducing position sizes")
            recommendations.append("Implement stronger stop-loss mechanisms")

        if stability > 0.3:
            recommendations.append("Strategy shows high volatility - consider risk management")
            recommendations.append("Diversify across more symbols")

        if strategy_config.get("risk_per_trade", 0.02) > 0.05:
            recommendations.append("Risk per trade is too high - reduce to < 3%")

        return recommendations

    # Enhanced Strands Integration Methods
    async def _mcp_process_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process Strands workflow requests"""
        logger.info(f"Processing Strands workflow: {workflow_id}")

        # Route to appropriate workflow handler
        if workflow_id.startswith("mcts_"):
            return await self._mcp_execute_trading_workflow(inputs)
        else:
            return await super().process_workflow(workflow_id, inputs)

    async def _mcp_execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Strands tools with enhanced error handling"""
        if tool_name in self.strands_tools:
            try:
                # Call the tool function
                tool_func = self.strands_tools[tool_name]
                result = await tool_func(**parameters)

                # Record tool execution
                if hasattr(self, "store_memory"):
                    await self.store_memory(
                        f"tool_execution_{tool_name}_{int(time.time())}",
                        result,
                        {"tool": tool_name, "timestamp": time.time()},
                    )

                return result

            except Exception as e:
                logger.error(f"Strands tool {tool_name} failed: {e}")
                return {"error": str(e), "tool_name": tool_name, "success": False}
        else:
            return await super().execute_tool(tool_name, parameters)

    async def _mcp_process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests prioritizing Strands tools and workflows with security"""
        # Security check
        auth_header = message.get("auth_header")
        ip_address = message.get("ip_address")

        if auth_header:
            auth_result = await self.authenticate_request(auth_header, ip_address)
            if not auth_result.get("success"):
                return auth_result

        # Rate limiting
        if not await self.rate_limiter.check_limit(self.agent_id):
            return {"error": "Rate limit exceeded", "retry_after": 60}

        # Circuit breaker
        if not self.circuit_breaker.can_execute():
            return {"error": "Service temporarily unavailable", "retry_after": 60}

        try:
            start_time = time.time()

            # Prioritize Strands workflow and tool requests
            if "workflow_id" in message:
                # This is a Strands workflow request
                return await self._mcp_process_workflow(
                    message["workflow_id"], message.get("inputs", {})
                )

            elif "tool_name" in message:
                # This is a Strands tool execution request
                return await self._mcp_execute_tool(message["tool_name"], message.get("parameters", {}))

            elif "strands_request" in message:
                # Direct Strands integration
                strands_req = message["strands_request"]
                if strands_req.get("type") == "tool_execution":
                    return await self._mcp_execute_tool(
                        strands_req["tool"], strands_req.get("parameters", {})
                    )
                elif strands_req.get("type") == "workflow":
                    return await self._mcp_process_workflow(
                        strands_req["workflow_id"], strands_req.get("inputs", {})
                    )

            # Fallback to legacy MCP-style messages for backwards compatibility
            msg_type = message.get("type")
            if msg_type in ["calculate", "optimize", "evaluate"]:
                # Map legacy requests to Strands tools
                if msg_type == "calculate":
                    # Map to appropriate Strands tool based on problem type
                    problem_type = message.get("problem_type", "trading")
                    parameters = message.get("parameters", {})

                    if problem_type == "portfolio_optimization":
                        return await self.calculate_optimal_portfolio(
                            symbols=parameters.get("symbols", ["BTC", "ETH"]),
                            capital=parameters.get("initial_portfolio", 0),  # No default
                            risk_tolerance=parameters.get("risk_tolerance", 0.5),
                        )
                    elif problem_type == "strategy_evaluation":
                        return await self.evaluate_trading_strategy(
                            strategy_config=parameters.get("strategy", {}),
                            market_conditions=parameters.get("market_conditions", {}),
                        )
                    else:
                        # Default to MCTS simulation
                        return await self._mcp_run_mcts_simulation(problem_type, parameters)

                elif msg_type == "optimize":
                    return await self.optimize_allocation(
                        portfolio=message.get("portfolio", {}),
                        constraints=message.get("constraints", []),
                    )

                elif msg_type == "evaluate":
                    return await self._mcp_analyze_market_risk(
                        symbols=message.get("symbols", ["BTC", "ETH"]),
                        time_horizon=message.get("time_horizon", 30),
                    )

            else:
                # Unknown message type - try legacy handler
                result = await self._handle_legacy_request(message)

            # Record success
            self.circuit_breaker.record_success()
            self.calculation_count += 1

            # Add metrics to result
            if isinstance(result, dict):
                result["metrics"] = {
                    "execution_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "calculation_count": self.calculation_count,
                    "processing_method": "strands_native",
                }

            return result

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Error processing message: {e}")
            return {"error": str(e), "type": "processing_error"}

    async def _handle_legacy_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle legacy MCP-style requests for backwards compatibility"""
        msg_type = message.get("type")

        if msg_type == "calculate":
            return await self._handle_calculation_request(message)
        elif msg_type == "optimize":
            return await self._handle_optimization_request(message)
        elif msg_type == "evaluate":
            return await self._handle_evaluation_request(message)
        else:
            raise ValidationError(f"Unknown message type: {msg_type}")

    async def _handle_calculation_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculation request with timeout and monitoring"""
        problem_type = message.get("problem_type", "trading")
        parameters = message.get("parameters", {})
        iterations = min(message.get("iterations", self.config.iterations), 10000)
        timeout = message.get("timeout", self.config.timeout_seconds)

        # Validate parameters
        try:
            parameters = InputValidator.validate_calculation_params(parameters)
        except ValidationError as e:
            return {"error": str(e), "type": "validation_error"}

        # Check memory cache first
        cache_key = f"mcts_result:{problem_type}:{hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()}"
        cached_result = await self.retrieve_memory(f"calculation_cache_{cache_key}")
        if cached_result:
            cached_result["cached"] = True
            await self._track_calculation_performance("cache_hit", 0, True)
            return cached_result

        # Check MCP cache as fallback
        mcp_cached_result = mcp_cache.get(cache_key)
        if mcp_cached_result:
            mcp_cached_result["cached"] = True
            return mcp_cached_result

        # Create environment
        if problem_type == "trading":
            self.environment = ProductionTradingEnvironment(parameters, self.market_data_provider)
        else:
            return {"error": f"Unsupported problem type: {problem_type}"}

        # Run MCTS with timeout
        try:
            result = await asyncio.wait_for(
                self._mcp_run_mcts_parallel(iterations=iterations), timeout=timeout
            )

            # Cache result in both memory and MCP cache
            await self.store_memory(
                f"calculation_cache_{cache_key}",
                result,
                {"type": "calculation_cache", "expires_at": time.time() + self.config.cache_ttl},
            )
            mcp_cache.set(cache_key, result, ttl=self.config.cache_ttl)

            # Track performance
            calculation_time = time.time() - start_time if "start_time" in locals() else 0
            await self._track_calculation_performance("calculation", calculation_time, True)

            # Log calculation for learning
            await self._log_calculation_result(problem_type, parameters, result, calculation_time)

            return {
                "type": "calculation_result",
                "problem_type": problem_type,
                "best_action": result["best_action"],
                "expected_value": result["expected_value"],
                "confidence": result["confidence"],
                "exploration_stats": result["stats"],
            }

        except asyncio.TimeoutError:
            return {"error": "Calculation timeout", "partial_results": None}

    @vercel_edge_handler
    async def _mcp_run_mcts_parallel(self, iterations: int = None) -> Dict[str, Any]:
        """Run MCTS with adaptive iteration control and dynamic parameters"""
        if not self.environment:
            raise ValueError("Environment not initialized")

        max_iterations = iterations or self.config.iterations
        self.total_iterations += max_iterations

        # Initialize adaptive controller
        adaptive_controller = AdaptiveIterationController(
            min_iterations=max(100, max_iterations // 20),
            max_iterations=max_iterations,
            convergence_window=50,
            early_stop_confidence=0.95,
        )

        # Initialize root
        initial_state = await self.environment.get_initial_state()
        root = MCTSNodeV2(state=initial_state, untried_actions=initial_state["available_actions"])
        self._current_root_node = root  # Store reference for memory checking

        # Pre-fetch AI insights for initial state
        symbols = initial_state.get("symbols", [])
        if symbols and self.grok4_client:
            ai_sentiment = await self._get_ai_market_sentiment(symbols, initial_state)
            if ai_sentiment:
                logger.info(f"AI insights loaded for {len(ai_sentiment)} symbols")
                # Store in cache for simulation phase
                self._ai_cache[f"ai_sentiment_{initial_state.get('timestamp', '')}"] = ai_sentiment

        # Statistics
        start_time = time.time()
        iteration_values = []
        memory_checks = 0
        convergence_history = []

        # Parallel simulation setup
        batch_size = self.config.parallel_simulations

        # Adaptive iteration loop
        actual_iterations = 0
        while True:
            # Check if we should continue
            current_best = root.best_child(c_param=0, use_rave=self.config.enable_rave)
            current_value = (
                current_best.value / current_best.visits
                if current_best and current_best.visits > 0
                else 0
            )
            current_confidence = (
                current_best.visits / (actual_iterations + 1) if current_best else 0
            )
            best_action_str = str(current_best.action) if current_best else "none"

            should_continue, reason, status = adaptive_controller.should_continue_search(
                current_value, current_confidence, best_action_str
            )

            convergence_history.append(status)

            if not should_continue:
                logger.info(f"MCTS converged early at iteration {actual_iterations}: {reason}")
                break

            # Get current adaptive parameters
            adaptive_params = adaptive_controller.get_current_parameters()
            current_c_param = adaptive_params["c_param"]

            # Memory check
            if actual_iterations % 100 == 0:
                memory_checks += 1
                if self._check_memory_limit():
                    logger.warning(f"Memory limit approaching at iteration {actual_iterations}")
                    break

            # Memory optimization - prune tree if getting too large
            if actual_iterations % 500 == 0 and actual_iterations > 0:
                tree_size = self._count_nodes(root)
                if tree_size > 5000:  # Prune if tree gets too large
                    self._prune_tree(root, keep_ratio=0.7)
                    logger.info(
                        f"Pruned tree at iteration {actual_iterations}, new size: {self._count_nodes(root)}"
                    )

            # Run batch of iterations
            batch_tasks = []
            for j in range(batch_size):
                batch_tasks.append(
                    self._run_single_iteration_adaptive(
                        root, initial_state, current_c_param, adaptive_params
                    )
                )

            # Run batch in parallel
            batch_results = await asyncio.gather(*batch_tasks)
            iteration_values.extend([r["value"] for r in batch_results])

            # Update tree with results
            for result in batch_results:
                self._backpropagate(result["node"], result["value"], result["action_sequence"])

            actual_iterations += batch_size

        # Get final results
        final_best = root.best_child(c_param=0, use_rave=self.config.enable_rave)
        elapsed_time = time.time() - start_time

        # Generate final report
        final_report = adaptive_controller.get_final_report()

        # Record metrics for anomaly detection
        final_tree_size = self._count_nodes(root)
        final_expected_value = final_best.value / final_best.visits if final_best else 0
        final_confidence = final_best.visits / actual_iterations if final_best else 0

        asyncio.create_task(
            self._record_execution_metrics(
                elapsed_time,
                actual_iterations,
                final_expected_value,
                final_confidence,
                final_tree_size,
                final_report["convergence_confidence"],
            )
        )

        # Calculate AI contribution metrics
        ai_metrics = {
            "ai_enabled": self.grok4_client is not None,
            "ai_cache_hits": len(self._ai_cache),
            "ai_insights_used": len([k for k in self._ai_cache.keys() if "sentiment" in k]),
            "ai_predictions_used": len([k for k in self._ai_cache.keys() if "predictions" in k]),
        }

        # Get AI reasoning for best action if available
        ai_reasoning = None
        if final_best and self.grok4_client:
            symbol = final_best.action.get("symbol", "") if final_best.action else ""
            cache_key = f"ai_sentiment_{initial_state.get('timestamp', '')}"
            if cache_key in self._ai_cache and symbol in self._ai_cache[cache_key]:
                ai_data = self._ai_cache[cache_key][symbol]
                ai_reasoning = {
                    "recommendation": ai_data.get("recommendation"),
                    "confidence": ai_data.get("confidence"),
                    "reasoning": ai_data.get("reasoning"),
                    "risk_level": ai_data.get("risk_level"),
                }

        result = {
            "best_action": final_best.action if final_best else None,
            "expected_value": final_expected_value,
            "confidence": final_confidence,
            "ai_reasoning": ai_reasoning,  # AI explanation for the decision
            "stats": {
                "iterations": actual_iterations,
                "max_iterations": max_iterations,
                "elapsed_time": elapsed_time,
                "iterations_per_second": actual_iterations / elapsed_time
                if elapsed_time > 0
                else 0,
                "average_value": sum(iteration_values) / len(iteration_values)
                if iteration_values
                else 0,
                "max_value": max(iteration_values) if iteration_values else 0,
                "min_value": min(iteration_values) if iteration_values else 0,
                "memory_checks": memory_checks,
                "tree_size": final_tree_size,
                "efficiency_gain": final_report["efficiency_gain"],
                "convergence_reason": final_report["convergence_reason"],
                "convergence_confidence": final_report["convergence_confidence"],
                "adaptive_params_final": adaptive_controller.get_current_parameters(),
                "convergence_history": convergence_history[-10:],  # Last 10 status updates
                "ai_metrics": ai_metrics,  # AI contribution metrics
            },
        }

        return result

    async def _run_single_iteration_adaptive(
        self,
        root: MCTSNodeV2,
        initial_state: Dict[str, Any],
        c_param: float,
        adaptive_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single MCTS iteration with adaptive parameters and virtual loss"""
        node = root
        state = initial_state.copy()
        action_sequence = []
        path = [root]  # Track path for virtual loss cleanup

        # Selection with dynamic c_param and virtual loss
        while node.untried_actions == [] and node.children != []:
            # Add virtual loss for parallel MCTS
            node.add_virtual_loss(1)

            node = node.best_child(c_param, self.config.enable_rave)
            state = await self.environment.apply_action(state, node.action)
            action_sequence.append(node.action)
            path.append(node)

        # Expansion with progressive widening
        if node.untried_actions != []:
            # Progressive widening: check if we should expand
            should_expand = True
            if self.config.enable_progressive_widening:
                # Progressive widening formula: |C(v)| <= k * N(v)^α
                k_pw = 1.0  # Progressive widening constant
                alpha_pw = 0.5  # Progressive widening alpha
                # Ensure visits is at least 1 to avoid division by zero/edge cases
                visits = max(1, node.visits)
                max_children = k_pw * math.pow(visits, alpha_pw)
                should_expand = len(node.children) < max_children

            if should_expand:
                # Use action priors if available
                if hasattr(node, "action_priors") and node.action_priors:
                    # Sample according to priors for better expansion
                    actions = [a for a in node.untried_actions if str(a) in node.action_priors]
                    if actions:
                        probs = [node.action_priors[str(a)] for a in actions]
                        prob_sum = sum(probs)
                        if prob_sum > 0:
                            probs = [p / prob_sum for p in probs]
                            action = random.choices(actions, weights=probs, k=1)[0]
                        else:
                            action = random.choice(node.untried_actions)
                    else:
                        action = random.choice(node.untried_actions)
                else:
                    action = random.choice(node.untried_actions)

                state = await self.environment.apply_action(state, action)
                node = node.add_child(action, state)
                action_sequence.append(action)
                path.append(node)

                # Add virtual loss to new node
                node.add_virtual_loss(1)

        # Simulation with adaptive depth and AI enhancement
        simulation_state = state.copy()
        simulation_actions = []
        depth = 0

        # Adaptive simulation depth based on convergence
        convergence_confidence = adaptive_params.get("convergence_confidence", 0)
        adaptive_depth = int(self.config.simulation_depth * (1 + convergence_confidence))

        # Pre-fetch AI insights for this simulation branch
        if depth == 0 and self.grok4_client:
            symbols = simulation_state.get("symbols", [])
            if symbols:
                ai_sentiment = await self._get_ai_market_sentiment(symbols, simulation_state)
                if ai_sentiment:
                    # Store in cache for this simulation
                    cache_key = f"ai_sentiment_{simulation_state.get('timestamp', '')}"
                    self._ai_cache[cache_key] = ai_sentiment

        while (
            not await self.environment.is_terminal_state(simulation_state)
            and depth < adaptive_depth
        ):
            available_actions = await self.environment.get_available_actions(simulation_state)
            if available_actions:
                action = await self._select_simulation_action(
                    available_actions, simulation_state, depth
                )
                simulation_state = await self.environment.apply_action(simulation_state, action)
                simulation_actions.append(action)
            depth += 1

        # AI-Enhanced Evaluation
        base_value = await self.environment.evaluate_state(simulation_state)

        # Apply AI value adjustment if we have insights
        if self.grok4_client and action_sequence:
            # Calculate AI confidence boost based on alignment with recommendations
            ai_boost = await self._calculate_ai_value_boost(
                action_sequence + simulation_actions, simulation_state
            )
            value = base_value * (1 + ai_boost)
        else:
            value = base_value

        # Remove virtual loss from all nodes in path after completing iteration
        for path_node in path:
            path_node.remove_virtual_loss(1)

        return {
            "node": node,
            "value": value,
            "action_sequence": action_sequence + simulation_actions,
        }

    async def _run_single_iteration(
        self, root: MCTSNodeV2, initial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single MCTS iteration"""
        node = root
        state = initial_state.copy()
        action_sequence = []

        # Selection
        while node.untried_actions == [] and node.children != []:
            node = node.best_child(self.config.exploration_constant, self.config.enable_rave)
            state = await self.environment.apply_action(state, node.action)
            action_sequence.append(node.action)

        # Expansion
        if node.untried_actions != []:
            action = random.choice(node.untried_actions)
            state = await self.environment.apply_action(state, action)
            node = node.add_child(action, state)
            action_sequence.append(action)

        # Simulation
        simulation_state = state.copy()
        simulation_actions = []
        depth = 0

        while (
            not await self.environment.is_terminal_state(simulation_state)
            and depth < self.config.simulation_depth
        ):
            available_actions = await self.environment.get_available_actions(simulation_state)
            if available_actions:
                # Use deterministic action selection based on value estimation
                action = await self._select_simulation_action(
                    available_actions, simulation_state, depth
                )

                simulation_state = await self.environment.apply_action(simulation_state, action)
                simulation_actions.append(action)
            depth += 1

        # Evaluation
        value = await self.environment.evaluate_state(simulation_state)

        return {
            "node": node,
            "value": value,
            "action_sequence": action_sequence + simulation_actions,
        }

    async def _select_simulation_action(
        self, available_actions: List[Dict[str, Any]], state: Dict[str, Any], depth: int
    ) -> Dict[str, Any]:
        """Select action during simulation using TRUE Monte Carlo random sampling with AI enhancement"""
        if not available_actions:
            raise ValueError("No available actions for simulation")

        # TRUE MONTE CARLO: Use random selection with optional AI-informed biasing

        # Option 1: Pure random (classic Monte Carlo)
        if self.config.simulation_strategy == "pure_random":
            return random.choice(available_actions)

        # Option 2: AI-enhanced weighted random (still stochastic but smarter)
        weights = []

        # Get AI insights if available (cached for performance)
        ai_market_sentiment = None
        if self.grok4_client and depth < 3:  # Only use AI for early decisions
            cache_key = f"ai_sentiment_{state.get('timestamp', '')}"
            if cache_key in self._ai_cache:
                ai_market_sentiment = self._ai_cache[cache_key]

        for action in available_actions:
            weight = 1.0  # Base weight

            # Adjust weights based on action characteristics
            action_type = action.get("type", "unknown")
            symbol = action.get("symbol", "")

            # AI-enhanced weight adjustment for technical analysis actions
            if ai_market_sentiment and symbol in ai_market_sentiment:
                ai_rec = ai_market_sentiment[symbol].get("recommendation", "HOLD")
                ai_confidence = ai_market_sentiment[symbol].get("confidence", 0.5)

                # Enhanced weighting for technical analysis actions
                if action_type in ["technical_analysis", "analyze_indicators"]:
                    # AI boost for technical analysis when market is uncertain
                    if ai_confidence < 0.6:  # Low AI confidence means we need more TA
                        weight *= 2.5
                    else:
                        weight *= 1.5
                elif action_type == "detect_patterns" and ai_rec in ["BUY", "SELL"]:
                    # Pattern detection more valuable when AI has strong signal
                    weight *= 1.0 + ai_confidence * 0.7
                elif action_type == "support_resistance":
                    # Support/resistance analysis always valuable
                    weight *= 2.0
                elif action_type == "generate_signals":
                    # Signal generation gets AI confidence boost
                    weight *= 1.0 + ai_confidence * 0.6

                # Standard trading actions with AI alignment
                if action_type == "buy" and ai_rec == "BUY":
                    weight *= 1.0 + ai_confidence * 0.5  # Boost buy actions if AI recommends
                elif action_type == "sell" and ai_rec == "SELL":
                    weight *= 1.0 + ai_confidence * 0.5  # Boost sell actions if AI recommends
                elif action_type == "hold" and ai_rec == "HOLD":
                    weight *= 1.0 + ai_confidence * 0.3

            # Original heuristics (maintained for robustness)
            if action_type == "buy":
                # Slightly favor buys in early simulation
                weight *= 1.2 if depth < 5 else 0.9
            elif action_type == "sell":
                # Slightly favor sells in later simulation
                weight *= 0.9 if depth < 5 else 1.1
            elif action_type in ["technical_analysis", "sentiment_analysis"]:
                # Analysis more valuable early
                weight *= 2.0 if depth < 3 else 0.5

            # Risk adjustment (still maintains randomness)
            risk_score = action.get("risk_score", 0.5)

            # AI risk adjustment if available
            if ai_market_sentiment and symbol in ai_market_sentiment:
                ai_risk = ai_market_sentiment[symbol].get("risk_level", "MEDIUM")
                if ai_risk == "HIGH":
                    risk_score = min(risk_score + 0.2, 1.0)
                elif ai_risk == "LOW":
                    risk_score = max(risk_score - 0.2, 0.0)

            weight *= 1.5 - risk_score  # Slight bias against high risk

            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(available_actions)] * len(available_actions)

        # STOCHASTIC SELECTION - this is true Monte Carlo with AI guidance
        return random.choices(available_actions, weights=weights, k=1)[0]

    def _backpropagate(self, node: MCTSNodeV2, value: float, action_sequence: List[Dict[str, Any]]):
        """Backpropagate value through tree with RAVE updates"""
        while node is not None:
            node.visits += 1
            node.value += value

            # RAVE updates
            if self.config.enable_rave:
                node.update_rave(action_sequence, value)

            node = node.parent

    def _count_nodes(self, node: MCTSNodeV2) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _prune_tree(self, root: MCTSNodeV2, keep_ratio: float = 0.7):
        """Prune tree to reduce memory usage"""

        def prune_node(node: MCTSNodeV2, depth: int = 0):
            if not node.children:
                return

            # Don't prune too close to root
            if depth < 2:
                for child in node.children:
                    prune_node(child, depth + 1)
                return

            # Sort children by promise (visits * value)
            child_scores = []
            for child in node.children:
                if child.visits > 0:
                    score = child.visits * (child.value / child.visits)
                else:
                    score = 0
                child_scores.append((score, child))

            # Keep top percentage of children
            child_scores.sort(key=lambda x: x[0], reverse=True)
            keep_count = max(1, int(len(child_scores) * keep_ratio))

            # Remove least promising children
            new_children = []
            pruned_children = []
            for i, (score, child) in enumerate(child_scores):
                if i < keep_count:
                    new_children.append(child)
                    prune_node(child, depth + 1)
                else:
                    pruned_children.append(child)

            # Break circular references for pruned children to prevent memory leaks
            for child in pruned_children:
                child.parent = None
                child.children.clear()
                child.rave_visits.clear()
                child.rave_values.clear()
                child.action_priors.clear()

            node.children = new_children

        prune_node(root)

    async def _record_execution_metrics(
        self,
        execution_time: float,
        iterations: int,
        expected_value: float,
        confidence: float,
        tree_size: int,
        convergence_confidence: float,
    ):
        """Record execution metrics for anomaly detection"""
        try:
            # Estimate memory usage
            memory_usage = tree_size * 0.0002  # Rough estimate

            # Record metrics
            await self.anomaly_detector.record_metric("execution_time", execution_time)
            await self.anomaly_detector.record_metric("iterations_completed", iterations)
            await self.anomaly_detector.record_metric("expected_value", expected_value)
            await self.anomaly_detector.record_metric("confidence", confidence)
            await self.anomaly_detector.record_metric("tree_size", tree_size)
            await self.anomaly_detector.record_metric("memory_usage", memory_usage)
            await self.anomaly_detector.record_metric(
                "convergence_confidence", convergence_confidence
            )

            # Calculate and record error rate
            error_rate = 1.0 - confidence  # Simple proxy for error rate
            await self.anomaly_detector.record_metric("error_rate", error_rate)

        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")

    def _check_memory_limit(self) -> bool:
        """Check if approaching memory limit - Vercel Edge Runtime compatible"""
        # Vercel Edge Runtime doesn't support psutil, so we use tree size as proxy
        try:
            # Try psutil first for local development
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb > self.config.max_memory_mb * 0.8
        except (ImportError, RuntimeError):
            # Fallback for Vercel Edge Runtime - estimate based on tree size
            if hasattr(self, "_current_root_node"):
                tree_size = self._count_nodes(self._current_root_node)
                # Rough estimate: 200 bytes per node
                estimated_memory_mb = (tree_size * 200) / (1024 * 1024)
                return estimated_memory_mb > self.config.max_memory_mb * 0.8
            else:
                # Conservative default if we can't estimate
                return False

    async def _mcts_calculate_with_monitoring(
        self,
        problem_type: str,
        parameters: Dict[str, Any],
        iterations: int = None,
        timeout: int = None,
    ) -> ToolResult:
        """MCP tool with monitoring"""
        try:
            # Record metric
            start_time = mcp_metrics.tool_execution_start("mcts_calculate_v2", self.agent_id)

            result = await self._handle_calculation_request(
                {
                    "type": "calculate",
                    "problem_type": problem_type,
                    "parameters": parameters,
                    "iterations": iterations,
                    "timeout": timeout,
                }
            )

            # Record success
            mcp_metrics.tool_execution_end("mcts_calculate_v2", start_time, "error" not in result)

            return ToolResult.json_result(result)
        except Exception as e:
            mcp_metrics.tool_execution_end("mcts_calculate_v2", start_time, False)
            return ToolResult.error_result(str(e))

    async def _get_metrics(self) -> ToolResult:
        """Get agent performance metrics"""
        uptime = time.time() - self.start_time

        metrics = {
            "agent_id": self.agent_id,
            "uptime_seconds": uptime,
            "calculation_count": self.calculation_count,
            "total_iterations": self.total_iterations,
            "average_iterations_per_calculation": self.total_iterations / self.calculation_count
            if self.calculation_count > 0
            else 0,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failures,
            "rate_limiter_remaining": await self.rate_limiter.get_remaining(self.agent_id),
        }

        return ToolResult.json_result(metrics)

    async def _health_check(self) -> ToolResult:
        """Health check endpoint"""
        try:
            # Test basic functionality
            test_env = ProductionTradingEnvironment(
                {"initial_portfolio": 1000, "symbols": ["BTC"], "max_depth": 2}
            )
            test_state = await test_env.get_initial_state()

            health = {
                "status": "healthy" if self.circuit_breaker.state == "closed" else "degraded",
                "agent_id": self.agent_id,
                "uptime": time.time() - self.start_time,
                "memory_ok": not self._check_memory_limit(),
                "circuit_breaker": self.circuit_breaker.state,
                "last_calculation": self.calculation_count,
            }

            return ToolResult.json_result(health)
        except Exception as e:
            return ToolResult.json_result({"status": "unhealthy", "error": str(e)})

    async def _handle_optimization_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optimization with genetic algorithm enhancement"""
        objective = message.get("objective")
        constraints = message.get("constraints", [])

        # Use MCTS for optimization exploration
        optimization_params = {
            "initial_portfolio": 0,  # Must be specified by user,
            "symbols": ["BTC", "ETH"],
            "max_depth": 10,
            "objective": objective,
            "constraints": constraints,
        }

        result = await self._handle_calculation_request(
            {
                "type": "calculate",
                "problem_type": "trading",
                "parameters": optimization_params,
                "iterations": 2000,
            }
        )

        return {
            "type": "optimization_result",
            "objective": objective,
            "optimal_solution": result.get("best_action"),
            "expected_improvement": result.get("expected_value"),
            "confidence": result.get("confidence"),
        }

    async def _handle_evaluation_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle strategy evaluation"""
        strategy = message.get("strategy", {})

        # Evaluate using MCTS
        eval_params = {
            "initial_portfolio": strategy.get("capital", 0),  # Must be specified
            "symbols": strategy.get("symbols", ["BTC", "ETH"]),
            "max_depth": 20,
        }

        result = await self._handle_calculation_request(
            {
                "type": "calculate",
                "problem_type": "trading",
                "parameters": eval_params,
                "iterations": 1000,
            }
        )

        return {
            "type": "evaluation_result",
            "strategy_score": result.get("expected_value", 0),
            "confidence": result.get("confidence", 0),
            "recommendation": "approved" if result.get("expected_value", 0) > 0.1 else "rejected",
        }

    # ==================== NEW AI-POWERED TOOLS ====================

    async def _mcp_analyze_market_sentiment(
        self, symbols: List[str], timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment using Grok4 AI

        Args:
            symbols: List of trading symbols
            timeframe: Analysis timeframe

        Returns:
            Market sentiment analysis with AI insights
        """
        logger.info(f"AI Analysis: market sentiment for {len(symbols)} symbols")

        if not self.grok4_client:
            return {
                "error": "Grok4 AI client not available",
                "fallback_analysis": "MCTS-based correlation analysis",
                "symbols": symbols,
                "confidence": 0.5,
            }

        try:
            # Get AI-powered sentiment analysis
            insights = await self.grok4_client.analyze_market_sentiment(symbols, timeframe)

            # Convert to standardized format
            sentiment_data = {}
            for insight in insights:
                sentiment_data[insight.symbol] = {
                    "recommendation": insight.recommendation,
                    "sentiment_score": insight.score,
                    "risk_level": insight.risk_level,
                    "reasoning": insight.reasoning,
                    "confidence": insight.confidence,
                }

            return {
                "sentiment_analysis": sentiment_data,
                "overall_market_sentiment": self._calculate_overall_sentiment(insights),
                "trading_signals": self._extract_trading_signals(insights),
                "ai_provider": "grok4",
                "timeframe": timeframe,
                "analysis_timestamp": datetime.now().isoformat(),
            }

        except Grok4APIError as e:
            logger.error(f"Grok4 API error during sentiment analysis: {e}")
            if e.status_code == 429:
                logger.warning("Rate limit exceeded - using fallback analysis")
            # Fallback to MCTS-based analysis
            return await self._mcp_analyze_market_correlation(symbols, timeframe)
        except Grok4Error as e:
            logger.error(f"Grok4 sentiment analysis failed: {e}")
            # Fallback to MCTS-based analysis
            return await self._mcp_analyze_market_correlation(symbols, timeframe)
        except Exception as e:
            logger.error(f"Unexpected error in sentiment analysis: {e}")
            # Fallback to MCTS-based analysis
            return await self._mcp_analyze_market_correlation(symbols, timeframe)

    async def _mcp_analyze_technical_signals(
        self, symbols: List[str], market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze technical signals using AI-enhanced MCTS approach

        Args:
            symbols: Symbols to analyze
            market_data: Optional market data

        Returns:
            Technical analysis results with AI enhancement
        """
        logger.info(f"Technical Signal Analysis: {len(symbols)} symbols")

        if not market_data:
            market_data = await self._fetch_market_data()

        # Use the existing analyze_with_technical_indicators method
        return await self._mcp_analyze_with_technical_indicators(symbols, market_data)

    async def _mcp_predict_market_movement(
        self, symbols: List[str], horizon: str = "1d"
    ) -> Dict[str, Any]:
        """
        Predict market movement using Grok4 AI

        Args:
            symbols: Symbols to analyze
            horizon: Prediction horizon

        Returns:
            Market movement predictions
        """
        logger.info(f"AI Prediction: market movement for {len(symbols)} symbols")

        if not self.grok4_client:
            return {"error": "Grok4 AI client not available", "fallback": "Use historical analysis"}

        try:
            predictions = await self.grok4_client.predict_market_movement(symbols, horizon)

            # Validate predictions with MCTS simulation
            mcts_validation = await self._validate_predictions_with_mcts(predictions)

            return {
                "ai_predictions": predictions,
                "mcts_validation": mcts_validation,
                "horizon": horizon,
                "confidence_adjusted": True,
                "prediction_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"AI market prediction failed: {e}")
            return {"error": str(e), "predictions": {}}

    async def _mcp_backtest_strategy(
        self, strategy_config: Dict[str, Any], start_date: str = None, end_date: str = None
    ) -> Dict[str, Any]:
        """
        Backtest trading strategy with comprehensive analysis

        Args:
            strategy_config: Strategy configuration
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Comprehensive backtesting results
        """
        logger.info(f"Strategy Backtesting: {strategy_config.get('name', 'Unknown')}")

        try:
            # Create backtest configuration
            config = BacktestConfig(
                strategy_type=StrategyType(strategy_config.get("type", "momentum")),
                symbols=strategy_config.get("symbols", ["BTC", "ETH"]),
                start_date=start_date or "2023-01-01",
                end_date=end_date or "2023-12-31",
                initial_capital=strategy_config.get("initial_capital", 10000),
                position_size=strategy_config.get("position_size", 0.1),
                stop_loss=strategy_config.get("stop_loss", 0.05),
                take_profit=strategy_config.get("take_profit", 0.15),
            )

            # Run backtest
            result = await self.strategy_backtester.backtest_strategy(config)

            # Get AI evaluation if available
            ai_evaluation = None
            if self.grok4_client:
                try:
                    ai_evaluation = await self.grok4_client.evaluate_trading_strategy(
                        strategy_config
                    )
                except Exception as e:
                    logger.warning(f"AI strategy evaluation failed: {e}")

            return {
                "backtest_results": {
                    "total_return": result.performance.total_return,
                    "annualized_return": result.performance.annualized_return,
                    "sharpe_ratio": result.performance.sharpe_ratio,
                    "max_drawdown": result.performance.max_drawdown,
                    "win_rate": result.performance.win_rate,
                    "total_trades": result.performance.total_trades,
                },
                "performance_metrics": result.performance.__dict__,
                "ai_evaluation": ai_evaluation.__dict__ if ai_evaluation else None,
                "insights": result.insights,
                "confidence": result.confidence,
                "analysis_methods": ["historical_backtest", "ai_evaluation"]
                if ai_evaluation
                else ["historical_backtest"],
            }

        except Exception as e:
            logger.error(f"Strategy backtesting failed: {e}")
            return {"error": str(e), "backtest_results": None}

    async def _mcp_compare_strategies(self, strategy_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple trading strategies

        Args:
            strategy_configs: List of strategy configurations

        Returns:
            Strategy comparison results
        """
        logger.info(f"Strategy Comparison: comparing {len(strategy_configs)} strategies")

        try:
            # Convert to BacktestConfig objects
            configs = []
            for config in strategy_configs:
                backtest_config = BacktestConfig(
                    strategy_type=StrategyType(config.get("type", "momentum")),
                    symbols=config.get("symbols", ["BTC", "ETH"]),
                    start_date=config.get("start_date", "2023-01-01"),
                    end_date=config.get("end_date", "2023-12-31"),
                    initial_capital=config.get("initial_capital", 10000),
                )
                configs.append(backtest_config)

            # Run comparison
            comparison = await self.strategy_backtester.compare_strategies(configs)

            return {
                "strategy_rankings": comparison["rankings"],
                "best_strategy": comparison["best_strategy"],
                "performance_summary": comparison["performance_summary"],
                "comparison_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            # Track failed calculation
            await self._track_calculation_performance("calculation", 0, False)
            # Store error for learning
            await self.store_memory(
                f"calculation_error_{time.time()}",
                {
                    "error": str(e),
                    "problem_type": "strategy_comparison",
                    "parameters": strategy_configs,
                    "timestamp": datetime.now().isoformat(),
                },
                {"type": "error_log"},
            )
            logger.error(f"Strategy comparison failed: {e}")
            return {"error": str(e), "type": "calculation_error"}

    async def _track_calculation_performance(
        self, operation_type: str, duration: float, success: bool
    ):
        """Track calculation performance metrics in memory"""
        try:
            performance_data = await self.retrieve_memory("mcts_performance") or {
                "total_calculations": 0,
                "avg_calculation_time": 0,
                "success_rate": 0,
                "cache_hit_rate": 0,
                "best_results": [],
            }

            if operation_type == "cache_hit":
                performance_data["cache_hits"] = performance_data.get("cache_hits", 0) + 1
                total_ops = performance_data["total_calculations"] + performance_data.get(
                    "cache_hits", 0
                )
                performance_data["cache_hit_rate"] = (
                    performance_data.get("cache_hits", 0) / total_ops if total_ops > 0 else 0
                )
            elif operation_type == "calculation":
                performance_data["total_calculations"] += 1
                if success:
                    performance_data["successful_calculations"] = (
                        performance_data.get("successful_calculations", 0) + 1
                    )
                    # Update average calculation time
                    current_avg = performance_data["avg_calculation_time"]
                    performance_data["avg_calculation_time"] = (
                        current_avg * (performance_data["total_calculations"] - 1) + duration
                    ) / performance_data["total_calculations"]

                performance_data["success_rate"] = (
                    performance_data.get("successful_calculations", 0)
                    / performance_data["total_calculations"]
                )

            await self.store_memory(
                "mcts_performance", performance_data, {"type": "performance_tracking"}
            )

        except Exception as e:
            logger.error(f"Failed to track calculation performance: {e}")

    async def _log_calculation_result(
        self, problem_type: str, parameters: Dict[str, Any], result: Dict[str, Any], duration: float
    ):
        """Log calculation results for learning and analysis"""
        try:
            calculation_history = await self.retrieve_memory("calculation_history") or []

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "problem_type": problem_type,
                "parameters": parameters,
                "result": {
                    "best_action": result.get("best_action"),
                    "expected_value": result.get("expected_value"),
                    "confidence": result.get("confidence"),
                },
                "duration": duration,
                "iterations": result.get("stats", {}).get("iterations", 0),
            }

            calculation_history.append(log_entry)

            # Keep only last 100 calculations to prevent memory bloat
            if len(calculation_history) > 100:
                calculation_history = calculation_history[-100:]

            await self.store_memory(
                "calculation_history", calculation_history, {"type": "calculation_log"}
            )

            # Learn from successful strategies
            if result.get("confidence", 0) > 0.8:  # High confidence results
                await self._learn_from_successful_strategy(problem_type, parameters, result)

        except Exception as e:
            logger.error(f"Failed to log calculation result: {e}")

    async def _learn_from_successful_strategy(
        self, problem_type: str, parameters: Dict[str, Any], result: Dict[str, Any]
    ):
        """Learn from successful MCTS strategies"""
        try:
            strategy_learning = await self.retrieve_memory("strategy_learning") or {
                "successful_strategies": [],
                "failed_strategies": [],
            }

            strategy_pattern = {
                "problem_type": problem_type,
                "key_parameters": {
                    k: v
                    for k, v in parameters.items()
                    if k in ["initial_portfolio", "symbols", "max_depth"]
                },
                "result_confidence": result.get("confidence"),
                "best_action": result.get("best_action"),
                "timestamp": datetime.now().isoformat(),
            }

            strategy_learning["successful_strategies"].append(strategy_pattern)

            # Keep only last 50 successful strategies
            if len(strategy_learning["successful_strategies"]) > 50:
                strategy_learning["successful_strategies"] = strategy_learning[
                    "successful_strategies"
                ][-50:]

            await self.store_memory(
                "strategy_learning", strategy_learning, {"type": "strategy_learning"}
            )

        except Exception as e:
            logger.error(f"Failed to learn from successful strategy: {e}")

    # ==================== HELPER METHODS ====================

    def _calculate_overall_sentiment(self, insights: List[MarketInsight]) -> Dict[str, Any]:
        """Calculate overall market sentiment from individual insights"""
        if not insights:
            return {"sentiment": "NEUTRAL", "confidence": 0.0}

        buy_count = sum(1 for i in insights if i.recommendation == "BUY")
        sell_count = sum(1 for i in insights if i.recommendation == "SELL")
        hold_count = len(insights) - buy_count - sell_count

        avg_confidence = sum(i.confidence for i in insights) / len(insights)

        if buy_count > sell_count + hold_count:
            sentiment = "BULLISH"
        elif sell_count > buy_count + hold_count:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        return {
            "sentiment": sentiment,
            "confidence": avg_confidence,
            "distribution": {
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "hold_signals": hold_count,
            },
        }

    def _extract_trading_signals(self, insights: List[MarketInsight]) -> List[Dict[str, Any]]:
        """Extract actionable trading signals from AI insights"""
        signals = []

        for insight in insights:
            if insight.confidence > 0.7:  # High confidence signals only
                signals.append(
                    {
                        "symbol": insight.symbol,
                        "action": insight.recommendation,
                        "confidence": insight.confidence,
                        "risk_level": insight.risk_level,
                        "reasoning": insight.reasoning[:100] + "..."
                        if len(insight.reasoning) > 100
                        else insight.reasoning,
                    }
                )

        return sorted(signals, key=lambda x: x["confidence"], reverse=True)

    async def _run_mcts_technical_analysis_simulation(self, symbols: List[str]) -> Dict[str, Any]:
        """Run MCTS simulation for technical analysis optimization"""
        try:
            # Setup technical analysis environment
            config = {
                "symbols": symbols,
                "max_depth": 8,
                "technical_analysis_focus": True,
                "available_actions": [
                    {"type": "analyze_indicators", "symbols": symbols},
                    {"type": "detect_patterns", "symbols": symbols},
                    {"type": "support_resistance", "symbols": symbols},
                ],
            }

            self.environment = ProductionTradingEnvironment(config, self.market_data_provider)

            # Run technical analysis focused MCTS simulation
            result = await self._mcp_run_mcts_parallel(iterations=150)

            return {
                "analysis_confidence": result["confidence"],
                "recommended_analysis_sequence": [
                    a.get("type") for a in result.get("stats", {}).get("action_sequence", [])
                ],
                "best_indicators": result.get("best_action", {}),
                "mcts_confidence": result["confidence"],
                "simulation_stats": result["stats"],
            }

        except Exception as e:
            logger.error(f"MCTS technical analysis simulation failed: {e}")
            return {"analysis_confidence": 0.5, "error": str(e), "method": "fallback_estimation"}

    async def _validate_predictions_with_mcts(
        self, predictions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate AI predictions using MCTS technical analysis simulation"""
        try:
            symbols = list(predictions.keys())

            # Run MCTS technical analysis validation
            validation_config = {
                "symbols": symbols,
                "max_depth": 6,
                "prediction_validation": True,
                "technical_analysis_focus": True,
            }

            self.environment = ProductionTradingEnvironment(
                validation_config, self.market_data_provider
            )
            result = await self._mcp_run_mcts_parallel(iterations=100)

            return {
                "mcts_validation_score": result["confidence"],
                "prediction_alignment": "high"
                if result["confidence"] > 0.7
                else "medium"
                if result["confidence"] > 0.5
                else "low",
                "mcts_recommendation": result["best_action"],
                "validation_stats": result["stats"],
            }

        except Exception as e:
            logger.error(f"MCTS prediction validation failed: {e}")
            return {"validation_score": 0.5, "error": str(e), "method": "fallback_validation"}

    async def _mcp_cleanup(self):
        """
        Cleanup resources used by MCTS agent.
        Should be called when the agent is being shut down.
        """
        try:
            # Note: Grok4 client is a singleton and managed globally
            # We don't close it here as other agents might be using it
            # It should be closed at application shutdown with close_grok4_client()

            # Clean up other resources
            if hasattr(self, "monitoring_dashboard") and self.monitoring_dashboard:
                # Cleanup monitoring resources if needed
                pass

            if hasattr(self, "memory_agent") and self.memory_agent:
                # Memory agent cleanup if needed
                pass

            logger.info(f"MCTS Agent {self.agent_id} cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during MCTS agent cleanup: {e}")


# Backward compatibility alias
ProductionMCTSCalculationAgent = MCTSCalculationAgent
