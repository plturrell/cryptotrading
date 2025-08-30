"""
Trading Algorithm Agent - Trading strategy analysis and signal generation.

This agent analyzes markets and generates trading signals using 10 different strategies.
It provides analysis, backtesting, and signal generation capabilities.

IMPORTANT: This agent generates SIGNALS ONLY - it does NOT execute real trades or manage portfolios.
All "orders" are actually trade signals for analysis purposes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field

from ..base import BaseAgent, AgentStatus
from ...protocols.a2a.a2a_protocol import A2AMessage, MessageType
from ...ai.grok4_client import Grok4Client
from ....data.providers.real_only_provider import RealOnlyDataProvider

logger = logging.getLogger(__name__)


class TradingStrategy(Enum):
    """Available trading strategies."""
    GRID_TRADING = "grid_trading"
    DCA = "dollar_cost_averaging"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum_trend"
    MEAN_REVERSION = "mean_reversion"
    SCALPING = "scalping"
    MARKET_MAKING = "market_making"
    BREAKOUT = "breakout_trading"
    ML_PREDICTIVE = "ml_predictive"
    MULTI_STRATEGY = "multi_strategy_hybrid"


@dataclass
class TradingSignal:
    """Trading signal with confidence and metadata."""
    strategy: TradingStrategy
    action: str  # BUY, SELL, HOLD
    symbol: str
    price: Decimal
    quantity: Decimal
    confidence: float
    reason: str
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderGrid:
    """Grid trading order structure."""
    symbol: str
    center_price: Decimal
    grid_spacing: Decimal
    grid_levels: int
    buy_orders: List[Tuple[Decimal, Decimal]]  # (price, quantity)
    sell_orders: List[Tuple[Decimal, Decimal]]
    total_investment: Decimal


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity between exchanges."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    spread: Decimal
    profit_percentage: float
    volume: Decimal
    estimated_profit: Decimal


class TradingAlgorithmAgent(BaseAgent):
    """
    Advanced trading algorithm agent implementing multiple strategies.
    
    Leverages existing A2A infrastructure for:
    - MCTS calculations from MCTSCalculationAgent
    - Technical analysis from existing agents
    - AI insights from Grok4 integration
    - Real-time data from RealOnlyDataProvider
    """
    
    def __init__(self, agent_id: str = "trading_algorithm_agent"):
        super().__init__(agent_id=agent_id)
        
        # Initialize components
        self.data_provider = RealOnlyDataProvider()
        self.grok4_client = Grok4Client()
        
        # Strategy configurations
        self.active_strategies: Dict[TradingStrategy, bool] = {
            strategy: False for strategy in TradingStrategy
        }
        
        # Trading state
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.order_book: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Risk management
        self.max_position_size = Decimal("0.1")  # 10% of portfolio per position
        self.max_drawdown = 0.15  # 15% maximum drawdown
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Strategy-specific parameters
        self.strategy_params = self._initialize_strategy_params()
        
        # MCP tools registration
        self._register_mcp_tools()
    
    def _initialize_strategy_params(self) -> Dict[TradingStrategy, Dict[str, Any]]:
        """Initialize default parameters for each strategy."""
        return {
            TradingStrategy.GRID_TRADING: {
                "grid_spacing_percentage": 0.02,  # 2% spacing
                "grid_levels": 10,
                "investment_per_grid": Decimal("100"),
                "rebalance_threshold": 0.05
            },
            TradingStrategy.DCA: {
                "interval_hours": 24,
                "base_amount": Decimal("100"),
                "smart_dca_enabled": True,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "volume_multiplier": 1.5
            },
            TradingStrategy.ARBITRAGE: {
                "min_spread_percentage": 0.005,  # 0.5% minimum
                "max_slippage": 0.002,
                "latency_threshold_ms": 100,
                "exchanges": ["binance", "coinbase", "kraken"]
            },
            TradingStrategy.MOMENTUM: {
                "lookback_periods": 20,
                "ma_short": 50,
                "ma_long": 200,
                "rsi_threshold": 65,
                "volume_confirmation": True,
                "trend_strength_min": 0.7
            },
            TradingStrategy.MEAN_REVERSION: {
                "bollinger_periods": 20,
                "bollinger_std": 2,
                "z_score_threshold": 2.5,
                "mean_periods": 50,
                "pairs_correlation_min": 0.8
            },
            TradingStrategy.SCALPING: {
                "profit_target_percentage": 0.003,  # 0.3%
                "stop_loss_percentage": 0.001,  # 0.1%
                "max_trades_per_hour": 20,
                "min_volume_usd": 10000,
                "order_book_depth": 5
            },
            TradingStrategy.MARKET_MAKING: {
                "spread_percentage": 0.002,  # 0.2%
                "inventory_target": 0.5,
                "max_inventory_deviation": 0.2,
                "quote_refresh_seconds": 5,
                "dynamic_spread": True
            },
            TradingStrategy.BREAKOUT: {
                "lookback_periods": 50,
                "volume_surge_multiplier": 2.0,
                "breakout_confirmation_bars": 3,
                "false_breakout_filter": True,
                "atr_multiplier": 1.5
            },
            TradingStrategy.ML_PREDICTIVE: {
                "model_type": "ensemble",
                "features": ["price", "volume", "rsi", "macd", "sentiment"],
                "prediction_horizon": 24,  # hours
                "confidence_threshold": 0.75,
                "retrain_frequency_days": 7
            },
            TradingStrategy.MULTI_STRATEGY: {
                "allocation_mode": "dynamic",
                "rebalance_frequency_hours": 24,
                "strategy_weights": {},
                "risk_parity": True,
                "correlation_window": 30
            }
        }
    
    def _register_mcp_tools(self):
        """Register MCP tools for each trading strategy."""
        self.mcp_tools = {
            # Grid Trading Tools
            "grid_create": self._mcp_grid_create,
            "grid_rebalance": self._mcp_grid_rebalance,
            "grid_monitor": self._mcp_grid_monitor,
            
            # DCA Tools
            "dca_execute": self._mcp_dca_execute,
            "dca_smart_adjust": self._mcp_dca_smart_adjust,
            "dca_schedule": self._mcp_dca_schedule,
            
            # Arbitrage Tools
            "arbitrage_scan": self._mcp_arbitrage_scan,
            "arbitrage_execute": self._mcp_arbitrage_execute,
            "arbitrage_monitor": self._mcp_arbitrage_monitor,
            
            # Momentum Tools
            "momentum_scan": self._mcp_momentum_scan,
            "momentum_enter": self._mcp_momentum_enter,
            "momentum_trail_stop": self._mcp_momentum_trail_stop,
            
            # Mean Reversion Tools
            "mean_reversion_identify": self._mcp_mean_reversion_identify,
            "mean_reversion_trade": self._mcp_mean_reversion_trade,
            "pairs_trading_execute": self._mcp_pairs_trading_execute,
            
            # Scalping Tools
            "scalping_scan": self._mcp_scalping_scan,
            "scalping_execute": self._mcp_scalping_execute,
            "scalping_monitor": self._mcp_scalping_monitor,
            
            # Market Making Tools
            "market_making_quote": self._mcp_market_making_quote,
            "market_making_adjust": self._mcp_market_making_adjust,
            "inventory_manage": self._mcp_inventory_manage,
            
            # Breakout Tools
            "breakout_detect": self._mcp_breakout_detect,
            "breakout_confirm": self._mcp_breakout_confirm,
            "breakout_trade": self._mcp_breakout_trade,
            
            # ML Predictive Tools
            "ml_predict": self._mcp_ml_predict,
            "ml_train": self._mcp_ml_train,
            "ml_backtest": self._mcp_ml_backtest,
            
            # Multi-Strategy Tools
            "strategy_allocate": self._mcp_strategy_allocate,
            "strategy_optimize": self._mcp_strategy_optimize,
            "strategy_switch": self._mcp_strategy_switch,
            
            # Risk Management Tools
            "risk_calculate": self._mcp_risk_calculate,
            "position_size": self._mcp_position_size,
            "portfolio_optimize": self._mcp_portfolio_optimize
        }
    
    # ============= Grid Trading Strategy =============
    
    async def _mcp_grid_create(self, symbol: str, investment: Decimal) -> OrderGrid:
        """Create a grid trading setup for a symbol."""
        logger.info(f"Creating grid for {symbol} with investment {investment}")
        
        # Get current price
        current_price = await self._get_current_price(symbol)
        params = self.strategy_params[TradingStrategy.GRID_TRADING]
        
        # Calculate grid parameters
        spacing = current_price * Decimal(str(params["grid_spacing_percentage"]))
        levels = params["grid_levels"]
        
        # Generate buy and sell orders
        buy_orders = []
        sell_orders = []
        
        for i in range(1, levels + 1):
            # Buy orders below current price
            buy_price = current_price - (spacing * i)
            buy_quantity = params["investment_per_grid"] / buy_price
            buy_orders.append((buy_price, buy_quantity))
            
            # Sell orders above current price
            sell_price = current_price + (spacing * i)
            sell_quantity = params["investment_per_grid"] / sell_price
            sell_orders.append((sell_price, sell_quantity))
        
        grid = OrderGrid(
            symbol=symbol,
            center_price=current_price,
            grid_spacing=spacing,
            grid_levels=levels,
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            total_investment=investment
        )
        
        # Generate grid signals
        await self._generate_grid_signals(grid)
        
        return grid
    
    async def _mcp_grid_rebalance(self, symbol: str) -> Dict[str, Any]:
        """Rebalance grid when price moves significantly."""
        current_price = await self._get_current_price(symbol)
        existing_grid = self.order_book.get(symbol, {}).get("grid")
        
        if not existing_grid:
            return {"status": "no_grid", "symbol": symbol}
        
        # Check if rebalancing needed
        price_change = abs(current_price - existing_grid["center_price"]) / existing_grid["center_price"]
        
        if price_change > self.strategy_params[TradingStrategy.GRID_TRADING]["rebalance_threshold"]:
            # Mark existing grid as inactive
            self.order_book[symbol]["grid"]["active"] = False
            
            # Create new grid
            new_grid = await self._mcp_grid_create(symbol, existing_grid["total_investment"])
            
            return {
                "status": "rebalanced",
                "old_center": existing_grid["center_price"],
                "new_center": new_grid.center_price,
                "price_change": price_change
            }
        
        return {"status": "no_rebalance_needed", "price_change": price_change}
    
    async def _mcp_grid_monitor(self, symbol: str) -> Dict[str, Any]:
        """Monitor grid trading performance."""
        grid_data = self.order_book.get(symbol, {}).get("grid")
        if not grid_data:
            return {"status": "no_grid", "symbol": symbol}
        
        # Calculate metrics (analysis only)
        filled_signals = grid_data.get("filled_signals", [])
        total_profit = sum(signal.get("theoretical_profit", 0) for signal in filled_signals)
        
        return {
            "symbol": symbol,
            "center_price": grid_data["center_price"],
            "current_price": await self._get_current_price(symbol),
            "filled_signals": len(filled_signals),
            "total_profit": total_profit,
            "active_orders": len(grid_data.get("active_orders", [])),
            "grid_spacing": grid_data["grid_spacing"]
        }
    
    # ============= DCA Strategy =============
    
    async def _mcp_dca_execute(self, symbol: str, amount: Optional[Decimal] = None) -> Dict[str, Any]:
        """Execute a DCA purchase."""
        params = self.strategy_params[TradingStrategy.DCA]
        
        # Determine purchase amount
        if amount is None:
            amount = params["base_amount"]
        
        # Smart DCA adjustment
        if params["smart_dca_enabled"]:
            adjustment = await self._calculate_dca_adjustment(symbol)
            amount = amount * Decimal(str(adjustment))
        
        # Execute purchase
        current_price = await self._get_current_price(symbol)
        quantity = amount / current_price
        
        order = await self._generate_trade_signal(
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            price=current_price,
            order_type="MARKET"
        )
        
        # Track signal for analysis
        if symbol not in self.positions:
            self.positions[symbol] = {"signals": []}
        self.positions[symbol]["signals"].append({
            "quantity": quantity,
            "price": current_price,
            "timestamp": datetime.now()
        })
        
        return {
            "symbol": symbol,
            "amount": amount,
            "quantity": quantity,
            "price": current_price,
            "order_id": order.get("order_id"),
            "smart_adjustment": params["smart_dca_enabled"]
        }
    
    async def _mcp_dca_smart_adjust(self, symbol: str) -> Dict[str, Any]:
        """Calculate smart DCA adjustments based on market conditions."""
        params = self.strategy_params[TradingStrategy.DCA]
        
        # Get technical indicators
        rsi = await self._calculate_rsi(symbol)
        volume_ratio = await self._calculate_volume_ratio(symbol)
        
        # Calculate adjustment multiplier
        multiplier = 1.0
        
        if rsi < params["rsi_oversold"]:
            # Increase purchase during oversold
            multiplier += (params["rsi_oversold"] - rsi) / 100
        elif rsi > params["rsi_overbought"]:
            # Decrease purchase during overbought
            multiplier -= (rsi - params["rsi_overbought"]) / 100
        
        # Volume confirmation
        if volume_ratio > params["volume_multiplier"]:
            multiplier *= 1.2  # Increase during high volume
        
        # Get AI sentiment
        sentiment = await self._get_ai_sentiment(symbol)
        if sentiment["confidence"] > 0.7:
            if sentiment["sentiment"] == "bullish":
                multiplier *= 1.1
            elif sentiment["sentiment"] == "bearish":
                multiplier *= 0.9
        
        return {
            "symbol": symbol,
            "base_multiplier": multiplier,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "sentiment": sentiment,
            "recommended_amount": params["base_amount"] * Decimal(str(multiplier))
        }
    
    async def _mcp_dca_schedule(self, symbol: str, frequency_hours: int) -> Dict[str, Any]:
        """Schedule recurring DCA purchases."""
        # Create scheduled task
        task_id = f"dca_{symbol}_{datetime.now().timestamp()}"
        
        async def dca_task():
            while self.active_strategies[TradingStrategy.DCA]:
                await self._mcp_dca_execute(symbol)
                await asyncio.sleep(frequency_hours * 3600)
        
        # Start task
        asyncio.create_task(dca_task())
        
        return {
            "task_id": task_id,
            "symbol": symbol,
            "frequency_hours": frequency_hours,
            "status": "scheduled"
        }
    
    # ============= Arbitrage Strategy =============
    
    async def _mcp_arbitrage_scan(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities across exchanges."""
        opportunities = []
        params = self.strategy_params[TradingStrategy.ARBITRAGE]
        
        for symbol in symbols:
            # Get prices from multiple exchanges
            prices = await self._get_multi_exchange_prices(symbol, params["exchanges"])
            
            if len(prices) < 2:
                continue
            
            # Find arbitrage opportunities
            for buy_exchange, buy_price in prices.items():
                for sell_exchange, sell_price in prices.items():
                    if buy_exchange == sell_exchange:
                        continue
                    
                    spread = sell_price - buy_price
                    spread_percentage = spread / buy_price
                    
                    if spread_percentage > params["min_spread_percentage"]:
                        # Calculate potential profit
                        volume = await self._get_arbitrage_volume(symbol, buy_exchange, sell_exchange)
                        estimated_profit = volume * spread
                        
                        opportunity = ArbitrageOpportunity(
                            symbol=symbol,
                            buy_exchange=buy_exchange,
                            sell_exchange=sell_exchange,
                            buy_price=buy_price,
                            sell_price=sell_price,
                            spread=spread,
                            profit_percentage=float(spread_percentage),
                            volume=volume,
                            estimated_profit=estimated_profit
                        )
                        opportunities.append(opportunity)
        
        # Sort by profit potential
        opportunities.sort(key=lambda x: x.estimated_profit, reverse=True)
        
        return opportunities
    
    async def _mcp_arbitrage_execute(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """Generate arbitrage trading signals (NO ACTUAL EXECUTION)."""
        params = self.strategy_params[TradingStrategy.ARBITRAGE]
        
        # Check latency
        latency = await self._check_exchange_latency(
            [opportunity.buy_exchange, opportunity.sell_exchange]
        )
        
        if max(latency.values()) > params["latency_threshold_ms"]:
            return {
                "status": "failed",
                "reason": "latency_too_high",
                "latency": latency
            }
        
        # Generate simultaneous trading signals
        try:
            # Generate buy signal
            buy_order = await self._generate_trade_signal(
                symbol=opportunity.symbol,
                side="BUY",
                quantity=opportunity.volume,
                price=opportunity.buy_price,
                exchange=opportunity.buy_exchange,
                order_type="LIMIT"
            )
            
            # Generate sell signal
            sell_order = await self._generate_trade_signal(
                symbol=opportunity.symbol,
                side="SELL",
                quantity=opportunity.volume,
                price=opportunity.sell_price,
                exchange=opportunity.sell_exchange,
                order_type="LIMIT"
            )
            
            # Calculate theoretical profit
            theoretical_profit = (sell_order["suggested_price"] - buy_order["suggested_price"]) * opportunity.volume
            
            return {
                "status": "success",
                "buy_order": buy_order,
                "sell_order": sell_order,
                "theoretical_profit": theoretical_profit,
                "estimated_profit": opportunity.estimated_profit,
                "expected_slippage": theoretical_profit - opportunity.estimated_profit
            }
            
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {e}")
            return {
                "status": "failed",
                "reason": str(e),
                "opportunity": opportunity
            }
    
    async def _mcp_arbitrage_monitor(self) -> Dict[str, Any]:
        """Monitor arbitrage performance and opportunities."""
        # Get current opportunities
        symbols = await self._get_top_volume_symbols(20)
        opportunities = await self._mcp_arbitrage_scan(symbols)
        
        # Get performance metrics
        arbitrage_trades = self.performance_metrics.get("arbitrage_trades", [])
        
        total_profit = sum(trade.get("theoretical_profit", 0) for trade in arbitrage_trades)
        success_rate = len([t for t in arbitrage_trades if t["status"] == "success"]) / max(len(arbitrage_trades), 1)
        
        return {
            "active_opportunities": len(opportunities),
            "top_opportunity": opportunities[0] if opportunities else None,
            "total_trades": len(arbitrage_trades),
            "total_profit": total_profit,
            "success_rate": success_rate,
            "average_spread": np.mean([o.profit_percentage for o in opportunities]) if opportunities else 0
        }
    
    # ============= Momentum/Trend Following Strategy =============
    
    async def _mcp_momentum_scan(self, symbols: List[str]) -> List[TradingSignal]:
        """Scan for momentum trading opportunities."""
        signals = []
        params = self.strategy_params[TradingStrategy.MOMENTUM]
        
        for symbol in symbols:
            # Get price data
            prices = await self._get_price_history(symbol, params["lookback_periods"])
            
            if len(prices) < params["ma_long"]:
                continue
            
            # Calculate indicators
            ma_short = prices[-params["ma_short"]:].mean()
            ma_long = prices[-params["ma_long"]:].mean()
            rsi = await self._calculate_rsi(symbol)
            
            # Check momentum conditions
            if ma_short > ma_long and rsi > params["rsi_threshold"]:
                # Volume confirmation
                if params["volume_confirmation"]:
                    volume_surge = await self._check_volume_surge(symbol)
                    if not volume_surge:
                        continue
                
                # Calculate trend strength
                trend_strength = (ma_short - ma_long) / ma_long
                
                if trend_strength > params["trend_strength_min"]:
                    current_price = prices[-1]
                    
                    signal = TradingSignal(
                        strategy=TradingStrategy.MOMENTUM,
                        action="BUY",
                        symbol=symbol,
                        price=current_price,
                        quantity=await self._calculate_position_size(symbol, current_price),
                        confidence=min(trend_strength, 1.0),
                        reason=f"Strong momentum: MA crossover, RSI={rsi:.2f}",
                        stop_loss=current_price * Decimal("0.95"),
                        take_profit=current_price * Decimal("1.10"),
                        metadata={
                            "ma_short": float(ma_short),
                            "ma_long": float(ma_long),
                            "rsi": rsi,
                            "trend_strength": trend_strength
                        }
                    )
                    signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    async def _mcp_momentum_enter(self, signal: TradingSignal) -> Dict[str, Any]:
        """Enter a momentum trade."""
        # Risk management check
        risk_check = await self._check_risk_limits(signal)
        if not risk_check["approved"]:
            return {
                "status": "rejected",
                "reason": risk_check["reason"],
                "signal": signal
            }
        
        # Generate signal with stop loss and take profit
        order = await self._generate_trade_signal(
            symbol=signal.symbol,
            side=signal.action,
            quantity=signal.quantity,
            price=signal.price,
            order_type="MARKET",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        # Set up trailing stop
        if order["status"] == "filled":
            await self._setup_trailing_stop(
                symbol=signal.symbol,
                initial_stop=signal.stop_loss,
                trail_percentage=0.05
            )
        
        return {
            "status": "success",
            "order": order,
            "signal": signal,
            "risk_metrics": risk_check
        }
    
    async def _mcp_momentum_trail_stop(self, symbol: str) -> Dict[str, Any]:
        """Update trailing stop for momentum position."""
        position = self.positions.get(symbol)
        
        if not position:
            return {"status": "no_position", "symbol": symbol}
        
        current_price = await self._get_current_price(symbol)
        
        # Update trailing stop if price has moved favorably
        if current_price > position["highest_price"]:
            position["highest_price"] = current_price
            new_stop = current_price * Decimal("0.95")  # 5% trailing stop
            
            if new_stop > position["stop_loss"]:
                position["stop_loss"] = new_stop
                
                # Update order
                await self._update_stop_loss(symbol, new_stop)
                
                return {
                    "status": "updated",
                    "symbol": symbol,
                    "new_stop": new_stop,
                    "highest_price": position["highest_price"]
                }
        
        return {
            "status": "no_update",
            "symbol": symbol,
            "current_stop": position["stop_loss"],
            "highest_price": position["highest_price"]
        }
    
    # ============= Mean Reversion Strategy =============
    
    async def _mcp_mean_reversion_identify(self, symbols: List[str]) -> List[TradingSignal]:
        """Identify mean reversion opportunities."""
        signals = []
        params = self.strategy_params[TradingStrategy.MEAN_REVERSION]
        
        for symbol in symbols:
            # Get price history
            prices = await self._get_price_history(symbol, params["mean_periods"])
            
            if len(prices) < params["bollinger_periods"]:
                continue
            
            # Calculate Bollinger Bands
            mean = prices[-params["bollinger_periods"]:].mean()
            std = prices[-params["bollinger_periods"]:].std()
            upper_band = mean + (std * params["bollinger_std"])
            lower_band = mean - (std * params["bollinger_std"])
            
            current_price = prices[-1]
            
            # Calculate Z-score
            z_score = (current_price - mean) / std if std > 0 else 0
            
            # Check for mean reversion opportunity
            signal = None
            
            if abs(z_score) > params["z_score_threshold"]:
                if current_price < lower_band:
                    # Oversold - potential buy
                    signal = TradingSignal(
                        strategy=TradingStrategy.MEAN_REVERSION,
                        action="BUY",
                        symbol=symbol,
                        price=current_price,
                        quantity=await self._calculate_position_size(symbol, current_price),
                        confidence=min(abs(z_score) / 3, 1.0),
                        reason=f"Oversold: Z-score={z_score:.2f}",
                        stop_loss=current_price * Decimal("0.98"),
                        take_profit=mean,
                        metadata={
                            "z_score": z_score,
                            "mean": float(mean),
                            "upper_band": float(upper_band),
                            "lower_band": float(lower_band)
                        }
                    )
                elif current_price > upper_band:
                    # Overbought - potential sell/short
                    signal = TradingSignal(
                        strategy=TradingStrategy.MEAN_REVERSION,
                        action="SELL",
                        symbol=symbol,
                        price=current_price,
                        quantity=await self._calculate_position_size(symbol, current_price),
                        confidence=min(abs(z_score) / 3, 1.0),
                        reason=f"Overbought: Z-score={z_score:.2f}",
                        stop_loss=current_price * Decimal("1.02"),
                        take_profit=mean,
                        metadata={
                            "z_score": z_score,
                            "mean": float(mean),
                            "upper_band": float(upper_band),
                            "lower_band": float(lower_band)
                        }
                    )
                
                if signal:
                    signals.append(signal)
        
        return signals
    
    async def _mcp_mean_reversion_trade(self, signal: TradingSignal) -> Dict[str, Any]:
        """Generate mean reversion trade signal (NO ACTUAL EXECUTION)."""
        # Check if we already have a position
        if signal.symbol in self.positions:
            return {
                "status": "rejected",
                "reason": "already_in_position",
                "signal": signal
            }
        
        # Generate signal
        order = await self._generate_trade_signal(
            symbol=signal.symbol,
            side=signal.action,
            quantity=signal.quantity,
            price=signal.price,
            order_type="LIMIT",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        return {
            "status": "success",
            "order": order,
            "signal": signal,
            "expected_profit": abs(signal.take_profit - signal.price) * signal.quantity
        }
    
    async def _mcp_pairs_trading_execute(self, pair1: str, pair2: str) -> Dict[str, Any]:
        """Execute pairs trading strategy."""
        params = self.strategy_params[TradingStrategy.MEAN_REVERSION]
        
        # Calculate correlation
        prices1 = await self._get_price_history(pair1, 100)
        prices2 = await self._get_price_history(pair2, 100)
        
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        
        if correlation < params["pairs_correlation_min"]:
            return {
                "status": "rejected",
                "reason": "insufficient_correlation",
                "correlation": correlation
            }
        
        # Calculate spread
        spread = prices1[-1] / prices2[-1]
        mean_spread = (prices1 / prices2).mean()
        std_spread = (prices1 / prices2).std()
        z_score = (spread - mean_spread) / std_spread if std_spread > 0 else 0
        
        # Trade if spread is extreme
        if abs(z_score) > params["z_score_threshold"]:
            if z_score > 0:
                # Spread too high - sell pair1, buy pair2
                action1 = "SELL"
                action2 = "BUY"
            else:
                # Spread too low - buy pair1, sell pair2
                action1 = "BUY"
                action2 = "SELL"
            
            # Calculate quantities
            investment = Decimal("1000")  # Base investment
            quantity1 = investment / prices1[-1]
            quantity2 = investment / prices2[-1]
            
            # Generate signals
            order1 = await self._generate_trade_signal(
                symbol=pair1,
                side=action1,
                quantity=quantity1,
                price=prices1[-1],
                order_type="MARKET"
            )
            
            order2 = await self._generate_trade_signal(
                symbol=pair2,
                side=action2,
                quantity=quantity2,
                price=prices2[-1],
                order_type="MARKET"
            )
            
            return {
                "status": "success",
                "pair1": {"symbol": pair1, "action": action1, "order": order1},
                "pair2": {"symbol": pair2, "action": action2, "order": order2},
                "z_score": z_score,
                "correlation": correlation
            }
        
        return {
            "status": "no_opportunity",
            "z_score": z_score,
            "correlation": correlation
        }
    
    # ============= Helper Methods =============
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol."""
        data = await self.data_provider.get_latest_price(symbol)
        return Decimal(str(data["price"]))
    
    async def _get_price_history(self, symbol: str, periods: int) -> np.ndarray:
        """Get historical prices."""
        data = await self.data_provider.get_historical_data(
            symbol=symbol,
            interval="1h",
            limit=periods
        )
        return np.array([float(d["close"]) for d in data])
    
    async def _calculate_rsi(self, symbol: str, periods: int = 14) -> float:
        """Calculate RSI indicator."""
        prices = await self._get_price_history(symbol, periods + 1)
        
        deltas = np.diff(prices)
        gains = deltas[deltas > 0].sum() / periods
        losses = -deltas[deltas < 0].sum() / periods
        
        if losses == 0:
            return 100
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _calculate_position_size(self, symbol: str, price: Decimal) -> Decimal:
        """Calculate appropriate position size based on risk management."""
        portfolio_value = await self._get_portfolio_value()
        risk_amount = portfolio_value * Decimal(str(self.risk_per_trade))
        
        # Calculate position size based on stop loss distance
        stop_distance = price * Decimal("0.05")  # 5% default stop
        position_size = risk_amount / stop_distance
        
        # Apply maximum position size limit
        max_size = portfolio_value * self.max_position_size
        position_size = min(position_size, max_size / price)
        
        return position_size
    
    async def _get_portfolio_value(self) -> Decimal:
        """Calculate theoretical portfolio value for strategy analysis."""
        # Returns a default value for strategy calculations
        # NOTE: This does NOT connect to any real portfolio
        return Decimal("10000")
    
    async def _generate_trade_signal(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        order_type: str = "MARKET",
        exchange: Optional[str] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Generate a trade signal for analysis (NO ACTUAL ORDER EXECUTION)."""
        # This generates signals only - NO real trading
        signal = {
            "signal_id": f"signal_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "suggested_price": price,
            "signal_type": order_type,
            "status": "signal_generated",
            "timestamp": datetime.now(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "note": "SIGNAL ONLY - No actual order execution"
        }
        
        # Track signal for analysis
        if symbol not in self.order_book:
            self.order_book[symbol] = []
        self.order_book[symbol].append(signal)
        
        return signal
    
    async def _get_ai_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get AI sentiment analysis from Grok4."""
        prompt = f"Analyze market sentiment for {symbol} cryptocurrency. Return sentiment (bullish/bearish/neutral) and confidence (0-1)."
        
        response = await self.grok4_client.analyze(prompt)
        
        return {
            "sentiment": response.get("sentiment", "neutral"),
            "confidence": response.get("confidence", 0.5),
            "analysis": response.get("analysis", "")
        }
    
    async def _generate_grid_signals(self, grid: OrderGrid) -> None:
        """Generate grid trading signals (analysis only)."""
        if grid.symbol not in self.order_book:
            self.order_book[grid.symbol] = {}
        
        self.order_book[grid.symbol]["grid"] = {
            "center_price": grid.center_price,
            "grid_spacing": grid.grid_spacing,
            "total_investment": grid.total_investment,
            "buy_signals": grid.buy_orders,
            "sell_signals": grid.sell_orders,
            "active": True,
            "filled_signals": []
        }
        
        logger.info(f"Generated {len(grid.buy_orders)} buy and {len(grid.sell_orders)} sell signals for {grid.symbol}")
    
    async def process_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Process incoming A2A messages."""
        if message.type == MessageType.ANALYSIS_REQUEST:
            # Route to appropriate strategy
            strategy = message.payload.get("strategy")
            
            if strategy == "grid_trading":
                result = await self._mcp_grid_create(
                    message.payload["symbol"],
                    Decimal(message.payload["investment"])
                )
            elif strategy == "momentum":
                signals = await self._mcp_momentum_scan(
                    message.payload.get("symbols", [])
                )
                result = {"signals": signals}
            else:
                result = {"error": "Unknown strategy"}
            
            return A2AMessage(
                type=MessageType.ANALYSIS_RESPONSE,
                sender=self.agent_id,
                receiver=message.sender,
                payload=result
            )
        
        return await super().process_message(message)