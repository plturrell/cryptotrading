"""
Extended trading strategies for TradingAlgorithmAgent.

Implements Scalping, Market Making, Breakout, ML Predictive, and Multi-Strategy algorithms.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TradingStrategiesExtension:
    """Extended trading strategies for the TradingAlgorithmAgent."""

    # ============= Scalping Strategy =============

    async def _mcp_scalping_scan(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Scan for scalping opportunities with high-frequency patterns."""
        opportunities = []
        params = self.strategy_params[TradingStrategy.SCALPING]

        for symbol in symbols:
            # Check volume requirement
            volume_usd = await self._get_volume_usd(symbol)
            if volume_usd < params["min_volume_usd"]:
                continue

            # Get order book depth
            order_book = await self._get_order_book(symbol, params["order_book_depth"])

            # Analyze bid-ask spread
            if not order_book["bids"] or not order_book["asks"]:
                continue

            best_bid = Decimal(str(order_book["bids"][0]["price"]))
            best_ask = Decimal(str(order_book["asks"][0]["price"]))
            spread = best_ask - best_bid
            spread_percentage = spread / best_bid

            # Check for scalping opportunity
            if spread_percentage < params["profit_target_percentage"]:
                # Look for microstructure patterns
                pattern = await self._detect_microstructure_pattern(order_book)

                if pattern["detected"]:
                    opportunity = {
                        "symbol": symbol,
                        "pattern": pattern["type"],
                        "entry_price": best_bid + (spread * Decimal("0.3")),
                        "exit_price": best_ask - (spread * Decimal("0.3")),
                        "potential_profit": spread * Decimal("0.4"),
                        "volume": min(
                            order_book["bids"][0]["quantity"], order_book["asks"][0]["quantity"]
                        ),
                        "confidence": pattern["confidence"],
                        "holding_time_seconds": pattern.get("expected_duration", 30),
                    }
                    opportunities.append(opportunity)

        # Sort by profit potential
        opportunities.sort(key=lambda x: x["potential_profit"] * x["volume"], reverse=True)

        # Limit to max trades per hour
        max_trades = params["max_trades_per_hour"]
        return opportunities[:max_trades]

    async def _mcp_scalping_execute(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scalping trade signals with timing analysis."""
        params = self.strategy_params[TradingStrategy.SCALPING]

        # Calculate position size
        position_size = await self._calculate_scalping_position_size(
            opportunity["symbol"], opportunity["entry_price"]
        )

        # Generate entry signal
        entry_order = await self._generate_trade_signal(
            symbol=opportunity["symbol"],
            side="BUY",
            quantity=position_size,
            price=opportunity["entry_price"],
            order_type="LIMIT",
            time_in_force="IOC",  # Immediate or cancel
        )

        if entry_order["status"] != "filled":
            return {"status": "failed", "reason": "entry_not_filled", "opportunity": opportunity}

        # Set strict stop loss
        stop_loss = entry_order["suggested_price"] * (1 - params["stop_loss_percentage"])

        # Monitor for exit
        start_time = datetime.now()
        exit_order = None

        while (datetime.now() - start_time).seconds < opportunity["holding_time_seconds"]:
            current_price = await self._get_current_price(opportunity["symbol"])

            # Check stop loss
            if current_price <= stop_loss:
                exit_order = await self._generate_trade_signal(
                    symbol=opportunity["symbol"],
                    side="SELL",
                    quantity=position_size,
                    price=current_price,
                    order_type="MARKET",
                )
                break

            # Check profit target
            if current_price >= opportunity["exit_price"]:
                exit_order = await self._generate_trade_signal(
                    symbol=opportunity["symbol"],
                    side="SELL",
                    quantity=position_size,
                    price=opportunity["exit_price"],
                    order_type="LIMIT",
                )
                break

            await asyncio.sleep(0.1)  # High frequency monitoring

        # Force exit if time limit reached
        if not exit_order:
            current_price = await self._get_current_price(opportunity["symbol"])
            exit_order = await self._generate_trade_signal(
                symbol=opportunity["symbol"],
                side="SELL",
                quantity=position_size,
                price=current_price,
                order_type="MARKET",
            )

        # Calculate profit
        profit = (exit_order["suggested_price"] - entry_order["suggested_price"]) * position_size

        return {
            "status": "completed",
            "entry_order": entry_order,
            "exit_order": exit_order,
            "profit": profit,
            "holding_time": (datetime.now() - start_time).seconds,
            "profit_percentage": float(profit / (entry_order["suggested_price"] * position_size)),
        }

    async def _mcp_scalping_monitor(self) -> Dict[str, Any]:
        """Monitor scalping performance and adjust parameters."""
        scalping_trades = self.performance_metrics.get("scalping_trades", [])

        if not scalping_trades:
            return {"status": "no_trades", "recommendations": []}

        # Calculate metrics
        total_trades = len(scalping_trades)
        profitable_trades = [t for t in scalping_trades if t["profit"] > 0]
        win_rate = len(profitable_trades) / total_trades

        avg_profit = np.mean([t["profit"] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t["profit"] for t in scalping_trades if t["profit"] < 0]) or 1
        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else 0

        avg_holding_time = np.mean([t["holding_time"] for t in scalping_trades])

        # Generate recommendations
        recommendations = []

        if win_rate < 0.6:
            recommendations.append("Tighten entry criteria - win rate too low")

        if profit_factor < 1.5:
            recommendations.append("Adjust profit targets - risk/reward ratio suboptimal")

        if avg_holding_time > 60:
            recommendations.append("Reduce holding time - positions held too long for scalping")

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_holding_time": avg_holding_time,
            "total_profit": sum(t["profit"] for t in scalping_trades),
            "recommendations": recommendations,
            "trades_per_hour": total_trades
            / max(
                (scalping_trades[-1]["timestamp"] - scalping_trades[0]["timestamp"]).total_seconds()
                / 3600,
                1,
            ),
        }

    # ============= Market Making Strategy =============

    async def _mcp_market_making_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate market making quotes."""
        params = self.strategy_params[TradingStrategy.MARKET_MAKING]

        # Get current market state
        mid_price = await self._get_mid_price(symbol)
        volatility = await self._calculate_volatility(symbol)
        order_book = await self._get_order_book(symbol, 10)

        # Calculate dynamic spread if enabled
        if params["dynamic_spread"]:
            base_spread = params["spread_percentage"]

            # Adjust for volatility
            volatility_adjustment = volatility * 0.5

            # Adjust for inventory
            inventory = self.positions.get(symbol, {}).get("quantity", 0)
            inventory_ratio = (
                inventory / params["inventory_target"] if params["inventory_target"] > 0 else 0
            )
            inventory_adjustment = (inventory_ratio - 1) * 0.001  # Skew prices based on inventory

            bid_spread = base_spread + volatility_adjustment - inventory_adjustment
            ask_spread = base_spread + volatility_adjustment + inventory_adjustment
        else:
            bid_spread = ask_spread = params["spread_percentage"]

        # Calculate quote prices
        bid_price = mid_price * (1 - bid_spread)
        ask_price = mid_price * (1 + ask_spread)

        # Calculate quote sizes based on inventory
        base_size = Decimal("100")  # Base quote size

        if inventory_ratio > params["inventory_target"] + params["max_inventory_deviation"]:
            # Too much inventory - increase ask size, decrease bid size
            bid_size = base_size * Decimal("0.5")
            ask_size = base_size * Decimal("1.5")
        elif inventory_ratio < params["inventory_target"] - params["max_inventory_deviation"]:
            # Too little inventory - increase bid size, decrease ask size
            bid_size = base_size * Decimal("1.5")
            ask_size = base_size * Decimal("0.5")
        else:
            bid_size = ask_size = base_size

        # Place quotes
        quotes = {
            "symbol": symbol,
            "bid": {"price": bid_price, "size": bid_size, "spread": bid_spread},
            "ask": {"price": ask_price, "size": ask_size, "spread": ask_spread},
            "mid_price": mid_price,
            "volatility": volatility,
            "inventory_ratio": inventory_ratio,
            "timestamp": datetime.now(),
        }

        # Submit orders
        await self._submit_market_making_orders(quotes)

        return quotes

    async def _mcp_market_making_adjust(self, symbol: str) -> Dict[str, Any]:
        """Adjust market making parameters based on conditions."""
        params = self.strategy_params[TradingStrategy.MARKET_MAKING]

        # Get recent fills
        recent_fills = await self._get_recent_fills(symbol, minutes=30)

        if not recent_fills:
            return {"status": "no_fills", "adjustments": None}

        # Analyze fill imbalance
        buy_fills = [f for f in recent_fills if f["side"] == "BUY"]
        sell_fills = [f for f in recent_fills if f["side"] == "SELL"]

        fill_ratio = len(buy_fills) / max(len(sell_fills), 1)

        adjustments = {}

        # Adjust spreads based on fill imbalance
        if fill_ratio > 1.5:
            # More buys than sells - widen bid spread
            adjustments["bid_spread_adjustment"] = 1.2
            adjustments["ask_spread_adjustment"] = 0.9
        elif fill_ratio < 0.67:
            # More sells than buys - widen ask spread
            adjustments["bid_spread_adjustment"] = 0.9
            adjustments["ask_spread_adjustment"] = 1.2
        else:
            adjustments["bid_spread_adjustment"] = 1.0
            adjustments["ask_spread_adjustment"] = 1.0

        # Calculate profitability
        total_profit = sum(f.get("profit", 0) for f in recent_fills)
        avg_profit_per_trade = total_profit / len(recent_fills)

        if avg_profit_per_trade < 0:
            # Losing money - widen spreads
            adjustments["overall_spread_adjustment"] = 1.3

        # Apply adjustments
        await self._apply_market_making_adjustments(symbol, adjustments)

        return {
            "status": "adjusted",
            "fill_ratio": fill_ratio,
            "total_fills": len(recent_fills),
            "total_profit": total_profit,
            "adjustments": adjustments,
        }

    async def _mcp_inventory_manage(self, symbol: str) -> Dict[str, Any]:
        """Manage inventory to maintain target levels."""
        params = self.strategy_params[TradingStrategy.MARKET_MAKING]

        current_inventory = self.positions.get(symbol, {}).get("quantity", 0)
        target_inventory = params["inventory_target"]

        deviation = abs(current_inventory - target_inventory) / max(target_inventory, 1)

        if deviation > params["max_inventory_deviation"]:
            # Need to rebalance
            rebalance_quantity = target_inventory - current_inventory

            if rebalance_quantity > 0:
                # Need to buy
                order = await self._generate_trade_signal(
                    symbol=symbol,
                    side="BUY",
                    quantity=abs(rebalance_quantity),
                    price=await self._get_current_price(symbol),
                    order_type="MARKET",
                )
            else:
                # Need to sell
                order = await self._generate_trade_signal(
                    symbol=symbol,
                    side="SELL",
                    quantity=abs(rebalance_quantity),
                    price=await self._get_current_price(symbol),
                    order_type="MARKET",
                )

            return {
                "status": "rebalanced",
                "previous_inventory": current_inventory,
                "target_inventory": target_inventory,
                "rebalance_quantity": rebalance_quantity,
                "order": order,
            }

        return {
            "status": "balanced",
            "current_inventory": current_inventory,
            "target_inventory": target_inventory,
            "deviation": deviation,
        }

    # ============= Breakout Trading Strategy =============

    async def _mcp_breakout_detect(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Detect potential breakout patterns."""
        breakouts = []
        params = self.strategy_params[TradingStrategy.BREAKOUT]

        for symbol in symbols:
            # Get price history
            prices = await self._get_price_history(symbol, params["lookback_periods"])
            volumes = await self._get_volume_history(symbol, params["lookback_periods"])

            if len(prices) < params["lookback_periods"]:
                continue

            # Identify support and resistance levels
            resistance = np.max(prices[-params["lookback_periods"] :])
            support = np.min(prices[-params["lookback_periods"] :])
            current_price = prices[-1]

            # Calculate ATR for volatility
            atr = await self._calculate_atr(symbol, 14)

            # Check for breakout conditions
            breakout = None

            # Resistance breakout
            if current_price > resistance * 0.995:  # Within 0.5% of resistance
                # Check volume surge
                recent_volume = volumes[-1]
                avg_volume = np.mean(volumes[:-1])
                volume_surge = recent_volume / avg_volume if avg_volume > 0 else 0

                if volume_surge > params["volume_surge_multiplier"]:
                    breakout = {
                        "symbol": symbol,
                        "type": "resistance",
                        "level": resistance,
                        "current_price": current_price,
                        "volume_surge": volume_surge,
                        "atr": atr,
                        "strength": (current_price - resistance) / resistance,
                        "target": resistance + (atr * params["atr_multiplier"]),
                        "stop_loss": resistance - (atr * 0.5),
                    }

            # Support breakout (breakdown)
            elif current_price < support * 1.005:  # Within 0.5% of support
                recent_volume = volumes[-1]
                avg_volume = np.mean(volumes[:-1])
                volume_surge = recent_volume / avg_volume if avg_volume > 0 else 0

                if volume_surge > params["volume_surge_multiplier"]:
                    breakout = {
                        "symbol": symbol,
                        "type": "support",
                        "level": support,
                        "current_price": current_price,
                        "volume_surge": volume_surge,
                        "atr": atr,
                        "strength": (support - current_price) / support,
                        "target": support - (atr * params["atr_multiplier"]),
                        "stop_loss": support + (atr * 0.5),
                    }

            if breakout:
                # Check for chart patterns
                pattern = await self._detect_chart_pattern(prices)
                breakout["pattern"] = pattern
                breakouts.append(breakout)

        return breakouts

    async def _mcp_breakout_confirm(self, breakout: Dict[str, Any]) -> Dict[str, Any]:
        """Confirm breakout with additional validation."""
        params = self.strategy_params[TradingStrategy.BREAKOUT]

        confirmations = []
        confidence = 0.5

        # Wait for confirmation bars
        confirmation_prices = []
        for _ in range(params["breakout_confirmation_bars"]):
            await asyncio.sleep(60)  # Wait 1 minute per bar
            price = await self._get_current_price(breakout["symbol"])
            confirmation_prices.append(price)

        # Check if price stays beyond breakout level
        if breakout["type"] == "resistance":
            confirmed_bars = sum(1 for p in confirmation_prices if p > breakout["level"])
        else:
            confirmed_bars = sum(1 for p in confirmation_prices if p < breakout["level"])

        if confirmed_bars >= params["breakout_confirmation_bars"] - 1:
            confirmations.append("price_confirmation")
            confidence += 0.2

        # Check volume persistence
        recent_volumes = await self._get_volume_history(breakout["symbol"], 5)
        if np.mean(recent_volumes) > np.mean(recent_volumes[:-5]) * 1.5:
            confirmations.append("volume_persistence")
            confidence += 0.15

        # False breakout filter
        if params["false_breakout_filter"]:
            false_breakout_probability = await self._calculate_false_breakout_probability(breakout)
            if false_breakout_probability < 0.3:
                confirmations.append("low_false_breakout_risk")
                confidence += 0.15

        # AI confirmation
        ai_analysis = await self._get_ai_sentiment(breakout["symbol"])
        if ai_analysis["confidence"] > 0.7:
            if (breakout["type"] == "resistance" and ai_analysis["sentiment"] == "bullish") or (
                breakout["type"] == "support" and ai_analysis["sentiment"] == "bearish"
            ):
                confirmations.append("ai_confirmation")
                confidence += 0.1

        return {
            "confirmed": len(confirmations) >= 2,
            "confidence": min(confidence, 1.0),
            "confirmations": confirmations,
            "breakout": breakout,
            "confirmation_prices": confirmation_prices,
        }

    async def _mcp_breakout_trade(self, confirmation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate breakout trade signal after confirmation."""
        if not confirmation["confirmed"]:
            return {
                "status": "rejected",
                "reason": "insufficient_confirmation",
                "confirmation": confirmation,
            }

        breakout = confirmation["breakout"]

        # Determine trade direction
        if breakout["type"] == "resistance":
            side = "BUY"
        else:
            side = "SELL"

        # Calculate position size with ATR-based risk
        position_size = await self._calculate_atr_position_size(
            breakout["symbol"], breakout["current_price"], breakout["atr"]
        )

        # Generate signal with dynamic stop and target
        order = await self._generate_trade_signal(
            symbol=breakout["symbol"],
            side=side,
            quantity=position_size,
            price=await self._get_current_price(breakout["symbol"]),
            order_type="MARKET",
            stop_loss=breakout["stop_loss"],
            take_profit=breakout["target"],
        )

        # Set up monitoring for pullback entry
        if order["status"] == "filled":
            asyncio.create_task(self._monitor_breakout_pullback(breakout["symbol"], breakout))

        return {
            "status": "executed",
            "order": order,
            "breakout": breakout,
            "confidence": confirmation["confidence"],
            "risk_reward_ratio": abs(breakout["target"] - breakout["current_price"])
            / abs(breakout["stop_loss"] - breakout["current_price"]),
        }

    # ============= ML/AI Predictive Strategy =============

    async def _mcp_ml_predict(self, symbol: str, horizon_hours: int = 24) -> Dict[str, Any]:
        """Generate ML predictions for price movement."""
        params = self.strategy_params[TradingStrategy.ML_PREDICTIVE]

        # Prepare features
        features = await self._prepare_ml_features(symbol, params["features"])

        # Load or create model
        model = await self._get_or_create_ml_model(symbol, params["model_type"])

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))

        # Generate predictions
        if params["model_type"] == "ensemble":
            # Use ensemble of models
            predictions = []
            confidences = []

            # Random Forest
            rf_model = model["random_forest"]
            rf_pred = rf_model.predict(features_scaled)[0]
            rf_confidence = np.max(rf_model.predict_proba(features_scaled)[0])
            predictions.append(rf_pred)
            confidences.append(rf_confidence)

            # Gradient Boosting
            gb_model = model["gradient_boosting"]
            gb_pred = gb_model.predict(features_scaled)[0]
            predictions.append(gb_pred)
            confidences.append(0.7)  # GB doesn't have predict_proba for regression

            # Combine predictions
            final_prediction = np.mean(predictions)
            final_confidence = np.mean(confidences)

        else:
            final_prediction = model.predict(features_scaled)[0]
            final_confidence = 0.6  # Default confidence

        # Interpret prediction
        current_price = await self._get_current_price(symbol)
        predicted_price = current_price * (1 + final_prediction)

        # Generate trading signal
        if final_confidence > params["confidence_threshold"]:
            if final_prediction > 0.01:  # 1% upside
                action = "BUY"
                reason = f"ML predicts {final_prediction*100:.2f}% increase"
            elif final_prediction < -0.01:  # 1% downside
                action = "SELL"
                reason = f"ML predicts {abs(final_prediction)*100:.2f}% decrease"
            else:
                action = "HOLD"
                reason = "Insufficient predicted movement"
        else:
            action = "HOLD"
            reason = f"Low confidence: {final_confidence:.2f}"

        # Feature importance
        if hasattr(model.get("random_forest"), "feature_importances_"):
            feature_importance = dict(
                zip(params["features"], model["random_forest"].feature_importances_)
            )
        else:
            feature_importance = {}

        return {
            "symbol": symbol,
            "prediction": final_prediction,
            "predicted_price": predicted_price,
            "current_price": current_price,
            "confidence": final_confidence,
            "horizon_hours": horizon_hours,
            "action": action,
            "reason": reason,
            "feature_importance": feature_importance,
            "model_type": params["model_type"],
        }

    async def _mcp_ml_train(self, symbols: List[str]) -> Dict[str, Any]:
        """Train ML models on historical data."""
        params = self.strategy_params[TradingStrategy.ML_PREDICTIVE]

        training_results = {}

        for symbol in symbols:
            # Prepare training data
            X, y = await self._prepare_training_data(symbol, params["features"])

            if len(X) < 100:
                training_results[symbol] = {"status": "insufficient_data"}
                continue

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train ensemble models
            models = {}

            # Random Forest Classifier
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            rf_score = rf_model.score(X_test_scaled, y_test)
            models["random_forest"] = rf_model

            # Gradient Boosting Regressor
            gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            gb_model.fit(X_train_scaled, y_train.astype(float))
            gb_score = gb_model.score(X_test_scaled, y_test.astype(float))
            models["gradient_boosting"] = gb_model

            # Save models
            model_path = f"models/{symbol}_ml_model.pkl"
            joblib.dump(models, model_path)

            # Save scaler
            scaler_path = f"models/{symbol}_scaler.pkl"
            joblib.dump(scaler, scaler_path)

            training_results[symbol] = {
                "status": "success",
                "rf_accuracy": rf_score,
                "gb_r2_score": gb_score,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "model_path": model_path,
            }

        return {
            "trained_symbols": len(training_results),
            "results": training_results,
            "timestamp": datetime.now(),
        }

    async def _mcp_ml_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Backtest ML predictions on historical data."""
        # Load model
        model = await self._get_or_create_ml_model(symbol, "ensemble")

        # Get historical data
        historical_data = await self._get_historical_data_range(symbol, start_date, end_date)

        trades = []
        total_profit = 0

        for i in range(len(historical_data) - 24):  # 24 hour prediction horizon
            # Prepare features for this point
            features = await self._prepare_ml_features_at_point(
                historical_data, i, self.strategy_params[TradingStrategy.ML_PREDICTIVE]["features"]
            )

            # Get prediction
            prediction = await self._mcp_ml_predict(symbol, 24)

            if prediction["action"] != "HOLD":
                # Simulate trade
                entry_price = historical_data[i]["close"]
                exit_price = historical_data[i + 24]["close"]

                if prediction["action"] == "BUY":
                    profit = exit_price - entry_price
                else:
                    profit = entry_price - exit_price

                trades.append(
                    {
                        "timestamp": historical_data[i]["timestamp"],
                        "action": prediction["action"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "profit": profit,
                        "prediction": prediction["prediction"],
                        "actual": (exit_price - entry_price) / entry_price,
                    }
                )

                total_profit += profit

        # Calculate metrics
        if trades:
            win_rate = len([t for t in trades if t["profit"] > 0]) / len(trades)
            avg_profit = (
                np.mean([t["profit"] for t in trades if t["profit"] > 0])
                if any(t["profit"] > 0 for t in trades)
                else 0
            )
            avg_loss = (
                np.mean([t["profit"] for t in trades if t["profit"] < 0])
                if any(t["profit"] < 0 for t in trades)
                else 0
            )
            sharpe_ratio = (
                np.mean([t["profit"] for t in trades]) / np.std([t["profit"] for t in trades])
                if trades
                else 0
            )
        else:
            win_rate = avg_profit = avg_loss = sharpe_ratio = 0

        return {
            "symbol": symbol,
            "period": f"{start_date} to {end_date}",
            "total_trades": len(trades),
            "total_profit": total_profit,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades[-10:],  # Last 10 trades for review
        }

    # ============= Multi-Strategy Hybrid System =============

    async def _mcp_strategy_allocate(self) -> Dict[str, Any]:
        """Dynamically allocate capital across strategies."""
        params = self.strategy_params[TradingStrategy.MULTI_STRATEGY]

        # Get performance metrics for each strategy
        strategy_performance = {}

        for strategy in TradingStrategy:
            if strategy == TradingStrategy.MULTI_STRATEGY:
                continue

            # Calculate strategy metrics
            metrics = await self._calculate_strategy_metrics(strategy)
            strategy_performance[strategy] = metrics

        # Calculate allocations
        if params["allocation_mode"] == "dynamic":
            # Risk parity allocation
            if params["risk_parity"]:
                allocations = await self._calculate_risk_parity_allocation(strategy_performance)
            else:
                # Performance-based allocation
                allocations = await self._calculate_performance_allocation(strategy_performance)
        else:
            # Fixed allocation
            allocations = params["strategy_weights"]

        # Apply constraints
        total_allocation = sum(allocations.values())
        if total_allocation > 1.0:
            # Normalize to 100%
            allocations = {k: v / total_allocation for k, v in allocations.items()}

        # Update active strategies
        for strategy, allocation in allocations.items():
            self.active_strategies[strategy] = allocation > 0.01  # Activate if > 1% allocation

        return {
            "allocations": allocations,
            "strategy_performance": strategy_performance,
            "allocation_mode": params["allocation_mode"],
            "timestamp": datetime.now(),
        }

    async def _mcp_strategy_optimize(self) -> Dict[str, Any]:
        """Optimize strategy parameters based on performance."""
        optimization_results = {}

        for strategy in TradingStrategy:
            if strategy == TradingStrategy.MULTI_STRATEGY:
                continue

            # Get recent performance
            performance = await self._get_strategy_performance(strategy, days=30)

            if not performance["trades"]:
                continue

            # Optimize based on strategy type
            if strategy == TradingStrategy.GRID_TRADING:
                # Optimize grid spacing
                optimal_spacing = await self._optimize_grid_spacing(performance)
                self.strategy_params[strategy]["grid_spacing_percentage"] = optimal_spacing
                optimization_results[strategy] = {"grid_spacing": optimal_spacing}

            elif strategy == TradingStrategy.MOMENTUM:
                # Optimize MA periods
                optimal_periods = await self._optimize_ma_periods(performance)
                self.strategy_params[strategy]["ma_short"] = optimal_periods["short"]
                self.strategy_params[strategy]["ma_long"] = optimal_periods["long"]
                optimization_results[strategy] = optimal_periods

            elif strategy == TradingStrategy.SCALPING:
                # Optimize profit targets
                optimal_targets = await self._optimize_scalping_targets(performance)
                self.strategy_params[strategy]["profit_target_percentage"] = optimal_targets[
                    "profit"
                ]
                self.strategy_params[strategy]["stop_loss_percentage"] = optimal_targets["stop"]
                optimization_results[strategy] = optimal_targets

        return {
            "optimized_strategies": len(optimization_results),
            "results": optimization_results,
            "timestamp": datetime.now(),
        }

    async def _mcp_strategy_switch(self, market_condition: str) -> Dict[str, Any]:
        """Switch strategies based on market conditions."""
        # Define strategy preferences for market conditions
        strategy_preferences = {
            "trending_up": [
                TradingStrategy.MOMENTUM,
                TradingStrategy.BREAKOUT,
                TradingStrategy.DCA,
            ],
            "trending_down": [
                TradingStrategy.MEAN_REVERSION,
                TradingStrategy.DCA,
                TradingStrategy.ARBITRAGE,
            ],
            "ranging": [
                TradingStrategy.GRID_TRADING,
                TradingStrategy.MEAN_REVERSION,
                TradingStrategy.MARKET_MAKING,
            ],
            "high_volatility": [
                TradingStrategy.SCALPING,
                TradingStrategy.GRID_TRADING,
                TradingStrategy.ARBITRAGE,
            ],
            "low_volatility": [
                TradingStrategy.MARKET_MAKING,
                TradingStrategy.MOMENTUM,
                TradingStrategy.BREAKOUT,
            ],
        }

        # Get preferred strategies for current condition
        preferred = strategy_preferences.get(market_condition, [])

        # Deactivate non-preferred strategies
        for strategy in TradingStrategy:
            if strategy == TradingStrategy.MULTI_STRATEGY:
                continue
            self.active_strategies[strategy] = strategy in preferred

        # Allocate capital to active strategies
        allocations = await self._mcp_strategy_allocate()

        return {
            "market_condition": market_condition,
            "active_strategies": [s for s, active in self.active_strategies.items() if active],
            "allocations": allocations["allocations"],
            "timestamp": datetime.now(),
        }

    # ============= Risk Management Tools =============

    async def _mcp_risk_calculate(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        positions = portfolio.get("positions", {})

        if not positions:
            return {"status": "no_positions", "risk_metrics": {}}

        # Calculate Value at Risk (VaR)
        returns = []
        for symbol, position in positions.items():
            historical_returns = await self._get_historical_returns(symbol, 30)
            returns.extend(historical_returns)

        if returns:
            var_95 = np.percentile(returns, 5)  # 95% VaR
            cvar_95 = np.mean([r for r in returns if r <= var_95])  # Conditional VaR
        else:
            var_95 = cvar_95 = 0

        # Calculate position correlations
        correlation_matrix = await self._calculate_correlation_matrix(list(positions.keys()))

        # Calculate portfolio beta
        portfolio_beta = await self._calculate_portfolio_beta(positions)

        # Calculate maximum drawdown
        portfolio_values = await self._get_portfolio_history(30)
        if portfolio_values:
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0

        # Calculate Sharpe ratio
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)
        else:
            sharpe_ratio = 0

        return {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "portfolio_beta": portfolio_beta,
            "correlation_risk": np.max(correlation_matrix) if correlation_matrix.size > 0 else 0,
            "position_count": len(positions),
            "total_exposure": sum(p.get("value", 0) for p in positions.values()),
        }

    async def _mcp_position_size(
        self, symbol: str, entry_price: Decimal, stop_loss: Decimal
    ) -> Dict[str, Any]:
        """Calculate optimal position size using Kelly Criterion and risk management."""
        # Get win rate and average win/loss
        historical_trades = await self._get_symbol_trade_history(symbol)

        if historical_trades:
            wins = [t for t in historical_trades if t["profit"] > 0]
            losses = [t for t in historical_trades if t["profit"] < 0]

            win_rate = len(wins) / len(historical_trades)
            avg_win = np.mean([t["profit"] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t["profit"] for t in losses])) if losses else 1

            # Kelly Criterion
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.02  # Default 2%
        else:
            kelly_fraction = 0.02  # Default for no history

        # Risk-based position sizing
        portfolio_value = await self._get_portfolio_value()
        risk_amount = portfolio_value * Decimal(str(self.risk_per_trade))

        stop_distance = abs(entry_price - stop_loss)
        risk_based_size = risk_amount / stop_distance if stop_distance > 0 else 0

        # Combine Kelly and risk-based sizing
        kelly_size = portfolio_value * Decimal(str(kelly_fraction)) / entry_price

        # Use the smaller of the two
        final_size = min(risk_based_size, kelly_size)

        # Apply maximum position limit
        max_size = portfolio_value * self.max_position_size / entry_price
        final_size = min(final_size, max_size)

        return {
            "symbol": symbol,
            "recommended_size": final_size,
            "kelly_fraction": kelly_fraction,
            "risk_based_size": risk_based_size,
            "kelly_size": kelly_size,
            "max_allowed": max_size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_amount": risk_amount,
        }

    async def _mcp_portfolio_optimize(
        self, symbols: List[str], target_return: float = 0.1
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation using Modern Portfolio Theory."""
        # Get historical returns for all symbols
        returns_data = {}
        for symbol in symbols:
            returns = await self._get_historical_returns(symbol, 365)
            returns_data[symbol] = returns

        # Convert to DataFrame for easier calculation
        returns_df = pd.DataFrame(returns_data)

        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 365  # Annualized
        cov_matrix = returns_df.cov() * 365  # Annualized

        # Optimize for maximum Sharpe ratio
        num_assets = len(symbols)

        def neg_sharpe_ratio(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - 0.02) / portfolio_std  # Risk-free rate = 2%

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
            {"type": "ineq", "fun": lambda x: x},  # No short selling
        ]

        # Initial guess
        initial_weights = np.array([1 / num_assets] * num_assets)

        # Optimize
        from scipy.optimize import minimize

        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method="SLSQP",
            constraints=constraints,
            bounds=[(0, 1) for _ in range(num_assets)],
        )

        optimal_weights = result.x

        # Calculate portfolio metrics
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_std

        # Create allocation dictionary
        allocations = {
            symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 0.01
        }

        return {
            "allocations": allocations,
            "expected_return": portfolio_return,
            "expected_volatility": portfolio_std,
            "sharpe_ratio": sharpe_ratio,
            "optimization_status": "success" if result.success else "failed",
            "symbols": symbols,
        }
