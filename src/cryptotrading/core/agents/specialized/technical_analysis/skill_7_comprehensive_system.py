"""
Skill 7: Comprehensive Technical Analysis System
STRAND tools for combining all technical analysis skills into unified trading signals and risk assessment
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_signal_strength(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Overall Signal Strength from Multiple Indicators

    Args:
        signals: List of signal dictionaries from various TA skills

    Returns:
        Dictionary with aggregated signal strength analysis
    """
    try:
        if not signals:
            return {
                "success": True,
                "overall_signal": "neutral",
                "strength": 0.0,
                "confidence": 0.0,
                "signal_breakdown": {},
                "timestamp": pd.Timestamp.now().isoformat(),
            }

        # Signal strength weights
        strength_weights = {"weak": 1.0, "medium": 2.0, "strong": 3.0, "very_strong": 4.0}

        # Skill importance weights
        skill_weights = {
            "momentum_indicators": 1.0,
            "momentum_volatility": 1.2,
            "volume_analysis": 1.1,
            "support_resistance": 1.3,
            "chart_patterns": 1.4,
            "harmonic_patterns": 1.5,
        }

        buy_score = 0.0
        sell_score = 0.0
        neutral_score = 0.0
        total_weight = 0.0

        signal_breakdown = {
            "buy_signals": [],
            "sell_signals": [],
            "neutral_signals": [],
            "by_skill": {},
        }

        for signal in signals:
            signal_type = signal.get("signal", "neutral")
            strength = signal.get("strength", "medium")
            skill = signal.get("skill", "unknown")
            indicator = signal.get("indicator", "Unknown")

            # Calculate weighted score
            strength_weight = strength_weights.get(strength, 2.0)
            skill_weight = skill_weights.get(skill, 1.0)
            signal_weight = strength_weight * skill_weight

            # Add to appropriate category
            if signal_type == "buy":
                buy_score += signal_weight
                signal_breakdown["buy_signals"].append(
                    {
                        "indicator": indicator,
                        "strength": strength,
                        "weight": signal_weight,
                        "note": signal.get("note", ""),
                    }
                )
            elif signal_type == "sell":
                sell_score += signal_weight
                signal_breakdown["sell_signals"].append(
                    {
                        "indicator": indicator,
                        "strength": strength,
                        "weight": signal_weight,
                        "note": signal.get("note", ""),
                    }
                )
            else:
                neutral_score += signal_weight
                signal_breakdown["neutral_signals"].append(
                    {
                        "indicator": indicator,
                        "strength": strength,
                        "weight": signal_weight,
                        "note": signal.get("note", ""),
                    }
                )

            # Track by skill
            if skill not in signal_breakdown["by_skill"]:
                signal_breakdown["by_skill"][skill] = {"buy": 0, "sell": 0, "neutral": 0}
            signal_breakdown["by_skill"][skill][signal_type] += 1

            total_weight += signal_weight

        # Determine overall signal
        if total_weight > 0:
            buy_ratio = buy_score / total_weight
            sell_ratio = sell_score / total_weight
            neutral_ratio = neutral_score / total_weight

            # Overall signal determination
            if buy_ratio > 0.6:
                overall_signal = "strong_buy"
                strength = buy_ratio
            elif buy_ratio > 0.4:
                overall_signal = "buy"
                strength = buy_ratio
            elif sell_ratio > 0.6:
                overall_signal = "strong_sell"
                strength = sell_ratio
            elif sell_ratio > 0.4:
                overall_signal = "sell"
                strength = sell_ratio
            else:
                overall_signal = "neutral"
                strength = neutral_ratio

            # Confidence based on signal consensus
            max_ratio = max(buy_ratio, sell_ratio, neutral_ratio)
            confidence = max_ratio
        else:
            overall_signal = "neutral"
            strength = 0.0
            confidence = 0.0

        return {
            "success": True,
            "overall_signal": overall_signal,
            "strength": float(strength),
            "confidence": float(confidence),
            "scores": {
                "buy_score": float(buy_score),
                "sell_score": float(sell_score),
                "neutral_score": float(neutral_score),
                "total_weight": float(total_weight),
            },
            "signal_breakdown": signal_breakdown,
            "signal_count": len(signals),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Signal strength calculation failed: {e}")
        return {"success": False, "error": str(e)}


def assess_market_regime(data: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
    """
    STRAND Tool: Assess Current Market Regime (Trending/Ranging/Volatile)

    Args:
        data: OHLCV DataFrame
        lookback: Period for regime analysis

    Returns:
        Dictionary with market regime assessment
    """
    try:
        if len(data) < lookback:
            return {
                "success": True,
                "regime": "insufficient_data",
                "confidence": 0.0,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

        recent_data = data.tail(lookback)

        # Calculate regime indicators
        returns = recent_data["close"].pct_change().dropna()

        # Trend strength (ADX-like calculation)
        high_low = recent_data["high"] - recent_data["low"]
        high_close_prev = abs(recent_data["high"] - recent_data["close"].shift(1))
        low_close_prev = abs(recent_data["low"] - recent_data["close"].shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()

        # Directional movement
        up_move = recent_data["high"] - recent_data["high"].shift(1)
        down_move = recent_data["low"].shift(1) - recent_data["low"]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean().iloc[-1]

        # Volatility metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        rolling_vol = returns.rolling(window=10).std()
        vol_of_vol = rolling_vol.std()

        # Range metrics
        price_range = (recent_data["high"].max() - recent_data["low"].min()) / recent_data[
            "close"
        ].mean()
        avg_daily_range = ((recent_data["high"] - recent_data["low"]) / recent_data["close"]).mean()

        # Trend consistency
        trend_direction = 1 if recent_data["close"].iloc[-1] > recent_data["close"].iloc[0] else -1
        consistent_moves = sum(1 for r in returns if (r > 0) == (trend_direction > 0)) / len(
            returns
        )

        # Regime classification
        regime_scores = {"trending": 0.0, "ranging": 0.0, "volatile": 0.0}

        # Trending indicators
        if adx > 25:  # Strong trend
            regime_scores["trending"] += 0.4
        elif adx > 20:
            regime_scores["trending"] += 0.2

        if consistent_moves > 0.6:
            regime_scores["trending"] += 0.3

        if abs(returns.mean()) > 0.001:  # Directional bias
            regime_scores["trending"] += 0.2

        # Ranging indicators
        if adx < 20:  # Weak trend
            regime_scores["ranging"] += 0.3

        if price_range < 0.2:  # Narrow range
            regime_scores["ranging"] += 0.3

        if abs(returns.mean()) < 0.0005:  # No directional bias
            regime_scores["ranging"] += 0.2

        # Volatile indicators
        if volatility > 0.5:  # High volatility
            regime_scores["volatile"] += 0.4
        elif volatility > 0.3:
            regime_scores["volatile"] += 0.2

        if vol_of_vol > 0.01:  # Volatile volatility
            regime_scores["volatile"] += 0.3

        if avg_daily_range > 0.05:  # Large daily ranges
            regime_scores["volatile"] += 0.2

        # Determine regime
        dominant_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[dominant_regime]

        return {
            "success": True,
            "regime": dominant_regime,
            "confidence": float(confidence),
            "metrics": {
                "adx": float(adx) if not np.isnan(adx) else 0.0,
                "volatility": float(volatility),
                "price_range": float(price_range),
                "trend_consistency": float(consistent_moves),
                "avg_daily_range": float(avg_daily_range),
            },
            "regime_scores": {k: float(v) for k, v in regime_scores.items()},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Market regime assessment failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_risk_metrics(
    data: pd.DataFrame, signals: List[Dict[str, Any]], position_size: float = 1.0
) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Risk Metrics for Trading Signals

    Args:
        data: OHLCV DataFrame
        signals: List of trading signals
        position_size: Position size multiplier

    Returns:
        Dictionary with risk assessment metrics
    """
    try:
        current_price = data["close"].iloc[-1]
        returns = data["close"].pct_change().dropna()

        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        var_95 = returns.quantile(0.05)  # 5% VaR
        max_drawdown = self._calculate_max_drawdown(data["close"])

        # Signal-based risk assessment
        risk_signals = []
        conflicting_signals = 0

        buy_signals = [s for s in signals if s.get("signal") == "buy"]
        sell_signals = [s for s in signals if s.get("signal") == "sell"]

        # Check for conflicting signals
        if len(buy_signals) > 0 and len(sell_signals) > 0:
            conflicting_signals = min(len(buy_signals), len(sell_signals))
            risk_signals.append(
                {
                    "type": "conflicting_signals",
                    "severity": "medium",
                    "description": f"{conflicting_signals} conflicting signals detected",
                }
            )

        # Volatility risk
        if volatility > 0.6:
            risk_signals.append(
                {
                    "type": "high_volatility",
                    "severity": "high",
                    "description": f"High volatility: {volatility:.1%}",
                }
            )
        elif volatility > 0.4:
            risk_signals.append(
                {
                    "type": "elevated_volatility",
                    "severity": "medium",
                    "description": f"Elevated volatility: {volatility:.1%}",
                }
            )

        # Drawdown risk
        if max_drawdown < -0.2:
            risk_signals.append(
                {
                    "type": "high_drawdown",
                    "severity": "high",
                    "description": f"Recent max drawdown: {max_drawdown:.1%}",
                }
            )

        # Calculate position risk
        stop_loss_levels = []
        take_profit_levels = []

        for signal in signals:
            if "stop_loss" in signal:
                stop_loss_levels.append(signal["stop_loss"])
            if "target" in signal:
                take_profit_levels.append(signal["target"])

        # Risk-reward calculation
        if stop_loss_levels and take_profit_levels:
            avg_stop = np.mean(stop_loss_levels)
            avg_target = np.mean(take_profit_levels)

            risk_amount = abs(current_price - avg_stop) / current_price
            reward_amount = abs(avg_target - current_price) / current_price

            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        else:
            risk_reward_ratio = 0
            risk_amount = 0
            reward_amount = 0

        # Overall risk score (0-100, higher = riskier)
        risk_score = 0

        # Volatility component (0-40 points)
        risk_score += min(volatility * 80, 40)

        # Drawdown component (0-30 points)
        risk_score += min(abs(max_drawdown) * 150, 30)

        # Signal conflict component (0-20 points)
        if conflicting_signals > 0:
            risk_score += min(conflicting_signals * 10, 20)

        # Risk-reward component (0-10 points)
        if risk_reward_ratio < 1:
            risk_score += 10
        elif risk_reward_ratio < 2:
            risk_score += 5

        risk_score = min(risk_score, 100)

        # Risk level classification
        if risk_score >= 70:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "success": True,
            "risk_level": risk_level,
            "risk_score": float(risk_score),
            "metrics": {
                "volatility": float(volatility),
                "var_95": float(var_95),
                "max_drawdown": float(max_drawdown),
                "risk_reward_ratio": float(risk_reward_ratio),
                "position_risk": float(risk_amount * position_size),
                "potential_reward": float(reward_amount * position_size),
            },
            "risk_signals": risk_signals,
            "conflicting_signals": conflicting_signals,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Risk metrics calculation failed: {e}")
        return {"success": False, "error": str(e)}


def _calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown from price series"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def generate_trading_recommendation(
    data: pd.DataFrame, all_signals: List[Dict[str, Any]], risk_tolerance: str = "medium"
) -> Dict[str, Any]:
    """
    STRAND Tool: Generate Final Trading Recommendation

    Args:
        data: OHLCV DataFrame
        all_signals: Combined signals from all TA skills
        risk_tolerance: Risk tolerance level (low/medium/high)

    Returns:
        Dictionary with final trading recommendation
    """
    try:
        # Get signal strength analysis
        signal_strength = calculate_signal_strength(all_signals)

        # Get market regime
        market_regime = assess_market_regime(data)

        # Get risk metrics
        risk_metrics = calculate_risk_metrics(data, all_signals)

        current_price = data["close"].iloc[-1]

        # Risk tolerance thresholds
        risk_thresholds = {
            "low": {"max_risk_score": 30, "min_confidence": 0.7, "min_signals": 3},
            "medium": {"max_risk_score": 60, "min_confidence": 0.6, "min_signals": 2},
            "high": {"max_risk_score": 80, "min_confidence": 0.5, "min_signals": 1},
        }

        threshold = risk_thresholds.get(risk_tolerance, risk_thresholds["medium"])

        # Decision logic
        recommendation = {
            "action": "hold",
            "confidence": 0.0,
            "position_size": 0.0,
            "entry_price": current_price,
            "stop_loss": None,
            "take_profit": None,
            "reasoning": [],
        }

        if (
            not signal_strength["success"]
            or not market_regime["success"]
            or not risk_metrics["success"]
        ):
            recommendation["reasoning"].append("Insufficient data for analysis")
            return {
                "success": True,
                "recommendation": recommendation,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

        # Check if conditions are met for trading
        signal_count = signal_strength["signal_count"]
        confidence = signal_strength["confidence"]
        risk_score = risk_metrics["risk_score"]
        overall_signal = signal_strength["overall_signal"]

        # Risk checks
        if risk_score > threshold["max_risk_score"]:
            recommendation["reasoning"].append(f"Risk score too high: {risk_score:.0f}")
        elif confidence < threshold["min_confidence"]:
            recommendation["reasoning"].append(f"Confidence too low: {confidence:.1%}")
        elif signal_count < threshold["min_signals"]:
            recommendation["reasoning"].append(f"Insufficient signals: {signal_count}")
        else:
            # Generate recommendation
            if overall_signal in ["strong_buy", "buy"]:
                recommendation["action"] = "buy"
                recommendation["confidence"] = confidence

                # Position sizing based on confidence and risk
                base_size = (
                    0.1 if risk_tolerance == "low" else 0.2 if risk_tolerance == "medium" else 0.3
                )
                confidence_multiplier = confidence
                risk_multiplier = max(0.5, 1 - (risk_score / 100))

                recommendation["position_size"] = (
                    base_size * confidence_multiplier * risk_multiplier
                )

                # Set stop loss and take profit
                atr = data["high"].rolling(14).max() - data["low"].rolling(14).min()
                avg_atr = atr.mean()

                recommendation["stop_loss"] = current_price - (2 * avg_atr)
                recommendation["take_profit"] = current_price + (3 * avg_atr)

                recommendation["reasoning"].extend(
                    [
                        f"Strong {overall_signal} signal with {confidence:.1%} confidence",
                        f"Risk score acceptable: {risk_score:.0f}",
                        f"Market regime: {market_regime['regime']}",
                    ]
                )

            elif overall_signal in ["strong_sell", "sell"]:
                recommendation["action"] = "sell"
                recommendation["confidence"] = confidence

                # Position sizing for short
                base_size = (
                    0.1 if risk_tolerance == "low" else 0.2 if risk_tolerance == "medium" else 0.3
                )
                confidence_multiplier = confidence
                risk_multiplier = max(0.5, 1 - (risk_score / 100))

                recommendation["position_size"] = -(
                    base_size * confidence_multiplier * risk_multiplier
                )

                # Set stop loss and take profit for short
                atr = data["high"].rolling(14).max() - data["low"].rolling(14).min()
                avg_atr = atr.mean()

                recommendation["stop_loss"] = current_price + (2 * avg_atr)
                recommendation["take_profit"] = current_price - (3 * avg_atr)

                recommendation["reasoning"].extend(
                    [
                        f"Strong {overall_signal} signal with {confidence:.1%} confidence",
                        f"Risk score acceptable: {risk_score:.0f}",
                        f"Market regime: {market_regime['regime']}",
                    ]
                )
            else:
                recommendation["reasoning"].append(f"Neutral signal: {overall_signal}")

        return {
            "success": True,
            "recommendation": recommendation,
            "supporting_analysis": {
                "signal_strength": signal_strength,
                "market_regime": market_regime,
                "risk_metrics": risk_metrics,
            },
            "parameters": {"risk_tolerance": risk_tolerance},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Trading recommendation generation failed: {e}")
        return {"success": False, "error": str(e)}


def create_comprehensive_system_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for comprehensive technical analysis system

    Returns:
        List of tool specifications for STRAND framework
    """
    return [
        {
            "name": "calculate_signal_strength",
            "function": calculate_signal_strength,
            "description": "Calculate overall signal strength from multiple TA indicators",
            "parameters": {"signals": "List of signal dictionaries from various TA skills"},
            "category": "technical_analysis",
            "skill": "comprehensive_system",
        },
        {
            "name": "assess_market_regime",
            "function": assess_market_regime,
            "description": "Assess current market regime (trending/ranging/volatile)",
            "parameters": {"data": "OHLCV DataFrame", "lookback": "Period for regime analysis"},
            "category": "technical_analysis",
            "skill": "comprehensive_system",
        },
        {
            "name": "calculate_risk_metrics",
            "function": calculate_risk_metrics,
            "description": "Calculate comprehensive risk metrics for trading signals",
            "parameters": {
                "data": "OHLCV DataFrame",
                "signals": "List of trading signals",
                "position_size": "Position size multiplier",
            },
            "category": "technical_analysis",
            "skill": "comprehensive_system",
        },
        {
            "name": "generate_trading_recommendation",
            "function": generate_trading_recommendation,
            "description": "Generate final trading recommendation based on all TA analysis",
            "parameters": {
                "data": "OHLCV DataFrame",
                "all_signals": "Combined signals from all TA skills",
                "risk_tolerance": "Risk tolerance level (low/medium/high)",
            },
            "category": "technical_analysis",
            "skill": "comprehensive_system",
        },
    ]
