"""
Skill 2: Momentum and Volatility Indicators
STRAND tools for MACD, Bollinger Bands, ATR, and Stochastic Oscillator
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# TA-Lib imports
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Install with: conda install -c conda-forge ta-lib")

logger = logging.getLogger(__name__)


def calculate_macd(
    data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate MACD (Moving Average Convergence Divergence)

    Args:
        data: OHLCV DataFrame
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        Dictionary with MACD results and signals
    """
    try:
        if TALIB_AVAILABLE:
            macd_line, signal_line, histogram = talib.MACD(
                data["close"].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period,
            )
            macd_line = pd.Series(macd_line, index=data.index)
            signal_line = pd.Series(signal_line, index=data.index)
            histogram = pd.Series(histogram, index=data.index)
        else:
            # Custom MACD implementation
            ema_fast = data["close"].ewm(span=fast_period).mean()
            ema_slow = data["close"].ewm(span=slow_period).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line

        results = {
            "MACD": macd_line.tolist(),
            "MACD_Signal": signal_line.tolist(),
            "MACD_Histogram": histogram.tolist(),
        }

        signals = []

        # MACD crossover signals
        if (
            len(macd_line) > 1
            and not pd.isna(macd_line.iloc[-1])
            and not pd.isna(signal_line.iloc[-1])
        ):
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]

            if current_macd > current_signal and prev_macd <= prev_signal:
                signals.append(
                    {
                        "type": "macd_bullish_crossover",
                        "indicator": "MACD",
                        "signal": "buy",
                        "strength": "strong",
                        "macd_value": current_macd,
                        "signal_value": current_signal,
                    }
                )
            elif current_macd < current_signal and prev_macd >= prev_signal:
                signals.append(
                    {
                        "type": "macd_bearish_crossover",
                        "indicator": "MACD",
                        "signal": "sell",
                        "strength": "strong",
                        "macd_value": current_macd,
                        "signal_value": current_signal,
                    }
                )

        # Zero line crossover
        if len(macd_line) > 1 and not pd.isna(macd_line.iloc[-1]):
            current_macd = macd_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]

            if current_macd > 0 and prev_macd <= 0:
                signals.append(
                    {
                        "type": "macd_zero_line_bullish",
                        "indicator": "MACD",
                        "signal": "buy",
                        "strength": "medium",
                        "macd_value": current_macd,
                    }
                )
            elif current_macd < 0 and prev_macd >= 0:
                signals.append(
                    {
                        "type": "macd_zero_line_bearish",
                        "indicator": "MACD",
                        "signal": "sell",
                        "strength": "medium",
                        "macd_value": current_macd,
                    }
                )

        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "parameters": {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_bollinger_bands(
    data: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Bollinger Bands

    Args:
        data: OHLCV DataFrame
        period: Moving average period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Dictionary with Bollinger Bands results and signals
    """
    try:
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                data["close"].values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            upper = pd.Series(upper, index=data.index)
            middle = pd.Series(middle, index=data.index)
            lower = pd.Series(lower, index=data.index)
        else:
            # Custom Bollinger Bands implementation
            middle = data["close"].rolling(window=period).mean()
            std = data["close"].rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)

        # Calculate %B and bandwidth
        percent_b = (data["close"] - lower) / (upper - lower)
        bandwidth = (upper - lower) / middle

        results = {
            "BB_Upper": upper.tolist(),
            "BB_Middle": middle.tolist(),
            "BB_Lower": lower.tolist(),
            "BB_PercentB": percent_b.tolist(),
            "BB_Bandwidth": bandwidth.tolist(),
        }

        signals = []

        # Bollinger Band signals
        current_price = data["close"].iloc[-1]
        if not pd.isna(upper.iloc[-1]) and not pd.isna(lower.iloc[-1]):
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            current_percent_b = percent_b.iloc[-1]

            # Band touch signals
            if current_price >= current_upper:
                signals.append(
                    {
                        "type": "bb_upper_touch",
                        "indicator": "Bollinger Bands",
                        "signal": "sell",
                        "strength": "medium",
                        "price": current_price,
                        "upper_band": current_upper,
                        "percent_b": current_percent_b,
                    }
                )
            elif current_price <= current_lower:
                signals.append(
                    {
                        "type": "bb_lower_touch",
                        "indicator": "Bollinger Bands",
                        "signal": "buy",
                        "strength": "medium",
                        "price": current_price,
                        "lower_band": current_lower,
                        "percent_b": current_percent_b,
                    }
                )

            # Squeeze detection (low volatility)
            if len(bandwidth) > 20:
                current_bandwidth = bandwidth.iloc[-1]
                avg_bandwidth = bandwidth.iloc[-20:].mean()

                if current_bandwidth < avg_bandwidth * 0.5:
                    signals.append(
                        {
                            "type": "bb_squeeze",
                            "indicator": "Bollinger Bands",
                            "signal": "neutral",
                            "strength": "high",
                            "note": "Low volatility - potential breakout coming",
                            "bandwidth": current_bandwidth,
                            "avg_bandwidth": avg_bandwidth,
                        }
                    )

        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "parameters": {"period": period, "std_dev": std_dev},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_atr(data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Average True Range (ATR)

    Args:
        data: OHLCV DataFrame
        period: ATR period (default: 14)

    Returns:
        Dictionary with ATR results and volatility analysis
    """
    try:
        if TALIB_AVAILABLE:
            atr = pd.Series(
                talib.ATR(
                    data["high"].values, data["low"].values, data["close"].values, timeperiod=period
                ),
                index=data.index,
            )
        else:
            # Custom ATR implementation
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())

            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()

        results = {"ATR": atr.tolist()}
        signals = []

        # Volatility analysis
        if len(atr) > period:
            current_atr = atr.iloc[-1]
            avg_atr = atr.iloc[-period:].mean()

            if not pd.isna(current_atr) and not pd.isna(avg_atr):
                volatility_ratio = current_atr / avg_atr

                if volatility_ratio > 1.5:
                    signals.append(
                        {
                            "type": "high_volatility",
                            "indicator": "ATR",
                            "signal": "neutral",
                            "strength": "high",
                            "note": "High volatility detected",
                            "atr_value": current_atr,
                            "volatility_ratio": volatility_ratio,
                        }
                    )
                elif volatility_ratio < 0.7:
                    signals.append(
                        {
                            "type": "low_volatility",
                            "indicator": "ATR",
                            "signal": "neutral",
                            "strength": "medium",
                            "note": "Low volatility - potential breakout setup",
                            "atr_value": current_atr,
                            "volatility_ratio": volatility_ratio,
                        }
                    )

        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "parameters": {"period": period},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"ATR calculation failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_stochastic(
    data: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    overbought: float = 80.0,
    oversold: float = 20.0,
) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Stochastic Oscillator

    Args:
        data: OHLCV DataFrame
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
        overbought: Overbought threshold (default: 80)
        oversold: Oversold threshold (default: 20)

    Returns:
        Dictionary with Stochastic results and signals
    """
    try:
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(
                data["high"].values,
                data["low"].values,
                data["close"].values,
                fastk_period=k_period,
                slowk_period=d_period,
                slowd_period=d_period,
            )
            stoch_k = pd.Series(slowk, index=data.index)
            stoch_d = pd.Series(slowd, index=data.index)
        else:
            # Custom Stochastic implementation
            lowest_low = data["low"].rolling(window=k_period).min()
            highest_high = data["high"].rolling(window=k_period).max()

            k_percent = 100 * ((data["close"] - lowest_low) / (highest_high - lowest_low))
            stoch_k = k_percent.rolling(window=d_period).mean()
            stoch_d = stoch_k.rolling(window=d_period).mean()

        results = {"Stoch_K": stoch_k.tolist(), "Stoch_D": stoch_d.tolist()}

        signals = []

        # Stochastic signals
        if not pd.isna(stoch_k.iloc[-1]) and not pd.isna(stoch_d.iloc[-1]):
            current_k = stoch_k.iloc[-1]
            current_d = stoch_d.iloc[-1]

            # Overbought/Oversold signals
            if current_k > overbought and current_d > overbought:
                signals.append(
                    {
                        "type": "stoch_overbought",
                        "indicator": "Stochastic",
                        "signal": "sell",
                        "strength": "medium",
                        "stoch_k": current_k,
                        "stoch_d": current_d,
                        "threshold": overbought,
                    }
                )
            elif current_k < oversold and current_d < oversold:
                signals.append(
                    {
                        "type": "stoch_oversold",
                        "indicator": "Stochastic",
                        "signal": "buy",
                        "strength": "medium",
                        "stoch_k": current_k,
                        "stoch_d": current_d,
                        "threshold": oversold,
                    }
                )

            # %K and %D crossover signals
            if len(stoch_k) > 1 and len(stoch_d) > 1:
                prev_k = stoch_k.iloc[-2]
                prev_d = stoch_d.iloc[-2]

                if current_k > current_d and prev_k <= prev_d:
                    strength = "strong" if current_k < oversold else "medium"
                    signals.append(
                        {
                            "type": "stoch_bullish_crossover",
                            "indicator": "Stochastic",
                            "signal": "buy",
                            "strength": strength,
                            "stoch_k": current_k,
                            "stoch_d": current_d,
                        }
                    )
                elif current_k < current_d and prev_k >= prev_d:
                    strength = "strong" if current_k > overbought else "medium"
                    signals.append(
                        {
                            "type": "stoch_bearish_crossover",
                            "indicator": "Stochastic",
                            "signal": "sell",
                            "strength": strength,
                            "stoch_k": current_k,
                            "stoch_d": current_d,
                        }
                    )

        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "parameters": {
                "k_period": k_period,
                "d_period": d_period,
                "overbought": overbought,
                "oversold": oversold,
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Stochastic calculation failed: {e}")
        return {"success": False, "error": str(e)}


def analyze_momentum_volatility(
    data: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    stoch_k: int = 14,
    stoch_d: int = 3,
) -> Dict[str, Any]:
    """
    STRAND Tool: Comprehensive Momentum and Volatility Analysis

    Args:
        data: OHLCV DataFrame
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        bb_period: Bollinger Bands period
        bb_std: Bollinger Bands standard deviation
        atr_period: ATR period
        stoch_k: Stochastic %K period
        stoch_d: Stochastic %D period

    Returns:
        Comprehensive momentum and volatility analysis
    """
    try:
        # Calculate all indicators
        macd_result = calculate_macd(data, macd_fast, macd_slow, macd_signal)
        bb_result = calculate_bollinger_bands(data, bb_period, bb_std)
        atr_result = calculate_atr(data, atr_period)
        stoch_result = calculate_stochastic(data, stoch_k, stoch_d)

        # Combine results
        all_indicators = {}
        all_signals = []

        for result in [macd_result, bb_result, atr_result, stoch_result]:
            if result["success"]:
                all_indicators.update(result["indicators"])
                all_signals.extend(result["signals"])

        # Generate momentum analysis
        momentum_signals = [s for s in all_signals if s["signal"] in ["buy", "sell"]]
        buy_momentum = len([s for s in momentum_signals if s["signal"] == "buy"])
        sell_momentum = len([s for s in momentum_signals if s["signal"] == "sell"])

        if buy_momentum > sell_momentum:
            momentum_direction = "bullish"
        elif sell_momentum > buy_momentum:
            momentum_direction = "bearish"
        else:
            momentum_direction = "neutral"

        # Volatility analysis
        volatility_signals = [s for s in all_signals if "volatility" in s["type"]]
        high_vol_signals = len([s for s in volatility_signals if "high" in s["type"]])
        low_vol_signals = len([s for s in volatility_signals if "low" in s["type"]])

        if high_vol_signals > 0:
            volatility_state = "high"
        elif low_vol_signals > 0:
            volatility_state = "low"
        else:
            volatility_state = "normal"

        # Calculate confidence score
        total_signals = len(momentum_signals)
        confidence_score = min(total_signals / 6.0, 1.0)  # Max confidence at 6+ signals

        return {
            "success": True,
            "indicators": all_indicators,
            "signals": all_signals,
            "analysis": {
                "momentum_direction": momentum_direction,
                "volatility_state": volatility_state,
                "buy_signals": buy_momentum,
                "sell_signals": sell_momentum,
                "confidence_score": confidence_score,
                "signal_strength": "strong"
                if confidence_score > 0.7
                else "medium"
                if confidence_score > 0.4
                else "weak",
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Momentum volatility analysis failed: {e}")
        return {"success": False, "error": str(e)}


def create_momentum_volatility_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for momentum and volatility indicators

    Returns:
        List of tool specifications for STRAND framework
    """
    return [
        {
            "name": "calculate_macd",
            "function": calculate_macd,
            "description": "Calculate MACD with crossover and zero-line signals",
            "parameters": {
                "data": "OHLCV DataFrame",
                "fast_period": "Fast EMA period (default: 12)",
                "slow_period": "Slow EMA period (default: 26)",
                "signal_period": "Signal line period (default: 9)",
            },
            "category": "technical_analysis",
            "skill": "momentum_volatility",
        },
        {
            "name": "calculate_bollinger_bands",
            "function": calculate_bollinger_bands,
            "description": "Calculate Bollinger Bands with squeeze detection",
            "parameters": {
                "data": "OHLCV DataFrame",
                "period": "Moving average period (default: 20)",
                "std_dev": "Standard deviation multiplier (default: 2.0)",
            },
            "category": "technical_analysis",
            "skill": "momentum_volatility",
        },
        {
            "name": "calculate_atr",
            "function": calculate_atr,
            "description": "Calculate ATR with volatility analysis",
            "parameters": {"data": "OHLCV DataFrame", "period": "ATR period (default: 14)"},
            "category": "technical_analysis",
            "skill": "momentum_volatility",
        },
        {
            "name": "calculate_stochastic",
            "function": calculate_stochastic,
            "description": "Calculate Stochastic Oscillator with crossover signals",
            "parameters": {
                "data": "OHLCV DataFrame",
                "k_period": "%K period (default: 14)",
                "d_period": "%D period (default: 3)",
                "overbought": "Overbought threshold (default: 80)",
                "oversold": "Oversold threshold (default: 20)",
            },
            "category": "technical_analysis",
            "skill": "momentum_volatility",
        },
        {
            "name": "analyze_momentum_volatility",
            "function": analyze_momentum_volatility,
            "description": "Comprehensive momentum and volatility analysis using MACD, Bollinger Bands, ATR, and Stochastic",
            "parameters": {
                "data": "OHLCV DataFrame",
                "macd_fast": "MACD fast period",
                "macd_slow": "MACD slow period",
                "macd_signal": "MACD signal period",
                "bb_period": "Bollinger Bands period",
                "bb_std": "Bollinger Bands standard deviation",
                "atr_period": "ATR period",
                "stoch_k": "Stochastic %K period",
                "stoch_d": "Stochastic %D period",
            },
            "category": "technical_analysis",
            "skill": "momentum_volatility",
        },
    ]
