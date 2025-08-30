"""
Skill 4: Support and Resistance Level Detection
STRAND tools for automated support/resistance detection using pivot points, psychological levels, and Fibonacci retracements
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


def detect_pivot_points(data: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Pivot Points (Local Highs and Lows)

    Args:
        data: OHLCV DataFrame
        window: Window size for pivot detection (default: 5)

    Returns:
        Dictionary with pivot points and potential S/R levels
    """
    try:
        highs = data["high"].values
        lows = data["low"].values

        # Find local maxima and minima
        high_peaks = argrelextrema(highs, np.greater, order=window)[0]
        low_valleys = argrelextrema(lows, np.less, order=window)[0]

        # Create pivot points
        pivot_highs = []
        pivot_lows = []

        for idx in high_peaks:
            pivot_highs.append(
                {
                    "index": int(idx),
                    "date": data.index[idx].isoformat()
                    if hasattr(data.index[idx], "isoformat")
                    else str(data.index[idx]),
                    "price": float(highs[idx]),
                    "type": "resistance",
                }
            )

        for idx in low_valleys:
            pivot_lows.append(
                {
                    "index": int(idx),
                    "date": data.index[idx].isoformat()
                    if hasattr(data.index[idx], "isoformat")
                    else str(data.index[idx]),
                    "price": float(lows[idx]),
                    "type": "support",
                }
            )

        # Combine and sort by recency
        all_pivots = pivot_highs + pivot_lows
        all_pivots.sort(key=lambda x: x["index"], reverse=True)

        # Get recent pivots (last 20)
        recent_pivots = all_pivots[:20]

        results = {
            "pivot_highs": pivot_highs,
            "pivot_lows": pivot_lows,
            "recent_pivots": recent_pivots,
            "total_pivots": len(all_pivots),
        }

        signals = []
        current_price = data["close"].iloc[-1]

        # Check proximity to recent pivot levels
        for pivot in recent_pivots[:10]:  # Check last 10 pivots
            distance_pct = abs(current_price - pivot["price"]) / current_price * 100

            if distance_pct < 1.0:  # Within 1% of pivot level
                signals.append(
                    {
                        "type": f"near_{pivot['type']}_level",
                        "indicator": "Pivot Points",
                        "signal": "neutral",
                        "strength": "high",
                        "price": current_price,
                        "level": pivot["price"],
                        "distance_pct": distance_pct,
                        "note": f"Price near {pivot['type']} level at {pivot['price']:.2f}",
                    }
                )

        return {
            "success": True,
            "levels": results,
            "signals": signals,
            "parameters": {"window": window},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Pivot point detection failed: {e}")
        return {"success": False, "error": str(e)}


def detect_psychological_levels(
    data: pd.DataFrame, round_numbers: List[int] = None
) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Psychological Levels (Round Numbers)

    Args:
        data: OHLCV DataFrame
        round_numbers: List of round number intervals (default: [1, 5, 10, 50, 100])

    Returns:
        Dictionary with psychological levels analysis
    """
    try:
        if round_numbers is None:
            round_numbers = [1, 5, 10, 50, 100]

        current_price = data["close"].iloc[-1]
        price_range = data["close"].max() - data["close"].min()

        psychological_levels = []

        # Find relevant round numbers based on price range
        for interval in round_numbers:
            if (
                interval <= price_range * 0.1
            ):  # Only consider intervals that make sense for the price range
                # Find round levels above and below current price
                lower_level = (current_price // interval) * interval
                upper_level = lower_level + interval

                psychological_levels.extend(
                    [
                        {
                            "level": float(lower_level),
                            "type": "psychological",
                            "interval": interval,
                            "distance_from_current": abs(current_price - lower_level),
                        },
                        {
                            "level": float(upper_level),
                            "type": "psychological",
                            "interval": interval,
                            "distance_from_current": abs(current_price - upper_level),
                        },
                    ]
                )

        # Sort by distance from current price
        psychological_levels.sort(key=lambda x: x["distance_from_current"])

        # Remove duplicates and keep closest levels
        unique_levels = []
        seen_levels = set()

        for level in psychological_levels:
            level_key = round(level["level"], 2)
            if level_key not in seen_levels:
                unique_levels.append(level)
                seen_levels.add(level_key)

        # Keep only closest 10 levels
        closest_levels = unique_levels[:10]

        signals = []

        # Check proximity to psychological levels
        for level in closest_levels:
            distance_pct = abs(current_price - level["level"]) / current_price * 100

            if distance_pct < 2.0:  # Within 2% of psychological level
                level_type = "resistance" if level["level"] > current_price else "support"
                signals.append(
                    {
                        "type": f"near_psychological_{level_type}",
                        "indicator": "Psychological Levels",
                        "signal": "neutral",
                        "strength": "medium",
                        "price": current_price,
                        "level": level["level"],
                        "distance_pct": distance_pct,
                        "interval": level["interval"],
                        "note": f"Price near psychological {level_type} at {level['level']:.2f}",
                    }
                )

        return {
            "success": True,
            "levels": {"psychological_levels": closest_levels, "current_price": current_price},
            "signals": signals,
            "parameters": {"round_numbers": round_numbers},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Psychological levels detection failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_fibonacci_retracements(data: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Fibonacci Retracement Levels

    Args:
        data: OHLCV DataFrame
        lookback: Period to find swing high/low (default: 50)

    Returns:
        Dictionary with Fibonacci retracement levels
    """
    try:
        # Find swing high and low in lookback period
        recent_data = data.tail(lookback)
        swing_high = recent_data["high"].max()
        swing_low = recent_data["low"].min()
        swing_high_idx = recent_data["high"].idxmax()
        swing_low_idx = recent_data["low"].idxmin()

        # Determine trend direction
        if swing_high_idx > swing_low_idx:
            trend = "uptrend"
            range_price = swing_high - swing_low
        else:
            trend = "downtrend"
            range_price = swing_high - swing_low

        # Fibonacci ratios
        fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fib_levels = []

        for ratio in fib_ratios:
            if trend == "uptrend":
                level = swing_high - (range_price * ratio)
            else:
                level = swing_low + (range_price * ratio)

            fib_levels.append(
                {"ratio": ratio, "level": float(level), "type": "fibonacci", "trend": trend}
            )

        current_price = data["close"].iloc[-1]
        signals = []

        # Check proximity to Fibonacci levels
        for fib in fib_levels:
            distance_pct = abs(current_price - fib["level"]) / current_price * 100

            if distance_pct < 1.5:  # Within 1.5% of Fibonacci level
                level_type = "resistance" if fib["level"] > current_price else "support"
                strength = "strong" if fib["ratio"] in [0.382, 0.5, 0.618] else "medium"

                signals.append(
                    {
                        "type": f"near_fibonacci_{level_type}",
                        "indicator": "Fibonacci",
                        "signal": "neutral",
                        "strength": strength,
                        "price": current_price,
                        "level": fib["level"],
                        "ratio": fib["ratio"],
                        "distance_pct": distance_pct,
                        "trend": trend,
                        "note": f"Price near {fib['ratio']:.1%} Fibonacci {level_type} at {fib['level']:.2f}",
                    }
                )

        return {
            "success": True,
            "levels": {
                "fibonacci_levels": fib_levels,
                "swing_high": float(swing_high),
                "swing_low": float(swing_low),
                "trend": trend,
                "range": float(range_price),
            },
            "signals": signals,
            "parameters": {"lookback": lookback},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Fibonacci retracement calculation failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_level_strength(
    data: pd.DataFrame, levels: List[Dict], touch_tolerance: float = 0.02, min_touches: int = 2
) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Support/Resistance Level Strength

    Args:
        data: OHLCV DataFrame
        levels: List of level dictionaries with 'level' and 'type' keys
        touch_tolerance: Price tolerance for level touches (default: 2%)
        min_touches: Minimum touches to consider level valid (default: 2)

    Returns:
        Dictionary with level strength analysis
    """
    try:
        strengthened_levels = []

        for level_info in levels:
            level_price = level_info["level"]
            level_type = level_info.get("type", "unknown")

            touches = 0
            bounces = 0
            breaks = 0
            volume_at_touches = []

            # Check each candle for touches
            for i in range(len(data)):
                candle_high = data["high"].iloc[i]
                candle_low = data["low"].iloc[i]
                candle_close = data["close"].iloc[i]
                candle_volume = data["volume"].iloc[i]

                # Check if price touched the level
                tolerance = level_price * touch_tolerance

                if level_type == "support" or level_type == "psychological":
                    # For support, check if low touched level
                    if abs(candle_low - level_price) <= tolerance:
                        touches += 1
                        volume_at_touches.append(candle_volume)

                        # Check if it bounced (closed above level)
                        if candle_close > level_price:
                            bounces += 1
                        else:
                            breaks += 1

                elif level_type == "resistance":
                    # For resistance, check if high touched level
                    if abs(candle_high - level_price) <= tolerance:
                        touches += 1
                        volume_at_touches.append(candle_volume)

                        # Check if it bounced (closed below level)
                        if candle_close < level_price:
                            bounces += 1
                        else:
                            breaks += 1

            # Calculate strength metrics
            if touches >= min_touches:
                bounce_rate = bounces / touches if touches > 0 else 0
                avg_volume = np.mean(volume_at_touches) if volume_at_touches else 0
                overall_avg_volume = data["volume"].mean()
                volume_ratio = avg_volume / overall_avg_volume if overall_avg_volume > 0 else 1

                # Strength score (0-100)
                strength_score = min(
                    100,
                    (
                        (touches * 15)
                        + (bounce_rate * 40)  # More touches = stronger
                        + (min(volume_ratio, 2) * 20)  # Higher bounce rate = stronger
                        + (  # Higher volume = stronger (capped at 2x)
                            25 if breaks == 0 else max(0, 25 - breaks * 5)
                        )  # Fewer breaks = stronger
                    ),
                )

                strengthened_levels.append(
                    {
                        **level_info,
                        "touches": touches,
                        "bounces": bounces,
                        "breaks": breaks,
                        "bounce_rate": bounce_rate,
                        "avg_volume_ratio": volume_ratio,
                        "strength_score": strength_score,
                        "strength_rating": (
                            "very_strong"
                            if strength_score >= 80
                            else "strong"
                            if strength_score >= 60
                            else "medium"
                            if strength_score >= 40
                            else "weak"
                        ),
                    }
                )

        # Sort by strength score
        strengthened_levels.sort(key=lambda x: x["strength_score"], reverse=True)

        # Generate signals for strong levels
        signals = []
        current_price = data["close"].iloc[-1]

        for level in strengthened_levels[:5]:  # Check top 5 strongest levels
            if level["strength_score"] >= 60:  # Only strong levels
                distance_pct = abs(current_price - level["level"]) / current_price * 100

                if distance_pct < 3.0:  # Within 3% of strong level
                    level_type = level["type"]
                    if level_type == "resistance" and current_price < level["level"]:
                        signals.append(
                            {
                                "type": "approaching_strong_resistance",
                                "indicator": "Level Strength",
                                "signal": "sell",
                                "strength": level["strength_rating"],
                                "price": current_price,
                                "level": level["level"],
                                "strength_score": level["strength_score"],
                                "touches": level["touches"],
                                "bounce_rate": level["bounce_rate"],
                                "note": f"Approaching strong resistance at {level['level']:.2f}",
                            }
                        )
                    elif level_type == "support" and current_price > level["level"]:
                        signals.append(
                            {
                                "type": "approaching_strong_support",
                                "indicator": "Level Strength",
                                "signal": "buy",
                                "strength": level["strength_rating"],
                                "price": current_price,
                                "level": level["level"],
                                "strength_score": level["strength_score"],
                                "touches": level["touches"],
                                "bounce_rate": level["bounce_rate"],
                                "note": f"Approaching strong support at {level['level']:.2f}",
                            }
                        )

        return {
            "success": True,
            "levels": strengthened_levels,
            "signals": signals,
            "parameters": {"touch_tolerance": touch_tolerance, "min_touches": min_touches},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Level strength calculation failed: {e}")
        return {"success": False, "error": str(e)}


def detect_support_resistance_flips(
    data: pd.DataFrame, levels: List[Dict], flip_tolerance: float = 0.03
) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Support/Resistance Level Flips

    Args:
        data: OHLCV DataFrame
        levels: List of level dictionaries
        flip_tolerance: Price tolerance for flip detection (default: 3%)

    Returns:
        Dictionary with flip analysis
    """
    try:
        flipped_levels = []
        signals = []
        current_price = data["close"].iloc[-1]

        for level_info in levels:
            level_price = level_info["level"]
            original_type = level_info.get("type", "unknown")

            # Look for price action that suggests a flip
            recent_data = data.tail(20)  # Last 20 periods

            flip_detected = False
            flip_type = None

            # Check for support-to-resistance flip
            if original_type == "support":
                # Price was above support, then broke below, now resistance
                above_count = 0
                below_count = 0

                for i in range(len(recent_data)):
                    close_price = recent_data["close"].iloc[i]
                    tolerance = level_price * flip_tolerance

                    if close_price > level_price + tolerance:
                        above_count += 1
                    elif close_price < level_price - tolerance:
                        below_count += 1

                # If we see both above and below action, and currently below
                if above_count > 0 and below_count > 0 and current_price < level_price:
                    flip_detected = True
                    flip_type = "support_to_resistance"

            # Check for resistance-to-support flip
            elif original_type == "resistance":
                # Price was below resistance, then broke above, now support
                above_count = 0
                below_count = 0

                for i in range(len(recent_data)):
                    close_price = recent_data["close"].iloc[i]
                    tolerance = level_price * flip_tolerance

                    if close_price > level_price + tolerance:
                        above_count += 1
                    elif close_price < level_price - tolerance:
                        below_count += 1

                # If we see both above and below action, and currently above
                if above_count > 0 and below_count > 0 and current_price > level_price:
                    flip_detected = True
                    flip_type = "resistance_to_support"

            if flip_detected:
                new_type = "resistance" if flip_type == "support_to_resistance" else "support"

                flipped_levels.append(
                    {
                        **level_info,
                        "original_type": original_type,
                        "new_type": new_type,
                        "flip_type": flip_type,
                        "flip_detected": True,
                    }
                )

                # Generate flip signal
                distance_pct = abs(current_price - level_price) / current_price * 100

                if distance_pct < 2.0:  # Within 2% of flipped level
                    signal_direction = "buy" if new_type == "support" else "sell"

                    signals.append(
                        {
                            "type": f"level_flip_{flip_type}",
                            "indicator": "Support/Resistance Flip",
                            "signal": signal_direction,
                            "strength": "strong",
                            "price": current_price,
                            "level": level_price,
                            "original_type": original_type,
                            "new_type": new_type,
                            "distance_pct": distance_pct,
                            "note": f"Level flipped from {original_type} to {new_type} at {level_price:.2f}",
                        }
                    )

        return {
            "success": True,
            "flipped_levels": flipped_levels,
            "signals": signals,
            "parameters": {"flip_tolerance": flip_tolerance},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Support/resistance flip detection failed: {e}")
        return {"success": False, "error": str(e)}


def analyze_support_resistance(
    data: pd.DataFrame, pivot_window: int = 5, lookback: int = 50, min_touches: int = 2
) -> Dict[str, Any]:
    """
    STRAND Tool: Comprehensive Support and Resistance Analysis

    Args:
        data: OHLCV DataFrame
        pivot_window: Window for pivot point detection
        lookback: Lookback period for analysis
        min_touches: Minimum touches for level validation

    Returns:
        Comprehensive support and resistance analysis
    """
    try:
        # Get all types of levels
        pivot_result = detect_pivot_points(data, pivot_window)
        psychological_result = detect_psychological_levels(data)
        fibonacci_result = calculate_fibonacci_retracements(data, lookback)

        # Combine all levels
        all_levels = []

        if pivot_result["success"]:
            for pivot in pivot_result["levels"]["recent_pivots"]:
                all_levels.append(
                    {
                        "level": pivot["price"],
                        "type": pivot["type"],
                        "source": "pivot",
                        "date": pivot["date"],
                    }
                )

        if psychological_result["success"]:
            for level in psychological_result["levels"]["psychological_levels"]:
                level_type = "resistance" if level["level"] > data["close"].iloc[-1] else "support"
                all_levels.append(
                    {
                        "level": level["level"],
                        "type": level_type,
                        "source": "psychological",
                        "interval": level["interval"],
                    }
                )

        if fibonacci_result["success"]:
            for fib in fibonacci_result["levels"]["fibonacci_levels"]:
                level_type = "resistance" if fib["level"] > data["close"].iloc[-1] else "support"
                all_levels.append(
                    {
                        "level": fib["level"],
                        "type": level_type,
                        "source": "fibonacci",
                        "ratio": fib["ratio"],
                    }
                )

        # Calculate level strength
        strength_result = calculate_level_strength(data, all_levels, min_touches=min_touches)

        # Detect flips
        flip_result = detect_support_resistance_flips(data, all_levels)

        # Combine all signals
        all_signals = []
        for result in [
            pivot_result,
            psychological_result,
            fibonacci_result,
            strength_result,
            flip_result,
        ]:
            if result["success"] and "signals" in result:
                all_signals.extend(result["signals"])

        # Analysis summary
        strong_levels = [l for l in strength_result["levels"] if l.get("strength_score", 0) >= 60]
        current_price = data["close"].iloc[-1]

        nearby_support = [
            l for l in strong_levels if l["type"] == "support" and l["level"] < current_price
        ]
        nearby_resistance = [
            l for l in strong_levels if l["type"] == "resistance" and l["level"] > current_price
        ]

        # Sort by distance from current price
        nearby_support.sort(key=lambda x: current_price - x["level"])
        nearby_resistance.sort(key=lambda x: x["level"] - current_price)

        return {
            "success": True,
            "levels": {
                "all_levels": all_levels,
                "strong_levels": strong_levels,
                "nearby_support": nearby_support[:3],  # Closest 3 support levels
                "nearby_resistance": nearby_resistance[:3],  # Closest 3 resistance levels
                "flipped_levels": flip_result.get("flipped_levels", []),
            },
            "signals": all_signals,
            "analysis": {
                "total_levels": len(all_levels),
                "strong_levels_count": len(strong_levels),
                "nearest_support": nearby_support[0]["level"] if nearby_support else None,
                "nearest_resistance": nearby_resistance[0]["level"] if nearby_resistance else None,
                "current_price": current_price,
            },
            "parameters": {
                "pivot_window": pivot_window,
                "lookback": lookback,
                "min_touches": min_touches,
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Support/resistance analysis failed: {e}")
        return {"success": False, "error": str(e)}


def create_support_resistance_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for support and resistance analysis

    Returns:
        List of tool specifications for STRAND framework
    """
    return [
        {
            "name": "detect_pivot_points",
            "function": detect_pivot_points,
            "description": "Detect pivot points (local highs and lows) for S/R level identification",
            "parameters": {
                "data": "OHLCV DataFrame",
                "window": "Window size for pivot detection (default: 5)",
            },
            "category": "technical_analysis",
            "skill": "support_resistance",
        },
        {
            "name": "detect_psychological_levels",
            "function": detect_psychological_levels,
            "description": "Detect psychological levels based on round numbers",
            "parameters": {
                "data": "OHLCV DataFrame",
                "round_numbers": "List of round number intervals",
            },
            "category": "technical_analysis",
            "skill": "support_resistance",
        },
        {
            "name": "calculate_fibonacci_retracements",
            "function": calculate_fibonacci_retracements,
            "description": "Calculate Fibonacci retracement levels from recent swing high/low",
            "parameters": {
                "data": "OHLCV DataFrame",
                "lookback": "Period to find swing high/low (default: 50)",
            },
            "category": "technical_analysis",
            "skill": "support_resistance",
        },
        {
            "name": "calculate_level_strength",
            "function": calculate_level_strength,
            "description": "Calculate strength of S/R levels based on touches, bounces, and volume",
            "parameters": {
                "data": "OHLCV DataFrame",
                "levels": "List of level dictionaries",
                "touch_tolerance": "Price tolerance for level touches (default: 0.02)",
                "min_touches": "Minimum touches to consider level valid (default: 2)",
            },
            "category": "technical_analysis",
            "skill": "support_resistance",
        },
        {
            "name": "detect_support_resistance_flips",
            "function": detect_support_resistance_flips,
            "description": "Detect when support becomes resistance or vice versa",
            "parameters": {
                "data": "OHLCV DataFrame",
                "levels": "List of level dictionaries",
                "flip_tolerance": "Price tolerance for flip detection (default: 0.03)",
            },
            "category": "technical_analysis",
            "skill": "support_resistance",
        },
        {
            "name": "analyze_support_resistance",
            "function": analyze_support_resistance,
            "description": "Comprehensive S/R analysis using pivots, psychological levels, and Fibonacci",
            "parameters": {
                "data": "OHLCV DataFrame",
                "pivot_window": "Window for pivot detection",
                "lookback": "Lookback period for analysis",
                "min_touches": "Minimum touches for level validation",
            },
            "category": "technical_analysis",
            "skill": "support_resistance",
        },
    ]
