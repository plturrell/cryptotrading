"""
Skill 3: Volume-Based Indicators and Analysis
STRAND tools for OBV, VWAP, A/D Line, and MFI
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


def calculate_obv(data: pd.DataFrame) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate On-Balance Volume (OBV)

    Args:
        data: OHLCV DataFrame

    Returns:
        Dictionary with OBV results and volume trend analysis
    """
    try:
        if TALIB_AVAILABLE:
            obv = pd.Series(
                talib.OBV(data["close"].values, data["volume"].values), index=data.index
            )
        else:
            # Custom OBV implementation
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = data["volume"].iloc[0]

            for i in range(1, len(data)):
                if data["close"].iloc[i] > data["close"].iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + data["volume"].iloc[i]
                elif data["close"].iloc[i] < data["close"].iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - data["volume"].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]

        results = {"OBV": obv.tolist()}
        signals = []

        # OBV trend analysis
        if len(obv) >= 20:
            obv_ma = obv.rolling(window=20).mean()
            current_obv = obv.iloc[-1]
            current_ma = obv_ma.iloc[-1]

            if not pd.isna(current_ma):
                if current_obv > current_ma:
                    signals.append(
                        {
                            "type": "obv_bullish_trend",
                            "indicator": "OBV",
                            "signal": "buy",
                            "strength": "medium",
                            "obv_value": current_obv,
                            "obv_ma": current_ma,
                        }
                    )
                elif current_obv < current_ma:
                    signals.append(
                        {
                            "type": "obv_bearish_trend",
                            "indicator": "OBV",
                            "signal": "sell",
                            "strength": "medium",
                            "obv_value": current_obv,
                            "obv_ma": current_ma,
                        }
                    )

        # OBV divergence detection
        if len(obv) >= 10 and len(data) >= 10:
            recent_obv = obv.iloc[-5:]
            recent_price = data["close"].iloc[-5:]

            obv_trend = recent_obv.iloc[-1] - recent_obv.iloc[0]
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]

            if price_trend > 0 and obv_trend < 0:
                signals.append(
                    {
                        "type": "obv_bearish_divergence",
                        "indicator": "OBV",
                        "signal": "sell",
                        "strength": "strong",
                        "note": "Price rising but volume declining",
                    }
                )
            elif price_trend < 0 and obv_trend > 0:
                signals.append(
                    {
                        "type": "obv_bullish_divergence",
                        "indicator": "OBV",
                        "signal": "buy",
                        "strength": "strong",
                        "note": "Price falling but volume accumulating",
                    }
                )

        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"OBV calculation failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_vwap(data: pd.DataFrame) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Volume Weighted Average Price (VWAP)

    Args:
        data: OHLCV DataFrame

    Returns:
        Dictionary with VWAP results and price-volume analysis
    """
    try:
        # Calculate typical price
        typical_price = (data["high"] + data["low"] + data["close"]) / 3

        # Calculate VWAP
        vwap = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()

        results = {"VWAP": vwap.tolist()}
        signals = []

        # VWAP signals
        current_price = data["close"].iloc[-1]
        current_vwap = vwap.iloc[-1]

        if not pd.isna(current_vwap):
            price_vs_vwap = (current_price - current_vwap) / current_vwap * 100

            if current_price > current_vwap:
                signals.append(
                    {
                        "type": "price_above_vwap",
                        "indicator": "VWAP",
                        "signal": "buy",
                        "strength": "medium",
                        "price": current_price,
                        "vwap": current_vwap,
                        "deviation_pct": price_vs_vwap,
                    }
                )
            elif current_price < current_vwap:
                signals.append(
                    {
                        "type": "price_below_vwap",
                        "indicator": "VWAP",
                        "signal": "sell",
                        "strength": "medium",
                        "price": current_price,
                        "vwap": current_vwap,
                        "deviation_pct": price_vs_vwap,
                    }
                )

            # Strong deviation signals
            if abs(price_vs_vwap) > 2.0:
                signal_type = (
                    "strong_deviation_above" if price_vs_vwap > 0 else "strong_deviation_below"
                )
                signal_direction = "sell" if price_vs_vwap > 0 else "buy"  # Mean reversion

                signals.append(
                    {
                        "type": signal_type,
                        "indicator": "VWAP",
                        "signal": signal_direction,
                        "strength": "strong",
                        "note": "Strong deviation from VWAP - potential mean reversion",
                        "deviation_pct": price_vs_vwap,
                    }
                )

        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"VWAP calculation failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_ad_line(data: pd.DataFrame) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Accumulation/Distribution Line

    Args:
        data: OHLCV DataFrame

    Returns:
        Dictionary with A/D Line results and accumulation/distribution analysis
    """
    try:
        if TALIB_AVAILABLE:
            ad_line = pd.Series(
                talib.AD(
                    data["high"].values,
                    data["low"].values,
                    data["close"].values,
                    data["volume"].values,
                ),
                index=data.index,
            )
        else:
            # Custom A/D Line implementation
            money_flow_multiplier = (
                (data["close"] - data["low"]) - (data["high"] - data["close"])
            ) / (data["high"] - data["low"])
            money_flow_multiplier = money_flow_multiplier.fillna(0)  # Handle division by zero
            money_flow_volume = money_flow_multiplier * data["volume"]
            ad_line = money_flow_volume.cumsum()

        results = {"AD_Line": ad_line.tolist()}
        signals = []

        # A/D Line trend analysis
        if len(ad_line) >= 20:
            ad_ma = ad_line.rolling(window=20).mean()
            current_ad = ad_line.iloc[-1]
            current_ma = ad_ma.iloc[-1]

            if not pd.isna(current_ma):
                if current_ad > current_ma:
                    signals.append(
                        {
                            "type": "ad_accumulation",
                            "indicator": "A/D Line",
                            "signal": "buy",
                            "strength": "medium",
                            "ad_value": current_ad,
                            "ad_ma": current_ma,
                        }
                    )
                elif current_ad < current_ma:
                    signals.append(
                        {
                            "type": "ad_distribution",
                            "indicator": "A/D Line",
                            "signal": "sell",
                            "strength": "medium",
                            "ad_value": current_ad,
                            "ad_ma": current_ma,
                        }
                    )

        # A/D Line divergence detection
        if len(ad_line) >= 10:
            recent_ad = ad_line.iloc[-5:]
            recent_price = data["close"].iloc[-5:]

            ad_trend = recent_ad.iloc[-1] - recent_ad.iloc[0]
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]

            if price_trend > 0 and ad_trend < 0:
                signals.append(
                    {
                        "type": "ad_bearish_divergence",
                        "indicator": "A/D Line",
                        "signal": "sell",
                        "strength": "strong",
                        "note": "Price rising but distribution occurring",
                    }
                )
            elif price_trend < 0 and ad_trend > 0:
                signals.append(
                    {
                        "type": "ad_bullish_divergence",
                        "indicator": "A/D Line",
                        "signal": "buy",
                        "strength": "strong",
                        "note": "Price falling but accumulation occurring",
                    }
                )

        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"A/D Line calculation failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_mfi(
    data: pd.DataFrame, period: int = 14, overbought: float = 80.0, oversold: float = 20.0
) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Money Flow Index (MFI)

    Args:
        data: OHLCV DataFrame
        period: MFI period (default: 14)
        overbought: Overbought threshold (default: 80)
        oversold: Oversold threshold (default: 20)

    Returns:
        Dictionary with MFI results and money flow analysis
    """
    try:
        if TALIB_AVAILABLE:
            mfi = pd.Series(
                talib.MFI(
                    data["high"].values,
                    data["low"].values,
                    data["close"].values,
                    data["volume"].values,
                    timeperiod=period,
                ),
                index=data.index,
            )
        else:
            # Custom MFI implementation
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            money_flow = typical_price * data["volume"]

            # Positive and negative money flow
            positive_flow = pd.Series(0.0, index=data.index)
            negative_flow = pd.Series(0.0, index=data.index)

            for i in range(1, len(data)):
                if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
                    negative_flow.iloc[i] = money_flow.iloc[i]

            # Money Flow Ratio and MFI
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()

            money_flow_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + money_flow_ratio))
            mfi = mfi.fillna(50)  # Handle division by zero

        results = {"MFI": mfi.tolist()}
        signals = []

        # MFI signals
        current_mfi = mfi.iloc[-1]
        if not pd.isna(current_mfi):
            if current_mfi > overbought:
                signals.append(
                    {
                        "type": "mfi_overbought",
                        "indicator": "MFI",
                        "signal": "sell",
                        "strength": "medium",
                        "mfi_value": current_mfi,
                        "threshold": overbought,
                    }
                )
            elif current_mfi < oversold:
                signals.append(
                    {
                        "type": "mfi_oversold",
                        "indicator": "MFI",
                        "signal": "buy",
                        "strength": "medium",
                        "mfi_value": current_mfi,
                        "threshold": oversold,
                    }
                )

            # MFI divergence detection
            if len(mfi) >= 10:
                recent_mfi = mfi.iloc[-5:]
                recent_price = data["close"].iloc[-5:]

                mfi_trend = recent_mfi.iloc[-1] - recent_mfi.iloc[0]
                price_trend = recent_price.iloc[-1] - recent_price.iloc[0]

                if price_trend > 0 and mfi_trend < 0:
                    signals.append(
                        {
                            "type": "mfi_bearish_divergence",
                            "indicator": "MFI",
                            "signal": "sell",
                            "strength": "strong",
                            "note": "Price rising but money flow declining",
                        }
                    )
                elif price_trend < 0 and mfi_trend > 0:
                    signals.append(
                        {
                            "type": "mfi_bullish_divergence",
                            "indicator": "MFI",
                            "signal": "buy",
                            "strength": "strong",
                            "note": "Price falling but money flow improving",
                        }
                    )

        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "parameters": {"period": period, "overbought": overbought, "oversold": oversold},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"MFI calculation failed: {e}")
        return {"success": False, "error": str(e)}


def detect_volume_anomalies(
    data: pd.DataFrame, lookback: int = 20, threshold_multiplier: float = 2.0
) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Volume Anomalies and Spikes

    Args:
        data: OHLCV DataFrame
        lookback: Lookback period for average volume (default: 20)
        threshold_multiplier: Multiplier for anomaly detection (default: 2.0)

    Returns:
        Dictionary with volume anomaly analysis
    """
    try:
        # Calculate volume statistics
        volume_ma = data["volume"].rolling(window=lookback).mean()
        volume_std = data["volume"].rolling(window=lookback).std()

        # Detect anomalies
        current_volume = data["volume"].iloc[-1]
        current_ma = volume_ma.iloc[-1]
        current_std = volume_std.iloc[-1]

        signals = []

        if not pd.isna(current_ma) and not pd.isna(current_std):
            volume_z_score = (current_volume - current_ma) / current_std

            if current_volume > current_ma * threshold_multiplier:
                signals.append(
                    {
                        "type": "volume_spike",
                        "indicator": "Volume",
                        "signal": "neutral",
                        "strength": "high",
                        "note": "Unusual volume spike detected",
                        "volume": current_volume,
                        "avg_volume": current_ma,
                        "multiplier": current_volume / current_ma,
                        "z_score": volume_z_score,
                    }
                )

            # Volume drying up
            if current_volume < current_ma * 0.5:
                signals.append(
                    {
                        "type": "volume_drying_up",
                        "indicator": "Volume",
                        "signal": "neutral",
                        "strength": "medium",
                        "note": "Volume drying up - potential breakout setup",
                        "volume": current_volume,
                        "avg_volume": current_ma,
                        "ratio": current_volume / current_ma,
                    }
                )

        # Price-volume relationship analysis
        recent_data = data.iloc[-5:]
        if len(recent_data) >= 2:
            price_change = recent_data["close"].iloc[-1] - recent_data["close"].iloc[0]
            volume_trend = recent_data["volume"].iloc[-1] - recent_data["volume"].iloc[0]

            if price_change > 0 and volume_trend > 0:
                signals.append(
                    {
                        "type": "bullish_volume_confirmation",
                        "indicator": "Volume",
                        "signal": "buy",
                        "strength": "strong",
                        "note": "Price rise confirmed by volume increase",
                    }
                )
            elif price_change < 0 and volume_trend > 0:
                signals.append(
                    {
                        "type": "bearish_volume_confirmation",
                        "indicator": "Volume",
                        "signal": "sell",
                        "strength": "strong",
                        "note": "Price decline confirmed by volume increase",
                    }
                )
            elif price_change > 0 and volume_trend < 0:
                signals.append(
                    {
                        "type": "weak_bullish_move",
                        "indicator": "Volume",
                        "signal": "neutral",
                        "strength": "medium",
                        "note": "Price rise not confirmed by volume - weak move",
                    }
                )

        return {
            "success": True,
            "signals": signals,
            "analysis": {
                "current_volume": current_volume,
                "avg_volume": current_ma,
                "volume_ratio": current_volume / current_ma if not pd.isna(current_ma) else None,
                "volume_z_score": volume_z_score if "volume_z_score" in locals() else None,
            },
            "parameters": {"lookback": lookback, "threshold_multiplier": threshold_multiplier},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Volume anomaly detection failed: {e}")
        return {"success": False, "error": str(e)}


def analyze_volume_indicators(
    data: pd.DataFrame, mfi_period: int = 14, volume_lookback: int = 20
) -> Dict[str, Any]:
    """
    STRAND Tool: Comprehensive Volume Analysis

    Args:
        data: OHLCV DataFrame
        mfi_period: MFI calculation period
        volume_lookback: Volume anomaly detection lookback

    Returns:
        Comprehensive volume analysis results
    """
    try:
        # Calculate all volume indicators
        obv_result = calculate_obv(data)
        vwap_result = calculate_vwap(data)
        ad_result = calculate_ad_line(data)
        mfi_result = calculate_mfi(data, mfi_period)
        volume_anomaly_result = detect_volume_anomalies(data, volume_lookback)

        # Combine results
        all_indicators = {}
        all_signals = []

        for result in [obv_result, vwap_result, ad_result, mfi_result]:
            if result["success"]:
                all_indicators.update(result["indicators"])
                all_signals.extend(result["signals"])

        if volume_anomaly_result["success"]:
            all_signals.extend(volume_anomaly_result["signals"])

        # Volume sentiment analysis
        volume_signals = [s for s in all_signals if s["signal"] in ["buy", "sell"]]
        buy_volume = len([s for s in volume_signals if s["signal"] == "buy"])
        sell_volume = len([s for s in volume_signals if s["signal"] == "sell"])

        if buy_volume > sell_volume:
            volume_sentiment = "bullish"
        elif sell_volume > buy_volume:
            volume_sentiment = "bearish"
        else:
            volume_sentiment = "neutral"

        # Volume strength analysis
        strong_signals = len([s for s in all_signals if s["strength"] == "strong"])
        total_signals = len(all_signals)

        volume_strength = (
            "strong"
            if strong_signals > total_signals * 0.5
            else "medium"
            if strong_signals > 0
            else "weak"
        )

        # Calculate confidence score
        confidence_score = min(total_signals / 8.0, 1.0)  # Max confidence at 8+ signals

        return {
            "success": True,
            "indicators": all_indicators,
            "signals": all_signals,
            "analysis": {
                "volume_sentiment": volume_sentiment,
                "volume_strength": volume_strength,
                "buy_signals": buy_volume,
                "sell_signals": sell_volume,
                "total_signals": total_signals,
                "confidence_score": confidence_score,
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Volume analysis failed: {e}")
        return {"success": False, "error": str(e)}


def create_volume_analysis_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for volume analysis

    Returns:
        List of tool specifications for STRAND framework
    """
    return [
        {
            "name": "calculate_obv",
            "function": calculate_obv,
            "description": "Calculate On-Balance Volume with trend and divergence analysis",
            "parameters": {"data": "OHLCV DataFrame"},
            "category": "technical_analysis",
            "skill": "volume_analysis",
        },
        {
            "name": "calculate_vwap",
            "function": calculate_vwap,
            "description": "Calculate Volume Weighted Average Price with deviation analysis",
            "parameters": {"data": "OHLCV DataFrame"},
            "category": "technical_analysis",
            "skill": "volume_analysis",
        },
        {
            "name": "calculate_ad_line",
            "function": calculate_ad_line,
            "description": "Calculate Accumulation/Distribution Line with trend analysis",
            "parameters": {"data": "OHLCV DataFrame"},
            "category": "technical_analysis",
            "skill": "volume_analysis",
        },
        {
            "name": "calculate_mfi",
            "function": calculate_mfi,
            "description": "Calculate Money Flow Index with divergence detection",
            "parameters": {
                "data": "OHLCV DataFrame",
                "period": "MFI period (default: 14)",
                "overbought": "Overbought threshold (default: 80)",
                "oversold": "Oversold threshold (default: 20)",
            },
            "category": "technical_analysis",
            "skill": "volume_analysis",
        },
        {
            "name": "detect_volume_anomalies",
            "function": detect_volume_anomalies,
            "description": "Detect volume spikes and anomalies with price-volume analysis",
            "parameters": {
                "data": "OHLCV DataFrame",
                "lookback": "Lookback period (default: 20)",
                "threshold_multiplier": "Anomaly threshold multiplier (default: 2.0)",
            },
            "category": "technical_analysis",
            "skill": "volume_analysis",
        },
        {
            "name": "analyze_volume_indicators",
            "function": analyze_volume_indicators,
            "description": "Comprehensive volume analysis using OBV, VWAP, A/D Line, MFI, and anomaly detection",
            "parameters": {
                "data": "OHLCV DataFrame",
                "mfi_period": "MFI period",
                "volume_lookback": "Volume anomaly lookback period",
            },
            "category": "technical_analysis",
            "skill": "volume_analysis",
        },
    ]
