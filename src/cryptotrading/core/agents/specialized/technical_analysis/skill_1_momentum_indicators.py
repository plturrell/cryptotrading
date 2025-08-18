"""
Skill 1: Momentum Indicators (SMA, EMA, RSI)
STRAND tools for fundamental technical indicators
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# TA-Lib imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Install with: conda install -c conda-forge ta-lib")

logger = logging.getLogger(__name__)

def calculate_sma(data: pd.DataFrame, periods: List[int] = None) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Simple Moving Averages
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods to calculate (default: [9, 21, 50, 200])
    
    Returns:
        Dictionary with SMA results and signals
    """
    periods = periods or [9, 21, 50, 200]
    
    try:
        results = {}
        signals = []
        
        for period in periods:
            if TALIB_AVAILABLE:
                sma = pd.Series(
                    talib.SMA(data['close'].values, timeperiod=period),
                    index=data.index
                )
            else:
                sma = data['close'].rolling(window=period).mean()
            
            results[f'SMA_{period}'] = sma.tolist()
            
            # Generate crossover signals
            current_price = data['close'].iloc[-1]
            current_sma = sma.iloc[-1]
            prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
            prev_sma = sma.iloc[-2] if len(sma) > 1 else current_sma
            
            if not pd.isna(current_sma) and not pd.isna(prev_sma):
                if current_price > current_sma and prev_price <= prev_sma:
                    signals.append({
                        "type": "sma_crossover",
                        "indicator": f"SMA_{period}",
                        "signal": "buy",
                        "strength": "medium",
                        "price": current_price,
                        "sma_value": current_sma
                    })
                elif current_price < current_sma and prev_price >= prev_sma:
                    signals.append({
                        "type": "sma_crossover",
                        "indicator": f"SMA_{period}",
                        "signal": "sell",
                        "strength": "medium",
                        "price": current_price,
                        "sma_value": current_sma
                    })
        
        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "periods": periods,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"SMA calculation failed: {e}")
        return {"success": False, "error": str(e)}

def calculate_ema(data: pd.DataFrame, periods: List[int] = None) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Exponential Moving Averages
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods to calculate (default: [9, 21, 50, 200])
    
    Returns:
        Dictionary with EMA results and signals
    """
    periods = periods or [9, 21, 50, 200]
    
    try:
        results = {}
        signals = []
        
        for period in periods:
            if TALIB_AVAILABLE:
                ema = pd.Series(
                    talib.EMA(data['close'].values, timeperiod=period),
                    index=data.index
                )
            else:
                ema = data['close'].ewm(span=period).mean()
            
            results[f'EMA_{period}'] = ema.tolist()
            
            # Generate crossover signals
            current_price = data['close'].iloc[-1]
            current_ema = ema.iloc[-1]
            prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
            prev_ema = ema.iloc[-2] if len(ema) > 1 else current_ema
            
            if not pd.isna(current_ema) and not pd.isna(prev_ema):
                if current_price > current_ema and prev_price <= prev_ema:
                    signals.append({
                        "type": "ema_crossover",
                        "indicator": f"EMA_{period}",
                        "signal": "buy",
                        "strength": "strong",
                        "price": current_price,
                        "ema_value": current_ema
                    })
                elif current_price < current_ema and prev_price >= prev_ema:
                    signals.append({
                        "type": "ema_crossover",
                        "indicator": f"EMA_{period}",
                        "signal": "sell",
                        "strength": "strong",
                        "price": current_price,
                        "ema_value": current_ema
                    })
        
        # Generate EMA crossover signals (fast vs slow)
        if len(periods) >= 2:
            fast_ema = results[f'EMA_{periods[0]}']
            slow_ema = results[f'EMA_{periods[1]}']
            
            if (fast_ema[-1] > slow_ema[-1] and 
                len(fast_ema) > 1 and len(slow_ema) > 1 and
                fast_ema[-2] <= slow_ema[-2]):
                signals.append({
                    "type": "ema_golden_cross",
                    "indicator": f"EMA_{periods[0]}_x_EMA_{periods[1]}",
                    "signal": "buy",
                    "strength": "very_strong",
                    "fast_ema": fast_ema[-1],
                    "slow_ema": slow_ema[-1]
                })
            elif (fast_ema[-1] < slow_ema[-1] and 
                  len(fast_ema) > 1 and len(slow_ema) > 1 and
                  fast_ema[-2] >= slow_ema[-2]):
                signals.append({
                    "type": "ema_death_cross",
                    "indicator": f"EMA_{periods[0]}_x_EMA_{periods[1]}",
                    "signal": "sell",
                    "strength": "very_strong",
                    "fast_ema": fast_ema[-1],
                    "slow_ema": slow_ema[-1]
                })
        
        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "periods": periods,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"EMA calculation failed: {e}")
        return {"success": False, "error": str(e)}

def calculate_rsi(data: pd.DataFrame, period: int = 14, 
                 overbought: float = 70.0, oversold: float = 30.0) -> Dict[str, Any]:
    """
    STRAND Tool: Calculate Relative Strength Index
    
    Args:
        data: OHLCV DataFrame
        period: RSI period (default: 14)
        overbought: Overbought threshold (default: 70)
        oversold: Oversold threshold (default: 30)
    
    Returns:
        Dictionary with RSI results and signals
    """
    try:
        if TALIB_AVAILABLE:
            rsi = pd.Series(
                talib.RSI(data['close'].values, timeperiod=period),
                index=data.index
            )
        else:
            # Custom RSI implementation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        results = {"RSI": rsi.tolist()}
        signals = []
        
        # Generate RSI signals
        current_rsi = rsi.iloc[-1]
        if not pd.isna(current_rsi):
            if current_rsi > overbought:
                signals.append({
                    "type": "rsi_overbought",
                    "indicator": "RSI",
                    "signal": "sell",
                    "strength": "medium",
                    "rsi_value": current_rsi,
                    "threshold": overbought
                })
            elif current_rsi < oversold:
                signals.append({
                    "type": "rsi_oversold",
                    "indicator": "RSI",
                    "signal": "buy",
                    "strength": "medium",
                    "rsi_value": current_rsi,
                    "threshold": oversold
                })
            
            # RSI divergence detection (simplified)
            if len(rsi) >= 10:
                recent_rsi = rsi.iloc[-5:]
                recent_price = data['close'].iloc[-5:]
                
                if (recent_price.iloc[-1] > recent_price.iloc[0] and 
                    recent_rsi.iloc[-1] < recent_rsi.iloc[0]):
                    signals.append({
                        "type": "rsi_bearish_divergence",
                        "indicator": "RSI",
                        "signal": "sell",
                        "strength": "strong",
                        "rsi_value": current_rsi
                    })
                elif (recent_price.iloc[-1] < recent_price.iloc[0] and 
                      recent_rsi.iloc[-1] > recent_rsi.iloc[0]):
                    signals.append({
                        "type": "rsi_bullish_divergence",
                        "indicator": "RSI",
                        "signal": "buy",
                        "strength": "strong",
                        "rsi_value": current_rsi
                    })
        
        return {
            "success": True,
            "indicators": results,
            "signals": signals,
            "period": period,
            "overbought": overbought,
            "oversold": oversold,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        return {"success": False, "error": str(e)}

def analyze_momentum_indicators(data: pd.DataFrame, 
                           sma_periods: List[int] = None,
                           ema_periods: List[int] = None,
                           rsi_period: int = 14,
                           rsi_overbought: float = 70.0,
                           rsi_oversold: float = 30.0) -> Dict[str, Any]:
    """
    STRAND Tool: Comprehensive Momentum Indicators Analysis
    
    Args:
        data: OHLCV DataFrame
        sma_periods: SMA periods
        ema_periods: EMA periods  
        rsi_period: RSI period
        rsi_overbought: RSI overbought threshold
        rsi_oversold: RSI oversold threshold
    
    Returns:
        Comprehensive analysis results
    """
    try:
        # Calculate all indicators
        sma_result = calculate_sma(data, sma_periods)
        ema_result = calculate_ema(data, ema_periods)
        rsi_result = calculate_rsi(data, rsi_period, rsi_overbought, rsi_oversold)
        
        # Combine results
        all_indicators = {}
        all_signals = []
        
        if sma_result["success"]:
            all_indicators.update(sma_result["indicators"])
            all_signals.extend(sma_result["signals"])
        
        if ema_result["success"]:
            all_indicators.update(ema_result["indicators"])
            all_signals.extend(ema_result["signals"])
        
        if rsi_result["success"]:
            all_indicators.update(rsi_result["indicators"])
            all_signals.extend(rsi_result["signals"])
        
        # Generate overall sentiment
        buy_signals = len([s for s in all_signals if s["signal"] == "buy"])
        sell_signals = len([s for s in all_signals if s["signal"] == "sell"])
        
        if buy_signals > sell_signals:
            overall_sentiment = "bullish"
        elif sell_signals > buy_signals:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
        
        # Calculate confidence score
        total_signals = buy_signals + sell_signals
        confidence_score = min(total_signals / 5.0, 1.0)  # Max confidence at 5+ signals
        
        return {
            "success": True,
            "indicators": all_indicators,
            "signals": all_signals,
            "analysis": {
                "overall_sentiment": overall_sentiment,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "confidence_score": confidence_score,
                "signal_strength": "strong" if confidence_score > 0.7 else "medium" if confidence_score > 0.4 else "weak"
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Momentum indicators analysis failed: {e}")
        return {"success": False, "error": str(e)}

def create_momentum_indicators_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for momentum indicators
    
    Returns:
        List of tool specifications for STRAND framework
    """
    return [
        {
            "name": "calculate_sma",
            "function": calculate_sma,
            "description": "Calculate Simple Moving Averages with crossover signals",
            "parameters": {
                "data": "OHLCV DataFrame",
                "periods": "List of periods (default: [9,21,50,200])"
            },
            "category": "technical_analysis",
            "skill": "momentum_indicators"
        },
        {
            "name": "calculate_ema", 
            "function": calculate_ema,
            "description": "Calculate Exponential Moving Averages with crossover signals",
            "parameters": {
                "data": "OHLCV DataFrame",
                "periods": "List of periods (default: [9,21,50,200])"
            },
            "category": "technical_analysis",
            "skill": "momentum_indicators"
        },
        {
            "name": "calculate_rsi",
            "function": calculate_rsi,
            "description": "Calculate RSI with overbought/oversold signals and divergence detection",
            "parameters": {
                "data": "OHLCV DataFrame",
                "period": "RSI period (default: 14)",
                "overbought": "Overbought threshold (default: 70)",
                "oversold": "Oversold threshold (default: 30)"
            },
            "category": "technical_analysis",
            "skill": "momentum_indicators"
        },
        {
            "name": "analyze_momentum_indicators",
            "function": analyze_momentum_indicators,
            "description": "Comprehensive analysis using SMA, EMA, and RSI indicators",
            "parameters": {
                "data": "OHLCV DataFrame",
                "sma_periods": "SMA periods",
                "ema_periods": "EMA periods",
                "rsi_period": "RSI period",
                "rsi_overbought": "RSI overbought threshold",
                "rsi_oversold": "RSI oversold threshold"
            },
            "category": "technical_analysis",
            "skill": "momentum_indicators"
        }
    ]
