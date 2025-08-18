"""
Skill 5: Chart Pattern Recognition
STRAND tools for detecting common chart patterns like triangles, flags, head & shoulders, double tops/bottoms
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

def detect_triangles(data: pd.DataFrame, window: int = 10, min_points: int = 4) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Triangle Patterns (Ascending, Descending, Symmetrical)
    """
    try:
        highs = data['high'].values
        lows = data['low'].values
        
        peaks = argrelextrema(highs, np.greater, order=window)[0]
        troughs = argrelextrema(lows, np.less, order=window)[0]
        
        recent_peaks = peaks[-20:] if len(peaks) >= 20 else peaks
        recent_troughs = troughs[-20:] if len(troughs) >= 20 else troughs
        
        triangles = []
        
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            peak_x = np.array(recent_peaks).reshape(-1, 1)
            peak_y = np.array([highs[i] for i in recent_peaks])
            peak_reg = LinearRegression().fit(peak_x, peak_y)
            peak_slope = peak_reg.coef_[0]
            
            trough_x = np.array(recent_troughs).reshape(-1, 1)
            trough_y = np.array([lows[i] for i in recent_troughs])
            trough_reg = LinearRegression().fit(trough_x, trough_y)
            trough_slope = trough_reg.coef_[0]
            
            slope_threshold = 0.001
            
            if abs(peak_slope) < slope_threshold and trough_slope > slope_threshold:
                pattern_type = "ascending_triangle"
                bias = "bullish"
            elif peak_slope < -slope_threshold and abs(trough_slope) < slope_threshold:
                pattern_type = "descending_triangle"
                bias = "bearish"
            elif abs(peak_slope - (-abs(trough_slope))) < slope_threshold * 2:
                pattern_type = "symmetrical_triangle"
                bias = "neutral"
            else:
                pattern_type = "undefined"
                bias = "neutral"
            
            if pattern_type != "undefined":
                triangles.append({
                    "type": pattern_type,
                    "bias": bias,
                    "peak_slope": float(peak_slope),
                    "trough_slope": float(trough_slope),
                    "recent_peaks": [{"index": int(i), "price": float(highs[i])} for i in recent_peaks],
                    "recent_troughs": [{"index": int(i), "price": float(lows[i])} for i in recent_troughs]
                })
        
        signals = []
        current_price = data['close'].iloc[-1]
        
        for triangle in triangles:
            if triangle["type"] == "ascending_triangle":
                resistance_level = max([p["price"] for p in triangle["recent_peaks"]])
                if current_price > resistance_level * 1.01:
                    signals.append({
                        "type": "triangle_breakout",
                        "indicator": "Ascending Triangle",
                        "signal": "buy",
                        "strength": "strong",
                        "price": current_price,
                        "breakout_level": resistance_level
                    })
        
        return {
            "success": True,
            "patterns": triangles,
            "signals": signals,
            "parameters": {"window": window, "min_points": min_points},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Triangle detection failed: {e}")
        return {"success": False, "error": str(e)}

def detect_head_and_shoulders(data: pd.DataFrame, window: int = 5, 
                             shoulder_tolerance: float = 0.05) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Head and Shoulders Patterns
    """
    try:
        highs = data['high'].values
        lows = data['low'].values
        
        peaks = argrelextrema(highs, np.greater, order=window)[0]
        troughs = argrelextrema(lows, np.less, order=window)[0]
        
        patterns = []
        
        if len(peaks) >= 3:
            recent_peaks = peaks[-10:] if len(peaks) >= 10 else peaks
            
            for i in range(len(recent_peaks) - 2):
                left_shoulder_idx = recent_peaks[i]
                head_idx = recent_peaks[i + 1]
                right_shoulder_idx = recent_peaks[i + 2]
                
                left_shoulder_price = highs[left_shoulder_idx]
                head_price = highs[head_idx]
                right_shoulder_price = highs[right_shoulder_idx]
                
                if head_price > left_shoulder_price and head_price > right_shoulder_price:
                    shoulder_diff = abs(left_shoulder_price - right_shoulder_price) / max(left_shoulder_price, right_shoulder_price)
                    
                    if shoulder_diff <= shoulder_tolerance:
                        relevant_troughs = [t for t in troughs if left_shoulder_idx < t < right_shoulder_idx]
                        
                        if len(relevant_troughs) >= 1:
                            neckline_level = np.mean([lows[t] for t in relevant_troughs])
                            pattern_height = head_price - neckline_level
                            
                            patterns.append({
                                "type": "head_and_shoulders",
                                "bias": "bearish",
                                "head_price": float(head_price),
                                "neckline_level": float(neckline_level),
                                "target_price": float(neckline_level - pattern_height)
                            })
        
        signals = []
        current_price = data['close'].iloc[-1]
        
        for pattern in patterns:
            if current_price < pattern["neckline_level"] * 0.99:
                signals.append({
                    "type": "head_shoulders_breakdown",
                    "indicator": "Head and Shoulders",
                    "signal": "sell",
                    "strength": "strong",
                    "price": current_price,
                    "target": pattern["target_price"]
                })
        
        return {
            "success": True,
            "patterns": patterns,
            "signals": signals,
            "parameters": {"window": window, "shoulder_tolerance": shoulder_tolerance},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Head and shoulders detection failed: {e}")
        return {"success": False, "error": str(e)}

def detect_double_tops_bottoms(data: pd.DataFrame, window: int = 5,
                              similarity_tolerance: float = 0.02) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Double Top and Double Bottom Patterns
    """
    try:
        highs = data['high'].values
        lows = data['low'].values
        
        peaks = argrelextrema(highs, np.greater, order=window)[0]
        troughs = argrelextrema(lows, np.less, order=window)[0]
        
        patterns = []
        
        # Double tops
        if len(peaks) >= 2:
            recent_peaks = peaks[-10:] if len(peaks) >= 10 else peaks
            
            for i in range(len(recent_peaks) - 1):
                for j in range(i + 1, len(recent_peaks)):
                    peak1_idx = recent_peaks[i]
                    peak2_idx = recent_peaks[j]
                    peak1_price = highs[peak1_idx]
                    peak2_price = highs[peak2_idx]
                    
                    price_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                    
                    if price_diff <= similarity_tolerance:
                        valley_indices = [t for t in troughs if peak1_idx < t < peak2_idx]
                        
                        if valley_indices:
                            valley_price = lows[valley_indices[0]]
                            valley_depth = min(peak1_price, peak2_price) - valley_price
                            
                            if valley_depth > (max(peak1_price, peak2_price) * 0.02):
                                patterns.append({
                                    "type": "double_top",
                                    "bias": "bearish",
                                    "resistance_level": float(max(peak1_price, peak2_price)),
                                    "support_level": float(valley_price),
                                    "target_price": float(valley_price - valley_depth)
                                })
        
        # Double bottoms
        if len(troughs) >= 2:
            recent_troughs = troughs[-10:] if len(troughs) >= 10 else troughs
            
            for i in range(len(recent_troughs) - 1):
                for j in range(i + 1, len(recent_troughs)):
                    trough1_idx = recent_troughs[i]
                    trough2_idx = recent_troughs[j]
                    trough1_price = lows[trough1_idx]
                    trough2_price = lows[trough2_idx]
                    
                    price_diff = abs(trough1_price - trough2_price) / max(trough1_price, trough2_price)
                    
                    if price_diff <= similarity_tolerance:
                        peak_indices = [p for p in peaks if trough1_idx < p < trough2_idx]
                        
                        if peak_indices:
                            peak_price = highs[peak_indices[0]]
                            peak_height = peak_price - max(trough1_price, trough2_price)
                            
                            if peak_height > (max(trough1_price, trough2_price) * 0.02):
                                patterns.append({
                                    "type": "double_bottom",
                                    "bias": "bullish",
                                    "support_level": float(min(trough1_price, trough2_price)),
                                    "resistance_level": float(peak_price),
                                    "target_price": float(peak_price + peak_height)
                                })
        
        signals = []
        current_price = data['close'].iloc[-1]
        
        for pattern in patterns:
            if pattern["type"] == "double_top" and current_price < pattern["support_level"] * 0.99:
                signals.append({
                    "type": "double_top_breakdown",
                    "indicator": "Double Top",
                    "signal": "sell",
                    "strength": "strong",
                    "price": current_price,
                    "target": pattern["target_price"]
                })
            elif pattern["type"] == "double_bottom" and current_price > pattern["resistance_level"] * 1.01:
                signals.append({
                    "type": "double_bottom_breakout",
                    "indicator": "Double Bottom",
                    "signal": "buy",
                    "strength": "strong",
                    "price": current_price,
                    "target": pattern["target_price"]
                })
        
        return {
            "success": True,
            "patterns": patterns,
            "signals": signals,
            "parameters": {"window": window, "similarity_tolerance": similarity_tolerance},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Double tops/bottoms detection failed: {e}")
        return {"success": False, "error": str(e)}

def detect_flags_pennants(data: pd.DataFrame, trend_window: int = 20,
                         pattern_window: int = 10) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Flag and Pennant Patterns
    """
    try:
        patterns = []
        
        if len(data) < trend_window + pattern_window:
            return {
                "success": True,
                "patterns": [],
                "signals": [],
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        trend_start = len(data) - trend_window - pattern_window
        trend_end = len(data) - pattern_window
        pattern_start = trend_end
        pattern_end = len(data) - 1
        
        if trend_start < 0:
            trend_start = 0
        
        trend_data = data.iloc[trend_start:trend_end]
        pattern_data = data.iloc[pattern_start:pattern_end]
        
        trend_change = (trend_data['close'].iloc[-1] - trend_data['close'].iloc[0]) / trend_data['close'].iloc[0]
        
        if abs(trend_change) >= 0.05:  # 5% minimum trend
            trend_direction = "up" if trend_change > 0 else "down"
            pattern_range = (pattern_data['high'].max() - pattern_data['low'].min()) / pattern_data['close'].mean()
            
            if pattern_range < 0.05:  # Tight consolidation
                flagpole_height = abs(trend_data['close'].iloc[-1] - trend_data['close'].iloc[0])
                current_price = data['close'].iloc[-1]
                
                if trend_direction == "up":
                    pattern_type = "bullish_flag"
                    target_price = current_price + flagpole_height
                    bias = "bullish"
                else:
                    pattern_type = "bearish_flag"
                    target_price = current_price - flagpole_height
                    bias = "bearish"
                
                patterns.append({
                    "type": pattern_type,
                    "bias": bias,
                    "target_price": float(target_price),
                    "flagpole_height": float(flagpole_height)
                })
        
        signals = []
        current_price = data['close'].iloc[-1]
        
        for pattern in patterns:
            pattern_data = data.iloc[pattern_start:pattern_end]
            
            if pattern["bias"] == "bullish":
                breakout_level = pattern_data['high'].max()
                if current_price > breakout_level * 1.005:
                    signals.append({
                        "type": f"{pattern['type']}_breakout",
                        "indicator": pattern["type"].replace("_", " ").title(),
                        "signal": "buy",
                        "strength": "strong",
                        "price": current_price,
                        "target": pattern["target_price"]
                    })
        
        return {
            "success": True,
            "patterns": patterns,
            "signals": signals,
            "parameters": {"trend_window": trend_window, "pattern_window": pattern_window},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Flags/pennants detection failed: {e}")
        return {"success": False, "error": str(e)}

def analyze_chart_patterns(data: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
    """
    STRAND Tool: Comprehensive Chart Pattern Analysis
    """
    try:
        triangles = detect_triangles(data, window)
        head_shoulders = detect_head_and_shoulders(data, window)
        double_patterns = detect_double_tops_bottoms(data, window)
        flags_pennants = detect_flags_pennants(data)
        
        all_patterns = []
        all_signals = []
        
        for result in [triangles, head_shoulders, double_patterns, flags_pennants]:
            if result["success"]:
                all_patterns.extend(result.get("patterns", []))
                all_signals.extend(result.get("signals", []))
        
        # Pattern summary
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_type = pattern["type"]
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            "success": True,
            "patterns": {
                "all_patterns": all_patterns,
                "triangles": triangles.get("patterns", []),
                "head_shoulders": head_shoulders.get("patterns", []),
                "double_patterns": double_patterns.get("patterns", []),
                "flags_pennants": flags_pennants.get("patterns", [])
            },
            "signals": all_signals,
            "summary": {
                "total_patterns": len(all_patterns),
                "pattern_counts": pattern_counts,
                "total_signals": len(all_signals)
            },
            "parameters": {"window": window},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chart pattern analysis failed: {e}")
        return {"success": False, "error": str(e)}

def create_chart_pattern_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for chart pattern recognition
    """
    return [
        {
            "name": "detect_triangles",
            "function": detect_triangles,
            "description": "Detect triangle patterns (ascending, descending, symmetrical)",
            "parameters": {
                "data": "OHLCV DataFrame",
                "window": "Window for peak/trough detection",
                "min_points": "Minimum points for pattern validation"
            },
            "category": "technical_analysis",
            "skill": "chart_patterns"
        },
        {
            "name": "detect_head_and_shoulders",
            "function": detect_head_and_shoulders,
            "description": "Detect head and shoulders patterns",
            "parameters": {
                "data": "OHLCV DataFrame",
                "window": "Window for peak detection",
                "shoulder_tolerance": "Tolerance for shoulder height similarity"
            },
            "category": "technical_analysis",
            "skill": "chart_patterns"
        },
        {
            "name": "detect_double_tops_bottoms",
            "function": detect_double_tops_bottoms,
            "description": "Detect double top and double bottom patterns",
            "parameters": {
                "data": "OHLCV DataFrame",
                "window": "Window for peak/trough detection",
                "similarity_tolerance": "Price similarity tolerance"
            },
            "category": "technical_analysis",
            "skill": "chart_patterns"
        },
        {
            "name": "detect_flags_pennants",
            "function": detect_flags_pennants,
            "description": "Detect flag and pennant continuation patterns",
            "parameters": {
                "data": "OHLCV DataFrame",
                "trend_window": "Window to identify preceding trend",
                "pattern_window": "Window for flag/pennant pattern"
            },
            "category": "technical_analysis",
            "skill": "chart_patterns"
        },
        {
            "name": "analyze_chart_patterns",
            "function": analyze_chart_patterns,
            "description": "Comprehensive chart pattern analysis using all detection methods",
            "parameters": {
                "data": "OHLCV DataFrame",
                "window": "Window for pattern detection"
            },
            "category": "technical_analysis",
            "skill": "chart_patterns"
        }
    ]
