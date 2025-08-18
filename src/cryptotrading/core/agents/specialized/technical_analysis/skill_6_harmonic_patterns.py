"""
Skill 6: Advanced Pattern Recognition
STRAND tools for detecting complex patterns like Elliott Wave, Harmonic patterns, Gartley, Butterfly, etc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from scipy.signal import argrelextrema
from scipy.stats import linregress

logger = logging.getLogger(__name__)

def detect_elliott_wave_impulse(data: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Elliott Wave Impulse Patterns (5-wave structure)
    
    Args:
        data: OHLCV DataFrame
        window: Window for peak/trough detection
    
    Returns:
        Dictionary with detected Elliott Wave impulse patterns
    """
    try:
        highs = data['high'].values
        lows = data['low'].values
        
        # Find significant peaks and troughs
        peaks = argrelextrema(highs, np.greater, order=window)[0]
        troughs = argrelextrema(lows, np.less, order=window)[0]
        
        # Combine and sort by index
        all_points = []
        for peak in peaks:
            all_points.append({"index": peak, "price": highs[peak], "type": "peak"})
        for trough in troughs:
            all_points.append({"index": trough, "price": lows[trough], "type": "trough"})
        
        all_points.sort(key=lambda x: x["index"])
        
        impulse_patterns = []
        
        # Look for 5-wave impulse patterns (peak-trough-peak-trough-peak for bullish)
        if len(all_points) >= 9:  # Need at least 9 points for complete pattern
            for i in range(len(all_points) - 8):
                sequence = all_points[i:i+9]
                
                # Check for bullish impulse: T-P-T-P-T-P-T-P-T
                if (sequence[0]["type"] == "trough" and sequence[1]["type"] == "peak" and
                    sequence[2]["type"] == "trough" and sequence[3]["type"] == "peak" and
                    sequence[4]["type"] == "trough" and sequence[5]["type"] == "peak" and
                    sequence[6]["type"] == "trough" and sequence[7]["type"] == "peak" and
                    sequence[8]["type"] == "trough"):
                    
                    # Elliott Wave rules validation
                    wave1_start = sequence[0]["price"]  # Wave 1 start
                    wave1_end = sequence[1]["price"]    # Wave 1 end
                    wave2_end = sequence[2]["price"]    # Wave 2 end
                    wave3_end = sequence[3]["price"]    # Wave 3 end
                    wave4_end = sequence[4]["price"]    # Wave 4 end
                    wave5_end = sequence[5]["price"]    # Wave 5 end
                    
                    # Rule 1: Wave 2 never retraces more than 100% of Wave 1
                    wave1_length = wave1_end - wave1_start
                    wave2_retrace = wave1_end - wave2_end
                    
                    # Rule 2: Wave 3 is never the shortest wave
                    wave3_length = wave3_end - wave2_end
                    wave5_length = wave5_end - wave4_end
                    
                    # Rule 3: Wave 4 never enters the price territory of Wave 1
                    wave4_low = wave4_end
                    
                    if (wave2_retrace <= wave1_length and  # Rule 1
                        wave3_length >= max(wave1_length, wave5_length) and  # Rule 2
                        wave4_low > wave1_end):  # Rule 3
                        
                        # Calculate Fibonacci ratios
                        fib_ratios = {
                            "wave2_retrace": wave2_retrace / wave1_length if wave1_length > 0 else 0,
                            "wave3_extension": wave3_length / wave1_length if wave1_length > 0 else 0,
                            "wave4_retrace": (wave3_end - wave4_end) / wave3_length if wave3_length > 0 else 0,
                            "wave5_extension": wave5_length / wave1_length if wave1_length > 0 else 0
                        }
                        
                        impulse_patterns.append({
                            "type": "bullish_impulse",
                            "bias": "bullish",
                            "waves": {
                                "wave1": {"start": wave1_start, "end": wave1_end, "length": wave1_length},
                                "wave2": {"end": wave2_end, "retrace": wave2_retrace},
                                "wave3": {"end": wave3_end, "length": wave3_length},
                                "wave4": {"end": wave4_end},
                                "wave5": {"end": wave5_end, "length": wave5_length}
                            },
                            "fibonacci_ratios": fib_ratios,
                            "start_index": sequence[0]["index"],
                            "end_index": sequence[8]["index"],
                            "confidence": self._calculate_elliott_confidence(fib_ratios)
                        })
        
        # Generate signals
        signals = []
        current_price = data['close'].iloc[-1]
        
        for pattern in impulse_patterns:
            if pattern["confidence"] > 0.6:  # High confidence patterns only
                wave5_end = pattern["waves"]["wave5"]["end"]
                
                # Signal potential end of impulse wave
                if abs(current_price - wave5_end) / current_price < 0.02:  # Within 2%
                    signals.append({
                        "type": "elliott_wave_completion",
                        "indicator": "Elliott Wave",
                        "signal": "sell" if pattern["type"] == "bullish_impulse" else "buy",
                        "strength": "strong",
                        "price": current_price,
                        "pattern_type": pattern["type"],
                        "confidence": pattern["confidence"],
                        "note": f"Elliott Wave {pattern['type']} completion near {wave5_end:.2f}"
                    })
        
        return {
            "success": True,
            "patterns": impulse_patterns,
            "signals": signals,
            "parameters": {"window": window},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Elliott Wave detection failed: {e}")
        return {"success": False, "error": str(e)}

def _calculate_elliott_confidence(fib_ratios: Dict[str, float]) -> float:
    """Calculate confidence score for Elliott Wave pattern based on Fibonacci ratios"""
    confidence = 0.0
    
    # Wave 2 typically retraces 50-61.8% of Wave 1
    if 0.5 <= fib_ratios["wave2_retrace"] <= 0.618:
        confidence += 0.25
    elif 0.382 <= fib_ratios["wave2_retrace"] <= 0.786:
        confidence += 0.15
    
    # Wave 3 typically extends 161.8% of Wave 1
    if 1.5 <= fib_ratios["wave3_extension"] <= 1.8:
        confidence += 0.3
    elif 1.2 <= fib_ratios["wave3_extension"] <= 2.0:
        confidence += 0.2
    
    # Wave 4 typically retraces 23.6-38.2% of Wave 3
    if 0.236 <= fib_ratios["wave4_retrace"] <= 0.382:
        confidence += 0.25
    elif 0.1 <= fib_ratios["wave4_retrace"] <= 0.5:
        confidence += 0.15
    
    # Wave 5 typically equals Wave 1 or extends to 61.8% of Wave 1
    if 0.8 <= fib_ratios["wave5_extension"] <= 1.2:
        confidence += 0.2
    elif 0.6 <= fib_ratios["wave5_extension"] <= 1.4:
        confidence += 0.1
    
    return min(confidence, 1.0)

def detect_harmonic_patterns(data: pd.DataFrame, window: int = 5,
                           tolerance: float = 0.05) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Harmonic Patterns (Gartley, Butterfly, Bat, Crab)
    
    Args:
        data: OHLCV DataFrame
        window: Window for peak/trough detection
        tolerance: Fibonacci ratio tolerance (5%)
    
    Returns:
        Dictionary with detected harmonic patterns
    """
    try:
        highs = data['high'].values
        lows = data['low'].values
        
        peaks = argrelextrema(highs, np.greater, order=window)[0]
        troughs = argrelextrema(lows, np.less, order=window)[0]
        
        # Combine and sort points
        all_points = []
        for peak in peaks:
            all_points.append({"index": peak, "price": highs[peak], "type": "peak"})
        for trough in troughs:
            all_points.append({"index": trough, "price": lows[trough], "type": "trough"})
        
        all_points.sort(key=lambda x: x["index"])
        
        harmonic_patterns = []
        
        # Look for 5-point harmonic patterns (X-A-B-C-D)
        if len(all_points) >= 5:
            for i in range(len(all_points) - 4):
                points = all_points[i:i+5]
                
                # Extract XABCD points
                X = points[0]["price"]
                A = points[1]["price"]
                B = points[2]["price"]
                C = points[3]["price"]
                D = points[4]["price"]
                
                # Calculate Fibonacci ratios
                XA = abs(A - X)
                AB = abs(B - A)
                BC = abs(C - B)
                CD = abs(D - C)
                XD = abs(D - X)
                AC = abs(C - A)
                
                if XA > 0 and AB > 0 and BC > 0 and CD > 0:
                    ratios = {
                        "AB_XA": AB / XA,
                        "BC_AB": BC / AB,
                        "CD_BC": CD / BC,
                        "XD_XA": XD / XA,
                        "AC_AB": AC / AB if AB > 0 else 0
                    }
                    
                    # Check for specific harmonic patterns
                    pattern_type = self._identify_harmonic_pattern(ratios, tolerance)
                    
                    if pattern_type:
                        # Determine bullish/bearish based on point sequence
                        if points[0]["type"] != points[4]["type"]:  # X and D are different types
                            bias = "bullish" if points[4]["type"] == "trough" else "bearish"
                        else:
                            bias = "neutral"
                        
                        harmonic_patterns.append({
                            "type": pattern_type,
                            "bias": bias,
                            "points": {
                                "X": {"index": points[0]["index"], "price": X},
                                "A": {"index": points[1]["index"], "price": A},
                                "B": {"index": points[2]["index"], "price": B},
                                "C": {"index": points[3]["index"], "price": C},
                                "D": {"index": points[4]["index"], "price": D}
                            },
                            "ratios": ratios,
                            "completion_level": D,
                            "target_levels": self._calculate_harmonic_targets(X, A, B, C, D, pattern_type)
                        })
        
        # Generate signals
        signals = []
        current_price = data['close'].iloc[-1]
        
        for pattern in harmonic_patterns:
            completion_level = pattern["completion_level"]
            
            # Signal when price reaches completion level (D point)
            if abs(current_price - completion_level) / current_price < 0.015:  # Within 1.5%
                signal_direction = "buy" if pattern["bias"] == "bullish" else "sell"
                
                signals.append({
                    "type": f"harmonic_{pattern['type']}_completion",
                    "indicator": f"Harmonic {pattern['type'].title()}",
                    "signal": signal_direction,
                    "strength": "strong",
                    "price": current_price,
                    "completion_level": completion_level,
                    "targets": pattern["target_levels"],
                    "note": f"Harmonic {pattern['type']} pattern completion at D point"
                })
        
        return {
            "success": True,
            "patterns": harmonic_patterns,
            "signals": signals,
            "parameters": {"window": window, "tolerance": tolerance},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Harmonic pattern detection failed: {e}")
        return {"success": False, "error": str(e)}

def _identify_harmonic_pattern(ratios: Dict[str, float], tolerance: float) -> Optional[str]:
    """Identify specific harmonic pattern based on Fibonacci ratios"""
    
    def within_tolerance(actual: float, expected: float, tol: float) -> bool:
        return abs(actual - expected) <= tol
    
    # Gartley Pattern
    if (within_tolerance(ratios["AB_XA"], 0.618, tolerance) and
        within_tolerance(ratios["BC_AB"], 0.382, tolerance) and
        within_tolerance(ratios["CD_BC"], 1.272, tolerance) and
        within_tolerance(ratios["XD_XA"], 0.786, tolerance)):
        return "gartley"
    
    # Butterfly Pattern
    if (within_tolerance(ratios["AB_XA"], 0.786, tolerance) and
        within_tolerance(ratios["BC_AB"], 0.382, tolerance) and
        within_tolerance(ratios["CD_BC"], 1.618, tolerance) and
        within_tolerance(ratios["XD_XA"], 1.27, tolerance)):
        return "butterfly"
    
    # Bat Pattern
    if (within_tolerance(ratios["AB_XA"], 0.382, tolerance) and
        within_tolerance(ratios["BC_AB"], 0.382, tolerance) and
        within_tolerance(ratios["CD_BC"], 1.618, tolerance) and
        within_tolerance(ratios["XD_XA"], 0.886, tolerance)):
        return "bat"
    
    # Crab Pattern
    if (within_tolerance(ratios["AB_XA"], 0.382, tolerance) and
        within_tolerance(ratios["BC_AB"], 0.382, tolerance) and
        within_tolerance(ratios["CD_BC"], 2.24, tolerance) and
        within_tolerance(ratios["XD_XA"], 1.618, tolerance)):
        return "crab"
    
    return None

def _calculate_harmonic_targets(X: float, A: float, B: float, C: float, D: float, 
                              pattern_type: str) -> List[float]:
    """Calculate target levels for harmonic patterns"""
    targets = []
    
    # Common Fibonacci retracement levels from D point
    CD_range = abs(D - C)
    
    if pattern_type in ["gartley", "bat"]:
        # Conservative targets
        targets.extend([
            D + CD_range * 0.382,
            D + CD_range * 0.618,
            C  # Return to C level
        ])
    elif pattern_type in ["butterfly", "crab"]:
        # More aggressive targets
        targets.extend([
            D + CD_range * 0.618,
            D + CD_range * 1.0,
            D + CD_range * 1.618
        ])
    
    return sorted(targets)

def detect_wyckoff_phases(data: pd.DataFrame, volume_window: int = 20,
                         price_window: int = 50) -> Dict[str, Any]:
    """
    STRAND Tool: Detect Wyckoff Market Phases (Accumulation/Distribution)
    
    Args:
        data: OHLCV DataFrame
        volume_window: Window for volume analysis
        price_window: Window for price range analysis
    
    Returns:
        Dictionary with detected Wyckoff phases
    """
    try:
        if len(data) < max(volume_window, price_window):
            return {
                "success": True,
                "phases": [],
                "signals": [],
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        phases = []
        
        # Calculate volume and price metrics
        data['volume_ma'] = data['volume'].rolling(window=volume_window).mean()
        data['price_range'] = data['high'] - data['low']
        data['price_range_ma'] = data['price_range'].rolling(window=price_window).mean()
        
        # Look for Wyckoff phases in recent data
        recent_data = data.tail(price_window * 2)
        
        # Phase 1: Accumulation/Distribution Detection
        for i in range(price_window, len(recent_data) - price_window):
            window_data = recent_data.iloc[i-price_window:i+price_window]
            
            # Calculate phase characteristics
            price_volatility = window_data['price_range'].std()
            volume_trend = linregress(range(len(window_data)), window_data['volume'].values)[0]
            price_trend = linregress(range(len(window_data)), window_data['close'].values)[0]
            
            # High volume, low price movement = potential accumulation/distribution
            avg_volume = window_data['volume'].mean()
            avg_range = window_data['price_range'].mean()
            
            volume_ratio = avg_volume / data['volume'].mean() if data['volume'].mean() > 0 else 1
            range_ratio = avg_range / data['price_range'].mean() if data['price_range'].mean() > 0 else 1
            
            # Wyckoff characteristics: High volume, narrow range
            if volume_ratio > 1.2 and range_ratio < 0.8:
                if abs(price_trend) < 0.001:  # Sideways movement
                    phase_type = "accumulation" if volume_trend > 0 else "distribution"
                    
                    phases.append({
                        "type": f"wyckoff_{phase_type}",
                        "bias": "bullish" if phase_type == "accumulation" else "bearish",
                        "start_index": i - price_window,
                        "end_index": i + price_window,
                        "volume_ratio": float(volume_ratio),
                        "range_ratio": float(range_ratio),
                        "volume_trend": float(volume_trend),
                        "price_trend": float(price_trend),
                        "confidence": min(volume_ratio * (2 - range_ratio), 2.0) / 2.0
                    })
        
        # Generate signals
        signals = []
        current_price = data['close'].iloc[-1]
        
        for phase in phases:
            if phase["confidence"] > 0.6:  # High confidence phases
                if phase["type"] == "wyckoff_accumulation":
                    signals.append({
                        "type": "wyckoff_accumulation_breakout",
                        "indicator": "Wyckoff Accumulation",
                        "signal": "buy",
                        "strength": "strong",
                        "price": current_price,
                        "confidence": phase["confidence"],
                        "note": "Potential breakout from Wyckoff accumulation phase"
                    })
                elif phase["type"] == "wyckoff_distribution":
                    signals.append({
                        "type": "wyckoff_distribution_breakdown",
                        "indicator": "Wyckoff Distribution",
                        "signal": "sell",
                        "strength": "strong",
                        "price": current_price,
                        "confidence": phase["confidence"],
                        "note": "Potential breakdown from Wyckoff distribution phase"
                    })
        
        return {
            "success": True,
            "phases": phases,
            "signals": signals,
            "parameters": {"volume_window": volume_window, "price_window": price_window},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Wyckoff phase detection failed: {e}")
        return {"success": False, "error": str(e)}

def analyze_advanced_patterns(data: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
    """
    STRAND Tool: Comprehensive Advanced Pattern Analysis
    
    Args:
        data: OHLCV DataFrame
        window: Window for pattern detection
    
    Returns:
        Comprehensive advanced pattern analysis
    """
    try:
        # Run all advanced pattern detection tools
        elliott_result = detect_elliott_wave_impulse(data, window)
        harmonic_result = detect_harmonic_patterns(data, window)
        wyckoff_result = detect_wyckoff_phases(data)
        
        # Combine all patterns and signals
        all_patterns = []
        all_signals = []
        
        for result in [elliott_result, harmonic_result, wyckoff_result]:
            if result["success"]:
                all_patterns.extend(result.get("patterns", []))
                all_signals.extend(result.get("signals", []))
        
        # Pattern analysis summary
        pattern_counts = {}
        high_confidence_patterns = []
        
        for pattern in all_patterns:
            pattern_type = pattern["type"]
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            # Collect high confidence patterns
            confidence = pattern.get("confidence", 0.5)
            if confidence > 0.7:
                high_confidence_patterns.append(pattern)
        
        return {
            "success": True,
            "patterns": {
                "all_patterns": all_patterns,
                "elliott_waves": elliott_result.get("patterns", []),
                "harmonic_patterns": harmonic_result.get("patterns", []),
                "wyckoff_phases": wyckoff_result.get("patterns", []),
                "high_confidence": high_confidence_patterns
            },
            "signals": all_signals,
            "analysis": {
                "total_patterns": len(all_patterns),
                "pattern_counts": pattern_counts,
                "high_confidence_count": len(high_confidence_patterns),
                "total_signals": len(all_signals)
            },
            "parameters": {"window": window},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Advanced pattern analysis failed: {e}")
        return {"success": False, "error": str(e)}

def create_advanced_pattern_tools() -> List[Dict[str, Any]]:
    """
    Create STRAND tools for advanced pattern recognition
    
    Returns:
        List of tool specifications for STRAND framework
    """
    return [
        {
            "name": "detect_elliott_wave_impulse",
            "function": detect_elliott_wave_impulse,
            "description": "Detect Elliott Wave impulse patterns (5-wave structure)",
            "parameters": {
                "data": "OHLCV DataFrame",
                "window": "Window for peak/trough detection"
            },
            "category": "technical_analysis",
            "skill": "harmonic_patterns"
        },
        {
            "name": "detect_harmonic_patterns",
            "function": detect_harmonic_patterns,
            "description": "Detect harmonic patterns (Gartley, Butterfly, Bat, Crab)",
            "parameters": {
                "data": "OHLCV DataFrame",
                "window": "Window for peak/trough detection",
                "tolerance": "Fibonacci ratio tolerance"
            },
            "category": "technical_analysis",
            "skill": "harmonic_patterns"
        },
        {
            "name": "detect_wyckoff_phases",
            "function": detect_wyckoff_phases,
            "description": "Detect Wyckoff accumulation and distribution phases",
            "parameters": {
                "data": "OHLCV DataFrame",
                "volume_window": "Window for volume analysis",
                "price_window": "Window for price range analysis"
            },
            "category": "technical_analysis",
            "skill": "harmonic_patterns"
        },
        {
            "name": "analyze_advanced_patterns",
            "function": analyze_advanced_patterns,
            "description": "Comprehensive advanced pattern analysis using Elliott Wave, Harmonic, and Wyckoff",
            "parameters": {
                "data": "OHLCV DataFrame",
                "window": "Window for pattern detection"
            },
            "category": "technical_analysis",
            "skill": "harmonic_patterns"
        }
    ]
