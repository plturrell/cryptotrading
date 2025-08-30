"""
Technical Indicators Service - Real calculations
No mock data - uses actual mathematical formulas
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..infrastructure.monitoring import get_logger

logger = get_logger("services.technical_indicators")


class TechnicalIndicatorsService:
    """Real technical indicators calculations"""

    def __init__(self):
        self.indicators_cache = {}

    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return []

        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1 : i + 1]) / period
            sma_values.append(sma)

        return sma_values

    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return []

        multiplier = 2 / (period + 1)
        ema_values = []

        # Start with SMA for first value
        ema = sum(prices[:period]) / period
        ema_values.append(ema)

        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = (prices[i] - ema) * multiplier + ema
            ema_values.append(ema)

        return ema_values

    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return []

        # Calculate price changes
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]

        rsi_values = []

        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            # Smoothed averages
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

            # Calculate RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

        return rsi_values

    def calculate_macd(
        self,
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict[str, List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow_period:
            return {"macd": [], "signal": [], "histogram": []}

        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)

        # Align EMAs (slow EMA starts later)
        start_diff = len(ema_fast) - len(ema_slow)
        if start_diff > 0:
            ema_fast = ema_fast[start_diff:]

        # Calculate MACD line
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(ema_slow))]

        # Calculate signal line (EMA of MACD)
        signal_line = self.calculate_ema(macd_line, signal_period)

        # Calculate histogram
        macd_aligned = macd_line[len(macd_line) - len(signal_line) :]
        histogram = [macd_aligned[i] - signal_line[i] for i in range(len(signal_line))]

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def calculate_bollinger_bands(
        self, prices: List[float], period: int = 20, std_dev: float = 2
    ) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {"upper": [], "middle": [], "lower": []}

        middle_band = self.calculate_sma(prices, period)
        upper_band = []
        lower_band = []

        for i in range(period - 1, len(prices)):
            price_slice = prices[i - period + 1 : i + 1]
            std = np.std(price_slice)
            sma = middle_band[i - period + 1]

            upper_band.append(sma + (std_dev * std))
            lower_band.append(sma - (std_dev * std))

        return {"upper": upper_band, "middle": middle_band, "lower": lower_band}

    def calculate_volatility(self, prices: List[float], period: int = 20) -> List[float]:
        """Calculate price volatility (rolling standard deviation)"""
        if len(prices) < period:
            return []

        volatilities = []
        for i in range(period - 1, len(prices)):
            price_slice = prices[i - period + 1 : i + 1]
            returns = [
                (price_slice[j] - price_slice[j - 1]) / price_slice[j - 1]
                for j in range(1, len(price_slice))
            ]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            volatilities.append(volatility)

        return volatilities

    def calculate_support_resistance(
        self, prices: List[float], window: int = 20
    ) -> Dict[str, List[float]]:
        """Calculate support and resistance levels"""
        if len(prices) < window * 2:
            return {"support": [], "resistance": []}

        support_levels = []
        resistance_levels = []

        for i in range(window, len(prices) - window):
            price_window = prices[i - window : i + window + 1]
            current_price = prices[i]

            # Local minima (support)
            if current_price == min(price_window):
                support_levels.append(current_price)

            # Local maxima (resistance)
            if current_price == max(price_window):
                resistance_levels.append(current_price)

        # Remove duplicates and sort
        support_levels = sorted(list(set(support_levels)))
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)

        return {
            "support": support_levels[:5],  # Top 5 support levels
            "resistance": resistance_levels[:5],  # Top 5 resistance levels
        }

    def calculate_all_indicators(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate all technical indicators for price data"""
        if not price_data:
            return {}

        # Extract price arrays
        close_prices = [float(item.get("Close", item.get("price", 0))) for item in price_data]
        high_prices = [float(item.get("High", item.get("price", 0))) for item in price_data]
        low_prices = [float(item.get("Low", item.get("price", 0))) for item in price_data]
        volumes = [float(item.get("Volume", item.get("volume", 0))) for item in price_data]

        if not close_prices or all(p == 0 for p in close_prices):
            logger.warning("No valid price data for indicator calculation")
            return {}

        try:
            # Calculate all indicators
            indicators = {
                "sma_20": self.calculate_sma(close_prices, 20),
                "sma_50": self.calculate_sma(close_prices, 50),
                "ema_12": self.calculate_ema(close_prices, 12),
                "ema_26": self.calculate_ema(close_prices, 26),
                "rsi_14": self.calculate_rsi(close_prices, 14),
                "macd": self.calculate_macd(close_prices),
                "bollinger_bands": self.calculate_bollinger_bands(close_prices),
                "volatility": self.calculate_volatility(close_prices),
                "support_resistance": self.calculate_support_resistance(close_prices),
            }

            # Get latest values for current indicators
            current_indicators = {}
            for name, values in indicators.items():
                if isinstance(values, list) and values:
                    current_indicators[f"current_{name}"] = values[-1]
                elif isinstance(values, dict):
                    current_dict = {}
                    for key, val_list in values.items():
                        if isinstance(val_list, list) and val_list:
                            current_dict[key] = val_list[-1]
                    if current_dict:
                        current_indicators[f"current_{name}"] = current_dict

            return {
                "indicators": indicators,
                "current": current_indicators,
                "data_points": len(close_prices),
                "calculated_at": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}

    def get_trend_analysis(self, price_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze trend based on technical indicators"""
        indicators = self.calculate_all_indicators(price_data)

        if not indicators or "current" not in indicators:
            return {"short_term": "unknown", "medium_term": "unknown", "long_term": "unknown"}

        current = indicators["current"]

        # Short-term trend (RSI and recent price action)
        short_term = "neutral"
        if "current_rsi_14" in current:
            rsi = current["current_rsi_14"]
            if rsi > 70:
                short_term = "overbought"
            elif rsi < 30:
                short_term = "oversold"
            elif rsi > 50:
                short_term = "bullish"
            else:
                short_term = "bearish"

        # Medium-term trend (MACD)
        medium_term = "neutral"
        if "current_macd" in current and isinstance(current["current_macd"], dict):
            macd_data = current["current_macd"]
            if "histogram" in macd_data and macd_data["histogram"] > 0:
                medium_term = "bullish"
            elif "histogram" in macd_data and macd_data["histogram"] < 0:
                medium_term = "bearish"

        # Long-term trend (SMA comparison)
        long_term = "neutral"
        if "current_sma_20" in current and "current_sma_50" in current:
            sma_20 = current["current_sma_20"]
            sma_50 = current["current_sma_50"]
            if sma_20 > sma_50:
                long_term = "bullish"
            else:
                long_term = "bearish"

        return {"short_term": short_term, "medium_term": medium_term, "long_term": long_term}
