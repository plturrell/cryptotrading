"""
Vercel API route for model serving with full quality
Uses the production model serving infrastructure
"""

import os
import sys
import json
from datetime import datetime
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cryptotrading.core.ml.model_server import model_serving_api
import yfinance as yf
import numpy as np
import pandas as pd


def handler(request):
    """
    Vercel serverless function handler for model serving
    POST /api/ml/serve

    Serves predictions using full model quality
    """
    try:
        # Parse request body
        body = json.loads(request.body) if request.body else {}

        symbol = body.get("symbol", "BTC").upper()
        model_id = body.get("model_id", symbol)
        horizon = body.get("horizon", "24h")

        # Get real market data for features
        try:
            yahoo_symbol = f"{symbol}-USD" if symbol in ["BTC", "ETH"] else symbol
            ticker = yf.Ticker(yahoo_symbol)

            # Get historical data
            history = ticker.history(period="30d", interval="1h")

            if history.empty:
                raise ValueError(f"No market data available for {symbol}")

            # Prepare features from real data
            close_prices = history["Close"].values
            volumes = history["Volume"].values

            # Calculate technical indicators
            features = {
                "price": float(close_prices[-1]),
                "price_change_24h": float(
                    (close_prices[-1] - close_prices[-24]) / close_prices[-24] * 100
                ),
                "volume": float(volumes[-1]),
                "volume_ratio": float(volumes[-1] / np.mean(volumes[-20:]))
                if len(volumes) >= 20
                else 1.0,
                "rsi_14": calculate_rsi(close_prices, 14),
                "macd_signal": calculate_macd_signal(close_prices),
                "bb_position": calculate_bollinger_position(close_prices),
                "volatility_20": float(
                    np.std(np.diff(close_prices[-20:]) / close_prices[-21:-1]) * np.sqrt(24 * 365)
                ),
            }

            # Add more features as needed
            for i in [5, 10, 20]:
                if len(close_prices) > i:
                    features[f"sma_{i}"] = float(np.mean(close_prices[-i:]))
                    features[f"price_to_sma_{i}"] = float(close_prices[-1] / features[f"sma_{i}"])

        except Exception as e:
            return {
                "statusCode": 503,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {
                        "error": f"Unable to fetch real market data: {str(e)}",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
            }

        # Make prediction using model serving API
        result = asyncio.run(model_serving_api.predict(model_id, features))

        # Check if prediction was successful
        if "error" in result:
            return {
                "statusCode": 503,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result),
            }

        # Format response
        prediction_data = result.get("prediction", {})

        response = {
            "symbol": symbol,
            "model_id": model_id,
            "current_price": features["price"],
            "predicted_price": prediction_data.get("price", features["price"]),
            "price_change_percent": prediction_data.get("change_percent", 0),
            "confidence": result.get("confidence", 0),
            "horizon": horizon,
            "features_used": len(features),
            "model_version": result.get("model_version", "unknown"),
            "cached": result.get("cached", False),
            "latency_ms": result.get("latency_ms", 0),
            "timestamp": datetime.now().isoformat(),
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "public, s-maxage=60",  # Cache for 1 minute
            },
            "body": json.dumps(response),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "timestamp": datetime.now().isoformat()}),
        }


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    if len(prices) < period:
        return 50.0

    deltas = np.diff(prices)
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]

    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)


def calculate_macd_signal(prices):
    """Calculate MACD signal"""
    if len(prices) < 26:
        return 0.0

    # Calculate EMAs
    ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
    ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]

    macd = ema_12 - ema_26
    return float(macd)


def calculate_bollinger_position(prices, period=20):
    """Calculate position within Bollinger Bands"""
    if len(prices) < period:
        return 0.5

    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])

    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)

    current_price = prices[-1]

    if upper_band == lower_band:
        return 0.5

    position = (current_price - lower_band) / (upper_band - lower_band)
    return float(max(0, min(1, position)))


# For testing locally
if __name__ == "__main__":

    class MockRequest:
        def __init__(self):
            self.body = json.dumps({"symbol": "BTC", "horizon": "24h"})

    result = handler(MockRequest())
    print(json.dumps(result, indent=2))
