"""
Vercel API route for ML predictions
"""

import os
import sys
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from cryptotrading.core.ml.model_server import model_serving_api
    import asyncio
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports for basic functionality
    pass


def handler(request):
    """
    Vercel serverless function handler for ML predictions
    GET /api/ml/predict?symbol=BTC&horizon=24h
    """
    try:
        # Parse query parameters
        symbol = request.args.get("symbol", "BTC").upper()
        horizon = request.args.get("horizon", "24h")

        # Get real-time price data
        try:
            # Import locally to avoid heavy dependencies in edge function
            import yfinance as yf

            # Map symbol to Yahoo Finance format
            yahoo_symbol = f"{symbol}-USD" if symbol in ["BTC", "ETH"] else symbol
            ticker = yf.Ticker(yahoo_symbol)

            # Get recent history for features
            history = ticker.history(period="30d", interval="1h")

            if history.empty:
                raise ValueError(f"No historical data available for {symbol}")

            # Extract price and volume data
            close_prices = history["Close"].values
            volumes = history["Volume"].values
            current_price = float(close_prices[-1])

        except Exception as e:
            # FAIL if we can't get real data
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

        # Calculate features from real data
        features = {
            "price": current_price,
            "price_change_24h": float(
                (close_prices[-1] - close_prices[-24]) / close_prices[-24] * 100
            ),
            "volume": float(volumes[-1]),
            "volume_ratio": float(volumes[-1] / np.mean(volumes[-20:]))
            if len(volumes) >= 20
            else 1.0,
            "rsi_14": calculate_rsi(close_prices, 14),
            "volatility_20": float(
                np.std(np.diff(close_prices[-20:]) / close_prices[-21:-1]) * np.sqrt(24 * 365)
            ),
        }

        # Add SMA features
        for period in [5, 10, 20]:
            if len(close_prices) > period:
                sma = float(np.mean(close_prices[-period:]))
                features[f"sma_{period}"] = sma
                features[f"price_to_sma_{period}"] = float(close_prices[-1] / sma)

        # Make prediction using model serving API
        result = asyncio.run(model_serving_api.predict(symbol, features))

        # Check if prediction was successful
        if "error" in result:
            return {
                "statusCode": 503,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {
                        "error": result["error"],
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
            }

        # Calculate predicted price
        prediction_value = result.get("prediction", 0)
        if isinstance(prediction_value, (list, np.ndarray)):
            prediction_value = float(prediction_value[0])
        else:
            prediction_value = float(prediction_value)

        predicted_price = current_price * (1 + prediction_value / 100)

        # Format response
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": round(predicted_price, 2),
            "price_change_percent": round(prediction_value, 2),
            "confidence": result.get("confidence", 0),
            "horizon": horizon,
            "features_used": len(features),
            "model_version": result.get("model_version", "unknown"),
            "cached": result.get("cached", False),
            "timestamp": datetime.now().isoformat(),
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "public, s-maxage=300",
            },
            "body": json.dumps(result),
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


# For testing locally
if __name__ == "__main__":

    class MockRequest:
        def __init__(self):
            self.args = {"symbol": "BTC", "horizon": "24h"}

    result = handler(MockRequest())
    print(json.dumps(result, indent=2))
