"""
Vercel API route for batch ML predictions
"""

import os
import sys
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from cryptotrading.core.ml.model_server import model_serving_api
    import asyncio
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")

def handler(request):
    """
    Vercel serverless function handler for batch ML predictions
    POST /api/ml/batch
    Body: {"symbols": ["BTC", "ETH"], "horizon": "24h"}
    """
    try:
        # Parse request body
        if request.method == 'POST':
            body = request.get_json()
            symbols = body.get('symbols', ['BTC', 'ETH'])
            horizon = body.get('horizon', '24h')
        else:
            # GET request fallback
            symbols = request.args.get('symbols', 'BTC,ETH').split(',')
            horizon = request.args.get('horizon', '24h')
        
        # Make predictions for each symbol
        predictions = []
        
        # Process all symbols in parallel
        async def process_symbol(symbol):
            symbol = symbol.upper().strip()
            
            # Get REAL market data or fail
            try:
                import yfinance as yf
                yahoo_symbol = f"{symbol}-USD" if symbol in ['BTC', 'ETH'] else symbol
                ticker = yf.Ticker(yahoo_symbol)
                
                # Get historical data
                history = ticker.history(period="30d", interval="1h")
                
                if history.empty:
                    raise ValueError(f"No historical data available for {symbol}")
                
                # Extract data
                close_prices = history['Close'].values
                volumes = history['Volume'].values
                current_price = float(close_prices[-1])
                
                # Calculate features
                features = {
                    'price': current_price,
                    'price_change_24h': float((close_prices[-1] - close_prices[-24]) / close_prices[-24] * 100),
                    'volume': float(volumes[-1]),
                    'volume_ratio': float(volumes[-1] / np.mean(volumes[-20:])) if len(volumes) >= 20 else 1.0,
                    'rsi_14': calculate_rsi(close_prices, 14),
                    'volatility_20': float(np.std(np.diff(close_prices[-20:]) / close_prices[-21:-1]) * np.sqrt(24 * 365))
                }
                
                # Add SMA features
                for period in [5, 10, 20]:
                    if len(close_prices) > period:
                        sma = float(np.mean(close_prices[-period:]))
                        features[f'sma_{period}'] = sma
                        features[f'price_to_sma_{period}'] = float(close_prices[-1] / sma)
                
                # Make prediction using model serving API
                result = await model_serving_api.predict(symbol, features)
                
                # Format response
                if 'error' not in result:
                    prediction_value = result.get('prediction', 0)
                    if isinstance(prediction_value, (list, np.ndarray)):
                        prediction_value = float(prediction_value[0])
                    else:
                        prediction_value = float(prediction_value)
                    
                    predicted_price = current_price * (1 + prediction_value / 100)
                    
                    return {
                        'symbol': symbol,
                        'current_price': current_price,
                        'predicted_price': round(predicted_price, 2),
                        'price_change_percent': round(prediction_value, 2),
                        'confidence': result.get('confidence', 0),
                        'horizon': horizon,
                        'model_version': result.get('model_version', 'unknown'),
                        'cached': result.get('cached', False),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'symbol': symbol,
                        'error': result['error'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except Exception as e:
                return {
                    'symbol': symbol,
                    'error': f'Failed to get data: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Process all symbols concurrently
        predictions = asyncio.run(
            asyncio.gather(*[process_symbol(symbol) for symbol in symbols])
        )
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'public, s-maxage=300'
            },
            'body': json.dumps({
                'predictions': predictions,
                'count': len(predictions),
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
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
            self.method = 'POST'
            self.args = {}
            
        def get_json(self):
            return {'symbols': ['BTC', 'ETH'], 'horizon': '24h'}
    
    result = handler(MockRequest())
    print(json.dumps(result, indent=2))