"""
WSGI entry point for rex.com on GoDaddy hosting
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Create data directory if it doesn't exist
os.makedirs(project_root / 'data', exist_ok=True)

from flask import Flask, render_template, send_from_directory, request
from flask_cors import CORS
from flask_restx import Api, Resource
from dotenv import load_dotenv

# Import observability components
try:
    from cryptotrading.infrastructure.monitoring.dashboard import register_observability_routes, create_dashboard_route
    from cryptotrading.infrastructure.monitoring.tracer import instrument_flask_app
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

class CustomJSONEncoder:
    """Custom JSON encoder to handle pandas and numpy types"""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, np.datetime64):
            return pd.to_datetime(obj).strftime('%Y-%m-%d')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif pd.isna(obj):
            return None
        return str(obj)

# Check if running on Vercel
IS_VERCEL = os.getenv('VERCEL') == '1'

# Load environment variables
load_dotenv()

# Initialize database
try:
    from cryptotrading.data.database import get_db
    db = get_db()
except Exception as e:
    print(f"Database initialization warning: {e}")
    db = None

app = Flask(__name__, 
           static_folder='webapp',
           template_folder='webapp')

# Initialize observability
if OBSERVABILITY_AVAILABLE:
    try:
        # Instrument Flask app for tracing
        instrument_flask_app(app)
        
        # Register observability routes
        register_observability_routes(app)
        create_dashboard_route(app)
        
        print("✅ Observability enabled: /observability/dashboard.html")
    except Exception as e:
        print(f"⚠️  Observability setup failed: {e}")
        OBSERVABILITY_AVAILABLE = False

# CORS configuration for rex.com
CORS(app, origins=[
    'https://rex.com',
    'https://www.rex.com',
    'https://xn--e1afmkfd.com',  # Punycode for rex.com
    'https://www.xn--e1afmkfd.com'
])

# API configuration
api = Api(
    app,
    version='1.0',
    title='rex.com API',
    description='Professional cryptocurrency trading platform API',
    doc='/api/'
)

# SAP UI5 routes
@app.route('/')
def index():
    """Serve SAP UI5 application"""
    return send_from_directory('webapp', 'index.html')

@app.route('/manifest.json')
def manifest():
    """Serve SAP UI5 manifest"""
    return send_from_directory('.', 'manifest.json')

@app.route('/webapp/<path:filename>')
def webapp_files(filename):
    """Serve SAP UI5 webapp files"""
    return send_from_directory('webapp', filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'platform': 'rex.com', 'version': '0.1.0'}

# API endpoints
@api.route('/api/trading/status')
class TradingStatus(Resource):
    def get(self):
        """Get trading system status"""
        return {'status': 'active', 'platform': 'rex.com'}

@api.route('/api/market/data')
class MarketData(Resource):
    def get(self):
        """Get market data from Yahoo Finance with full observability"""
        from cryptotrading.infrastructure.monitoring import get_logger, get_business_metrics, trace_context, track_errors, ErrorSeverity, ErrorCategory
        
        logger = get_logger("api.market")
        business_metrics = get_business_metrics()
        
        symbol = request.args.get('symbol', 'BTC')
        start_time = time.time()
        
        with trace_context(f"market_data_api_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("api.endpoint", "/api/market/data")
                
                logger.info(f"Fetching market data for {symbol}", extra={
                    "symbol": symbol,
                    "endpoint": "/api/market/data",
                    "method": "GET"
                })
                
                from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
                
                yahoo_client = YahooFinanceClient()
                data = yahoo_client.get_realtime_price(symbol)
                
                if data is None:
                    error_msg = f"No price data available for {symbol}"
                    logger.warning(error_msg, extra={"symbol": symbol})
                    
                    duration_ms = (time.time() - start_time) * 1000
                    business_metrics.track_api_request("/api/market/data", "GET", 503, duration_ms)
                    
                    return {"error": error_msg}, 503
                
                # Track successful API call
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request("/api/market/data", "GET", 200, duration_ms)
                
                logger.info(f"Successfully retrieved market data for {symbol}", extra={
                    "symbol": symbol,
                    "price": data.get("price"),
                    "volume": data.get("volume")
                })
                
                span.set_attribute("price", str(data.get("price", 0)))
                span.set_attribute("success", "true")
                
                return data
                
            except Exception as e:
                logger.error(f"Failed to fetch market data for {symbol}", error=e, extra={
                    "symbol": symbol,
                    "endpoint": "/api/market/data"
                })
                
                # Track error
                from cryptotrading.infrastructure.monitoring import get_error_tracker
                error_tracker = get_error_tracker()
                error_tracker.track_error(e, severity=ErrorSeverity.HIGH, category=ErrorCategory.API_ERROR)
                
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request("/api/market/data", "GET", 500, duration_ms)
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                return {"error": f"Failed to fetch market data: {str(e)}"}, 500

@api.route('/api/ai/analyze')
class AIAnalysis(Resource):
    def post(self):
        """AI market analysis using Claude-4-Sonnet with full observability"""
        from cryptotrading.infrastructure.monitoring import get_logger, get_business_metrics, trace_context, ErrorSeverity, ErrorCategory
        
        logger = get_logger("api.ai.analyze")
        business_metrics = get_business_metrics()
        
        start_time = time.time()
        data = api.payload
        symbol = data.get('symbol', 'BTC')
        
        with trace_context(f"ai_analysis_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("ai.model", "claude-4-sonnet")
                span.set_attribute("api.endpoint", "/api/ai/analyze")
                
                logger.info(f"Starting AI analysis for {symbol}", extra={
                    "symbol": symbol,
                    "model": "claude-4-sonnet",
                    "endpoint": "/api/ai/analyze",
                    "method": "POST"
                })
                
                from cryptotrading.core.ai import AIGatewayClient
                
                ai = AIGatewayClient()
                analysis = ai.analyze_market(data)
                
                # Track successful AI operation
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request("/api/ai/analyze", "POST", 200, duration_ms)
                business_metrics.track_ai_operation(
                    operation="market_analysis",
                    model="claude-4-sonnet",
                    symbol=symbol,
                    success=True,
                    duration_ms=duration_ms
                )
                
                logger.info(f"AI analysis completed for {symbol}", extra={
                    "symbol": symbol,
                    "analysis_length": len(str(analysis)) if analysis else 0,
                    "duration_ms": duration_ms
                })
                
                span.set_attribute("success", "true")
                span.set_attribute("analysis_length", len(str(analysis)) if analysis else 0)
                
                return {
                    'analysis': analysis,
                    'model': 'claude-4-sonnet',
                    'symbol': symbol,
                    'duration_ms': duration_ms
                }
                
            except Exception as e:
                logger.error(f"AI analysis failed for {symbol}", error=e, extra={
                    "symbol": symbol,
                    "model": "claude-4-sonnet",
                    "endpoint": "/api/ai/analyze"
                })
                
                # Track error
                from cryptotrading.infrastructure.monitoring import get_error_tracker
                error_tracker = get_error_tracker()
                error_tracker.track_error(e, severity=ErrorSeverity.HIGH, category=ErrorCategory.AI_ERROR)
                
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request("/api/ai/analyze", "POST", 500, duration_ms)
                business_metrics.track_ai_operation(
                    operation="market_analysis",
                    model="claude-4-sonnet",
                    symbol=symbol,
                    success=False,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                return {"error": str(e)}, 500

@api.route('/api/ai/news/<string:symbol>')
class CryptoNews(Resource):
    def get(self, symbol):
        """Get real-time crypto news via Perplexity"""
        from cryptotrading.core.ml.perplexity import PerplexityClient
        
        perplexity = PerplexityClient()
        news = perplexity.search_crypto_news(symbol.upper())
        return news

@api.route('/api/ai/signals/<string:symbol>')
class TradingSignals(Resource):
    def get(self, symbol):
        """Get AI trading signals via Perplexity"""
        from cryptotrading.core.ml.perplexity import PerplexityClient
        
        timeframe = '4h'  # Default timeframe
        perplexity = PerplexityClient()
        signals = perplexity.get_trading_signals(symbol.upper(), timeframe)
        return signals

@api.route('/api/wallet/balance')
class WalletBalance(Resource):
    def get(self):
        """Wallet balance - MetaMask client removed"""
        return {
            "error": "Wallet balance functionality requires blockchain integration",
            "message": "MetaMask client was removed as it required Infura API key",
            "suggested_alternatives": [
                "Use Web3.py directly with your own Infura/Alchemy key",
                "Use Etherscan API for wallet balance queries"
            ]
        }

@api.route('/api/wallet/monitor')
class WalletMonitor(Resource):
    def get(self):
        """Wallet monitoring - MetaMask client removed"""
        return {
            "error": "Wallet monitoring functionality requires blockchain integration",
            "message": "MetaMask client was removed as it required Infura API key",
            "suggested_alternatives": [
                "Use Web3.py directly with your own Infura/Alchemy key",
                "Integrate with wallet monitoring services like Moralis"
            ]
        }

@api.route('/api/defi/opportunities')
class DeFiOpportunities(Resource):
    def get(self):
        """DeFi opportunities - blockchain clients removed"""
        return {
            "error": "DeFi analysis functionality requires blockchain integration",
            "message": "Ethereum client was removed as it contained fake DeFi data",
            "suggested_alternatives": [
                "Use real DeFi protocols like Aave or Compound APIs",
                "Integrate with DeFi aggregators like DeFiPulse or Zapper"
            ]
        }

@api.route('/api/wallet/gas')
class GasPrice(Resource):
    def get(self):
        """Gas price information - blockchain clients removed"""
        return {
            "error": "Gas price functionality requires blockchain integration",
            "message": "MetaMask and Ethereum clients were removed as they required external API keys",
            "suggested_alternatives": [
                "Use CoinGecko API for ETH price information",
                "Integrate with a gas tracking service like EtherScan"
            ]
        }

@api.route('/api/market/overview')
class MarketOverview(Resource):
    def get(self):
        """Get market overview for multiple symbols"""
        from cryptotrading.data.market_data import MarketDataAggregator
        
        symbols = request.args.get('symbols', 'bitcoin,ethereum,binancecoin').split(',')
        
        aggregator = MarketDataAggregator()
        overview = aggregator.get_market_overview(symbols)
        return overview

@api.route('/api/market/dex/opportunities')
class DexOpportunities(Resource):
    def get(self):
        """Get DEX trading opportunities"""
        from cryptotrading.data.market_data import MarketDataAggregator
        
        min_liquidity = float(request.args.get('min_liquidity', 10000))
        
        aggregator = MarketDataAggregator()
        opportunities = aggregator.get_dex_opportunities(min_liquidity)
        return {"opportunities": opportunities, "count": len(opportunities)}

@api.route('/api/market/dex/trending')
class DexTrending(Resource):
    def get(self):
        """Get trending DEX pools"""
        from cryptotrading.data.market_data import GeckoTerminalClient
        
        network = request.args.get('network', None)
        
        client = GeckoTerminalClient()
        trending = client.get_trending_pools(network)
        return trending

@api.route('/api/market/dex/pool/<string:network>/<string:address>')
class DexPool(Resource):
    def get(self, network, address):
        """Get specific DEX pool data"""
        from cryptotrading.data.market_data import GeckoTerminalClient
        
        client = GeckoTerminalClient()
        pool_data = client.get_pool_by_address(network, address)
        volume_data = client.get_pool_volume(network, address)
        
        return {
            "pool": pool_data,
            "volume": volume_data
        }

@api.route('/api/market/historical/<string:symbol>')
class HistoricalData(Resource):
    def get(self, symbol):
        """Get historical market data from Yahoo Finance with full observability"""
        from cryptotrading.infrastructure.monitoring import get_logger, get_business_metrics, trace_context, ErrorSeverity, ErrorCategory
        from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
        from datetime import datetime, timedelta
        
        logger = get_logger("api.historical")
        business_metrics = get_business_metrics()
        
        days = int(request.args.get('days', 7))
        start_time = time.time()
        
        with trace_context(f"historical_data_api_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("days", days)
                span.set_attribute("api.endpoint", "/api/market/historical")
                
                logger.info(f"Fetching {days} days of historical data for {symbol}", extra={
                    "symbol": symbol,
                    "days": days,
                    "endpoint": "/api/market/historical",
                    "method": "GET"
                })
                
                yahoo_client = YahooFinanceClient()
                
                # Calculate date range
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                # Download historical data
                df = yahoo_client.download_data(symbol, start_date, end_date, save=False)
                
                if df is None or df.empty:
                    error_msg = f"No historical data available for {symbol}"
                    logger.warning(error_msg, extra={"symbol": symbol, "days": days})
                    
                    duration_ms = (time.time() - start_time) * 1000
                    business_metrics.track_api_request("/api/market/historical", "GET", 404, duration_ms)
                    
                    span.set_attribute("success", "false")
                    span.set_attribute("error_message", error_msg)
                    
                    return {"error": error_msg}, 404
                
                # Convert to JSON-serializable format
                import numpy as np
                
                # Reset index to make Date a column
                df_clean = df.reset_index()
                
                # Convert DataFrame to records with proper JSON serialization
                records = []
                for _, row in df_clean.iterrows():
                    record = {}
                    for col, value in row.items():
                        if pd.isna(value):
                            record[col] = None
                        elif isinstance(value, (pd.Timestamp, np.datetime64)):
                            record[col] = pd.to_datetime(value).strftime('%Y-%m-%d')
                        elif isinstance(value, np.integer):
                            record[col] = int(value)
                        elif isinstance(value, np.floating):
                            record[col] = float(value)
                        else:
                            record[col] = value
                    records.append(record)
                
                historical_data = {
                    "symbol": symbol,
                    "days": days,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data": records,
                    "count": len(df)
                }
                
                # Track successful API call
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request("/api/market/historical", "GET", 200, duration_ms)
                business_metrics.track_data_processing(
                    source="yahoo_finance",
                    symbol=symbol,
                    records_processed=len(df),
                    success=True,
                    duration_ms=duration_ms
                )
                
                logger.info(f"Successfully retrieved {len(df)} historical records for {symbol}", extra={
                    "symbol": symbol,
                    "records_count": len(df),
                    "days": days,
                    "date_range": {"start": start_date, "end": end_date}
                })
                
                span.set_attribute("records_count", len(df))
                span.set_attribute("success", "true")
                
                return historical_data
                
            except Exception as e:
                logger.error(f"Failed to fetch historical data for {symbol}", error=e, extra={
                    "symbol": symbol,
                    "days": days,
                    "endpoint": "/api/market/historical"
                })
                
                # Track error
                from cryptotrading.infrastructure.monitoring import get_error_tracker
                error_tracker = get_error_tracker()
                error_tracker.track_error(e, severity=ErrorSeverity.HIGH, category=ErrorCategory.API_ERROR)
                
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request("/api/market/historical", "GET", 500, duration_ms)
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                return {"error": f"Failed to fetch historical data: {str(e)}"}, 500

@api.route('/api/data/download/<string:symbol>')
class DownloadHistoricalData(Resource):
    def get(self, symbol):
        """Download historical data for model training"""
        from cryptotrading.data.historical import HistoricalDataAggregator
        
        source = request.args.get('source', 'all')
        
        aggregator = HistoricalDataAggregator()
        
        if source == 'all':
            data = aggregator.download_all_sources(symbol)
            return {
                "symbol": symbol,
                "sources": list(data.keys()),
                "total_rows": sum(len(df) for df in data.values()),
                "status": "downloaded"
            }
        else:
            return {"error": f"Source {source} not supported"}

@api.route('/api/data/training/<string:symbol>')
class TrainingDataset(Resource):
    def get(self, symbol):
        """Get or create training dataset with indicators"""
        from cryptotrading.data.historical import HistoricalDataAggregator
        
        lookback_days = int(request.args.get('days', 365))
        
        aggregator = HistoricalDataAggregator()
        
        # Try to load existing dataset first
        df = aggregator.load_training_dataset(symbol)
        
        if df is None:
            # Create new dataset
            df = aggregator.create_training_dataset(symbol, lookback_days=lookback_days)
        
        if df.empty:
            return {"error": "No data available"}
        
        return {
            "symbol": symbol,
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            },
            "indicators": [col for col in df.columns if any(ind in col for ind in ['sma', 'ema', 'rsi', 'macd', 'bb'])]
        }

@api.route('/api/data/available')
class AvailableDatasets(Resource):
    def get(self):
        """List available training datasets"""
        from cryptotrading.data.historical import HistoricalDataAggregator
        
        aggregator = HistoricalDataAggregator()
        datasets = aggregator.get_available_datasets()
        
        return {
            "datasets": datasets,
            "count": len(datasets)
        }

@api.route('/api/limits')
class APILimits(Resource):
    def get(self):
        """Get current API rate limit status"""
        from cryptotrading.utils import rate_limiter
        
        return {
            "limits": rate_limiter.get_all_limits(),
            "timestamp": datetime.now().isoformat()
        }

@api.route('/api/ai/strategy')
class AIStrategy(Resource):
    def post(self):
        """Generate personalized trading strategy using Claude-4"""
        try:
            from cryptotrading.core.ai import AIGatewayClient
            
            user_profile = api.payload
            ai = AIGatewayClient()
            strategy = ai.generate_trading_strategy(user_profile)
            
            # Store in blob storage
            try:
                from cryptotrading.data.storage import put_json_blob
                blob_result = put_json_blob(
                    f"strategies/user_{user_profile.get('user_id', 'anonymous')}.json",
                    strategy
                )
                strategy['storage_url'] = blob_result.get('url')
            except:
                pass
            
            return strategy
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/ai/sentiment')
class AISentiment(Resource):
    def post(self):
        """Analyze news sentiment using Claude-4"""
        try:
            from cryptotrading.core.ai import AIGatewayClient
            
            news_items = api.payload.get('news', [])
            ai = AIGatewayClient()
            sentiment = ai.analyze_news_sentiment(news_items)
            
            return sentiment
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/storage/signals/<string:symbol>')
class StoredSignals(Resource):
    def get(self, symbol):
        """Get stored trading signals from Vercel Blob"""
        try:
            from cryptotrading.data.storage import VercelBlobClient
            
            client = VercelBlobClient()
            signals = client.get_latest_signals(symbol)
            
            return {
                "symbol": symbol,
                "signals": signals,
                "count": len(signals)
            }
        except Exception as e:
            return {"error": str(e)}, 500
    
    def post(self, symbol):
        """Store new trading signal in Vercel Blob"""
        try:
            from cryptotrading.data.storage import VercelBlobClient
            
            signal_data = api.payload
            client = VercelBlobClient()
            result = client.store_trading_signal(symbol, signal_data)
            
            return result
        except Exception as e:
            return {"error": str(e)}, 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return {'error': 'Not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500

if __name__ == '__main__':
    # Development server
    app.run(debug=False, host='0.0.0.0', port=5001)

# WSGI application for GoDaddy
application = app