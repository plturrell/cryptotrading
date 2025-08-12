"""
WSGI entry point for рекс.com on GoDaddy hosting
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

# Load environment variables
load_dotenv()

# Initialize database
try:
    from src.рекс.database import get_db
    db = get_db()
except Exception as e:
    print(f"Database initialization warning: {e}")
    db = None

app = Flask(__name__, 
           static_folder='webapp',
           template_folder='webapp')

# CORS configuration for рекс.com
CORS(app, origins=[
    'https://рекс.com',
    'https://www.рекс.com',
    'https://xn--e1afmkfd.com',  # Punycode for рекс.com
    'https://www.xn--e1afmkfd.com'
])

# API configuration
api = Api(
    app,
    version='1.0',
    title='рекс.com API',
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
    return {'status': 'healthy', 'platform': 'рекс.com', 'version': '0.1.0'}

# API endpoints
@api.route('/api/trading/status')
class TradingStatus(Resource):
    def get(self):
        """Get trading system status"""
        return {'status': 'active', 'platform': 'рекс.com'}

@api.route('/api/market/data')
class MarketData(Resource):
    def get(self):
        """Get aggregated market data from multiple sources"""
        try:
            from src.рекс.market_data import MarketDataAggregator
            
            symbol = request.args.get('symbol', 'bitcoin')
            network = request.args.get('network', None)
            
            aggregator = MarketDataAggregator()
            data = aggregator.get_aggregated_price(symbol, network)
            
            if "error" in data:
                return {"error": data["error"]}, 503
                
            return data
        except Exception as e:
            return {"error": f"Failed to fetch market data: {str(e)}"}, 500

@api.route('/api/ai/analyze')
class AIAnalysis(Resource):
    def post(self):
        """AI market analysis using DeepSeek R1"""
        from src.рекс.ml.deepseek import DeepSeekR1
        
        data = api.payload
        ai = DeepSeekR1()
        analysis = ai.analyze_market(data)
        
        # Save to database if available
        if db:
            try:
                db.save_ai_analysis(
                    symbol=data.get('symbol', 'BTC'),
                    model='deepseek-r1',
                    analysis_type='market',
                    analysis=analysis
                )
            except Exception as e:
                print(f"Database save error: {e}")
        
        return {
            'analysis': analysis,
            'model': 'deepseek-r1',
            'symbol': data.get('symbol', 'BTC')
        }

@api.route('/api/ai/news/<string:symbol>')
class CryptoNews(Resource):
    def get(self, symbol):
        """Get real-time crypto news via Perplexity"""
        from src.рекс.ml.perplexity import PerplexityClient
        
        perplexity = PerplexityClient()
        news = perplexity.search_crypto_news(symbol.upper())
        return news

@api.route('/api/ai/signals/<string:symbol>')
class TradingSignals(Resource):
    def get(self, symbol):
        """Get AI trading signals via Perplexity"""
        from src.рекс.ml.perplexity import PerplexityClient
        
        timeframe = '4h'  # Default timeframe
        perplexity = PerplexityClient()
        signals = perplexity.get_trading_signals(symbol.upper(), timeframe)
        return signals

@api.route('/api/wallet/balance')
class WalletBalance(Resource):
    def get(self):
        """Get MetaMask wallet balance"""
        from src.рекс.blockchain import MetaMaskClient
        
        client = MetaMaskClient()
        balance = client.get_balance()
        return balance

@api.route('/api/wallet/monitor')
class WalletMonitor(Resource):
    def get(self):
        """Monitor wallet and DeFi opportunities"""
        from src.рекс.blockchain import MetaMaskClient
        
        client = MetaMaskClient()
        status = client.monitor_wallet()
        return status

@api.route('/api/defi/opportunities')
class DeFiOpportunities(Resource):
    def get(self):
        """Get DeFi opportunities for the wallet"""
        from src.рекс.blockchain import EthereumClient
        
        wallet = "0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1"
        client = EthereumClient()
        opportunities = client.analyze_defi_opportunities(wallet)
        return {"opportunities": opportunities, "wallet": wallet}

@api.route('/api/wallet/gas')
class GasPrice(Resource):
    def get(self):
        """Get current gas prices and optimization"""
        from src.рекс.blockchain import MetaMaskClient, EthereumClient
        
        metamask = MetaMaskClient()
        eth_client = EthereumClient()
        
        gas_price = metamask.get_gas_price()
        optimization = eth_client.get_gas_optimization()
        
        return {
            "current": gas_price,
            "optimization": optimization
        }

@api.route('/api/market/overview')
class MarketOverview(Resource):
    def get(self):
        """Get market overview for multiple symbols"""
        from src.рекс.market_data import MarketDataAggregator
        
        symbols = request.args.get('symbols', 'bitcoin,ethereum,binancecoin').split(',')
        
        aggregator = MarketDataAggregator()
        overview = aggregator.get_market_overview(symbols)
        return overview

@api.route('/api/market/dex/opportunities')
class DexOpportunities(Resource):
    def get(self):
        """Get DEX trading opportunities"""
        from src.рекс.market_data import MarketDataAggregator
        
        min_liquidity = float(request.args.get('min_liquidity', 10000))
        
        aggregator = MarketDataAggregator()
        opportunities = aggregator.get_dex_opportunities(min_liquidity)
        return {"opportunities": opportunities, "count": len(opportunities)}

@api.route('/api/market/dex/trending')
class DexTrending(Resource):
    def get(self):
        """Get trending DEX pools"""
        from src.рекс.market_data import GeckoTerminalClient
        
        network = request.args.get('network', None)
        
        client = GeckoTerminalClient()
        trending = client.get_trending_pools(network)
        return trending

@api.route('/api/market/dex/pool/<string:network>/<string:address>')
class DexPool(Resource):
    def get(self, network, address):
        """Get specific DEX pool data"""
        from src.рекс.market_data import GeckoTerminalClient
        
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
        """Get historical market data"""
        from src.рекс.market_data import MarketDataAggregator
        
        days = int(request.args.get('days', 7))
        
        aggregator = MarketDataAggregator()
        historical = aggregator.get_historical_data(symbol, days)
        return historical

@api.route('/api/data/download/<string:symbol>')
class DownloadHistoricalData(Resource):
    def get(self, symbol):
        """Download historical data for model training"""
        from src.рекс.historical_data import HistoricalDataAggregator
        
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
        from src.рекс.historical_data import HistoricalDataAggregator
        
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
        from src.рекс.historical_data import HistoricalDataAggregator
        
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
        from src.рекс.utils import rate_limiter
        
        return {
            "limits": rate_limiter.get_all_limits(),
            "timestamp": datetime.now().isoformat()
        }

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