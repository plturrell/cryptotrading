"""
Market API v1 - Market data endpoints
"""
from flask import request
from flask_restx import Namespace, Resource, fields
from ...services.market_service import MarketDataService

market_ns = Namespace('market', description='Market data operations')

# Models for documentation
market_data_model = market_ns.model('MarketData', {
    'symbol': fields.String(required=True, description='Symbol'),
    'price': fields.Float(description='Current price'),
    'volume': fields.Float(description='24h volume'),
    'change_24h': fields.Float(description='24h price change %')
})

historical_data_model = market_ns.model('HistoricalData', {
    'symbol': fields.String(required=True, description='Symbol'),
    'days': fields.Integer(description='Number of days'),
    'data': fields.List(fields.Raw, description='Historical price data'),
    'count': fields.Integer(description='Number of records')
})

# Initialize service
market_service = MarketDataService()


@market_ns.route('/data')
class MarketData(Resource):
    @market_ns.doc('get_market_data')
    @market_ns.marshal_with(market_data_model)
    @market_ns.param('symbol', 'Cryptocurrency symbol (e.g., BTC)')
    def get(self):
        """Get real-time market data for a symbol"""
        symbol = request.args.get('symbol', 'BTC')
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            data = loop.run_until_complete(market_service.get_realtime_price(symbol))
            return data
        except Exception as e:
            market_ns.abort(500, f"Failed to fetch market data: {str(e)}")
        finally:
            loop.close()


@market_ns.route('/overview')
class MarketOverview(Resource):
    @market_ns.doc('get_market_overview')
    @market_ns.param('symbols', 'Comma-separated list of symbols (e.g., BTC,ETH,BNB)')
    def get(self):
        """Get market overview for multiple symbols"""
        symbols_param = request.args.get('symbols', 'BTC,ETH,BNB')
        symbols = [s.strip() for s in symbols_param.split(',')]
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            data = loop.run_until_complete(market_service.get_market_overview(symbols))
            return data
        except Exception as e:
            market_ns.abort(500, f"Failed to fetch market overview: {str(e)}")
        finally:
            loop.close()


@market_ns.route('/historical/<string:symbol>')
class HistoricalData(Resource):
    @market_ns.doc('get_historical_data')
    @market_ns.marshal_with(historical_data_model)
    @market_ns.param('days', 'Number of days of historical data (default: 7)')
    def get(self, symbol):
        """Get historical market data for a symbol"""
        days = int(request.args.get('days', 7))
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            data = loop.run_until_complete(market_service.get_historical_data(symbol, days))
            return data
        except Exception as e:
            market_ns.abort(500, f"Failed to fetch historical data: {str(e)}")
        finally:
            loop.close()
