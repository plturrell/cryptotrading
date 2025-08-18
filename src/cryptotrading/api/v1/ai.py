"""
AI API v1 - AI analysis endpoints
"""
from flask import request
from flask_restx import Namespace, Resource, fields
from ...services.ai_service import AIAnalysisService

ai_ns = Namespace('ai', description='AI analysis operations')

# Models for documentation
ai_analysis_model = ai_ns.model('AIAnalysis', {
    'analysis': fields.String(description='AI market analysis'),
    'model': fields.String(description='AI model used'),
    'symbol': fields.String(description='Analyzed symbol'),
    'duration_ms': fields.Float(description='Analysis duration in milliseconds')
})

strategy_model = ai_ns.model('TradingStrategy', {
    'strategy': fields.Raw(description='Generated trading strategy'),
    'user_profile': fields.Raw(description='User profile used'),
    'storage_url': fields.String(description='Storage URL if saved')
})

# Initialize service
ai_service = AIAnalysisService()


@ai_ns.route('/analyze')
class AIAnalysis(Resource):
    @ai_ns.doc('ai_market_analysis')
    @ai_ns.marshal_with(ai_analysis_model)
    @ai_ns.expect(ai_ns.model('AnalysisRequest', {
        'symbol': fields.String(required=True, description='Symbol to analyze'),
        'market_data': fields.Raw(description='Market data context')
    }))
    def post(self):
        """AI market analysis using Claude-4-Sonnet"""
        data = request.get_json() or {}
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            analysis = loop.run_until_complete(ai_service.analyze_market(data))
            return analysis
        except Exception as e:
            ai_ns.abort(500, f"AI analysis failed: {str(e)}")
        finally:
            loop.close()


@ai_ns.route('/news/<string:symbol>')
class CryptoNews(Resource):
    @ai_ns.doc('get_crypto_news')
    @ai_ns.param('symbol', 'Cryptocurrency symbol')
    def get(self, symbol):
        """Get real-time crypto news via Perplexity"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            news = loop.run_until_complete(ai_service.get_crypto_news(symbol))
            return news
        except Exception as e:
            ai_ns.abort(500, f"News fetch failed: {str(e)}")
        finally:
            loop.close()


@ai_ns.route('/strategy')
class AIStrategy(Resource):
    @ai_ns.doc('generate_strategy')
    @ai_ns.marshal_with(strategy_model)
    @ai_ns.expect(ai_ns.model('UserProfile', {
        'user_id': fields.String(description='User identifier'),
        'risk_tolerance': fields.String(description='Risk tolerance level'),
        'investment_amount': fields.Float(description='Investment amount'),
        'experience_level': fields.String(description='Trading experience level')
    }))
    def post(self):
        """Generate personalized trading strategy using Claude-4"""
        user_profile = request.get_json() or {}
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            strategy = loop.run_until_complete(ai_service.generate_trading_strategy(user_profile))
            return strategy
        except Exception as e:
            ai_ns.abort(500, f"Strategy generation failed: {str(e)}")
        finally:
            loop.close()


@ai_ns.route('/sentiment')
class AISentiment(Resource):
    @ai_ns.doc('analyze_sentiment')
    @ai_ns.expect(ai_ns.model('SentimentRequest', {
        'news': fields.List(fields.Raw, required=True, description='News articles to analyze')
    }))
    def post(self):
        """Analyze news sentiment using Claude-4"""
        data = request.get_json() or {}
        news_items = data.get('news', [])
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            sentiment = loop.run_until_complete(ai_service.analyze_news_sentiment(news_items))
            return sentiment
        except Exception as e:
            ai_ns.abort(500, f"Sentiment analysis failed: {str(e)}")
        finally:
            loop.close()
