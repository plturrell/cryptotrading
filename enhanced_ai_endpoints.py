"""
Enhanced AI API Endpoints - Demonstrating Grok4 Intelligence
New endpoints that showcase the advanced AI capabilities
"""
from flask import Flask, request, jsonify
from flask_restx import Api, Resource
import asyncio
import logging
from datetime import datetime

# Import our enhanced AI service
from src.cryptotrading.services.ai_service import AIAnalysisService

app = Flask(__name__)
api = Api(app, version='1.0', title='Enhanced AI Trading API',
          description='Advanced AI-powered trading analysis using Grok4 intelligence')

# Initialize AI service
ai_service = AIAnalysisService()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@api.route('/api/ai/enhanced/market-predictions')
class MarketPredictions(Resource):
    def post(self):
        """
        Predict market movements using Grok4's prediction engine
        
        Example payload:
        {
            "symbols": ["BTC", "ETH", "ADA"],
            "horizon": "1d"
        }
        """
        try:
            data = request.get_json()
            symbols = data.get('symbols', ['BTC'])
            horizon = data.get('horizon', '1d')
            
            if not symbols:
                return {"error": "Symbols list is required"}, 400
            
            # Use async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    ai_service.predict_market_movements(symbols, horizon)
                )
                
                return {
                    "status": "success",
                    "result": result,
                    "capabilities": "grok4_market_prediction",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Market prediction failed: {e}")
            return {"error": str(e), "status": "failed"}, 500

@api.route('/api/ai/enhanced/correlation-analysis')
class CorrelationAnalysis(Resource):
    def post(self):
        """
        Analyze correlation patterns using Grok4's correlation analysis
        
        Example payload:
        {
            "symbols": ["BTC", "ETH", "ADA", "SOL"],
            "timeframe": "1d"
        }
        """
        try:
            data = request.get_json()
            symbols = data.get('symbols', ['BTC', 'ETH'])
            timeframe = data.get('timeframe', '1d')
            
            if len(symbols) < 2:
                return {"error": "At least 2 symbols required for correlation analysis"}, 400
            
            # Use async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    ai_service.analyze_correlations(symbols, timeframe)
                )
                
                return {
                    "status": "success",
                    "result": result,
                    "capabilities": "grok4_correlation_analysis",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {"error": str(e), "status": "failed"}, 500

@api.route('/api/ai/enhanced/comprehensive-analysis')
class ComprehensiveAnalysis(Resource):
    def post(self):
        """
        Comprehensive AI analysis combining multiple Grok4 capabilities
        
        Example payload:
        {
            "symbols": ["BTC", "ETH"],
            "horizon": "1d",
            "include_predictions": true,
            "include_correlations": true
        }
        """
        try:
            data = request.get_json()
            symbols = data.get('symbols', ['BTC'])
            horizon = data.get('horizon', '1d')
            
            include_predictions = data.get('include_predictions', False)
            include_correlations = data.get('include_correlations', False)
            
            results = {}
            
            # Use async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Market analysis for each symbol
                market_analyses = {}
                for symbol in symbols:
                    analysis_data = {'symbol': symbol, 'timeframe': horizon}
                    market_analyses[symbol] = loop.run_until_complete(
                        ai_service.analyze_market(analysis_data)
                    )
                
                results['market_analysis'] = market_analyses
                
                # Market predictions
                if include_predictions:
                    results['predictions'] = loop.run_until_complete(
                        ai_service.predict_market_movements(symbols, horizon)
                    )
                
                # Correlation analysis
                if include_correlations and len(symbols) > 1:
                    results['correlations'] = loop.run_until_complete(
                        ai_service.analyze_correlations(symbols)
                    )
                
                return {
                    "status": "success",
                    "results": results,
                    "capabilities": "grok4_comprehensive_analysis",
                    "symbols_analyzed": symbols,
                    "analysis_components": {
                        "market_analysis": True,
                        "predictions": include_predictions,
                        "correlations": include_correlations and len(symbols) > 1
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"error": str(e), "status": "failed"}, 500

@api.route('/api/ai/enhanced/capabilities')
class AICapabilities(Resource):
    def get(self):
        """
        Get information about enhanced AI capabilities
        """
        return {
            "ai_engine": "Grok4",
            "status": "enhanced_intelligence_active",
            "capabilities": {
                "market_analysis": {
                    "description": "Advanced market sentiment analysis with predictions",
                    "endpoint": "/api/ai/analyze",
                    "features": ["sentiment_scoring", "market_insights", "prediction_integration"]
                },
                "market_predictions": {
                    "description": "AI-powered market movement predictions",
                    "endpoint": "/api/ai/enhanced/market-predictions",
                    "features": ["direction_prediction", "confidence_scoring", "factor_analysis"]
                },
                "correlation_analysis": {
                    "description": "Advanced correlation pattern analysis",
                    "endpoint": "/api/ai/enhanced/correlation-analysis",
                    "features": ["correlation_matrix", "diversification_scoring", "clustering"]
                },
                "news_sentiment": {
                    "description": "Enhanced news sentiment analysis",
                    "endpoint": "/api/ai/sentiment",
                    "features": ["market_impact", "symbol_extraction", "confidence_scoring"]
                },
                "comprehensive_analysis": {
                    "description": "Multi-faceted AI analysis combining market insights",
                    "endpoint": "/api/ai/enhanced/comprehensive-analysis",
                    "features": ["integrated_insights", "holistic_view", "actionable_recommendations"]
                }
            },
            "improvements_over_generic_gateway": [
                "Real AI intelligence instead of generic responses",
                "Advanced market prediction capabilities",
                "Correlation analysis with clustering",
                "Enhanced sentiment analysis with market impact",
                "Comprehensive multi-dimensional analysis"
            ],
            "api_configuration": {
                "api_key_required": True,
                "environment_variable": "XAI_API_KEY or GROK4_API_KEY",
                "model": "grok-2-1212",
                "real_ai": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == '__main__':
    print("🚀 Starting Enhanced AI Trading API with Grok4 Intelligence")
    print("📊 New capabilities:")
    print("   • AI-powered market predictions") 
    print("   • Correlation pattern analysis")
    print("   • Comprehensive multi-dimensional analysis")
    print("\n🔗 Enhanced endpoints:")
    print("   • POST /api/ai/enhanced/market-predictions")
    print("   • POST /api/ai/enhanced/correlation-analysis")
    print("   • POST /api/ai/enhanced/comprehensive-analysis")
    print("   • GET  /api/ai/enhanced/capabilities")
    print("\n⚠️  Configure XAI_API_KEY for real AI intelligence")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
