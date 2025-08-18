"""
Trading API v1 - Trading operations
"""
from flask import request
from flask_restx import Namespace, Resource, fields
from datetime import datetime

trading_ns = Namespace('trading', description='Trading operations')

# Models for documentation
intelligent_decision_model = trading_ns.model('IntelligentDecision', {
    'symbol': fields.String(description='Symbol analyzed'),
    'recommendation': fields.String(description='Trading recommendation'),
    'confidence': fields.Float(description='Decision confidence'),
    'reasoning': fields.String(description='Decision reasoning'),
    'risk_assessment': fields.Raw(description='Risk assessment details')
})

knowledge_model = trading_ns.model('AccumulatedKnowledge', {
    'symbol': fields.String(description='Symbol'),
    'total_interactions': fields.Integer(description='Total interactions'),
    'success_patterns': fields.Integer(description='Number of success patterns'),
    'performance': fields.Raw(description='Performance metrics')
})


@trading_ns.route('/intelligent/<string:symbol>')
class IntelligentTradingDecision(Resource):
    @trading_ns.doc('get_intelligent_decision')
    @trading_ns.marshal_with(intelligent_decision_model)
    @trading_ns.expect(trading_ns.model('TradingContext', {
        'market_data': fields.Raw(description='Current market data'),
        'portfolio': fields.Raw(description='Current portfolio')
    }), validate=False)
    def post(self, symbol):
        """Get intelligent trading decision using accumulated knowledge and AI"""
        start_time = time.time()
        
        try:
            data = request.get_json() or {}
            
            # Get market data and portfolio from request
            market_data = data.get('market_data', {})
            portfolio = data.get('portfolio', {'USD': 10000})
            
            # If no market data provided, get current data
            if not market_data:
                from ...data.providers.real_only_provider import RealDataProvider
                provider = RealDataProvider()
                current_data = provider.get_current_price(symbol)
                market_data = {
                    'price': current_data.get('price', 0),
                    'volume': current_data.get('volume', 0),
                    'change_24h': current_data.get('change_24h', 0)
                }
            
            # Run intelligent analysis
            import asyncio
            
            async def get_intelligent_decision():
                from ...core.intelligence.intelligence_hub import get_intelligence_hub, IntelligenceContext
                
                hub = await get_intelligence_hub()
                
                context = IntelligenceContext(
                    session_id=f"api_{int(time.time())}",
                    symbol=symbol.upper(),
                    market_data=market_data,
                    portfolio=portfolio,
                    timestamp=datetime.utcnow()
                )
                
                return await hub.analyze_and_decide(context)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                combined_intelligence = loop.run_until_complete(get_intelligent_decision())
                
                duration_ms = (time.time() - start_time) * 1000
                
                return {
                    'symbol': symbol.upper(),
                    'recommendation': combined_intelligence.final_recommendation,
                    'confidence': combined_intelligence.confidence,
                    'reasoning': combined_intelligence.reasoning,
                    'ai_insights_count': len(combined_intelligence.ai_insights),
                    'mcts_decision': combined_intelligence.mcts_decision,
                    'risk_assessment': combined_intelligence.risk_assessment,
                    'duration_ms': duration_ms,
                    'intelligence_type': 'accumulated_knowledge',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            finally:
                loop.close()
                
        except Exception as e:
            trading_ns.abort(500, f"Intelligent trading analysis failed: {str(e)}")


@trading_ns.route('/knowledge/<string:symbol>')
class AccumulatedKnowledge(Resource):
    @trading_ns.doc('get_accumulated_knowledge')
    @trading_ns.marshal_with(knowledge_model)
    def get(self, symbol):
        """Get accumulated knowledge and performance for a symbol"""
        try:
            import asyncio
            
            async def get_knowledge():
                from ...core.intelligence.knowledge_accumulator import get_knowledge_accumulator
                from ...core.intelligence.decision_audit import get_audit_trail
                
                accumulator = await get_knowledge_accumulator()
                audit_trail = get_audit_trail()
                
                # Get accumulated knowledge
                session_id = f"knowledge_query_{int(time.time())}"
                knowledge = await accumulator.get_accumulated_knowledge(session_id)
                
                # Get symbol-specific performance
                performance = await audit_trail.get_performance_metrics(symbol, days=30)
                
                # Get recent lessons learned
                lessons = await audit_trail.get_lessons_learned(symbol, days=7)
                
                return {
                    'symbol': symbol.upper(),
                    'total_interactions': knowledge.total_interactions,
                    'success_patterns': len(knowledge.success_patterns),
                    'failure_patterns': len(knowledge.failure_patterns),
                    'market_insights': knowledge.market_insights.get(symbol.upper(), {}),
                    'performance': {
                        'total_decisions': performance.total_decisions,
                        'success_rate': performance.success_rate,
                        'avg_profit_per_decision': performance.avg_profit_per_decision,
                        'total_profit_loss': performance.total_profit_loss
                    },
                    'recent_lessons': lessons[:5],  # Last 5 lessons
                    'agent_performance': dict(knowledge.agent_performance),
                    'confidence_calibration': knowledge.confidence_calibration
                }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(get_knowledge())
                return result
            finally:
                loop.close()
                
        except Exception as e:
            trading_ns.abort(500, f"Knowledge retrieval failed: {str(e)}")
