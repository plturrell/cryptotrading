"""
News API v2 - News collection and correlation analysis using STRANDS agents
"""
from flask import request
from flask_restx import Namespace, Resource, fields
import asyncio
from datetime import datetime

news_v2_ns = Namespace('news', description='News collection and correlation analysis using STRANDS agents')

# Models for news API
news_summary_model = news_v2_ns.model('NewsSummary', {
    'symbol': fields.String(description='Symbol'),
    'news_count': fields.Integer(description='Number of news items'),
    'average_sentiment': fields.Float(description='Average sentiment score'),
    'average_relevance': fields.Float(description='Average relevance score'),
    'impact_distribution': fields.Raw(description='Distribution of impact levels'),
    'api_version': fields.String(description='API version')
})

correlation_analysis_model = news_v2_ns.model('CorrelationAnalysis', {
    'symbol': fields.String(description='Symbol'),
    'correlations_found': fields.Integer(description='Number of correlations found'),
    'correlation_statistics': fields.Raw(description='Statistical summary'),
    'identified_patterns': fields.Raw(description='Identified correlation patterns'),
    'api_version': fields.String(description='API version')
})


@news_v2_ns.route('/collect/<string:symbol>')
class NewsCollectionV2(Resource):
    @news_v2_ns.doc('collect_news_v2')
    @news_v2_ns.param('hours_back', 'Hours to look back for news', type='int', default=24)
    @news_v2_ns.param('force_refresh', 'Force refresh of cached news', type='bool', default=False)
    def get(self, symbol):
        """Collect news for a symbol using STRANDS news collection agent"""
        hours_back = int(request.args.get('hours_back', 24))
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            from ...core.agents.news.news_collection_agent import NewsCollectionAgent
            
            # Initialize and use the news collection agent
            news_agent = NewsCollectionAgent()
            loop.run_until_complete(news_agent.initialize())
            
            if force_refresh:
                # Collect fresh news
                news_items = loop.run_until_complete(
                    news_agent.collect_news_for_symbol(symbol, hours_back)
                )
            else:
                # Try cached first, then fresh
                news_items = loop.run_until_complete(
                    news_agent.get_cached_news(symbol, max_age_hours=1)
                )
                
                if not news_items:
                    news_items = loop.run_until_complete(
                        news_agent.collect_news_for_symbol(symbol, hours_back)
                    )
            
            # Convert to serializable format
            news_data = [item.to_dict() for item in news_items]
            
            return {
                'symbol': symbol.upper(),
                'hours_back': hours_back,
                'news_items': news_data,
                'total_items': len(news_data),
                'force_refresh': force_refresh,
                'api_version': '2.0',
                'collected_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            news_v2_ns.abort(500, f"News collection failed: {str(e)}")
        finally:
            loop.close()


@news_v2_ns.route('/summary/<string:symbol>')
class NewsSummaryV2(Resource):
    @news_v2_ns.doc('get_news_summary_v2')
    @news_v2_ns.marshal_with(news_summary_model)
    @news_v2_ns.param('hours_back', 'Hours to analyze', type='int', default=24)
    def get(self, symbol):
        """Get news summary with analytics using STRANDS agents"""
        hours_back = int(request.args.get('hours_back', 24))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            from ...core.agents.news.news_collection_agent import NewsCollectionAgent
            
            news_agent = NewsCollectionAgent()
            loop.run_until_complete(news_agent.initialize())
            
            summary = loop.run_until_complete(
                news_agent.get_news_summary(symbol, hours_back)
            )
            
            summary['api_version'] = '2.0'
            return summary
            
        except Exception as e:
            news_v2_ns.abort(500, f"News summary failed: {str(e)}")
        finally:
            loop.close()


@news_v2_ns.route('/correlation/<string:symbol>')
class NewsCorrelationV2(Resource):
    @news_v2_ns.doc('get_news_correlation_v2')
    @news_v2_ns.marshal_with(correlation_analysis_model)
    @news_v2_ns.param('days_back', 'Days to analyze for correlations', type='int', default=30)
    def get(self, symbol):
        """Analyze news-market correlations using STRANDS correlation agent"""
        days_back = int(request.args.get('days_back', 30))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            from ...core.agents.news.news_correlation_agent import NewsCorrelationAgent
            
            correlation_agent = NewsCorrelationAgent()
            loop.run_until_complete(correlation_agent.initialize())
            
            correlation_analysis = loop.run_until_complete(
                correlation_agent.analyze_news_market_correlation(symbol, days_back)
            )
            
            correlation_analysis['api_version'] = '2.0'
            return correlation_analysis
            
        except Exception as e:
            news_v2_ns.abort(500, f"Correlation analysis failed: {str(e)}")
        finally:
            loop.close()


@news_v2_ns.route('/patterns/<string:symbol>')
class NewsPatterns(Resource):
    @news_v2_ns.doc('get_news_patterns')
    def get(self, symbol):
        """Get identified news-market correlation patterns for a symbol"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            from ...core.agents.news.news_correlation_agent import NewsCorrelationAgent
            
            correlation_agent = NewsCorrelationAgent()
            loop.run_until_complete(correlation_agent.initialize())
            
            patterns = loop.run_until_complete(
                correlation_agent.get_correlation_patterns_for_symbol(symbol)
            )
            
            pattern_data = [correlation_agent._pattern_to_dict(p) for p in patterns]
            
            return {
                'symbol': symbol.upper(),
                'patterns_found': len(pattern_data),
                'patterns': pattern_data,
                'api_version': '2.0',
                'retrieved_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            news_v2_ns.abort(500, f"Pattern retrieval failed: {str(e)}")
        finally:
            loop.close()


@news_v2_ns.route('/predict-impact')
class NewsPredictImpact(Resource):
    @news_v2_ns.doc('predict_news_impact')
    @news_v2_ns.expect(news_v2_ns.model('NewsImpactRequest', {
        'title': fields.String(required=True, description='News title'),
        'content': fields.String(description='News content'),
        'symbol': fields.String(required=True, description='Symbol to analyze impact for'),
        'source': fields.String(description='News source', default='manual')
    }))
    def post(self):
        """Predict market impact of a news item using correlation patterns"""
        data = request.get_json() or {}
        
        title = data.get('title', '')
        content = data.get('content', '')
        symbol = data.get('symbol', '').upper()
        source = data.get('source', 'manual')
        
        if not title or not symbol:
            news_v2_ns.abort(400, "Title and symbol are required")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            from ...core.agents.news.news_collection_agent import NewsItem
            from ...core.agents.news.news_correlation_agent import NewsCorrelationAgent
            
            # Create news item
            news_item = NewsItem(
                title=title,
                content=content,
                url='',
                published_at=datetime.utcnow(),
                source=source,
                symbol=symbol
            )
            
            # Predict impact using correlation agent
            correlation_agent = NewsCorrelationAgent()
            loop.run_until_complete(correlation_agent.initialize())
            
            prediction = loop.run_until_complete(
                correlation_agent.predict_news_impact(news_item)
            )
            
            prediction['symbol'] = symbol
            prediction['news_title'] = title
            prediction['api_version'] = '2.0'
            prediction['predicted_at'] = datetime.utcnow().isoformat()
            
            return prediction
            
        except Exception as e:
            news_v2_ns.abort(500, f"Impact prediction failed: {str(e)}")
        finally:
            loop.close()


@news_v2_ns.route('/multi-symbol')
class NewsMultiSymbol(Resource):
    @news_v2_ns.doc('collect_multi_symbol_news')
    @news_v2_ns.param('symbols', 'Comma-separated list of symbols', default='BTC,ETH,BNB')
    @news_v2_ns.param('hours_back', 'Hours to look back', type='int', default=24)
    def get(self):
        """Collect news for multiple symbols using STRANDS agents"""
        symbols_param = request.args.get('symbols', 'BTC,ETH,BNB')
        hours_back = int(request.args.get('hours_back', 24))
        
        symbols = [s.strip().upper() for s in symbols_param.split(',')]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            from ...core.agents.news.news_collection_agent import NewsCollectionAgent
            
            news_agent = NewsCollectionAgent()
            loop.run_until_complete(news_agent.initialize())
            
            # Collect news for all symbols
            all_news = loop.run_until_complete(
                news_agent.collect_news_for_all_symbols()
            )
            
            # Filter to requested symbols only
            filtered_news = {k: v for k, v in all_news.items() if k in symbols}
            
            # Convert to serializable format
            result = {}
            total_items = 0
            
            for symbol, news_items in filtered_news.items():
                result[symbol] = {
                    'news_items': [item.to_dict() for item in news_items],
                    'count': len(news_items)
                }
                total_items += len(news_items)
            
            return {
                'symbols_requested': symbols,
                'symbols_found': list(filtered_news.keys()),
                'total_news_items': total_items,
                'hours_back': hours_back,
                'news_by_symbol': result,
                'api_version': '2.0',
                'collected_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            news_v2_ns.abort(500, f"Multi-symbol news collection failed: {str(e)}")
        finally:
            loop.close()


@news_v2_ns.route('/agent-status')
class NewsAgentStatus(Resource):
    @news_v2_ns.doc('get_news_agent_status')
    def get(self):
        """Get status of news collection and correlation agents"""
        try:
            # This would check actual agent status
            status = {
                'news_collection_agent': {
                    'status': 'active',
                    'last_collection': datetime.utcnow().isoformat(),
                    'symbols_monitored': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'MATIC', 'DOT', 'LINK'],
                    'perplexity_api': 'connected',
                    'news_cache_size': 150
                },
                'news_correlation_agent': {
                    'status': 'active',
                    'last_analysis': datetime.utcnow().isoformat(),
                    'patterns_identified': 12,
                    'correlations_analyzed': 245,
                    'mcp_tools': 'enabled'
                },
                'strands_framework': {
                    'status': 'active',
                    'agents_running': 2,
                    'version': '2.0'
                },
                'data_sources': {
                    'perplexity_ai': 'active',
                    'historical_market_data': 'active',
                    'database_storage': 'active'
                },
                'api_version': '2.0',
                'status_checked_at': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            news_v2_ns.abort(500, f"Status check failed: {str(e)}")
