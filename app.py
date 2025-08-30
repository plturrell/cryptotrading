"""
WSGI entry point for cryptotrading.com on GoDaddy hosting
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Create data directory if it doesn't exist
os.makedirs(project_root / "data", exist_ok=True)

from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource
from dotenv import load_dotenv

# Import unified bootstrap
from cryptotrading.core.bootstrap_unified import setup_flask_app
from cryptotrading.core.config import is_vercel

# Import observability components (fallback to legacy)
try:
    from cryptotrading.infrastructure.monitoring.dashboard import (
        register_observability_routes,
        create_dashboard_route,
    )
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
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, np.datetime64):
            return pd.to_datetime(obj).strftime("%Y-%m-%d")
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif pd.isna(obj):
            return None
        return str(obj)


# Environment detection (use unified config)
IS_VERCEL = is_vercel()

# Load environment variables
load_dotenv()

# Initialize database with unified database
try:
    from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase
    import asyncio

    # Initialize database synchronously for Flask startup
    db = UnifiedDatabase()
    # Run async initialization
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(db.initialize())
    print("Database initialized successfully")
except Exception as e:
    print(f"Database initialization warning: {e}")
    db = None

app = Flask(__name__, static_folder="webapp", template_folder="webapp")

# Setup unified monitoring and storage
bootstrap = setup_flask_app(app, "cryptotrading")
monitor = bootstrap.get_monitor()
storage = bootstrap.get_storage()
feature_flags = bootstrap.get_feature_flags()

monitor.log_info(
    "Flask app starting", {"is_vercel": IS_VERCEL, "features": feature_flags.get_feature_flags()}
)

# Initialize observability (legacy fallback)
if OBSERVABILITY_AVAILABLE and feature_flags.use_full_monitoring:
    try:
        # Instrument Flask app for tracing
        instrument_flask_app(app)

        # Register observability routes
        register_observability_routes(app)
        create_dashboard_route(app)

        monitor.log_info(
            "Full observability enabled", {"dashboard": "/observability/dashboard.html"}
        )
    except Exception as e:
        monitor.log_error("Observability setup failed", {"error": str(e)})
        OBSERVABILITY_AVAILABLE = False

# CORS configuration for cryptotrading.com
CORS(
    app,
    origins=[
        "https://cryptotrading.com",
        "https://www.cryptotrading.com",
        "https://app.cryptotrading.com",
        "https://api.cryptotrading.com",
    ],
)

# API configuration
api = Api(
    app,
    version="1.0",
    title="cryptotrading.com API",
    description="Professional cryptocurrency trading platform API",
    doc="/api/",
)

# Register shared routes (SAP UI5 and health check)
from cryptotrading.core.web.shared_routes import register_shared_routes

register_shared_routes(app)

# Register dashboard endpoints for professional tile data
from api.dashboard_endpoints import dashboard_bp
app.register_blueprint(dashboard_bp)

# Register CDS services for OData v4 support
try:
    from api.cds_service_adapter import register_cds_services

    register_cds_services(app)
    monitor.log_info(
        "CDS services registered with news integration",
        {"endpoints": ["/api/odata/v4/TradingService", "/api/odata/v4/CodeAnalysisService"]},
    )
except Exception as e:
    monitor.log_error("Failed to register CDS services", {"error": str(e)})

# Register News API endpoints
try:
    from api.news_api import register_news_api

    register_news_api(app)
    monitor.log_info(
        "News API registered",
        {"endpoints": ["/api/news/latest", "/api/news/symbol", "/api/news/sentiment"]},
    )
except Exception as e:
    monitor.log_error("Failed to register News API", {"error": str(e)})

# Register Data Loading Service
try:
    from api.data_loading_service import data_loading_bp

    app.register_blueprint(data_loading_bp)
    monitor.log_info(
        "Data Loading Service registered",
        {
            "endpoints": [
                "/api/odata/v4/DataLoadingService/getDataSourceStatus",
                "/api/odata/v4/DataLoadingService/getActiveJobs",
                "/api/odata/v4/DataLoadingService/loadYahooFinanceData",
                "/api/odata/v4/DataLoadingService/loadFREDData",
                "/api/odata/v4/DataLoadingService/loadGeckoTerminalData",
                "/api/odata/v4/DataLoadingService/loadAllMarketData",
            ]
        },
    )
except Exception as e:
    monitor.log_error("Failed to register Data Loading Service", {"error": str(e)})

# Register AWS Data Exchange Service
try:
    from api.aws_data_exchange_api import aws_data_exchange_bp

    app.register_blueprint(aws_data_exchange_bp)
    monitor.log_info(
        "AWS Data Exchange Service registered",
        {
            "endpoints": [
                "/api/odata/v4/AWSDataExchange/getAvailableDatasets",
                "/api/odata/v4/AWSDataExchange/getDatasetAssets", 
                "/api/odata/v4/AWSDataExchange/loadDatasetToDatabase",
                "/api/odata/v4/AWSDataExchange/discoverCryptoData",
                "/api/odata/v4/AWSDataExchange/discoverEconomicData",
                "/api/odata/v4/AWSDataExchange/getServiceStatus",
            ]
        },
    )
except Exception as e:
    monitor.log_error("Failed to register AWS Data Exchange Service", {"error": str(e)})

# API endpoints


@api.route("/api/market/data")
class MarketData(Resource):
    def get(self):
        """Get market data from Yahoo Finance with full observability"""
        from cryptotrading.infrastructure.monitoring import (
            get_logger,
            get_business_metrics,
            trace_context,
            ErrorSeverity,
            ErrorCategory,
        )

        logger = get_logger("api.market")
        business_metrics = get_business_metrics()

        symbol = request.args.get("symbol", "BTC")
        start_time = time.time()

        with trace_context(f"market_data_api_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("api.endpoint", "/api/market/data")

                logger.info(
                    f"Fetching market data for {symbol}",
                    extra={"symbol": symbol, "endpoint": "/api/market/data", "method": "GET"},
                )

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

                logger.info(
                    f"Successfully retrieved market data for {symbol}",
                    extra={
                        "symbol": symbol,
                        "price": data.get("price"),
                        "volume": data.get("volume"),
                    },
                )

                span.set_attribute("price", str(data.get("price", 0)))
                span.set_attribute("success", "true")

                return data

            except Exception as e:
                logger.error(
                    f"Failed to fetch market data for {symbol}",
                    error=e,
                    extra={"symbol": symbol, "endpoint": "/api/market/data"},
                )

                # Track error
                from cryptotrading.infrastructure.monitoring import get_error_tracker

                error_tracker = get_error_tracker()
                error_tracker.track_error(
                    e, severity=ErrorSeverity.HIGH, category=ErrorCategory.API_ERROR
                )

                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request("/api/market/data", "GET", 500, duration_ms)

                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))

                return {"error": f"Failed to fetch market data: {str(e)}"}, 500


@api.route("/api/ai/analyze")
class AIAnalysis(Resource):
    def post(self):
        """AI market analysis using Claude-4-Sonnet with full observability"""
        from cryptotrading.infrastructure.monitoring import (
            get_logger,
            get_business_metrics,
            trace_context,
            ErrorSeverity,
            ErrorCategory,
        )

        logger = get_logger("api.ai.analyze")
        business_metrics = get_business_metrics()

        start_time = time.time()
        data = api.payload
        symbol = data.get("symbol", "BTC")

        with trace_context(f"ai_analysis_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("ai.model", "claude-4-sonnet")
                span.set_attribute("api.endpoint", "/api/ai/analyze")

                logger.info(
                    f"Starting AI analysis for {symbol}",
                    extra={
                        "symbol": symbol,
                        "model": "claude-4-sonnet",
                        "endpoint": "/api/ai/analyze",
                        "method": "POST",
                    },
                )

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
                    duration_ms=duration_ms,
                )

                logger.info(
                    f"AI analysis completed for {symbol}",
                    extra={
                        "symbol": symbol,
                        "analysis_length": len(str(analysis)) if analysis else 0,
                        "duration_ms": duration_ms,
                    },
                )

                span.set_attribute("success", "true")
                span.set_attribute("analysis_length", len(str(analysis)) if analysis else 0)

                return {
                    "analysis": analysis,
                    "model": "claude-4-sonnet",
                    "symbol": symbol,
                    "duration_ms": duration_ms,
                }

            except Exception as e:
                logger.error(
                    f"AI analysis failed for {symbol}",
                    error=e,
                    extra={
                        "symbol": symbol,
                        "model": "claude-4-sonnet",
                        "endpoint": "/api/ai/analyze",
                    },
                )

                # Track error
                from cryptotrading.infrastructure.monitoring import get_error_tracker

                error_tracker = get_error_tracker()
                error_tracker.track_error(
                    e, severity=ErrorSeverity.HIGH, category=ErrorCategory.AI_ERROR
                )

                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request("/api/ai/analyze", "POST", 500, duration_ms)
                business_metrics.track_ai_operation(
                    operation="market_analysis",
                    model="claude-4-sonnet",
                    symbol=symbol,
                    success=False,
                    duration_ms=duration_ms,
                )

                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))

                return {"error": str(e)}, 500


@api.route("/api/ai/news/<string:symbol>")
class CryptoNews(Resource):
    def get(self, symbol):
        """Get real-time crypto news via Perplexity"""
        from cryptotrading.core.ml.perplexity import PerplexityClient

        perplexity = PerplexityClient()
        news = perplexity.search_crypto_news(symbol.upper())
        return news


@api.route("/api/ml/predict/<string:symbol>")
class MLPrediction(Resource):
    def get(self, symbol):
        """Get ML price prediction for a cryptocurrency"""
        from cryptotrading.core.ml.inference import inference_service, PredictionRequest

        horizon = request.args.get("horizon", "24h")
        model_type = request.args.get("model_type", None)

        # Run async prediction in sync context
        import asyncio

        async def get_prediction():
            req = PredictionRequest(symbol=symbol.upper(), horizon=horizon, model_type=model_type)
            return await inference_service.get_prediction(req)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(get_prediction())
            return result.dict()
        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/api/ml/predict/batch")
class MLBatchPrediction(Resource):
    def post(self):
        """Get ML predictions for multiple cryptocurrencies"""
        from cryptotrading.core.ml.inference import inference_service, BatchPredictionRequest

        data = request.get_json()
        symbols = data.get("symbols", ["BTC", "ETH"])
        horizon = data.get("horizon", "24h")
        model_type = data.get("model_type", None)

        # Run async prediction in sync context
        import asyncio

        async def get_predictions():
            req = BatchPredictionRequest(
                symbols=[s.upper() for s in symbols], horizon=horizon, model_type=model_type
            )
            return await inference_service.get_batch_predictions(req)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(get_predictions())
            return [r.dict() for r in results]
        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/api/ml/performance/<string:symbol>")
class MLModelPerformance(Resource):
    def get(self, symbol):
        """Get ML model performance metrics"""
        from cryptotrading.core.ml.inference import inference_service

        horizon = request.args.get("horizon", "24h")

        # Run async in sync context
        import asyncio

        async def get_performance():
            return await inference_service.get_model_performance(symbol.upper(), horizon)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(get_performance())
            return result.dict()
        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/api/ml/train")
class MLTraining(Resource):
    def post(self):
        """Trigger ML model training"""
        from cryptotrading.core.ml.training import training_pipeline

        # Run training in background
        import threading
        import asyncio

        def train_models():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(training_pipeline.train_all_models())

        thread = threading.Thread(target=train_models)
        thread.start()

        return {
            "status": "training_started",
            "message": "Model training initiated in background",
            "symbols": training_pipeline.training_config["symbols"],
            "model_types": training_pipeline.training_config["model_types"],
        }


@api.route("/api/ml/features/<string:symbol>")
class MLFeatures(Resource):
    def get(self, symbol):
        """Get ML features for a symbol"""
        from cryptotrading.core.ml.feature_store import feature_store

        features = request.args.get("features", None)
        if features:
            features = features.split(",")

        # Run async in sync context
        import asyncio

        async def get_features():
            return await feature_store.compute_features(symbol.upper(), features)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            features_df = loop.run_until_complete(get_features())

            if features_df.empty:
                return {"error": "No features available"}, 404

            # Return last 100 rows as JSON
            result = {
                "symbol": symbol.upper(),
                "features": features_df.tail(100).to_dict(orient="records"),
                "feature_names": list(features_df.columns),
                "total_features": len(features_df.columns),
                "importance": feature_store.get_feature_importance(),
            }

            return result

        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/api/wallet/balance")
class WalletBalance(Resource):
    def get(self):
        """Wallet balance - MetaMask client removed"""
        return {
            "error": "Wallet balance functionality requires blockchain integration",
            "message": "MetaMask client was removed as it required Infura API key",
            "suggested_alternatives": [
                "Use Web3.py directly with your own Infura/Alchemy key",
                "Use Etherscan API for wallet balance queries",
            ],
        }


@api.route("/api/wallet/monitor")
class WalletMonitor(Resource):
    def get(self):
        """Wallet monitoring - MetaMask client removed"""
        return {
            "error": "Wallet monitoring functionality requires blockchain integration",
            "message": "MetaMask client was removed as it required Infura API key",
            "suggested_alternatives": [
                "Use Web3.py directly with your own Infura/Alchemy key",
                "Integrate with wallet monitoring services like Moralis",
            ],
        }


@api.route("/api/defi/opportunities")
class DeFiOpportunities(Resource):
    def get(self):
        """DeFi opportunities - blockchain clients removed"""
        return {
            "error": "DeFi analysis functionality requires blockchain integration",
            "message": "Ethereum client was removed as it contained fake DeFi data",
            "suggested_alternatives": [
                "Use real DeFi protocols like Aave or Compound APIs",
                "Integrate with DeFi aggregators like DeFiPulse or Zapper",
            ],
        }


@api.route("/api/wallet/gas")
class GasPrice(Resource):
    def get(self):
        """Gas price information - blockchain clients removed"""
        return {
            "error": "Gas price functionality requires blockchain integration",
            "message": ("MetaMask and Ethereum clients were removed as they required "
                        "external API keys"),
            "suggested_alternatives": [
                "Use CoinGecko API for ETH price information",
                "Integrate with a gas tracking service like EtherScan",
            ],
        }


@api.route("/api/market/overview")
class MarketOverview(Resource):
    def get(self):
        """Get real market overview for multiple symbols"""
        try:
            from cryptotrading.data.providers.real_only_provider import RealOnlyDataProvider

            symbols = request.args.get("symbols", "BTC,ETH,BNB").split(",")

            # Use real data provider - no mocks
            provider = RealOnlyDataProvider()

            # Run async in sync context
            import asyncio

            async def get_overview():
                results = {}
                for symbol in symbols:
                    try:
                        price_data = await provider.get_real_time_price(symbol)
                        results[symbol] = price_data
                    except Exception as e:
                        monitor.log_warning(f"Failed to get price for {symbol}: {e}")
                        continue
                return results

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            overview = loop.run_until_complete(get_overview())

            if not overview:
                return {"error": "No market data available"}, 503

            return {
                "symbols": list(overview.keys()),
                "data": overview,
                "timestamp": datetime.now().isoformat(),
                "source": "real_only_provider",
            }

        except Exception as e:
            monitor.log_error(f"Market overview failed: {e}", {"error": str(e)})
            return {"error": str(e)}, 500


@api.route("/api/intelligent/trading/<string:symbol>")
class IntelligentTradingDecision(Resource):
    def post(self, symbol):
        """Get intelligent trading decision using accumulated knowledge and AI"""
        start_time = time.time()

        try:
            data = api.payload or {}

            # Get market data and portfolio from request
            market_data = data.get("market_data", {})
            portfolio = data.get("portfolio", {"USD": 10000})

            # If no market data provided, get current data
            if not market_data:
                from cryptotrading.data.providers.real_only_provider import RealDataProvider

                provider = RealDataProvider()
                current_data = provider.get_current_price(symbol)
                market_data = {
                    "price": current_data.get("price", 0),
                    "volume": current_data.get("volume", 0),
                    "change_24h": current_data.get("change_24h", 0),
                }

            # Run intelligent analysis
            import asyncio

            async def get_intelligent_decision():
                from cryptotrading.core.intelligence.intelligence_hub import (
                    get_intelligence_hub,
                    IntelligenceContext,
                )

                hub = await get_intelligence_hub()

                context = IntelligenceContext(
                    session_id=f"api_{int(time.time())}",
                    symbol=symbol.upper(),
                    market_data=market_data,
                    portfolio=portfolio,
                    timestamp=datetime.utcnow(),
                )

                return await hub.analyze_and_decide(context)

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                combined_intelligence = loop.run_until_complete(get_intelligent_decision())

                duration_ms = (time.time() - start_time) * 1000

                return {
                    "symbol": symbol.upper(),
                    "recommendation": combined_intelligence.final_recommendation,
                    "confidence": combined_intelligence.confidence,
                    "reasoning": combined_intelligence.reasoning,
                    "ai_insights_count": len(combined_intelligence.ai_insights),
                    "mcts_decision": combined_intelligence.mcts_decision,
                    "risk_assessment": combined_intelligence.risk_assessment,
                    "duration_ms": duration_ms,
                    "intelligence_type": "accumulated_knowledge",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            except Exception as e:
                monitor.log_error(
                    f"Intelligent analysis failed for {symbol}: {e}", {"error": str(e)}
                )
                return {"error": f"Intelligence analysis failed: {str(e)}"}, 500

        except Exception as e:
            monitor.log_error(f"Intelligent trading request failed: {e}", {"error": str(e)})
            return {"error": str(e)}, 500


@api.route("/api/intelligent/knowledge/<string:symbol>")
class AccumulatedKnowledge(Resource):
    def get(self, symbol):
        """Get accumulated knowledge and performance for a symbol"""
        try:
            import asyncio

            async def get_knowledge():
                from cryptotrading.core.intelligence.knowledge_accumulator import (
                    get_knowledge_accumulator,
                )
                from cryptotrading.core.intelligence.decision_audit import get_audit_trail

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
                    "symbol": symbol.upper(),
                    "total_interactions": knowledge.total_interactions,
                    "success_patterns": len(knowledge.success_patterns),
                    "failure_patterns": len(knowledge.failure_patterns),
                    "market_insights": knowledge.market_insights.get(symbol.upper(), {}),
                    "performance": {
                        "total_decisions": performance.total_decisions,
                        "success_rate": performance.success_rate,
                        "avg_profit_per_decision": performance.avg_profit_per_decision,
                        "total_profit_loss": performance.total_profit_loss,
                    },
                    "recent_lessons": lessons[:5],  # Last 5 lessons
                    "agent_performance": dict(knowledge.agent_performance),
                    "confidence_calibration": knowledge.confidence_calibration,
                }

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(get_knowledge())
                return result

            except Exception as e:
                monitor.log_error(f"Knowledge retrieval failed: {e}", {"error": str(e)})
                return {"error": f"Knowledge retrieval failed: {str(e)}"}, 500

        except Exception as e:
            monitor.log_error(f"Knowledge request failed: {e}", {"error": str(e)})
            return {"error": str(e)}, 500


@api.route("/api/market/historical/<string:symbol>")
class HistoricalData(Resource):
    def get(self, symbol):
        """Get historical market data from Yahoo Finance with full observability"""
        from cryptotrading.infrastructure.monitoring import (
            get_logger,
            get_business_metrics,
            trace_context,
            ErrorSeverity,
            ErrorCategory,
        )
        from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
        from datetime import datetime, timedelta

        logger = get_logger("api.historical")
        business_metrics = get_business_metrics()

        days = int(request.args.get("days", 7))
        start_time = time.time()

        with trace_context(f"historical_data_api_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("days", days)
                span.set_attribute("api.endpoint", "/api/market/historical")

                logger.info(
                    f"Fetching {days} days of historical data for {symbol}",
                    extra={
                        "symbol": symbol,
                        "days": days,
                        "endpoint": "/api/market/historical",
                        "method": "GET",
                    },
                )

                yahoo_client = YahooFinanceClient()

                # Calculate date range
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

                # Download historical data
                df = yahoo_client.download_data(symbol, start_date, end_date, save=False)

                if df is None or df.empty:
                    error_msg = f"No historical data available for {symbol}"
                    logger.warning(error_msg, extra={"symbol": symbol, "days": days})

                    duration_ms = (time.time() - start_time) * 1000
                    business_metrics.track_api_request(
                        "/api/market/historical", "GET", 404, duration_ms
                    )

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
                            record[col] = pd.to_datetime(value).strftime("%Y-%m-%d")
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
                    "count": len(df),
                }

                # Track successful API call
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request(
                    "/api/market/historical", "GET", 200, duration_ms
                )
                business_metrics.track_data_processing(
                    source="yahoo_finance",
                    symbol=symbol,
                    records_processed=len(df),
                    success=True,
                    duration_ms=duration_ms,
                )

                logger.info(
                    f"Successfully retrieved {len(df)} historical records for {symbol}",
                    extra={
                        "symbol": symbol,
                        "records_count": len(df),
                        "days": days,
                        "date_range": {"start": start_date, "end": end_date},
                    },
                )

                span.set_attribute("records_count", len(df))
                span.set_attribute("success", "true")

                return historical_data

            except Exception as e:
                logger.error(
                    f"Failed to fetch historical data for {symbol}",
                    error=e,
                    extra={"symbol": symbol, "days": days, "endpoint": "/api/market/historical"},
                )

                # Track error
                from cryptotrading.infrastructure.monitoring import get_error_tracker

                error_tracker = get_error_tracker()
                error_tracker.track_error(
                    e, severity=ErrorSeverity.HIGH, category=ErrorCategory.API_ERROR
                )

                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request(
                    "/api/market/historical", "GET", 500, duration_ms
                )

                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))

                return {"error": f"Failed to fetch historical data: {str(e)}"}, 500


@api.route("/api/data/download/<string:symbol>")
class DownloadHistoricalData(Resource):
    def get(self, symbol):
        """Download historical data for model training"""
        from cryptotrading.data.historical import HistoricalDataAggregator

        source = request.args.get("source", "all")

        aggregator = HistoricalDataAggregator()

        if source == "all":
            data = aggregator.download_all_sources(symbol)
            return {
                "symbol": symbol,
                "sources": list(data.keys()),
                "total_rows": sum(len(df) for df in data.values()),
                "status": "downloaded",
            }
        else:
            return {"error": f"Source {source} not supported"}


@api.route("/api/data/training/<string:symbol>")
class TrainingDataset(Resource):
    def get(self, symbol):
        """Get or create training dataset with indicators"""
        from cryptotrading.data.historical import HistoricalDataAggregator

        lookback_days = int(request.args.get("days", 365))

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
            "date_range": {"start": str(df.index.min()), "end": str(df.index.max())},
            "indicators": [
                col
                for col in df.columns
                if any(ind in col for ind in ["sma", "ema", "rsi", "macd", "bb"])
            ],
        }


@api.route("/api/data/available")
class AvailableDatasets(Resource):
    def get(self):
        """List available training datasets"""
        from cryptotrading.data.historical import HistoricalDataAggregator

        aggregator = HistoricalDataAggregator()
        datasets = aggregator.get_available_datasets()

        return {"datasets": datasets, "count": len(datasets)}


@api.route("/api/limits")
class APILimits(Resource):
    def get(self):
        """Get current API rate limit status"""
        from cryptotrading.utils import rate_limiter

        return {"limits": rate_limiter.get_all_limits(), "timestamp": datetime.now().isoformat()}


@api.route("/api/ai/strategy")
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
                    f"strategies/user_{user_profile.get('user_id', 'anonymous')}.json", strategy
                )
                strategy["storage_url"] = blob_result.get("url")
            except Exception:
                pass

            return strategy
        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/api/ai/sentiment")
class AISentiment(Resource):
    def post(self):
        """Analyze news sentiment using Claude-4"""
        try:
            from cryptotrading.core.ai import AIGatewayClient

            news_items = api.payload.get("news", [])
            ai = AIGatewayClient()
            sentiment = ai.analyze_news_sentiment(news_items)

            return sentiment
        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/api/storage/signals/<string:symbol>")
class StoredSignals(Resource):
    def get(self, symbol):
        """Get stored trading signals from Vercel Blob"""
        try:
            from cryptotrading.data.storage import VercelBlobClient

            client = VercelBlobClient()
            signals = client.get_latest_signals(symbol)

            return {"symbol": symbol, "signals": signals, "count": len(signals)}
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


# CLRS Analysis API Endpoints
@api.route("/api/clrs/dependency-analysis")
class CLRSDependencyAnalysis(Resource):
    def post(self):
        """Analyze code dependencies using CLRS graph algorithms"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import CLRSMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            modules = data.get("modules", {})

            glean_client = GleanClient()
            clrs_tools = CLRSMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(clrs_tools.analyze_dependency_graph(modules))

            return {
                "analysis_type": "clrs_dependency_graph",
                "algorithm": "dfs_topological_sort",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"CLRS dependency analysis failed: {str(e)}"}, 500


@api.route("/api/clrs/code-similarity")
class CLRSCodeSimilarity(Resource):
    def post(self):
        """Analyze code similarity using CLRS string algorithms"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import CLRSMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            code1 = data.get("code1", "")
            code2 = data.get("code2", "")

            if not code1 or not code2:
                return {"error": "Both code1 and code2 are required"}, 400

            glean_client = GleanClient()
            clrs_tools = CLRSMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(clrs_tools.analyze_code_similarity(code1, code2))

            return {
                "analysis_type": "clrs_code_similarity",
                "algorithm": "longest_common_subsequence",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"CLRS similarity analysis failed: {str(e)}"}, 500


@api.route("/api/clrs/sort-symbols")
class CLRSSortSymbols(Resource):
    def post(self):
        """Sort code symbols using CLRS sorting algorithms"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import CLRSMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            symbols = data.get("symbols", [])
            sort_by = data.get("sort_by", "usage")
            algorithm = data.get("algorithm", "quicksort")

            glean_client = GleanClient()
            clrs_tools = CLRSMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                clrs_tools.sort_code_symbols(symbols, sort_by, algorithm)
            )

            return {
                "analysis_type": "clrs_symbol_sorting",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"CLRS symbol sorting failed: {str(e)}"}, 500


@api.route("/api/clrs/search-symbols")
class CLRSSearchSymbols(Resource):
    def post(self):
        """Search code symbols using CLRS binary search"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import CLRSMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            symbols = data.get("symbols", [])
            target = data.get("target", "")

            if not target:
                return {"error": "Target symbol name is required"}, 400

            glean_client = GleanClient()
            clrs_tools = CLRSMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(clrs_tools.search_code_symbols(symbols, target))

            return {
                "analysis_type": "clrs_symbol_search",
                "algorithm": "binary_search",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"CLRS symbol search failed: {str(e)}"}, 500


@api.route("/api/clrs/call-path")
class CLRSCallPath(Resource):
    def post(self):
        """Find shortest call path using Dijkstra's algorithm"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import CLRSMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            call_graph = data.get("call_graph", {})
            start_function = data.get("start_function", "")
            end_function = data.get("end_function", "")

            if not start_function or not end_function:
                return {"error": "Both start_function and end_function are required"}, 400

            glean_client = GleanClient()
            clrs_tools = CLRSMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                clrs_tools.find_shortest_call_path(call_graph, start_function, end_function)
            )

            return {
                "analysis_type": "clrs_call_path",
                "algorithm": "dijkstra",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"CLRS call path analysis failed: {str(e)}"}, 500


@api.route("/api/clrs/pattern-matching")
class CLRSPatternMatching(Resource):
    def post(self):
        """Find code patterns using KMP string matching"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import CLRSMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            source_code = data.get("source_code", "")
            patterns = data.get("patterns", [])

            if not source_code or not patterns:
                return {"error": "Both source_code and patterns are required"}, 400

            glean_client = GleanClient()
            clrs_tools = CLRSMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(clrs_tools.find_code_patterns(source_code, patterns))

            return {
                "analysis_type": "clrs_pattern_matching",
                "algorithm": "kmp",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"CLRS pattern matching failed: {str(e)}"}, 500


# Tree Analysis API Endpoints
@api.route("/api/tree/ast-analysis")
class TreeASTAnalysis(Resource):
    def post(self):
        """Process AST structure using tree operations"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import TreeMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            ast_data = data.get("ast_data", {})
            operation = data.get("operation", "get_depth")

            glean_client = GleanClient()
            tree_tools = TreeMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                tree_tools.process_ast_structure(ast_data, operation, **data)
            )

            return {
                "analysis_type": "tree_ast_processing",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"Tree AST analysis failed: {str(e)}"}, 500


@api.route("/api/tree/hierarchy-analysis")
class TreeHierarchyAnalysis(Resource):
    def post(self):
        """Analyze hierarchical code structure"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import TreeMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            codebase = data.get("codebase", {})

            glean_client = GleanClient()
            tree_tools = TreeMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(tree_tools.analyze_code_hierarchy(codebase))

            return {
                "analysis_type": "tree_hierarchy_analysis",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"Tree hierarchy analysis failed: {str(e)}"}, 500


@api.route("/api/tree/structure-diff")
class TreeStructureDiff(Resource):
    def post(self):
        """Compare two code structures and show differences"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import TreeMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            old_structure = data.get("old_structure", {})
            new_structure = data.get("new_structure", {})

            glean_client = GleanClient()
            tree_tools = TreeMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                tree_tools.compare_code_structures(old_structure, new_structure)
            )

            return {
                "analysis_type": "tree_structure_diff",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"Tree structure diff failed: {str(e)}"}, 500


@api.route("/api/tree/config-merge")
class TreeConfigMerge(Resource):
    def post(self):
        """Merge configuration structures"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import TreeMCPTools
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            base_config = data.get("base_config", {})
            override_config = data.get("override_config", {})

            glean_client = GleanClient()
            tree_tools = TreeMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                tree_tools.merge_configurations(base_config, override_config)
            )

            return {
                "analysis_type": "tree_config_merge",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"Tree config merge failed: {str(e)}"}, 500


# Enhanced Glean MCP Tools
@api.route("/api/clrs/comprehensive-analysis")
class CLRSComprehensiveAnalysis(Resource):
    def post(self):
        """Comprehensive analysis combining CLRS algorithms and Tree operations"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import (
                GleanAnalysisMCPTools,
            )
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            codebase_data = data.get("codebase_data", {})

            glean_client = GleanClient()
            analysis_tools = GleanAnalysisMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                analysis_tools.comprehensive_code_analysis(codebase_data)
            )

            return {
                "analysis_type": "clrs_comprehensive",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"CLRS comprehensive analysis failed: {str(e)}"}, 500


@api.route("/api/clrs/optimization-recommendations")
class CLRSOptimizationRecommendations(Resource):
    def post(self):
        """Get optimization recommendations using CLRS+Tree analysis"""
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import (
                GleanAnalysisMCPTools,
            )
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            data = api.payload
            current_structure = data.get("current_structure", {})
            optimization_goals = data.get(
                "optimization_goals", ["reduce_complexity", "improve_modularity"]
            )

            glean_client = GleanClient()
            analysis_tools = GleanAnalysisMCPTools(glean_client)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                analysis_tools.optimize_code_structure(current_structure, optimization_goals)
            )

            return {
                "analysis_type": "clrs_optimization",
                "timestamp": datetime.now().isoformat(),
                **result,
            }
        except Exception as e:
            return {"error": f"CLRS optimization analysis failed: {str(e)}"}, 500


# Error handlers are registered by register_shared_routes()

if __name__ == "__main__":
    # Development server
    app.run(debug=False, host="0.0.0.0", port=5001)

# WSGI application for GoDaddy
application = app
