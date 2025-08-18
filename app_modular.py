"""
Modular Flask Application - Refactored from monolithic app.py
Uses service layer architecture and API versioning
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

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from dotenv import load_dotenv

# Import unified bootstrap
from cryptotrading.core.bootstrap_unified import setup_flask_app
from cryptotrading.core.config import is_vercel

# Import API blueprints
from cryptotrading.api.v1 import api_v1_bp
from cryptotrading.api.v2 import api_v2_bp
from cryptotrading.api.websocket import websocket_bp, init_websocket

# Import observability components
try:
    from cryptotrading.infrastructure.monitoring.dashboard import register_observability_routes, create_dashboard_route
    from cryptotrading.infrastructure.monitoring.tracer import instrument_flask_app
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Environment detection
IS_VERCEL = is_vercel()

# Load environment variables
load_dotenv()

# Initialize database with unified database
try:
    from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase
    import asyncio
    
    db = UnifiedDatabase()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(db.initialize())
    print("Database initialized successfully")
except Exception as e:
    print(f"Database initialization warning: {e}")
    db = None

# Create Flask app
app = Flask(__name__, 
           static_folder='webapp',
           template_folder='webapp')

# Initialize SocketIO for WebSocket support
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='threading',
                   logger=True,
                   engineio_logger=True)

# Setup unified monitoring and storage
bootstrap = setup_flask_app(app, "cryptotrading")
monitor = bootstrap.get_monitor()
storage = bootstrap.get_storage()
feature_flags = bootstrap.get_feature_flags()

monitor.log_info("Modular Flask app starting", {
    "is_vercel": IS_VERCEL,
    "features": feature_flags.get_feature_flags(),
    "websocket_enabled": True
})

# Initialize observability
if OBSERVABILITY_AVAILABLE and feature_flags.use_full_monitoring:
    try:
        instrument_flask_app(app)
        register_observability_routes(app)
        create_dashboard_route(app)
        monitor.log_info("Full observability enabled")
    except Exception as e:
        monitor.log_error("Observability setup failed", {"error": str(e)})
        OBSERVABILITY_AVAILABLE = False

# CORS configuration
CORS(app, origins=[
    'https://cryptotrading.com',
    'https://www.cryptotrading.com',
    'https://app.cryptotrading.com',
    'https://api.cryptotrading.com'
])

# Register API blueprints with versioning
app.register_blueprint(api_v1_bp)
app.register_blueprint(api_v2_bp)
app.register_blueprint(websocket_bp)

# Initialize WebSocket manager
websocket_manager = init_websocket(socketio)

# Register shared routes
from cryptotrading.core.web.shared_routes import register_shared_routes
register_shared_routes(app)

# Legacy API compatibility endpoints (redirects to v1)
@app.route('/api/market/data')
def legacy_market_data():
    """Legacy endpoint - redirect to v1"""
    from flask import redirect, request
    return redirect(f"/api/v1/market/data?{request.query_string.decode()}")

@app.route('/api/ai/analyze', methods=['POST'])
def legacy_ai_analyze():
    """Legacy endpoint - redirect to v1"""
    from flask import redirect
    return redirect("/api/v1/ai/analyze")

@app.route('/api/ml/predict/<string:symbol>')
def legacy_ml_predict(symbol):
    """Legacy endpoint - redirect to v1"""
    from flask import redirect, request
    return redirect(f"/api/v1/ml/predict/{symbol}?{request.query_string.decode()}")

# Enhanced health check with service status
@app.route('/health')
def health_check():
    """Enhanced health check with service dependencies"""
    from datetime import datetime
    import time
    
    start_time = time.time()
    
    # Check service health
    services_status = {}
    
    # Check database
    try:
        if db:
            cursor = db.db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            services_status["database"] = "healthy"
        else:
            services_status["database"] = "unavailable"
    except Exception as e:
        services_status["database"] = f"error: {str(e)}"
    
    # Check WebSocket
    try:
        if websocket_manager:
            services_status["websocket"] = "healthy"
        else:
            services_status["websocket"] = "unavailable"
    except Exception as e:
        services_status["websocket"] = f"error: {str(e)}"
    
    # Check data pipelines
    try:
        from cryptotrading.pipelines.data_orchestrator import DataOrchestrator
        orchestrator = DataOrchestrator()
        pipeline_status = orchestrator.get_pipeline_status()
        services_status["pipelines"] = {
            "status": "healthy",
            "running_count": len(pipeline_status.get("running_pipelines", []))
        }
    except Exception as e:
        services_status["pipelines"] = f"error: {str(e)}"
    
    # Overall health
    all_healthy = all(
        status == "healthy" or (isinstance(status, dict) and status.get("status") == "healthy")
        for status in services_status.values()
    )
    
    response_time = (time.time() - start_time) * 1000
    
    health_data = {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0-modular",
        "environment": "vercel" if IS_VERCEL else "local",
        "response_time_ms": round(response_time, 2),
        "services": services_status,
        "features": {
            "api_versioning": True,
            "websocket_support": True,
            "observability": OBSERVABILITY_AVAILABLE,
            "data_pipelines": True
        }
    }
    
    status_code = 200 if all_healthy else 503
    return health_data, status_code

# API status endpoint
@app.route('/api/status')
def api_status():
    """API status and versioning information"""
    return {
        "api_versions": {
            "v1": {
                "status": "stable",
                "endpoints": [
                    "/api/v1/market/*",
                    "/api/v1/ai/*",
                    "/api/v1/ml/*",
                    "/api/v1/trading/*"
                ],
                "description": "Legacy API - maintained for backward compatibility"
            },
            "v2": {
                "status": "beta",
                "endpoints": [
                    "/api/v2/market/*",
                    "/api/v2/ai/*",
                    "/api/v2/ml/*",
                    "/api/v2/trading/*"
                ],
                "description": "Enhanced API with improved performance and features"
            }
        },
        "websocket": {
            "endpoint": "/socket.io/",
            "events": [
                "subscribe_market",
                "subscribe_predictions",
                "get_realtime_analysis"
            ]
        },
        "documentation": {
            "v1": "/api/v1/docs",
            "v2": "/api/v2/docs",
            "swagger_ui": "/api/"
        }
    }

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    monitor.log_info("WebSocket client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    monitor.log_info("WebSocket client disconnected")

# Pipeline management endpoints
@app.route('/api/pipelines/status')
def pipelines_status():
    """Get data pipelines status"""
    try:
        from cryptotrading.pipelines.data_orchestrator import DataOrchestrator
        orchestrator = DataOrchestrator()
        status = orchestrator.get_pipeline_status()
        return status
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/pipelines/start', methods=['POST'])
def start_pipelines():
    """Start data pipelines"""
    try:
        from cryptotrading.pipelines.data_orchestrator import DataOrchestrator
        import asyncio
        
        orchestrator = DataOrchestrator()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        started_count = loop.run_until_complete(orchestrator.start_all_pipelines())
        
        return {
            "status": "success",
            "started_pipelines": started_count,
            "message": f"Started {started_count} pipelines"
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/pipelines/stop', methods=['POST'])
def stop_pipelines():
    """Stop data pipelines"""
    try:
        from cryptotrading.pipelines.data_orchestrator import DataOrchestrator
        import asyncio
        
        orchestrator = DataOrchestrator()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stopped_count = loop.run_until_complete(orchestrator.stop_all_pipelines())
        
        return {
            "status": "success",
            "stopped_pipelines": stopped_count,
            "message": f"Stopped {stopped_count} pipelines"
        }
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    # Development server with WebSocket support
    socketio.run(app, debug=False, host='0.0.0.0', port=5001)

# WSGI application for production
application = app
