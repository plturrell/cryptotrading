"""
MCP API Endpoint for Vercel Deployment
Production-ready MCP server endpoint
"""
import asyncio
import json
import logging
from typing import Dict, Any
from flask import Flask, request, jsonify
import os

# MCP imports
from cryptotrading.core.protocols.mcp.enhanced_server import create_enhanced_mcp_server
from cryptotrading.core.protocols.mcp.auth import AuthMiddleware, AuthContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global MCP server instance
mcp_server = None
auth_middleware = AuthMiddleware()

def initialize_mcp_server():
    """Initialize MCP server with production configuration"""
    global mcp_server
    
    if mcp_server is not None:
        return mcp_server
    
    # Production configuration
    config = {
        'auth_enabled': True,
        'cache_enabled': True,
        'rate_limiting_enabled': True,
        'metrics_enabled': True,
        'health_checks_enabled': True,
        'multi_tenant_enabled': True,
        'plugins_enabled': False,  # Disabled until real implementations
        'events_enabled': True,
        'strand_integration_enabled': True,
        'fiori_integration_enabled': True,
        'auth_config': {
            'api_keys': [os.getenv('MCP_API_KEY', 'default-production-key')],
            'jwt_secret': os.getenv('JWT_SECRET', 'production-jwt-secret')
        },
        'tenant_config': {
            'default_tenant': {
                'tenant_id': 'production',
                'name': 'Production Tenant',
                'trading_enabled': True,
                'portfolio_limit': 100,
                'api_rate_limit': 1000
            }
        }
    }
    
    # Create enhanced server
    mcp_server = create_enhanced_mcp_server(config)
    
    # Initialize asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(mcp_server.initialize_enhanced_features(config))
    
    logger.info("MCP server initialized for production")
    return mcp_server

def create_app():
    """Create Flask app for Vercel"""
    app = Flask(__name__)
    
    @app.route('/api/mcp', methods=['POST'])
    def mcp_endpoint():
        """Main MCP endpoint"""
        try:
            # Initialize server if needed
            server = initialize_mcp_server()
            
            # Get request data
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON'}), 400
            
            # Authenticate request
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'error': 'Authorization header required'}), 401
            
            # Extract token
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
            elif auth_header.startswith('ApiKey '):
                token = auth_header[7:]
            else:
                return jsonify({'error': 'Invalid authorization format'}), 401
            
            # Validate authentication
            auth_context = None
            if token:
                # Try API key first
                user_info = auth_middleware.validate_api_key(token)
                if user_info:
                    auth_context = AuthContext(
                        user_id=user_info.get('user_id', 'api_user'),
                        tenant_id=user_info.get('tenant_id', 'production'),
                        permissions=user_info.get('permissions', ['*']),
                        metadata={'auth_type': 'api_key'}
                    )
                else:
                    # Try JWT
                    try:
                        jwt_payload = auth_middleware.validate_jwt_token(token)
                        if jwt_payload:
                            auth_context = AuthContext(
                                user_id=jwt_payload.get('user_id', 'jwt_user'),
                                tenant_id=jwt_payload.get('tenant_id', 'production'),
                                permissions=jwt_payload.get('permissions', ['*']),
                                metadata={'auth_type': 'jwt'}
                            )
                    except Exception as e:
                        logger.error(f"JWT validation error: {e}")
            
            if not auth_context:
                return jsonify({'error': 'Invalid authentication'}), 401
            
            # Process MCP request
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    server.handle_enhanced_request(data, auth_context)
                )
                return jsonify(result)
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"MCP endpoint error: {e}")
            return jsonify({
                'jsonrpc': '2.0',
                'id': data.get('id') if data else None,
                'error': {
                    'code': -32000,
                    'message': str(e)
                }
            }), 500
    
    @app.route('/api/mcp/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            server = initialize_mcp_server()
            
            # Basic health check
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                health_status = loop.run_until_complete(
                    server.health_checker.check_all_components()
                )
                
                return jsonify({
                    'status': 'healthy' if health_status['overall_status'] == 'healthy' else 'unhealthy',
                    'details': health_status,
                    'timestamp': health_status.get('timestamp')
                })
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 500
    
    @app.route('/api/mcp/info', methods=['GET'])
    def server_info():
        """Server information endpoint"""
        try:
            server = initialize_mcp_server()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                info = loop.run_until_complete(server.get_enhanced_server_info())
                return jsonify(info)
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Server info error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/mcp/metrics', methods=['GET'])
    def metrics_endpoint():
        """Metrics endpoint"""
        try:
            from cryptotrading.core.protocols.mcp.metrics import mcp_metrics
            
            metrics_data = mcp_metrics.export_metrics('json')
            return jsonify(json.loads(metrics_data))
            
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app

# Create the Flask app
app = create_app()

# Vercel handler
def handler(request):
    """Vercel serverless handler"""
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    # For local development
    app.run(debug=True, port=8080)
