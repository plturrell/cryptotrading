"""
Vercel API Route for MCP Server

Production-ready MCP server endpoint with full security integration.
Provides HTTP to MCP protocol bridge for Vercel deployment.

Features:
- JWT and API key authentication
- Rate limiting via Vercel Edge Config
- Input validation and sanitization
- CORS support
- Request/response logging
- Error handling
- Environment-based configuration
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs

# Import MCP components
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.protocol import MCPRequest, MCPResponse
from cryptotrading.core.protocols.mcp.transport import MCPTransport
from cryptotrading.core.protocols.mcp.security import (
    VercelSecurityMiddleware,
    AuthenticationError,
    RateLimitExceeded,
    ValidationError
)
from cryptotrading.core.protocols.mcp.tools import MCPTool
from cryptotrading.infrastructure.database import UnifiedDatabase, DatabaseConfig

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTTPTransport(MCPTransport):
    """HTTP transport for MCP over Vercel API routes"""
    
    def __init__(self):
        super().__init__()
        self.is_connected = True
        self.responses = []
    
    async def connect(self) -> bool:
        self.is_connected = True
        return True
    
    async def disconnect(self):
        self.is_connected = False
    
    async def send_message(self, message: str):
        """Store response for HTTP return"""
        self.responses.append(message)
    
    async def receive_messages(self):
        """Not used in HTTP mode"""
        pass


class VercelMCPHandler(BaseHTTPRequestHandler):
    """Vercel MCP request handler"""
    
    # Class-level server instance for efficiency
    _server = None
    _initialized = False
    _database = None
    
    @classmethod
    async def get_database(cls):
        """Get or create database instance"""
        if cls._database is None:
            cls._database = UnifiedDatabase()
            await cls._database.initialize()
            logger.info("Database initialized for MCP server")
        return cls._database
    
    @classmethod
    def get_server(cls):
        """Get or create MCP server instance"""
        if cls._server is None or not cls._initialized:
            transport = HTTPTransport()
            cls._server = MCPServer(
                name=os.getenv("MCP_SERVER_NAME", "vercel-mcp-server"),
                version=os.getenv("MCP_SERVER_VERSION", "1.0.0"),
                transport=transport,
                security_middleware=VercelSecurityMiddleware()
            )
            
            # Setup crypto trading tools
            cls._setup_tools()
            cls._server.is_initialized = True
            cls._initialized = True
            logger.info("MCP server initialized for Vercel")
        
        return cls._server
    
    @classmethod
    def _setup_tools(cls):
        """Setup crypto trading tools and advanced CLRS+Tree analysis tools"""
        server = cls._server
        
        # Import ALL segregated MCP tools
        try:
            from src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools import (
                create_all_segregated_tools
            )
            from src.cryptotrading.infrastructure.analysis.mcp_agent_segregation import (
                SecureToolWrapper, 
                get_segregation_manager
            )
            from src.cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import (
                GLEAN_MCP_TOOLS
            )
            
            # Create all segregated tools (crypto trading + analysis)
            try:
                all_segregated_tools = create_all_segregated_tools()
                segregation_manager = get_segregation_manager()
            except Exception as import_error:
                logger.error("Failed to create segregated tools: %s", import_error)
                all_segregated_tools = {}
                segregation_manager = None
            
            # Wrap each tool with security enforcement
            if segregation_manager and all_segregated_tools:
                for tool_name, tool in all_segregated_tools.items():
                    try:
                        wrapped_tool = SecureToolWrapper(tool, segregation_manager)
                        server.add_tool(wrapped_tool)
                        logger.info("Added segregated tool: %s", tool_name)
                    except Exception as tool_error:
                        logger.error("Failed to add tool %s: %s", tool_name, tool_error)
            
            # Add Glean MCP tools
            for tool_name, tool_config in GLEAN_MCP_TOOLS.items():
                try:
                    server.add_tool(tool_config["function"], tool_config["metadata"])
                    logger.info("Added Glean MCP tool: %s", tool_name)
                except Exception as glean_error:
                    logger.error("Failed to add Glean tool %s: %s", tool_name, glean_error)
            
            logger.info("Added ALL segregated MCP tools to server - crypto trading, analysis, and Glean tools")
            
        except ImportError as e:
            logger.warning("Could not import segregated MCP tools: %s", e)
            # Fallback to basic tools without segregation (not recommended for production)
            logger.warning("Running without agent segregation - NOT RECOMMENDED FOR PRODUCTION")
        except Exception as e:
            logger.error("Unexpected error setting up MCP tools: %s", e)
        
        logger.info("Added %d tools to MCP server", len(server.tools))
    
    def do_OPTIONS(self):
        """Handle CORS preflight request"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def do_POST(self):
        """Handle POST request"""
        try:
            # Get content length
            content_length = int(self.headers.get('Content-Length', 0))
            
            # Read request body
            if content_length > 0:
                body = self.rfile.read(content_length).decode('utf-8')
            else:
                body = ""
            
            # Process request
            response = asyncio.run(self.process_mcp_request(body))
            
            # Send response
            self.send_response(response['status'])
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response['body'].encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            self.send_error_response(500, str(e))
    
    async def process_mcp_request(self, body: str) -> Dict[str, Any]:
        """Process MCP request with security"""
        try:
            # Parse JSON request
            if not body:
                return {
                    'status': 400,
                    'body': json.dumps({'error': 'Empty request body'})
                }
            
            try:
                request_data = json.loads(body)
            except json.JSONDecodeError as e:
                return {
                    'status': 400,
                    'body': json.dumps({'error': f'Invalid JSON: {str(e)}'})
                }
            
            # Get server instance
            server = self.get_server()
            
            # Extract headers
            headers = dict(self.headers)
            
            # Clear previous responses
            server.transport.responses.clear()
            
            # Process through MCP server with security
            await server._handle_message(body, headers)
            
            # Get response
            if server.transport.responses:
                response_body = server.transport.responses[0]
                return {
                    'status': 200,
                    'body': response_body
                }
            else:
                return {
                    'status': 500,
                    'body': json.dumps({'error': 'No response generated'})
                }
                
        except Exception as e:
            logger.error(f"MCP processing error: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": request_data.get("id") if 'request_data' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            return {
                'status': 500,
                'body': json.dumps(error_response)
            }
    
    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
        self.send_header('Access-Control-Max-Age', '86400')
    
    def send_error_response(self, status: int, message: str):
        """Send error response"""
        self.send_response(status)
        self.send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        error_body = json.dumps({'error': message})
        self.wfile.write(error_body.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to use proper logging"""
        logger.info(format % args)


def handler(request):
    """Vercel handler function"""
    import io
    import sys
    from urllib.parse import urlparse, parse_qs
    
    # Create fake socket for HTTPRequestHandler
    class FakeSocket:
        def __init__(self, data):
            self.file = io.BytesIO(data)
        
        def makefile(self, mode):
            return self.file
    
    # Build HTTP request
    method = request.method
    url = request.url
    headers = dict(request.headers) if hasattr(request, 'headers') else {}
    body = request.body if hasattr(request, 'body') else b''
    
    # Build HTTP request string
    parsed_url = urlparse(url)
    path = parsed_url.path
    query = parsed_url.query
    
    request_line = f"{method} {path}"
    if query:
        request_line += f"?{query}"
    request_line += " HTTP/1.1\r\n"
    
    # Build headers
    header_lines = []
    for key, value in headers.items():
        header_lines.append(f"{key}: {value}\r\n")
    
    if body and 'Content-Length' not in headers:
        header_lines.append(f"Content-Length: {len(body)}\r\n")
    
    # Combine request
    http_request = request_line + "".join(header_lines) + "\r\n"
    http_request_bytes = http_request.encode('utf-8') + body
    
    # Create fake socket
    fake_socket = FakeSocket(http_request_bytes)
    
    # Capture response
    class ResponseCapture:
        def __init__(self):
            self.status = 200
            self.headers = {}
            self.body = b''
        
        def start_response(self, status, headers):
            self.status = int(status.split()[0])
            self.headers = dict(headers)
    
    response_capture = ResponseCapture()
    
    # Create handler
    class CaptureHandler(VercelMCPHandler):
        def __init__(self, capture):
            self.capture = capture
            self.rfile = fake_socket.makefile('rb')
            self.wfile = io.BytesIO()
            self.headers = {}
            
            # Parse the request
            self.raw_requestline = self.rfile.readline()
            self.parse_request()
            
            # Process based on method
            if self.command == 'OPTIONS':
                self.do_OPTIONS()
            elif self.command == 'POST':
                self.do_POST()
            else:
                self.send_error_response(405, 'Method not allowed')
            
            # Capture response
            self.capture.body = self.wfile.getvalue()
        
        def send_response(self, code, message=None):
            self.capture.status = code
        
        def send_header(self, keyword, value):
            self.capture.headers[keyword] = value
        
        def end_headers(self):
            pass
    
    # Process request
    try:
        handler_instance = CaptureHandler(response_capture)
        
        return {
            'statusCode': response_capture.status,
            'headers': response_capture.headers,
            'body': response_capture.body.decode('utf-8')
        }
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Internal server error'})
        }


# For local testing
if __name__ == "__main__":
    from http.server import HTTPServer
    
    server = HTTPServer(('localhost', 8080), VercelMCPHandler)
    print("MCP server running on http://localhost:8080")
    print("Test with: curl -X POST http://localhost:8080 -H 'Content-Type: application/json' -d '{\"jsonrpc\":\"2.0\",\"method\":\"tools/list\",\"id\":\"test\"}'")
    server.serve_forever()