"""
Vercel Python Function for MCTS Calculation
Provides Python-based serverless endpoint
"""
import os
import sys
import json
import time
from http.server import BaseHTTPRequestHandler

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import MCTS components
from src.cryptotrading.core.agents.specialized.mcts_calculation_agent import (
    ProductionMCTSCalculationAgent,
    ProductionTradingEnvironment,
    MCTSConfig
)


class handler(BaseHTTPRequestHandler):
    """Vercel Python function handler"""
    
    def do_POST(self):
        """Handle POST requests for MCTS calculations"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            request_data = json.loads(body) if body else {}
            
            # Validate request
            if 'parameters' not in request_data:
                self.send_error(400, "Missing required field: parameters")
                return
            
            # Create MCTS agent
            config = MCTSConfig()
            agent = ProductionMCTSCalculationAgent(
                agent_id=f"vercel_mcts_{int(time.time())}",
                config=config
            )
            
            # Set up environment
            env_config = request_data['parameters']
            agent.environment = ProductionTradingEnvironment(env_config)
            
            # Run MCTS calculation
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            iterations = request_data.get('iterations', config.iterations)
            result = loop.run_until_complete(
                agent.run_mcts_parallel(iterations=iterations)
            )
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'success': True,
                'result': result,
                'agent_id': agent.agent_id,
                'runtime': 'python',
                'timestamp': time.time()
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        health = {
            'status': 'healthy',
            'runtime': 'python',
            'version': '1.0.0',
            'timestamp': time.time()
        }
        
        self.wfile.write(json.dumps(health).encode())