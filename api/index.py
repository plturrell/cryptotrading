"""
Web Interface for MCP Server
Provides documentation, testing interface, and API explorer
"""

import json
import os
from datetime import datetime


def generate_html() -> str:
    """Generate HTML interface for MCP server"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Server - API Documentation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; color: #333; background: #f8fafc;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px 0; text-align: center; margin-bottom: 30px;
            border-radius: 10px;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }
        .card { 
            background: white; padding: 25px; border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;
        }
        .card h2 { color: #2d3748; margin-bottom: 15px; font-size: 1.3em; }
        .endpoints { margin-bottom: 30px; }
        .endpoint { 
            background: #f7fafc; padding: 15px; margin: 10px 0; border-radius: 8px;
            border-left: 4px solid #4299e1;
        }
        .method { 
            display: inline-block; padding: 4px 8px; border-radius: 4px; 
            color: white; font-size: 0.8em; font-weight: bold; margin-right: 10px;
        }
        .method.get { background: #48bb78; }
        .method.post { background: #ed8936; }
        .test-area { 
            background: white; padding: 25px; border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px;
        }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: 600; }
        .form-group select, .form-group textarea, .form-group input {
            width: 100%; padding: 10px; border: 1px solid #cbd5e0; border-radius: 6px;
            font-family: monospace; font-size: 14px;
        }
        .form-group textarea { height: 120px; resize: vertical; }
        .btn { 
            background: #4299e1; color: white; padding: 12px 24px; border: none;
            border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 600;
        }
        .btn:hover { background: #3182ce; }
        .btn:disabled { background: #a0aec0; cursor: not-allowed; }
        .response { 
            margin-top: 20px; padding: 15px; background: #f7fafc; border-radius: 6px;
            border: 1px solid #e2e8f0; font-family: monospace; white-space: pre-wrap;
            max-height: 400px; overflow-y: auto;
        }
        .success { border-left: 4px solid #48bb78; background: #f0fff4; }
        .error { border-left: 4px solid #f56565; background: #fed7d7; }
        .status { display: inline-block; padding: 4px 8px; border-radius: 4px; margin-left: 10px; }
        .status.online { background: #c6f6d5; color: #22543d; }
        .status.offline { background: #fed7e2; color: #742a2a; }
        .examples { background: #f7fafc; padding: 20px; border-radius: 8px; margin-top: 20px; }
        .examples h3 { margin-bottom: 15px; color: #2d3748; }
        .example { background: white; padding: 15px; margin: 10px 0; border-radius: 6px; border: 1px solid #e2e8f0; }
        .example h4 { color: #4a5568; margin-bottom: 10px; }
        .code { background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 6px; font-family: monospace; overflow-x: auto; }
        .footer { text-align: center; padding: 30px; color: #718096; border-top: 1px solid #e2e8f0; margin-top: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 MCP Server</h1>
            <p>Model Context Protocol API Documentation & Testing Interface</p>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📊 Server Status</h2>
                <p>Service: <strong>MCP Server</strong> <span id="status" class="status offline">Checking...</span></p>
                <p>Version: <strong>1.0.0</strong></p>
                <p>Protocol: <strong>MCP 2024-11-05</strong></p>
                <p>Environment: <strong id="environment">Production</strong></p>
                <p>Timestamp: <strong id="timestamp">Loading...</strong></p>
            </div>

            <div class="card">
                <h2>🔧 Available Tools</h2>
                <div id="tools-list">
                    <p>Loading tools...</p>
                </div>
            </div>
        </div>

        <div class="endpoints">
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/mcp</strong>
                <p>Get server status, documentation, and health information</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/mcp</strong>
                <p>MCP JSON-RPC endpoint for tool execution and resource access</p>
            </div>
        </div>

        <div class="test-area">
            <h2>🧪 API Tester</h2>
            <p>Test MCP methods directly from your browser:</p>
            
            <div class="form-group">
                <label for="method-select">Method:</label>
                <select id="method-select">
                    <option value="tools/list">tools/list - List available tools</option>
                    <option value="tools/call">tools/call - Execute a tool</option>
                    <option value="resources/list">resources/list - List resources</option>
                    <option value="prompts/list">prompts/list - List prompts</option>
                    <option value="ping">ping - Health check</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="params-input">Parameters (JSON):</label>
                <textarea id="params-input" placeholder='{"name": "echo", "arguments": {"message": "Hello MCP!"}}'></textarea>
            </div>
            
            <button class="btn" onclick="testAPI()">Send Request</button>
            
            <div id="test-response" class="response" style="display: none;"></div>
        </div>

        <div class="examples">
            <h3>📚 Example Requests</h3>
            
            <div class="example">
                <h4>List Tools</h4>
                <div class="code">curl -X POST /api/mcp -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "method": "tools/list", 
  "id": "1"
}'</div>
            </div>
            
            <div class="example">
                <h4>Execute Echo Tool</h4>
                <div class="code">curl -X POST /api/mcp -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "echo",
    "arguments": {"message": "Hello World!"}
  },
  "id": "2"
}'</div>
            </div>
            
            <div class="example">
                <h4>Get Portfolio</h4>
                <div class="code">curl -X POST /api/mcp -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_portfolio",
    "arguments": {}
  },
  "id": "3"
}'</div>
            </div>
        </div>

        <div class="footer">
            <p>Built with ❤️ using the Model Context Protocol</p>
            <p><a href="https://github.com/modelcontextprotocol/specification" target="_blank">MCP Specification</a> | 
               <a href="/api/mcp" target="_blank">API Status</a></p>
        </div>
    </div>

    <script>
        // Check server status
        async function checkStatus() {
            try {
                const response = await fetch('/api/mcp');
                const data = await response.json();
                
                document.getElementById('status').textContent = 'Online';
                document.getElementById('status').className = 'status online';
                document.getElementById('environment').textContent = data.environment || 'Unknown';
                document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();
            } catch (error) {
                document.getElementById('status').textContent = 'Offline';
                document.getElementById('status').className = 'status offline';
            }
        }

        // Load available tools
        async function loadTools() {
            try {
                const response = await fetch('/api/mcp', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        jsonrpc: '2.0',
                        method: 'tools/list',
                        id: 'tools-list'
                    })
                });
                
                const data = await response.json();
                const tools = data.result?.tools || [];
                
                const toolsList = document.getElementById('tools-list');
                if (tools.length === 0) {
                    toolsList.innerHTML = '<p>No tools available</p>';
                } else {
                    toolsList.innerHTML = tools.map(tool => 
                        `<p><strong>${tool.name}</strong>: ${tool.description}</p>`
                    ).join('');
                }
            } catch (error) {
                document.getElementById('tools-list').innerHTML = '<p>Error loading tools</p>';
            }
        }

        // Test API function
        async function testAPI() {
            const method = document.getElementById('method-select').value;
            const paramsText = document.getElementById('params-input').value.trim();
            const responseDiv = document.getElementById('test-response');
            
            let params = {};
            if (paramsText) {
                try {
                    params = JSON.parse(paramsText);
                } catch (error) {
                    responseDiv.innerHTML = `Error: Invalid JSON parameters\\n${error.message}`;
                    responseDiv.className = 'response error';
                    responseDiv.style.display = 'block';
                    return;
                }
            }
            
            const request = {
                jsonrpc: '2.0',
                method: method,
                params: params,
                id: Date.now().toString()
            };
            
            try {
                responseDiv.innerHTML = 'Sending request...';
                responseDiv.className = 'response';
                responseDiv.style.display = 'block';
                
                const response = await fetch('/api/mcp', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(request)
                });
                
                const data = await response.json();
                responseDiv.innerHTML = JSON.stringify(data, null, 2);
                responseDiv.className = response.ok ? 'response success' : 'response error';
                
            } catch (error) {
                responseDiv.innerHTML = `Error: ${error.message}`;
                responseDiv.className = 'response error';
            }
        }

        // Update parameters based on selected method
        document.getElementById('method-select').addEventListener('change', function() {
            const method = this.value;
            const paramsInput = document.getElementById('params-input');
            
            const examples = {
                'tools/call': '{"name": "echo", "arguments": {"message": "Hello MCP!"}}',
                'resources/read': '{"uri": "config://settings"}',
                'prompts/get': '{"name": "analyze_code", "arguments": {"language": "python", "code": "print(\\'hello\\')"}}'
            };
            
            paramsInput.value = examples[method] || '';
        });

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            checkStatus();
            loadTools();
        });
    </script>
</body>
</html>
"""


def handler(request):
    """Vercel handler for web interface"""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html", "Cache-Control": "public, max-age=3600"},
        "body": generate_html(),
    }


# Export for Vercel
def app(request):
    """Main Vercel handler"""
    return handler(request)
