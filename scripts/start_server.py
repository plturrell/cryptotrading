#!/usr/bin/env python3
"""
Start development servers for the crypto trading platform
"""

import sys
import os
import subprocess
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

class CryptTradingHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler for serving the webapp"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="webapp", **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def start_webapp_server():
    """Start the webapp server"""
    print("🌐 Starting Fiori webapp server on http://localhost:8080")
    server = HTTPServer(('localhost', 8080), CryptTradingHTTPRequestHandler)
    server.serve_forever()

def start_auth_api():
    """Start the authentication API server"""
    print("🔐 Starting Authentication API on http://localhost:5000")
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run([sys.executable, "api/auth_api.py"])

def main():
    """Start both servers"""
    
    # Change to project directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("🚀 Starting Cryptocurrency Trading Platform Servers")
    print("=" * 60)
    
    # Start authentication API in background
    api_thread = threading.Thread(target=start_auth_api, daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(2)
    
    print("\n🎯 Servers running:")
    print("   📱 Fiori App:     http://localhost:8080/login.html")
    print("   🔐 Auth API:      http://localhost:5001/api/health")
    print("   📚 Login Guide:   Use the demo user selector or manual login")
    print("\n👥 Demo Users:")
    print("   • Craig Wright (Admin):  craig / Craig2024!")
    print("   • Irina Petrova (Trader): irina / Irina2024!")
    print("   • Dasha Ivanova (Analyst): dasha / Dasha2024!")
    print("   • Dany Chen (Trader):     dany / Dany2024!")
    print("\n🛑 Press Ctrl+C to stop servers")
    print("=" * 60)
    
    try:
        # Start webapp server (main thread)
        start_webapp_server()
    except KeyboardInterrupt:
        print("\n\n🛑 Servers stopped. Goodbye!")

if __name__ == "__main__":
    main()