#!/usr/bin/env python3
"""
Setup MCP API Keys
Creates the required API key for MCP server authentication
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptotrading.core.protocols.mcp.security.auth import APIKeyManager

def setup_api_keys():
    """Setup API keys for MCP server"""
    print("Setting up MCP API keys...")
    
    # Create API key manager
    api_key_manager = APIKeyManager()
    
    # Create the production API key
    key_id, api_key = api_key_manager.create_api_key(
        name="production-mcp-key",
        scopes=["*"],  # All scopes
        rate_limit=1000  # High rate limit for production
    )
    
    print(f"Created API key: {api_key}")
    print(f"Key ID: {key_id}")
    print(f"Use this key in requests: X-API-Key: {api_key}")
    
    # Verify the key works
    verified = api_key_manager.verify_api_key(api_key)
    if verified:
        print("✅ API key verification successful")
    else:
        print("❌ API key verification failed")
    
    return api_key

if __name__ == "__main__":
    api_key = setup_api_keys()
    
    # Update .env file with the API key
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            content = f.read()
        
        # Replace the MCP_API_KEY value
        if "MCP_API_KEY=" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("MCP_API_KEY="):
                    lines[i] = f"MCP_API_KEY={api_key}"
                    break
            
            with open(env_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"✅ Updated .env file with new API key")
        else:
            print("❌ MCP_API_KEY not found in .env file")
    else:
        print("❌ .env file not found")
