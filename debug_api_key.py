#!/usr/bin/env python3
"""
Debug API Key Verification
"""
import sys
import hashlib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptotrading.core.protocols.mcp.security.auth import APIKeyManager

def debug_api_key():
    """Debug API key verification"""
    api_key_manager = APIKeyManager()
    
    # Get the current API key from the stored keys
    if not api_key_manager.keys:
        print("No API keys found")
        return
    
    # Get first key for testing
    stored_key = list(api_key_manager.keys.values())[0]
    print(f"Stored key ID: {stored_key.key_id}")
    print(f"Stored key hash: {stored_key.key_hash}")
    
    # Test with the actual key value (use the key_id as the key)
    test_key = stored_key.key_id  # Use the stored key_id as the actual key
    test_hash = hashlib.sha256(test_key.encode()).hexdigest()
    print(f"Test key: {test_key}")
    print(f"Test hash: {test_hash}")
    print(f"Hashes match: {test_hash == stored_key.key_hash}")
    
    # Try verification
    verified = api_key_manager.verify_api_key(test_key)
    print(f"Verification result: {verified}")
    
    if verified:
        print("✅ API key verification successful")
    else:
        print("❌ API key verification failed")

if __name__ == "__main__":
    debug_api_key()
