#!/usr/bin/env python3
"""
Test AI Gateway and Vercel Blob storage integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.рекс.ai import AIGatewayClient
from src.рекс.storage import VercelBlobClient, put_json_blob
import json

def test_ai_gateway():
    """Test AI Gateway with Claude-4-Sonnet"""
    print("\n=== Testing AI Gateway (Claude-4-Sonnet) ===")
    
    # Initialize client
    ai = AIGatewayClient()
    
    # Test 1: Market Analysis
    print("\n1. Testing Market Analysis...")
    market_data = {
        "symbol": "BTC",
        "price": 65000,
        "volume_24h": 28500000000,
        "change_24h": 2.5,
        "indicators": {
            "rsi": 65,
            "macd": 150,
            "ma_50": 63000,
            "ma_200": 58000
        }
    }
    
    analysis = ai.analyze_market(market_data)
    print(f"Signal: {analysis.get('signal')}")
    print(f"Confidence: {analysis.get('confidence')}%")
    print(f"Analysis: {analysis.get('analysis', '')[:200]}...")
    
    # Test 2: Trading Signals
    print("\n2. Testing Trading Signal Generation...")
    technical_data = {
        "rsi": 72,
        "macd": 200,
        "bollinger_upper": 66000,
        "bollinger_lower": 62000,
        "volume_trend": "increasing"
    }
    
    signal = ai.get_trading_signals("ETH", "4h", technical_data)
    print(f"Trading Signal: {signal}")
    
    # Test 3: Strategy Generation
    print("\n3. Testing Strategy Generation...")
    user_profile = {
        "risk_tolerance": "medium",
        "capital": 10000,
        "experience": "intermediate",
        "goals": "steady growth",
        "time_horizon": "6 months"
    }
    
    strategy = ai.generate_trading_strategy(user_profile)
    print(f"Strategy generated: {len(strategy.get('strategy', ''))} characters")
    
    return True

def test_vercel_blob():
    """Test Vercel Blob storage"""
    print("\n=== Testing Vercel Blob Storage ===")
    
    try:
        # Initialize client
        blob = VercelBlobClient()
        
        # Test 1: Store Trading Signal
        print("\n1. Storing Trading Signal...")
        signal_data = {
            "signal": "BUY",
            "confidence": 85,
            "entry_price": 65000,
            "stop_loss": 63000,
            "take_profit": 68000,
            "reasoning": "Strong bullish momentum with RSI confirmation"
        }
        
        result = blob.store_trading_signal("BTC", signal_data)
        print(f"Stored at: {result.get('url')}")
        
        # Test 2: Store Market Analysis
        print("\n2. Storing Market Analysis...")
        analysis_data = {
            "market_sentiment": "BULLISH",
            "btc_analysis": "Strong support at 63k",
            "eth_analysis": "Breaking resistance at 3.5k",
            "defi_trends": ["Uniswap volume increasing", "Aave TVL growing"]
        }
        
        result = blob.store_market_analysis(analysis_data)
        print(f"Stored at: {result.get('url')}")
        
        # Test 3: List Stored Signals
        print("\n3. Listing Stored Signals...")
        signals = blob.get_latest_signals("BTC", 5)
        print(f"Found {len(signals)} stored signals")
        
        # Test 4: Quick JSON Upload
        print("\n4. Testing Quick JSON Upload...")
        test_data = {"test": "data", "platform": "рекс.com"}
        result = put_json_blob("test/quick_upload.json", test_data)
        print(f"Quick upload result: {result}")
        
        return True
        
    except Exception as e:
        print(f"Vercel Blob error: {e}")
        print("Make sure BLOB_READ_WRITE_TOKEN is set in environment")
        return False

def main():
    """Run all tests"""
    print("Testing AI Gateway and Vercel Blob Storage")
    print("=" * 50)
    
    # Test AI Gateway
    ai_success = False
    try:
        ai_success = test_ai_gateway()
    except Exception as e:
        print(f"AI Gateway test failed: {e}")
    
    # Test Vercel Blob
    blob_success = False
    try:
        blob_success = test_vercel_blob()
    except Exception as e:
        print(f"Vercel Blob test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"AI Gateway (Claude-4-Sonnet): {'✓ PASSED' if ai_success else '✗ FAILED'}")
    print(f"Vercel Blob Storage: {'✓ PASSED' if blob_success else '✗ FAILED'}")
    
    if not blob_success:
        print("\nNote: Vercel Blob requires BLOB_READ_WRITE_TOKEN environment variable")
        print("Get your token from: https://vercel.com/dashboard/stores")

if __name__ == "__main__":
    main()