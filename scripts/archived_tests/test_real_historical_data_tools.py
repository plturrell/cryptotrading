#!/usr/bin/env python3
"""
Test script to validate the real historical data tools work
Tests Yahoo Finance and FRED data integration without mocks
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptotrading.core.protocols.mcp.tools import CryptoTradingTools

async def test_yahoo_finance_tool():
    """Test Yahoo Finance data tool"""
    print("🔍 Testing Yahoo Finance data tool...")
    
    tool = CryptoTradingTools.get_yahoo_finance_data_tool()
    
    # Test BTC data download
    result = await tool.execute({
        "symbol": "BTC", 
        "interval": "1d",
        "prepare_for_training": True
    })
    
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Result Status: {'✅ SUCCESS' if not result.isError else '❌ ERROR'}")
    
    if result.content:
        for content in result.content:
            if content.type == "resource" and content.mimeType == "application/json":
                import json
                data = json.loads(content.data)
                print(f"Symbol: {data.get('symbol')}")
                print(f"Data Points: {data.get('data_points')}")
                print(f"Status: {data.get('status')}")
                if data.get('status') == 'success':
                    print(f"Latest Price: ${data.get('latest_price'):.2f}")
                    print(f"Columns: {data.get('columns')}")
                else:
                    print(f"Error: {data.get('error')}")
            else:
                print(f"Content: {content.text}")
    
    print()
    return not result.isError

async def test_fred_tool():
    """Test FRED economic data tool"""
    print("🔍 Testing FRED economic data tool...")
    
    tool = CryptoTradingTools.get_fred_economic_data_tool()
    
    # Test 10-year treasury data
    result = await tool.execute({
        "series_id": "DGS10",
        "frequency": "d"
    })
    
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Result Status: {'✅ SUCCESS' if not result.isError else '❌ ERROR'}")
    
    if result.content:
        for content in result.content:
            if content.type == "resource" and content.mimeType == "application/json":
                import json
                data = json.loads(content.data)
                print(f"Series: {data.get('series_id')}")
                print(f"Description: {data.get('description')}")
                print(f"Data Points: {data.get('data_points')}")
                print(f"Status: {data.get('status')}")
                if data.get('status') == 'success':
                    print(f"Latest Value: {data.get('latest_value')}")
                else:
                    print(f"Error: {data.get('error')}")
            else:
                print(f"Content: {content.text}")
    
    print()
    return not result.isError

async def test_crypto_indicators_tool():
    """Test crypto-relevant indicators tool"""
    print("🔍 Testing crypto-relevant indicators tool...")
    
    tool = CryptoTradingTools.get_crypto_relevant_indicators_tool()
    
    # Test downloading multiple economic indicators
    result = await tool.execute({
        "include_liquidity_metrics": True
    })
    
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Result Status: {'✅ SUCCESS' if not result.isError else '❌ ERROR'}")
    
    if result.content:
        for content in result.content:
            if content.type == "resource" and content.mimeType == "application/json":
                import json
                data = json.loads(content.data)
                print(f"Series Downloaded: {data.get('series_downloaded')}")
                print(f"Status: {data.get('status')}")
                if data.get('status') == 'success':
                    print(f"Series Available: {list(data.get('series_data', {}).keys())}")
                    liquidity = data.get('liquidity_metrics')
                    if liquidity:
                        print(f"Liquidity Metrics: {liquidity.get('columns')}")
                else:
                    print(f"Error: {data.get('error')}")
            else:
                print(f"Content: {content.text}")
    
    print()
    return not result.isError

async def test_comprehensive_dataset_tool():
    """Test comprehensive trading dataset tool"""
    print("🔍 Testing comprehensive trading dataset tool...")
    
    tool = CryptoTradingTools.get_comprehensive_trading_dataset_tool()
    
    # Test loading small dataset
    result = await tool.execute({
        "symbols": ["BTC", "ETH"],
        "align_data": True
    })
    
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Result Status: {'✅ SUCCESS' if not result.isError else '❌ ERROR'}")
    
    if result.content:
        for content in result.content:
            if content.type == "resource" and content.mimeType == "application/json":
                import json
                data = json.loads(content.data)
                print(f"Symbols: {data.get('symbols')}")
                print(f"Sources: {data.get('sources')}")
                print(f"Status: {data.get('status')}")
                if data.get('status') == 'success':
                    summary = data.get('data_summary', {})
                    print(f"Total Files: {summary.get('total_files')}")
                    print(f"Yahoo Files: {summary.get('yahoo_finance', {}).get('file_count')}")
                    print(f"FRED Files: {summary.get('fred', {}).get('file_count')}")
                else:
                    print(f"Error: {data.get('error')}")
            else:
                print(f"Content: {content.text}")
    
    print()
    return not result.isError

async def main():
    """Run all tool tests"""
    print("🚀 Testing Real Historical Data Tools")
    print("=" * 50)
    
    # Check if required environment variables are set
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        print("⚠️  FRED_API_KEY not set - FRED tests may fail")
    
    results = []
    
    # Test each tool
    results.append(await test_yahoo_finance_tool())
    results.append(await test_fred_tool())
    results.append(await test_crypto_indicators_tool())
    results.append(await test_comprehensive_dataset_tool())
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All historical data tools are working!")
        return 0
    else:
        print("⚠️  Some tools failed - check logs above")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))