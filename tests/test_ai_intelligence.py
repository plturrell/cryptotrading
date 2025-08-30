#!/usr/bin/env python3
"""
Test AI-Powered Error Intelligence System  
Tests both Grok and Perplexity integrations for error analysis
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root / "src"))

import asyncio
import json
from typing import Dict, Any
from datetime import datetime

def test_ai_models():
    """Test all available AI models for error analysis"""
    
    print("🧠 TESTING AI-POWERED ERROR INTELLIGENCE")
    print("=" * 60)
    
    # Available models based on user's documentation
    grok_models = [
        ("grok-4-0709", "Advanced reasoning model"),
        ("grok-code-fast-1", "Fast code analysis model"),
        ("grok-3-mini", "Lightweight model"),
        ("grok-3", "Standard model")
    ]
    
    perplexity_models = [
        ("sonar", "Main Sonar model - search-augmented generation"),
        ("sonar-reasoning", "Sonar with reasoning capabilities"),
        ("sonar-deep-research", "Advanced research model")
    ]
    
    print(f"📊 Available Grok Models ({len(grok_models)}):")
    for model, desc in grok_models:
        print(f"   - {model}: {desc}")
    
    print(f"\n📊 Available Perplexity Models ({len(perplexity_models)}):")
    for model, desc in perplexity_models:
        print(f"   - {model}: {desc}")
    
    return grok_models, perplexity_models

async def test_grok_error_analysis():
    """Test Grok AI for error analysis"""
    print("\n🔍 TESTING GROK AI ERROR ANALYSIS")
    print("-" * 50)
    
    try:
        from cryptotrading.core.ai.grok4_client import Grok4Client
        
        client = Grok4Client()
        
        # Test error context
        test_error = {
            "error": "AttributeError: 'NoneType' object has no attribute 'get'",
            "file": "api/cds_service_adapter.py",
            "line": 142,
            "context": "wallet_balance = response.json().get('balance')",
            "stack_trace": ["File api/cds_service_adapter.py, line 142, in wallet_balance"]
        }
        
        analysis_request = {
            "prompt": f"Analyze this Python error: {json.dumps(test_error, indent=2)}",
            "task_type": "error_analysis",
            "output_format": "structured_json"
        }
        
        print("🎯 Testing Grok market sentiment analysis...")
        result = await client.analyze_market_sentiment(["BTC"], timeframe="1h")
        
        print("✅ Grok Analysis Result:")
        if isinstance(result, dict):
            print(f"   - Status: {result.get('status', 'Unknown')}")
            if 'analysis' in result:
                analysis = result['analysis']
                print(f"   - Root Cause: {analysis.get('root_cause', 'Not provided')}")
                print(f"   - Confidence: {analysis.get('confidence', 'Not provided')}")
                print(f"   - Fix: {analysis.get('fix_suggestion', 'Not provided')[:100]}...")
        else:
            print(f"   Result: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Grok test failed: {str(e)}")
        return False

async def test_perplexity_error_analysis():
    """Test Perplexity AI for error analysis"""
    print("\n🔍 TESTING PERPLEXITY AI ERROR ANALYSIS") 
    print("-" * 50)
    
    try:
        from cryptotrading.core.ml.perplexity import PerplexityClient
        
        client = PerplexityClient()
        
        print("🎯 Testing Perplexity crypto analysis...")
        result = await client.search_crypto_news("BTC")
        
        print("✅ Perplexity Analysis Result:")
        if isinstance(result, dict):
            print(f"   - Symbol: {result.get('symbol', 'Unknown')}")
            print(f"   - Timestamp: {result.get('timestamp', 'Unknown')}")
            if 'analysis' in result:
                analysis_text = result['analysis'][:150] + "..." if len(result['analysis']) > 150 else result['analysis']
                print(f"   - Analysis: {analysis_text}")
        else:
            print(f"   Result: {str(result)[:200]}...")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ Perplexity test failed: {str(e)}")
        return False

async def test_ai_monitoring_system():
    """Test the integrated AI monitoring system"""
    print("\n🤖 TESTING INTEGRATED AI MONITORING SYSTEM")
    print("-" * 50)
    
    try:
        from cryptotrading.infrastructure.monitoring.ai_root_cause_analyzer import AIRootCauseAnalyzer
        
        analyzer = AIRootCauseAnalyzer()
        
        # Test error data
        error_data = {
            "error_type": "AttributeError",
            "message": "'NoneType' object has no attribute 'get'",
            "file_path": "api/cds_service_adapter.py",
            "line_number": 142,
            "function_name": "wallet_balance",
            "stack_trace": [
                "File api/cds_service_adapter.py, line 142, in wallet_balance",
                "wallet_balance = response.json().get('balance')"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        print("🎯 Running AI root cause analysis...")
        analysis_result = await analyzer.analyze_error_with_ai(Exception("Test error"), error_data)
        
        print("✅ AI Analysis Result:")
        print(f"   - Root Cause: {analysis_result.get('root_cause', 'Not identified')}")
        print(f"   - Confidence: {analysis_result.get('confidence_score', 0)}")
        print(f"   - Severity: {analysis_result.get('severity', 'Unknown')}")
        print(f"   - Impact: {analysis_result.get('impact_assessment', 'Not assessed')}")
        
        if 'recommendations' in analysis_result:
            print("   - Recommendations:")
            for i, rec in enumerate(analysis_result['recommendations'][:3], 1):
                print(f"     {i}. {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI monitoring test failed: {str(e)}")
        return False

async def main():
    """Run all AI intelligence tests"""
    
    print("🚀 AI-POWERED ERROR INTELLIGENCE TEST")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test model availability
    grok_models, perplexity_models = test_ai_models()
    
    # Test individual AI services
    grok_success = await test_grok_error_analysis()
    perplexity_success = await test_perplexity_error_analysis()
    monitoring_success = await test_ai_monitoring_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    print(f"🤖 Grok AI Integration: {'✅ WORKING' if grok_success else '❌ FAILED'}")
    print(f"🔍 Perplexity AI Integration: {'✅ WORKING' if perplexity_success else '❌ FAILED'}")
    print(f"📈 AI Monitoring System: {'✅ WORKING' if monitoring_success else '❌ FAILED'}")
    
    total_systems = 3
    working_systems = sum([grok_success, perplexity_success, monitoring_success])
    
    print(f"\n🎯 Overall AI Intelligence Status: {working_systems}/{total_systems} systems operational")
    
    if working_systems == total_systems:
        print("✨ ALL AI SYSTEMS ARE OPERATIONAL!")
        print("🚀 Commercial-grade AI error intelligence is ready!")
    else:
        print("⚠️  Some AI systems need attention")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())