#!/usr/bin/env python3
"""
Test Real A2A System with Observability Integration
Tests the actual historical loader agent and database agent with full observability
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_historical_agent():
    """Test the real historical loader agent with observability"""
    
    print("🧪 Testing Real Historical Loader Agent with Observability")
    print("=" * 60)
    
    try:
        # Import real agent
        from cryptotrading.core.agents.specialized.historical_loader_agent import HistoricalLoaderAgent
        
        # Create agent instance  
        agent = HistoricalLoaderAgent()
        
        print("✅ Historical agent created successfully")
        
        # Test the process_request method with observability
        request = "Load 7 days of BTC-USD historical data from Yahoo Finance"
        result = await agent.process_request(request)
        
        print(f"📊 Agent Response: {result}")
        
        # Test the load_symbol_data tool directly
        tools = agent._create_tools()
        load_tool = next((tool for tool in tools if hasattr(tool, '__name__') and 'load_symbol_data' in tool.__name__), None)
        
        if load_tool:
            print("\n🔧 Testing load_symbol_data tool...")
            tool_result = load_tool("BTC", 7, False)
            print(f"📈 Tool Result Success: {tool_result.get('success', False)}")
            if tool_result.get('success'):
                data = tool_result.get('data', {})
                print(f"   Records: {data.get('records_count', 0)}")
                print(f"   Symbol: {data.get('symbol', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error testing historical agent: {e}")
        import traceback
        traceback.print_exc()

async def test_database_agent():
    """Test the real database agent with observability"""
    
    print("\n🧪 Testing Real Database Agent with Observability")
    print("=" * 60)
    
    try:
        # Import real agent
        from cryptotrading.core.agents.specialized.database_agent import DatabaseAgent
        
        # Create agent instance
        agent = DatabaseAgent()
        
        print("✅ Database agent created successfully")
        
        # Test store_historical_data tool
        tools = agent._create_tools()
        store_tool = next((tool for tool in tools if hasattr(tool, '__name__') and 'store_historical_data' in tool.__name__), None)
        
        if store_tool:
            print("\n🔧 Testing store_historical_data tool...")
            
            # Create test data payload
            test_payload = {
                "symbol": "BTC-USD",
                "data": [
                    {
                        "date": "2024-01-01",
                        "open": 45000.0,
                        "high": 46000.0,
                        "low": 44000.0,
                        "close": 45500.0,
                        "volume": 1000000
                    },
                    {
                        "date": "2024-01-02", 
                        "open": 45500.0,
                        "high": 47000.0,
                        "low": 45000.0,
                        "close": 46500.0,
                        "volume": 1200000
                    }
                ]
            }
            
            tool_result = store_tool(test_payload, "sqlite", False)  # No AI analysis for speed
            print(f"💾 Storage Result Success: {tool_result.get('success', False)}")
            if tool_result.get('success'):
                print(f"   Records Stored: {tool_result.get('records_stored', 0)}")
        
    except Exception as e:
        print(f"❌ Error testing database agent: {e}")
        import traceback
        traceback.print_exc()

def test_observability_components():
    """Test core observability components"""
    
    print("\n🧪 Testing Observability Components")
    print("=" * 40)
    
    try:
        # Test imports
        from cryptotrading.infrastructure.monitoring import (
            get_logger, get_tracer, get_error_tracker, get_metrics,
            get_business_metrics, trace_context
        )
        
        # Test logger
        logger = get_logger("test")
        logger.info("Test log message", extra={"test": "value"})
        print("✅ Logger working")
        
        # Test tracer  
        tracer = get_tracer("test-service")
        with trace_context("test_operation") as span:
            span.set_attribute("test.attribute", "test_value")
        print("✅ Tracer working")
        
        # Test error tracker
        error_tracker = get_error_tracker()
        try:
            raise ValueError("Test error for tracking")
        except ValueError as e:
            error_id = error_tracker.track_error(e)
            print(f"✅ Error tracker working - Error ID: {error_id}")
        
        # Test metrics
        metrics = get_metrics()
        metrics.counter("test.counter", 1.0, {"test": "tag"})
        print("✅ Metrics working")
        
        # Test business metrics
        business_metrics = get_business_metrics()
        business_metrics.track_api_request("/test", "GET", 200, 150.0)
        print("✅ Business metrics working")
        
        # Get summary
        error_summary = error_tracker.get_error_summary(hours=1)
        metrics_summary = metrics.get_all_metrics_summary(hours=1)
        
        print(f"\n📊 Observability Status:")
        print(f"   Errors tracked: {error_summary['total_errors']}")
        print(f"   Metrics tracked: {metrics_summary['total_metrics']}")
        
    except Exception as e:
        print(f"❌ Error testing observability: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🎯 Rex Trading System - Real Observability Integration Test")
    print("=" * 60)
    
    # Test observability components first
    test_observability_components()
    
    # Test real agents with observability
    await test_historical_agent()
    await test_database_agent()
    
    print("\n🎉 Real Observability Integration Test Complete!")
    print("=" * 60)
    print("📊 To view observability data:")
    print("   1. Start Flask app: python app.py") 
    print("   2. Visit: http://localhost:5000/observability/dashboard.html")
    print("   3. Check logs in: logs/errors.log")
    print("   4. API endpoints:")
    print("      - GET /observability/health")
    print("      - GET /observability/metrics") 
    print("      - GET /observability/errors/summary")

if __name__ == "__main__":
    asyncio.run(main())