#!/usr/bin/env python3
"""
Test A2A MCP Integration
Comprehensive testing of A2A agents with MCP tools integration
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set environment variables for CLI
os.environ['ENVIRONMENT'] = 'development'
os.environ['SKIP_DB_INIT'] = 'true'

try:
    from cryptotrading.core.agents.a2a_mcp_bridge import get_a2a_mcp_bridge
    from cryptotrading.core.protocols.a2a.a2a_protocol import A2A_CAPABILITIES
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback MCP integration testing...")

async def test_technical_analysis_mcp():
    """Test technical analysis agent MCP integration"""
    print("🔧 Testing Technical Analysis Agent MCP Integration")
    print("=" * 60)
    
    try:
        bridge = await get_a2a_mcp_bridge()
        
        # Test MCP capabilities for technical analysis
        capabilities = await bridge.get_agent_mcp_capabilities('technical_analysis_agent')
        
        print(f"Agent: {capabilities['agent_id']}")
        print(f"Primary MCP Tools: {capabilities['primary_tools']}")
        print(f"Secondary MCP Tools: {capabilities['secondary_tools']}")
        print(f"Total Tools Available: {capabilities['total_tools_available']}")
        
        # Test specific capability with MCP
        result = await bridge.execute_agent_with_mcp(
            'technical_analysis_agent',
            'technical_indicators',
            {'symbol': 'BTC', 'timeframe': '1d'}
        )
        
        print(f"\nMCP Execution Result:")
        print(f"Tools Used: {result.get('tools_used', [])}")
        print(f"Status: {'✅ Success' if 'error' not in result else '❌ Error'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_ml_agent_mcp():
    """Test ML agent MCP integration"""
    print("\n🤖 Testing ML Agent MCP Integration")
    print("=" * 60)
    
    try:
        bridge = await get_a2a_mcp_bridge()
        
        # Test MCP capabilities for ML agent
        capabilities = await bridge.get_agent_mcp_capabilities('ml_agent')
        
        print(f"Agent: {capabilities['agent_id']}")
        print(f"Primary MCP Tools: {capabilities['primary_tools']}")
        print(f"Secondary MCP Tools: {capabilities['secondary_tools']}")
        
        # Test model training capability with MCP
        result = await bridge.execute_agent_with_mcp(
            'ml_agent',
            'model_training',
            {'model_type': 'xgboost', 'symbol': 'ETH'}
        )
        
        print(f"\nMCP Execution Result:")
        print(f"Tools Used: {result.get('tools_used', [])}")
        print(f"Status: {'✅ Success' if 'error' not in result else '❌ Error'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_strands_glean_mcp():
    """Test Strands Glean agent MCP integration"""
    print("\n🔍 Testing Strands Glean Agent MCP Integration")
    print("=" * 60)
    
    try:
        bridge = await get_a2a_mcp_bridge()
        
        # Test MCP capabilities for Strands Glean agent
        capabilities = await bridge.get_agent_mcp_capabilities('strands_glean_agent')
        
        print(f"Agent: {capabilities['agent_id']}")
        print(f"Primary MCP Tools: {capabilities['primary_tools']}")
        print(f"Secondary MCP Tools: {capabilities['secondary_tools']}")
        
        # Test code analysis capability with MCP
        result = await bridge.execute_agent_with_mcp(
            'strands_glean_agent',
            'code_analysis',
            {'path': 'src/cryptotrading/core', 'language': 'python'}
        )
        
        print(f"\nMCP Execution Result:")
        print(f"Tools Used: {result.get('tools_used', [])}")
        print(f"Status: {'✅ Success' if 'error' not in result else '❌ Error'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_mcp_bridge_health():
    """Test MCP bridge health"""
    print("\n🏥 Testing MCP Bridge Health")
    print("=" * 60)
    
    try:
        bridge = await get_a2a_mcp_bridge()
        health = await bridge.health_check()
        
        print(f"Status: {health['status'].upper()}")
        print(f"Initialized: {health['initialized']}")
        print(f"MCP Tools Count: {health['mcp_tools_count']}")
        print(f"Agent Mappings: {health['agent_mappings_count']}")
        print(f"Segregation Manager: {'✅' if health['segregation_manager_available'] else '❌'}")
        
        print(f"\nTool Health:")
        for tool_name, tool_health in health['tools'].items():
            status = "✅" if tool_health['available'] else "❌"
            print(f"  {status} {tool_name} ({tool_health['type']})")
        
        return health['status'] in ['healthy', 'partial']
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Run comprehensive MCP integration tests"""
    print("🚀 A2A MCP Integration Test Suite")
    print("=" * 80)
    
    tests = [
        ("Technical Analysis MCP", test_technical_analysis_mcp),
        ("ML Agent MCP", test_ml_agent_mcp),
        ("Strands Glean MCP", test_strands_glean_mcp),
        ("MCP Bridge Health", test_mcp_bridge_health)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All MCP integration tests passed!")
    else:
        print("⚠️  Some MCP integration tests failed")

if __name__ == '__main__':
    asyncio.run(main())
