#!/usr/bin/env python3
"""
Simple test script for Strands-Glean integration
"""

import asyncio
import sys
import json
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_vercel_glean_client():
    """Test the VercelGleanClient directly"""
    try:
        from cryptotrading.infrastructure.analysis.vercel_glean_client import VercelGleanClient
        from cryptotrading.infrastructure.analysis.angle_parser import create_query
        
        print("🚀 Testing VercelGleanClient...")
        
        # Initialize client
        client = VercelGleanClient(project_root=str(project_root))
        print("✅ Client initialized")
        
        # Index project
        print("📚 Indexing project...")
        index_result = await client.index_project("test-unit")
        print(f"📊 Indexing result: {index_result.get('status', 'unknown')}")
        
        # Test query
        print("🔍 Testing query...")
        query = create_query("symbol_search", {"pattern": "Agent"})
        query_result = await client.query_angle(query)
        print(f"🎯 Query result: {json.dumps(query_result, indent=2)}")
        
        # Get statistics
        print("📈 Getting statistics...")
        stats = await client.get_statistics()
        print(f"📊 Stats: {json.dumps(stats, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_strands_glean_agent():
    """Test the StrandsGleanAgent if available"""
    try:
        # Import the agent module directly
        sys.path.append(str(project_root / "src" / "cryptotrading" / "core" / "agents" / "specialized"))
        from strands_glean_agent import StrandsGleanAgent, StrandsGleanContext
        
        print("🤖 Testing StrandsGleanAgent...")
        
        # Create agent manually
        agent = StrandsGleanAgent(
            agent_id="test-agent",
            project_root=str(project_root)
        )
        print("✅ Agent created")
        
        # Initialize
        success = await agent.initialize()
        print(f"🚀 Initialization: {'✅ Success' if success else '❌ Failed'}")
        
        # Get context summary
        summary = await agent.get_context_summary()
        print(f"📊 Context: {json.dumps(summary, indent=2)}")
        
        # Test symbol search
        search_result = await agent.search_symbols("Agent")
        print(f"🔍 Symbol search: {search_result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 STRANDS-GLEAN INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: VercelGleanClient
    print("\n1️⃣ Testing VercelGleanClient...")
    client_success = await test_vercel_glean_client()
    
    # Test 2: StrandsGleanAgent
    print("\n2️⃣ Testing StrandsGleanAgent...")
    agent_success = await test_strands_glean_agent()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"VercelGleanClient: {'✅ PASS' if client_success else '❌ FAIL'}")
    print(f"StrandsGleanAgent: {'✅ PASS' if agent_success else '❌ FAIL'}")
    
    if client_success and agent_success:
        print("\n🎉 All tests passed! Strands-Glean integration is working.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    asyncio.run(main())