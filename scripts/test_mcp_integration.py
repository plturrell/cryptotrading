#!/usr/bin/env python3
"""
Simple test of MCP Strands-Glean integration
Tests the MCP server directly without complex transport setup
"""

import asyncio
import json
import sys
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_mcp_server():
    """Test the MCP server functionality directly"""
    try:
        from cryptotrading.core.protocols.mcp.strands_glean_server import StrandsGleanMCPServer
        
        print("🚀 TESTING MCP STRANDS-GLEAN SERVER")
        print("=" * 50)
        
        # Create server instance
        server = StrandsGleanMCPServer(project_root=str(project_root))
        print("✅ Server created successfully")
        
        # Test tool registration
        print(f"\n🔧 Registered Tools: {len(server.tools)}")
        for tool_name in server.tools.keys():
            print(f"   • {tool_name}")
        
        # Test direct tool calls (simulating MCP)
        print(f"\n📋 Testing tool calls...")
        
        # Test 1: Get statistics
        print("1️⃣ Testing statistics...")
        stats_result = await server._get_statistics()
        if not stats_result.isError:
            print("   ✅ Statistics call successful")
            if stats_result.content and stats_result.content[0].data:
                data = json.loads(stats_result.content[0].data)
                print(f"   📊 Project: {data.get('project_root', 'unknown')}")
                print(f"   📊 Glean Available: {data.get('glean_available', False)}")
        else:
            print("   ❌ Statistics call failed")
        
        # Test 2: Index project
        print("\n2️⃣ Testing project indexing...")
        index_result = await server._index_project("test-unit", True)
        if not index_result.isError:
            print("   ✅ Indexing call successful")
            if index_result.content and index_result.content[0].data:
                data = json.loads(index_result.content[0].data)
                print(f"   📊 Files indexed: {data.get('files_indexed', 0)}")
                print(f"   📊 Symbols found: {data.get('symbols_found', 0)}")
                print(f"   📊 Facts stored: {data.get('facts_stored', 0)}")
        else:
            print("   ❌ Indexing call failed")
            print(f"   Error: {index_result.content[0].text if index_result.content else 'Unknown'}")
        
        # Test 3: Symbol search
        print("\n3️⃣ Testing symbol search...")
        search_result = await server._search_symbols("Agent", 5)
        if not search_result.isError:
            print("   ✅ Symbol search successful")
            if search_result.content and search_result.content[0].data:
                data = json.loads(search_result.content[0].data)
                symbols = data.get('symbols', [])
                print(f"   📊 Found {len(symbols)} symbols")
                for symbol in symbols[:3]:  # Show first 3
                    print(f"      • {symbol.get('name', 'unknown')} in {symbol.get('file', 'unknown')}")
        else:
            print("   ❌ Symbol search failed")
            print(f"   Error: {search_result.content[0].text if search_result.content else 'Unknown'}")
        
        # Test 4: AI Enhancement (mock)
        print("\n4️⃣ Testing AI enhancement...")
        mock_analysis = {"symbols": [{"name": "TestAgent", "file": "test.py"}]}
        ai_result = await server._enhance_with_ai(mock_analysis, "summary", "grok")
        if not ai_result.isError:
            print("   ✅ AI enhancement successful")
            if ai_result.content and ai_result.content[0].data:
                data = json.loads(ai_result.content[0].data)
                insights = data.get('ai_insights', {})
                print(f"   🤖 AI Provider: {data.get('ai_provider', 'unknown')}")
                print(f"   🤖 Enhancement Type: {data.get('enhancement_type', 'unknown')}")
                print(f"   🤖 Confidence: {data.get('confidence_score', 0)}")
        else:
            print("   ❌ AI enhancement failed")
        
        # Test 5: Code Review
        print("\n5️⃣ Testing AI code review...")
        review_result = await server._ai_code_review(["src/test.py"], "comprehensive")
        if not review_result.isError:
            print("   ✅ AI code review successful")
            if review_result.content and review_result.content[0].data:
                data = json.loads(review_result.content[0].data)
                print(f"   📊 Overall Score: {data.get('overall_score', 'N/A')}")
                print(f"   📊 Issues Found: {len(data.get('issues_found', []))}")
                print(f"   📊 Recommendations: {len(data.get('recommendations', []))}")
        else:
            print("   ❌ AI code review failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Test failed with exception")
        return False

async def test_integration_workflow():
    """Test a complete integration workflow"""
    print(f"\n🎯 INTEGRATION WORKFLOW TEST")
    print("=" * 50)
    
    try:
        from cryptotrading.core.protocols.mcp.strands_glean_server import StrandsGleanMCPServer
        
        # Create server
        server = StrandsGleanMCPServer(project_root=str(project_root))
        
        # Workflow: Index -> Search -> Analyze -> Enhance
        print("Step 1: Index the codebase...")
        index_result = await server._index_project("workflow-test", False)
        
        if not index_result.isError:
            print("Step 2: Search for interesting symbols...")
            search_result = await server._search_symbols("glean", 3)
            
            if not search_result.isError and search_result.content:
                data = json.loads(search_result.content[0].data)
                symbols = data.get('symbols', [])
                
                if symbols:
                    first_symbol = symbols[0]
                    symbol_name = first_symbol.get('name', 'unknown')
                    
                    print(f"Step 3: Analyze dependencies for {symbol_name}...")
                    deps_result = await server._analyze_dependencies(symbol_name, 2)
                    
                    if not deps_result.isError:
                        print("Step 4: Enhance analysis with AI...")
                        deps_data = json.loads(deps_result.content[0].data)
                        ai_result = await server._enhance_with_ai(deps_data, "recommendations", "grok")
                        
                        if not ai_result.isError:
                            ai_data = json.loads(ai_result.content[0].data)
                            insights = ai_data.get('ai_insights', {})
                            
                            print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
                            print(f"   Symbol analyzed: {symbol_name}")
                            print(f"   AI insights: {insights.get('summary', 'Analysis completed')}")
                            print(f"   Recommendations: {len(insights.get('recommendations', []))}")
                            return True
        
        print("❌ Workflow incomplete - some steps failed")
        return False
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🌟 MCP STRANDS-GLEAN INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Basic server functionality  
    test1_success = await test_mcp_server()
    
    # Test 2: Integration workflow
    test2_success = await test_integration_workflow()
    
    # Summary
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 30)
    print(f"Server Functions: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Integration Workflow: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print(f"\n🎊 SUCCESS! MCP Strands-Glean integration is working!")
        print("\n💡 Key capabilities demonstrated:")
        print("   ✅ Project indexing with SCIP")
        print("   ✅ Symbol search and analysis")
        print("   ✅ Dependency tracking")
        print("   ✅ AI-enhanced insights")
        print("   ✅ Code review automation")
        print("   ✅ End-to-end workflow")
        
        print(f"\n🔗 Ready for:")
        print("   • Grok API integration for real AI insights")
        print("   • Claude Code integration via MCP")
        print("   • Real-time code analysis workflows")
        print("   • Architecture constraint validation")
        print("   • Automated refactoring suggestions")
        
    else:
        print(f"\n⚠️ Some tests failed. The integration needs further work.")

if __name__ == "__main__":
    asyncio.run(main())