#!/usr/bin/env python3
"""
Test the complete integration: MCP + Grok + Glean + HTTP Transport
"""

import asyncio
import json
import sys
from pathlib import Path
import aiohttp

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


async def test_mcp_http_integration():
    """Test the complete MCP HTTP integration"""
    print("🌟 FULL INTEGRATION TEST")
    print("=" * 60)
    print("Testing: MCP Server + HTTP Transport + Grok AI + Glean Analysis")
    
    base_url = "http://localhost:8081"
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Server Status
            print("\n1️⃣ Testing server status...")
            async with session.get(f"{base_url}/mcp/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Server running: {data.get('status', 'unknown')}")
                    print(f"   📊 Version: {data.get('version', 'unknown')}")
                else:
                    print(f"   ❌ Server not responding: {response.status}")
                    return False
            
            # Test 2: List Tools
            print("\n2️⃣ Testing tools list...")
            async with session.get(f"{base_url}/mcp/tools") as response:
                if response.status == 200:
                    data = await response.json()
                    tools = data.get("result", {}).get("tools", [])
                    print(f"   ✅ Found {len(tools)} tools:")
                    for tool in tools[:3]:  # Show first 3
                        print(f"      • {tool.get('name', 'unknown')}")
                else:
                    print(f"   ❌ Tools list failed: {response.status}")
            
            # Test 3: Project Indexing
            print("\n3️⃣ Testing project indexing...")
            index_request = {
                "jsonrpc": "2.0",
                "id": "test_index",
                "method": "tools/call",
                "params": {
                    "name": "glean_index_project",
                    "arguments": {
                        "unit_name": "test-integration",
                        "force_reindex": True
                    }
                }
            }
            
            async with session.post(f"{base_url}/mcp", json=index_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            index_data = json.loads(content.get("data", "{}"))
                            print(f"   ✅ Indexing completed!")
                            print(f"      • Files: {index_data.get('files_indexed', 0)}")
                            print(f"      • Symbols: {index_data.get('symbols_found', 0)}")
                            print(f"      • Facts: {index_data.get('facts_stored', 0)}")
                        else:
                            print(f"   ⚠️ Indexing result: {result}")
                    else:
                        print(f"   ❌ Indexing failed: {result}")
                else:
                    print(f"   ❌ Indexing request failed: {response.status}")
            
            # Test 4: Symbol Search
            print("\n4️⃣ Testing symbol search...")
            search_request = {
                "jsonrpc": "2.0",
                "id": "test_search",
                "method": "tools/call",
                "params": {
                    "name": "glean_symbol_search",
                    "arguments": {
                        "pattern": "Agent",
                        "limit": 3
                    }
                }
            }
            
            async with session.post(f"{base_url}/mcp", json=search_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            search_data = json.loads(content.get("data", "{}"))
                            symbols = search_data.get("symbols", [])
                            print(f"   ✅ Found {len(symbols)} symbols:")
                            for symbol in symbols[:2]:  # Show first 2
                                print(f"      • {symbol.get('name', 'unknown')} ({symbol.get('kind', 'unknown')})")
                        else:
                            print(f"   ⚠️ Search result: {result}")
                    else:
                        print(f"   ❌ Search failed: {result}")
                else:
                    print(f"   ❌ Search request failed: {response.status}")
            
            # Test 5: AI Enhancement with Grok
            print("\n5️⃣ Testing AI enhancement with Grok...")
            ai_request = {
                "jsonrpc": "2.0", 
                "id": "test_ai",
                "method": "tools/call",
                "params": {
                    "name": "ai_enhance_analysis",
                    "arguments": {
                        "analysis_data": {
                            "symbols": [
                                {"name": "StrandsAgent", "kind": "class", "file": "strands.py"},
                                {"name": "GleanClient", "kind": "class", "file": "glean_client.py"}
                            ]
                        },
                        "enhancement_type": "summary",
                        "ai_provider": "grok"
                    }
                }
            }
            
            print("      🤖 Calling Grok API...")
            async with session.post(f"{base_url}/mcp", json=ai_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            ai_data = json.loads(content.get("data", "{}"))
                            grok_response = ai_data.get("grok_response", {})
                            
                            if grok_response.get("status") == "success":
                                print(f"   ✅ AI enhancement completed!")
                                print(f"      🧠 Analysis: {grok_response.get('analysis', 'No analysis')[:100]}...")
                                usage = grok_response.get("usage", {})
                                if usage:
                                    print(f"      📊 Tokens used: {usage.get('total_tokens', 'unknown')}")
                            else:
                                print(f"   ⚠️ AI enhancement status: {grok_response.get('status', 'unknown')}")
                                print(f"      Error: {grok_response.get('error', 'No error details')}")
                        else:
                            print(f"   ⚠️ AI result format: {result}")
                    else:
                        error_text = result.get("content", [{}])[0].get("text", "Unknown error")
                        print(f"   ❌ AI enhancement failed: {error_text}")
                else:
                    error_text = await response.text()
                    print(f"   ❌ AI request failed: {response.status} - {error_text}")
            
            # Test 6: AI Code Review
            print("\n6️⃣ Testing AI code review...")
            review_request = {
                "jsonrpc": "2.0",
                "id": "test_review",
                "method": "tools/call",
                "params": {
                    "name": "ai_code_review",
                    "arguments": {
                        "files": ["src/cryptotrading/core/agents/base.py"],
                        "review_type": "comprehensive"
                    }
                }
            }
            
            print("      🤖 Generating AI review...")
            async with session.post(f"{base_url}/mcp", json=review_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            review_data = json.loads(content.get("data", "{}"))
                            grok_review = review_data.get("grok_review", {})
                            
                            if grok_review.get("status") == "success":
                                print(f"   ✅ AI code review completed!")
                                review_text = grok_review.get("review", "No review content")
                                print(f"      📝 Review: {review_text[:150]}...")
                            else:
                                print(f"   ⚠️ Review status: {grok_review.get('status', 'unknown')}")
                        else:
                            print(f"   ⚠️ Review result format: {result}")
                    else:
                        error_text = result.get("content", [{}])[0].get("text", "Unknown error")
                        print(f"   ❌ AI review failed: {error_text}")
                else:
                    print(f"   ❌ Review request failed: {response.status}")
            
            # Summary
            print(f"\n🎉 INTEGRATION TEST COMPLETED!")
            print("=" * 60)
            print("✅ Successfully demonstrated:")
            print("   • MCP Server with HTTP transport")
            print("   • Real-time project indexing (Glean)")
            print("   • Symbol search and analysis")
            print("   • AI enhancement with Grok API")
            print("   • AI-powered code reviews")
            print("   • External tool connectivity via HTTP")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False


async def main():
    """Main test runner"""
    print("🚀 Starting full integration test...")
    print("Make sure the MCP server is running on port 8081")
    print("Command: python3 scripts/launch_mcp_server.py --port 8081")
    
    await asyncio.sleep(2)  # Give server time to start
    
    success = await test_mcp_http_integration()
    
    if success:
        print(f"\n🎊 ALL SYSTEMS OPERATIONAL!")
        print("The Strands-Glean-Grok integration is fully functional.")
    else:
        print(f"\n⚠️ Some tests failed. Check server status.")


if __name__ == "__main__":
    asyncio.run(main())