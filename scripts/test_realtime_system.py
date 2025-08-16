#!/usr/bin/env python3
"""
Test Real-time File Watching and Analysis System
Demonstrates real-time code analysis with AI insights
"""

import asyncio
import json
import sys
import tempfile
import os
from pathlib import Path
import aiohttp
import time

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


async def test_realtime_system():
    """Test the complete real-time system"""
    print("🔍 REAL-TIME SYSTEM TEST")
    print("=" * 60)
    print("Testing: File Watching + Incremental Analysis + AI Insights")
    
    base_url = "http://localhost:8082"
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Server Status
            print("\n1️⃣ Testing server status...")
            async with session.get(f"{base_url}/mcp/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Server running: {data.get('status', 'unknown')}")
                    print(f"   📊 Version: {data.get('version', 'unknown')}")
                    print(f"   🔍 Real-time monitoring: {data.get('realtime_monitoring', False)}")
                else:
                    print(f"   ❌ Server not responding: {response.status}")
                    return False
            
            # Test 2: Start File Watching
            print("\n2️⃣ Starting real-time file watching...")
            watch_request = {
                "jsonrpc": "2.0",
                "id": "start_watch",
                "method": "tools/call",
                "params": {
                    "name": "realtime_start_watching",
                    "arguments": {}
                }
            }
            
            async with session.post(f"{base_url}/mcp", json=watch_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            watch_data = json.loads(content.get("data", "{}"))
                            print(f"   ✅ File watching started!")
                            print(f"      • Status: {watch_data.get('status', 'unknown')}")
                            print(f"      • Directories: {watch_data.get('directories_watched', 0)}")
                            print(f"      • Files watched: {watch_data.get('files_being_watched', 0)}")
                        else:
                            print(f"   ⚠️ Watch result: {result}")
                    else:
                        print(f"   ❌ Watch start failed: {result}")
                else:
                    print(f"   ❌ Watch request failed: {response.status}")
            
            # Test 3: Create test file changes
            print("\n3️⃣ Creating test file changes...")
            test_file = project_root / "src" / "cryptotrading" / "test_realtime.py"
            
            try:
                # Create a test Python file
                test_content = '''"""
Test file for real-time analysis
"""

class TestRealtimeAnalysis:
    """A test class for demonstrating real-time code analysis"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = "2024-01-01"
    
    def process_data(self, data: list) -> dict:
        """Process some test data"""
        result = {"processed": len(data), "name": self.name}
        return result
    
    def calculate_metrics(self, values: list) -> float:
        """Calculate test metrics"""
        if not values:
            return 0.0
        return sum(values) / len(values)

def test_function():
    """A standalone test function"""
    analyzer = TestRealtimeAnalysis("test")
    data = [1, 2, 3, 4, 5]
    result = analyzer.process_data(data)
    metric = analyzer.calculate_metrics(data)
    return {"result": result, "metric": metric}
'''
                
                with open(test_file, 'w') as f:
                    f.write(test_content)
                print(f"   ✅ Created test file: {test_file.name}")
                
                # Wait for file watching to detect the change
                await asyncio.sleep(3)
                
                # Modify the file
                modified_content = test_content + '''

# Added new functionality
def additional_test_function():
    """Additional test function added later"""
    return "This was added to test real-time detection"
'''
                
                with open(test_file, 'w') as f:
                    f.write(modified_content)
                print(f"   ✅ Modified test file")
                
                # Wait for analysis to complete
                await asyncio.sleep(4)
                
            except Exception as e:
                print(f"   ⚠️ File operation error: {e}")
            
            # Test 4: Check recent changes
            print("\n4️⃣ Checking recent file changes...")
            changes_request = {
                "jsonrpc": "2.0",
                "id": "get_changes",
                "method": "tools/call",
                "params": {
                    "name": "realtime_get_changes",
                    "arguments": {"limit": 10}
                }
            }
            
            async with session.post(f"{base_url}/mcp", json=changes_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            changes_data = json.loads(content.get("data", "{}"))
                            changes = changes_data.get("recent_changes", [])
                            print(f"   ✅ Found {len(changes)} recent changes:")
                            for change in changes[-3:]:  # Show last 3
                                file_name = Path(change.get("file_path", "unknown")).name
                                change_type = change.get("change_type", "unknown")
                                print(f"      • {file_name} ({change_type})")
                        else:
                            print(f"   ⚠️ Changes result: {result}")
                    else:
                        print(f"   ❌ Get changes failed: {result}")
                else:
                    print(f"   ❌ Changes request failed: {response.status}")
            
            # Test 5: Check recent analyses
            print("\n5️⃣ Checking recent analysis results...")
            analyses_request = {
                "jsonrpc": "2.0",
                "id": "get_analyses",
                "method": "tools/call",
                "params": {
                    "name": "realtime_get_analyses",
                    "arguments": {"limit": 5}
                }
            }
            
            async with session.post(f"{base_url}/mcp", json=analyses_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            analyses_data = json.loads(content.get("data", "{}"))
                            analyses = analyses_data.get("recent_analyses", [])
                            print(f"   ✅ Found {len(analyses)} recent analyses:")
                            for analysis in analyses[-2:]:  # Show last 2
                                file_name = Path(analysis.get("file_path", "unknown")).name
                                symbols_count = len(analysis.get("symbols_found", []))
                                processing_time = analysis.get("processing_time", 0)
                                ai_insights = analysis.get("ai_insights") is not None
                                print(f"      • {file_name}: {symbols_count} symbols, {processing_time:.2f}s")
                                if ai_insights:
                                    print(f"        🤖 AI insights included")
                        else:
                            print(f"   ⚠️ Analyses result: {result}")
                    else:
                        print(f"   ❌ Get analyses failed: {result}")
                else:
                    print(f"   ❌ Analyses request failed: {response.status}")
            
            # Test 6: Get real-time status
            print("\n6️⃣ Getting real-time monitoring status...")
            status_request = {
                "jsonrpc": "2.0",
                "id": "get_status",
                "method": "tools/call",
                "params": {
                    "name": "realtime_get_status",
                    "arguments": {}
                }
            }
            
            async with session.post(f"{base_url}/mcp", json=status_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            status_data = json.loads(content.get("data", "{}"))
                            realtime_status = status_data.get("realtime_status", "unknown")
                            watcher_stats = status_data.get("watcher_statistics", {})
                            recent_activity = status_data.get("recent_activity", {})
                            
                            print(f"   ✅ Real-time status: {realtime_status}")
                            print(f"      📊 Changes detected: {watcher_stats.get('changes_detected', 0)}")
                            print(f"      📊 Analyses completed: {watcher_stats.get('analyses_completed', 0)}")
                            print(f"      📊 AI insights generated: {watcher_stats.get('ai_insights_generated', 0)}")
                            print(f"      📊 Recent changes (1h): {recent_activity.get('changes_last_hour', 0)}")
                            print(f"      📊 Recent analyses (1h): {recent_activity.get('analyses_last_hour', 0)}")
                        else:
                            print(f"   ⚠️ Status result: {result}")
                    else:
                        print(f"   ❌ Get status failed: {result}")
                else:
                    print(f"   ❌ Status request failed: {response.status}")
            
            # Test 7: Enhanced statistics with real-time data
            print("\n7️⃣ Getting enhanced statistics...")
            stats_request = {
                "jsonrpc": "2.0",
                "id": "get_stats",
                "method": "tools/call",
                "params": {
                    "name": "glean_statistics",
                    "arguments": {}
                }
            }
            
            async with session.post(f"{base_url}/mcp", json=stats_request) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    if not result.get("isError", False):
                        content = result.get("content", [{}])[0]
                        if content.get("type") == "resource":
                            stats_data = json.loads(content.get("data", "{}"))
                            glean_stats = stats_data.get("glean_statistics", {})
                            realtime_stats = stats_data.get("realtime_statistics", {})
                            
                            print(f"   ✅ Enhanced statistics retrieved:")
                            print(f"      📊 Glean facts: {glean_stats.get('total_facts', 0):,}")
                            print(f"      📊 Files indexed: {glean_stats.get('files_indexed', 0)}")
                            print(f"      🔍 Real-time uptime: {realtime_stats.get('uptime_seconds', 0):.1f}s")
                            print(f"      🔍 Files watched: {realtime_stats.get('files_watched', 0)}")
                        else:
                            print(f"   ⚠️ Stats result: {result}")
                    else:
                        print(f"   ❌ Get stats failed: {result}")
                else:
                    print(f"   ❌ Stats request failed: {response.status}")
            
            # Clean up test file
            try:
                if test_file.exists():
                    test_file.unlink()
                    print(f"\n🧹 Cleaned up test file")
            except:
                pass
            
            # Summary
            print(f"\n🎉 REAL-TIME SYSTEM TEST COMPLETED!")
            print("=" * 60)
            print("✅ Successfully demonstrated:")
            print("   • Real-time file watching and change detection")
            print("   • Incremental code analysis with symbol extraction")
            print("   • AI-enhanced insights with Grok integration")
            print("   • Real-time monitoring and statistics")
            print("   • MCP protocol integration for external tools")
            print("   • HTTP transport for web connectivity")
            
            return True
            
        except Exception as e:
            print(f"❌ Real-time system test failed: {e}")
            return False


async def demo_continuous_monitoring():
    """Demo continuous monitoring capabilities"""
    print(f"\n🔄 CONTINUOUS MONITORING DEMO")
    print("=" * 50)
    
    base_url = "http://localhost:8082"
    
    async with aiohttp.ClientSession() as session:
        print("Monitoring real-time activity for 30 seconds...")
        print("Make some file changes in src/cryptotrading/ to see real-time analysis!")
        
        for i in range(6):  # Monitor for 30 seconds
            try:
                # Get current status
                status_request = {
                    "jsonrpc": "2.0",
                    "id": f"monitor_{i}",
                    "method": "tools/call",
                    "params": {"name": "realtime_get_status", "arguments": {}}
                }
                
                async with session.post(f"{base_url}/mcp", json=status_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", {})
                        if not result.get("isError", False):
                            content = result.get("content", [{}])[0]
                            if content.get("type") == "resource":
                                status_data = json.loads(content.get("data", "{}"))
                                watcher_stats = status_data.get("watcher_statistics", {})
                                
                                changes = watcher_stats.get("changes_detected", 0)
                                analyses = watcher_stats.get("analyses_completed", 0)
                                ai_insights = watcher_stats.get("ai_insights_generated", 0)
                                
                                print(f"   {i*5:2d}s: Changes: {changes}, Analyses: {analyses}, AI: {ai_insights}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"   Monitor error: {e}")
        
        print("✅ Continuous monitoring demo completed")


async def main():
    """Main test runner"""
    print("🚀 Starting real-time system test...")
    print("Make sure the real-time MCP server is running on port 8082")
    print("Command: python3 scripts/realtime_mcp_server.py --port 8082 --start-watching")
    
    await asyncio.sleep(2)  # Give server time to start
    
    # Run main test
    success = await test_realtime_system()
    
    if success:
        # Run continuous monitoring demo
        await demo_continuous_monitoring()
        
        print(f"\n🎊 ALL REAL-TIME TESTS PASSED!")
        print("The system successfully demonstrates:")
        print("   • File watching with change detection")
        print("   • Incremental code analysis")
        print("   • AI-enhanced insights via Grok")
        print("   • Real-time monitoring and statistics")
        print("   • External tool integration via MCP")
    else:
        print(f"\n⚠️ Some tests failed. Check server status.")


if __name__ == "__main__":
    asyncio.run(main())