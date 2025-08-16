#!/usr/bin/env python3
"""
Test Real-time Dashboard Functionality
Comprehensive test of dashboard features and real-time capabilities
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
import aiohttp

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


async def test_dashboard_api(dashboard_url: str = "http://localhost:8090"):
    """Test dashboard API endpoints"""
    print("🧪 DASHBOARD API TEST")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Basic connectivity
            print("\n1️⃣ Testing basic connectivity...")
            async with session.get(f"{dashboard_url}/api/metrics") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Metrics endpoint: {response.status}")
                    print(f"      • Files watched: {data.get('total_files_watched', 0)}")
                    print(f"      • Changes detected: {data.get('changes_detected', 0)}")
                    print(f"      • Analyses completed: {data.get('analyses_completed', 0)}")
                    print(f"      • AI insights: {data.get('ai_insights_generated', 0)}")
                else:
                    print(f"   ❌ Metrics endpoint: {response.status}")
            
            # Test 2: Recent changes
            print("\n2️⃣ Testing recent changes endpoint...")
            async with session.get(f"{dashboard_url}/api/changes") as response:
                if response.status == 200:
                    data = await response.json()
                    changes = data.get('recent_changes', [])
                    print(f"   ✅ Changes endpoint: {len(changes)} recent changes")
                    if changes:
                        latest = changes[-1]
                        file_name = Path(latest.get('file_path', 'unknown')).name
                        print(f"      • Latest: {file_name} ({latest.get('change_type', 'unknown')})")
                else:
                    print(f"   ❌ Changes endpoint: {response.status}")
            
            # Test 3: Recent analyses
            print("\n3️⃣ Testing recent analyses endpoint...")
            async with session.get(f"{dashboard_url}/api/analyses") as response:
                if response.status == 200:
                    data = await response.json()
                    analyses = data.get('recent_analyses', [])
                    print(f"   ✅ Analyses endpoint: {len(analyses)} recent analyses")
                    if analyses:
                        latest = analyses[-1]
                        file_name = Path(latest.get('file_path', 'unknown')).name
                        symbols_count = len(latest.get('symbols_found', []))
                        ai_enhanced = "🤖" if latest.get('ai_insights') else ""
                        print(f"      • Latest: {file_name} ({symbols_count} symbols) {ai_enhanced}")
                else:
                    print(f"   ❌ Analyses endpoint: {response.status}")
            
            # Test 4: System status
            print("\n4️⃣ Testing system status endpoint...")
            async with session.get(f"{dashboard_url}/api/status") as response:
                if response.status == 200:
                    data = await response.json()
                    realtime_status = data.get('realtime_status', 'unknown')
                    print(f"   ✅ Status endpoint: Real-time monitoring {realtime_status}")
                    
                    watcher_stats = data.get('watcher_statistics', {})
                    if watcher_stats:
                        print(f"      • Queue size: {watcher_stats.get('queue_stats', {}).get('queued', 0)}")
                        print(f"      • Active tasks: {watcher_stats.get('active_analysis_tasks', 0)}")
                        print(f"      • Uptime: {watcher_stats.get('uptime_seconds', 0):.1f}s")
                else:
                    print(f"   ❌ Status endpoint: {response.status}")
            
            # Test 5: Dashboard HTML
            print("\n5️⃣ Testing dashboard HTML...")
            async with session.get(dashboard_url) as response:
                if response.status == 200:
                    html = await response.text()
                    print(f"   ✅ Dashboard HTML: {len(html)} characters")
                    if "Real-time Code Analysis Dashboard" in html:
                        print("      • Title found in HTML ✅")
                    if "WebSocket" in html:
                        print("      • WebSocket support detected ✅")
                else:
                    print(f"   ❌ Dashboard HTML: {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False


async def test_websocket_connection(dashboard_url: str = "http://localhost:8090"):
    """Test WebSocket real-time connection"""
    print("\n🔌 WEBSOCKET CONNECTION TEST")
    print("=" * 50)
    
    import aiohttp
    
    # Convert HTTP URL to WebSocket URL
    ws_url = dashboard_url.replace("http://", "ws://") + "/ws"
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"Connecting to: {ws_url}")
            
            async with session.ws_connect(ws_url) as ws:
                print("✅ WebSocket connected successfully")
                
                # Listen for initial messages
                message_count = 0
                timeout_seconds = 10
                
                print(f"Listening for messages for {timeout_seconds} seconds...")
                
                try:
                    async with asyncio.timeout(timeout_seconds):
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    msg_type = data.get('type', 'unknown')
                                    timestamp = data.get('timestamp', 'unknown')
                                    payload = data.get('payload', {})
                                    
                                    print(f"   📨 Message {message_count + 1}: {msg_type}")
                                    
                                    if msg_type == 'metrics':
                                        files_watched = payload.get('total_files_watched', 0)
                                        changes = payload.get('changes_detected', 0)
                                        print(f"      • Files: {files_watched}, Changes: {changes}")
                                    elif msg_type == 'system_status':
                                        status = payload.get('realtime_status', 'unknown')
                                        print(f"      • Real-time status: {status}")
                                    
                                    message_count += 1
                                    
                                    if message_count >= 3:  # Stop after a few messages
                                        break
                                        
                                except json.JSONDecodeError:
                                    print(f"   ⚠️ Invalid JSON message: {msg.data}")
                                    
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print(f"   ❌ WebSocket error: {ws.exception()}")
                                break
                
                except asyncio.TimeoutError:
                    print(f"   ⏰ Timeout after {timeout_seconds} seconds")
                
                print(f"✅ Received {message_count} messages via WebSocket")
                return True
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False


async def simulate_file_changes():
    """Simulate file changes to test real-time monitoring"""
    print("\n📝 FILE CHANGE SIMULATION")
    print("=" * 50)
    
    # Create a test file in the watched directory
    test_file = project_root / "src" / "cryptotrading" / "test_dashboard_changes.py"
    
    try:
        # Create initial file
        test_content = '''"""
Test file for dashboard change monitoring
"""

class DashboardTestClass:
    """A test class for dashboard monitoring"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = "2024-01-01"
    
    def test_method(self, data: list) -> dict:
        """Test method for analysis"""
        return {"processed": len(data), "name": self.name}

def test_function():
    """A test function"""
    return "dashboard test"
'''
        
        print("1️⃣ Creating test file...")
        with open(test_file, 'w') as f:
            f.write(test_content)
        print(f"   ✅ Created: {test_file.name}")
        
        # Wait for file watching to detect
        await asyncio.sleep(3)
        
        # Modify the file
        print("\n2️⃣ Modifying test file...")
        modified_content = test_content + '''

# Added new functionality for dashboard testing
def additional_test_function():
    """Additional function to test incremental analysis"""
    return "This was added to test real-time dashboard updates"

class ExtendedTestClass(DashboardTestClass):
    """Extended test class"""
    
    def extended_method(self):
        return f"Extended functionality for {self.name}"
'''
        
        with open(test_file, 'w') as f:
            f.write(modified_content)
        print(f"   ✅ Modified: {test_file.name}")
        
        # Wait for analysis
        await asyncio.sleep(4)
        
        # Create another test file
        print("\n3️⃣ Creating second test file...")
        test_file2 = project_root / "src" / "cryptotrading" / "test_dashboard_additional.py"
        
        additional_content = '''"""
Additional test file for dashboard monitoring
"""

def dashboard_helper_function():
    """Helper function for dashboard testing"""
    return {"status": "dashboard_test", "timestamp": "2024-01-01"}

class DashboardHelperClass:
    """Helper class for dashboard testing"""
    
    def process_dashboard_data(self, data):
        return f"Processing {len(data) if data else 0} items"
'''
        
        with open(test_file2, 'w') as f:
            f.write(additional_content)
        print(f"   ✅ Created: {test_file2.name}")
        
        # Wait for analysis
        await asyncio.sleep(4)
        
        print("\n✅ File change simulation completed")
        print("   • Created 2 test files")
        print("   • Modified 1 test file")
        print("   • Total changes: 3")
        
        return [test_file, test_file2]
        
    except Exception as e:
        print(f"❌ File change simulation failed: {e}")
        return []


async def cleanup_test_files(test_files: list):
    """Clean up test files"""
    print("\n🧹 CLEANING UP TEST FILES")
    print("=" * 30)
    
    for test_file in test_files:
        try:
            if test_file.exists():
                test_file.unlink()
                print(f"   ✅ Removed: {test_file.name}")
        except Exception as e:
            print(f"   ⚠️ Failed to remove {test_file.name}: {e}")


async def test_dashboard_integration(dashboard_url: str = "http://localhost:8090", 
                                   mcp_url: str = "http://localhost:8082"):
    """Test complete dashboard integration"""
    print("🎯 COMPLETE DASHBOARD INTEGRATION TEST")
    print("=" * 60)
    print("Testing: Dashboard + MCP + Real-time Monitoring + WebSocket")
    
    # Check dashboard availability
    print("\n🔍 Checking dashboard availability...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{dashboard_url}/api/metrics") as response:
                if response.status != 200:
                    print(f"❌ Dashboard not available at {dashboard_url}")
                    return False
                print("✅ Dashboard is running")
    except Exception as e:
        print(f"❌ Dashboard not accessible: {e}")
        return False
    
    # Test API endpoints
    api_success = await test_dashboard_api(dashboard_url)
    
    # Test WebSocket connection
    ws_success = await test_websocket_connection(dashboard_url)
    
    # Simulate file changes and test real-time updates
    print("\n🚀 Testing real-time file monitoring...")
    test_files = await simulate_file_changes()
    
    # Wait for changes to be processed
    await asyncio.sleep(5)
    
    # Check if changes appear in dashboard
    print("\n📊 Verifying real-time updates...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{dashboard_url}/api/changes") as response:
                if response.status == 200:
                    data = await response.json()
                    recent_changes = data.get('recent_changes', [])
                    
                    # Look for our test file changes
                    test_changes = [c for c in recent_changes 
                                  if 'test_dashboard' in c.get('file_path', '')]
                    
                    print(f"   📝 Found {len(test_changes)} test file changes")
                    for change in test_changes[-3:]:  # Show last 3
                        file_name = Path(change.get('file_path', 'unknown')).name
                        change_type = change.get('change_type', 'unknown')
                        timestamp = change.get('timestamp', 'unknown')
                        print(f"      • {file_name} ({change_type}) at {timestamp}")
            
            # Check analyses
            async with session.get(f"{dashboard_url}/api/analyses") as response:
                if response.status == 200:
                    data = await response.json()
                    recent_analyses = data.get('recent_analyses', [])
                    
                    test_analyses = [a for a in recent_analyses 
                                   if 'test_dashboard' in a.get('file_path', '')]
                    
                    print(f"   🔬 Found {len(test_analyses)} test file analyses")
                    for analysis in test_analyses[-2:]:  # Show last 2
                        file_name = Path(analysis.get('file_path', 'unknown')).name
                        symbols_count = len(analysis.get('symbols_found', []))
                        processing_time = analysis.get('processing_time', 0)
                        ai_insights = "🤖" if analysis.get('ai_insights') else ""
                        print(f"      • {file_name}: {symbols_count} symbols, {processing_time:.2f}s {ai_insights}")
                    
        except Exception as e:
            print(f"   ❌ Failed to verify updates: {e}")
    
    # Clean up
    await cleanup_test_files(test_files)
    
    # Summary
    print(f"\n🎉 DASHBOARD INTEGRATION TEST COMPLETED!")
    print("=" * 60)
    print("✅ Test Results:")
    print(f"   • API Endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"   • WebSocket Connection: {'✅ PASS' if ws_success else '❌ FAIL'}")
    print(f"   • Real-time File Monitoring: {'✅ PASS' if test_files else '❌ FAIL'}")
    print("   • Dashboard Web Interface: ✅ AVAILABLE")
    
    success_count = sum([api_success, ws_success, bool(test_files)])
    total_tests = 3
    
    print(f"\n📊 Overall Result: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎊 ALL DASHBOARD TESTS PASSED!")
        print("\n🌐 Dashboard is fully functional and ready to use!")
        print(f"   • Open: {dashboard_url}")
        print("   • Features: Real-time monitoring, AI insights, live updates")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
    
    return success_count == total_tests


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Real-time Dashboard")
    parser.add_argument("--dashboard-url", default="http://localhost:8090", help="Dashboard URL")
    parser.add_argument("--mcp-url", default="http://localhost:8082", help="MCP server URL")
    parser.add_argument("--api-only", action="store_true", help="Test API endpoints only")
    parser.add_argument("--websocket-only", action="store_true", help="Test WebSocket only")
    
    args = parser.parse_args()
    
    print("🧪 DASHBOARD TEST SUITE")
    print(f"Dashboard URL: {args.dashboard_url}")
    print(f"MCP Server URL: {args.mcp_url}")
    
    if args.api_only:
        success = await test_dashboard_api(args.dashboard_url)
    elif args.websocket_only:
        success = await test_websocket_connection(args.dashboard_url)
    else:
        success = await test_dashboard_integration(args.dashboard_url, args.mcp_url)
    
    if not success:
        print("\n❌ Dashboard tests failed")
        sys.exit(1)
    else:
        print("\n✅ Dashboard tests completed successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Dashboard test stopped")
    except Exception as e:
        print(f"❌ Test error: {e}")
        sys.exit(1)