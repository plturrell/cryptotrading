#!/usr/bin/env python3
"""
Simple Dashboard Test
Quick test to check dashboard functionality
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


async def test_simple_dashboard():
    """Test basic dashboard functionality"""
    print("🧪 SIMPLE DASHBOARD TEST")
    print("=" * 40)
    
    try:
        from cryptotrading.core.dashboard.realtime_dashboard import create_dashboard
        
        print("✅ Dashboard module imported successfully")
        
        # Create dashboard
        dashboard = await create_dashboard("http://localhost:8082")
        print("✅ Dashboard instance created")
        
        # Try to start server
        if await dashboard.start_server("localhost", 8091):
            print("✅ Dashboard server started on port 8091")
            print("🌐 Dashboard URL: http://localhost:8091")
            
            # Keep running for a short test
            print("⏳ Running for 10 seconds for testing...")
            await asyncio.sleep(10)
            
            await dashboard.stop_server()
            print("✅ Dashboard stopped successfully")
            return True
        else:
            print("❌ Failed to start dashboard server")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test"""
    print("🚀 Testing Dashboard Components")
    success = await test_simple_dashboard()
    
    if success:
        print("\n🎉 Dashboard test completed successfully!")
        print("The dashboard is working and can be started.")
    else:
        print("\n❌ Dashboard test failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test stopped")
    except Exception as e:
        print(f"❌ Test error: {e}")
        sys.exit(1)