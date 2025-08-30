#!/usr/bin/env python3
"""
Rex Crypto Trading Platform - System Startup Script
Complete system initialization with all services
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from deploy.build_deploy_framework import UnifiedFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main startup sequence"""
    print("🚀 Rex Crypto Trading Platform - System Startup")
    print("=" * 60)
    
    framework = UnifiedFramework(str(project_root))
    
    # Full system startup
    success = framework.full_system_startup()
    
    if success:
        print("\n✅ System startup completed successfully!")
        print("\n📊 Available Services:")
        print("   • Russian Crypto News Service")
        print("   • Image Enhancement (Web scraping, Charts, Search)")
        print("   • Database with full schema")
        print("   • REST API endpoints")
        print("   • SAP UI5 Frontend")
        
        print("\n🌐 Access Points:")
        print("   • Web App: http://localhost:8080")
        print("   • API: http://localhost:5000/api")
        print("   • Health Check: http://localhost:5000/api/health")
        
        print("\n🔧 Management Commands:")
        print("   • Health Check: python startup.py health")
        print("   • Build & Deploy: python startup.py build")
        print("   • Deploy to Vercel: python startup.py deploy")
    else:
        print("\n❌ System startup failed!")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        framework = UnifiedFramework(str(project_root))
        
        if command == "health":
            health_status = framework.starter.health_check()
            print("\n🏥 System Health Status:")
            for service, status in health_status.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {service.replace('_', ' ').title()}")
        elif command == "build":
            framework.build_and_deploy_new_system()
        elif command == "deploy":
            framework.deploy_to_github_and_vercel()
    else:
        asyncio.run(main())
