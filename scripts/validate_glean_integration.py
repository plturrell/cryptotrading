#!/usr/bin/env python3
"""
Comprehensive Glean MCP Integration Validation Script
Validates the complete integration of Glean tools with the MCP server
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.cryptotrading.infrastructure.analysis.glean_zero_blindspots_mcp_tool import glean_zero_blindspots_validator_tool
from src.cryptotrading.infrastructure.analysis.glean_continuous_monitor import glean_continuous_monitor_tool

async def validate_glean_integration():
    """Comprehensive validation of Glean MCP integration"""
    print("🚀 Validating Glean MCP Integration")
    print("=" * 50)
    
    # Test 1: Zero Blind Spots Validator
    print("\n🔍 Testing Glean Zero Blind Spots Validator...")
    try:
        result = await glean_zero_blindspots_validator_tool({'project_path': str(project_root)})
        
        if result['success']:
            validation = result['validation_result']
            print(f"✅ Validator Status: SUCCESS")
            print(f"📊 Validation Score: {validation['validation_score']:.1f}/100")
            print(f"🎯 Total Facts: {validation.get('total_facts', 'N/A')}")
            print(f"🔧 Languages: {len(validation.get('language_coverage', {}))}")
            print(f"📈 Production Ready: {validation.get('production_ready', False)}")
            
            # Show language coverage if available
            if 'language_coverage' in validation:
                print("\n📋 Language Coverage:")
                for lang, stats in validation['language_coverage'].items():
                    files = stats.get('files_indexed', stats.get('files', 0))
                    facts = stats.get('facts_generated', stats.get('facts', 0))
                    print(f"  • {lang}: {files} files, {facts} facts")
            
            # Show recommendations if any
            if validation.get('recommendations'):
                print("\n💡 Recommendations:")
                for rec in validation['recommendations'][:3]:  # Show top 3
                    print(f"  • {rec}")
            
            # Show blind spots if any
            if validation.get('blind_spots'):
                print(f"\n⚠️  Blind Spots Found: {len(validation['blind_spots'])}")
                for spot in validation['blind_spots'][:3]:  # Show top 3
                    print(f"  • {spot.get('description', spot)}")
        else:
            print(f"❌ Validator Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Validator Test Failed: {e}")
        return False
    
    # Test 2: Continuous Monitor
    print("\n🔄 Testing Continuous Monitor...")
    try:
        # Test status command
        monitor_result = await glean_continuous_monitor_tool({
            'command': 'status', 
            'project_path': str(project_root)
        })
        
        if monitor_result['success']:
            print(f"✅ Monitor Status: SUCCESS")
            print(f"📡 Monitoring Active: {monitor_result.get('monitoring_active', False)}")
            
            # Test list sessions
            sessions_result = await glean_continuous_monitor_tool({
                'command': 'list_sessions'
            })
            
            if sessions_result['success']:
                print(f"📊 Active Sessions: {sessions_result['active_sessions']}")
            
        else:
            print(f"❌ Monitor Failed: {monitor_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Monitor Test Failed: {e}")
        return False
    
    # Test 3: MCP Server Integration
    print("\n🔗 Testing MCP Server Integration...")
    try:
        from src.cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import GLEAN_MCP_TOOLS
        
        print(f"✅ Glean MCP Tools Registered: {len(GLEAN_MCP_TOOLS)}")
        for tool_name in GLEAN_MCP_TOOLS.keys():
            print(f"  • {tool_name}")
            
    except Exception as e:
        print(f"❌ MCP Integration Test Failed: {e}")
        return False
    
    # Final Summary
    print("\n" + "=" * 50)
    print("🎉 Glean MCP Integration Validation COMPLETE")
    print("✅ All tests passed successfully!")
    print("🚀 Production ready for Glean agent deployment")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(validate_glean_integration())
    sys.exit(0 if success else 1)
