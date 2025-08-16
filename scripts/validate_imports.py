#!/usr/bin/env python3
"""
Import Validation Test
Tests that all critical imports work with the new src/cryptotrading/ structure.
"""

import sys
import traceback
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import(module_name, description):
    """Test importing a module and report results"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: {module_name} - {e}")
        return False

def main():
    """Run import validation tests"""
    print("🔍 Validating imports for new src/cryptotrading/ structure...\n")
    
    tests = [
        # Core structure
        ("cryptotrading", "Main package"),
        ("cryptotrading.core", "Core package"),
        ("cryptotrading.core.agents", "Agents package"),
        ("cryptotrading.core.agents.base", "Base agent"),
        ("cryptotrading.core.agents.memory", "Memory agent"),
        ("cryptotrading.core.agents.strands", "Strands agent"),
        
        # Data layer
        ("cryptotrading.data", "Data package"),
        ("cryptotrading.data.database", "Database package"),
        ("cryptotrading.data.storage", "Storage package"),
        ("cryptotrading.data.historical", "Historical data"),
        
        # Protocols
        ("cryptotrading.core.protocols", "Protocols package"),
        ("cryptotrading.core.protocols.mcp", "MCP protocol"),
        ("cryptotrading.core.protocols.a2a", "A2A protocol"),
        
        # Infrastructure
        ("cryptotrading.infrastructure", "Infrastructure package"),
        ("cryptotrading.infrastructure.logging", "Logging"),
        ("cryptotrading.infrastructure.monitoring", "Monitoring"),
        ("cryptotrading.infrastructure.security", "Security"),
        ("cryptotrading.infrastructure.registry", "Registry"),
        
        # ML and AI
        ("cryptotrading.core.ml", "ML package"),
        ("cryptotrading.core.ai", "AI package"),
        
        # Utils
        ("cryptotrading.utils", "Utils package"),
    ]
    
    passed = 0
    total = len(tests)
    
    for module_name, description in tests:
        if test_import(module_name, description):
            passed += 1
    
    print(f"\n📊 Import Validation Results:")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print(f"\n🎉 All imports working perfectly with new structure!")
        return 0
    else:
        print(f"\n⚠️  Some imports need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
