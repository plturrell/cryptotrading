#!/usr/bin/env python3
"""
Final verification of Glean agent imports
Tests that all imports work correctly in the actual system
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import(module_name, import_statement):
    """Test a specific import"""
    try:
        exec(import_statement)
        print(f"✓ {module_name}")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        return False

def verify_all_imports():
    """Verify all Glean agent imports work correctly"""
    print("🔧 VERIFYING GLEAN AGENT IMPORTS")
    print("=" * 50)
    
    imports_to_test = [
        # Data schemas
        ("glean_data_schemas", 
         "from cryptotrading.infrastructure.analysis.glean_data_schemas import ALL_DATA_TRACKING_SCHEMAS, DataInputFact"),
        
        # SCIP data flow indexer  
        ("scip_data_flow_indexer",
         "from cryptotrading.infrastructure.analysis.scip_data_flow_indexer import DataFlowSCIPIndexer, DataFlowVisitor"),
        
        # Runtime collectors
        ("glean_runtime_collectors",
         "from cryptotrading.infrastructure.analysis.glean_runtime_collectors import track_data_input, track_factor"),
        
        # Angle queries
        ("crypto_angle_queries", 
         "from cryptotrading.infrastructure.analysis.crypto_angle_queries import create_crypto_query, CRYPTO_ANGLE_QUERIES"),
        
        # Extended agent capabilities
        ("strands_glean_agent",
         "from cryptotrading.core.agents.specialized.strands_glean_agent import DataFlowAnalysisCapability"),
    ]
    
    results = []
    for name, import_stmt in imports_to_test:
        success = test_import(name, import_stmt)
        results.append((name, success))
    
    print(f"\n{'='*50}")
    print("TESTING CORE FUNCTIONALITY")
    print("=" * 50)
    
    # Test core functionality
    try:
        # Test schema creation
        from cryptotrading.infrastructure.analysis.glean_data_schemas import DataInputFact
        fact = DataInputFact(
            function="test", file="test.py", line=1, input_id="test",
            data_type="test", source="test"
        )
        fact_dict = fact.to_fact()
        assert fact_dict["predicate"] == "crypto.DataInput"
        print("✓ Schema fact creation works")
    except Exception as e:
        print(f"✗ Schema fact creation failed: {e}")
        results.append(("schema_functionality", False))
    
    try:
        # Test query creation
        from cryptotrading.infrastructure.analysis.crypto_angle_queries import create_crypto_query
        query = create_crypto_query("data_inputs_by_symbol", {"symbol": "BTC-USD"})
        assert "crypto.DataInput" in query
        assert "BTC-USD" in query
        print("✓ Angle query creation works")
    except Exception as e:
        print(f"✗ Angle query creation failed: {e}")
        results.append(("query_functionality", False))
    
    try:
        # Test visitor
        from cryptotrading.infrastructure.analysis.scip_data_flow_indexer import DataFlowVisitor
        import ast
        code = "import yfinance as yf"
        tree = ast.parse(code)
        visitor = DataFlowVisitor("test.py")
        visitor.visit(tree)
        print("✓ Data flow visitor works")
    except Exception as e:
        print(f"✗ Data flow visitor failed: {e}")
        results.append(("visitor_functionality", False))
    
    # Summary
    print(f"\n{'='*50}")
    print("IMPORT VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print(f"\nOverall: {passed}/{total} components working")
    
    if passed == total:
        print("\n🎉 ALL GLEAN AGENT IMPORTS VERIFIED!")
        print("✓ No missing imports")
        print("✓ No import errors") 
        print("✓ All functionality accessible")
    else:
        print(f"\n⚠️  {total - passed} components have import issues")
    
    return passed == total

if __name__ == "__main__":
    success = verify_all_imports()
    sys.exit(0 if success else 1)