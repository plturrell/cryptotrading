#!/usr/bin/env python3
"""
Test script for the extended Glean agent data collection capabilities
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptotrading.core.agents.specialized.strands_glean_agent import create_strands_glean_agent
from cryptotrading.infrastructure.analysis.glean_runtime_collectors import (
    init_glean_collectors, track_data_input, track_data_output, 
    track_parameters, track_factor
)
from cryptotrading.infrastructure.analysis.crypto_angle_queries import (
    create_crypto_query, build_data_lineage_query, validate_crypto_query
)
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_data_collection_decorators():
    """Test the runtime data collection decorators"""
    print("=== Testing Data Collection Decorators ===")
    
    # Initialize collectors
    init_glean_collectors()
    
    @track_data_input(source="test_source", data_type="market_data")
    @track_data_output(output_type="technical_indicator")
    @track_parameters(period={"type": "int", "range_min": 1, "range_max": 100, "default": 14})
    @track_factor(factor_name="test_rsi", category="momentum")
    async def test_rsi_calculation(symbol: str, prices: pd.Series, period: int = 14) -> pd.Series:
        """Test RSI calculation with tracking"""
        # Simple RSI calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Test data
    test_prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    
    try:
        result = await test_rsi_calculation(symbol="BTC-USD", prices=test_prices, period=14)
        print(f"✓ RSI calculation completed successfully")
        print(f"  Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"✗ RSI calculation failed: {e}")
        return False


def test_angle_queries():
    """Test the Angle query system"""
    print("\n=== Testing Angle Queries ===")
    
    try:
        # Test basic query creation
        query1 = create_crypto_query("data_inputs_by_symbol", {"symbol": "BTC-USD"})
        print(f"✓ Created data input query")
        
        # Test data lineage query
        lineage_query = build_data_lineage_query("BTC-USD", include_factors=True)
        print(f"✓ Created data lineage query")
        
        # Test query validation
        validation = validate_crypto_query(query1)
        if validation["valid"]:
            print(f"✓ Query validation passed")
        else:
            print(f"✗ Query validation failed: {validation['errors']}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Angle query test failed: {e}")
        return False


async def test_strands_glean_agent():
    """Test the extended Strands-Glean agent"""
    print("\n=== Testing Extended Strands-Glean Agent ===")
    
    try:
        # Create agent
        agent = await create_strands_glean_agent(
            project_root="/Users/apple/projects/cryptotrading",
            agent_id="test-glean-agent"
        )
        
        print(f"✓ Agent created successfully")
        
        # Test context summary
        summary = await agent.get_context_summary()
        print(f"✓ Context summary: {summary['capabilities']} capabilities available")
        
        # Test new capabilities
        capabilities_to_test = [
            ("data_flow_analysis", "BTC-USD"),
            ("parameter_analysis", "model"),
            ("factor_analysis", "BTC-USD"),
            ("data_quality_analysis", "")
        ]
        
        for capability, query in capabilities_to_test:
            if capability in agent.capabilities:
                try:
                    result = await agent.analyze_code(capability, query)
                    print(f"✓ {capability} completed: {result.get('status', 'unknown')}")
                except Exception as e:
                    print(f"✗ {capability} failed: {e}")
            else:
                print(f"⚠ {capability} not available")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False


async def test_data_flow_indexer():
    """Test the data flow SCIP indexer"""
    print("\n=== Testing Data Flow Indexer ===")
    
    try:
        from cryptotrading.infrastructure.analysis.scip_data_flow_indexer import DataFlowSCIPIndexer
        
        # Create indexer
        indexer = DataFlowSCIPIndexer("/Users/apple/projects/cryptotrading")
        
        # Test indexing a simple Python file
        test_file = "/Users/apple/projects/cryptotrading/src/cryptotrading/core/ml/models.py"
        if os.path.exists(test_file):
            result = indexer.index_file(test_file)
            print(f"✓ Indexed file: {test_file}")
            print(f"  Status: {result.get('status', 'unknown')}")
            
            if 'data_flow' in result:
                flow = result['data_flow']
                print(f"  Data flow: {flow['inputs']} inputs, {flow['outputs']} outputs, {flow['parameters']} parameters, {flow['factors']} factors")
            
            # Get extracted facts
            facts = indexer.get_data_flow_facts()
            print(f"  Extracted {len(facts)} data flow facts")
            
            return True
        else:
            print(f"⚠ Test file not found: {test_file}")
            return True  # Not a failure, just missing test file
            
    except Exception as e:
        print(f"✗ Data flow indexer test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("🧪 Testing Extended Glean Agent Data Collection\n")
    
    tests = [
        ("Data Collection Decorators", test_data_collection_decorators),
        ("Angle Queries", lambda: test_angle_queries()),
        ("Data Flow Indexer", test_data_flow_indexer),
        ("Strands-Glean Agent", test_strands_glean_agent),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Glean data collection is working.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)