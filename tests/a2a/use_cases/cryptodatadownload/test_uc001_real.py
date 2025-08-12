#!/usr/bin/env python3
"""
REAL Test Implementation for UC001: CryptoDataDownload Schema Discovery
NO MOCKS - Actual API calls to CryptoDataDownload
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from рекс.a2a.agents.data_management_agent import data_management_agent

async def test_real_cryptodatadownload_discovery():
    """Real test against CryptoDataDownload API"""
    print("=== REAL UC001 Test: CryptoDataDownload Schema Discovery ===")
    print(f"Start Time: {datetime.now()}")
    print("-" * 60)
    
    # Test 1: Discover real schema from CryptoDataDownload
    print("\n1. Testing Real Schema Discovery from CryptoDataDownload...")
    
    try:
        # Call the real data management agent
        result = await data_management_agent.analyze_data_source(
            "cryptodatadownload",
            {
                "exchange": "binance",
                "pair": "BTCUSDT", 
                "timeframe": "d"
            }
        )
        
        print(f"\nDiscovery Result Success: {result.get('success')}")
        
        if result.get('success'):
            # Display discovered structure
            print("\n2. Discovered Data Structure:")
            print(f"   Source: {result.get('source')}")
            print(f"   URL Pattern: {result.get('url_pattern')}")
            
            structure = result.get('structure', {})
            print(f"   Format: {structure.get('format')}")
            print(f"   Total Columns: {structure.get('total_columns')}")
            print(f"   Header Rows: {structure.get('header_rows')}")
            
            # Show discovered columns
            print("\n3. Discovered Columns:")
            columns = structure.get('columns', {})
            for col_name, col_info in list(columns.items())[:5]:  # First 5 columns
                print(f"   - {col_name}:")
                print(f"     Type: {col_info.get('data_type')}")
                print(f"     Sample Values: {col_info.get('sample_values', [])[:2]}")
                print(f"     DB Mapping: {col_info.get('database_mapping')}")
            
            # Show quality metrics
            print("\n4. Calculated Quality Metrics (from real data):")
            quality = result.get('sap_resource_discovery', {}).get('Governance', {}).get('QualityMetrics', {})
            print(f"   Completeness: {quality.get('Completeness', 0):.3f}")
            print(f"   Accuracy: {quality.get('Accuracy', 0):.3f}")
            print(f"   Consistency: {quality.get('Consistency', 0):.3f}")
            print(f"   Sample Size: {quality.get('SampleSize', 0)} rows")
            
            # Show SAP CAP Schema
            print("\n5. Generated SAP CAP Schema:")
            cap_schema = result.get('sap_cap_schema', {})
            print(f"   Entity Name: {cap_schema.get('entity_name')}")
            print(f"   Namespace: {cap_schema.get('namespace')}")
            print("   CDS Definition (first 500 chars):")
            cds_def = cap_schema.get('cds_definition', '')[:500]
            print(f"   {cds_def}...")
            
            # Test 2: Store the discovered schema
            print("\n6. Testing Schema Storage (SQLite)...")
            
            store_result = await data_management_agent.agent(
                f"Store schema with data: {json.dumps(result)} and storage_type: sqlite"
            )
            
            store_data = json.loads(store_result) if isinstance(store_result, str) else store_result
            print(f"   Storage Success: {store_data.get('success')}")
            
            if store_data.get('success'):
                data_product_id = store_data.get('data_product_id')
                print(f"   Data Product ID: {data_product_id}")
                print(f"   Schema Hash: {store_data.get('schema_hash')}")
                
                # Test 3: Retrieve the stored schema
                print("\n7. Testing Schema Retrieval...")
                
                retrieve_result = await data_management_agent.agent(
                    f"Get schema for data_product_id: {data_product_id}"
                )
                
                retrieve_data = json.loads(retrieve_result) if isinstance(retrieve_result, str) else retrieve_result
                
                if retrieve_data.get('source'):
                    print(f"   Retrieved Successfully: Yes")
                    print(f"   Source Matches: {retrieve_data.get('source') == 'cryptodatadownload'}")
                else:
                    print(f"   Retrieved Successfully: No")
                
                # Test 4: List all schemas
                print("\n8. Testing List Schemas...")
                
                list_result = await data_management_agent.agent(
                    "List schemas"
                )
                
                list_data = json.loads(list_result) if isinstance(list_result, str) else list_result
                print(f"   Total Schemas: {list_data.get('count', 0)}")
                
                if list_data.get('schemas'):
                    for schema in list_data['schemas'][:3]:  # First 3
                        print(f"   - {schema.get('data_product_id')}: {schema.get('validation_status')}")
            
        else:
            print(f"\nDiscovery Failed: {result.get('error')}")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Completed")

async def test_real_performance():
    """Test real performance against BR001 (< 30 seconds)"""
    print("\n=== Performance Test: BR001 Compliance ===")
    
    import time
    start_time = time.time()
    
    result = await data_management_agent.analyze_data_source(
        "cryptodatadownload",
        {"exchange": "binance", "pair": "ETHUSDT", "timeframe": "d"}
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"BR001 Compliance (<30s): {'PASS' if execution_time < 30 else 'FAIL'}")
    
    return execution_time < 30

async def test_multiple_exchanges():
    """Test discovery across multiple exchanges"""
    print("\n=== Testing Multiple Exchanges ===")
    
    exchanges = ["binance", "coinbase", "kraken"]
    results = []
    
    for exchange in exchanges:
        print(f"\nTesting {exchange}...")
        result = await data_management_agent.analyze_data_source(
            "cryptodatadownload",
            {"exchange": exchange, "pair": "BTCUSD", "timeframe": "d"}
        )
        
        success = result.get('success', False)
        print(f"  {exchange}: {'✓' if success else '✗'}")
        
        if success:
            quality = result.get('sap_resource_discovery', {}).get('Governance', {}).get('QualityMetrics', {})
            print(f"    Completeness: {quality.get('Completeness', 0):.3f}")
            print(f"    Sample Size: {quality.get('SampleSize', 0)}")
        
        results.append((exchange, success))
    
    return results

if __name__ == "__main__":
    print("Starting REAL tests (no mocks)...")
    print("This will make actual HTTP requests to CryptoDataDownload\n")
    
    # Run the real tests
    asyncio.run(test_real_cryptodatadownload_discovery())
    
    # Run performance test
    perf_passed = asyncio.run(test_real_performance())
    
    # Test multiple exchanges
    exchange_results = asyncio.run(test_multiple_exchanges())
    
    # Summary
    print("\n=== TEST SUMMARY ===")
    print(f"Performance Test: {'PASS' if perf_passed else 'FAIL'}")
    print("\nExchange Coverage:")
    for exchange, success in exchange_results:
        print(f"  {exchange}: {'PASS' if success else 'FAIL'}")
    
    print("\nAll tests completed with REAL data!")