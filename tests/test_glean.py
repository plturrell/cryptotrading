#!/usr/bin/env python3
"""
Simple test for Glean data collection capabilities
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_schemas():
    """Test data schemas"""
    print("=== Testing Data Schemas ===")
    
    try:
        from cryptotrading.infrastructure.analysis.glean_data_schemas import (
            DataInputFact, DataOutputFact, ParameterFact, FactorFact
        )
        
        # Test creating facts
        input_fact = DataInputFact(
            function="test_func",
            file="test.py",
            line=10,
            input_id="test123",
            data_type="market_data",
            source="yahoo_finance",
            symbol="BTC-USD"
        )
        
        fact_dict = input_fact.to_fact()
        print(f"✓ Created input fact: {fact_dict['predicate']}")
        
        output_fact = DataOutputFact(
            function="test_func",
            file="test.py", 
            line=20,
            output_id="out123",
            output_type="prediction",
            data_shape={"type": "float", "value": "123.45"}
        )
        
        output_dict = output_fact.to_fact()
        print(f"✓ Created output fact: {output_dict['predicate']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Schema test failed: {e}")
        return False

def test_angle_queries():
    """Test angle queries"""
    print("\n=== Testing Angle Queries ===")
    
    try:
        from cryptotrading.infrastructure.analysis.crypto_angle_queries import (
            create_crypto_query, validate_crypto_query
        )
        
        # Test creating a query
        query = create_crypto_query("data_inputs_by_symbol", {"symbol": "BTC-USD"})
        print(f"✓ Created query")
        
        # Test validation
        validation = validate_crypto_query(query)
        if validation["valid"]:
            print(f"✓ Query is valid")
            print(f"  Predicates used: {validation['predicates_used']}")
        else:
            print(f"✗ Query invalid: {validation['errors']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Angle query test failed: {e}")
        return False

def test_data_flow_visitor():
    """Test the data flow AST visitor"""
    print("\n=== Testing Data Flow Visitor ===")
    
    try:
        import ast
        from cryptotrading.infrastructure.analysis.scip_data_flow_indexer import DataFlowVisitor
        
        # Test Python code with data operations
        test_code = """
import yfinance as yf
import pandas as pd

def calculate_rsi(symbol, period=14):
    # Data input
    data = yf.download(symbol, period="1y")
    
    # Parameter
    window_size = period
    
    # Factor calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Data output
    return rsi
"""
        
        # Parse and visit
        tree = ast.parse(test_code)
        visitor = DataFlowVisitor("test.py")
        visitor.visit(tree)
        
        print(f"✓ Analyzed code successfully")
        print(f"  Found {len(visitor.data_inputs)} data inputs")
        print(f"  Found {len(visitor.data_outputs)} data outputs") 
        print(f"  Found {len(visitor.parameters)} parameters")
        print(f"  Found {len(visitor.factors)} factors")
        
        # Print some details
        if visitor.data_inputs:
            print(f"  First input: {visitor.data_inputs[0].source}")
        if visitor.parameters:
            print(f"  First parameter: {visitor.parameters[0].name}")
            
        return True
        
    except Exception as e:
        print(f"✗ Data flow visitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_runtime_decorators():
    """Test runtime data collection decorators"""
    print("\n=== Testing Runtime Decorators ===")
    
    try:
        from cryptotrading.infrastructure.analysis.glean_runtime_collectors import (
            track_data_input, track_data_output, track_factor
        )
        import pandas as pd
        
        # Create a simple decorated function
        @track_data_input(source="test", data_type="prices")
        @track_data_output(output_type="indicator")
        @track_factor(factor_name="simple_ma", category="trend")
        async def calculate_ma(prices: pd.Series, window: int = 5) -> pd.Series:
            return prices.rolling(window=window).mean()
        
        # Test data
        test_prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Call the function
        result = await calculate_ma(test_prices, window=3)
        
        print(f"✓ Decorated function executed successfully")
        print(f"  Result length: {len(result)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Runtime decorator test failed: {e}")
        return False

async def run_tests():
    """Run all tests"""
    print("🧪 Testing Glean Data Collection Components\n")
    
    tests = [
        ("Data Schemas", lambda: test_schemas()),
        ("Angle Queries", lambda: test_angle_queries()),
        ("Data Flow Visitor", lambda: test_data_flow_visitor()),
        ("Runtime Decorators", test_runtime_decorators),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
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
        print("🎉 All core components working!")
    else:
        print("⚠️  Some components need fixes.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)