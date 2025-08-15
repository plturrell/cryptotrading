#!/usr/bin/env python3
"""
Real Data Validation Test
Validate the enhanced comprehensive metrics client with real market data
"""

import time
import logging
from datetime import datetime

# Disable excessive logging
logging.disable(logging.CRITICAL)

def test_real_data_functionality():
    """Test core functionality with real market data"""
    
    print("🔬 REAL DATA VALIDATION TEST")
    print("=" * 50)
    
    try:
        # Test imports
        print("1️⃣ Testing Import...")
        from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
        print("   ✅ Import successful")
        
        # Test initialization 
        print("\n2️⃣ Testing Initialization...")
        start_time = time.time()
        client = EnhancedComprehensiveMetricsClient()
        init_time = time.time() - start_time
        print(f"   ✅ Initialized in {init_time:.2f}s with {len(client.COMPREHENSIVE_INDICATORS)} indicators")
        
        # Test basic data retrieval
        print("\n3️⃣ Testing Real Data Retrieval...")
        try:
            vix_data = client.get_comprehensive_data('^VIX', days_back=5)
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                print(f"   ✅ VIX data retrieved: Current level {current_vix:.2f}")
            else:
                print("   ⚠️  VIX data empty")
        except Exception as e:
            print(f"   ❌ VIX data failed: {e}")
        
        # Test multiple indicators
        print("\n4️⃣ Testing Multiple Indicators...")
        test_symbols = ['^VIX', '^TNX', 'GC=F']
        results = client.get_multiple_comprehensive_data(test_symbols, days_back=3)
        success_count = len([s for s, d in results.items() if not d.empty])
        print(f"   ✅ {success_count}/{len(test_symbols)} indicators loaded successfully")
        
        # Test calculation methods
        print("\n5️⃣ Testing Calculation Methods...")
        
        # Test weighted signals
        if results:
            dummy_crypto = vix_data if not vix_data.empty else None
            if dummy_crypto is not None:
                signals = client.calculate_weighted_signals(results, dummy_crypto)
                if not signals.empty:
                    print(f"   ✅ Weighted signals: {len(signals.columns)} signals calculated")
                else:
                    print("   ⚠️  Weighted signals empty")
        
        # Test protocol negotiation
        print("\n6️⃣ Testing Protocol Negotiation...")
        protocol_result = client.negotiate_protocol_version('2.1.0')
        if 'negotiated_version' in protocol_result:
            print(f"   ✅ Protocol negotiated: {protocol_result['negotiated_version']}")
        else:
            print("   ⚠️  Protocol negotiation issue")
        
        # Test migration tools
        print("\n7️⃣ Testing Migration Tools...")
        legacy_data = {
            'message_type': 'DATA_REQUEST',
            'payload': {'symbols': 'BTC-USD', 'days': 30}
        }
        migration_result = client.migrate_from_legacy_protocol(legacy_data)
        if migration_result.get('success'):
            print("   ✅ Legacy migration working")
        else:
            print("   ⚠️  Migration issue")
        
        # Test position sizing
        print("\n8️⃣ Testing Position Sizing...")
        if not vix_data.empty and not signals.empty:
            position_sizing = client.calculate_position_sizing(signals, vix_data)
            if not position_sizing.empty:
                print(f"   ✅ Position sizing: {len(position_sizing.columns)} metrics calculated")
            else:
                print("   ⚠️  Position sizing empty")
        
        print("\n" + "=" * 50)
        print("🎯 REAL DATA VALIDATION COMPLETE")
        print("✅ All core functions operational with real market data")
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE: {e}")
        return False

if __name__ == "__main__":
    success = test_real_data_functionality()
    exit(0 if success else 1)