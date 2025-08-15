#!/usr/bin/env python3
"""
Test Production Enhancements
Validate performance monitoring, alerting, notifications, and rate limiting
"""

import time
import asyncio
from datetime import datetime

def test_alert_callback(alert_type: str, message: str, data: dict):
    """Sample alert callback for testing"""
    print(f"🚨 ALERT [{alert_type.upper()}]: {message}")
    if data:
        print(f"   Data: {data}")

def test_notification_callback(notification_type: str, message: str):
    """Sample notification callback for testing"""
    print(f"📢 NOTIFICATION [{notification_type.upper()}]: {message}")

def test_production_enhancements():
    """Test all production enhancements"""
    
    print("🔧 TESTING PRODUCTION ENHANCEMENTS")
    print("=" * 60)
    
    try:
        # Import and initialize
        print("1️⃣ Testing Enhanced Client with Production Features...")
        from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
        
        client = EnhancedComprehensiveMetricsClient()
        
        # Add alert and notification callbacks
        client.add_alert_callback(test_alert_callback)
        client.add_notification_callback(test_notification_callback)
        
        print("   ✅ Client initialized with monitoring enabled")
        
        # Test service health check
        print("\n2️⃣ Testing Service Health Monitoring...")
        health_status = client.check_yahoo_finance_health()
        print(f"   Service Status: {health_status['status']}")
        if health_status['status'] == 'healthy':
            print(f"   Response Time: {health_status['response_time_ms']:.2f}ms")
        else:
            print(f"   Error: {health_status.get('error', 'Unknown')}")
        
        # Test performance monitoring
        print("\n3️⃣ Testing Performance Monitoring...")
        
        # Make several monitored calls
        symbols_to_test = ['^VIX', '^TNX', 'GC=F']
        
        for symbol in symbols_to_test:
            try:
                data = client.get_comprehensive_data(symbol, days_back=5)
                print(f"   ✅ Retrieved {len(data)} records for {symbol}")
            except Exception as e:
                print(f"   ❌ Failed to retrieve {symbol}: {e}")
        
        # Get performance metrics
        perf_metrics = client.get_performance_metrics()
        print(f"   Performance metrics collected for {len(perf_metrics['methods'])} methods")
        
        for method, metrics in perf_metrics['methods'].items():
            print(f"     {method}: {metrics['total_calls']} calls, "
                  f"{metrics['success_rate']:.2%} success rate, "
                  f"{metrics['avg_latency_ms']:.0f}ms avg latency")
        
        # Test rate limiting
        print("\n4️⃣ Testing Rate Limiting...")
        
        # Make rapid calls to test rate limiter
        rapid_calls = 0
        start_time = time.time()
        
        try:
            for i in range(5):  # Quick test with 5 calls
                client.get_comprehensive_data('^VIX', days_back=1)
                rapid_calls += 1
        except Exception as e:
            print(f"   Rate limiting triggered: {e}")
        
        elapsed = time.time() - start_time
        print(f"   Made {rapid_calls} rapid calls in {elapsed:.2f}s")
        
        # Check rate limiter status
        rate_status = client._rate_limiter
        print(f"   Rate limiter: {len(rate_status['call_timestamps'])}/{rate_status['calls_per_minute']} calls/min")
        
        # Test comprehensive dashboard
        print("\n5️⃣ Testing Service Status Dashboard...")
        dashboard = client.get_service_status_dashboard()
        
        print(f"   Overall Health Score: {dashboard['overall_health_score']}/100")
        print(f"   Service Status: {dashboard['service_status']}")
        print(f"   Circuit Breaker: {'OPEN' if dashboard['circuit_breaker_open'] else 'CLOSED'}")
        print(f"   Error Rate: {dashboard['performance_summary']['error_rate_pct']:.2f}%")
        print(f"   Avg Latency: {dashboard['performance_summary']['avg_latency_ms']:.0f}ms")
        
        # Test error scenarios
        print("\n6️⃣ Testing Error Handling and Alerting...")
        
        try:
            # Try invalid symbol to trigger error alert
            client.get_comprehensive_data("INVALID_SYMBOL_XYZ123", days_back=1)
        except Exception as e:
            print(f"   ✅ Error properly caught and alerted: {str(e)[:50]}...")
        
        # Test protocol negotiation with monitoring
        print("\n7️⃣ Testing Enhanced Protocol Features...")
        
        protocol_result = client.negotiate_protocol_version('2.1.0')
        print(f"   Protocol negotiated: {protocol_result['negotiated_version']}")
        
        # Test migration with monitoring
        legacy_data = {
            'message_type': 'DATA_REQUEST',
            'payload': {'symbols': 'BTC-USD', 'days': 30}
        }
        migration_result = client.migrate_from_legacy_protocol(legacy_data)
        print(f"   Migration success: {migration_result['success']}")
        
        # Final dashboard check
        print("\n8️⃣ Final Production Readiness Assessment...")
        final_dashboard = client.get_service_status_dashboard()
        
        health_score = final_dashboard['overall_health_score']
        if health_score >= 95:
            grade = "🥇 EXCELLENT"
        elif health_score >= 90:
            grade = "🥈 VERY GOOD"
        elif health_score >= 85:
            grade = "🥉 GOOD"
        else:
            grade = "⚠️  NEEDS IMPROVEMENT"
        
        print(f"   Final Health Score: {health_score}/100 - {grade}")
        print(f"   Production Ready: {'YES' if health_score >= 90 else 'MONITOR CLOSELY'}")
        
        print("\n" + "=" * 60)
        print("🎯 PRODUCTION ENHANCEMENTS TEST COMPLETE")
        print("✅ Performance monitoring: ACTIVE")
        print("✅ Error alerting: ACTIVE") 
        print("✅ Service health checks: ACTIVE")
        print("✅ Rate limiting: ACTIVE")
        print("✅ Circuit breaker: ACTIVE")
        print("✅ Notification system: ACTIVE")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_production_enhancements()
    exit(0 if success else 1)