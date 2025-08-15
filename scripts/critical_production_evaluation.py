#!/usr/bin/env python3
"""
CRITICAL PRODUCTION EVALUATION
Brutally honest assessment of the implementation for production readiness
This evaluation will expose real issues that could cause production failures
"""

import asyncio
import time
import sys
import os
import logging
import traceback
import psutil
import gc
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Disable logging for clean output
logging.disable(logging.CRITICAL)

class CriticalProductionEvaluator:
    """
    Merciless production readiness evaluation
    Tests everything that could break in production
    """
    
    def __init__(self):
        self.critical_issues = []
        self.warnings = []
        self.memory_baseline = 0
        self.performance_issues = []
    
    def add_critical_issue(self, issue: str, details: str = ""):
        """Record a critical issue that would cause production failure"""
        self.critical_issues.append({
            'issue': issue,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        print(f"❌ CRITICAL: {issue}")
        if details:
            print(f"   Details: {details}")
    
    def add_warning(self, warning: str, details: str = ""):
        """Record a warning that could cause issues"""
        self.warnings.append({
            'warning': warning,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        print(f"⚠️  WARNING: {warning}")
        if details:
            print(f"   Details: {details}")
    
    def measure_memory(self, operation: str):
        """Measure memory usage for operation"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if self.memory_baseline == 0:
            self.memory_baseline = memory_mb
        else:
            increase = memory_mb - self.memory_baseline
            if increase > 50:  # More than 50MB increase
                self.add_critical_issue(
                    f"Memory leak detected in {operation}",
                    f"Memory increased by {increase:.1f}MB"
                )
    
    async def evaluate_critical_production_readiness(self) -> Dict[str, Any]:
        """Conduct brutal production readiness evaluation"""
        
        print("🔥 CRITICAL PRODUCTION EVALUATION - NO MERCY MODE")
        print("=" * 80)
        print("Testing everything that could break in production...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_rating': 0,
            'critical_issues': [],
            'warnings': [],
            'test_results': {},
            'production_ready': False
        }
        
        try:
            # 1. Import and Basic Functionality Test
            print("\n1️⃣ Testing Imports and Basic Functionality...")
            await self.test_imports_and_basic_functionality()
            
            # 2. Error Handling and Edge Cases
            print("\n2️⃣ Testing Error Handling and Edge Cases...")
            await self.test_error_handling_edge_cases()
            
            # 3. Performance and Memory Tests
            print("\n3️⃣ Testing Performance and Memory Usage...")
            await self.test_performance_and_memory()
            
            # 4. Real Data Integration Tests
            print("\n4️⃣ Testing Real Data Integration...")
            await self.test_real_data_integration()
            
            # 5. Concurrency and Threading Tests
            print("\n5️⃣ Testing Concurrency and Threading...")
            await self.test_concurrency_and_threading()
            
            # 6. Configuration and Deployment Tests
            print("\n6️⃣ Testing Configuration and Deployment...")
            await self.test_configuration_and_deployment()
            
            # 7. Security and Input Validation
            print("\n7️⃣ Testing Security and Input Validation...")
            await self.test_security_and_validation()
            
            # 8. Production Scenario Simulation
            print("\n8️⃣ Simulating Production Scenarios...")
            await self.test_production_scenarios()
            
        except Exception as e:
            self.add_critical_issue(
                "Test suite crashed",
                f"Exception: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
        
        # Calculate final rating
        results['critical_issues'] = self.critical_issues
        results['warnings'] = self.warnings
        results['overall_rating'] = self.calculate_brutal_rating()
        results['production_ready'] = len(self.critical_issues) == 0 and results['overall_rating'] >= 90
        
        return results
    
    async def test_imports_and_basic_functionality(self):
        """Test if basic imports and initialization work"""
        try:
            # Test import
            from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            
            # Test initialization
            self.measure_memory("initialization")
            client = EnhancedComprehensiveMetricsClient()
            self.measure_memory("post_initialization")
            
            # Test basic configuration
            if len(client.COMPREHENSIVE_INDICATORS) < 50:
                self.add_critical_issue(
                    "Insufficient indicators for production",
                    f"Only {len(client.COMPREHENSIVE_INDICATORS)} indicators, need 50+"
                )
            
            # Test method availability
            required_methods = [
                'calculate_weighted_signals', 'calculate_position_sizing',
                'get_threshold_alerts', 'calculate_options_analytics',
                'calculate_ensemble_correlations', 'stream_real_time_indicators',
                'negotiate_protocol_version', 'migrate_from_legacy_protocol'
            ]
            
            missing_methods = [method for method in required_methods if not hasattr(client, method)]
            if missing_methods:
                self.add_critical_issue(
                    "Critical methods missing",
                    f"Missing: {', '.join(missing_methods)}"
                )
            
            print("✅ Basic functionality test passed")
            
        except ImportError as e:
            self.add_critical_issue("Import failure", f"Cannot import required modules: {e}")
        except Exception as e:
            self.add_critical_issue("Initialization failure", f"Failed to initialize: {e}")
    
    async def test_error_handling_edge_cases(self):
        """Test error handling with malicious/edge case inputs"""
        try:
            from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            client = EnhancedComprehensiveMetricsClient()
            
            # Test 1: Empty data inputs
            try:
                result = client.calculate_weighted_signals({}, pd.DataFrame())
                if not isinstance(result, pd.DataFrame):
                    self.add_critical_issue("Invalid return type for empty input")
            except Exception as e:
                self.add_warning("Exception on empty input", str(e))
            
            # Test 2: Invalid symbols
            try:
                result = client.get_indicator_info("TOTALLY_INVALID_SYMBOL_XYZ123")
                if 'error' not in result:
                    self.add_warning("Should return error for invalid symbol")
            except Exception as e:
                self.add_warning("Exception on invalid symbol", str(e))
            
            # Test 3: Massive data inputs (memory bomb test)
            try:
                huge_data = pd.DataFrame({
                    'Close': np.random.randn(100000),
                    'Date': pd.date_range('2000-01-01', periods=100000)
                }).set_index('Date')
                
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                result = client.calculate_weighted_signals({'^VIX': huge_data}, huge_data)
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                memory_used = end_memory - start_memory
                if memory_used > 500:  # More than 500MB
                    self.add_critical_issue(
                        "Memory usage too high for large datasets",
                        f"Used {memory_used:.1f}MB for 100k records"
                    )
                
            except Exception as e:
                self.add_warning("Failed on large dataset", str(e))
            
            # Test 4: SQL Injection-style inputs
            malicious_inputs = [
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../etc/passwd",
                "\x00\x01\x02\x03",  # Null bytes
                "A" * 10000  # Buffer overflow attempt
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    result = client.get_indicator_info(malicious_input)
                    # Should not crash, should handle gracefully
                except Exception as e:
                    self.add_warning(f"Exception on malicious input: {malicious_input[:20]}", str(e))
            
            # Test 5: Protocol negotiation with invalid versions
            invalid_versions = [
                "not.a.version",
                "999999.999999.999999",
                "",
                None,
                123,
                ["1.0.0"],  # Wrong type
                {"version": "1.0.0"}  # Wrong type
            ]
            
            for invalid_version in invalid_versions:
                try:
                    result = client.negotiate_protocol_version(invalid_version)
                    if 'error' not in result and 'fallback_version' not in result:
                        self.add_warning(f"Should handle invalid version gracefully: {invalid_version}")
                except Exception as e:
                    # Some exceptions are OK, but shouldn't crash the whole system
                    pass
            
            print("✅ Error handling tests completed")
            
        except Exception as e:
            self.add_critical_issue("Error handling test failed", str(e))
    
    async def test_performance_and_memory(self):
        """Test performance and memory usage under realistic loads"""
        try:
            from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            
            # Test 1: Multiple client initialization (memory leak test)
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            clients = []
            for i in range(10):
                client = EnhancedComprehensiveMetricsClient()
                clients.append(client)
            
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_per_client = (memory_after - memory_start) / 10
            
            if memory_per_client > 50:  # More than 50MB per client
                self.add_critical_issue(
                    "Excessive memory usage per client",
                    f"{memory_per_client:.1f}MB per client"
                )
            
            # Cleanup
            del clients
            gc.collect()
            
            # Test 2: Performance benchmarks
            client = EnhancedComprehensiveMetricsClient()
            
            # Test indicator info retrieval speed
            start_time = time.time()
            for i in range(100):
                client.get_indicator_info('^VIX')
            indicator_time = time.time() - start_time
            
            if indicator_time > 10:  # Should be under 10 seconds for 100 calls
                self.add_warning(
                    "Slow indicator info retrieval",
                    f"{indicator_time:.2f}s for 100 calls"
                )
            
            # Test 3: Protocol negotiation speed
            start_time = time.time()
            for i in range(1000):
                client.negotiate_protocol_version('2.1.0')
            protocol_time = time.time() - start_time
            
            if protocol_time > 5:  # Should be under 5 seconds for 1000 calls
                self.add_warning(
                    "Slow protocol negotiation",
                    f"{protocol_time:.2f}s for 1000 calls"
                )
            
            print("✅ Performance tests completed")
            
        except Exception as e:
            self.add_critical_issue("Performance test failed", str(e))
    
    async def test_real_data_integration(self):
        """Test integration with real Yahoo Finance data"""
        try:
            from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            client = EnhancedComprehensiveMetricsClient()
            
            # Test 1: Real data retrieval
            try:
                start_time = time.time()
                vix_data = client.get_comprehensive_data('^VIX', days_back=5)
                data_time = time.time() - start_time
                
                if vix_data.empty:
                    self.add_critical_issue("Cannot retrieve real market data for VIX")
                elif data_time > 30:  # Should be under 30 seconds
                    self.add_warning(f"Slow data retrieval: {data_time:.2f}s for VIX")
                
            except Exception as e:
                self.add_critical_issue("Real data retrieval failed", str(e))
            
            # Test 2: API rate limits
            start_time = time.time()
            successful_calls = 0
            failed_calls = 0
            
            test_symbols = ['^VIX', '^TNX', 'GC=F', '^GSPC', 'TLT']
            
            for symbol in test_symbols:
                try:
                    data = client.get_comprehensive_data(symbol, days_back=2)
                    if not data.empty:
                        successful_calls += 1
                    else:
                        failed_calls += 1
                except Exception:
                    failed_calls += 1
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
            
            total_time = time.time() - start_time
            
            if failed_calls > successful_calls:
                self.add_critical_issue(
                    "High failure rate for data retrieval",
                    f"{failed_calls}/{len(test_symbols)} calls failed"
                )
            
            if total_time > 60:  # Should complete within 60 seconds
                self.add_warning(f"Slow bulk data retrieval: {total_time:.2f}s")
            
            # Test 3: Data quality validation
            if successful_calls > 0:
                sample_data = client.get_comprehensive_data('^VIX', days_back=5)
                if not sample_data.empty:
                    # Check for required columns
                    required_columns = ['Open', 'High', 'Low', 'Close']
                    missing_columns = [col for col in required_columns if col not in sample_data.columns]
                    
                    if missing_columns:
                        self.add_critical_issue(
                            "Missing required data columns",
                            f"Missing: {', '.join(missing_columns)}"
                        )
                    
                    # Check for data quality issues
                    if sample_data['Close'].isna().sum() > len(sample_data) * 0.1:
                        self.add_warning("High percentage of missing close prices")
                    
                    if (sample_data['High'] < sample_data['Low']).any():
                        self.add_critical_issue("Data quality issue: High < Low detected")
            
            print("✅ Real data integration tests completed")
            
        except Exception as e:
            self.add_critical_issue("Real data integration test failed", str(e))
    
    async def test_concurrency_and_threading(self):
        """Test concurrent access and threading safety"""
        try:
            from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            
            # Test 1: Concurrent client access
            clients = [EnhancedComprehensiveMetricsClient() for _ in range(5)]
            
            async def concurrent_operation(client, operation_id):
                try:
                    # Simulate concurrent operations
                    result1 = client.get_indicator_info('^VIX')
                    result2 = client.negotiate_protocol_version('2.1.0')
                    
                    # Small delay
                    await asyncio.sleep(0.1)
                    
                    result3 = client.get_indicator_info('^TNX')
                    
                    return {'success': True, 'operation_id': operation_id}
                except Exception as e:
                    return {'success': False, 'operation_id': operation_id, 'error': str(e)}
            
            # Run concurrent operations
            tasks = [concurrent_operation(client, i) for i, client in enumerate(clients)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            failed_operations = sum(1 for result in results if isinstance(result, Exception) or not result.get('success', False))
            
            if failed_operations > 0:
                self.add_warning(
                    "Concurrency issues detected",
                    f"{failed_operations}/{len(tasks)} operations failed"
                )
            
            # Test 2: Streaming concurrency
            streaming_errors = []
            
            async def test_streaming_callback(data):
                # Simulate processing
                await asyncio.sleep(0.01)
            
            try:
                # Start multiple streams
                stream_tasks = []
                for i in range(3):
                    task = asyncio.create_task(
                        clients[0].stream_real_time_indicators(
                            ['^VIX'], test_streaming_callback, interval_seconds=1
                        )
                    )
                    stream_tasks.append(task)
                
                # Let them run briefly
                await asyncio.sleep(2)
                
                # Cancel all streams
                for task in stream_tasks:
                    task.cancel()
                
                # Wait for cancellation
                await asyncio.gather(*stream_tasks, return_exceptions=True)
                
            except Exception as e:
                self.add_warning("Streaming concurrency issue", str(e))
            
            print("✅ Concurrency tests completed")
            
        except Exception as e:
            self.add_critical_issue("Concurrency test failed", str(e))
    
    async def test_configuration_and_deployment(self):
        """Test configuration management and deployment readiness"""
        try:
            # Test 1: Configuration file access
            config_path = "config/enhanced_indicators.yaml"
            if not os.path.exists(config_path):
                self.add_critical_issue(
                    "Configuration file missing",
                    f"Cannot find {config_path}"
                )
            
            # Test 2: Environment variable handling
            original_env = os.environ.copy()
            
            try:
                # Test missing environment variables
                if 'XAI_API_KEY' in os.environ:
                    del os.environ['XAI_API_KEY']
                if 'GROK_API_KEY' in os.environ:
                    del os.environ['GROK_API_KEY']
                
                # Should still work without API keys for basic functionality
                from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
                client = EnhancedComprehensiveMetricsClient()
                
                # Basic operations should work
                info = client.get_indicator_info('^VIX')
                if 'error' in info:
                    self.add_warning("Basic operations fail without API keys")
                
            finally:
                # Restore environment
                os.environ.clear()
                os.environ.update(original_env)
            
            # Test 3: Import path issues
            try:
                import sys
                original_path = sys.path.copy()
                
                # Remove current directory from path
                if '' in sys.path:
                    sys.path.remove('')
                if '.' in sys.path:
                    sys.path.remove('.')
                
                # Try import (should still work with proper package structure)
                try:
                    from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
                except ImportError:
                    self.add_critical_issue("Import fails without current directory in path")
                
            finally:
                sys.path = original_path
            
            print("✅ Configuration and deployment tests completed")
            
        except Exception as e:
            self.add_critical_issue("Configuration test failed", str(e))
    
    async def test_security_and_validation(self):
        """Test security and input validation"""
        try:
            from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            client = EnhancedComprehensiveMetricsClient()
            
            # Test 1: File path traversal attempts
            dangerous_paths = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config",
                "/etc/shadow",
                "C:\\Windows\\System32\\config\\SAM"
            ]
            
            for dangerous_path in dangerous_paths:
                try:
                    # Try to initialize with dangerous path
                    dangerous_client = EnhancedComprehensiveMetricsClient(config_path=dangerous_path)
                    self.add_critical_issue(
                        "Path traversal vulnerability",
                        f"Accepts dangerous path: {dangerous_path}"
                    )
                except:
                    # Should fail - this is good
                    pass
            
            # Test 2: Large input validation
            giant_symbol_list = ['SYM' + str(i) for i in range(10000)]
            
            try:
                start_time = time.time()
                # This should either fail gracefully or handle efficiently
                client.get_multiple_comprehensive_data(giant_symbol_list, days_back=1)
                processing_time = time.time() - start_time
                
                if processing_time > 300:  # More than 5 minutes
                    self.add_critical_issue(
                        "No input size limits",
                        f"Processed 10k symbols for {processing_time:.1f}s"
                    )
            except Exception:
                # Failing is acceptable for giant inputs
                pass
            
            # Test 3: Type safety
            wrong_types = [
                123,  # Integer instead of string
                [],   # List instead of string
                {},   # Dict instead of string
                None  # None instead of string
            ]
            
            type_errors = 0
            for wrong_type in wrong_types:
                try:
                    client.get_indicator_info(wrong_type)
                except TypeError:
                    type_errors += 1
                except:
                    # Other exceptions are OK
                    pass
            
            if type_errors < len(wrong_types) * 0.5:
                self.add_warning(
                    "Weak type validation",
                    f"Only {type_errors}/{len(wrong_types)} type errors caught"
                )
            
            print("✅ Security and validation tests completed")
            
        except Exception as e:
            self.add_critical_issue("Security test failed", str(e))
    
    async def test_production_scenarios(self):
        """Simulate real production scenarios"""
        try:
            from cryptotrading.core.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            
            # Scenario 1: High-frequency trading simulation
            client = EnhancedComprehensiveMetricsClient()
            
            start_time = time.time()
            operations = 0
            errors = 0
            
            # Simulate 1000 rapid operations
            for i in range(1000):
                try:
                    if i % 3 == 0:
                        client.get_indicator_info('^VIX')
                    elif i % 3 == 1:
                        client.negotiate_protocol_version('2.1.0')
                    else:
                        client.get_threshold_alerts({'^VIX': pd.DataFrame({'Close': [25]})})
                    
                    operations += 1
                    
                    # Micro delay to simulate realistic usage
                    await asyncio.sleep(0.001)
                    
                except Exception:
                    errors += 1
            
            total_time = time.time() - start_time
            ops_per_second = operations / total_time
            error_rate = errors / (operations + errors)
            
            if ops_per_second < 100:  # Should handle at least 100 ops/second
                self.add_warning(
                    "Low throughput for high-frequency scenario",
                    f"Only {ops_per_second:.1f} ops/second"
                )
            
            if error_rate > 0.01:  # Error rate should be under 1%
                self.add_critical_issue(
                    "High error rate in production scenario",
                    f"{error_rate*100:.2f}% error rate"
                )
            
            # Scenario 2: Memory pressure simulation
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create multiple clients and use them intensively
            clients = [EnhancedComprehensiveMetricsClient() for _ in range(20)]
            
            for client in clients:
                # Perform memory-intensive operations
                for _ in range(10):
                    client.get_all_indicators_info()  # This loads all indicator info
            
            memory_peak = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = memory_peak - memory_start
            
            if memory_growth > 1000:  # More than 1GB growth
                self.add_critical_issue(
                    "Excessive memory growth under load",
                    f"Memory grew by {memory_growth:.1f}MB"
                )
            
            # Cleanup
            del clients
            gc.collect()
            
            print("✅ Production scenario tests completed")
            
        except Exception as e:
            self.add_critical_issue("Production scenario test failed", str(e))
    
    def calculate_brutal_rating(self) -> int:
        """Calculate brutally honest rating"""
        base_score = 100
        
        # Critical issues - each one is a major deduction
        critical_penalty = len(self.critical_issues) * 15
        
        # Warnings - smaller deductions but still significant
        warning_penalty = len(self.warnings) * 3
        
        # Additional penalties for specific issue types
        security_issues = sum(1 for issue in self.critical_issues if 'security' in issue['issue'].lower() or 'vulnerability' in issue['issue'].lower())
        memory_issues = sum(1 for issue in self.critical_issues if 'memory' in issue['issue'].lower())
        performance_issues = sum(1 for issue in self.critical_issues if 'slow' in issue['issue'].lower() or 'performance' in issue['issue'].lower())
        
        # Security issues are extra critical
        security_penalty = security_issues * 10
        
        # Memory issues can kill production
        memory_penalty = memory_issues * 8
        
        # Performance issues affect user experience
        performance_penalty = performance_issues * 5
        
        final_score = base_score - critical_penalty - warning_penalty - security_penalty - memory_penalty - performance_penalty
        
        return max(0, final_score)


async def main():
    """Run the brutal evaluation"""
    print("🔥 PRODUCTION READINESS EVALUATION - BRUTAL MODE")
    print("This evaluation will find REAL issues that could break production")
    print("Better to find them now than in production...")
    print()
    
    evaluator = CriticalProductionEvaluator()
    results = await evaluator.evaluate_critical_production_readiness()
    
    print("\n" + "="*80)
    print("📊 BRUTAL EVALUATION RESULTS")
    print("="*80)
    
    print(f"Overall Rating: {results['overall_rating']}/100")
    print(f"Production Ready: {'YES' if results['production_ready'] else 'NO'}")
    print(f"Critical Issues: {len(results['critical_issues'])}")
    print(f"Warnings: {len(results['warnings'])}")
    
    if results['critical_issues']:
        print(f"\n❌ CRITICAL ISSUES (Each costs ~15 points):")
        for i, issue in enumerate(results['critical_issues'], 1):
            print(f"{i}. {issue['issue']}")
            if issue['details']:
                print(f"   → {issue['details']}")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS (Each costs ~3 points):")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"{i}. {warning['warning']}")
            if warning['details']:
                print(f"   → {warning['details']}")
    
    print(f"\n🎯 HONEST ASSESSMENT:")
    
    if results['overall_rating'] >= 95:
        print("🥇 EXCELLENT - Ready for production with minor optimizations")
    elif results['overall_rating'] >= 85:
        print("🥈 GOOD - Production ready with some improvements needed")
    elif results['overall_rating'] >= 75:
        print("🥉 ACCEPTABLE - Can go to production but monitor closely")
    elif results['overall_rating'] >= 60:
        print("⚠️  NEEDS WORK - Significant issues must be fixed before production")
    else:
        print("❌ NOT READY - Too many critical issues for production")
    
    # Save results
    with open('brutal_evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: brutal_evaluation_results.json")
    
    return 0 if results['overall_rating'] >= 85 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))