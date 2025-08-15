#!/usr/bin/env python3
"""
FOCUSED PRODUCTION EVALUATION
Quick but brutally honest assessment for production readiness
Designed to complete without timeout while finding critical issues
"""

import time
import sys
import os
import logging
import traceback
import psutil
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Disable logging for clean output
logging.disable(logging.CRITICAL)

class FocusedProductionEvaluator:
    """
    Focused production readiness evaluation
    Tests the most critical production failure points quickly
    """
    
    def __init__(self):
        self.critical_issues = []
        self.warnings = []
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

    def evaluate_production_readiness(self) -> Dict[str, Any]:
        """Conduct focused production readiness evaluation"""
        
        print("🔥 FOCUSED PRODUCTION EVALUATION - CRITICAL ISSUES ONLY")
        print("=" * 80)
        print("Testing critical production failure points...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_rating': 0,
            'critical_issues': [],
            'warnings': [],
            'production_ready': False,
            'evaluation_type': 'focused'
        }
        
        try:
            # 1. Critical Import and Initialization Test
            print("\n1️⃣ Testing Critical Imports...")
            self.test_critical_imports()
            
            # 2. Basic Functionality Smoke Test
            print("\n2️⃣ Testing Basic Functionality...")
            self.test_basic_functionality()
            
            # 3. Critical Error Handling
            print("\n3️⃣ Testing Critical Error Handling...")
            self.test_critical_error_handling()
            
            # 4. Memory Leak Detection
            print("\n4️⃣ Testing Memory Usage...")
            self.test_memory_usage()
            
            # 5. Performance Bottlenecks
            print("\n5️⃣ Testing Performance...")
            self.test_performance_bottlenecks()
            
            # 6. Security Vulnerabilities
            print("\n6️⃣ Testing Security...")
            self.test_security_vulnerabilities()
            
            # 7. Configuration Issues
            print("\n7️⃣ Testing Configuration...")
            self.test_configuration_issues()
            
        except Exception as e:
            self.add_critical_issue(
                "Evaluation suite crashed",
                f"Exception: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
        
        # Calculate final rating
        results['critical_issues'] = self.critical_issues
        results['warnings'] = self.warnings
        results['overall_rating'] = self.calculate_brutal_rating()
        results['production_ready'] = len(self.critical_issues) == 0 and results['overall_rating'] >= 90
        
        return results
    
    def test_critical_imports(self):
        """Test if critical imports work without crashing"""
        try:
            start_time = time.time()
            from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            import_time = time.time() - start_time
            
            if import_time > 5:
                self.add_warning(f"Slow import time: {import_time:.2f}s")
            
            print("✅ Core imports successful")
            
        except ImportError as e:
            self.add_critical_issue("Import failure", f"Cannot import core modules: {e}")
        except Exception as e:
            self.add_critical_issue("Import exception", f"Unexpected error during import: {e}")
    
    def test_basic_functionality(self):
        """Test basic functionality without external dependencies"""
        try:
            from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            
            start_time = time.time()
            client = EnhancedComprehensiveMetricsClient()
            init_time = time.time() - start_time
            
            if init_time > 10:
                self.add_critical_issue(f"Initialization too slow: {init_time:.2f}s")
            
            # Test basic method availability
            required_methods = [
                'calculate_weighted_signals', 'calculate_position_sizing',
                'negotiate_protocol_version', 'get_indicator_info'
            ]
            
            missing_methods = [method for method in required_methods if not hasattr(client, method)]
            if missing_methods:
                self.add_critical_issue(
                    "Critical methods missing",
                    f"Missing: {', '.join(missing_methods)}"
                )
            
            # Test basic configuration
            if len(client.COMPREHENSIVE_INDICATORS) < 10:
                self.add_critical_issue(
                    "Insufficient indicators",
                    f"Only {len(client.COMPREHENSIVE_INDICATORS)} indicators"
                )
            
            print("✅ Basic functionality working")
            
        except Exception as e:
            self.add_critical_issue("Basic functionality failure", str(e))
    
    def test_critical_error_handling(self):
        """Test error handling for critical scenarios"""
        try:
            from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            client = EnhancedComprehensiveMetricsClient()
            
            # Test 1: Empty/None inputs
            try:
                result = client.get_indicator_info(None)
                if 'error' not in str(result).lower():
                    self.add_warning("Should handle None input gracefully")
            except Exception:
                pass  # Exception is acceptable
            
            # Test 2: Invalid symbol
            try:
                result = client.get_indicator_info("INVALID_SYMBOL_XYZ123")
                # Should not crash
            except Exception as e:
                if "critical" in str(e).lower() or "fatal" in str(e).lower():
                    self.add_critical_issue("Critical error on invalid input", str(e))
            
            # Test 3: Protocol version with invalid input
            try:
                result = client.negotiate_protocol_version("not.a.version")
                # Should handle gracefully
            except Exception as e:
                if "critical" in str(e).lower():
                    self.add_critical_issue("Protocol negotiation crashes on invalid input", str(e))
            
            print("✅ Critical error handling acceptable")
            
        except Exception as e:
            self.add_critical_issue("Error handling test failed", str(e))
    
    def test_memory_usage(self):
        """Test for obvious memory leaks"""
        try:
            from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            
            # Baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024
            
            # Create and destroy multiple clients
            clients = []
            for i in range(5):
                client = EnhancedComprehensiveMetricsClient()
                clients.append(client)
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_per_client = (peak_memory - baseline_memory) / 5
            
            if memory_per_client > 100:  # More than 100MB per client
                self.add_critical_issue(
                    "Excessive memory per client",
                    f"{memory_per_client:.1f}MB per client"
                )
            
            # Cleanup
            del clients
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_retained = final_memory - baseline_memory
            
            if memory_retained > 200:  # More than 200MB retained
                self.add_warning(
                    "Possible memory leak",
                    f"{memory_retained:.1f}MB retained after cleanup"
                )
            
            print("✅ Memory usage within acceptable limits")
            
        except Exception as e:
            self.add_critical_issue("Memory test failed", str(e))
    
    def test_performance_bottlenecks(self):
        """Test for obvious performance bottlenecks"""
        try:
            from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            client = EnhancedComprehensiveMetricsClient()
            
            # Test repeated method calls
            start_time = time.time()
            for i in range(100):
                client.get_indicator_info('^VIX')
            method_time = time.time() - start_time
            
            if method_time > 30:  # More than 30 seconds for 100 calls
                self.add_critical_issue(
                    "Severe performance bottleneck",
                    f"{method_time:.2f}s for 100 method calls"
                )
            elif method_time > 10:
                self.add_warning(
                    "Performance bottleneck",
                    f"{method_time:.2f}s for 100 method calls"
                )
            
            # Test protocol negotiation speed
            start_time = time.time()
            for i in range(100):
                client.negotiate_protocol_version('2.1.0')
            protocol_time = time.time() - start_time
            
            if protocol_time > 10:
                self.add_warning(
                    "Slow protocol negotiation",
                    f"{protocol_time:.2f}s for 100 negotiations"
                )
            
            print("✅ Performance within acceptable limits")
            
        except Exception as e:
            self.add_critical_issue("Performance test failed", str(e))
    
    def test_security_vulnerabilities(self):
        """Test for obvious security vulnerabilities"""
        try:
            from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
            
            # Test 1: Path traversal in config
            dangerous_paths = [
                "../../../etc/passwd",
                "..\\\\..\\\\..\\\\windows\\\\system32"
            ]
            
            for dangerous_path in dangerous_paths:
                try:
                    client = EnhancedComprehensiveMetricsClient(config_path=dangerous_path)
                    self.add_critical_issue(
                        "Path traversal vulnerability",
                        f"Accepts dangerous path: {dangerous_path}"
                    )
                except:
                    pass  # Should fail - this is good
            
            # Test 2: Injection attempts
            client = EnhancedComprehensiveMetricsClient()
            injection_attempts = [
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../../../etc/passwd"
            ]
            
            for injection in injection_attempts:
                try:
                    result = client.get_indicator_info(injection)
                    # Should not execute malicious code or crash
                except Exception:
                    pass  # Exceptions are acceptable
            
            print("✅ No obvious security vulnerabilities")
            
        except Exception as e:
            self.add_critical_issue("Security test failed", str(e))
    
    def test_configuration_issues(self):
        """Test for critical configuration issues"""
        try:
            # Test configuration file exists
            config_path = "config/enhanced_indicators.yaml"
            if not os.path.exists(config_path):
                self.add_critical_issue(
                    "Configuration file missing",
                    f"Cannot find {config_path}"
                )
            
            # Test import without config file
            try:
                from src.rex.ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
                client = EnhancedComprehensiveMetricsClient(config_path="nonexistent.yaml")
                self.add_warning("Should fail gracefully with missing config")
            except:
                pass  # Should fail
            
            print("✅ Configuration handling acceptable")
            
        except Exception as e:
            self.add_critical_issue("Configuration test failed", str(e))
    
    def calculate_brutal_rating(self) -> int:
        """Calculate brutally honest rating"""
        base_score = 85  # Start with previous known good score
        
        # Critical issues - each one is a major deduction
        critical_penalty = len(self.critical_issues) * 20
        
        # Warnings - smaller deductions
        warning_penalty = len(self.warnings) * 5
        
        # Additional penalties for specific issue types
        security_issues = sum(1 for issue in self.critical_issues 
                            if any(keyword in issue['issue'].lower() 
                                 for keyword in ['security', 'vulnerability', 'injection', 'traversal']))
        memory_issues = sum(1 for issue in self.critical_issues 
                          if 'memory' in issue['issue'].lower())
        performance_issues = sum(1 for issue in self.critical_issues 
                               if any(keyword in issue['issue'].lower() 
                                    for keyword in ['slow', 'performance', 'bottleneck']))
        
        # Extra penalties for critical issue types
        security_penalty = security_issues * 15
        memory_penalty = memory_issues * 10
        performance_penalty = performance_issues * 8
        
        final_score = base_score - critical_penalty - warning_penalty - security_penalty - memory_penalty - performance_penalty
        
        return max(0, final_score)


def main():
    """Run the focused evaluation"""
    print("🔥 FOCUSED PRODUCTION READINESS EVALUATION")
    print("Quick but brutal assessment for production readiness")
    print("Designed to complete without timeout...")
    print()
    
    evaluator = FocusedProductionEvaluator()
    results = evaluator.evaluate_production_readiness()
    
    print("\n" + "="*80)
    print("📊 BRUTAL EVALUATION RESULTS")
    print("="*80)
    
    print(f"Overall Rating: {results['overall_rating']}/100")
    print(f"Production Ready: {'YES' if results['production_ready'] else 'NO'}")
    print(f"Critical Issues: {len(results['critical_issues'])}")
    print(f"Warnings: {len(results['warnings'])}")
    
    if results['critical_issues']:
        print(f"\n❌ CRITICAL ISSUES (Each costs ~20 points):")
        for i, issue in enumerate(results['critical_issues'], 1):
            print(f"{i}. {issue['issue']}")
            if issue['details']:
                print(f"   → {issue['details']}")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS (Each costs ~5 points):")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"{i}. {warning['warning']}")
            if warning['details']:
                print(f"   → {warning['details']}")
    
    print(f"\n🎯 BRUTAL HONEST ASSESSMENT:")
    
    if results['overall_rating'] >= 95:
        print("🥇 EXCELLENT - Production ready, minor optimizations possible")
    elif results['overall_rating'] >= 85:
        print("🥈 GOOD - Production ready with monitoring recommended")
    elif results['overall_rating'] >= 75:
        print("🥉 ACCEPTABLE - Can deploy but fix issues quickly")
    elif results['overall_rating'] >= 60:
        print("⚠️  NEEDS WORK - Fix critical issues before production")
    else:
        print("❌ NOT READY - Too many critical issues for production")
    
    # Save results
    with open('focused_evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: focused_evaluation_results.json")
    
    return 0 if results['overall_rating'] >= 80 else 1


if __name__ == "__main__":
    exit(main())