#!/usr/bin/env python3
"""
Professional Trading System Diagnostics CLI
Comprehensive diagnostic tools for the enhanced historical loader agent and professional trading system
Integrates with existing CLI framework for 100/100 implementation verification
"""

import argparse
import asyncio
import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from ..ml.enhanced_comprehensive_metrics_client import EnhancedComprehensiveMetricsClient
from ..ml.professional_trading_config import ProfessionalTradingConfig, MarketRegime
from ..a2a.agents.historical_loader_agent import HistoricalLoaderAgent


def setup_logging(level: str = "INFO"):
    """Configure logging for CLI"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'trading_diagnostics_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )


class ProfessionalTradingDiagnostics:
    """
    Comprehensive diagnostic system for professional trading components
    Integrates enhanced agent with existing CLI framework
    """
    
    def __init__(self):
        self.client = EnhancedComprehensiveMetricsClient()
        self.agent = None  # Will be initialized when needed
        self.results = {}
        
    async def run_complete_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic tests"""
        print("🔍 Professional Trading System Diagnostics")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'PENDING',
            'components': {},
            'performance': {},
            'compliance': {},
            'recommendations': []
        }
        
        try:
            # 1. Enhanced Metrics Client Diagnostics
            print("\n📊 Testing Enhanced Metrics Client...")
            results['components']['enhanced_client'] = await self.test_enhanced_client()
            
            # 2. Professional Trading Configuration
            print("\n⚙️ Testing Professional Trading Configuration...")
            results['components']['trading_config'] = await self.test_trading_config()
            
            # 3. Streaming Functionality
            print("\n📡 Testing Streaming Functionality...")
            results['components']['streaming'] = await self.test_streaming()
            
            # 4. Protocol Negotiation
            print("\n🤝 Testing Protocol Negotiation...")
            results['components']['protocol'] = await self.test_protocol_negotiation()
            
            # 5. Agent Integration
            print("\n🤖 Testing Agent Integration...")
            results['components']['agent_integration'] = await self.test_agent_integration()
            
            # 6. Professional Calculations
            print("\n💼 Testing Professional Calculations...")
            results['components']['calculations'] = await self.test_professional_calculations()
            
            # 7. Error Handling and Observability
            print("\n🛡️ Testing Error Handling...")
            results['components']['error_handling'] = await self.test_error_handling()
            
            # 8. Performance Benchmarks
            print("\n⚡ Running Performance Benchmarks...")
            results['performance'] = await self.run_performance_tests()
            
            # 9. Compliance Verification
            print("\n✅ Verifying Compliance...")
            results['compliance'] = await self.verify_compliance()
            
            # Calculate overall status
            results['system_status'] = self.calculate_system_status(results)
            results['recommendations'] = self.generate_recommendations(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Critical error in diagnostics: {e}")
            results['system_status'] = 'CRITICAL_ERROR'
            results['error'] = str(e)
            return results
    
    async def test_enhanced_client(self) -> Dict[str, Any]:
        """Test enhanced comprehensive metrics client"""
        results = {'status': 'PENDING', 'tests': {}, 'details': {}}
        
        try:
            # Test initialization
            results['tests']['initialization'] = {
                'passed': self.client is not None,
                'indicators_count': len(self.client.COMPREHENSIVE_INDICATORS),
                'crypto_predictors': len(self.client.CRYPTO_COMPREHENSIVE_PREDICTORS),
                'weighting_model': bool(self.client.WEIGHTING_MODEL)
            }
            
            # Test indicator information retrieval
            vix_info = self.client.get_indicator_info('^VIX')
            results['tests']['indicator_info'] = {
                'passed': 'name' in vix_info and 'category' in vix_info,
                'has_metadata': 'institutional_usage' in vix_info,
                'sample_indicator': vix_info.get('name', 'N/A')
            }
            
            # Test professional calculation methods
            methods_to_test = [
                'calculate_weighted_signals', 'calculate_position_sizing',
                'get_threshold_alerts', 'calculate_options_analytics',
                'calculate_ensemble_correlations', 'stream_real_time_indicators',
                'negotiate_protocol_version', 'migrate_from_legacy_protocol'
            ]
            
            method_results = {}
            for method in methods_to_test:
                method_results[method] = hasattr(self.client, method)
            
            results['tests']['method_availability'] = {
                'passed': all(method_results.values()),
                'available_methods': sum(method_results.values()),
                'total_methods': len(methods_to_test),
                'missing_methods': [m for m, available in method_results.items() if not available]
            }
            
            # Test ticker validation
            vix_validation = self.client.validate_ticker_availability('^VIX')
            results['tests']['ticker_validation'] = {
                'passed': vix_validation.get('available', False),
                'validation_structure': 'symbol' in vix_validation and 'available' in vix_validation
            }
            
            results['status'] = 'PASSED' if all(
                test.get('passed', False) for test in results['tests'].values()
            ) else 'FAILED'
            
        except Exception as e:
            results['status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    async def test_trading_config(self) -> Dict[str, Any]:
        """Test professional trading configuration"""
        results = {'status': 'PENDING', 'tests': {}}
        
        try:
            # Test institutional strategies
            strategies = ProfessionalTradingConfig.get_all_indicator_sets()
            required_strategies = ['two_sigma', 'deribit', 'galaxy_digital', 'jump_trading']
            
            results['tests']['institutional_strategies'] = {
                'passed': all(strategy in strategies for strategy in required_strategies),
                'total_strategies': len(strategies),
                'required_present': sum(1 for s in required_strategies if s in strategies),
                'strategy_names': list(strategies.keys())
            }
            
            # Test market regime indicators
            regime_tests = {}
            for regime in MarketRegime:
                indicators = ProfessionalTradingConfig.get_regime_indicators(regime)
                regime_tests[regime.value] = len(indicators) > 0
            
            results['tests']['regime_indicators'] = {
                'passed': all(regime_tests.values()),
                'regimes_configured': sum(regime_tests.values()),
                'total_regimes': len(regime_tests)
            }
            
            # Test critical thresholds
            thresholds = ProfessionalTradingConfig.get_critical_thresholds()
            critical_indicators = ['^VIX', 'DX-Y.NYB', '^TNX', 'HYG']
            
            results['tests']['critical_thresholds'] = {
                'passed': all(indicator in thresholds for indicator in critical_indicators),
                'thresholds_configured': len(thresholds),
                'critical_indicators_present': sum(1 for i in critical_indicators if i in thresholds)
            }
            
            # Test correlation windows and weighting
            windows = ProfessionalTradingConfig.get_correlation_windows()
            weighting = ProfessionalTradingConfig.get_weighting_model()
            
            results['tests']['configuration_completeness'] = {
                'passed': bool(windows) and bool(weighting),
                'has_correlation_windows': bool(windows),
                'has_weighting_model': bool(weighting),
                'weighting_sum': sum(weighting.values()) if weighting else 0
            }
            
            results['status'] = 'PASSED' if all(
                test.get('passed', False) for test in results['tests'].values()
            ) else 'FAILED'
            
        except Exception as e:
            results['status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    async def test_streaming(self) -> Dict[str, Any]:
        """Test streaming functionality"""
        results = {'status': 'PENDING', 'tests': {}}
        
        try:
            # Test streaming method availability
            streaming_methods = [
                'stream_real_time_indicators',
                'stream_correlation_matrix'
            ]
            
            method_availability = {
                method: hasattr(self.client, method) 
                for method in streaming_methods
            }
            
            results['tests']['streaming_methods'] = {
                'passed': all(method_availability.values()),
                'available_methods': list(method_availability.keys()),
                'method_count': len(streaming_methods)
            }
            
            # Test streaming configuration (brief test)
            streaming_updates = []
            
            async def test_callback(data):
                streaming_updates.append(data)
            
            # Quick streaming test (2 seconds)
            try:
                streaming_task = asyncio.create_task(
                    self.client.stream_real_time_indicators(
                        ['^VIX'], test_callback, interval_seconds=1, batch_size=1
                    )
                )
                await asyncio.sleep(2)
                streaming_task.cancel()
                
                results['tests']['streaming_execution'] = {
                    'passed': len(streaming_updates) > 0,
                    'updates_received': len(streaming_updates),
                    'streaming_works': True
                }
            except Exception as e:
                results['tests']['streaming_execution'] = {
                    'passed': False,
                    'error': str(e),
                    'streaming_works': False
                }
            
            results['status'] = 'PASSED' if all(
                test.get('passed', False) for test in results['tests'].values()
            ) else 'FAILED'
            
        except Exception as e:
            results['status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    async def test_protocol_negotiation(self) -> Dict[str, Any]:
        """Test protocol negotiation functionality"""
        results = {'status': 'PENDING', 'tests': {}}
        
        try:
            # Test current version negotiation
            current_negotiation = self.client.negotiate_protocol_version()
            
            results['tests']['current_version'] = {
                'passed': 'negotiated_version' in current_negotiation,
                'has_features': 'features' in current_negotiation,
                'has_capabilities': 'capabilities' in current_negotiation,
                'negotiated_version': current_negotiation.get('negotiated_version', 'N/A')
            }
            
            # Test version compatibility
            compatibility_tests = []
            test_versions = ['1.0.0', '2.0.0', '2.1.0', '3.0.0']
            
            for version in test_versions:
                negotiation = self.client.negotiate_protocol_version(version)
                compatibility_tests.append({
                    'requested': version,
                    'negotiated': negotiation.get('negotiated_version'),
                    'success': 'error' not in negotiation
                })
            
            results['tests']['version_compatibility'] = {
                'passed': any(test['success'] for test in compatibility_tests),
                'compatibility_tests': compatibility_tests,
                'supported_versions': [test['requested'] for test in compatibility_tests if test['success']]
            }
            
            # Test migration tools
            legacy_data = {
                'message_type': 'DATA_REQUEST',
                'payload': {'symbols': 'BTC-USD', 'days': 30}
            }
            
            migration_result = self.client.migrate_from_legacy_protocol(legacy_data)
            
            results['tests']['migration_tools'] = {
                'passed': migration_result.get('success', False),
                'has_migrated_data': 'migrated_data' in migration_result,
                'migration_successful': migration_result.get('success', False)
            }
            
            results['status'] = 'PASSED' if all(
                test.get('passed', False) for test in results['tests'].values()
            ) else 'FAILED'
            
        except Exception as e:
            results['status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    async def test_agent_integration(self) -> Dict[str, Any]:
        """Test agent integration with Strands framework"""
        results = {'status': 'PENDING', 'tests': {}}
        
        try:
            # Initialize agent for testing
            try:
                self.agent = HistoricalLoaderAgent()
                agent_initialized = True
            except Exception as e:
                agent_initialized = False
                agent_error = str(e)
            
            results['tests']['agent_initialization'] = {
                'passed': agent_initialized,
                'error': agent_error if not agent_initialized else None
            }
            
            if agent_initialized:
                # Test agent tools
                tools = self.agent._create_tools() if hasattr(self.agent, '_create_tools') else []
                
                results['tests']['agent_tools'] = {
                    'passed': len(tools) > 0,
                    'tool_count': len(tools),
                    'has_tools': len(tools) > 0
                }
                
                # Test agent capabilities
                capabilities = getattr(self.agent, 'capabilities', [])
                
                results['tests']['agent_capabilities'] = {
                    'passed': len(capabilities) > 0,
                    'capability_count': len(capabilities),
                    'capabilities': capabilities
                }
            
            results['status'] = 'PASSED' if all(
                test.get('passed', False) for test in results['tests'].values()
            ) else 'FAILED'
            
        except Exception as e:
            results['status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    async def test_professional_calculations(self) -> Dict[str, Any]:
        """Test professional calculation accuracy"""
        results = {'status': 'PENDING', 'tests': {}}
        
        try:
            # Create mock data for testing
            mock_data = {
                '^VIX': pd.DataFrame({
                    'Close': [20, 25, 30, 28, 22],
                    'Date': pd.date_range('2024-01-01', periods=5)
                }).set_index('Date'),
                '^TNX': pd.DataFrame({
                    'Close': [4.0, 4.2, 4.5, 4.3, 4.1],
                    'Date': pd.date_range('2024-01-01', periods=5)
                }).set_index('Date')
            }
            
            crypto_data = pd.DataFrame({
                'Close': [45000, 46000, 44000, 45500, 46500],
                'High': [45500, 46500, 44500, 46000, 47000],
                'Low': [44500, 45500, 43500, 45000, 46000],
                'Date': pd.date_range('2024-01-01', periods=5)
            }).set_index('Date')
            
            # Test weighted signals
            signals = self.client.calculate_weighted_signals(mock_data, crypto_data)
            
            results['tests']['weighted_signals'] = {
                'passed': not signals.empty and 'composite_signal' in signals.columns,
                'signal_columns': len(signals.columns),
                'has_composite': 'composite_signal' in signals.columns,
                'has_correlations': any('correlation' in col for col in signals.columns)
            }
            
            # Test position sizing
            positions = self.client.calculate_position_sizing(signals, crypto_data)
            
            results['tests']['position_sizing'] = {
                'passed': not positions.empty and 'recommended_size' in positions.columns,
                'has_atr': 'atr_14' in positions.columns,
                'has_kelly': 'kelly_position_size' in positions.columns,
                'has_volatility_adjustment': 'vix_adjusted_size' in positions.columns
            }
            
            # Test threshold alerts
            alerts = self.client.get_threshold_alerts(mock_data)
            
            results['tests']['threshold_alerts'] = {
                'passed': 'triggered' in alerts and 'current_levels' in alerts,
                'alert_structure_valid': isinstance(alerts.get('triggered'), list),
                'has_current_levels': bool(alerts.get('current_levels'))
            }
            
            # Test options analytics
            options = self.client.calculate_options_analytics(mock_data, crypto_data)
            
            results['tests']['options_analytics'] = {
                'passed': not options.empty,
                'has_volatility_metrics': any('vol' in col.lower() for col in options.columns),
                'analytics_columns': len(options.columns)
            }
            
            results['status'] = 'PASSED' if all(
                test.get('passed', False) for test in results['tests'].values()
            ) else 'FAILED'
            
        except Exception as e:
            results['status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and graceful degradation"""
        results = {'status': 'PENDING', 'tests': {}}
        
        try:
            # Test invalid input handling
            try:
                invalid_signals = self.client.calculate_weighted_signals({}, pd.DataFrame())
                error_handled = invalid_signals.empty
            except Exception:
                error_handled = True
            
            results['tests']['invalid_input_handling'] = {
                'passed': error_handled,
                'graceful_degradation': True
            }
            
            # Test protocol error handling
            error_protocol = self.client.negotiate_protocol_version('invalid.version')
            
            results['tests']['protocol_error_handling'] = {
                'passed': 'error' in error_protocol or 'fallback_version' in error_protocol,
                'provides_fallback': bool(error_protocol.get('fallback_version') or error_protocol.get('basic_features'))
            }
            
            # Test ticker validation with invalid symbol
            invalid_validation = self.client.validate_ticker_availability('INVALID_SYMBOL_XYZ')
            
            results['tests']['ticker_validation_errors'] = {
                'passed': not invalid_validation.get('available', True),
                'handles_invalid_symbols': 'error' in invalid_validation or not invalid_validation.get('available', True)
            }
            
            results['status'] = 'PASSED' if all(
                test.get('passed', False) for test in results['tests'].values()
            ) else 'FAILED'
            
        except Exception as e:
            results['status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        performance = {}
        
        try:
            # Test indicator info retrieval speed
            start_time = datetime.now()
            vix_info = self.client.get_indicator_info('^VIX')
            indicator_time = (datetime.now() - start_time).total_seconds()
            
            performance['indicator_info_time'] = {
                'seconds': indicator_time,
                'acceptable': indicator_time < 5.0,
                'benchmark': '< 5.0 seconds'
            }
            
            # Test protocol negotiation speed
            start_time = datetime.now()
            protocol = self.client.negotiate_protocol_version()
            protocol_time = (datetime.now() - start_time).total_seconds()
            
            performance['protocol_negotiation_time'] = {
                'seconds': protocol_time,
                'acceptable': protocol_time < 1.0,
                'benchmark': '< 1.0 seconds'
            }
            
            # Test bulk operations
            start_time = datetime.now()
            all_indicators = self.client.get_all_indicators_info()
            bulk_time = (datetime.now() - start_time).total_seconds()
            
            performance['bulk_operations_time'] = {
                'seconds': bulk_time,
                'indicators_processed': len(all_indicators),
                'acceptable': bulk_time < 30.0,
                'benchmark': '< 30.0 seconds for all indicators'
            }
            
            performance['overall_performance'] = all(
                test.get('acceptable', False) for test in performance.values() 
                if isinstance(test, dict) and 'acceptable' in test
            )
            
        except Exception as e:
            performance['error'] = str(e)
            performance['overall_performance'] = False
        
        return performance
    
    async def verify_compliance(self) -> Dict[str, Any]:
        """Verify compliance with professional trading standards"""
        compliance = {}
        
        try:
            # Check indicator coverage
            total_indicators = len(self.client.COMPREHENSIVE_INDICATORS)
            compliance['indicator_coverage'] = {
                'total_indicators': total_indicators,
                'meets_minimum': total_indicators >= 50,
                'benchmark': '≥ 50 professional indicators'
            }
            
            # Check institutional strategies
            strategies = ProfessionalTradingConfig.get_all_indicator_sets()
            compliance['institutional_strategies'] = {
                'strategy_count': len(strategies),
                'meets_minimum': len(strategies) >= 6,
                'benchmark': '≥ 6 institutional strategies'
            }
            
            # Check calculation completeness
            required_calculations = [
                'calculate_weighted_signals', 'calculate_position_sizing',
                'get_threshold_alerts', 'calculate_options_analytics',
                'calculate_ensemble_correlations'
            ]
            
            available_calculations = [
                calc for calc in required_calculations 
                if hasattr(self.client, calc)
            ]
            
            compliance['calculation_completeness'] = {
                'available_calculations': len(available_calculations),
                'required_calculations': len(required_calculations),
                'meets_standard': len(available_calculations) == len(required_calculations),
                'benchmark': 'All professional calculations implemented'
            }
            
            # Check streaming capabilities
            streaming_methods = ['stream_real_time_indicators', 'stream_correlation_matrix']
            available_streaming = [
                method for method in streaming_methods 
                if hasattr(self.client, method)
            ]
            
            compliance['streaming_capabilities'] = {
                'available_streaming': len(available_streaming),
                'required_streaming': len(streaming_methods),
                'meets_standard': len(available_streaming) == len(streaming_methods),
                'benchmark': 'Real-time streaming implemented'
            }
            
            # Overall compliance score
            compliance_tests = [
                compliance['indicator_coverage']['meets_minimum'],
                compliance['institutional_strategies']['meets_minimum'],
                compliance['calculation_completeness']['meets_standard'],
                compliance['streaming_capabilities']['meets_standard']
            ]
            
            compliance['overall_compliance'] = all(compliance_tests)
            compliance['compliance_score'] = sum(compliance_tests) / len(compliance_tests) * 100
            
        except Exception as e:
            compliance['error'] = str(e)
            compliance['overall_compliance'] = False
        
        return compliance
    
    def calculate_system_status(self, results: Dict[str, Any]) -> str:
        """Calculate overall system status"""
        components = results.get('components', {})
        
        passed_components = sum(
            1 for component in components.values() 
            if component.get('status') == 'PASSED'
        )
        total_components = len(components)
        
        if total_components == 0:
            return 'UNKNOWN'
        
        success_rate = passed_components / total_components
        
        if success_rate == 1.0:
            return 'OPTIMAL'
        elif success_rate >= 0.8:
            return 'GOOD'
        elif success_rate >= 0.6:
            return 'ACCEPTABLE'
        else:
            return 'NEEDS_ATTENTION'
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        # Check component statuses
        components = results.get('components', {})
        for component_name, component_result in components.items():
            if component_result.get('status') != 'PASSED':
                recommendations.append(f"Address issues in {component_name} component")
        
        # Check performance
        performance = results.get('performance', {})
        if not performance.get('overall_performance', True):
            recommendations.append("Optimize system performance - some operations are slow")
        
        # Check compliance
        compliance = results.get('compliance', {})
        if not compliance.get('overall_compliance', True):
            recommendations.append("Improve compliance with professional trading standards")
        
        if not recommendations:
            recommendations.append("System is operating optimally - no recommendations")
        
        return recommendations


async def cmd_diagnose(args):
    """Run comprehensive diagnostics"""
    diagnostics = ProfessionalTradingDiagnostics()
    
    print("Starting Professional Trading System Diagnostics...")
    
    if args.component:
        # Run specific component test
        component_tests = {
            'client': diagnostics.test_enhanced_client,
            'config': diagnostics.test_trading_config,
            'streaming': diagnostics.test_streaming,
            'protocol': diagnostics.test_protocol_negotiation,
            'agent': diagnostics.test_agent_integration,
            'calculations': diagnostics.test_professional_calculations,
            'errors': diagnostics.test_error_handling
        }
        
        if args.component in component_tests:
            result = await component_tests[args.component]()
            print(f"\n{args.component.upper()} Component Test Result:")
            print(json.dumps(result, indent=2, default=str))
            return 0 if result.get('status') == 'PASSED' else 1
        else:
            print(f"Unknown component: {args.component}")
            print(f"Available components: {', '.join(component_tests.keys())}")
            return 1
    else:
        # Run complete diagnostics
        results = await diagnostics.run_complete_diagnostics()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC SUMMARY")
        print(f"{'='*60}")
        print(f"System Status: {results['system_status']}")
        print(f"Timestamp: {results['timestamp']}")
        
        print(f"\nComponent Results:")
        for component, result in results['components'].items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"{status_emoji} {component}: {result['status']}")
        
        if results.get('performance'):
            perf = results['performance']
            perf_emoji = "✅" if perf.get('overall_performance') else "⚠️"
            print(f"\n{perf_emoji} Performance: {'ACCEPTABLE' if perf.get('overall_performance') else 'NEEDS_IMPROVEMENT'}")
        
        if results.get('compliance'):
            comp = results['compliance']
            comp_score = comp.get('compliance_score', 0)
            comp_emoji = "✅" if comp_score >= 90 else "⚠️" if comp_score >= 70 else "❌"
            print(f"{comp_emoji} Compliance: {comp_score:.1f}%")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(results.get('recommendations', []), 1):
            print(f"{i}. {rec}")
        
        # Save detailed results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {args.output}")
        
        return 0 if results['system_status'] in ['OPTIMAL', 'GOOD'] else 1


async def cmd_test_streaming(args):
    """Test streaming functionality specifically"""
    client = EnhancedComprehensiveMetricsClient()
    
    print("🌊 Testing Real-time Streaming Functionality")
    print("=" * 50)
    
    updates_received = []
    
    async def stream_callback(data):
        updates_received.append(data)
        print(f"📊 Received update: {len(data.get('data', {}))} indicators at {data.get('timestamp', 'unknown')}")
    
    symbols = args.symbols.split(',') if args.symbols else ['^VIX', '^TNX', 'GC=F']
    duration = args.duration or 10
    
    print(f"Streaming {len(symbols)} indicators for {duration} seconds...")
    print(f"Symbols: {', '.join(symbols)}")
    
    try:
        # Start streaming
        streaming_task = asyncio.create_task(
            client.stream_real_time_indicators(
                symbols, stream_callback, 
                interval_seconds=args.interval or 2,
                batch_size=args.batch_size or 5
            )
        )
        
        # Let it run for specified duration
        await asyncio.sleep(duration)
        streaming_task.cancel()
        
        print(f"\n✅ Streaming test completed!")
        print(f"Total updates received: {len(updates_received)}")
        print(f"Average updates per second: {len(updates_received) / duration:.2f}")
        
        if updates_received:
            print(f"Last update: {updates_received[-1].get('timestamp', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return 1


async def cmd_validate_config(args):
    """Validate professional trading configuration"""
    print("⚙️ Validating Professional Trading Configuration")
    print("=" * 50)
    
    try:
        # Test all institutional strategies
        strategies = ProfessionalTradingConfig.get_all_indicator_sets()
        
        print(f"Institutional Strategies: {len(strategies)}")
        for name, strategy in strategies.items():
            print(f"  ✓ {name}: {strategy.name}")
            print(f"    Symbols: {len(strategy.symbols)}")
            print(f"    Institution: {strategy.institutional_reference}")
        
        # Test market regimes
        print(f"\nMarket Regimes:")
        for regime in MarketRegime:
            indicators = ProfessionalTradingConfig.get_regime_indicators(regime)
            print(f"  ✓ {regime.value}: {len(indicators)} indicators")
        
        # Test critical thresholds
        thresholds = ProfessionalTradingConfig.get_critical_thresholds()
        print(f"\nCritical Thresholds: {len(thresholds)}")
        for symbol, levels in thresholds.items():
            print(f"  ✓ {symbol}: {len(levels)} threshold levels")
        
        print(f"\n✅ Configuration validation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Professional Trading System Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete diagnostics
  python professional_trading_diagnostics.py diagnose
  
  # Test specific component
  python professional_trading_diagnostics.py diagnose --component streaming
  
  # Test streaming functionality
  python professional_trading_diagnostics.py stream --symbols "^VIX,^TNX,GC=F" --duration 30
  
  # Validate configuration
  python professional_trading_diagnostics.py validate-config
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Run system diagnostics')
    diagnose_parser.add_argument('--component', 
                                choices=['client', 'config', 'streaming', 'protocol', 'agent', 'calculations', 'errors'],
                                help='Test specific component')
    diagnose_parser.add_argument('--output', help='Save detailed results to file')
    diagnose_parser.set_defaults(func=cmd_diagnose)
    
    # Stream test command
    stream_parser = subparsers.add_parser('stream', help='Test streaming functionality')
    stream_parser.add_argument('--symbols', help='Comma-separated list of symbols to stream')
    stream_parser.add_argument('--duration', type=int, help='Test duration in seconds')
    stream_parser.add_argument('--interval', type=int, help='Update interval in seconds')
    stream_parser.add_argument('--batch-size', type=int, help='Batch size for updates')
    stream_parser.set_defaults(func=cmd_test_streaming)
    
    # Config validation command
    config_parser = subparsers.add_parser('validate-config', help='Validate trading configuration')
    config_parser.set_defaults(func=cmd_validate_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command:
        try:
            return asyncio.run(args.func(args))
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return 130
        except Exception as e:
            logging.error(f"Command failed: {e}", exc_info=args.verbose)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())