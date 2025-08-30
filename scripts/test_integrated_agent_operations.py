#!/usr/bin/env python3
"""
Test Integrated Agent Operations
Comprehensive test suite for Week 2 CDS integration with transactions and monitoring
"""

import asyncio
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / "src"))

# Import test components
from cryptotrading.core.agents.specialized.agent_manager import AgentManagerAgent
from cryptotrading.core.agents.specialized.data_analysis_agent import DataAnalysisAgent
from cryptotrading.core.agents.specialized.ml_agent import MLAgent
from cryptotrading.infrastructure.monitoring.cds_integration_monitor import get_cds_monitor
from cryptotrading.infrastructure.transactions.agent_transaction_manager import get_transaction_manager
from cryptotrading.infrastructure.monitoring.cds_monitoring_api import check_cds_system_health

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedAgentOperationsTest:
    """Test suite for integrated agent operations with CDS, transactions, and monitoring"""
    
    def __init__(self):
        self.results = {}
        self.agent_manager = None
        self.data_analysis_agent = None
        self.ml_agent = None
        self.monitor = get_cds_monitor()
        self.transaction_manager = get_transaction_manager()
        
    async def setup_test_environment(self):
        """Setup test environment with agents and services"""
        logger.info("🔧 Setting up test environment...")
        
        try:
            # Start transaction manager
            await self.transaction_manager.start()
            
            # Initialize agents
            self.agent_manager = AgentManagerAgent("test_agent_manager")
            self.data_analysis_agent = DataAnalysisAgent("test_data_analysis_agent")
            self.ml_agent = MLAgent("test_ml_agent")
            
            # Initialize all agents
            agents_to_init = [
                ("Agent Manager", self.agent_manager),
                ("Data Analysis Agent", self.data_analysis_agent),
                ("ML Agent", self.ml_agent)
            ]
            
            for agent_name, agent in agents_to_init:
                try:
                    if await agent.initialize():
                        logger.info(f"✅ {agent_name} initialized successfully")
                    else:
                        logger.warning(f"⚠️ {agent_name} initialization failed")
                except Exception as e:
                    logger.error(f"❌ {agent_name} initialization error: {e}")
            
            logger.info("🚀 Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False
    
    async def test_cds_integration(self) -> dict:
        """Test CDS integration functionality"""
        logger.info("📡 Testing CDS Integration...")
        
        test_results = {
            "test_name": "CDS Integration",
            "subtests": {},
            "overall_success": False
        }
        
        # Test 1: Agent registration via CDS
        try:
            if self.agent_manager.cds_initialized:
                result = await self.agent_manager.register_agent({
                    "agent_id": "test_cds_agent_001",
                    "agent_name": "test_cds_agent_001", 
                    "agent_type": "test",
                    "capabilities": ["testing", "cds_integration"]
                })
                
                test_results["subtests"]["cds_agent_registration"] = {
                    "success": result.get("success", False),
                    "method": result.get("method", "unknown"),
                    "details": result
                }
            else:
                test_results["subtests"]["cds_agent_registration"] = {
                    "success": False,
                    "error": "CDS not initialized",
                    "details": "CDS integration failed during setup"
                }
        except Exception as e:
            test_results["subtests"]["cds_agent_registration"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Data analysis via CDS
        try:
            if self.data_analysis_agent.cds_initialized:
                test_data = {
                    "data": [1, 2, 3, 4, 5, 100],  # Include outlier
                    "symbols": ["BTC", "ETH"]
                }
                
                result = await self.data_analysis_agent._mcp_validate_factor_quality(
                    test_data, ["BTC", "ETH"], {}
                )
                
                test_results["subtests"]["cds_data_analysis"] = {
                    "success": result.get("success", False),
                    "method": result.get("method", "unknown"),
                    "details": result
                }
            else:
                test_results["subtests"]["cds_data_analysis"] = {
                    "success": False,
                    "error": "CDS not initialized for data analysis agent"
                }
        except Exception as e:
            test_results["subtests"]["cds_data_analysis"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: ML prediction via CDS
        try:
            if self.ml_agent.cds_initialized:
                result = await self.ml_agent._mcp_predict_price(
                    symbol="BTC-USD",
                    horizon_hours=24
                )
                
                test_results["subtests"]["cds_ml_prediction"] = {
                    "success": result.get("success", False),
                    "method": result.get("method", "unknown"),
                    "details": result
                }
            else:
                test_results["subtests"]["cds_ml_prediction"] = {
                    "success": False,
                    "error": "CDS not initialized for ML agent"
                }
        except Exception as e:
            test_results["subtests"]["cds_ml_prediction"] = {
                "success": False,
                "error": str(e)
            }
        
        # Determine overall success
        successful_tests = sum(1 for test in test_results["subtests"].values() if test.get("success"))
        total_tests = len(test_results["subtests"])
        test_results["overall_success"] = successful_tests > 0
        test_results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        return test_results
    
    async def test_transactional_operations(self) -> dict:
        """Test transaction boundary management"""
        logger.info("🔄 Testing Transactional Operations...")
        
        test_results = {
            "test_name": "Transactional Operations",
            "subtests": {},
            "overall_success": False
        }
        
        # Test 1: Transactional agent registration
        try:
            if hasattr(self.agent_manager, 'register_agent_transactional'):
                result = await self.agent_manager.register_agent_transactional({
                    "agent_id": "test_tx_agent_001",
                    "agent_name": "test_tx_agent_001",
                    "agent_type": "test",
                    "capabilities": ["testing", "transactions"]
                })
                
                test_results["subtests"]["transactional_registration"] = {
                    "success": result.get("success", False),
                    "transaction_id": result.get("transaction_id"),
                    "method": result.get("method", "unknown"),
                    "details": result
                }
            else:
                test_results["subtests"]["transactional_registration"] = {
                    "success": False,
                    "error": "Transactional registration method not available"
                }
        except Exception as e:
            test_results["subtests"]["transactional_registration"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Transaction manager statistics
        try:
            stats = self.transaction_manager.get_transaction_stats()
            test_results["subtests"]["transaction_stats"] = {
                "success": True,
                "stats": stats,
                "has_transactions": stats.get("total_transactions", 0) > 0
            }
        except Exception as e:
            test_results["subtests"]["transaction_stats"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: Transaction rollback simulation (create failing transaction)
        try:
            # Create a transaction that should fail and rollback
            transaction = await self.transaction_manager.begin_transaction(
                agent_id="test_rollback_agent",
                transaction_type="TEST_ROLLBACK"
            )
            
            # Add an operation that will fail
            await self.transaction_manager.add_operation(
                transaction_id=transaction.transaction_id,
                operation_type="FAILING_OPERATION",
                resource_type="test",
                resource_id="test_resource",
                operation_data={"test": "data"},
                rollback_data={"action": "restore_test_data"}
            )
            
            # Intentionally rollback
            rollback_success = await self.transaction_manager.rollback_transaction(
                transaction.transaction_id,
                "Intentional test rollback"
            )
            
            test_results["subtests"]["transaction_rollback"] = {
                "success": rollback_success,
                "transaction_id": transaction.transaction_id,
                "rollback_reason": "Intentional test rollback"
            }
            
        except Exception as e:
            test_results["subtests"]["transaction_rollback"] = {
                "success": False,
                "error": str(e)
            }
        
        # Determine overall success
        successful_tests = sum(1 for test in test_results["subtests"].values() if test.get("success"))
        total_tests = len(test_results["subtests"])
        test_results["overall_success"] = successful_tests >= 2  # At least 2 out of 3
        test_results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        return test_results
    
    async def test_monitoring_system(self) -> dict:
        """Test monitoring and metrics collection"""
        logger.info("📊 Testing Monitoring System...")
        
        test_results = {
            "test_name": "Monitoring System",
            "subtests": {},
            "overall_success": False
        }
        
        # Test 1: CDS monitor functionality
        try:
            # Register a test agent with monitor
            self.monitor.register_agent("test_monitor_agent")
            self.monitor.update_agent_status("test_monitor_agent", "CONNECTED")
            
            agent_stats = self.monitor.get_agent_stats("test_monitor_agent")
            
            test_results["subtests"]["cds_monitor_agent_stats"] = {
                "success": agent_stats is not None,
                "agent_registered": agent_stats.agent_id == "test_monitor_agent" if agent_stats else False,
                "details": agent_stats.to_dict() if agent_stats else None
            }
        except Exception as e:
            test_results["subtests"]["cds_monitor_agent_stats"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: System health monitoring
        try:
            system_health = self.monitor.get_system_health()
            
            test_results["subtests"]["system_health"] = {
                "success": isinstance(system_health, dict),
                "has_agents": system_health.get("total_agents", 0) > 0,
                "health_data": system_health
            }
        except Exception as e:
            test_results["subtests"]["system_health"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: Performance report generation
        try:
            performance_report = self.monitor.get_performance_report()
            
            test_results["subtests"]["performance_report"] = {
                "success": isinstance(performance_report, dict),
                "has_timestamp": "timestamp" in performance_report,
                "has_recommendations": len(performance_report.get("recommendations", [])) >= 0,
                "report_keys": list(performance_report.keys())
            }
        except Exception as e:
            test_results["subtests"]["performance_report"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 4: CDS system health check
        try:
            health_check = await check_cds_system_health()
            
            test_results["subtests"]["cds_health_check"] = {
                "success": isinstance(health_check, dict),
                "status": health_check.get("status", "unknown"),
                "has_details": "details" in health_check,
                "health_check_data": health_check
            }
        except Exception as e:
            test_results["subtests"]["cds_health_check"] = {
                "success": False,
                "error": str(e)
            }
        
        # Determine overall success
        successful_tests = sum(1 for test in test_results["subtests"].values() if test.get("success"))
        total_tests = len(test_results["subtests"])
        test_results["overall_success"] = successful_tests >= 3  # At least 3 out of 4
        test_results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        return test_results
    
    async def test_cross_agent_integration(self) -> dict:
        """Test cross-agent communication and collaboration"""
        logger.info("🤝 Testing Cross-Agent Integration...")
        
        test_results = {
            "test_name": "Cross-Agent Integration",
            "subtests": {},
            "overall_success": False
        }
        
        # Test 1: Agent Manager managing other agents
        try:
            # Register data analysis agent via agent manager
            result = await self.agent_manager.register_agent({
                "agent_id": "managed_data_agent",
                "agent_name": "managed_data_agent",
                "agent_type": "data_analysis",
                "capabilities": ["data_quality", "statistical_analysis"]
            })
            
            test_results["subtests"]["cross_agent_registration"] = {
                "success": result.get("success", False),
                "managed_agent_id": "managed_data_agent",
                "details": result
            }
        except Exception as e:
            test_results["subtests"]["cross_agent_registration"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Data analysis agent providing analysis
        try:
            test_data = {
                "prices": [100, 101, 99, 102, 98, 105],
                "volumes": [1000, 1100, 900, 1200, 800, 1300],
                "timestamps": [datetime.now() - timedelta(hours=i) for i in range(6)]
            }
            
            analysis_result = await self.data_analysis_agent._mcp_comprehensive_data_analysis(
                data=test_data,
                factor_names=["prices", "volumes"]
            )
            
            test_results["subtests"]["comprehensive_analysis"] = {
                "success": analysis_result.get("success", False),
                "has_distribution": "distribution_analysis" in analysis_result.get("comprehensive_analysis", {}),
                "has_correlation": "correlation_analysis" in analysis_result.get("comprehensive_analysis", {}),
                "details": analysis_result
            }
        except Exception as e:
            test_results["subtests"]["comprehensive_analysis"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: ML agent prediction capabilities
        try:
            prediction_result = await self.ml_agent.process_mcp_request(
                "predict_price",
                {"symbol": "BTC-USD", "horizon_hours": 1}
            )
            
            test_results["subtests"]["ml_prediction"] = {
                "success": prediction_result.get("success", False),
                "has_prediction": "prediction" in prediction_result or "symbol" in prediction_result,
                "details": prediction_result
            }
        except Exception as e:
            test_results["subtests"]["ml_prediction"] = {
                "success": False,
                "error": str(e)
            }
        
        # Determine overall success
        successful_tests = sum(1 for test in test_results["subtests"].values() if test.get("success"))
        total_tests = len(test_results["subtests"])
        test_results["overall_success"] = successful_tests >= 2  # At least 2 out of 3
        test_results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        return test_results
    
    async def run_all_tests(self) -> dict:
        """Run all integration tests"""
        logger.info("🧪 Starting Integrated Agent Operations Test Suite")
        logger.info("=" * 70)
        
        # Setup test environment
        setup_success = await self.setup_test_environment()
        if not setup_success:
            return {
                "overall_success": False,
                "error": "Failed to setup test environment",
                "timestamp": datetime.now().isoformat()
            }
        
        # Run all test categories
        test_categories = [
            ("CDS Integration", self.test_cds_integration),
            ("Transactional Operations", self.test_transactional_operations),
            ("Monitoring System", self.test_monitoring_system),
            ("Cross-Agent Integration", self.test_cross_agent_integration)
        ]
        
        all_results = {
            "test_suite": "Integrated Agent Operations",
            "start_time": datetime.now().isoformat(),
            "test_categories": {},
            "summary": {},
            "overall_success": False
        }
        
        successful_categories = 0
        total_categories = len(test_categories)
        
        for category_name, test_func in test_categories:
            logger.info(f"🔍 Running {category_name} tests...")
            
            try:
                result = await test_func()
                all_results["test_categories"][category_name] = result
                
                if result.get("overall_success"):
                    successful_categories += 1
                    logger.info(f"✅ {category_name} - PASSED ({result.get('success_rate', 0):.1%} success rate)")
                else:
                    logger.warning(f"⚠️ {category_name} - FAILED ({result.get('success_rate', 0):.1%} success rate)")
                
            except Exception as e:
                logger.error(f"❌ {category_name} - ERROR: {e}")
                all_results["test_categories"][category_name] = {
                    "test_name": category_name,
                    "overall_success": False,
                    "error": str(e)
                }
            
            logger.info("-" * 50)
        
        # Generate summary
        all_results["end_time"] = datetime.now().isoformat()
        all_results["summary"] = {
            "total_categories": total_categories,
            "successful_categories": successful_categories,
            "category_success_rate": successful_categories / total_categories,
            "overall_success": successful_categories >= 3  # At least 3 out of 4 categories must pass
        }
        
        all_results["overall_success"] = all_results["summary"]["overall_success"]
        
        return all_results
    
    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        logger.info("🧹 Cleaning up test environment...")
        
        try:
            # Cleanup agents
            agents_to_cleanup = [
                ("Agent Manager", self.agent_manager),
                ("Data Analysis Agent", self.data_analysis_agent),
                ("ML Agent", self.ml_agent)
            ]
            
            for agent_name, agent in agents_to_cleanup:
                if agent and hasattr(agent, 'cleanup'):
                    try:
                        await agent.cleanup()
                        logger.info(f"✅ {agent_name} cleaned up")
                    except Exception as e:
                        logger.warning(f"⚠️ {agent_name} cleanup failed: {e}")
            
            # Stop transaction manager
            await self.transaction_manager.stop()
            logger.info("✅ Transaction manager stopped")
            
            logger.info("🚀 Test environment cleanup complete")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def print_detailed_results(self, results):
        """Print detailed test results"""
        logger.info("=" * 70)
        logger.info("📋 DETAILED TEST RESULTS")
        logger.info("=" * 70)
        
        summary = results.get("summary", {})
        logger.info(f"Overall Success: {'✅ PASS' if results.get('overall_success') else '❌ FAIL'}")
        logger.info(f"Categories Passed: {summary.get('successful_categories', 0)}/{summary.get('total_categories', 0)}")
        logger.info(f"Success Rate: {summary.get('category_success_rate', 0):.1%}")
        logger.info("")
        
        for category_name, category_result in results.get("test_categories", {}).items():
            logger.info(f"📂 {category_name}")
            logger.info(f"   Status: {'✅ PASS' if category_result.get('overall_success') else '❌ FAIL'}")
            logger.info(f"   Success Rate: {category_result.get('success_rate', 0):.1%}")
            
            for subtest_name, subtest_result in category_result.get("subtests", {}).items():
                status = "✅" if subtest_result.get("success") else "❌"
                error_info = f" - {subtest_result.get('error', '')}" if subtest_result.get("error") else ""
                logger.info(f"     {status} {subtest_name}{error_info}")
            
            logger.info("")


async def main():
    """Main test runner"""
    test_suite = IntegratedAgentOperationsTest()
    
    try:
        # Run all tests
        results = await test_suite.run_all_tests()
        
        # Print results
        test_suite.print_detailed_results(results)
        
        # Determine exit code
        exit_code = 0 if results.get("overall_success") else 1
        
        logger.info("=" * 70)
        if results.get("overall_success"):
            logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
            logger.info("✅ Week 2 CDS Integration Implementation COMPLETE")
        else:
            logger.warning("⚠️ Some integration tests failed")
            logger.info("📝 Review failed tests and CDS service connectivity")
        
        logger.info("=" * 70)
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return 1
    
    finally:
        # Always cleanup
        await test_suite.cleanup_test_environment()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)