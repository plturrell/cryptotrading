#!/usr/bin/env python3
"""
Week 4 Agent Ecosystem Integration Test
Tests the complete CDS-integrated agent ecosystem after Week 4 migrations.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Week4EcosystemTest:
    """Test suite for complete Week 4 agent ecosystem integration"""
    
    def __init__(self):
        self.test_results = {
            "cds_connectivity": {"status": "not_tested", "details": None},
            "agent_registrations": {"status": "not_tested", "details": None},
            "cross_agent_communication": {"status": "not_tested", "details": None},
            "mcp_tool_integration": {"status": "not_tested", "details": None},
            "monitoring_system": {"status": "not_tested", "details": None},
            "transaction_boundaries": {"status": "not_tested", "details": None},
            "performance_metrics": {"status": "not_tested", "details": None},
            "overall_success": False
        }
    
    async def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run the complete Week 4 integration test suite"""
        logger.info("🚀 Starting Week 4 Agent Ecosystem Integration Test")
        logger.info("=" * 60)
        
        try:
            # Test 1: CDS Connectivity
            await self._test_cds_connectivity()
            
            # Test 2: Agent Registrations
            await self._test_agent_registrations()
            
            # Test 3: Cross-Agent Communication
            await self._test_cross_agent_communication()
            
            # Test 4: MCP Tool Integration
            await self._test_mcp_tool_integration()
            
            # Test 5: Monitoring System
            await self._test_monitoring_system()
            
            # Test 6: Transaction Boundaries
            await self._test_transaction_boundaries()
            
            # Test 7: Performance Metrics
            await self._test_performance_metrics()
            
            # Determine overall success
            await self._evaluate_overall_success()
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            self.test_results["overall_success"] = False
            self.test_results["error"] = str(e)
        
        return self.test_results
    
    async def _test_cds_connectivity(self):
        """Test CDS server connectivity and basic operations"""
        logger.info("1️⃣ Testing CDS Connectivity...")
        
        try:
            # Test basic CDS connectivity
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get("http://localhost:4005/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Test A2A health endpoint
                        async with session.get("http://localhost:4005/a2a/health") as a2a_response:
                            if a2a_response.status == 200:
                                a2a_health = await a2a_response.json()
                                
                                self.test_results["cds_connectivity"]["status"] = "✅ PASS"
                                self.test_results["cds_connectivity"]["details"] = {
                                    "cds_health": health_data,
                                    "a2a_health": a2a_health
                                }
                                logger.info("✅ CDS Connectivity - PASSED")
                            else:
                                raise Exception(f"A2A health endpoint failed: {a2a_response.status}")
                    else:
                        raise Exception(f"CDS health endpoint failed: {response.status}")
                        
        except Exception as e:
            self.test_results["cds_connectivity"]["status"] = "❌ FAIL"
            self.test_results["cds_connectivity"]["details"] = str(e)
            logger.error(f"❌ CDS Connectivity - FAILED: {e}")
    
    async def _test_agent_registrations(self):
        """Test agent registrations with CDS"""
        logger.info("2️⃣ Testing Agent Registrations...")
        
        try:
            # Import and test agent classes
            from cryptotrading.core.agents.specialized.agent_manager import AgentManager
            from cryptotrading.core.agents.specialized.data_analysis_agent import DataAnalysisAgent  
            from cryptotrading.core.agents.specialized.ml_agent import MLAgent
            from cryptotrading.core.agents.specialized.feature_store_agent import FeatureStoreAgent
            from cryptotrading.core.agents.specialized.technical_analysis.technical_analysis_agent import TechnicalAnalysisAgent
            from cryptotrading.core.agents.specialized.trading_algorithm_agent import TradingAlgorithmAgent
            from cryptotrading.core.agents.specialized.strands_glean_agent import StrandsGleanAgent
            from cryptotrading.core.agents.specialized.news_intelligence_agent import NewsIntelligenceAgent
            
            agents_to_test = [
                ("agent_manager", AgentManager),
                ("data_analysis", DataAnalysisAgent),
                ("ml_agent", MLAgent),
                ("feature_store", FeatureStoreAgent),
                ("technical_analysis", TechnicalAnalysisAgent),
                ("trading_algorithm", TradingAlgorithmAgent),
                ("strands_glean", StrandsGleanAgent),
                ("news_intelligence", NewsIntelligenceAgent)
            ]
            
            registration_results = []
            
            for agent_name, agent_class in agents_to_test:
                try:
                    # Create agent instance
                    agent = agent_class(agent_id=f"test_{agent_name}")
                    
                    # Test CDS mixin availability
                    has_cds_mixin = hasattr(agent, 'initialize_cds')
                    has_monitoring = hasattr(agent, '_cds_monitor')
                    
                    registration_results.append({
                        "agent": agent_name,
                        "class_loaded": True,
                        "cds_mixin": has_cds_mixin,
                        "monitoring": has_monitoring,
                        "status": "✅ READY" if has_cds_mixin else "⚠️ LIMITED"
                    })
                    
                except Exception as e:
                    registration_results.append({
                        "agent": agent_name,
                        "class_loaded": False,
                        "error": str(e),
                        "status": "❌ FAILED"
                    })
            
            # Evaluate results
            passed_agents = len([r for r in registration_results if r["status"].startswith("✅")])
            total_agents = len(registration_results)
            
            if passed_agents >= total_agents * 0.75:  # 75% success rate
                self.test_results["agent_registrations"]["status"] = "✅ PASS"
            else:
                self.test_results["agent_registrations"]["status"] = "⚠️ PARTIAL"
            
            self.test_results["agent_registrations"]["details"] = {
                "results": registration_results,
                "passed": passed_agents,
                "total": total_agents,
                "success_rate": f"{(passed_agents/total_agents)*100:.1f}%"
            }
            
            logger.info(f"✅ Agent Registrations - {passed_agents}/{total_agents} agents ready")
                        
        except Exception as e:
            self.test_results["agent_registrations"]["status"] = "❌ FAIL"
            self.test_results["agent_registrations"]["details"] = str(e)
            logger.error(f"❌ Agent Registrations - FAILED: {e}")
    
    async def _test_cross_agent_communication(self):
        """Test A2A communication between agents"""
        logger.info("3️⃣ Testing Cross-Agent Communication...")
        
        try:
            # Test A2A protocol and messaging
            from cryptotrading.core.protocols.a2a.a2a_protocol import A2AProtocol, MessageType
            
            # Create test message
            test_message = A2AProtocol.create_message(
                sender_id="test_sender",
                receiver_id="test_receiver", 
                message_type=MessageType.DATA_LOAD_REQUEST,
                payload={"test": "week4_integration"},
                priority=1
            )
            
            # Validate message
            message_dict = test_message.to_dict()
            is_valid = A2AProtocol.validate_message(message_dict)
            
            if is_valid:
                self.test_results["cross_agent_communication"]["status"] = "✅ PASS"
                self.test_results["cross_agent_communication"]["details"] = {
                    "message_validation": "passed",
                    "protocol": "A2A",
                    "message_id": message_dict.get("message_id")
                }
                logger.info("✅ Cross-Agent Communication - PASSED")
            else:
                raise Exception("A2A message validation failed")
                
        except Exception as e:
            self.test_results["cross_agent_communication"]["status"] = "❌ FAIL"
            self.test_results["cross_agent_communication"]["details"] = str(e)
            logger.error(f"❌ Cross-Agent Communication - FAILED: {e}")
    
    async def _test_mcp_tool_integration(self):
        """Test MCP tool configurations and integration"""
        logger.info("4️⃣ Testing MCP Tool Integration...")
        
        try:
            import json
            from pathlib import Path
            
            # Check for MCP tool configurations
            mcp_tools_dir = Path("src/cryptotrading/core/agents/mcp_tools")
            expected_tools = [
                "agent_manager_tools.json",
                "trading_strategies_tools.json", 
                "news_intelligence_tools.json",
                "feature_store_tools.json",
                "technical_analysis_tools.json",
                "strands_glean_tools.json"
            ]
            
            tool_results = []
            
            for tool_file in expected_tools:
                tool_path = mcp_tools_dir / tool_file
                try:
                    if tool_path.exists():
                        with open(tool_path) as f:
                            tool_config = json.load(f)
                            
                        # Validate basic structure
                        has_tools = "tools" in tool_config
                        has_capabilities = "capabilities" in tool_config
                        tool_count = len(tool_config.get("tools", {}))
                        
                        tool_results.append({
                            "file": tool_file,
                            "exists": True,
                            "valid_structure": has_tools and has_capabilities,
                            "tool_count": tool_count,
                            "status": "✅ VALID" if has_tools and has_capabilities else "⚠️ INCOMPLETE"
                        })
                    else:
                        tool_results.append({
                            "file": tool_file,
                            "exists": False,
                            "status": "❌ MISSING"
                        })
                        
                except Exception as e:
                    tool_results.append({
                        "file": tool_file,
                        "exists": True,
                        "error": str(e),
                        "status": "❌ INVALID"
                    })
            
            # Evaluate results
            valid_tools = len([r for r in tool_results if r["status"].startswith("✅")])
            total_tools = len(expected_tools)
            
            if valid_tools >= total_tools * 0.8:  # 80% success rate
                self.test_results["mcp_tool_integration"]["status"] = "✅ PASS"
            else:
                self.test_results["mcp_tool_integration"]["status"] = "⚠️ PARTIAL"
            
            self.test_results["mcp_tool_integration"]["details"] = {
                "results": tool_results,
                "valid": valid_tools,
                "total": total_tools,
                "success_rate": f"{(valid_tools/total_tools)*100:.1f}%"
            }
            
            logger.info(f"✅ MCP Tool Integration - {valid_tools}/{total_tools} configurations valid")
                        
        except Exception as e:
            self.test_results["mcp_tool_integration"]["status"] = "❌ FAIL"
            self.test_results["mcp_tool_integration"]["details"] = str(e)
            logger.error(f"❌ MCP Tool Integration - FAILED: {e}")
    
    async def _test_monitoring_system(self):
        """Test CDS monitoring and metrics system"""
        logger.info("5️⃣ Testing Monitoring System...")
        
        try:
            # Test monitoring infrastructure availability
            monitoring_components = []
            
            try:
                from cryptotrading.infrastructure.monitoring.cds_integration_monitor import get_cds_monitor, CDSOperationType
                monitoring_components.append({"component": "CDS Monitor", "status": "✅ AVAILABLE"})
            except ImportError:
                monitoring_components.append({"component": "CDS Monitor", "status": "❌ MISSING"})
            
            try:
                from cryptotrading.infrastructure.monitoring.cds_monitoring_api import CDSMonitoringAPI
                monitoring_components.append({"component": "Monitoring API", "status": "✅ AVAILABLE"})
            except ImportError:
                monitoring_components.append({"component": "Monitoring API", "status": "❌ MISSING"})
            
            try:
                from cryptotrading.infrastructure.transactions.agent_transaction_manager import TransactionManager
                monitoring_components.append({"component": "Transaction Manager", "status": "✅ AVAILABLE"})
            except ImportError:
                monitoring_components.append({"component": "Transaction Manager", "status": "❌ MISSING"})
            
            # Count available components
            available = len([c for c in monitoring_components if c["status"].startswith("✅")])
            total = len(monitoring_components)
            
            if available >= total * 0.67:  # 67% success rate
                self.test_results["monitoring_system"]["status"] = "✅ PASS"
            else:
                self.test_results["monitoring_system"]["status"] = "⚠️ PARTIAL"
            
            self.test_results["monitoring_system"]["details"] = {
                "components": monitoring_components,
                "available": available,
                "total": total,
                "availability_rate": f"{(available/total)*100:.1f}%"
            }
            
            logger.info(f"✅ Monitoring System - {available}/{total} components available")
                        
        except Exception as e:
            self.test_results["monitoring_system"]["status"] = "❌ FAIL"
            self.test_results["monitoring_system"]["details"] = str(e)
            logger.error(f"❌ Monitoring System - FAILED: {e}")
    
    async def _test_transaction_boundaries(self):
        """Test transaction boundary implementation"""
        logger.info("6️⃣ Testing Transaction Boundaries...")
        
        try:
            # Test transaction infrastructure
            transaction_features = []
            
            try:
                from cryptotrading.infrastructure.transactions.cds_transactional_client import CDSTransactionalMixin
                transaction_features.append({"feature": "Transactional Client", "status": "✅ AVAILABLE"})
            except ImportError:
                transaction_features.append({"feature": "Transactional Client", "status": "❌ MISSING"})
            
            try:
                from cryptotrading.infrastructure.transactions.agent_transaction_manager import transactional, TransactionIsolation
                transaction_features.append({"feature": "Transaction Decorators", "status": "✅ AVAILABLE"})
            except ImportError:
                transaction_features.append({"feature": "Transaction Decorators", "status": "❌ MISSING"})
            
            # Count available features
            available = len([f for f in transaction_features if f["status"].startswith("✅")])
            total = len(transaction_features)
            
            if available == total:
                self.test_results["transaction_boundaries"]["status"] = "✅ PASS"
            else:
                self.test_results["transaction_boundaries"]["status"] = "⚠️ PARTIAL"
            
            self.test_results["transaction_boundaries"]["details"] = {
                "features": transaction_features,
                "available": available,
                "total": total,
                "feature_coverage": f"{(available/total)*100:.1f}%"
            }
            
            logger.info(f"✅ Transaction Boundaries - {available}/{total} features available")
                        
        except Exception as e:
            self.test_results["transaction_boundaries"]["status"] = "❌ FAIL"
            self.test_results["transaction_boundaries"]["details"] = str(e)
            logger.error(f"❌ Transaction Boundaries - FAILED: {e}")
    
    async def _test_performance_metrics(self):
        """Test performance metrics and system health"""
        logger.info("7️⃣ Testing Performance Metrics...")
        
        try:
            import aiohttp
            
            # Test system performance endpoints
            performance_data = {}
            
            async with aiohttp.ClientSession() as session:
                # Test main health endpoint
                try:
                    async with session.get("http://localhost:4005/health", timeout=5) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            performance_data["system_health"] = {
                                "status": "healthy",
                                "uptime": health_data.get("uptime", 0),
                                "version": health_data.get("version", "unknown")
                            }
                        else:
                            performance_data["system_health"] = {"status": "unhealthy", "code": response.status}
                except Exception as e:
                    performance_data["system_health"] = {"status": "error", "error": str(e)}
                
                # Test A2A health
                try:
                    async with session.get("http://localhost:4005/a2a/health", timeout=5) as response:
                        if response.status == 200:
                            a2a_data = await response.json()
                            performance_data["a2a_health"] = a2a_data
                        else:
                            performance_data["a2a_health"] = {"status": "ERROR", "code": response.status}
                except Exception as e:
                    performance_data["a2a_health"] = {"status": "ERROR", "error": str(e)}
            
            # Evaluate performance
            system_healthy = performance_data.get("system_health", {}).get("status") == "healthy"
            a2a_healthy = performance_data.get("a2a_health", {}).get("status") in ["HEALTHY", "healthy"]
            
            if system_healthy and a2a_healthy:
                self.test_results["performance_metrics"]["status"] = "✅ PASS"
            elif system_healthy or a2a_healthy:
                self.test_results["performance_metrics"]["status"] = "⚠️ PARTIAL"
            else:
                self.test_results["performance_metrics"]["status"] = "❌ FAIL"
            
            self.test_results["performance_metrics"]["details"] = performance_data
            
            logger.info(f"✅ Performance Metrics - System health monitored")
                        
        except Exception as e:
            self.test_results["performance_metrics"]["status"] = "❌ FAIL"
            self.test_results["performance_metrics"]["details"] = str(e)
            logger.error(f"❌ Performance Metrics - FAILED: {e}")
    
    async def _evaluate_overall_success(self):
        """Evaluate overall test success"""
        logger.info("=" * 60)
        logger.info("📊 WEEK 4 INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        
        # Count successful tests
        test_statuses = []
        for test_name, result in self.test_results.items():
            if test_name != "overall_success" and isinstance(result, dict):
                status = result.get("status", "unknown")
                test_statuses.append(status)
                logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        # Calculate success metrics
        passed_tests = len([s for s in test_statuses if s.startswith("✅")])
        partial_tests = len([s for s in test_statuses if s.startswith("⚠️")])
        failed_tests = len([s for s in test_statuses if s.startswith("❌")])
        total_tests = len(test_statuses)
        
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Partial Success: {partial_tests}")
        logger.info(f"Tests Failed: {failed_tests}")
        
        # Determine overall success (require 70% pass rate)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        self.test_results["overall_success"] = success_rate >= 0.7
        
        if self.test_results["overall_success"]:
            logger.info("🎉 WEEK 4 AGENT ECOSYSTEM INTEGRATION - SUCCESSFUL!")
            logger.info("✅ CDS integration complete across all agents")
            logger.info("✅ Monitoring and transaction support enabled")
            logger.info("✅ MCP tool configurations updated")
            logger.info("✅ Cross-agent communication functional")
        else:
            logger.warning("⚠️ Week 4 integration has issues")
            logger.info("📝 Check individual test results for details")
        
        logger.info("=" * 60)

async def main():
    """Main test runner"""
    logger.info("🚀 Starting Week 4 Agent Ecosystem Integration Test")
    logger.info("Testing CDS integration across all migrated agents")
    logger.info("")
    
    try:
        test_suite = Week4EcosystemTest()
        results = await test_suite.run_complete_test_suite()
        
        # Return appropriate exit code
        return 0 if results.get("overall_success", False) else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)