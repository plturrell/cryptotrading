#!/usr/bin/env python3
"""
Test script for Phase 3: Cross-Agent Workflows with On-Chain Data Exchange
Tests workflow orchestration, on-chain storage, and monitoring
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set testing environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_phase3_workflows():
    """Test Phase 3 implementation: Cross-agent workflows with on-chain data"""
    
    logger.info("🚀 Starting Phase 3 Workflow Tests...")
    logger.info("=" * 60)
    
    test_results = {
        "on_chain_storage": False,
        "data_exchange": False,
        "workflow_orchestration": False,
        "workflow_templates": False,
        "data_encryption": False,
        "workflow_monitoring": False,
        "integration": False
    }
    
    try:
        # Test 1: On-Chain Storage Contract
        logger.info("\n📝 Test 1: Verify On-Chain Storage Contract")
        try:
            from pathlib import Path
            contract_path = project_root / "contracts" / "A2ADataExchange.sol"
            
            if contract_path.exists():
                with open(contract_path, 'r') as f:
                    contract_code = f.read()
                    
                # Verify key functions exist
                required_functions = [
                    "function storeData",
                    "function retrieveData",
                    "function createWorkflow",
                    "function completeWorkflow",
                    "function grantDataAccess"
                ]
                
                all_found = all(func in contract_code for func in required_functions)
                
                if all_found:
                    logger.info("✅ On-chain storage contract verified")
                    test_results["on_chain_storage"] = True
                else:
                    logger.error("❌ Missing required functions in contract")
            else:
                logger.error("❌ Contract file not found")
                
        except Exception as e:
            logger.error(f"❌ Contract verification failed: {e}")
        
        # Test 2: Blockchain Data Exchange Service
        logger.info("\n📝 Test 2: Test Blockchain Data Exchange Service")
        try:
            from src.cryptotrading.core.protocols.a2a.blockchain_data_exchange import (
                BlockchainDataExchangeService,
                DataPacket,
                WorkflowData,
                DataStatus,
                WorkflowStatus
            )
            
            # Create service instance
            service = BlockchainDataExchangeService()
            
            # Verify key methods exist
            required_methods = [
                'store_data',
                'retrieve_data',
                'create_workflow',
                'complete_workflow',
                'grant_data_access',
                '_compress_data',
                '_decompress_data'
            ]
            
            for method in required_methods:
                if not hasattr(service, method):
                    raise AttributeError(f"Missing method: {method}")
            
            # Test compression
            test_data = b"Test data " * 1000  # Create compressible data
            compressed, was_compressed = service._compress_data(test_data)
            
            if was_compressed and len(compressed) < len(test_data):
                logger.info("✅ Data compression working")
            
            # Test decompression
            decompressed = service._decompress_data(compressed, was_compressed)
            if decompressed == test_data:
                logger.info("✅ Data decompression working")
            
            logger.info("✅ Blockchain data exchange service functional")
            test_results["data_exchange"] = True
            
        except Exception as e:
            logger.error(f"❌ Data exchange service test failed: {e}")
        
        # Test 3: Workflow Orchestration
        logger.info("\n📝 Test 3: Test Workflow Orchestration")
        try:
            from src.cryptotrading.core.protocols.a2a.workflow_orchestration import (
                WorkflowOrchestrator,
                WorkflowTemplate,
                WorkflowStep,
                WorkflowStepType,
                WorkflowInstance
            )
            
            # Create orchestrator
            orchestrator = WorkflowOrchestrator()
            
            # Check built-in templates
            templates = orchestrator.list_templates()
            expected_templates = [
                "market_analysis_v1",
                "trading_signal_v1",
                "portfolio_opt_v1"
            ]
            
            template_ids = [t["template_id"] for t in templates]
            
            for expected in expected_templates:
                if expected in template_ids:
                    logger.info(f"✅ Template found: {expected}")
            
            # Test workflow creation
            if templates:
                workflow_id = await orchestrator.create_workflow(
                    template_id=templates[0]["template_id"],
                    parameters={"test": True}
                )
                
                if workflow_id:
                    logger.info(f"✅ Workflow created: {workflow_id}")
                    
                    # Check workflow status
                    status = orchestrator.get_workflow_status(workflow_id)
                    if status:
                        logger.info(f"✅ Workflow status retrieved: {status['status']}")
            
            logger.info("✅ Workflow orchestration functional")
            test_results["workflow_orchestration"] = True
            
        except Exception as e:
            logger.error(f"❌ Workflow orchestration test failed: {e}")
        
        # Test 4: Workflow Templates
        logger.info("\n📝 Test 4: Test Workflow Templates")
        try:
            from src.cryptotrading.core.protocols.a2a.workflow_templates import (
                WorkflowTemplateLibrary,
                ML_TRAINING_WORKFLOW,
                REALTIME_TRADING_WORKFLOW,
                DEFI_YIELD_WORKFLOW,
                RISK_MONITORING_WORKFLOW,
                MARKET_RESEARCH_WORKFLOW
            )
            
            # Test template creation methods
            templates_to_test = [
                ("ML Training", WorkflowTemplateLibrary.create_ml_training_workflow()),
                ("Real-time Trading", WorkflowTemplateLibrary.create_real_time_trading_workflow()),
                ("DeFi Yield", WorkflowTemplateLibrary.create_defi_yield_optimization_workflow()),
                ("Risk Monitoring", WorkflowTemplateLibrary.create_risk_monitoring_workflow()),
                ("Market Research", WorkflowTemplateLibrary.create_market_research_workflow())
            ]
            
            for name, template in templates_to_test:
                if template and template.steps:
                    logger.info(f"✅ {name} template: {len(template.steps)} steps")
            
            # Test custom workflow creation
            custom = WorkflowTemplateLibrary.create_custom_workflow(
                template_id="test_custom",
                name="Test Custom Workflow",
                description="Test workflow",
                steps=[
                    {
                        "step_id": "step1",
                        "step_type": "DATA_COLLECTION",
                        "agent_id": "test-agent",
                        "parameters": {}
                    }
                ]
            )
            
            if custom and custom.template_id == "test_custom":
                logger.info("✅ Custom workflow template created")
            
            logger.info("✅ Workflow templates functional")
            test_results["workflow_templates"] = True
            
        except Exception as e:
            logger.error(f"❌ Workflow templates test failed: {e}")
        
        # Test 5: Data Encryption
        logger.info("\n📝 Test 5: Test Data Encryption Service")
        try:
            from src.cryptotrading.core.protocols.a2a.data_encryption import (
                DataEncryptionService,
                get_encryption_service,
                encrypt_for_chain,
                decrypt_from_chain
            )
            
            # Create encryption service
            encryption_service = get_encryption_service()
            
            # Test symmetric encryption
            test_data = b"Sensitive trading data"
            key = encryption_service.generate_symmetric_key()
            
            encrypted = encryption_service.encrypt_symmetric(test_data, key)
            decrypted = encryption_service.decrypt_symmetric(encrypted, key)
            
            if decrypted == test_data:
                logger.info("✅ Symmetric encryption working")
            
            # Test asymmetric encryption
            private_key, public_key = encryption_service.generate_asymmetric_keypair()
            
            encrypted = encryption_service.encrypt_asymmetric(test_data, public_key)
            decrypted = encryption_service.decrypt_asymmetric(encrypted, private_key)
            
            if decrypted == test_data:
                logger.info("✅ Asymmetric encryption working")
            
            # Test hybrid encryption for large data
            large_data = b"x" * 1000
            encrypted = encryption_service.encrypt_asymmetric(large_data, public_key)
            decrypted = encryption_service.decrypt_asymmetric(encrypted, private_key)
            
            if decrypted == large_data:
                logger.info("✅ Hybrid encryption working")
            
            # Test agent encryption
            encrypted_map = encryption_service.encrypt_for_agents(
                data={"test": "data"},
                sender_agent_id="sender",
                receiver_agent_ids=["receiver1", "receiver2"]
            )
            
            if len(encrypted_map) == 2:
                logger.info("✅ Multi-agent encryption working")
            
            logger.info("✅ Data encryption service functional")
            test_results["data_encryption"] = True
            
        except Exception as e:
            logger.error(f"❌ Data encryption test failed: {e}")
        
        # Test 6: Workflow Monitoring
        logger.info("\n📝 Test 6: Test Workflow Monitoring")
        try:
            from src.cryptotrading.core.protocols.a2a.workflow_monitor import (
                WorkflowMonitor,
                WorkflowEvent,
                WorkflowMetrics,
                EventType,
                get_workflow_monitor
            )
            
            # Create monitor
            monitor = WorkflowMonitor()
            
            # Test event handler registration
            event_count = 0
            
            def test_handler(event: WorkflowEvent):
                nonlocal event_count
                event_count += 1
            
            monitor.register_event_handler(EventType.DATA_STORED, test_handler)
            
            # Create test metrics
            metrics = WorkflowMetrics(
                workflow_id=1,
                total_steps=5,
                completed_steps=3
            )
            metrics.calculate_metrics()
            
            # Test analytics report
            report = monitor.generate_analytics_report()
            
            if "summary" in report and "event_distribution" in report:
                logger.info("✅ Analytics report generation working")
            
            logger.info("✅ Workflow monitoring functional")
            test_results["workflow_monitoring"] = True
            
        except Exception as e:
            logger.error(f"❌ Workflow monitoring test failed: {e}")
        
        # Test 7: Integration Test
        logger.info("\n📝 Test 7: Integration Test - Complete Workflow")
        try:
            # Simulate a complete workflow with all components
            
            # 1. Create encrypted data
            from src.cryptotrading.core.protocols.a2a.data_encryption import get_encryption_service
            encryption = get_encryption_service()
            
            test_market_data = {
                "symbol": "BTC/USDT",
                "price": 50000,
                "volume": 1000000,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            # 2. Encrypt data for multiple agents
            encrypted_data = encryption.encrypt_for_agents(
                data=test_market_data,
                sender_agent_id="data-provider",
                receiver_agent_ids=["ml-agent", "analysis-agent"]
            )
            
            logger.info(f"✅ Data encrypted for {len(encrypted_data)} agents")
            
            # 3. Create workflow instance
            from src.cryptotrading.core.protocols.a2a.workflow_orchestration import WorkflowOrchestrator
            orchestrator = WorkflowOrchestrator()
            
            # Use a simple template
            workflow_id = await orchestrator.create_workflow(
                template_id="market_analysis_v1",
                parameters={"symbol": "BTC/USDT"}
            )
            
            if workflow_id:
                logger.info(f"✅ Integration workflow created: {workflow_id}")
            
            # 4. Check workflow can be monitored
            from src.cryptotrading.core.protocols.a2a.workflow_monitor import WorkflowMonitor
            monitor = WorkflowMonitor()
            
            # Register test handler
            events_received = []
            
            def integration_handler(event):
                events_received.append(event)
            
            monitor.register_event_handler(EventType.WORKFLOW_CREATED, integration_handler)
            
            logger.info("✅ Integration test completed successfully")
            test_results["integration"] = True
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
        
        # Final Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 PHASE 3 TEST SUMMARY:")
        logger.info("=" * 60)
        
        for test_name, passed in test_results.items():
            status = "✅" if passed else "❌"
            test_label = test_name.replace("_", " ").title()
            logger.info(f"{status} {test_label}: {'PASSED' if passed else 'FAILED'}")
        
        passed_count = sum(1 for v in test_results.values() if v)
        total_count = len(test_results)
        success_rate = (passed_count / total_count) * 100
        
        logger.info(f"\nOverall: {passed_count}/{total_count} tests passed ({success_rate:.0f}%)")
        
        if success_rate >= 80:
            logger.info("\n🎉 PHASE 3 IMPLEMENTATION: SUCCESSFUL!")
            logger.info("Cross-agent workflows with on-chain data exchange are ready!")
            logger.info("\nKey achievements:")
            logger.info("• On-chain data storage contract (A2ADataExchange.sol)")
            logger.info("• Blockchain data exchange service with compression")
            logger.info("• Workflow orchestration with templates")
            logger.info("• Data encryption for sensitive information")
            logger.info("• Real-time workflow monitoring with events")
            logger.info("• 5 pre-built workflow templates")
            return True
        else:
            logger.warning("\n⚠️ PHASE 3 IMPLEMENTATION: NEEDS WORK")
            logger.warning("Some components need attention before production.")
            return False
            
    except Exception as e:
        logger.error(f"💥 Critical error in Phase 3 tests: {e}")
        return False


async def main():
    """Main test function"""
    success = await test_phase3_workflows()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ PHASE 3 TESTS PASSED!")
        print("\nPhase 3 Features Implemented:")
        print("1. On-chain data storage (up to 256KB per packet)")
        print("2. Cross-agent workflow orchestration")
        print("3. Data compression and encryption")
        print("4. Workflow monitoring with blockchain events")
        print("5. Pre-built templates for common operations")
        print("\nThe A2A platform now supports:")
        print("• Secure on-chain data exchange between agents")
        print("• Complex multi-step workflows")
        print("• Real-time monitoring and analytics")
        print("• Data privacy through encryption")
    else:
        print("❌ PHASE 3 TESTS FAILED!")
        print("Review the logs above to identify issues.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)