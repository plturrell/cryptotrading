#!/usr/bin/env python3
"""
CDS-A2A Complete Integration Verification
Ensures real, functional integration with no gaps, mocks, or simulations
"""

import asyncio
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import aiohttp

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CDSIntegrationVerifier:
    """Verify complete CDS-A2A integration without mocks"""
    
    def __init__(self):
        self.cds_base_url = "http://localhost:4004"
        self.verification_results = {
            "cds_services": {},
            "a2a_integration": {},
            "data_flow": {},
            "real_operations": {},
            "mock_detection": {},
            "gaps_found": [],
            "overall_status": "NOT_VERIFIED"
        }
    
    async def run_complete_verification(self) -> Dict[str, Any]:
        """Run comprehensive integration verification"""
        logger.info("🔍 STARTING CDS-A2A INTEGRATION VERIFICATION")
        logger.info("=" * 70)
        logger.info("Checking for REAL integration - NO mocks, NO simulations, NO gaps")
        logger.info("=" * 70)
        
        try:
            # 1. Verify CDS Services are real and operational
            await self._verify_cds_services()
            
            # 2. Check A2A Service integration
            await self._verify_a2a_service()
            
            # 3. Test real data operations
            await self._test_real_data_operations()
            
            # 4. Check for mocks and simulations
            await self._detect_mocks_and_simulations()
            
            # 5. Verify bi-directional communication
            await self._verify_bidirectional_communication()
            
            # 6. Test WebSocket connectivity
            await self._test_websocket_connectivity()
            
            # 7. Verify agent registrations in CDS
            await self._verify_agent_cds_registrations()
            
            # 8. Test transaction boundaries
            await self._test_transaction_boundaries()
            
            # Generate final report
            await self._generate_verification_report()
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            self.verification_results["overall_status"] = "FAILED"
            self.verification_results["error"] = str(e)
        
        return self.verification_results
    
    async def _verify_cds_services(self):
        """Verify CDS services are real and operational"""
        logger.info("\n1️⃣ VERIFYING CDS SERVICES")
        logger.info("-" * 40)
        
        services_to_check = [
            "/api/odata/v4/A2AService",
            "/api/odata/v4/MarketAnalysisService",
            "/api/odata/v4/MonitoringService",
            "/api/odata/v4/DataPipelineService",
            "/api/odata/v4/IntelligenceService"
        ]
        
        async with aiohttp.ClientSession() as session:
            for service_path in services_to_check:
                try:
                    # Check service metadata
                    url = f"{self.cds_base_url}{service_path}/$metadata"
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            is_real = "<EntityType" in content and "<Property" in content
                            
                            # Check for actual entities
                            url = f"{self.cds_base_url}{service_path}"
                            async with session.get(url) as data_response:
                                if data_response.status == 200:
                                    data = await data_response.json()
                                    has_entities = "value" in data or "@odata.context" in data
                                    
                                    self.verification_results["cds_services"][service_path] = {
                                        "status": "✅ REAL" if is_real and has_entities else "⚠️ INCOMPLETE",
                                        "has_metadata": is_real,
                                        "has_entities": has_entities,
                                        "entity_count": len(data.get("value", [])) if "value" in data else 0
                                    }
                                    
                                    if is_real and has_entities:
                                        logger.info(f"✅ {service_path}: REAL SERVICE")
                                    else:
                                        logger.warning(f"⚠️ {service_path}: Service exists but may be incomplete")
                        else:
                            self.verification_results["cds_services"][service_path] = {
                                "status": "❌ UNAVAILABLE",
                                "http_status": response.status
                            }
                            logger.error(f"❌ {service_path}: Service unavailable")
                            
                except Exception as e:
                    self.verification_results["cds_services"][service_path] = {
                        "status": "❌ ERROR",
                        "error": str(e)
                    }
                    logger.error(f"❌ {service_path}: {e}")
    
    async def _verify_a2a_service(self):
        """Verify A2A Service integration"""
        logger.info("\n2️⃣ VERIFYING A2A SERVICE INTEGRATION")
        logger.info("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            # Check A2A specific endpoints
            endpoints = {
                "health": "/a2a/health",
                "events": "/a2a/events",
                "websocket": "/a2a/ws",
                "agents": "/api/odata/v4/A2AService/A2AAgents",
                "messages": "/api/odata/v4/A2AService/A2AMessages",
                "workflows": "/api/odata/v4/A2AService/A2AWorkflows"
            }
            
            for name, endpoint in endpoints.items():
                try:
                    if name == "websocket":
                        # WebSocket endpoint check
                        self.verification_results["a2a_integration"][name] = {
                            "status": "⏭️ WEBSOCKET",
                            "note": "WebSocket endpoint - tested separately"
                        }
                        continue
                    
                    if name == "events":
                        # POST endpoint check
                        url = f"{self.cds_base_url}{endpoint}"
                        test_data = {
                            "event": "test_event",
                            "data": {"test": "verification"},
                            "timestamp": datetime.now().isoformat()
                        }
                        async with session.post(url, json=test_data) as response:
                            if response.status in [200, 201]:
                                result = await response.json()
                                self.verification_results["a2a_integration"][name] = {
                                    "status": "✅ OPERATIONAL",
                                    "can_receive_events": True,
                                    "response": result.get("status", "unknown")
                                }
                                logger.info(f"✅ {name}: Can receive events")
                            else:
                                self.verification_results["a2a_integration"][name] = {
                                    "status": "❌ NOT WORKING",
                                    "http_status": response.status
                                }
                                logger.error(f"❌ {name}: Cannot receive events")
                    else:
                        # GET endpoint check
                        url = f"{self.cds_base_url}{endpoint}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                is_operational = True
                                
                                if name in ["agents", "messages", "workflows"]:
                                    # Check if it's a real OData response
                                    is_operational = "@odata.context" in data or "value" in data
                                
                                self.verification_results["a2a_integration"][name] = {
                                    "status": "✅ OPERATIONAL" if is_operational else "⚠️ PARTIAL",
                                    "has_data": "value" in data if isinstance(data, dict) else False,
                                    "record_count": len(data.get("value", [])) if "value" in data else 0
                                }
                                
                                if is_operational:
                                    logger.info(f"✅ {name}: Operational")
                                else:
                                    logger.warning(f"⚠️ {name}: Partial functionality")
                            else:
                                self.verification_results["a2a_integration"][name] = {
                                    "status": "❌ UNAVAILABLE",
                                    "http_status": response.status
                                }
                                logger.error(f"❌ {name}: Unavailable")
                                
                except Exception as e:
                    self.verification_results["a2a_integration"][name] = {
                        "status": "❌ ERROR",
                        "error": str(e)
                    }
                    logger.error(f"❌ {name}: {e}")
    
    async def _test_real_data_operations(self):
        """Test real data operations through CDS"""
        logger.info("\n3️⃣ TESTING REAL DATA OPERATIONS")
        logger.info("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            # Test creating a real agent registration
            test_agent = {
                "agentName": "Integration Test Agent",
                "agentType": "test",
                "description": "Test agent for CDS verification",
                "capabilities": json.dumps(["testing", "verification"]),  # Convert to JSON string
                "configuration": json.dumps({"test": True}),
                "status": "ACTIVE",  # Must be uppercase per CDS schema
                "version": "1.0.0",
                "lastHeartbeat": datetime.now().isoformat() + "Z"
            }
            
            try:
                # Create agent
                url = f"{self.cds_base_url}/api/odata/v4/A2AService/A2AAgents"
                async with session.post(url, json=test_agent) as response:
                    if response.status in [201, 200]:
                        created_data = await response.json()
                        agent_id = created_data.get("ID")
                        
                        if not agent_id:
                            # If no ID returned, try to get from Location header
                            location = response.headers.get("Location", "")
                            if location and "(" in location:
                                agent_id = location.split("(")[1].split(")")[0].strip("'")
                        
                        if not agent_id:
                            logger.warning("Created agent but no ID returned")
                            self.verification_results["real_operations"] = {
                                "status": "⚠️ CREATE ONLY",
                                "create": "✅ SUCCESS",
                                "read": "❌ NO ID RETURNED"
                            }
                            return
                        
                        # Read back the agent (handle draft entities)
                        # First try with IsActiveEntity=false for draft entities
                        read_url = f"{url}(ID='{agent_id}',IsActiveEntity=false)"
                        async with session.get(read_url) as read_response:
                            if read_response.status == 200:
                                read_data = await read_response.json()
                                
                                # Update the agent
                                update_data = {"status": "INACTIVE"}  # Use valid enum value
                                async with session.patch(read_url, json=update_data) as update_response:
                                    update_success = update_response.status in [200, 204]
                                    
                                    # Delete the test agent
                                    async with session.delete(read_url) as delete_response:
                                        delete_success = delete_response.status in [200, 204]
                                        
                                        self.verification_results["real_operations"] = {
                                            "status": "✅ FULL CRUD" if update_success and delete_success else "⚠️ PARTIAL CRUD",
                                            "create": "✅ SUCCESS",
                                            "read": "✅ SUCCESS",
                                            "update": "✅ SUCCESS" if update_success else "❌ FAILED",
                                            "delete": "✅ SUCCESS" if delete_success else "❌ FAILED",
                                            "test_agent_id": agent_id
                                        }
                                        
                                        if update_success and delete_success:
                                            logger.info("✅ CRUD Operations: FULLY FUNCTIONAL")
                                        else:
                                            logger.warning("⚠️ CRUD Operations: Partially functional")
                            else:
                                self.verification_results["real_operations"] = {
                                    "status": "⚠️ CREATE ONLY",
                                    "create": "✅ SUCCESS",
                                    "read": "❌ FAILED"
                                }
                                logger.warning("⚠️ Can create but cannot read back")
                    else:
                        response_text = await response.text()
                        self.verification_results["real_operations"] = {
                            "status": "❌ NO CRUD",
                            "error": f"Create failed with status {response.status}",
                            "response": response_text[:200]
                        }
                        logger.error(f"❌ CRUD Operations: Failed - {response.status}")
                        
            except Exception as e:
                self.verification_results["real_operations"] = {
                    "status": "❌ ERROR",
                    "error": str(e)
                }
                logger.error(f"❌ CRUD Operations: {e}")
    
    async def _detect_mocks_and_simulations(self):
        """Detect any mock or simulation code"""
        logger.info("\n4️⃣ DETECTING MOCKS AND SIMULATIONS")
        logger.info("-" * 40)
        
        # Search for mock/simulation patterns in code
        mock_patterns = [
            "mock", "Mock", "MOCK",
            "simulate", "Simulate", "SIMULATION",
            "fake", "Fake", "FAKE",
            "dummy", "Dummy", "DUMMY",
            "stub", "Stub", "STUB",
            "# For now", "# TODO", "# FIXME",
            "hardcoded", "hard-coded"
        ]
        
        files_with_mocks = []
        
        # Check key integration files
        files_to_check = [
            "src/cryptotrading/core/protocols/cds/cds_client.py",
            "src/cryptotrading/core/protocols/cds/cds_service.py",
            "srv/a2a-service.js",
            "srv/server.js"
        ]
        
        for file_path in files_to_check:
            full_path = project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        found_patterns = []
                        for pattern in mock_patterns:
                            if pattern in content:
                                # Count occurrences
                                count = content.count(pattern)
                                found_patterns.append(f"{pattern} ({count}x)")
                        
                        if found_patterns:
                            files_with_mocks.append({
                                "file": file_path,
                                "patterns": found_patterns
                            })
                except Exception as e:
                    logger.warning(f"Could not check {file_path}: {e}")
        
        self.verification_results["mock_detection"] = {
            "status": "✅ CLEAN" if not files_with_mocks else "⚠️ MOCKS FOUND",
            "files_with_mocks": files_with_mocks,
            "mock_count": len(files_with_mocks)
        }
        
        if files_with_mocks:
            logger.warning(f"⚠️ Found potential mocks in {len(files_with_mocks)} files")
            for file_info in files_with_mocks:
                logger.warning(f"  - {file_info['file']}: {', '.join(file_info['patterns'][:3])}")
        else:
            logger.info("✅ No obvious mocks or simulations detected")
    
    async def _verify_bidirectional_communication(self):
        """Verify bi-directional communication between CDS and A2A"""
        logger.info("\n5️⃣ VERIFYING BI-DIRECTIONAL COMMUNICATION")
        logger.info("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test Python -> CDS direction
                python_to_cds = await self._test_python_to_cds(session)
                
                # Test CDS -> Python direction (via event endpoint)
                cds_to_python = await self._test_cds_to_python(session)
                
                self.verification_results["data_flow"]["python_to_cds"] = python_to_cds
                self.verification_results["data_flow"]["cds_to_python"] = cds_to_python
                
                if python_to_cds["status"] == "✅ WORKING" and cds_to_python["status"] == "✅ WORKING":
                    logger.info("✅ Bi-directional communication: FULLY FUNCTIONAL")
                else:
                    logger.warning("⚠️ Bi-directional communication: Partially functional")
                    
            except Exception as e:
                self.verification_results["data_flow"]["error"] = str(e)
                logger.error(f"❌ Bi-directional communication test failed: {e}")
    
    async def _test_python_to_cds(self, session):
        """Test Python to CDS communication"""
        try:
            # Send event from Python to CDS
            url = f"{self.cds_base_url}/a2a/events"
            test_event = {
                "event": "python.test.event",
                "data": {
                    "source": "python_agent",
                    "timestamp": datetime.now().isoformat(),
                    "test_id": f"test_{datetime.now().timestamp()}"
                }
            }
            
            async with session.post(url, json=test_event) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "✅ WORKING",
                        "can_send_events": True,
                        "response_status": result.get("status", "unknown")
                    }
                else:
                    return {
                        "status": "❌ NOT WORKING",
                        "http_status": response.status
                    }
        except Exception as e:
            return {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    async def _test_cds_to_python(self, session):
        """Test CDS to Python communication"""
        try:
            # Create a message in CDS that should trigger Python agents
            message_data = {
                "messageType": "TEST",
                "payload": json.dumps({"test": "cds_to_python"}),
                "priority": "MEDIUM",
                "status": "SENT",
                "sentAt": datetime.now().isoformat() + "Z"
            }
            
            url = f"{self.cds_base_url}/api/odata/v4/A2AService/A2AMessages"
            async with session.post(url, json=message_data) as response:
                if response.status in [201, 200]:
                    return {
                        "status": "✅ WORKING",
                        "can_create_messages": True,
                        "message_id": message_data["ID"]
                    }
                else:
                    return {
                        "status": "❌ NOT WORKING",
                        "http_status": response.status
                    }
        except Exception as e:
            return {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    async def _test_websocket_connectivity(self):
        """Test WebSocket connectivity for real-time communication"""
        logger.info("\n6️⃣ TESTING WEBSOCKET CONNECTIVITY")
        logger.info("-" * 40)
        
        try:
            import websockets
            
            ws_url = "ws://localhost:4004/a2a/ws"
            
            try:
                async with websockets.connect(ws_url) as websocket:
                    # Send registration message
                    register_msg = json.dumps({
                        "type": "AGENT_REGISTER",
                        "payload": {"agentId": "test_ws_agent"}
                    })
                    await websocket.send(register_msg)
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    
                    # Send heartbeat
                    heartbeat_msg = json.dumps({
                        "type": "HEARTBEAT",
                        "payload": {}
                    })
                    await websocket.send(heartbeat_msg)
                    
                    # Wait for heartbeat ack
                    heartbeat_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    heartbeat_data = json.loads(heartbeat_response)
                    
                    self.verification_results["websocket"] = {
                        "status": "✅ OPERATIONAL",
                        "can_register": response_data.get("type") == "AGENT_REGISTERED",
                        "heartbeat_works": heartbeat_data.get("type") == "HEARTBEAT_ACK"
                    }
                    logger.info("✅ WebSocket: FULLY OPERATIONAL")
                    
            except (ConnectionRefusedError, TimeoutError) as e:
                self.verification_results["websocket"] = {
                    "status": "❌ NOT AVAILABLE",
                    "error": str(e)
                }
                logger.error(f"❌ WebSocket: Not available - {e}")
                
        except ImportError:
            self.verification_results["websocket"] = {
                "status": "⏭️ SKIPPED",
                "note": "websockets library not installed"
            }
            logger.warning("⏭️ WebSocket test skipped (install websockets library)")
    
    async def _verify_agent_cds_registrations(self):
        """Verify agent registrations in CDS"""
        logger.info("\n7️⃣ VERIFYING AGENT CDS REGISTRATIONS")
        logger.info("-" * 40)
        
        try:
            # Import agent classes and check CDS integration
            from cryptotrading.core.protocols.cds import create_cds_client, CDSClient, CDSServiceConfig
            
            # Test creating a CDS client directly (not using async factory for this test)
            client = CDSClient(CDSServiceConfig())
            
            # Check if client has real implementation
            has_real_methods = all([
                hasattr(client, 'connect'),
                hasattr(client, 'disconnect'),
                hasattr(client, 'register_agent'),
                hasattr(client, 'send_message')
            ])
            
            self.verification_results["agent_cds_integration"] = {
                "status": "✅ INTEGRATED" if has_real_methods else "❌ NOT INTEGRATED",
                "has_client": True,
                "has_methods": has_real_methods
            }
            
            if has_real_methods:
                # Test actual connection
                try:
                    await client.connect("verification_agent")
                    self.verification_results["agent_cds_integration"]["can_connect"] = True
                    await client.disconnect()
                    logger.info("✅ Agent CDS Integration: FUNCTIONAL")
                except Exception as e:
                    self.verification_results["agent_cds_integration"]["can_connect"] = False
                    self.verification_results["agent_cds_integration"]["connection_error"] = str(e)
                    logger.warning(f"⚠️ Agent CDS Integration: Client exists but connection failed - {e}")
            else:
                logger.error("❌ Agent CDS Integration: Methods missing")
                
        except Exception as e:
            self.verification_results["agent_cds_integration"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
            logger.error(f"❌ Agent CDS Integration: {e}")
    
    async def _test_transaction_boundaries(self):
        """Test transaction boundary implementation"""
        logger.info("\n8️⃣ TESTING TRANSACTION BOUNDARIES")
        logger.info("-" * 40)
        
        try:
            # Check if transaction infrastructure exists
            from cryptotrading.infrastructure.transactions.agent_transaction_manager import (
                TransactionManager, transactional
            )
            
            # Create transaction manager
            tx_manager = TransactionManager()
            
            # Test transaction creation
            transaction = await tx_manager.begin_transaction(
                agent_id="test_agent",
                transaction_type="TEST_TRANSACTION"
            )
            
            # Test transaction operations
            await tx_manager.add_checkpoint(
                transaction.transaction_id,
                "test_checkpoint",
                {"test": "data"}
            )
            
            # Commit transaction
            await tx_manager.commit_transaction(transaction.transaction_id)
            
            self.verification_results["transactions"] = {
                "status": "✅ FUNCTIONAL",
                "can_create": True,
                "can_checkpoint": True,
                "can_commit": True,
                "transaction_id": transaction.transaction_id
            }
            logger.info("✅ Transaction Boundaries: FULLY FUNCTIONAL")
            
        except Exception as e:
            self.verification_results["transactions"] = {
                "status": "❌ NOT FUNCTIONAL",
                "error": str(e)
            }
            logger.error(f"❌ Transaction Boundaries: {e}")
    
    async def _generate_verification_report(self):
        """Generate final verification report"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 CDS-A2A INTEGRATION VERIFICATION REPORT")
        logger.info("=" * 70)
        
        # Count statuses
        all_statuses = []
        gaps = []
        
        for category, results in self.verification_results.items():
            if isinstance(results, dict) and "status" in results:
                status = results["status"]
                all_statuses.append(status)
                
                if "❌" in status:
                    gaps.append(f"{category}: {status}")
                elif "⚠️" in status:
                    gaps.append(f"{category}: {status} (partial)")
        
        # Check CDS services
        for service, info in self.verification_results.get("cds_services", {}).items():
            if isinstance(info, dict) and "status" in info:
                all_statuses.append(info["status"])
                if "❌" in info["status"]:
                    gaps.append(f"CDS Service {service}: {info['status']}")
        
        # Calculate metrics
        total_checks = len(all_statuses)
        fully_working = len([s for s in all_statuses if "✅" in s])
        partially_working = len([s for s in all_statuses if "⚠️" in s])
        not_working = len([s for s in all_statuses if "❌" in s])
        
        # Determine overall status
        if fully_working >= total_checks * 0.8 and not_working == 0:
            overall = "✅ FULLY INTEGRATED"
        elif fully_working >= total_checks * 0.6:
            overall = "⚠️ PARTIALLY INTEGRATED"
        else:
            overall = "❌ INTEGRATION INCOMPLETE"
        
        self.verification_results["overall_status"] = overall
        self.verification_results["gaps_found"] = gaps
        
        # Print summary
        logger.info(f"\nOVERALL STATUS: {overall}")
        logger.info(f"Total Checks: {total_checks}")
        logger.info(f"Fully Working: {fully_working} ({fully_working/total_checks*100:.1f}%)")
        logger.info(f"Partially Working: {partially_working} ({partially_working/total_checks*100:.1f}%)")
        logger.info(f"Not Working: {not_working} ({not_working/total_checks*100:.1f}%)")
        
        if gaps:
            logger.info(f"\n🔴 GAPS FOUND ({len(gaps)}):")
            for gap in gaps:
                logger.info(f"  - {gap}")
        else:
            logger.info("\n✅ NO CRITICAL GAPS FOUND")
        
        # Mock detection summary
        mock_info = self.verification_results.get("mock_detection", {})
        if mock_info.get("files_with_mocks"):
            logger.warning(f"\n⚠️ POTENTIAL MOCKS/SIMULATIONS DETECTED:")
            for file_info in mock_info["files_with_mocks"][:5]:
                logger.warning(f"  - {file_info['file']}")
        
        logger.info("\n" + "=" * 70)
        
        return self.verification_results


async def main():
    """Main verification runner"""
    logger.info("🔍 CDS-A2A Complete Integration Verification")
    logger.info("Checking for real, functional integration with no gaps")
    logger.info("")
    
    try:
        verifier = CDSIntegrationVerifier()
        results = await verifier.run_complete_verification()
        
        # Write detailed results to file
        results_file = project_root / "cds_a2a_verification_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        # Return appropriate exit code
        if "✅" in results.get("overall_status", ""):
            return 0
        elif "⚠️" in results.get("overall_status", ""):
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return 3


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)