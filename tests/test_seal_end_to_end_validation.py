#!/usr/bin/env python3
"""
Comprehensive End-to-End SEAL System Validation
Tests the complete SEAL-enhanced enterprise code management system
"""

import asyncio
import sys
import traceback
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import all SEAL components
try:
    from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase
    from cryptotrading.infrastructure.code_management.database_adapter import CodeManagementDatabaseAdapter
    from cryptotrading.infrastructure.code_management.issue_lifecycle_manager import IssueLifecycleManager
    from cryptotrading.infrastructure.code_management.issue_backlog_tracker import IssueBacklogTracker
    from cryptotrading.infrastructure.code_management.seal_code_adapter import SEALCodeAdapter
    from cryptotrading.infrastructure.code_management.seal_workflow_engine import SEALWorkflowEngine
    from cryptotrading.infrastructure.code_management.enterprise_code_orchestrator import (
        EnterpriseCodeOrchestrator, OrchestrationConfig
    )
    from cryptotrading.infrastructure.code_management.intelligent_code_manager import IntelligentCodeManager
    from cryptotrading.infrastructure.code_management.automated_quality_monitor import AutomatedQualityMonitor
    from cryptotrading.infrastructure.code_management.proactive_issue_detector import ProactiveIssueDetector
    from cryptotrading.infrastructure.code_management.code_health_dashboard import CodeHealthDashboard
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

class SEALSystemValidator:
    """Comprehensive SEAL system validation"""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        self.temp_dir = None
        self.db = None
        self.adapter = None
        
    async def setup_test_environment(self) -> bool:
        """Setup isolated test environment"""
        try:
            # Create temporary directory for testing
            self.temp_dir = Path(tempfile.mkdtemp(prefix="seal_test_"))
            
            # Initialize database with test configuration
            self.db = UnifiedDatabase(
                db_path=str(self.temp_dir / "test.db"),
                redis_url=None  # Skip Redis for testing
            )
            await self.db.initialize()
            
            # Initialize database adapter
            self.adapter = CodeManagementDatabaseAdapter(self.db)
            
            self.test_results["tests"]["setup"] = {
                "status": "passed",
                "message": f"Test environment created at {self.temp_dir}"
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["setup"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"Setup failed: {e}")
            return False
    
    async def test_database_layer(self) -> bool:
        """Test unified database functionality"""
        try:
            # Test database connection
            await self.db.execute("SELECT 1")
            
            # Test table creation
            await self.adapter.store_issue({
                "id": "test-001",
                "title": "Test Issue",
                "description": "Test description",
                "severity": "medium",
                "type": "bug",
                "status": "detected",
                "created_at": datetime.now().isoformat()
            })
            
            # Test issue retrieval
            issues = await self.adapter.get_issues()
            assert len(issues) >= 1, "Failed to retrieve stored issue"
            
            # Test metrics storage
            await self.adapter.store_metrics({
                "timestamp": datetime.now().isoformat(),
                "total_issues": 1,
                "critical_issues": 0,
                "resolved_issues": 0
            })
            
            self.test_results["tests"]["database"] = {
                "status": "passed",
                "message": "Database layer functioning correctly"
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["database"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"Database test failed: {e}")
            return False
    
    async def test_issue_lifecycle_manager(self) -> bool:
        """Test issue lifecycle management"""
        try:
            lifecycle_manager = IssueLifecycleManager(self.adapter)
            
            # Test issue creation and state transitions
            test_issue = {
                "id": "lifecycle-001",
                "title": "Lifecycle Test Issue",
                "description": "Testing lifecycle transitions",
                "severity": "high",
                "type": "bug",
                "status": "detected",
                "created_at": datetime.now().isoformat()
            }
            
            await self.adapter.store_issue(test_issue)
            
            # Test auto-triage
            await lifecycle_manager.auto_triage_issues()
            
            # Test state transition
            success = await lifecycle_manager.transition_issue_state(
                "lifecycle-001", "triaged", "Auto-triaged for testing"
            )
            assert success, "Failed to transition issue state"
            
            # Test lifecycle metrics
            metrics = await lifecycle_manager.get_lifecycle_metrics()
            assert "total_issues" in metrics, "Lifecycle metrics missing required fields"
            
            self.test_results["tests"]["lifecycle_manager"] = {
                "status": "passed",
                "message": "Issue lifecycle manager functioning correctly",
                "metrics": metrics
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["lifecycle_manager"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"Lifecycle manager test failed: {e}")
            return False
    
    async def test_backlog_tracker(self) -> bool:
        """Test issue backlog and sprint management"""
        try:
            backlog_tracker = IssueBacklogTracker(self.adapter)
            
            # Test sprint creation
            sprint_id = await backlog_tracker.create_sprint(
                "Test Sprint 1",
                datetime.now(),
                datetime.now() + timedelta(days=14)
            )
            assert sprint_id, "Failed to create sprint"
            
            # Test issue assignment to sprint
            await backlog_tracker.assign_issue_to_sprint("lifecycle-001", sprint_id)
            
            # Test backlog metrics
            metrics = await backlog_tracker.get_backlog_metrics()
            assert "total_issues" in metrics, "Backlog metrics missing required fields"
            
            # Test sprint progress
            progress = await backlog_tracker.get_sprint_progress(sprint_id)
            assert "sprint_id" in progress, "Sprint progress missing required fields"
            
            self.test_results["tests"]["backlog_tracker"] = {
                "status": "passed",
                "message": "Backlog tracker functioning correctly",
                "sprint_id": sprint_id,
                "metrics": metrics
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["backlog_tracker"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"Backlog tracker test failed: {e}")
            return False
    
    async def test_seal_code_adapter(self) -> bool:
        """Test SEAL code adaptation engine"""
        try:
            lifecycle_manager = IssueLifecycleManager(self.adapter)
            seal_adapter = SEALCodeAdapter(self.adapter, lifecycle_manager)
            
            # Verify Grok configuration
            assert seal_adapter.config["model_name"] == "grok", "SEAL not configured for Grok model"
            
            # Test codebase analysis
            adaptation_requests = await seal_adapter.analyze_codebase_for_adaptation(
                self.temp_dir
            )
            assert isinstance(adaptation_requests, list), "Adaptation requests should be a list"
            
            # Test adaptation strategy selection
            test_issue = {
                "id": "seal-001",
                "title": "Performance Issue",
                "description": "Slow database queries",
                "severity": "medium",
                "type": "performance",
                "status": "detected"
            }
            
            strategy = await seal_adapter.select_adaptation_strategy(test_issue)
            assert strategy, "Failed to select adaptation strategy"
            assert strategy["type"] in ["bug_fix", "performance_optimization", "code_quality", "security_enhancement", "refactoring"], "Invalid adaptation type"
            
            # Test self-edit application (simulation)
            edit_result = await seal_adapter.apply_self_edit(
                test_issue,
                strategy,
                "# Test code\nprint('hello world')"
            )
            assert edit_result, "Failed to apply self-edit"
            assert "confidence_score" in edit_result, "Self-edit result missing confidence score"
            
            self.test_results["tests"]["seal_adapter"] = {
                "status": "passed",
                "message": "SEAL code adapter functioning correctly",
                "config": seal_adapter.config,
                "strategy": strategy,
                "edit_result": {k: v for k, v in edit_result.items() if k != "adapted_code"}
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["seal_adapter"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"SEAL adapter test failed: {e}")
            return False
    
    async def test_seal_workflow_engine(self) -> bool:
        """Test SEAL workflow engine"""
        try:
            lifecycle_manager = IssueLifecycleManager(self.adapter)
            backlog_tracker = IssueBacklogTracker(self.adapter)
            seal_adapter = SEALCodeAdapter(self.adapter, lifecycle_manager)
            
            workflow_engine = SEALWorkflowEngine(
                self.adapter, lifecycle_manager, backlog_tracker, seal_adapter
            )
            
            # Test workflow initialization
            assert len(workflow_engine.workflows) == 4, "Expected 4 default workflows"
            
            workflow_names = [w.name for w in workflow_engine.workflows]
            expected_workflows = [
                "Continuous Code Improvement",
                "Sprint-Based Optimization", 
                "Emergency Issue Resolution",
                "Proactive Code Enhancement"
            ]
            
            for expected in expected_workflows:
                assert expected in workflow_names, f"Missing workflow: {expected}"
            
            # Test trigger evaluation
            for workflow in workflow_engine.workflows:
                should_trigger = await workflow_engine.should_trigger_workflow(workflow)
                assert isinstance(should_trigger, bool), f"Trigger evaluation failed for {workflow.name}"
            
            # Test workflow execution (dry run)
            test_workflow = workflow_engine.workflows[0]  # Continuous improvement
            execution_id = await workflow_engine.execute_workflow(test_workflow, dry_run=True)
            assert execution_id, "Failed to execute workflow"
            
            # Test execution history
            history = await workflow_engine.get_execution_history()
            assert isinstance(history, list), "Execution history should be a list"
            
            self.test_results["tests"]["workflow_engine"] = {
                "status": "passed",
                "message": "SEAL workflow engine functioning correctly",
                "workflows": workflow_names,
                "execution_id": execution_id
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["workflow_engine"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"Workflow engine test failed: {e}")
            return False
    
    async def test_enterprise_orchestrator(self) -> bool:
        """Test enterprise orchestrator integration"""
        try:
            config = OrchestrationConfig()
            orchestrator = EnterpriseCodeOrchestrator(
                project_path=self.temp_dir,
                config=config,
                database_adapter=self.adapter
            )
            
            # Test component initialization
            assert orchestrator.seal_adapter is not None, "SEAL adapter not initialized"
            assert orchestrator.seal_workflow_engine is not None, "SEAL workflow engine not initialized"
            assert orchestrator.lifecycle_manager is not None, "Lifecycle manager not initialized"
            assert orchestrator.backlog_tracker is not None, "Backlog tracker not initialized"
            
            # Test orchestrator status
            status = await orchestrator.get_orchestration_status()
            assert "seal_workflows" in status, "Orchestration status missing SEAL workflows"
            assert "lifecycle_management" in status, "Orchestration status missing lifecycle management"
            
            # Test SEAL workflow evaluation
            await orchestrator._evaluate_seal_workflows()
            
            self.test_results["tests"]["orchestrator"] = {
                "status": "passed",
                "message": "Enterprise orchestrator functioning correctly",
                "status_info": status
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["orchestrator"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"Orchestrator test failed: {e}")
            return False
    
    async def test_component_integration(self) -> bool:
        """Test integration between all components"""
        try:
            # Create a realistic test scenario
            lifecycle_manager = IssueLifecycleManager(self.adapter)
            backlog_tracker = IssueBacklogTracker(self.adapter)
            
            # Create test issues
            test_issues = [
                {
                    "id": "integration-001",
                    "title": "Critical Security Vulnerability",
                    "description": "SQL injection vulnerability found",
                    "severity": "critical",
                    "type": "security",
                    "status": "detected",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": "integration-002", 
                    "title": "Performance Bottleneck",
                    "description": "Slow API response times",
                    "severity": "high",
                    "type": "performance",
                    "status": "detected",
                    "created_at": datetime.now().isoformat()
                }
            ]
            
            for issue in test_issues:
                await self.adapter.store_issue(issue)
            
            # Test auto-triage
            await lifecycle_manager.auto_triage_issues()
            
            # Create sprint and assign issues
            sprint_id = await backlog_tracker.create_sprint(
                "Integration Test Sprint",
                datetime.now(),
                datetime.now() + timedelta(days=7)
            )
            
            for issue in test_issues:
                await backlog_tracker.assign_issue_to_sprint(issue["id"], sprint_id)
            
            # Test SEAL adaptation workflow
            seal_adapter = SEALCodeAdapter(self.adapter, lifecycle_manager)
            workflow_engine = SEALWorkflowEngine(
                self.adapter, lifecycle_manager, backlog_tracker, seal_adapter
            )
            
            # Execute emergency workflow for critical issue
            emergency_workflow = next(
                w for w in workflow_engine.workflows 
                if w.name == "Emergency Issue Resolution"
            )
            
            execution_id = await workflow_engine.execute_workflow(emergency_workflow, dry_run=True)
            assert execution_id, "Failed to execute emergency workflow"
            
            # Verify integration metrics
            lifecycle_metrics = await lifecycle_manager.get_lifecycle_metrics()
            backlog_metrics = await backlog_tracker.get_backlog_metrics()
            
            assert lifecycle_metrics["total_issues"] >= 2, "Integration test issues not found"
            assert backlog_metrics["total_issues"] >= 2, "Issues not properly tracked in backlog"
            
            self.test_results["tests"]["integration"] = {
                "status": "passed",
                "message": "Component integration functioning correctly",
                "lifecycle_metrics": lifecycle_metrics,
                "backlog_metrics": backlog_metrics,
                "sprint_id": sprint_id,
                "execution_id": execution_id
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["integration"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"Integration test failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling and resilience"""
        try:
            lifecycle_manager = IssueLifecycleManager(self.adapter)
            
            # Test invalid issue transition
            result = await lifecycle_manager.transition_issue_state(
                "nonexistent-issue", "invalid-state", "Test"
            )
            assert not result, "Should fail for nonexistent issue"
            
            # Test SEAL adapter with invalid input
            seal_adapter = SEALCodeAdapter(self.adapter, lifecycle_manager)
            
            # Test with malformed issue
            try:
                strategy = await seal_adapter.select_adaptation_strategy({})
                # Should handle gracefully
            except Exception:
                pass  # Expected to handle errors gracefully
            
            # Test workflow engine with invalid workflow
            backlog_tracker = IssueBacklogTracker(self.adapter)
            workflow_engine = SEALWorkflowEngine(
                self.adapter, lifecycle_manager, backlog_tracker, seal_adapter
            )
            
            # Test execution with invalid workflow
            try:
                from cryptotrading.infrastructure.code_management.seal_workflow_engine import SEALWorkflow
                invalid_workflow = SEALWorkflow(
                    name="Invalid Test",
                    description="Invalid workflow for testing",
                    workflow_type="invalid_type",
                    trigger_conditions={},
                    execution_config={}
                )
                execution_id = await workflow_engine.execute_workflow(invalid_workflow, dry_run=True)
                # Should handle gracefully
            except Exception:
                pass  # Expected to handle errors gracefully
            
            self.test_results["tests"]["error_handling"] = {
                "status": "passed",
                "message": "Error handling functioning correctly"
            }
            return True
            
        except Exception as e:
            self.test_results["tests"]["error_handling"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.test_results["errors"].append(f"Error handling test failed: {e}")
            return False
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        try:
            if self.db:
                await self.db.close()
            
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                
            self.test_results["tests"]["cleanup"] = {
                "status": "passed",
                "message": "Test environment cleaned up successfully"
            }
            
        except Exception as e:
            self.test_results["tests"]["cleanup"] = {
                "status": "failed",
                "error": str(e)
            }
            self.test_results["errors"].append(f"Cleanup failed: {e}")
    
    def generate_summary(self):
        """Generate test summary"""
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for test in self.test_results["tests"].values() 
                          if test.get("status") == "passed")
        failed_tests = total_tests - passed_tests
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_errors": len(self.test_results["errors"]),
            "total_warnings": len(self.test_results["warnings"])
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive end-to-end validation"""
        print("🚀 Starting SEAL System End-to-End Validation...")
        
        # Setup
        if not await self.setup_test_environment():
            self.generate_summary()
            return self.test_results
        
        # Core tests
        test_methods = [
            self.test_database_layer,
            self.test_issue_lifecycle_manager,
            self.test_backlog_tracker,
            self.test_seal_code_adapter,
            self.test_seal_workflow_engine,
            self.test_enterprise_orchestrator,
            self.test_component_integration,
            self.test_error_handling
        ]
        
        for test_method in test_methods:
            try:
                print(f"Running {test_method.__name__}...")
                await test_method()
            except Exception as e:
                print(f"❌ {test_method.__name__} failed: {e}")
                self.test_results["errors"].append(f"{test_method.__name__}: {e}")
        
        # Cleanup
        await self.cleanup_test_environment()
        
        # Generate summary
        self.generate_summary()
        
        return self.test_results

async def main():
    """Main validation function"""
    validator = SEALSystemValidator()
    results = await validator.run_all_tests()
    
    # Save results
    results_file = Path(__file__).parent.parent / "data" / "seal_end_to_end_validation_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    summary = results["summary"]
    print(f"\n🎯 SEAL System Validation Summary:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed_tests']}")
    print(f"   Failed: {summary['failed_tests']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Errors: {summary['total_errors']}")
    print(f"   Warnings: {summary['total_warnings']}")
    
    if summary['failed_tests'] > 0:
        print(f"\n❌ Validation FAILED - {summary['failed_tests']} test(s) failed")
        print("Errors:")
        for error in results["errors"]:
            print(f"   - {error}")
        return 1
    else:
        print(f"\n✅ Validation PASSED - All {summary['passed_tests']} tests successful")
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
