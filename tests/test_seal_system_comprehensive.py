#!/usr/bin/env python3
"""
Comprehensive SEAL System Validation - Fixed Version
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
from typing import Dict, List, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import all SEAL components
from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase, DatabaseConfig, DatabaseMode
from cryptotrading.infrastructure.code_management.database_adapter import CodeManagementDatabaseAdapter
from cryptotrading.infrastructure.code_management.issue_lifecycle_manager import IssueLifecycleManager
from cryptotrading.infrastructure.code_management.issue_backlog_tracker import IssueBacklogTracker
from cryptotrading.infrastructure.code_management.seal_code_adapter import SEALCodeAdapter
from cryptotrading.infrastructure.code_management.seal_workflow_engine import SEALWorkflowEngine
from cryptotrading.infrastructure.code_management.enterprise_code_orchestrator import (
    EnterpriseCodeOrchestrator, OrchestrationConfig
)

async def test_database_initialization():
    """Test database initialization and basic operations"""
    print("🔧 Testing Database Initialization...")
    
    temp_dir = Path(tempfile.mkdtemp(prefix="seal_db_test_"))
    try:
        # Test UnifiedDatabase
        config = DatabaseConfig(
            mode=DatabaseMode.LOCAL,
            sqlite_path=str(temp_dir / "test.db"),
            redis_url=None,
            enable_caching=False
        )
        db = UnifiedDatabase(config)
        await db.initialize()
        
        # Test basic query
        result = await db.execute("SELECT 1 as test")
        assert result is not None, "Database query failed"
        
        # Test adapter
        adapter = CodeManagementDatabaseAdapter(db)
        
        # Test issue storage
        test_issue = {
            "id": "test-001",
            "title": "Test Issue",
            "description": "Test description",
            "severity": "medium",
            "type": "bug",
            "status": "detected",
            "created_at": datetime.now().isoformat()
        }
        
        await adapter.store_issue(test_issue)
        issues = await adapter.get_issues()
        
        assert len(issues) >= 1, "Issue storage/retrieval failed"
        print(f"✅ Database test passed - {len(issues)} issues stored")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        print(traceback.format_exc())
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

async def test_lifecycle_manager():
    """Test issue lifecycle management"""
    print("🔄 Testing Issue Lifecycle Manager...")
    
    temp_dir = Path(tempfile.mkdtemp(prefix="seal_lifecycle_test_"))
    try:
        config = DatabaseConfig(
            mode=DatabaseMode.LOCAL,
            sqlite_path=str(temp_dir / "test.db"),
            redis_url=None,
            enable_caching=False
        )
        db = UnifiedDatabase(config)
        await db.initialize()
        adapter = CodeManagementDatabaseAdapter(db)
        
        lifecycle_manager = IssueLifecycleManager(adapter)
        
        # Create test issue
        test_issue = {
            "id": "lifecycle-001",
            "title": "Lifecycle Test",
            "description": "Testing lifecycle transitions",
            "severity": "high",
            "type": "bug",
            "status": "detected",
            "created_at": datetime.now().isoformat()
        }
        
        await adapter.store_issue(test_issue)
        
        # Test state transition
        success = await lifecycle_manager.transition_issue_state(
            "lifecycle-001", "triaged", "Auto-triaged for testing"
        )
        assert success, "State transition failed"
        
        # Test metrics
        metrics = await lifecycle_manager.get_lifecycle_metrics()
        assert "total_issues" in metrics, "Lifecycle metrics missing"
        
        print(f"✅ Lifecycle test passed - {metrics['total_issues']} issues tracked")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ Lifecycle test failed: {e}")
        print(traceback.format_exc())
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

async def test_seal_adapter():
    """Test SEAL code adaptation engine"""
    print("🤖 Testing SEAL Code Adapter...")
    
    temp_dir = Path(tempfile.mkdtemp(prefix="seal_adapter_test_"))
    try:
        config = DatabaseConfig(
            mode=DatabaseMode.LOCAL,
            sqlite_path=str(temp_dir / "test.db"),
            redis_url=None,
            enable_caching=False
        )
        db = UnifiedDatabase(config)
        await db.initialize()
        adapter = CodeManagementDatabaseAdapter(db)
        lifecycle_manager = IssueLifecycleManager(adapter)
        
        seal_adapter = SEALCodeAdapter(adapter, lifecycle_manager)
        
        # Verify Grok configuration
        assert seal_adapter.config["model_name"] == "grok", "SEAL not configured for Grok"
        
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
        assert strategy is not None, "Strategy selection failed"
        assert "type" in strategy, "Strategy missing type"
        
        # Test self-edit application
        edit_result = await seal_adapter.apply_self_edit(
            test_issue,
            strategy,
            "# Test code\nprint('hello world')"
        )
        
        assert edit_result is not None, "Self-edit application failed"
        assert "confidence_score" in edit_result, "Edit result missing confidence score"
        
        print(f"✅ SEAL adapter test passed - Model: {seal_adapter.config['model_name']}")
        print(f"   Strategy: {strategy['type']}, Confidence: {edit_result['confidence_score']}")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ SEAL adapter test failed: {e}")
        print(traceback.format_exc())
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

async def test_workflow_engine():
    """Test SEAL workflow engine"""
    print("⚙️ Testing SEAL Workflow Engine...")
    
    temp_dir = Path(tempfile.mkdtemp(prefix="seal_workflow_test_"))
    try:
        config = DatabaseConfig(
            mode=DatabaseMode.LOCAL,
            sqlite_path=str(temp_dir / "test.db"),
            redis_url=None,
            enable_caching=False
        )
        db = UnifiedDatabase(config)
        await db.initialize()
        adapter = CodeManagementDatabaseAdapter(db)
        lifecycle_manager = IssueLifecycleManager(adapter)
        backlog_tracker = IssueBacklogTracker(adapter)
        seal_adapter = SEALCodeAdapter(adapter, lifecycle_manager)
        
        workflow_engine = SEALWorkflowEngine(
            adapter, lifecycle_manager, backlog_tracker, seal_adapter
        )
        
        # Test workflow initialization
        assert len(workflow_engine.workflows) == 4, f"Expected 4 workflows, got {len(workflow_engine.workflows)}"
        
        workflow_names = [w.name for w in workflow_engine.workflows]
        expected_workflows = [
            "Continuous Code Improvement",
            "Sprint-Based Optimization", 
            "Emergency Issue Resolution",
            "Proactive Code Enhancement"
        ]
        
        for expected in expected_workflows:
            assert expected in workflow_names, f"Missing workflow: {expected}"
        
        # Test workflow execution (dry run)
        test_workflow = workflow_engine.workflows[0]
        execution_id = await workflow_engine.execute_workflow(test_workflow, dry_run=True)
        assert execution_id is not None, "Workflow execution failed"
        
        print(f"✅ Workflow engine test passed - {len(workflow_names)} workflows available")
        print(f"   Workflows: {', '.join(workflow_names)}")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ Workflow engine test failed: {e}")
        print(traceback.format_exc())
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

async def test_enterprise_orchestrator():
    """Test enterprise orchestrator integration"""
    print("🏢 Testing Enterprise Orchestrator...")
    
    temp_dir = Path(tempfile.mkdtemp(prefix="seal_orchestrator_test_"))
    try:
        config = DatabaseConfig(
            mode=DatabaseMode.LOCAL,
            sqlite_path=str(temp_dir / "test.db"),
            redis_url=None,
            enable_caching=False
        )
        db = UnifiedDatabase(config)
        await db.initialize()
        adapter = CodeManagementDatabaseAdapter(db)
        
        config = OrchestrationConfig()
        orchestrator = EnterpriseCodeOrchestrator(
            project_path=temp_dir,
            config=config,
            database_adapter=adapter
        )
        
        # Test component initialization
        assert orchestrator.seal_adapter is not None, "SEAL adapter not initialized"
        assert orchestrator.seal_workflow_engine is not None, "SEAL workflow engine not initialized"
        assert orchestrator.lifecycle_manager is not None, "Lifecycle manager not initialized"
        assert orchestrator.backlog_tracker is not None, "Backlog tracker not initialized"
        
        # Test orchestrator status
        status = await orchestrator.get_orchestration_status()
        assert "seal_workflows" in status, "Status missing SEAL workflows"
        assert "lifecycle_management" in status, "Status missing lifecycle management"
        
        print("✅ Enterprise orchestrator test passed")
        print(f"   Components: SEAL adapter, workflow engine, lifecycle manager, backlog tracker")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ Enterprise orchestrator test failed: {e}")
        print(traceback.format_exc())
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

async def test_end_to_end_integration():
    """Test complete end-to-end integration"""
    print("🔗 Testing End-to-End Integration...")
    
    temp_dir = Path(tempfile.mkdtemp(prefix="seal_e2e_test_"))
    try:
        config = DatabaseConfig(
            mode=DatabaseMode.LOCAL,
            sqlite_path=str(temp_dir / "test.db"),
            redis_url=None,
            enable_caching=False
        )
        db = UnifiedDatabase(config)
        await db.initialize()
        adapter = CodeManagementDatabaseAdapter(db)
        
        # Initialize all components
        lifecycle_manager = IssueLifecycleManager(adapter)
        backlog_tracker = IssueBacklogTracker(adapter)
        seal_adapter = SEALCodeAdapter(adapter, lifecycle_manager)
        workflow_engine = SEALWorkflowEngine(adapter, lifecycle_manager, backlog_tracker, seal_adapter)
        
        # Create test scenario
        test_issues = [
            {
                "id": "e2e-001",
                "title": "Critical Security Issue",
                "description": "SQL injection vulnerability",
                "severity": "critical",
                "type": "security",
                "status": "detected",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "e2e-002",
                "title": "Performance Bottleneck",
                "description": "Slow API responses",
                "severity": "high",
                "type": "performance",
                "status": "detected",
                "created_at": datetime.now().isoformat()
            }
        ]
        
        # Store issues
        for issue in test_issues:
            await adapter.store_issue(issue)
        
        # Test auto-triage
        await lifecycle_manager.auto_triage_issues()
        
        # Create sprint
        sprint_id = await backlog_tracker.create_sprint(
            "E2E Test Sprint",
            datetime.now(),
            datetime.now() + timedelta(days=7)
        )
        
        # Assign issues to sprint
        for issue in test_issues:
            await backlog_tracker.assign_issue_to_sprint(issue["id"], sprint_id)
        
        # Execute emergency workflow
        emergency_workflow = next(
            w for w in workflow_engine.workflows 
            if w.name == "Emergency Issue Resolution"
        )
        
        execution_id = await workflow_engine.execute_workflow(emergency_workflow, dry_run=True)
        assert execution_id is not None, "Emergency workflow execution failed"
        
        # Verify metrics
        lifecycle_metrics = await lifecycle_manager.get_lifecycle_metrics()
        backlog_metrics = await backlog_tracker.get_backlog_metrics()
        
        assert lifecycle_metrics["total_issues"] >= 2, "Issues not tracked in lifecycle"
        assert backlog_metrics["total_issues"] >= 2, "Issues not tracked in backlog"
        
        print("✅ End-to-end integration test passed")
        print(f"   Issues processed: {lifecycle_metrics['total_issues']}")
        print(f"   Sprint created: {sprint_id}")
        print(f"   Emergency workflow executed: {execution_id}")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ End-to-end integration test failed: {e}")
        print(traceback.format_exc())
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

async def main():
    """Run comprehensive SEAL system validation"""
    print("🚀 SEAL System Comprehensive Validation")
    print("=" * 50)
    
    tests = [
        ("Database Initialization", test_database_initialization),
        ("Issue Lifecycle Manager", test_lifecycle_manager),
        ("SEAL Code Adapter", test_seal_adapter),
        ("SEAL Workflow Engine", test_workflow_engine),
        ("Enterprise Orchestrator", test_enterprise_orchestrator),
        ("End-to-End Integration", test_end_to_end_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 SEAL System Validation Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - SEAL System is fully operational!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed - System needs attention")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
