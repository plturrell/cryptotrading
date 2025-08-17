"""
Integration tests for Enterprise Code Orchestrator using real components
Tests the full system with actual database and file operations
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

from cryptotrading.infrastructure.code_management.enterprise_code_orchestrator import (
    EnterpriseCodeOrchestrator, OrchestrationConfig
)
from cryptotrading.infrastructure.code_management.database_adapter import (
    CodeManagementDatabaseAdapter, CodeManagementMode
)
from cryptotrading.infrastructure.code_management.intelligent_code_manager import Issue
from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase
from cryptotrading.core.config.production_config import DatabaseConfig

@pytest.fixture
async def test_project_path():
    """Create a test project directory with sample code"""
    test_dir = Path(tempfile.mkdtemp())
    
    # Create project structure
    src_dir = test_dir / "src"
    src_dir.mkdir()
    
    # Add sample Python files
    (src_dir / "main.py").write_text("""
import os
import sys

def main():
    # TODO: Add proper error handling
    result = process_data()
    print(result)
    
def process_data():
    data = [1, 2, 3, 4, 5]
    # Missing docstring
    return sum(data)
    
if __name__ == "__main__":
    main()
""")
    
    (src_dir / "utils.py").write_text("""
def calculate_average(numbers):
    # Potential division by zero
    return sum(numbers) / len(numbers)
    
def unused_function():
    # This function is never called
    pass
    
class DataProcessor:
    def __init__(self):
        self.data = []
        
    def process(self, item):
        # Missing error handling
        self.data.append(item)
        return item * 2
""")
    
    # Add test file
    tests_dir = test_dir / "tests"
    tests_dir.mkdir()
    
    (tests_dir / "test_main.py").write_text("""
import pytest
from src.main import process_data

def test_process_data():
    result = process_data()
    assert result == 15
""")
    
    # Add config files
    (test_dir / ".gitignore").write_text("*.pyc\n__pycache__/\n.coverage\n")
    (test_dir / "README.md").write_text("# Test Project\n\nSample project for testing")
    
    # Create data directory
    (test_dir / "data").mkdir()
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(test_dir)

@pytest.fixture
async def test_database():
    """Create test database for code management"""
    db_config = DatabaseConfig(
        host="localhost",
        database=":memory:",
        connection_pool_size=5
    )
    
    db = UnifiedDatabase(db_config)
    await db.initialize()
    
    # Create code management tables
    async with db.pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS code_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INTEGER,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS code_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                component TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    yield db
    
    await db.close()

@pytest.fixture
async def database_adapter(test_database):
    """Create database adapter for code management"""
    config = {
        "mode": CodeManagementMode.LOCAL,
        "database_url": "sqlite:///:memory:"
    }
    
    adapter = CodeManagementDatabaseAdapter(config)
    adapter.db = test_database  # Use test database
    
    yield adapter

@pytest.fixture
async def orchestrator(test_project_path, database_adapter):
    """Create enterprise orchestrator with test configuration"""
    config = OrchestrationConfig(
        continuous_monitoring_interval=1,  # 1 second for tests
        quality_check_interval=1,
        proactive_scan_interval=1,
        dashboard_port=5555,  # Different port for tests
        auto_fix_enabled=True,
        notification_enabled=False,
        max_auto_fixes_per_cycle=5
    )
    
    orchestrator = EnterpriseCodeOrchestrator(
        project_path=test_project_path,
        config=config,
        database_adapter=database_adapter
    )
    
    yield orchestrator
    
    # Ensure cleanup
    await orchestrator.stop_monitoring()

class TestEnterpriseOrchestrator:
    """Integration tests for enterprise orchestrator"""
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, orchestrator):
        """Test health monitoring with real code analysis"""
        # Run health check
        health_metrics = await orchestrator.code_manager.comprehensive_health_check()
        
        # Verify metrics
        assert health_metrics.total_files > 0
        assert health_metrics.total_lines > 0
        assert health_metrics.issues_count >= 0
        assert 0 <= health_metrics.coverage_percentage <= 100
        assert 0 <= health_metrics.technical_debt_score <= 100
        
        # Check file analysis
        assert "main.py" in [Path(f).name for f in health_metrics.files_analyzed]
        assert "utils.py" in [Path(f).name for f in health_metrics.files_analyzed]
        
    @pytest.mark.asyncio
    async def test_quality_monitoring(self, orchestrator):
        """Test quality monitoring detects real issues"""
        # Run quality checks
        results = await orchestrator.quality_monitor.run_all_checks()
        issues = orchestrator.quality_monitor.process_results(results)
        
        # Should detect issues in sample code
        assert len(issues) > 0
        
        # Check issue types
        issue_types = {issue.issue_type for issue in issues}
        expected_types = {"missing_docstring", "todo_comment", "potential_bug"}
        
        # Should find at least some of these
        assert len(issue_types & expected_types) > 0
        
        # Verify issue details
        for issue in issues:
            assert issue.file_path
            assert issue.description
            assert issue.severity in ["low", "medium", "high", "critical"]
            
    @pytest.mark.asyncio
    async def test_proactive_detection(self, orchestrator):
        """Test proactive issue detection"""
        # Run proactive scan
        issues = await orchestrator.proactive_detector.scan_project()
        
        # Should detect potential issues
        assert len(issues) > 0
        
        # Check for specific issues
        issue_descriptions = [issue.description for issue in issues]
        
        # Should detect division by zero risk
        assert any("division" in desc.lower() for desc in issue_descriptions)
        
        # Should detect unused code
        assert any("unused" in desc.lower() for desc in issue_descriptions)
        
    @pytest.mark.asyncio
    async def test_database_persistence(self, orchestrator, database_adapter):
        """Test that issues are persisted to database"""
        # Run quality check
        results = await orchestrator.quality_monitor.run_all_checks()
        issues = orchestrator.quality_monitor.process_results(results)
        
        # Store issues
        for issue in issues[:3]:  # Store first 3
            await database_adapter.store_issue(issue)
            
        # Retrieve issues
        stored_issues = await database_adapter.get_issues()
        assert len(stored_issues) >= 3
        
        # Verify issue data
        stored_types = {issue["issue_type"] for issue in stored_issues}
        original_types = {issue.issue_type for issue in issues[:3]}
        assert stored_types == original_types
        
    @pytest.mark.asyncio
    async def test_concurrent_monitoring(self, orchestrator):
        """Test concurrent execution of monitoring tasks"""
        # Start monitoring tasks concurrently
        tasks = [
            orchestrator._start_health_monitoring(),
            orchestrator._start_quality_monitoring(),
            orchestrator._start_proactive_scanning()
        ]
        
        # Run for a short time
        monitor_task = asyncio.create_task(asyncio.gather(*tasks))
        
        # Let it run
        await asyncio.sleep(2)
        
        # Stop monitoring
        orchestrator.running = False
        monitor_task.cancel()
        
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
            
        # Verify metrics were collected
        metrics = orchestrator.metrics.get_all()
        assert metrics.get("last_health_check") is not None
        assert metrics.get("last_quality_check") is not None
        assert metrics.get("last_proactive_scan") is not None
        
    @pytest.mark.asyncio
    async def test_auto_fix_functionality(self, orchestrator, test_project_path):
        """Test auto-fix functionality with real file modifications"""
        # Create a file with fixable issues
        fixable_file = test_project_path / "src" / "fixable.py"
        fixable_file.write_text("""
# File with fixable issues
import os, sys  # Multiple imports on one line

def calculate(x,y):  # Missing spaces after commas
    return x+y  # Missing spaces around operator
    
list=[1,2,3,4,5]  # Missing spaces
dict={'a':1,'b':2}  # Missing spaces
""")
        
        # Run quality check
        results = await orchestrator.quality_monitor.run_all_checks()
        issues = orchestrator.quality_monitor.process_results(results)
        
        # Filter fixable issues
        fixable_issues = [i for i in issues if i.auto_fixable and "fixable.py" in i.file_path]
        
        if fixable_issues:
            # Apply fixes
            fixed = await orchestrator.quality_monitor.auto_fix_issues(fixable_issues[:2])
            
            # Verify file was modified
            updated_content = fixable_file.read_text()
            
            # Should have better formatting
            assert updated_content != """
# File with fixable issues
import os, sys  # Multiple imports on one line

def calculate(x,y):  # Missing spaces after commas
    return x+y  # Missing spaces around operator
    
list=[1,2,3,4,5]  # Missing spaces
dict={'a':1,'b':2}  # Missing spaces
"""
            
    @pytest.mark.asyncio
    async def test_event_system(self, orchestrator):
        """Test event system integration"""
        events_received = []
        
        # Subscribe to events
        def event_handler(event):
            events_received.append(event)
            
        orchestrator.subscribe_to_event("health_check_completed", event_handler)
        orchestrator.subscribe_to_event("quality_check_completed", event_handler)
        
        # Trigger health check
        await orchestrator._start_health_monitoring()
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        orchestrator.running = False
        
        # Should have received events
        assert len(events_received) > 0
        
        # Verify event structure
        for event in events_received:
            assert "type" in event
            assert "timestamp" in event
            assert "data" in event
            
    @pytest.mark.asyncio
    async def test_orchestration_report(self, orchestrator):
        """Test comprehensive orchestration report generation"""
        # Run some monitoring cycles
        orchestrator.running = True
        
        # Run health check
        await orchestrator._start_health_monitoring()
        
        # Generate report
        report = await orchestrator._generate_orchestration_report()
        
        # Verify report structure
        assert "timestamp" in report
        assert "system_status" in report
        assert "health_dashboard" in report
        assert "quality_summary" in report
        assert "proactive_summary" in report
        assert "config" in report
        
        # Verify data
        assert report["system_status"] == "running"
        assert report["total_fixes_applied"] >= 0
        
        # Stop monitoring
        orchestrator.running = False
        
    @pytest.mark.asyncio
    async def test_state_persistence(self, orchestrator, test_project_path):
        """Test orchestration state persistence"""
        # Generate and save state
        report = await orchestrator._generate_orchestration_report()
        await orchestrator._save_orchestration_state(report)
        
        # Verify file was created
        state_file = test_project_path / "data" / "orchestration_state.json"
        assert state_file.exists()
        
        # Load and verify content
        with open(state_file, "r") as f:
            saved_state = json.load(f)
            
        assert saved_state["timestamp"] == report["timestamp"]
        assert saved_state["system_status"] == report["system_status"]
        
    @pytest.mark.asyncio
    async def test_lifecycle_integration(self, orchestrator, database_adapter):
        """Test issue lifecycle management integration"""
        # Skip if no lifecycle manager
        if not orchestrator._lifecycle_manager:
            pytest.skip("Lifecycle manager not available")
            
        # Create some issues
        issues = [
            Issue(
                issue_type="bug",
                severity="high",
                file_path="src/main.py",
                line_number=10,
                description="Critical bug found",
                auto_fixable=True
            ),
            Issue(
                issue_type="code_smell",
                severity="medium",
                file_path="src/utils.py",
                line_number=5,
                description="Code improvement needed",
                auto_fixable=False
            )
        ]
        
        # Store issues
        for issue in issues:
            await database_adapter.store_issue(issue)
            
        # Run lifecycle management
        await orchestrator._start_lifecycle_management()
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        orchestrator.running = False
        
        # Verify metrics
        metrics = orchestrator.metrics.get_all()
        assert "issues_triaged" in metrics
        assert "backlog_size" in metrics