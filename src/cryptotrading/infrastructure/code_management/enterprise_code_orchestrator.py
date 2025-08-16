"""
Enterprise Code Management Orchestrator
Central orchestrator that coordinates all code management components for enterprise-grade operation
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .intelligent_code_manager import IntelligentCodeManager, CodeHealthMetrics
from .automated_quality_monitor import AutomatedQualityMonitor
from .proactive_issue_detector import ProactiveIssueDetector
from .code_health_dashboard import CodeHealthDashboard

logger = logging.getLogger(__name__)

@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration system"""
    continuous_monitoring_interval: int = 1800  # 30 minutes
    quality_check_interval: int = 900  # 15 minutes
    proactive_scan_interval: int = 3600  # 1 hour
    dashboard_port: int = 5001
    auto_fix_enabled: bool = True
    notification_enabled: bool = True
    max_auto_fixes_per_cycle: int = 10

class EnterpriseCodeOrchestrator:
    """Enterprise orchestrator for comprehensive code management"""
    
    def __init__(self, project_path: str, config: Optional[OrchestrationConfig] = None):
        self.project_path = Path(project_path)
        self.config = config or OrchestrationConfig()
        
        # Initialize all components
        self.code_manager = IntelligentCodeManager(project_path)
        self.quality_monitor = AutomatedQualityMonitor(project_path)
        self.proactive_detector = ProactiveIssueDetector(project_path)
        self.dashboard = CodeHealthDashboard(project_path, self.config.dashboard_port)
        
        # State tracking
        self.is_running = False
        self.last_health_check = None
        self.last_quality_check = None
        self.last_proactive_scan = None
        self.total_fixes_applied = 0
        
    async def start_enterprise_monitoring(self) -> None:
        """Start the complete enterprise monitoring system"""
        logger.info("🚀 Starting Enterprise Code Management System...")
        
        self.is_running = True
        
        # Start all monitoring tasks concurrently
        tasks = [
            asyncio.create_task(self._continuous_health_monitoring()),
            asyncio.create_task(self._quality_monitoring_loop()),
            asyncio.create_task(self._proactive_scanning_loop()),
            asyncio.create_task(self._dashboard_server()),
            asyncio.create_task(self._orchestration_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error("Error in enterprise monitoring: %s", e)
            self.is_running = False
    
    async def _continuous_health_monitoring(self) -> None:
        """Continuous health monitoring loop"""
        logger.info("🏥 Starting continuous health monitoring...")
        
        while self.is_running:
            try:
                # Run comprehensive health check
                health_metrics = await self.code_manager.comprehensive_health_check()
                self.last_health_check = datetime.now()
                
                # Log health status
                logger.info("Health check complete - Coverage: %.1f%%, Score: %.1f", 
                           health_metrics.coverage_percentage, 
                           health_metrics.technical_debt_score)
                
                # Check for critical health issues
                await self._handle_critical_health_issues(health_metrics)
                
                # Wait for next cycle
                await asyncio.sleep(self.config.continuous_monitoring_interval)
                
            except Exception as e:
                logger.error("Error in health monitoring: %s", e)
                await asyncio.sleep(60)
    
    async def _quality_monitoring_loop(self) -> None:
        """Quality monitoring loop"""
        logger.info("🔍 Starting quality monitoring loop...")
        
        while self.is_running:
            try:
                # Run quality checks
                results = await self.quality_monitor.run_all_checks()
                issues = self.quality_monitor.process_results(results)
                
                self.last_quality_check = datetime.now()
                
                # Auto-fix issues if enabled
                if self.config.auto_fix_enabled and issues:
                    fixed_issues = await self.quality_monitor.auto_fix_issues(issues[:self.config.max_auto_fixes_per_cycle])
                    self.total_fixes_applied += len(fixed_issues)
                    
                    if fixed_issues:
                        logger.info("Auto-fixed %d quality issues", len(fixed_issues))
                
                # Wait for next cycle
                await asyncio.sleep(self.config.quality_check_interval)
                
            except Exception as e:
                logger.error("Error in quality monitoring: %s", e)
                await asyncio.sleep(60)
    
    async def _proactive_scanning_loop(self) -> None:
        """Proactive issue scanning loop"""
        logger.info("🔮 Starting proactive scanning loop...")
        
        while self.is_running:
            try:
                # Run proactive scan
                issues = await self.proactive_detector.scan_project()
                self.last_proactive_scan = datetime.now()
                
                # Auto-fix proactive issues if enabled
                if self.config.auto_fix_enabled and issues:
                    auto_fixable = [i for i in issues if i.auto_fixable]
                    if auto_fixable:
                        fixed_issues = await self.proactive_detector.auto_fix_issues(auto_fixable[:self.config.max_auto_fixes_per_cycle])
                        self.total_fixes_applied += len(fixed_issues)
                        
                        if fixed_issues:
                            logger.info("Auto-fixed %d proactive issues", len(fixed_issues))
                
                # Wait for next cycle
                await asyncio.sleep(self.config.proactive_scan_interval)
                
            except Exception as e:
                logger.error("Error in proactive scanning: %s", e)
                await asyncio.sleep(60)
    
    async def _dashboard_server(self) -> None:
        """Run the dashboard server"""
        logger.info("📊 Starting dashboard server on port %d", self.config.dashboard_port)
        
        # Run dashboard in a separate thread to avoid blocking
        import threading
        
        def run_dashboard():
            self.dashboard.run(debug=False)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Keep the task alive
        while self.is_running:
            await asyncio.sleep(10)
    
    async def _orchestration_loop(self) -> None:
        """Main orchestration loop for coordination"""
        logger.info("🎼 Starting orchestration loop...")
        
        while self.is_running:
            try:
                # Generate orchestration report
                report = await self._generate_orchestration_report()
                
                # Save orchestration state
                await self._save_orchestration_state(report)
                
                # Check for system-wide issues
                await self._check_system_health()
                
                # Coordinate between components
                await self._coordinate_components()
                
                # Wait for next orchestration cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error("Error in orchestration loop: %s", e)
                await asyncio.sleep(60)
    
    async def _handle_critical_health_issues(self, health_metrics: CodeHealthMetrics) -> None:
        """Handle critical health issues"""
        critical_threshold = 70.0  # Coverage below 70% is critical
        
        if health_metrics.coverage_percentage < critical_threshold:
            logger.warning("🚨 CRITICAL: Code coverage below threshold (%.1f%% < %.1f%%)", 
                          health_metrics.coverage_percentage, critical_threshold)
            
            # Trigger immediate comprehensive scan
            await self._trigger_emergency_scan()
        
        if health_metrics.technical_debt_score < 60.0:
            logger.warning("🚨 CRITICAL: Technical debt score too low (%.1f)", 
                          health_metrics.technical_debt_score)
            
            # Trigger debt reduction recommendations
            await self._trigger_debt_reduction()
    
    async def _trigger_emergency_scan(self) -> None:
        """Trigger emergency comprehensive scan"""
        logger.info("🚨 Triggering emergency comprehensive scan...")
        
        # Run all scans immediately
        await asyncio.gather(
            self.quality_monitor.run_all_checks(),
            self.proactive_detector.scan_project(),
            self.code_manager.comprehensive_health_check()
        )
    
    async def _trigger_debt_reduction(self) -> None:
        """Trigger technical debt reduction process"""
        logger.info("🔧 Triggering technical debt reduction process...")
        
        # Generate refactoring recommendations
        recommendations = await self.code_manager.generate_refactoring_recommendations()
        
        # Log recommendations
        for rec in recommendations[:5]:  # Top 5 recommendations
            logger.info("💡 Refactoring recommendation: %s", rec.get("suggested_refactoring", ""))
    
    async def _generate_orchestration_report(self) -> Dict[str, Any]:
        """Generate comprehensive orchestration report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "running" if self.is_running else "stopped",
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_quality_check": self.last_quality_check.isoformat() if self.last_quality_check else None,
            "last_proactive_scan": self.last_proactive_scan.isoformat() if self.last_proactive_scan else None,
            "total_fixes_applied": self.total_fixes_applied,
            "health_dashboard": self.code_manager.get_health_dashboard(),
            "quality_summary": self.quality_monitor.get_quality_summary(),
            "proactive_summary": self.proactive_detector.get_issue_summary(),
            "config": asdict(self.config)
        }
    
    async def _save_orchestration_state(self, report: Dict[str, Any]) -> None:
        """Save orchestration state to file"""
        state_file = self.project_path / "data" / "orchestration_state.json"
        state_file.parent.mkdir(exist_ok=True)
        
        with open(state_file, "w") as f:
            json.dump(report, f, indent=2)
    
    async def _check_system_health(self) -> None:
        """Check overall system health"""
        now = datetime.now()
        
        # Check if components are running on schedule
        if self.last_health_check and (now - self.last_health_check).seconds > self.config.continuous_monitoring_interval * 2:
            logger.warning("⚠️ Health monitoring appears to be lagging")
        
        if self.last_quality_check and (now - self.last_quality_check).seconds > self.config.quality_check_interval * 2:
            logger.warning("⚠️ Quality monitoring appears to be lagging")
        
        if self.last_proactive_scan and (now - self.last_proactive_scan).seconds > self.config.proactive_scan_interval * 2:
            logger.warning("⚠️ Proactive scanning appears to be lagging")
    
    async def _coordinate_components(self) -> None:
        """Coordinate between different components"""
        # Share detected issues between components
        proactive_issues = self.proactive_detector.detected_issues
        quality_issues = getattr(self.quality_monitor, 'last_issues', [])
        
        # Merge issues into code manager
        all_issues = proactive_issues + quality_issues
        self.code_manager.issues_db.extend(all_issues)
        
        # Remove duplicates (simple deduplication by description)
        seen_descriptions = set()
        unique_issues = []
        for issue in self.code_manager.issues_db:
            if issue.description not in seen_descriptions:
                unique_issues.append(issue)
                seen_descriptions.add(issue.description)
        
        self.code_manager.issues_db = unique_issues
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring processes"""
        logger.info("🛑 Stopping Enterprise Code Management System...")
        self.is_running = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "is_running": self.is_running,
            "uptime": (datetime.now() - (self.last_health_check or datetime.now())).total_seconds(),
            "total_fixes_applied": self.total_fixes_applied,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_quality_check": self.last_quality_check.isoformat() if self.last_quality_check else None,
            "last_proactive_scan": self.last_proactive_scan.isoformat() if self.last_proactive_scan else None,
            "dashboard_url": f"http://localhost:{self.config.dashboard_port}",
            "config": asdict(self.config)
        }

# CLI Interface
async def main():
    """Main CLI interface for the enterprise code orchestrator"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Enterprise Code Management Orchestrator")
    parser.add_argument("project_path", help="Path to the project to monitor")
    parser.add_argument("--port", type=int, default=5001, help="Dashboard port")
    parser.add_argument("--health-interval", type=int, default=1800, help="Health check interval (seconds)")
    parser.add_argument("--quality-interval", type=int, default=900, help="Quality check interval (seconds)")
    parser.add_argument("--proactive-interval", type=int, default=3600, help="Proactive scan interval (seconds)")
    parser.add_argument("--no-auto-fix", action="store_true", help="Disable automatic fixing")
    parser.add_argument("--max-fixes", type=int, default=10, help="Maximum auto-fixes per cycle")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(args.project_path) / "data" / "orchestrator.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create configuration
    config = OrchestrationConfig(
        continuous_monitoring_interval=args.health_interval,
        quality_check_interval=args.quality_interval,
        proactive_scan_interval=args.proactive_interval,
        dashboard_port=args.port,
        auto_fix_enabled=not args.no_auto_fix,
        max_auto_fixes_per_cycle=args.max_fixes
    )
    
    # Create and start orchestrator
    orchestrator = EnterpriseCodeOrchestrator(args.project_path, config)
    
    try:
        print(f"🚀 Starting Enterprise Code Management System")
        print(f"📊 Dashboard: http://localhost:{args.port}")
        print(f"📁 Project: {args.project_path}")
        print(f"🔧 Auto-fix: {'Enabled' if config.auto_fix_enabled else 'Disabled'}")
        print("Press Ctrl+C to stop...")
        
        await orchestrator.start_enterprise_monitoring()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        await orchestrator.stop_monitoring()
        print("✅ System stopped")

if __name__ == "__main__":
    asyncio.run(main())
