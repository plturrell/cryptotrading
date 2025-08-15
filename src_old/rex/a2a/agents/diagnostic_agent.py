"""
Diagnostic Strands Agent for rex.com
Analyzes system diagnostics and integrates with Windsurf MCP to trigger automated fixes
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..core.agent import StrandsAgent
from ..core.memory import WorkingMemory
from ...diagnostics.analyzer import system_analyzer
from ...diagnostics.logger import diagnostic_logger
from ...diagnostics.tracer import request_tracer


class DiagnosticAgent(StrandsAgent):
    """
    Intelligent diagnostic agent that:
    1. Continuously monitors system health
    2. Analyzes logs, traces, and metrics
    3. Identifies issues and root causes
    4. Suggests and triggers automated fixes via Windsurf MCP
    5. Learns from fix outcomes to improve future diagnostics
    """
    
    def __init__(self, name: str = "DiagnosticAgent"):
        super().__init__(name)
        
        # Diagnostic capabilities
        self.monitoring_interval = 60  # seconds
        self.analysis_history = []
        self.fix_attempts = []
        self.learning_memory = WorkingMemory()
        
        # MCP integration for automated fixes
        self.mcp_enabled = True
        self.auto_fix_enabled = True
        
        # Initialize diagnostic skills
        self.register_skills()
        
    def register_skills(self):
        """Register diagnostic and analysis skills"""
        
        @self.skill("analyze_system_health")
        async def analyze_system_health(self) -> Dict[str, Any]:
            """Perform comprehensive system health analysis"""
            try:
                with request_tracer.trace_operation("diagnostic_analysis"):
                    analysis = system_analyzer.analyze_system_health()
                    
                    # Store analysis in memory
                    self.analysis_history.append(analysis)
                    
                    # Keep only recent analyses
                    if len(self.analysis_history) > 100:
                        self.analysis_history = self.analysis_history[-100:]
                    
                    diagnostic_logger.log_performance(
                        'diagnostic_analysis_completed',
                        1.0,
                        {'status': analysis['overall_status']}
                    )
                    
                    return analysis
                    
            except Exception as e:
                diagnostic_logger.log_exception(e, "Failed to analyze system health")
                return {'error': str(e), 'timestamp': datetime.now().isoformat()}
        
        @self.skill("identify_critical_issues")
        async def identify_critical_issues(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Identify critical issues requiring immediate attention"""
            critical_issues = []
            
            for issue in analysis.get('issues', []):
                if issue.get('severity') == 'critical':
                    # Enrich issue with context
                    enriched_issue = {
                        **issue,
                        'analysis_timestamp': analysis['timestamp'],
                        'system_status': analysis['overall_status'],
                        'affected_components': [issue['component']],
                        'priority_score': self._calculate_priority_score(issue, analysis)
                    }
                    critical_issues.append(enriched_issue)
            
            # Sort by priority score
            critical_issues.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return critical_issues
        
        @self.skill("suggest_automated_fixes")
        async def suggest_automated_fixes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Suggest automated fixes for identified issues"""
            fixes = system_analyzer.suggest_fixes(analysis)
            
            # Enrich fixes with automation potential
            automated_fixes = []
            for fix in fixes:
                automation_score = self._assess_automation_potential(fix)
                
                if automation_score > 0.7:  # High confidence for automation
                    automated_fix = {
                        **fix,
                        'automation_score': automation_score,
                        'can_auto_execute': True,
                        'estimated_success_rate': self._estimate_success_rate(fix),
                        'rollback_plan': self._generate_rollback_plan(fix)
                    }
                    automated_fixes.append(automated_fix)
            
            return automated_fixes
        
        @self.skill("execute_windsurf_mcp_fix")
        async def execute_windsurf_mcp_fix(self, fix: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a fix using Windsurf MCP integration"""
            if not self.mcp_enabled:
                return {'status': 'disabled', 'message': 'MCP integration disabled'}
            
            fix_id = f"fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            try:
                # Log fix attempt
                diagnostic_logger.app_logger.info(f"EXECUTING_FIX: {fix_id} - {fix['title']}")
                
                # Prepare MCP command based on fix type
                mcp_command = self._prepare_mcp_command(fix)
                
                # Execute via MCP (simulated for now - would integrate with actual MCP)
                result = await self._execute_mcp_command(mcp_command)
                
                # Record fix attempt
                fix_attempt = {
                    'fix_id': fix_id,
                    'fix': fix,
                    'command': mcp_command,
                    'result': result,
                    'timestamp': datetime.now().isoformat(),
                    'success': result.get('success', False)
                }
                
                self.fix_attempts.append(fix_attempt)
                
                # Log result
                if result.get('success'):
                    diagnostic_logger.app_logger.info(f"FIX_SUCCESS: {fix_id}")
                else:
                    diagnostic_logger.app_logger.error(f"FIX_FAILED: {fix_id} - {result.get('error')}")
                
                return fix_attempt
                
            except Exception as e:
                diagnostic_logger.log_exception(e, f"Failed to execute fix {fix_id}")
                return {
                    'fix_id': fix_id,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        @self.skill("monitor_fix_outcomes")
        async def monitor_fix_outcomes(self, fix_attempt: Dict[str, Any]) -> Dict[str, Any]:
            """Monitor the outcome of a fix attempt"""
            fix_id = fix_attempt['fix_id']
            
            # Wait for system to stabilize
            await asyncio.sleep(30)
            
            # Re-analyze system health
            post_fix_analysis = await self.analyze_system_health()
            
            # Compare with pre-fix state
            improvement_detected = self._assess_improvement(
                fix_attempt['fix'],
                post_fix_analysis
            )
            
            outcome = {
                'fix_id': fix_id,
                'improvement_detected': improvement_detected,
                'post_fix_analysis': post_fix_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update learning memory
            self.learning_memory.store(
                f"fix_outcome_{fix_id}",
                {
                    'fix_type': fix_attempt['fix']['type'],
                    'original_issue': fix_attempt['fix']['description'],
                    'success': improvement_detected,
                    'lessons_learned': self._extract_lessons(fix_attempt, outcome)
                }
            )
            
            return outcome
        
        @self.skill("continuous_monitoring")
        async def continuous_monitoring(self):
            """Continuous system monitoring and automated response"""
            while True:
                try:
                    # Perform health analysis
                    analysis = await self.analyze_system_health()
                    
                    # Check for critical issues
                    critical_issues = await self.identify_critical_issues(analysis)
                    
                    if critical_issues and self.auto_fix_enabled:
                        # Suggest automated fixes
                        fixes = await self.suggest_automated_fixes(analysis)
                        
                        # Execute high-confidence fixes
                        for fix in fixes:
                            if fix.get('can_auto_execute') and fix.get('automation_score', 0) > 0.8:
                                fix_attempt = await self.execute_windsurf_mcp_fix(fix)
                                
                                # Monitor outcome
                                if fix_attempt.get('success'):
                                    await self.monitor_fix_outcomes(fix_attempt)
                    
                    # Wait before next monitoring cycle
                    await asyncio.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    diagnostic_logger.log_exception(e, "Error in continuous monitoring")
                    await asyncio.sleep(60)  # Wait longer on error
    
    def _calculate_priority_score(self, issue: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """Calculate priority score for an issue"""
        base_score = 0.5
        
        # Severity weighting
        severity_weights = {'critical': 1.0, 'warning': 0.6, 'info': 0.3}
        severity_score = severity_weights.get(issue.get('severity', 'info'), 0.3)
        
        # Component impact weighting
        component_weights = {
            'api': 0.9,
            'yahoo_finance': 0.8,
            'database': 0.9,
            'frontend': 0.7,
            'traces': 0.5
        }
        component_score = component_weights.get(issue.get('component', ''), 0.5)
        
        # System-wide impact
        system_impact = 1.0 if analysis.get('overall_status') == 'critical' else 0.7
        
        return base_score * severity_score * component_score * system_impact
    
    def _assess_automation_potential(self, fix: Dict[str, Any]) -> float:
        """Assess how suitable a fix is for automation"""
        base_score = 0.5
        
        # Fix type scoring
        type_scores = {
            'code_fix': 0.8,
            'integration_fix': 0.9,
            'config_fix': 0.95,
            'dependency_fix': 0.7
        }
        
        type_score = type_scores.get(fix.get('type', ''), 0.5)
        
        # Risk assessment
        risk_factors = len(fix.get('files_to_check', []))
        risk_penalty = min(risk_factors * 0.1, 0.3)
        
        return max(0.0, min(1.0, base_score + type_score - risk_penalty))
    
    def _estimate_success_rate(self, fix: Dict[str, Any]) -> float:
        """Estimate success rate based on historical data"""
        # Check learning memory for similar fixes
        similar_fixes = self.learning_memory.search(f"fix_type:{fix.get('type', '')}")
        
        if similar_fixes:
            success_count = sum(1 for f in similar_fixes if f.get('success', False))
            return success_count / len(similar_fixes)
        
        # Default estimates based on fix type
        default_rates = {
            'code_fix': 0.7,
            'integration_fix': 0.8,
            'config_fix': 0.9,
            'dependency_fix': 0.6
        }
        
        return default_rates.get(fix.get('type', ''), 0.5)
    
    def _generate_rollback_plan(self, fix: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a rollback plan for a fix"""
        return {
            'backup_files': fix.get('files_to_check', []),
            'rollback_commands': [
                'git stash',
                'git checkout HEAD~1',
                'restart services'
            ],
            'verification_steps': [
                'check system health',
                'verify API endpoints',
                'test core functionality'
            ]
        }
    
    def _prepare_mcp_command(self, fix: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare MCP command for executing a fix"""
        if fix['type'] == 'integration_fix' and 'JSON serialization' in fix['description']:
            return {
                'action': 'edit_file',
                'file': 'app.py',
                'operation': 'add_custom_json_encoder',
                'details': {
                    'add_import': 'from flask.json import JSONEncoder',
                    'add_class': 'CustomJSONEncoder',
                    'modify_app_config': 'app.json_encoder = CustomJSONEncoder'
                }
            }
        
        return {
            'action': 'analyze_and_suggest',
            'context': fix,
            'request': 'Please analyze and suggest specific code changes'
        }
    
    async def _execute_mcp_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP command (simulated - would integrate with actual MCP)"""
        # Simulate MCP execution
        await asyncio.sleep(2)
        
        # For now, return success for demonstration
        return {
            'success': True,
            'message': f"Executed {command['action']} successfully",
            'changes_made': command.get('details', {}),
            'timestamp': datetime.now().isoformat()
        }
    
    def _assess_improvement(self, fix: Dict[str, Any], post_analysis: Dict[str, Any]) -> bool:
        """Assess if a fix improved the system"""
        # Simple heuristic: check if overall status improved
        return post_analysis.get('overall_status') in ['healthy', 'warning']
    
    def _extract_lessons(self, fix_attempt: Dict[str, Any], outcome: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from fix attempt"""
        lessons = []
        
        if outcome['improvement_detected']:
            lessons.append(f"Fix type '{fix_attempt['fix']['type']}' was effective")
            lessons.append("Automated execution successful")
        else:
            lessons.append(f"Fix type '{fix_attempt['fix']['type']}' needs refinement")
            lessons.append("Consider manual intervention for similar issues")
        
        return lessons
    
    async def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        current_analysis = await self.analyze_system_health()
        
        return {
            'current_status': current_analysis,
            'recent_analyses': self.analysis_history[-10:],
            'fix_attempts': self.fix_attempts[-20:],
            'learning_insights': self.learning_memory.get_all(),
            'agent_metrics': {
                'total_analyses': len(self.analysis_history),
                'total_fix_attempts': len(self.fix_attempts),
                'success_rate': self._calculate_success_rate(),
                'monitoring_uptime': self.monitoring_interval
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall fix success rate"""
        if not self.fix_attempts:
            return 0.0
        
        successful_fixes = sum(1 for attempt in self.fix_attempts if attempt.get('success', False))
        return successful_fixes / len(self.fix_attempts)


# Global diagnostic agent instance
diagnostic_agent = DiagnosticAgent()
