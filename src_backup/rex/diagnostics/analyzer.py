"""
System analyzer for rex.com diagnostics
Analyzes logs, traces, and system state to identify issues and suggest fixes
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd
from .logger import diagnostic_logger
from .tracer import request_tracer


class SystemAnalyzer:
    """
    Intelligent system analyzer that processes logs, traces, and metrics
    to identify patterns, issues, and suggest actionable fixes
    """
    
    def __init__(self, log_dir: str = "logs/diagnostics"):
        self.log_dir = Path(log_dir)
        self.analysis_cache = {}
        
    def analyze_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health analysis"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'components': {},
            'issues': [],
            'recommendations': [],
            'metrics': {}
        }
        
        # Analyze different components
        analysis['components']['api'] = self._analyze_api_health()
        analysis['components']['yahoo_finance'] = self._analyze_yahoo_finance_health()
        analysis['components']['frontend'] = self._analyze_frontend_health()
        analysis['components']['database'] = self._analyze_database_health()
        
        # Analyze traces
        analysis['components']['traces'] = self._analyze_trace_health()
        
        # Determine overall status
        component_statuses = [comp.get('status', 'unknown') for comp in analysis['components'].values()]
        if 'critical' in component_statuses:
            analysis['overall_status'] = 'critical'
        elif 'warning' in component_statuses:
            analysis['overall_status'] = 'warning'
        elif all(status == 'healthy' for status in component_statuses):
            analysis['overall_status'] = 'healthy'
        else:
            analysis['overall_status'] = 'degraded'
        
        # Collect all issues and recommendations
        for component_name, component_data in analysis['components'].items():
            if 'issues' in component_data:
                analysis['issues'].extend([
                    {'component': component_name, **issue} 
                    for issue in component_data['issues']
                ])
            if 'recommendations' in component_data:
                analysis['recommendations'].extend([
                    {'component': component_name, **rec} 
                    for rec in component_data['recommendations']
                ])
        
        return analysis
    
    def _analyze_api_health(self) -> Dict[str, Any]:
        """Analyze API endpoint health"""
        api_log_file = self.log_dir / 'api.log'
        if not api_log_file.exists():
            return {'status': 'unknown', 'message': 'No API logs found'}
        
        # Parse recent API logs
        recent_logs = self._parse_recent_logs(api_log_file, hours=1)
        
        # Analyze error rates
        total_requests = len(recent_logs)
        error_requests = len([log for log in recent_logs if 'error' in log or 'status_code' in log and int(log.get('status_code', 200)) >= 400])
        
        error_rate = error_requests / total_requests if total_requests > 0 else 0
        
        # Analyze response times
        response_times = [float(log.get('duration_seconds', 0)) for log in recent_logs if 'duration_seconds' in log]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Identify problematic endpoints
        endpoint_errors = defaultdict(int)
        for log in recent_logs:
            if 'error' in log or ('status_code' in log and int(log.get('status_code', 200)) >= 400):
                endpoint = log.get('endpoint', 'unknown')
                endpoint_errors[endpoint] += 1
        
        issues = []
        recommendations = []
        
        # Determine status
        if error_rate > 0.5:
            status = 'critical'
            issues.append({
                'severity': 'critical',
                'message': f'High error rate: {error_rate:.1%}',
                'details': f'{error_requests}/{total_requests} requests failing'
            })
            recommendations.append({
                'priority': 'high',
                'action': 'investigate_api_errors',
                'message': 'Investigate API errors immediately',
                'details': dict(endpoint_errors)
            })
        elif error_rate > 0.1:
            status = 'warning'
            issues.append({
                'severity': 'warning',
                'message': f'Elevated error rate: {error_rate:.1%}',
                'details': f'{error_requests}/{total_requests} requests failing'
            })
        elif avg_response_time > 5.0:
            status = 'warning'
            issues.append({
                'severity': 'warning',
                'message': f'Slow response times: {avg_response_time:.2f}s average',
                'details': 'API responses are slower than expected'
            })
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'metrics': {
                'total_requests': total_requests,
                'error_rate': error_rate,
                'avg_response_time': avg_response_time,
                'problematic_endpoints': dict(endpoint_errors)
            },
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _analyze_yahoo_finance_health(self) -> Dict[str, Any]:
        """Analyze Yahoo Finance integration health"""
        yahoo_log_file = self.log_dir / 'yahoo_finance.log'
        if not yahoo_log_file.exists():
            return {'status': 'unknown', 'message': 'No Yahoo Finance logs found'}
        
        recent_logs = self._parse_recent_logs(yahoo_log_file, hours=1)
        
        total_operations = len(recent_logs)
        failed_operations = len([log for log in recent_logs if not log.get('success', True)])
        
        failure_rate = failed_operations / total_operations if total_operations > 0 else 0
        
        # Analyze operation types
        operation_stats = Counter([log.get('operation', 'unknown') for log in recent_logs])
        
        issues = []
        recommendations = []
        
        if failure_rate > 0.3:
            status = 'critical'
            issues.append({
                'severity': 'critical',
                'message': f'High Yahoo Finance failure rate: {failure_rate:.1%}',
                'details': f'{failed_operations}/{total_operations} operations failing'
            })
            recommendations.append({
                'priority': 'high',
                'action': 'fix_yahoo_finance_integration',
                'message': 'Fix Yahoo Finance integration issues',
                'details': 'Check API keys, rate limits, and data serialization'
            })
        elif failure_rate > 0.1:
            status = 'warning'
            issues.append({
                'severity': 'warning',
                'message': f'Some Yahoo Finance failures: {failure_rate:.1%}',
                'details': f'{failed_operations}/{total_operations} operations failing'
            })
        elif total_operations == 0:
            status = 'warning'
            issues.append({
                'severity': 'warning',
                'message': 'No Yahoo Finance operations detected',
                'details': 'Yahoo Finance integration may not be active'
            })
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'metrics': {
                'total_operations': total_operations,
                'failure_rate': failure_rate,
                'operation_types': dict(operation_stats)
            },
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _analyze_frontend_health(self) -> Dict[str, Any]:
        """Analyze frontend health"""
        frontend_log_file = self.log_dir / 'frontend.log'
        if not frontend_log_file.exists():
            return {'status': 'unknown', 'message': 'No frontend logs found'}
        
        recent_logs = self._parse_recent_logs(frontend_log_file, hours=1)
        
        total_errors = len(recent_logs)
        error_types = Counter([log.get('error_type', 'unknown') for log in recent_logs])
        
        issues = []
        recommendations = []
        
        if total_errors > 50:
            status = 'critical'
            issues.append({
                'severity': 'critical',
                'message': f'High frontend error count: {total_errors}',
                'details': f'Error types: {dict(error_types)}'
            })
        elif total_errors > 10:
            status = 'warning'
            issues.append({
                'severity': 'warning',
                'message': f'Elevated frontend errors: {total_errors}',
                'details': f'Error types: {dict(error_types)}'
            })
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'metrics': {
                'total_errors': total_errors,
                'error_types': dict(error_types)
            },
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _analyze_database_health(self) -> Dict[str, Any]:
        """Analyze database health"""
        db_log_file = self.log_dir / 'database.log'
        if not db_log_file.exists():
            return {'status': 'unknown', 'message': 'No database logs found'}
        
        recent_logs = self._parse_recent_logs(db_log_file, hours=1)
        
        total_operations = len(recent_logs)
        failed_operations = len([log for log in recent_logs if not log.get('success', True)])
        
        failure_rate = failed_operations / total_operations if total_operations > 0 else 0
        
        issues = []
        recommendations = []
        
        if failure_rate > 0.1:
            status = 'critical'
            issues.append({
                'severity': 'critical',
                'message': f'Database failure rate: {failure_rate:.1%}',
                'details': f'{failed_operations}/{total_operations} operations failing'
            })
        elif total_operations == 0:
            status = 'warning'
            issues.append({
                'severity': 'warning',
                'message': 'No database operations detected',
                'details': 'Database may not be connected'
            })
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'metrics': {
                'total_operations': total_operations,
                'failure_rate': failure_rate
            },
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _analyze_trace_health(self) -> Dict[str, Any]:
        """Analyze distributed trace health"""
        trace_stats = request_tracer.get_trace_statistics()
        
        issues = []
        recommendations = []
        
        # Check for stuck traces
        active_traces = trace_stats.get('active_traces', 0)
        if active_traces > 10:
            status = 'warning'
            issues.append({
                'severity': 'warning',
                'message': f'Many active traces: {active_traces}',
                'details': 'Some operations may be stuck'
            })
        else:
            status = 'healthy'
        
        # Analyze operation performance
        op_stats = trace_stats.get('operation_statistics', {})
        for operation, stats in op_stats.items():
            if stats.get('error_rate', 0) > 0.2:
                issues.append({
                    'severity': 'warning',
                    'message': f'High error rate for {operation}: {stats["error_rate"]:.1%}',
                    'details': f'Errors: {stats["errors"]}/{stats["count"]}'
                })
            
            if stats.get('avg_duration', 0) > 10.0:
                issues.append({
                    'severity': 'warning',
                    'message': f'Slow operation {operation}: {stats["avg_duration"]:.2f}s',
                    'details': 'Operation taking longer than expected'
                })
        
        return {
            'status': status,
            'metrics': trace_stats,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _parse_recent_logs(self, log_file: Path, hours: int = 1) -> List[Dict]:
        """Parse recent log entries"""
        if not log_file.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_logs = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        # Try to parse as JSON (structured logs)
                        if '|' in line:
                            # Parse structured log format
                            parts = line.strip().split('|')
                            if len(parts) >= 4:
                                timestamp_str = parts[0].strip()
                                message = parts[3].strip()
                                
                                # Try to parse timestamp
                                try:
                                    log_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    if log_time > cutoff_time:
                                        # Try to parse JSON message
                                        try:
                                            log_data = json.loads(message)
                                            recent_logs.append(log_data)
                                        except json.JSONDecodeError:
                                            # Plain text message
                                            recent_logs.append({'message': message, 'timestamp': timestamp_str})
                                except ValueError:
                                    continue
                    except Exception:
                        continue
        except Exception as e:
            diagnostic_logger.log_exception(e, f"Failed to parse log file {log_file}")
        
        return recent_logs
    
    def suggest_fixes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest specific fixes based on analysis"""
        fixes = []
        
        for recommendation in analysis.get('recommendations', []):
            action = recommendation.get('action')
            
            if action == 'investigate_api_errors':
                fixes.append({
                    'type': 'code_fix',
                    'priority': 'high',
                    'title': 'Fix API Error Handling',
                    'description': 'Improve error handling and logging in API endpoints',
                    'files_to_check': ['app.py', 'src/rex/historical_data/yahoo_finance.py'],
                    'specific_actions': [
                        'Add try-catch blocks around Yahoo Finance calls',
                        'Implement proper JSON serialization for pandas objects',
                        'Add input validation for API parameters'
                    ]
                })
            
            elif action == 'fix_yahoo_finance_integration':
                fixes.append({
                    'type': 'integration_fix',
                    'priority': 'high',
                    'title': 'Fix Yahoo Finance JSON Serialization',
                    'description': 'Resolve pandas Timestamp serialization issues',
                    'files_to_check': ['app.py', 'src/rex/historical_data/yahoo_finance.py'],
                    'specific_actions': [
                        'Implement custom JSON encoder for pandas types',
                        'Convert Timestamp objects to strings before serialization',
                        'Add data validation before API responses'
                    ]
                })
        
        return fixes


# Global system analyzer instance
system_analyzer = SystemAnalyzer()
