"""
Diagnostic Dashboard API for rex.com
Provides real-time system health, logs, traces, and automated fix recommendations
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import asyncio
from .logger import diagnostic_logger
from .tracer import request_tracer
from .analyzer import system_analyzer

# Create blueprint for diagnostic endpoints
diagnostic_bp = Blueprint('diagnostics', __name__, url_prefix='/api/diagnostics')


@diagnostic_bp.route('/health', methods=['GET'])
def get_system_health():
    """Get comprehensive system health analysis"""
    try:
        analysis = system_analyzer.analyze_system_health()
        return jsonify(analysis)
    except Exception as e:
        diagnostic_logger.log_exception(e, "Failed to get system health")
        return jsonify({'error': str(e)}), 500


@diagnostic_bp.route('/traces', methods=['GET'])
def get_traces():
    """Get recent traces"""
    try:
        limit = int(request.args.get('limit', 50))
        
        # Get recent traces
        recent_traces = request_tracer.get_recent_traces(limit)
        active_traces = request_tracer.get_active_traces()
        trace_stats = request_tracer.get_trace_statistics()
        
        return jsonify({
            'recent_traces': recent_traces,
            'active_traces': active_traces,
            'statistics': trace_stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        diagnostic_logger.log_exception(e, "Failed to get traces")
        return jsonify({'error': str(e)}), 500


@diagnostic_bp.route('/logs/<component>', methods=['GET'])
def get_component_logs(component):
    """Get logs for a specific component"""
    try:
        hours = int(request.args.get('hours', 1))
        
        # This would parse actual log files
        # For now, return summary from error tracking
        error_summary = diagnostic_logger.get_error_summary()
        
        component_errors = {
            k: v for k, v in error_summary.items() 
            if k.startswith(f"{component}:")
        }
        
        return jsonify({
            'component': component,
            'hours': hours,
            'error_summary': component_errors,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        diagnostic_logger.log_exception(e, f"Failed to get logs for {component}")
        return jsonify({'error': str(e)}), 500


@diagnostic_bp.route('/fixes/suggest', methods=['POST'])
def suggest_fixes():
    """Suggest automated fixes based on current system state"""
    try:
        # Get current analysis
        analysis = system_analyzer.analyze_system_health()
        
        # Get fix suggestions
        fixes = system_analyzer.suggest_fixes(analysis)
        
        return jsonify({
            'analysis_summary': {
                'overall_status': analysis['overall_status'],
                'issue_count': len(analysis['issues']),
                'critical_issues': len([i for i in analysis['issues'] if i.get('severity') == 'critical'])
            },
            'suggested_fixes': fixes,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        diagnostic_logger.log_exception(e, "Failed to suggest fixes")
        return jsonify({'error': str(e)}), 500


@diagnostic_bp.route('/agent/status', methods=['GET'])
def get_agent_status():
    """Get diagnostic agent status"""
    try:
        # This would integrate with the actual diagnostic agent
        # For now, return mock status
        return jsonify({
            'agent_name': 'DiagnosticAgent',
            'status': 'active',
            'monitoring_enabled': True,
            'auto_fix_enabled': True,
            'last_analysis': datetime.now().isoformat(),
            'total_analyses': 0,
            'total_fixes': 0,
            'success_rate': 0.0
        })
    except Exception as e:
        diagnostic_logger.log_exception(e, "Failed to get agent status")
        return jsonify({'error': str(e)}), 500


@diagnostic_bp.route('/agent/trigger-analysis', methods=['POST'])
def trigger_analysis():
    """Manually trigger system analysis"""
    try:
        # Perform immediate analysis
        analysis = system_analyzer.analyze_system_health()
        
        # Log the manual trigger
        diagnostic_logger.app_logger.info("Manual system analysis triggered")
        
        return jsonify({
            'message': 'Analysis triggered successfully',
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        diagnostic_logger.log_exception(e, "Failed to trigger analysis")
        return jsonify({'error': str(e)}), 500


@diagnostic_bp.route('/performance/metrics', methods=['GET'])
def get_performance_metrics():
    """Get performance metrics"""
    try:
        hours = int(request.args.get('hours', 1))
        
        # This would parse performance logs
        # For now, return trace statistics as performance metrics
        trace_stats = request_tracer.get_trace_statistics()
        
        return jsonify({
            'time_range_hours': hours,
            'trace_statistics': trace_stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        diagnostic_logger.log_exception(e, "Failed to get performance metrics")
        return jsonify({'error': str(e)}), 500


@diagnostic_bp.route('/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        # Get all diagnostic information
        health_analysis = system_analyzer.analyze_system_health()
        trace_stats = request_tracer.get_trace_statistics()
        error_summary = diagnostic_logger.get_error_summary()
        
        # Prepare dashboard summary
        dashboard_data = {
            'system_health': {
                'overall_status': health_analysis['overall_status'],
                'components': {
                    name: {
                        'status': comp.get('status', 'unknown'),
                        'issues': len(comp.get('issues', [])),
                        'metrics': comp.get('metrics', {})
                    }
                    for name, comp in health_analysis['components'].items()
                }
            },
            'performance': {
                'active_traces': trace_stats.get('active_traces', 0),
                'completed_traces': trace_stats.get('completed_traces', 0),
                'operation_stats': trace_stats.get('operation_statistics', {})
            },
            'errors': {
                'total_errors': len(error_summary),
                'top_errors': dict(list(sorted(error_summary.items(), key=lambda x: x[1], reverse=True))[:10])
            },
            'recommendations': health_analysis.get('recommendations', []),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(dashboard_data)
    except Exception as e:
        diagnostic_logger.log_exception(e, "Failed to get dashboard data")
        return jsonify({'error': str(e)}), 500
