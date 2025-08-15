"""
Observability Dashboard API
Provides endpoints for monitoring traces, errors, and metrics
"""

from flask import Flask, jsonify, request
from typing import Dict, Any
import json

from .tracer import get_tracer
from .logger import get_logger  
from .error_tracker import get_error_tracker
from .metrics import get_metrics
from .integration import get_observability_health

logger = get_logger(__name__)
tracer = get_tracer()
error_tracker = get_error_tracker()
metrics = get_metrics()

def create_observability_blueprint():
    """Create Flask blueprint for observability endpoints"""
    from flask import Blueprint
    
    bp = Blueprint('observability', __name__, url_prefix='/observability')
    
    @bp.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        try:
            health_data = get_observability_health()
            return jsonify(health_data), 200
        except Exception as e:
            logger.error("Health check failed", error=e)
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/metrics', methods=['GET'])
    def get_metrics_summary():
        """Get metrics summary"""
        try:
            hours = int(request.args.get('hours', 1))
            format_type = request.args.get('format', 'json')
            
            if format_type == 'prometheus':
                prometheus_data = metrics.export_metrics('prometheus')
                return prometheus_data, 200, {'Content-Type': 'text/plain'}
            else:
                metrics_data = metrics.get_all_metrics_summary(hours)
                return jsonify(metrics_data), 200
                
        except Exception as e:
            logger.error("Failed to get metrics", error=e)
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/metrics/<metric_name>', methods=['GET'])
    def get_metric_details(metric_name: str):
        """Get details for specific metric"""
        try:
            hours = int(request.args.get('hours', 1))
            metric_data = metrics.get_metric_summary(metric_name, hours)
            
            if metric_data:
                return jsonify(metric_data), 200
            else:
                return jsonify({'error': 'Metric not found'}), 404
                
        except Exception as e:
            logger.error(f"Failed to get metric {metric_name}", error=e)
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/errors/summary', methods=['GET'])
    def get_errors_summary():
        """Get error summary"""
        try:
            hours = int(request.args.get('hours', 24))
            error_summary = error_tracker.get_error_summary(hours)
            return jsonify(error_summary), 200
            
        except Exception as e:
            logger.error("Failed to get error summary", error=e)
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/errors/<error_id>', methods=['GET'])
    def get_error_details(error_id: str):
        """Get details for specific error"""
        try:
            error_details = error_tracker.get_error_details(error_id)
            
            if error_details:
                return jsonify(error_details), 200
            else:
                return jsonify({'error': 'Error not found'}), 404
                
        except Exception as e:
            logger.error(f"Failed to get error {error_id}", error=e)
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/errors/fingerprint/<fingerprint>', methods=['GET'])
    def get_errors_by_fingerprint(fingerprint: str):
        """Get errors with the same fingerprint"""
        try:
            limit = int(request.args.get('limit', 10))
            errors = error_tracker.get_errors_by_fingerprint(fingerprint, limit)
            return jsonify({'fingerprint': fingerprint, 'errors': errors}), 200
            
        except Exception as e:
            logger.error(f"Failed to get errors by fingerprint {fingerprint}", error=e)
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/traces/<trace_id>', methods=['GET'])
    def get_trace_details(trace_id: str):
        """Get trace details (placeholder - would need trace storage)"""
        try:
            # This would need to be implemented with actual trace storage
            # For now, return a placeholder response
            return jsonify({
                'trace_id': trace_id,
                'message': 'Trace storage not yet implemented',
                'suggestion': 'Configure OTEL_EXPORTER_OTLP_ENDPOINT for trace collection'
            }), 200
            
        except Exception as e:
            logger.error(f"Failed to get trace {trace_id}", error=e)
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/dashboard', methods=['GET'])
    def get_dashboard_data():
        """Get comprehensive dashboard data"""
        try:
            hours = int(request.args.get('hours', 24))
            
            dashboard_data = {
                'timestamp': get_observability_health()['timestamp'],
                'time_range': {'hours': hours},
                'health': get_observability_health(),
                'metrics_summary': metrics.get_all_metrics_summary(hours),
                'errors_summary': error_tracker.get_error_summary(hours)
            }
            
            return jsonify(dashboard_data), 200
            
        except Exception as e:
            logger.error("Failed to get dashboard data", error=e)
            return jsonify({'error': str(e)}), 500
    
    return bp

def register_observability_routes(app: Flask):
    """Register observability routes with Flask app"""
    bp = create_observability_blueprint()
    app.register_blueprint(bp)
    logger.info("Observability routes registered")

# HTML Dashboard (basic)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Rex Trading - Observability Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .error { color: red; }
        .success { color: green; }
        .warning { color: orange; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .refresh-btn { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Rex Trading - Observability Dashboard</h1>
    
    <button class="refresh-btn" onclick="location.reload()">Refresh</button>
    
    <div id="health-status" class="metric-card">
        <h2>System Health</h2>
        <div id="health-content">Loading...</div>
    </div>
    
    <div id="metrics-summary" class="metric-card">
        <h2>Metrics Summary (Last 24h)</h2>
        <div id="metrics-content">Loading...</div>
    </div>
    
    <div id="errors-summary" class="metric-card">
        <h2>Error Summary (Last 24h)</h2>
        <div id="errors-content">Loading...</div>
    </div>

    <script>
        async function loadDashboard() {
            try {
                const response = await fetch('/observability/dashboard?hours=24');
                const data = await response.json();
                
                // Update health status
                const healthHtml = Object.entries(data.health).map(([component, status]) => {
                    if (component === 'timestamp') return '';
                    const statusClass = status.status === 'healthy' ? 'success' : 'error';
                    return `<div><strong>${component}:</strong> <span class="${statusClass}">${status.status}</span></div>`;
                }).join('');
                document.getElementById('health-content').innerHTML = healthHtml;
                
                // Update metrics
                const metricsHtml = `
                    <p><strong>Total Metrics:</strong> ${data.metrics_summary.total_metrics}</p>
                    <table>
                        <tr><th>Metric</th><th>Type</th><th>Count</th><th>Latest Value</th></tr>
                        ${Object.entries(data.metrics_summary.metrics).slice(0, 10).map(([name, metric]) => `
                            <tr>
                                <td>${name}</td>
                                <td>${metric.type}</td>
                                <td>${metric.count}</td>
                                <td>${metric.latest || metric.total || 'N/A'}</td>
                            </tr>
                        `).join('')}
                    </table>
                `;
                document.getElementById('metrics-content').innerHTML = metricsHtml;
                
                // Update errors
                const errorsHtml = `
                    <p><strong>Total Errors:</strong> <span class="error">${data.errors_summary.total_errors}</span></p>
                    <h4>By Severity:</h4>
                    ${Object.entries(data.errors_summary.by_severity).map(([severity, count]) => 
                        `<div><strong>${severity}:</strong> ${count}</div>`
                    ).join('')}
                    <h4>By Category:</h4>
                    ${Object.entries(data.errors_summary.by_category).map(([category, count]) => 
                        `<div><strong>${category}:</strong> ${count}</div>`
                    ).join('')}
                `;
                document.getElementById('errors-content').innerHTML = errorsHtml;
                
            } catch (error) {
                console.error('Failed to load dashboard:', error);
                document.getElementById('health-content').innerHTML = '<div class="error">Failed to load dashboard data</div>';
            }
        }
        
        // Load dashboard on page load
        loadDashboard();
        
        // Auto-refresh every 30 seconds
        setInterval(loadDashboard, 30000);
    </script>
</body>
</html>
"""

def create_dashboard_route(app: Flask):
    """Create HTML dashboard route"""
    @app.route('/observability/dashboard.html')
    def dashboard_html():
        return DASHBOARD_HTML