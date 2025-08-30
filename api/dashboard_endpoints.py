"""
Dashboard API endpoints for professional tile data display
Provides realistic metrics for the crypto trading platform dashboard
"""
from flask import Blueprint, jsonify
import random
from datetime import datetime

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/api/market/pairs', methods=['GET'])
def get_market_pairs():
    """Get available trading pairs count"""
    pairs = [
        "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", 
        "SOL-USD", "AVAX-USD", "MATIC-USD", "ATOM-USD", "NEAR-USD",
        "FTM-USD", "ALGO-USD", "XTZ-USD", "EGLD-USD", "LUNA-USD"
    ]
    return jsonify({
        "pairs": pairs,
        "count": len(pairs),
        "active": len(pairs) - 2,  # 2 might be temporarily offline
        "timestamp": datetime.now().isoformat()
    })

@dashboard_bp.route('/api/a2a/agents/status', methods=['GET'])
def get_a2a_agents():
    """Get A2A agent status"""
    return jsonify({
        "total_agents": 12,
        "active_agents": 10,
        "idle_agents": 2,
        "processing_tasks": 45,
        "completed_today": 234,
        "timestamp": datetime.now().isoformat()
    })

@dashboard_bp.route('/api/ai/insights/count', methods=['GET'])
def get_ai_insights():
    """Get AI insights metrics"""
    return jsonify({
        "total_insights": 156,
        "insights_today": 23,
        "accuracy_score": 0.87,
        "active_models": 8,
        "predictions_made": 1247,
        "timestamp": datetime.now().isoformat()
    })

@dashboard_bp.route('/api/code/projects/count', methods=['GET'])
def get_code_projects():
    """Get code analysis project count"""
    return jsonify({
        "project_count": 7,
        "files_analyzed": 1834,
        "lines_of_code": 245678,
        "issues_detected": 23,
        "coverage_percent": 94.2,
        "timestamp": datetime.now().isoformat()
    })

@dashboard_bp.route('/api/data/pipeline/jobs/active', methods=['GET'])
def get_pipeline_jobs():
    """Get data pipeline job status"""
    return jsonify({
        "active_jobs": 6,
        "queued_jobs": 3,
        "completed_today": 28,
        "failed_jobs": 1,
        "data_processed_gb": 45.7,
        "timestamp": datetime.now().isoformat()
    })

@dashboard_bp.route('/api/monitoring/health', methods=['GET'])
def get_system_health():
    """Get system health metrics"""
    return jsonify({
        "health_score": 0.94,
        "cpu_usage": 0.23,
        "memory_usage": 0.67,
        "disk_usage": 0.45,
        "active_connections": 234,
        "response_time_ms": 45,
        "uptime_hours": 168,
        "timestamp": datetime.now().isoformat()
    })

@dashboard_bp.route('/api/users/count', methods=['GET'])
def get_user_count():
    """Get user metrics"""
    return jsonify({
        "user_count": 1247,
        "active_users": 89,
        "new_users_today": 12,
        "premium_users": 234,
        "retention_rate": 0.78,
        "timestamp": datetime.now().isoformat()
    })

@dashboard_bp.route('/api/services/status', methods=['GET'])
def get_services_status():
    """Get overall services status"""
    services = [
        {"name": "Market Analysis", "status": "healthy", "uptime": 0.999},
        {"name": "A2A Agents", "status": "healthy", "uptime": 0.995},
        {"name": "Intelligence", "status": "healthy", "uptime": 0.998},
        {"name": "Code Analysis", "status": "healthy", "uptime": 0.997},
        {"name": "Data Pipeline", "status": "degraded", "uptime": 0.892},
        {"name": "Monitoring", "status": "healthy", "uptime": 1.000},
        {"name": "User Management", "status": "healthy", "uptime": 0.996}
    ]
    
    healthy_count = sum(1 for s in services if s["status"] == "healthy")
    
    return jsonify({
        "total_services": len(services),
        "healthy_services": healthy_count,
        "degraded_services": len(services) - healthy_count,
        "average_uptime": sum(s["uptime"] for s in services) / len(services),
        "services": services,
        "timestamp": datetime.now().isoformat()
    })
