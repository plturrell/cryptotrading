"""
Code Health Dashboard
Real-time web dashboard for monitoring code health, issues, and automated fixes
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from dataclasses import asdict

from .intelligent_code_manager import IntelligentCodeManager, CodeHealthMetrics
from .automated_quality_monitor import AutomatedQualityMonitor
from .database_adapter import CodeManagementDatabaseAdapter

class CodeHealthDashboard:
    """Web dashboard for code health monitoring"""
    
    def __init__(self, project_path: str, port: int = 5001, database_adapter: CodeManagementDatabaseAdapter = None):
        self.project_path = Path(project_path)
        self.port = port
        self.app = Flask(__name__, template_folder=str(self.project_path / "templates"))
        self.database_adapter = database_adapter
        self.code_manager = IntelligentCodeManager(project_path, database_adapter)
        self.quality_monitor = AutomatedQualityMonitor(project_path, database_adapter)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('code_health_dashboard.html')
        
        @self.app.route('/api/health')
        def get_health():
            """Get current health metrics"""
            return jsonify(self.code_manager.get_health_dashboard())
        
        @self.app.route('/api/quality')
        def get_quality():
            """Get quality monitoring summary"""
            return jsonify(self.quality_monitor.get_quality_summary())
        
        @self.app.route('/api/issues')
        def get_issues():
            """Get current issues"""
            if self.database_adapter:
                # Get issues from database
                try:
                    active_issues = asyncio.run(self.database_adapter.get_issues(status_filter="pending"))
                except Exception as e:
                    print(f"Error getting issues from database: {e}")
                    active_issues = []
            else:
                # Fallback to in-memory storage
                active_issues = [
                    issue for issue in self.code_manager.issues_db 
                    if issue.fix_status.value != "completed"
                ]
            
            return jsonify({
                "total": len(active_issues),
                "critical": len([i for i in active_issues if i.type.value == "critical"]),
                "auto_fixable": len([i for i in active_issues if i.auto_fixable]),
                "issues": [
                    {
                        "id": issue.id,
                        "type": issue.type.value,
                        "severity": issue.severity,
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "description": issue.description,
                        "auto_fixable": issue.auto_fixable,
                        "fix_status": issue.fix_status.value,
                        "detected_at": issue.detected_at
                    }
                    for issue in active_issues[:50]  # Limit to 50 for performance
                ]
            })
        
        @self.app.route('/api/trends')
        def get_trends():
            """Get health trends over time"""
            history = self.code_manager.health_history[-30:]  # Last 30 entries
            
            trends = {
                "timestamps": [h.timestamp for h in history],
                "coverage": [h.coverage_percentage for h in history],
                "technical_debt": [h.technical_debt_score for h in history],
                "maintainability": [h.maintainability_index for h in history],
                "security": [h.security_score for h in history]
            }
            
            return jsonify(trends)
        
        @self.app.route('/api/fix-issue', methods=['POST'])
        def fix_issue():
            """Trigger manual fix for an issue"""
            data = request.get_json()
            issue_id = data.get('issue_id')
            
            # Find the issue
            if self.database_adapter:
                try:
                    issues = asyncio.run(self.database_adapter.get_issues())
                    issue = next((i for i in issues if i.id == issue_id), None)
                except Exception:
                    issue = None
            else:
                issue = next((i for i in self.code_manager.issues_db if i.id == issue_id), None)
                
            if not issue:
                return jsonify({"error": "Issue not found"}), 404
            
            if not issue.auto_fixable:
                return jsonify({"error": "Issue is not auto-fixable"}), 400
            
            # Trigger async fix
            asyncio.create_task(self._fix_issue_async(issue))
            
            return jsonify({"message": "Fix initiated", "issue_id": issue_id})
        
        @self.app.route('/api/run-check', methods=['POST'])
        def run_check():
            """Trigger manual quality check"""
            # Trigger async quality check
            asyncio.create_task(self._run_quality_check_async())
            
            return jsonify({"message": "Quality check initiated"})
        
        @self.app.route('/api/refactoring-recommendations')
        def get_refactoring_recommendations():
            """Get refactoring recommendations"""
            # This would be populated by the intelligent code manager
            recommendations = [
                {
                    "type": "complexity_reduction",
                    "file": "src/cryptotrading/core/trading_engine.py",
                    "function": "execute_trade",
                    "current_complexity": 12,
                    "suggested_refactoring": "Extract method for validation logic",
                    "priority": "high"
                },
                {
                    "type": "duplication_removal",
                    "files": ["src/cryptotrading/data/loader.py", "src/cryptotrading/data/processor.py"],
                    "lines": [45, 67],
                    "suggested_refactoring": "Extract common data validation function",
                    "priority": "medium"
                }
            ]
            
            return jsonify({"recommendations": recommendations})
    
    async def _fix_issue_async(self, issue):
        """Async wrapper for fixing issues"""
        try:
            await self.code_manager.auto_fix_issues([issue])
        except Exception as e:
            print(f"Error fixing issue {issue.id}: {e}")
    
    async def _run_quality_check_async(self):
        """Async wrapper for quality checks"""
        try:
            results = await self.quality_monitor.run_all_checks()
            issues = self.quality_monitor.process_results(results)
            await self.quality_monitor.auto_fix_issues(issues)
        except Exception as e:
            print(f"Error running quality check: {e}")
    
    def create_dashboard_template(self):
        """Create the HTML template for the dashboard"""
        template_dir = self.project_path / "templates"
        template_dir.mkdir(exist_ok=True)
        
        template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Health Dashboard - cryptotrading.com</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f1419;
            color: #e6e6e6;
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 {
            color: white;
            font-size: 2rem;
            font-weight: 600;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 1px solid #333;
        }
        .card h3 {
            color: #60a5fa;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #333;
        }
        .metric:last-child { border-bottom: none; }
        .metric-value {
            font-weight: 600;
            font-size: 1.1rem;
        }
        .status-good { color: #10b981; }
        .status-warning { color: #f59e0b; }
        .status-critical { color: #ef4444; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
            transition: width 0.3s ease;
        }
        .issue-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .issue-item {
            background: #2a2a2a;
            margin: 0.5rem 0;
            padding: 0.75rem;
            border-radius: 6px;
            border-left: 4px solid #ef4444;
        }
        .issue-item.warning { border-left-color: #f59e0b; }
        .issue-item.info { border-left-color: #3b82f6; }
        .btn {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.2s;
        }
        .btn:hover { background: #2563eb; }
        .btn-success { background: #10b981; }
        .btn-success:hover { background: #059669; }
        .chart-container {
            position: relative;
            height: 250px;
            margin-top: 1rem;
        }
        .loading {
            text-align: center;
            color: #888;
            padding: 2rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Code Health Dashboard - cryptotrading.com</h1>
    </div>
    
    <div class="dashboard">
        <!-- Health Overview -->
        <div class="card">
            <h3>📊 Health Overview</h3>
            <div id="health-overview" class="loading">Loading...</div>
        </div>
        
        <!-- Quality Metrics -->
        <div class="card">
            <h3>🔍 Quality Metrics</h3>
            <div id="quality-metrics" class="loading">Loading...</div>
        </div>
        
        <!-- Active Issues -->
        <div class="card">
            <h3>⚠️ Active Issues</h3>
            <div style="margin-bottom: 1rem;">
                <button class="btn" onclick="runQualityCheck()">🔄 Run Check</button>
            </div>
            <div id="active-issues" class="loading">Loading...</div>
        </div>
        
        <!-- Trends Chart -->
        <div class="card" style="grid-column: span 2;">
            <h3>📈 Health Trends</h3>
            <div class="chart-container">
                <canvas id="trendsChart"></canvas>
            </div>
        </div>
        
        <!-- Refactoring Recommendations -->
        <div class="card">
            <h3>🧠 Refactoring Recommendations</h3>
            <div id="refactoring-recommendations" class="loading">Loading...</div>
        </div>
    </div>

    <script>
        let trendsChart;
        
        async function fetchData(endpoint) {
            try {
                const response = await fetch(`/api/${endpoint}`);
                return await response.json();
            } catch (error) {
                console.error(`Error fetching ${endpoint}:`, error);
                return null;
            }
        }
        
        function updateHealthOverview(data) {
            const container = document.getElementById('health-overview');
            const status = data.status === 'healthy' ? 'status-good' : 'status-warning';
            
            container.innerHTML = `
                <div class="metric">
                    <span>Status</span>
                    <span class="metric-value ${status}">${data.status}</span>
                </div>
                <div class="metric">
                    <span>Coverage</span>
                    <span class="metric-value">${data.metrics?.coverage_percentage?.toFixed(1) || 0}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${data.metrics?.coverage_percentage || 0}%"></div>
                </div>
                <div class="metric">
                    <span>Active Issues</span>
                    <span class="metric-value ${data.active_issues > 0 ? 'status-warning' : 'status-good'}">${data.active_issues || 0}</span>
                </div>
                <div class="metric">
                    <span>Critical Issues</span>
                    <span class="metric-value ${data.critical_issues > 0 ? 'status-critical' : 'status-good'}">${data.critical_issues || 0}</span>
                </div>
                <div class="metric">
                    <span>Auto-fixable</span>
                    <span class="metric-value status-good">${data.auto_fixable_issues || 0}</span>
                </div>
            `;
        }
        
        function updateQualityMetrics(data) {
            const container = document.getElementById('quality-metrics');
            
            if (!data || data.status === 'no_data') {
                container.innerHTML = '<div class="loading">No quality data available</div>';
                return;
            }
            
            container.innerHTML = `
                <div class="metric">
                    <span>Total Issues</span>
                    <span class="metric-value">${data.total_issues || 0}</span>
                </div>
                <div class="metric">
                    <span>Fixed Issues</span>
                    <span class="metric-value status-good">${data.fixed_issues || 0}</span>
                </div>
                <div class="metric">
                    <span>Trend</span>
                    <span class="metric-value ${data.trend === 'improving' ? 'status-good' : data.trend === 'declining' ? 'status-critical' : ''}">${data.trend || 'stable'}</span>
                </div>
            `;
        }
        
        function updateActiveIssues(data) {
            const container = document.getElementById('active-issues');
            
            if (!data.issues || data.issues.length === 0) {
                container.innerHTML = '<div class="status-good">✅ No active issues</div>';
                return;
            }
            
            const issuesHtml = data.issues.slice(0, 10).map(issue => {
                const severityClass = issue.severity >= 8 ? 'critical' : issue.severity >= 5 ? 'warning' : 'info';
                const fixButton = issue.auto_fixable ? 
                    `<button class="btn btn-success" onclick="fixIssue('${issue.id}')">🔧 Fix</button>` : '';
                
                return `
                    <div class="issue-item ${severityClass}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div><strong>${issue.type}</strong> (Severity: ${issue.severity})</div>
                                <div style="font-size: 0.9rem; color: #ccc;">${issue.description}</div>
                                <div style="font-size: 0.8rem; color: #888;">${issue.file}:${issue.line || '?'}</div>
                            </div>
                            ${fixButton}
                        </div>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = `<div class="issue-list">${issuesHtml}</div>`;
        }
        
        function updateTrendsChart(data) {
            const ctx = document.getElementById('trendsChart').getContext('2d');
            
            if (trendsChart) {
                trendsChart.destroy();
            }
            
            trendsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.timestamps?.slice(-20) || [],
                    datasets: [
                        {
                            label: 'Coverage %',
                            data: data.coverage?.slice(-20) || [],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Technical Debt',
                            data: data.technical_debt?.slice(-20) || [],
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Maintainability',
                            data: data.maintainability?.slice(-20) || [],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#e6e6e6' }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#888' },
                            grid: { color: '#333' }
                        },
                        y: {
                            ticks: { color: '#888' },
                            grid: { color: '#333' }
                        }
                    }
                }
            });
        }
        
        function updateRefactoringRecommendations(data) {
            const container = document.getElementById('refactoring-recommendations');
            
            if (!data.recommendations || data.recommendations.length === 0) {
                container.innerHTML = '<div class="status-good">✅ No refactoring needed</div>';
                return;
            }
            
            const recsHtml = data.recommendations.map(rec => `
                <div class="issue-item info">
                    <div><strong>${rec.type.replace('_', ' ')}</strong></div>
                    <div style="font-size: 0.9rem; color: #ccc;">${rec.suggested_refactoring}</div>
                    <div style="font-size: 0.8rem; color: #888;">Priority: ${rec.priority}</div>
                </div>
            `).join('');
            
            container.innerHTML = `<div class="issue-list">${recsHtml}</div>`;
        }
        
        async function fixIssue(issueId) {
            try {
                const response = await fetch('/api/fix-issue', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ issue_id: issueId })
                });
                
                if (response.ok) {
                    alert('Fix initiated! Refreshing data...');
                    setTimeout(loadData, 2000);
                } else {
                    alert('Failed to initiate fix');
                }
            } catch (error) {
                alert('Error initiating fix');
            }
        }
        
        async function runQualityCheck() {
            try {
                const response = await fetch('/api/run-check', { method: 'POST' });
                if (response.ok) {
                    alert('Quality check initiated! Results will appear shortly...');
                    setTimeout(loadData, 5000);
                }
            } catch (error) {
                alert('Error running quality check');
            }
        }
        
        async function loadData() {
            const [health, quality, issues, trends, recommendations] = await Promise.all([
                fetchData('health'),
                fetchData('quality'),
                fetchData('issues'),
                fetchData('trends'),
                fetchData('refactoring-recommendations')
            ]);
            
            if (health) updateHealthOverview(health);
            if (quality) updateQualityMetrics(quality);
            if (issues) updateActiveIssues(issues);
            if (trends) updateTrendsChart(trends);
            if (recommendations) updateRefactoringRecommendations(recommendations);
        }
        
        // Load data on page load and refresh every 30 seconds
        loadData();
        setInterval(loadData, 30000);
    </script>
</body>
</html>'''
        
        template_file = template_dir / "code_health_dashboard.html"
        with open(template_file, "w") as f:
            f.write(template_content)
        
        print(f"✅ Dashboard template created at {template_file}")
    
    def run(self, debug: bool = False):
        """Run the dashboard server"""
        self.create_dashboard_template()
        print(f"🚀 Starting Code Health Dashboard on http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

def main():
    """Main entry point for the dashboard"""
    import sys
    
    project_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/apple/projects/cryptotrading"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5001
    
    dashboard = CodeHealthDashboard(project_path, port)
    dashboard.run(debug=True)

if __name__ == "__main__":
    main()
