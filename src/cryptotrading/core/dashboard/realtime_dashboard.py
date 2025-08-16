"""
Real-time Code Analysis Dashboard
Web interface for visualizing real-time code insights, file changes, and AI analysis
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import aiohttp
from aiohttp import web, WSMsgType
# import aiofiles  # Not needed for this implementation
import weakref

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Dashboard metrics for real-time display"""
    total_files_watched: int = 0
    changes_detected: int = 0
    analyses_completed: int = 0
    ai_insights_generated: int = 0
    average_analysis_time: float = 0.0
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files_watched": self.total_files_watched,
            "changes_detected": self.changes_detected,
            "analyses_completed": self.analyses_completed,
            "ai_insights_generated": self.ai_insights_generated,
            "average_analysis_time": round(self.average_analysis_time, 2),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "last_updated": self.last_updated.isoformat()
        }


class RealtimeDashboard:
    """Real-time dashboard for code analysis insights"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8082"):
        self.mcp_server_url = mcp_server_url
        self.app = None
        self.server = None
        self.websockets: Set[web.WebSocketResponse] = set()
        self.metrics = DashboardMetrics()
        
        # Data cache for dashboard
        self.recent_changes: List[Dict[str, Any]] = []
        self.recent_analyses: List[Dict[str, Any]] = []
        self.system_status: Dict[str, Any] = {}
        
        # Update intervals
        self.metrics_update_interval = 5.0  # seconds
        self.data_update_interval = 2.0     # seconds
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
    async def create_app(self) -> web.Application:
        """Create the web application"""
        app = web.Application()
        
        # Static file serving
        dashboard_dir = Path(__file__).parent
        static_dir = dashboard_dir / "static"
        
        # Ensure static directory exists
        static_dir.mkdir(exist_ok=True)
        
        # Create static files if they don't exist
        await self._create_static_files(static_dir)
        
        # Routes
        app.router.add_get('/', self.serve_dashboard)
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_get('/api/metrics', self.get_metrics)
        app.router.add_get('/api/changes', self.get_recent_changes)
        app.router.add_get('/api/analyses', self.get_recent_analyses)
        app.router.add_get('/api/status', self.get_system_status)
        app.router.add_static('/static', static_dir)
        
        # CORS middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        app.middlewares.append(cors_middleware)
        
        self.app = app
        return app
    
    async def serve_dashboard(self, request) -> web.Response:
        """Serve the main dashboard HTML"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Code Analysis Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; color: white; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { 
            background: white; border-radius: 12px; padding: 20px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1); transition: transform 0.2s;
        }
        .card:hover { transform: translateY(-2px); }
        .card h3 { color: #667eea; margin-bottom: 15px; font-size: 1.3em; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-label { color: #666; }
        .metric-value { 
            font-weight: bold; color: #333; 
            padding: 4px 8px; background: #f0f3ff; border-radius: 4px;
        }
        .status-indicator { 
            display: inline-block; width: 12px; height: 12px; 
            border-radius: 50%; margin-right: 8px;
        }
        .status-active { background: #4ade80; }
        .status-inactive { background: #f87171; }
        .log-entry { 
            padding: 8px; margin: 4px 0; background: #f8fafc; 
            border-left: 3px solid #667eea; border-radius: 4px; font-size: 0.9em;
        }
        .log-timestamp { color: #64748b; font-size: 0.8em; }
        .ai-insight { 
            background: linear-gradient(135deg, #667eea20, #764ba220); 
            border: 1px solid #667eea40; 
        }
        .connection-status { 
            position: fixed; top: 20px; right: 20px; 
            padding: 10px 15px; border-radius: 20px; 
            color: white; font-weight: bold; z-index: 1000;
        }
        .connected { background: #4ade80; }
        .disconnected { background: #f87171; }
        .loading { 
            display: inline-block; width: 16px; height: 16px; 
            border: 2px solid #f3f3f3; border-top: 2px solid #667eea; 
            border-radius: 50%; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        <span class="loading"></span> Connecting...
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🔍 Real-time Code Analysis Dashboard</h1>
            <p>Monitor code changes, analysis results, and AI insights in real-time</p>
        </div>
        
        <div class="grid">
            <!-- System Metrics -->
            <div class="card">
                <h3>📊 System Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Files Watched</span>
                    <span class="metric-value" id="filesWatched">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Changes Detected</span>
                    <span class="metric-value" id="changesDetected">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Analyses Completed</span>
                    <span class="metric-value" id="analysesCompleted">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">AI Insights Generated</span>
                    <span class="metric-value" id="aiInsights">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Analysis Time</span>
                    <span class="metric-value" id="avgAnalysisTime">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">System Uptime</span>
                    <span class="metric-value" id="systemUptime">-</span>
                </div>
            </div>
            
            <!-- Real-time Status -->
            <div class="card">
                <h3>🔄 Real-time Status</h3>
                <div class="metric">
                    <span class="metric-label">
                        <span class="status-indicator" id="watchingStatus"></span>
                        File Watching
                    </span>
                    <span class="metric-value" id="watchingText">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">
                        <span class="status-indicator" id="mcpStatus"></span>
                        MCP Server
                    </span>
                    <span class="metric-value" id="mcpText">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">
                        <span class="status-indicator" id="aiStatus"></span>
                        AI Analysis
                    </span>
                    <span class="metric-value" id="aiText">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Queue Size</span>
                    <span class="metric-value" id="queueSize">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Tasks</span>
                    <span class="metric-value" id="activeTasks">-</span>
                </div>
            </div>
            
            <!-- Recent Changes -->
            <div class="card">
                <h3>📝 Recent File Changes</h3>
                <div id="recentChanges" style="max-height: 300px; overflow-y: auto;">
                    <div class="log-entry">No recent changes...</div>
                </div>
            </div>
            
            <!-- Recent Analyses -->
            <div class="card">
                <h3>🔬 Recent Analyses</h3>
                <div id="recentAnalyses" style="max-height: 300px; overflow-y: auto;">
                    <div class="log-entry">No recent analyses...</div>
                </div>
            </div>
            
            <!-- AI Insights -->
            <div class="card">
                <h3>🤖 AI Insights</h3>
                <div id="aiInsightsList" style="max-height: 300px; overflow-y: auto;">
                    <div class="log-entry">No AI insights yet...</div>
                </div>
            </div>
            
            <!-- System Activity Log -->
            <div class="card">
                <h3>📋 Activity Log</h3>
                <div id="activityLog" style="max-height: 300px; overflow-y: auto;">
                    <div class="log-entry">System starting...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        class RealtimeDashboard {
            constructor() {
                this.ws = null;
                this.reconnectInterval = 5000;
                this.maxReconnectAttempts = 10;
                this.reconnectAttempts = 0;
                this.activityLogEntries = [];
                this.maxLogEntries = 50;
                
                this.init();
            }
            
            init() {
                this.connectWebSocket();
                this.startDataPolling();
                this.addActivity('Dashboard initialized');
            }
            
            connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                try {
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {
                        this.updateConnectionStatus(true);
                        this.reconnectAttempts = 0;
                        this.addActivity('WebSocket connected');
                    };
                    
                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            this.handleRealtimeUpdate(data);
                        } catch (error) {
                            console.error('WebSocket message parse error:', error);
                        }
                    };
                    
                    this.ws.onclose = () => {
                        this.updateConnectionStatus(false);
                        this.addActivity('WebSocket disconnected');
                        this.scheduleReconnect();
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.addActivity('WebSocket error occurred');
                    };
                    
                } catch (error) {
                    console.error('WebSocket connection failed:', error);
                    this.updateConnectionStatus(false);
                    this.scheduleReconnect();
                }
            }
            
            scheduleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    setTimeout(() => {
                        this.addActivity(`Reconnection attempt ${this.reconnectAttempts}`);
                        this.connectWebSocket();
                    }, this.reconnectInterval);
                }
            }
            
            updateConnectionStatus(connected) {
                const status = document.getElementById('connectionStatus');
                if (connected) {
                    status.className = 'connection-status connected';
                    status.innerHTML = '🟢 Connected';
                } else {
                    status.className = 'connection-status disconnected';
                    status.innerHTML = '🔴 Disconnected';
                }
            }
            
            handleRealtimeUpdate(data) {
                switch (data.type) {
                    case 'metrics':
                        this.updateMetrics(data.payload);
                        break;
                    case 'file_change':
                        this.addFileChange(data.payload);
                        break;
                    case 'analysis_result':
                        this.addAnalysisResult(data.payload);
                        break;
                    case 'ai_insight':
                        this.addAIInsight(data.payload);
                        break;
                    case 'system_status':
                        this.updateSystemStatus(data.payload);
                        break;
                }
            }
            
            async startDataPolling() {
                setInterval(async () => {
                    try {
                        await Promise.all([
                            this.fetchMetrics(),
                            this.fetchRecentChanges(),
                            this.fetchRecentAnalyses(),
                            this.fetchSystemStatus()
                        ]);
                    } catch (error) {
                        console.error('Data polling error:', error);
                    }
                }, 5000);
            }
            
            async fetchMetrics() {
                try {
                    const response = await fetch('/api/metrics');
                    const data = await response.json();
                    this.updateMetrics(data);
                } catch (error) {
                    console.error('Failed to fetch metrics:', error);
                }
            }
            
            updateMetrics(metrics) {
                document.getElementById('filesWatched').textContent = metrics.total_files_watched || '-';
                document.getElementById('changesDetected').textContent = metrics.changes_detected || '-';
                document.getElementById('analysesCompleted').textContent = metrics.analyses_completed || '-';
                document.getElementById('aiInsights').textContent = metrics.ai_insights_generated || '-';
                document.getElementById('avgAnalysisTime').textContent = 
                    metrics.average_analysis_time ? `${metrics.average_analysis_time}s` : '-';
                document.getElementById('systemUptime').textContent = 
                    metrics.uptime_seconds ? this.formatUptime(metrics.uptime_seconds) : '-';
            }
            
            async fetchRecentChanges() {
                try {
                    const response = await fetch('/api/changes');
                    const data = await response.json();
                    this.updateRecentChanges(data.recent_changes || []);
                } catch (error) {
                    console.error('Failed to fetch recent changes:', error);
                }
            }
            
            updateRecentChanges(changes) {
                const container = document.getElementById('recentChanges');
                if (changes.length === 0) {
                    container.innerHTML = '<div class="log-entry">No recent changes...</div>';
                    return;
                }
                
                const html = changes.slice(-10).reverse().map(change => `
                    <div class="log-entry">
                        <div>${this.getFileName(change.file_path)} 
                            <span style="color: #667eea;">(${change.change_type})</span>
                        </div>
                        <div class="log-timestamp">${this.formatTimestamp(change.timestamp)}</div>
                    </div>
                `).join('');
                
                container.innerHTML = html;
            }
            
            async fetchRecentAnalyses() {
                try {
                    const response = await fetch('/api/analyses');
                    const data = await response.json();
                    this.updateRecentAnalyses(data.recent_analyses || []);
                } catch (error) {
                    console.error('Failed to fetch recent analyses:', error);
                }
            }
            
            updateRecentAnalyses(analyses) {
                const container = document.getElementById('recentAnalyses');
                if (analyses.length === 0) {
                    container.innerHTML = '<div class="log-entry">No recent analyses...</div>';
                    return;
                }
                
                const html = analyses.slice(-10).reverse().map(analysis => `
                    <div class="log-entry ${analysis.ai_insights ? 'ai-insight' : ''}">
                        <div>${this.getFileName(analysis.file_path)} 
                            <span style="color: #667eea;">
                                (${analysis.symbols_found ? analysis.symbols_found.length : 0} symbols)
                                ${analysis.ai_insights ? '🤖' : ''}
                            </span>
                        </div>
                        <div class="log-timestamp">
                            ${this.formatTimestamp(analysis.timestamp)} • 
                            ${analysis.processing_time ? analysis.processing_time.toFixed(2) : '0'}s
                        </div>
                    </div>
                `).join('');
                
                container.innerHTML = html;
            }
            
            async fetchSystemStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    this.updateSystemStatus(data);
                } catch (error) {
                    console.error('Failed to fetch system status:', error);
                }
            }
            
            updateSystemStatus(status) {
                // File watching status
                const watchingActive = status.realtime_status === 'active';
                document.getElementById('watchingStatus').className = 
                    `status-indicator ${watchingActive ? 'status-active' : 'status-inactive'}`;
                document.getElementById('watchingText').textContent = 
                    watchingActive ? 'Active' : 'Inactive';
                
                // MCP server status
                document.getElementById('mcpStatus').className = 'status-indicator status-active';
                document.getElementById('mcpText').textContent = 'Running';
                
                // AI status
                const aiActive = status.watcher_statistics?.ai_insights_generated > 0;
                document.getElementById('aiStatus').className = 
                    `status-indicator ${aiActive ? 'status-active' : 'status-inactive'}`;
                document.getElementById('aiText').textContent = aiActive ? 'Active' : 'Standby';
                
                // Queue and tasks
                document.getElementById('queueSize').textContent = 
                    status.watcher_statistics?.queue_stats?.queued || '-';
                document.getElementById('activeTasks').textContent = 
                    status.watcher_statistics?.active_analysis_tasks || '-';
            }
            
            addActivity(message) {
                const timestamp = new Date().toLocaleTimeString();
                this.activityLogEntries.unshift({ timestamp, message });
                
                if (this.activityLogEntries.length > this.maxLogEntries) {
                    this.activityLogEntries = this.activityLogEntries.slice(0, this.maxLogEntries);
                }
                
                const container = document.getElementById('activityLog');
                const html = this.activityLogEntries.map(entry => `
                    <div class="log-entry">
                        <div>${entry.message}</div>
                        <div class="log-timestamp">${entry.timestamp}</div>
                    </div>
                `).join('');
                
                container.innerHTML = html;
            }
            
            getFileName(filePath) {
                return filePath ? filePath.split('/').pop() : 'unknown';
            }
            
            formatTimestamp(timestamp) {
                try {
                    return new Date(timestamp).toLocaleTimeString();
                } catch {
                    return timestamp || 'unknown';
                }
            }
            
            formatUptime(seconds) {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                const secs = Math.floor(seconds % 60);
                
                if (hours > 0) {
                    return `${hours}h ${minutes}m ${secs}s`;
                } else if (minutes > 0) {
                    return `${minutes}m ${secs}s`;
                } else {
                    return `${secs}s`;
                }
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new RealtimeDashboard();
        });
    </script>
</body>
</html>
        """
        
        return web.Response(text=html_content, content_type='text/html')
    
    async def websocket_handler(self, request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Add to active connections
        self.websockets.add(ws)
        logger.info("WebSocket client connected")
        
        try:
            # Send initial data
            await self._send_to_websocket(ws, 'metrics', self.metrics.to_dict())
            await self._send_to_websocket(ws, 'system_status', self.system_status)
            
            # Handle incoming messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        # Handle client requests if needed
                    except json.JSONDecodeError:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            # Remove from active connections
            self.websockets.discard(ws)
            logger.info("WebSocket client disconnected")
        
        return ws
    
    async def _send_to_websocket(self, ws: web.WebSocketResponse, msg_type: str, payload: Any):
        """Send data to a WebSocket connection"""
        try:
            message = {
                "type": msg_type,
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            }
            await ws.send_str(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
    
    async def broadcast_to_websockets(self, msg_type: str, payload: Any):
        """Broadcast data to all WebSocket connections"""
        if not self.websockets:
            return
        
        message = {
            "type": msg_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all active connections
        disconnected = set()
        for ws in self.websockets:
            try:
                await ws.send_str(json.dumps(message))
            except Exception:
                disconnected.add(ws)
        
        # Clean up disconnected sockets
        self.websockets -= disconnected
    
    async def get_metrics(self, request) -> web.Response:
        """Get current metrics"""
        return web.json_response(self.metrics.to_dict())
    
    async def get_recent_changes(self, request) -> web.Response:
        """Get recent file changes"""
        return web.json_response({"recent_changes": self.recent_changes})
    
    async def get_recent_analyses(self, request) -> web.Response:
        """Get recent analysis results"""
        return web.json_response({"recent_analyses": self.recent_analyses})
    
    async def get_system_status(self, request) -> web.Response:
        """Get system status"""
        return web.json_response(self.system_status)
    
    async def _create_static_files(self, static_dir: Path):
        """Create necessary static files"""
        # Create empty files if needed - the HTML is self-contained
        pass
    
    async def start_data_updates(self):
        """Start background data update tasks"""
        # Metrics update task
        async def update_metrics():
            while True:
                try:
                    await self._fetch_metrics_from_mcp()
                    await asyncio.sleep(self.metrics_update_interval)
                except Exception as e:
                    logger.error(f"Metrics update error: {e}")
                    await asyncio.sleep(5)
        
        # Data update task
        async def update_data():
            while True:
                try:
                    await self._fetch_data_from_mcp()
                    await asyncio.sleep(self.data_update_interval)
                except Exception as e:
                    logger.error(f"Data update error: {e}")
                    await asyncio.sleep(5)
        
        # Start background tasks
        task1 = asyncio.create_task(update_metrics())
        task2 = asyncio.create_task(update_data())
        
        self.background_tasks.add(task1)
        self.background_tasks.add(task2)
        
        # Clean up completed tasks
        task1.add_done_callback(self.background_tasks.discard)
        task2.add_done_callback(self.background_tasks.discard)
    
    async def _fetch_metrics_from_mcp(self):
        """Fetch metrics from MCP server"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get real-time status
                status_request = {
                    "jsonrpc": "2.0",
                    "id": "dashboard_status",
                    "method": "tools/call",
                    "params": {"name": "realtime_get_status", "arguments": {}}
                }
                
                async with session.post(f"{self.mcp_server_url}/mcp", json=status_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", {})
                        if not result.get("isError", False):
                            content = result.get("content", [{}])[0]
                            if content.get("type") == "resource":
                                status_data = json.loads(content.get("data", "{}"))
                                await self._update_metrics_from_status(status_data)
                
                # Get statistics
                stats_request = {
                    "jsonrpc": "2.0",
                    "id": "dashboard_stats",
                    "method": "tools/call",
                    "params": {"name": "glean_statistics", "arguments": {}}
                }
                
                async with session.post(f"{self.mcp_server_url}/mcp", json=stats_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", {})
                        if not result.get("isError", False):
                            content = result.get("content", [{}])[0]
                            if content.get("type") == "resource":
                                stats_data = json.loads(content.get("data", "{}"))
                                await self._update_metrics_from_stats(stats_data)
                
        except Exception as e:
            logger.error(f"Failed to fetch metrics from MCP: {e}")
    
    async def _update_metrics_from_status(self, status_data: Dict[str, Any]):
        """Update metrics from status data"""
        watcher_stats = status_data.get("watcher_statistics", {})
        
        self.metrics.changes_detected = watcher_stats.get("changes_detected", 0)
        self.metrics.analyses_completed = watcher_stats.get("analyses_completed", 0)
        self.metrics.ai_insights_generated = watcher_stats.get("ai_insights_generated", 0)
        self.metrics.total_files_watched = watcher_stats.get("files_watched", 0)
        self.metrics.uptime_seconds = watcher_stats.get("uptime_seconds", 0)
        
        queue_stats = watcher_stats.get("queue_stats", {})
        self.metrics.average_analysis_time = queue_stats.get("avg_processing_time", 0)
        
        self.metrics.last_updated = datetime.now()
        self.system_status = status_data
        
        # Broadcast to WebSocket clients
        await self.broadcast_to_websockets('metrics', self.metrics.to_dict())
        await self.broadcast_to_websockets('system_status', self.system_status)
    
    async def _update_metrics_from_stats(self, stats_data: Dict[str, Any]):
        """Update metrics from statistics data"""
        realtime_stats = stats_data.get("realtime_statistics", {})
        if realtime_stats:
            # Update with any additional stats from Glean
            pass
    
    async def _fetch_data_from_mcp(self):
        """Fetch recent data from MCP server"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get recent changes
                changes_request = {
                    "jsonrpc": "2.0",
                    "id": "dashboard_changes",
                    "method": "tools/call",
                    "params": {"name": "realtime_get_changes", "arguments": {"limit": 20}}
                }
                
                async with session.post(f"{self.mcp_server_url}/mcp", json=changes_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", {})
                        if not result.get("isError", False):
                            content = result.get("content", [{}])[0]
                            if content.get("type") == "resource":
                                changes_data = json.loads(content.get("data", "{}"))
                                new_changes = changes_data.get("recent_changes", [])
                                
                                # Update and broadcast if new changes
                                if new_changes != self.recent_changes:
                                    self.recent_changes = new_changes
                                    await self.broadcast_to_websockets('file_changes', new_changes)
                
                # Get recent analyses
                analyses_request = {
                    "jsonrpc": "2.0",
                    "id": "dashboard_analyses",
                    "method": "tools/call",
                    "params": {"name": "realtime_get_analyses", "arguments": {"limit": 20}}
                }
                
                async with session.post(f"{self.mcp_server_url}/mcp", json=analyses_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", {})
                        if not result.get("isError", False):
                            content = result.get("content", [{}])[0]
                            if content.get("type") == "resource":
                                analyses_data = json.loads(content.get("data", "{}"))
                                new_analyses = analyses_data.get("recent_analyses", [])
                                
                                # Update and broadcast if new analyses
                                if new_analyses != self.recent_analyses:
                                    self.recent_analyses = new_analyses
                                    await self.broadcast_to_websockets('analysis_results', new_analyses)
                                    
                                    # Check for AI insights
                                    ai_insights = [a for a in new_analyses if a.get("ai_insights")]
                                    if ai_insights:
                                        await self.broadcast_to_websockets('ai_insights', ai_insights)
                
        except Exception as e:
            logger.error(f"Failed to fetch data from MCP: {e}")
    
    async def start_server(self, host: str = "localhost", port: int = 8090):
        """Start the dashboard server"""
        try:
            app = await self.create_app()
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, host, port)
            await site.start()
            
            self.server = (runner, site)
            
            # Start background data updates
            await self.start_data_updates()
            
            logger.info(f"🚀 Real-time Dashboard running on http://{host}:{port}")
            logger.info(f"   • Main Dashboard: http://{host}:{port}/")
            logger.info(f"   • API Metrics: http://{host}:{port}/api/metrics")
            logger.info(f"   • WebSocket: ws://{host}:{port}/ws")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            return False
    
    async def stop_server(self):
        """Stop the dashboard server"""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close WebSocket connections
        for ws in self.websockets:
            await ws.close()
        self.websockets.clear()
        
        # Stop server
        if self.server:
            runner, site = self.server
            await site.stop()
            await runner.cleanup()
            self.server = None
        
        logger.info("Dashboard server stopped")


# Factory function
async def create_dashboard(mcp_server_url: str = "http://localhost:8082") -> RealtimeDashboard:
    """Create a real-time dashboard instance"""
    return RealtimeDashboard(mcp_server_url)


# CLI for testing
async def main():
    """Main dashboard launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Real-time Code Analysis Dashboard")
    parser.add_argument("--host", default="localhost", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8090, help="Dashboard port")
    parser.add_argument("--mcp-url", default="http://localhost:8082", help="MCP server URL")
    
    args = parser.parse_args()
    
    # Create and start dashboard
    dashboard = await create_dashboard(args.mcp_url)
    
    try:
        if await dashboard.start_server(args.host, args.port):
            print(f"✅ Dashboard started successfully!")
            print(f"🌐 Open: http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        else:
            print("❌ Failed to start dashboard")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    finally:
        await dashboard.stop_server()


if __name__ == "__main__":
    asyncio.run(main())