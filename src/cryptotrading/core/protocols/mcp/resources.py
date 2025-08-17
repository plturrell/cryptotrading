"""
MCP Resources Implementation
Defines resource structures and access for MCP protocol
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path


@dataclass
class ResourceTemplate:
    """Template for parameterized resources"""
    uriTemplate: str
    name: str
    description: str
    mimeType: str


class Resource(ABC):
    """Abstract base class for MCP resources"""
    
    def __init__(self, uri: str, name: str, description: str, mime_type: str = "text/plain"):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type
    
    @abstractmethod
    async def read(self) -> str:
        """Read the resource content"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to MCP format"""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


class FileResource(Resource):
    """File-based resource"""
    
    def __init__(self, file_path: str, uri: str, name: str, description: str, mime_type: str = "text/plain"):
        super().__init__(uri, name, description, mime_type)
        self.file_path = Path(file_path)
    
    async def read(self) -> str:
        """Read file content"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Resource file not found: {self.file_path}")
        
        return self.file_path.read_text(encoding='utf-8')


class JsonResource(Resource):
    """JSON data resource"""
    
    def __init__(self, data: Dict[str, Any], uri: str, name: str, description: str):
        super().__init__(uri, name, description, "application/json")
        self.data = data
    
    async def read(self) -> str:
        """Return JSON data as string"""
        return json.dumps(self.data, indent=2)


class DynamicResource(Resource):
    """Dynamic resource with callback function"""
    
    def __init__(self, callback: Callable[[], Any], uri: str, name: str, description: str, mime_type: str = "application/json"):
        super().__init__(uri, name, description, mime_type)
        self.callback = callback
    
    async def read(self) -> str:
        """Execute callback and return result"""
        result = self.callback()
        if isinstance(result, dict) or isinstance(result, list):
            return json.dumps(result, indent=2)
        return str(result)


# Alias for backward compatibility
MCPResource = Resource

class CryptoTradingResources:
    """Crypto trading specific resources for MCP server"""
    
    @staticmethod
    def get_config_resource() -> Resource:
        """Get trading configuration resource"""
        def get_config():
            return {
                "trading_pairs": ["BTC-USD", "ETH-USD", "SOL-USD"],
                "risk_limits": {
                    "max_position_size": 0.1,
                    "max_daily_loss": 0.05,
                    "stop_loss_percentage": 0.02
                },
                "exchanges": ["binance", "coinbase", "kraken"],
                "strategies": ["momentum", "mean_reversion", "arbitrage"]
            }
        
        return DynamicResource(
            callback=get_config,
            uri="crypto://config/trading",
            name="Trading Configuration",
            description="Current trading system configuration and parameters"
        )
    
    @staticmethod
    def get_portfolio_resource() -> Resource:
        """Get current portfolio resource"""
        def get_portfolio():
            return {
                "timestamp": "2024-01-01T12:00:00Z",
                "total_value_usd": 100000.00,
                "positions": [
                    {
                        "symbol": "BTC",
                        "amount": 2.5,
                        "avg_price": 30000.00,
                        "current_price": 32000.00,
                        "unrealized_pnl": 5000.00
                    },
                    {
                        "symbol": "ETH",
                        "amount": 10.0,
                        "avg_price": 2000.00,
                        "current_price": 2100.00,
                        "unrealized_pnl": 1000.00
                    }
                ],
                "cash_balance": 44000.00,
                "performance": {
                    "daily_pnl": 1500.00,
                    "weekly_pnl": 5000.00,
                    "monthly_pnl": 12000.00
                }
            }
        
        return DynamicResource(
            callback=get_portfolio,
            uri="crypto://portfolio/current",
            name="Current Portfolio",
            description="Real-time portfolio positions and performance metrics"
        )
    
    @staticmethod
    def get_market_status_resource() -> Resource:
        """Get market status resource"""
        def get_market_status():
            return {
                "market_state": "open",
                "trading_active": True,
                "last_update": "2024-01-01T12:00:00Z",
                "exchanges": {
                    "binance": {"status": "online", "latency_ms": 50},
                    "coinbase": {"status": "online", "latency_ms": 75},
                    "kraken": {"status": "online", "latency_ms": 60}
                },
                "data_feeds": {
                    "price_feed": {"status": "active", "last_update": "2024-01-01T12:00:00Z"},
                    "order_book": {"status": "active", "depth": 10},
                    "trade_stream": {"status": "active", "msg_per_sec": 150}
                }
            }
        
        return DynamicResource(
            callback=get_market_status,
            uri="crypto://market/status",
            name="Market Status",
            description="Current market and system status information"
        )
    
    @staticmethod
    def get_risk_metrics_resource() -> Resource:
        """Get risk metrics resource"""
        def get_risk_metrics():
            return {
                "portfolio_var": {
                    "var_95_1d": -2500.00,
                    "var_99_1d": -4000.00,
                    "expected_shortfall": -5500.00
                },
                "position_limits": {
                    "max_single_position": 0.25,
                    "sector_concentration_limit": 0.40,
                    "correlation_limit": 0.70
                },
                "risk_alerts": [
                    {
                        "level": "warning",
                        "message": "ETH position approaching concentration limit",
                        "threshold": 0.22,
                        "current": 0.20
                    }
                ],
                "stress_test_results": {
                    "market_crash_scenario": -15000.00,
                    "volatility_spike_scenario": -8000.00,
                    "liquidity_crisis_scenario": -12000.00
                }
            }
        
        return DynamicResource(
            callback=get_risk_metrics,
            uri="crypto://risk/metrics",
            name="Risk Metrics",
            description="Portfolio risk analysis and stress testing results"
        )
    
    @staticmethod
    def get_strategy_performance_resource() -> Resource:
        """Get strategy performance resource"""
        def get_strategy_performance():
            return {
                "strategies": [
                    {
                        "name": "momentum_strategy",
                        "status": "active",
                        "allocated_capital": 30000.00,
                        "performance": {
                            "total_return": 0.15,
                            "sharpe_ratio": 1.8,
                            "max_drawdown": -0.08,
                            "win_rate": 0.65
                        },
                        "trades_today": 12,
                        "last_signal": "2024-01-01T11:45:00Z"
                    },
                    {
                        "name": "mean_reversion_strategy",
                        "status": "active",
                        "allocated_capital": 25000.00,
                        "performance": {
                            "total_return": 0.08,
                            "sharpe_ratio": 1.2,
                            "max_drawdown": -0.05,
                            "win_rate": 0.72
                        },
                        "trades_today": 8,
                        "last_signal": "2024-01-01T11:30:00Z"
                    }
                ],
                "aggregate_performance": {
                    "total_return": 0.12,
                    "sharpe_ratio": 1.6,
                    "correlation_matrix": {
                        "momentum_mean_reversion": 0.15
                    }
                }
            }
        
        return DynamicResource(
            callback=get_strategy_performance,
            uri="crypto://strategies/performance",
            name="Strategy Performance",
            description="Performance metrics and status for all active trading strategies"
        )
    
    @staticmethod
    def get_log_resource(log_type: str = "trading") -> Resource:
        """Get log resource"""
        def get_logs():
            logs = {
                "trading": [
                    {"timestamp": "2024-01-01T12:00:00Z", "level": "INFO", "message": "BTC buy order filled at $32000"},
                    {"timestamp": "2024-01-01T11:58:00Z", "level": "INFO", "message": "ETH sell signal generated"},
                    {"timestamp": "2024-01-01T11:55:00Z", "level": "WARNING", "message": "High volatility detected in SOL"}
                ],
                "system": [
                    {"timestamp": "2024-01-01T12:00:00Z", "level": "INFO", "message": "System health check passed"},
                    {"timestamp": "2024-01-01T11:59:00Z", "level": "ERROR", "message": "Temporary connection loss to Kraken"},
                    {"timestamp": "2024-01-01T11:58:00Z", "level": "INFO", "message": "Reconnected to all exchanges"}
                ]
            }
            return logs.get(log_type, [])
        
        return DynamicResource(
            callback=get_logs,
            uri=f"crypto://logs/{log_type}",
            name=f"{log_type.title()} Logs",
            description=f"Recent {log_type} log entries"
        )