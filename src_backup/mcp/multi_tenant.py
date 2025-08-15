"""
MCP Multi-Tenant Support
Lightweight multi-tenancy for crypto trading accounts
"""
import json
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from datetime import datetime
import logging

from .auth import AuthContext
from .cache import mcp_cache
from .metrics import mcp_metrics

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Tenant configuration"""
    tenant_id: str
    name: str
    trading_enabled: bool = True
    max_daily_trades: int = 100
    max_position_size: float = 10000.0
    allowed_symbols: List[str] = None
    risk_limits: Dict[str, float] = None
    api_rate_limits: Dict[str, int] = None
    
    def __post_init__(self):
        if self.allowed_symbols is None:
            self.allowed_symbols = ["BTC", "ETH", "USDT", "USDC"]
        if self.risk_limits is None:
            self.risk_limits = {
                "max_drawdown": 0.20,  # 20%
                "var_limit": 0.05,     # 5%
                "leverage_limit": 3.0
            }
        if self.api_rate_limits is None:
            self.api_rate_limits = {
                "requests_per_minute": 60,
                "requests_per_hour": 1000
            }


class TenantManager:
    """Manages tenant configurations and isolation"""
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self._load_default_tenants()
    
    def _load_default_tenants(self):
        """Load default tenant configurations"""
        # Default tenant for general use
        self.tenants["default"] = TenantConfig(
            tenant_id="default",
            name="Default Tenant",
            trading_enabled=True,
            max_daily_trades=50,
            max_position_size=5000.0
        )
        
        # Demo tenant with restricted access
        self.tenants["demo"] = TenantConfig(
            tenant_id="demo",
            name="Demo Account",
            trading_enabled=False,  # Demo mode - no real trading
            max_daily_trades=10,
            max_position_size=1000.0,
            allowed_symbols=["BTC", "ETH"],
            api_rate_limits={
                "requests_per_minute": 30,
                "requests_per_hour": 500
            }
        )
        
        # Premium tenant with higher limits
        self.tenants["premium"] = TenantConfig(
            tenant_id="premium",
            name="Premium Account",
            trading_enabled=True,
            max_daily_trades=500,
            max_position_size=50000.0,
            api_rate_limits={
                "requests_per_minute": 120,
                "requests_per_hour": 5000
            }
        )
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration"""
        return self.tenants.get(tenant_id)
    
    def create_tenant(self, config: TenantConfig) -> bool:
        """Create new tenant"""
        if config.tenant_id in self.tenants:
            return False
        
        self.tenants[config.tenant_id] = config
        logger.info(f"Created tenant: {config.tenant_id}")
        return True
    
    def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Update tenant configuration"""
        if tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[tenant_id]
        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        logger.info(f"Updated tenant: {tenant_id}")
        return True
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant"""
        if tenant_id in ["default", "demo", "premium"]:
            return False  # Cannot delete system tenants
        
        if tenant_id in self.tenants:
            del self.tenants[tenant_id]
            logger.info(f"Deleted tenant: {tenant_id}")
            return True
        
        return False
    
    def list_tenants(self) -> List[Dict[str, Any]]:
        """List all tenants"""
        return [
            {
                "tenant_id": config.tenant_id,
                "name": config.name,
                "trading_enabled": config.trading_enabled,
                "max_daily_trades": config.max_daily_trades,
                "max_position_size": config.max_position_size
            }
            for config in self.tenants.values()
        ]


class TenantIsolation:
    """Provides tenant isolation for data and operations"""
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
    
    def get_tenant_cache_key(self, tenant_id: str, base_key: str) -> str:
        """Generate tenant-specific cache key"""
        return f"tenant:{tenant_id}:{base_key}"
    
    def get_tenant_data(self, tenant_id: str, data_type: str, key: str) -> Optional[Any]:
        """Get tenant-specific data from cache"""
        cache_key = self.get_tenant_cache_key(tenant_id, f"{data_type}:{key}")
        return mcp_cache.cache.get(cache_key)
    
    def set_tenant_data(self, tenant_id: str, data_type: str, key: str, 
                       value: Any, ttl: int = 300) -> None:
        """Set tenant-specific data in cache"""
        cache_key = self.get_tenant_cache_key(tenant_id, f"{data_type}:{key}")
        mcp_cache.cache.set(cache_key, value, ttl)
    
    def delete_tenant_data(self, tenant_id: str, data_type: str = None) -> int:
        """Delete tenant-specific data"""
        if data_type:
            pattern = f"tenant:{tenant_id}:{data_type}"
        else:
            pattern = f"tenant:{tenant_id}"
        
        return mcp_cache.invalidate_pattern(pattern)
    
    def check_trading_permission(self, auth_context: AuthContext, 
                               operation: str, symbol: str = None) -> Dict[str, Any]:
        """Check if tenant has permission for trading operation"""
        tenant = self.tenant_manager.get_tenant(auth_context.tenant_id)
        if not tenant:
            return {
                "allowed": False,
                "reason": "Invalid tenant"
            }
        
        # Check if trading is enabled
        if not tenant.trading_enabled:
            return {
                "allowed": False,
                "reason": "Trading is disabled for this tenant"
            }
        
        # Check symbol restrictions
        if symbol and symbol not in tenant.allowed_symbols:
            return {
                "allowed": False,
                "reason": f"Symbol {symbol} not allowed for this tenant"
            }
        
        # Check daily trade limits
        today = datetime.now().strftime("%Y-%m-%d")
        trades_key = f"trades:{today}"
        daily_trades = self.get_tenant_data(auth_context.tenant_id, "counters", trades_key) or 0
        
        if daily_trades >= tenant.max_daily_trades:
            return {
                "allowed": False,
                "reason": "Daily trade limit exceeded"
            }
        
        return {
            "allowed": True,
            "remaining_trades": tenant.max_daily_trades - daily_trades
        }
    
    def record_trading_operation(self, auth_context: AuthContext, 
                               operation: str, symbol: str, amount: float):
        """Record trading operation for tenant"""
        # Update daily trade counter
        today = datetime.now().strftime("%Y-%m-%d")
        trades_key = f"trades:{today}"
        current_trades = self.get_tenant_data(auth_context.tenant_id, "counters", trades_key) or 0
        self.set_tenant_data(auth_context.tenant_id, "counters", trades_key, current_trades + 1, 86400)
        
        # Record metrics
        mcp_metrics.trading_operation(operation, symbol, True)
        mcp_metrics.collector.counter(
            "mcp.tenant.trading_operations",
            tags={
                "tenant": auth_context.tenant_id,
                "operation": operation,
                "symbol": symbol
            }
        )
    
    def check_position_limits(self, auth_context: AuthContext, 
                            symbol: str, amount: float) -> Dict[str, Any]:
        """Check position size limits"""
        tenant = self.tenant_manager.get_tenant(auth_context.tenant_id)
        if not tenant:
            return {"allowed": False, "reason": "Invalid tenant"}
        
        if amount > tenant.max_position_size:
            return {
                "allowed": False,
                "reason": f"Position size {amount} exceeds limit {tenant.max_position_size}"
            }
        
        return {"allowed": True}
    
    def get_tenant_portfolio(self, auth_context: AuthContext) -> Dict[str, Any]:
        """Get tenant-specific portfolio data"""
        portfolio_key = "portfolio:current"
        portfolio = self.get_tenant_data(auth_context.tenant_id, "portfolio", portfolio_key)
        
        if not portfolio:
            # Default empty portfolio
            portfolio = {
                "total_value": 0.0,
                "positions": {},
                "cash_balance": 0.0,
                "last_updated": datetime.now().isoformat()
            }
            self.set_tenant_data(auth_context.tenant_id, "portfolio", portfolio_key, portfolio)
        
        return portfolio
    
    def update_tenant_portfolio(self, auth_context: AuthContext, 
                              portfolio_data: Dict[str, Any]):
        """Update tenant-specific portfolio"""
        portfolio_key = "portfolio:current"
        portfolio_data["last_updated"] = datetime.now().isoformat()
        self.set_tenant_data(auth_context.tenant_id, "portfolio", portfolio_key, portfolio_data)


class TenantAwareTools:
    """Tenant-aware versions of MCP tools"""
    
    def __init__(self, tenant_isolation: TenantIsolation):
        self.isolation = tenant_isolation
    
    async def get_portfolio(self, auth_context: AuthContext, 
                          include_history: bool = False) -> Dict[str, Any]:
        """Get tenant-specific portfolio"""
        # Check permissions
        if not auth_context or not auth_context.tenant_id:
            raise ValueError("Valid authentication context required")
        
        portfolio = self.isolation.get_tenant_portfolio(auth_context)
        
        # Add tenant-specific metadata
        tenant = self.isolation.tenant_manager.get_tenant(auth_context.tenant_id)
        if tenant:
            portfolio["tenant_info"] = {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "trading_enabled": tenant.trading_enabled,
                "max_position_size": tenant.max_position_size
            }
        
        return portfolio
    
    async def execute_trade(self, auth_context: AuthContext, 
                          symbol: str, side: str, amount: float,
                          order_type: str = "market") -> Dict[str, Any]:
        """Execute tenant-aware trade"""
        # Check trading permissions
        permission_check = self.isolation.check_trading_permission(
            auth_context, "trade", symbol
        )
        
        if not permission_check["allowed"]:
            raise ValueError(permission_check["reason"])
        
        # Check position limits
        position_check = self.isolation.check_position_limits(
            auth_context, symbol, amount
        )
        
        if not position_check["allowed"]:
            raise ValueError(position_check["reason"])
        
        # Record the operation
        self.isolation.record_trading_operation(
            auth_context, "trade", symbol, amount
        )
        
        # Execute trade (mock implementation)
        trade_result = {
            "order_id": f"order_{auth_context.tenant_id}_{int(datetime.now().timestamp())}",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "order_type": order_type,
            "status": "filled",
            "tenant_id": auth_context.tenant_id,
            "timestamp": datetime.now().isoformat()
        }
        
        return trade_result
    
    async def get_risk_metrics(self, auth_context: AuthContext, 
                             scope: str = "portfolio") -> Dict[str, Any]:
        """Get tenant-specific risk metrics"""
        tenant = self.isolation.tenant_manager.get_tenant(auth_context.tenant_id)
        if not tenant:
            raise ValueError("Invalid tenant")
        
        portfolio = self.isolation.get_tenant_portfolio(auth_context)
        
        # Calculate basic risk metrics
        risk_metrics = {
            "tenant_id": auth_context.tenant_id,
            "portfolio_value": portfolio.get("total_value", 0),
            "risk_limits": tenant.risk_limits,
            "position_utilization": portfolio.get("total_value", 0) / tenant.max_position_size,
            "daily_trades_used": 0,  # Would be calculated from actual data
            "max_daily_trades": tenant.max_daily_trades
        }
        
        return risk_metrics


# Global instances
global_tenant_manager = TenantManager()
tenant_isolation = TenantIsolation(global_tenant_manager)
tenant_aware_tools = TenantAwareTools(tenant_isolation)


def get_tenant_config(tenant_id: str) -> Optional[TenantConfig]:
    """Get tenant configuration"""
    return global_tenant_manager.get_tenant(tenant_id)


def check_tenant_permission(auth_context: AuthContext, operation: str, 
                          symbol: str = None) -> Dict[str, Any]:
    """Check tenant permissions for operation"""
    return tenant_isolation.check_trading_permission(auth_context, operation, symbol)


def get_tenant_stats(tenant_id: str) -> Dict[str, Any]:
    """Get tenant usage statistics"""
    tenant = global_tenant_manager.get_tenant(tenant_id)
    if not tenant:
        return {"error": "Tenant not found"}
    
    # Get cached statistics
    today = datetime.now().strftime("%Y-%m-%d")
    daily_trades = tenant_isolation.get_tenant_data(tenant_id, "counters", f"trades:{today}") or 0
    
    return {
        "tenant_id": tenant_id,
        "name": tenant.name,
        "daily_trades": daily_trades,
        "max_daily_trades": tenant.max_daily_trades,
        "trading_enabled": tenant.trading_enabled,
        "utilization": {
            "daily_trades": daily_trades / tenant.max_daily_trades,
        }
    }
