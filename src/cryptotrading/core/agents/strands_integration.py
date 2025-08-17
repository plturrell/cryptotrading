"""
Strands Framework Integration Management
Manages integrations with external systems, APIs, and data sources
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Types of integrations"""
    API = "api"
    DATABASE = "database"
    MESSAGING = "messaging"
    FILE_SYSTEM = "file_system"
    WEBHOOK = "webhook"
    STREAM = "stream"

class IntegrationStatus(Enum):
    """Integration status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    DEGRADED = "degraded"

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    burst_size: int = 5
    
@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

@dataclass
class IntegrationMetrics:
    """Metrics for an integration"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    uptime_percentage: float = 100.0

class BaseIntegration(ABC):
    """Base class for all integrations"""
    
    def __init__(self, integration_id: str, config: Dict[str, Any]):
        self.integration_id = integration_id
        self.config = config
        self.status = IntegrationStatus.DISCONNECTED
        self.metrics = IntegrationMetrics()
        self.rate_limiter = None
        self.retry_config = RetryConfig()
        self._last_health_check = datetime.utcnow()
        self._connection_attempts = 0
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the integration"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the integration"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if integration is healthy"""
        pass
    
    @abstractmethod
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a request through this integration"""
        pass
    
    def update_metrics(self, success: bool, response_time: float = 0.0, error: str = None):
        """Update integration metrics"""
        self.metrics.total_requests += 1
        self.metrics.last_request_time = datetime.utcnow()
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            if error:
                self.metrics.last_error = error
        
        # Update average response time
        if response_time > 0:
            total_successful = self.metrics.successful_requests
            if total_successful > 0:
                self.metrics.avg_response_time = (
                    (self.metrics.avg_response_time * (total_successful - 1) + response_time) / total_successful
                )

class APIIntegration(BaseIntegration):
    """Integration for REST APIs"""
    
    def __init__(self, integration_id: str, config: Dict[str, Any]):
        super().__init__(integration_id, config)
        self.base_url = config.get('base_url', '')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 30.0)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        rate_limit = config.get('rate_limit', {})
        self.rate_limit_config = RateLimitConfig(**rate_limit) if rate_limit else RateLimitConfig()
        
    async def connect(self) -> bool:
        """Connect to the API"""
        try:
            self.status = IntegrationStatus.CONNECTING
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
            
            # Test connection
            health_ok = await self.health_check()
            if health_ok:
                self.status = IntegrationStatus.CONNECTED
                logger.info(f"API integration {self.integration_id} connected")
                return True
            else:
                self.status = IntegrationStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect API integration {self.integration_id}: {e}")
            self.status = IntegrationStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the API"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.status = IntegrationStatus.DISCONNECTED
            logger.info(f"API integration {self.integration_id} disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting API integration {self.integration_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check API health"""
        try:
            if not self.session:
                return False
            
            health_endpoint = self.config.get('health_endpoint', '/health')
            url = f"{self.base_url.rstrip('/')}{health_endpoint}"
            
            async with self.session.get(url) as response:
                is_healthy = response.status < 400
                self._last_health_check = datetime.utcnow()
                return is_healthy
                
        except Exception as e:
            logger.error(f"Health check failed for {self.integration_id}: {e}")
            return False
    
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API request"""
        if self.status != IntegrationStatus.CONNECTED:
            return {"success": False, "error": "Integration not connected"}
        
        start_time = time.time()
        
        try:
            # Apply rate limiting
            await self._apply_rate_limit()
            
            method = request.get('method', 'GET').upper()
            endpoint = request.get('endpoint', '/')
            url = f"{self.base_url.rstrip('/')}{endpoint}"
            params = request.get('params', {})
            data = request.get('data')
            headers = request.get('headers', {})
            
            # Execute request
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data if data else None,
                headers=headers
            ) as response:
                
                response_time = time.time() - start_time
                response_data = await response.text()
                
                try:
                    response_json = json.loads(response_data)
                except json.JSONDecodeError:
                    response_json = {"raw_response": response_data}
                
                success = response.status < 400
                
                result = {
                    "success": success,
                    "status_code": response.status,
                    "data": response_json,
                    "response_time": response_time,
                    "headers": dict(response.headers)
                }
                
                if not success:
                    result["error"] = f"HTTP {response.status}: {response_data}"
                
                self.update_metrics(success, response_time, 
                                  result.get("error") if not success else None)
                
                return result
                
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            self.update_metrics(False, response_time, error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "response_time": response_time
            }
    
    async def _apply_rate_limit(self):
        """Apply rate limiting to requests"""
        # Simple rate limiting implementation
        # In production, would use more sophisticated rate limiting
        if hasattr(self, '_last_request_time'):
            time_since_last = time.time() - self._last_request_time
            min_interval = 1.0 / self.rate_limit_config.requests_per_second
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
        
        self._last_request_time = time.time()

class DatabaseIntegration(BaseIntegration):
    """Integration for databases"""
    
    def __init__(self, integration_id: str, config: Dict[str, Any]):
        super().__init__(integration_id, config)
        self.connection_string = config.get('connection_string')
        self.pool = None
        
    async def connect(self) -> bool:
        """Connect to database"""
        try:
            self.status = IntegrationStatus.CONNECTING
            
            # Database connection logic would go here
            # For now, simulate connection
            await asyncio.sleep(0.1)
            
            self.status = IntegrationStatus.CONNECTED
            logger.info(f"Database integration {self.integration_id} connected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect database integration {self.integration_id}: {e}")
            self.status = IntegrationStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from database"""
        try:
            if self.pool:
                # Close database pool
                pass
            
            self.status = IntegrationStatus.DISCONNECTED
            logger.info(f"Database integration {self.integration_id} disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting database integration {self.integration_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            # Simple health check query
            # In production, would execute actual query
            return self.status == IntegrationStatus.CONNECTED
            
        except Exception as e:
            logger.error(f"Database health check failed for {self.integration_id}: {e}")
            return False
    
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database query"""
        if self.status != IntegrationStatus.CONNECTED:
            return {"success": False, "error": "Database not connected"}
        
        start_time = time.time()
        
        try:
            query = request.get('query', '')
            params = request.get('params', [])
            
            # Database query execution would go here
            # For now, simulate query
            await asyncio.sleep(0.01)
            
            response_time = time.time() - start_time
            result = {
                "success": True,
                "data": {"simulated": True, "query": query},
                "response_time": response_time,
                "rows_affected": 0
            }
            
            self.update_metrics(True, response_time)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            self.update_metrics(False, response_time, error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "response_time": response_time
            }

class IntegrationManager:
    """Manages all integrations for the Strands framework"""
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.integration_factories: Dict[IntegrationType, Callable] = {
            IntegrationType.API: APIIntegration,
            IntegrationType.DATABASE: DatabaseIntegration,
        }
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = 30.0  # seconds
        
    def register_integration_factory(self, integration_type: IntegrationType, 
                                   factory: Callable[[str, Dict[str, Any]], BaseIntegration]):
        """Register a factory for creating integrations of a specific type"""
        self.integration_factories[integration_type] = factory
    
    async def create_integration(self, integration_id: str, integration_type: IntegrationType,
                               config: Dict[str, Any]) -> bool:
        """Create and register a new integration"""
        try:
            if integration_type not in self.integration_factories:
                logger.error(f"No factory registered for integration type {integration_type}")
                return False
            
            factory = self.integration_factories[integration_type]
            integration = factory(integration_id, config)
            
            self.integrations[integration_id] = integration
            logger.info(f"Created integration {integration_id} of type {integration_type.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create integration {integration_id}: {e}")
            return False
    
    async def connect_integration(self, integration_id: str) -> bool:
        """Connect a specific integration"""
        if integration_id not in self.integrations:
            logger.error(f"Integration {integration_id} not found")
            return False
        
        integration = self.integrations[integration_id]
        return await integration.connect()
    
    async def disconnect_integration(self, integration_id: str) -> bool:
        """Disconnect a specific integration"""
        if integration_id not in self.integrations:
            logger.error(f"Integration {integration_id} not found")
            return False
        
        integration = self.integrations[integration_id]
        return await integration.disconnect()
    
    async def execute_request(self, integration_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a request through a specific integration"""
        if integration_id not in self.integrations:
            return {"success": False, "error": f"Integration {integration_id} not found"}
        
        integration = self.integrations[integration_id]
        return await integration.execute_request(request)
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all integrations"""
        results = {}
        
        for integration_id, integration in self.integrations.items():
            try:
                results[integration_id] = await integration.connect()
            except Exception as e:
                logger.error(f"Failed to connect integration {integration_id}: {e}")
                results[integration_id] = False
        
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect all integrations"""
        results = {}
        
        for integration_id, integration in self.integrations.items():
            try:
                results[integration_id] = await integration.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect integration {integration_id}: {e}")
                results[integration_id] = False
        
        return results
    
    async def start_monitoring(self):
        """Start monitoring all integrations"""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already running")
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started integration monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring integrations"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped integration monitoring")
    
    async def get_integration_status(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific integration"""
        if integration_id not in self.integrations:
            return None
        
        integration = self.integrations[integration_id]
        
        return {
            "integration_id": integration_id,
            "status": integration.status.value,
            "metrics": {
                "total_requests": integration.metrics.total_requests,
                "successful_requests": integration.metrics.successful_requests,
                "failed_requests": integration.metrics.failed_requests,
                "success_rate": (
                    integration.metrics.successful_requests / integration.metrics.total_requests
                    if integration.metrics.total_requests > 0 else 0.0
                ),
                "avg_response_time": integration.metrics.avg_response_time,
                "last_request_time": (
                    integration.metrics.last_request_time.isoformat()
                    if integration.metrics.last_request_time else None
                ),
                "last_error": integration.metrics.last_error
            }
        }
    
    async def get_all_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        status = {}
        
        for integration_id in self.integrations:
            status[integration_id] = await self.get_integration_status(integration_id)
        
        # Calculate overall health
        total_integrations = len(self.integrations)
        healthy_integrations = sum(
            1 for integration in self.integrations.values()
            if integration.status == IntegrationStatus.CONNECTED
        )
        
        overall_health = healthy_integrations / total_integrations if total_integrations > 0 else 1.0
        
        return {
            "overall_health": overall_health,
            "total_integrations": total_integrations,
            "healthy_integrations": healthy_integrations,
            "integrations": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Check health of all integrations
                for integration_id, integration in self.integrations.items():
                    try:
                        is_healthy = await integration.health_check()
                        
                        if not is_healthy and integration.status == IntegrationStatus.CONNECTED:
                            logger.warning(f"Integration {integration_id} health check failed")
                            integration.status = IntegrationStatus.DEGRADED
                        elif is_healthy and integration.status == IntegrationStatus.DEGRADED:
                            logger.info(f"Integration {integration_id} recovered")
                            integration.status = IntegrationStatus.CONNECTED
                        
                    except Exception as e:
                        logger.error(f"Error checking health of integration {integration_id}: {e}")
                        integration.status = IntegrationStatus.ERROR
                
                await asyncio.sleep(self._monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self._monitoring_interval)

# Global integration manager instance
_integration_manager: Optional[IntegrationManager] = None

def get_integration_manager() -> IntegrationManager:
    """Get the global integration manager instance"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager()
    return _integration_manager

# Helper functions for common integration operations
async def setup_yahoo_finance_integration() -> bool:
    """Setup Yahoo Finance API integration"""
    manager = get_integration_manager()
    
    config = {
        "base_url": "https://query1.finance.yahoo.com",
        "timeout": 30.0,
        "rate_limit": {
            "requests_per_second": 2.0,  # Conservative rate limiting
            "requests_per_minute": 100.0,
            "requests_per_hour": 1000.0
        },
        "headers": {
            "User-Agent": "Strands-Framework/1.0"
        },
        "health_endpoint": "/v8/finance/chart/AAPL"  # Use AAPL as health check
    }
    
    success = await manager.create_integration("yahoo_finance", IntegrationType.API, config)
    if success:
        return await manager.connect_integration("yahoo_finance")
    
    return False

async def setup_database_integration(connection_string: str) -> bool:
    """Setup database integration"""
    manager = get_integration_manager()
    
    config = {
        "connection_string": connection_string,
        "pool_size": 10,
        "timeout": 30.0
    }
    
    success = await manager.create_integration("main_database", IntegrationType.DATABASE, config)
    if success:
        return await manager.connect_integration("main_database")
    
    return False

# Rate limiting decorator
def rate_limited(requests_per_second: float = 1.0):
    """Decorator for rate limiting function calls"""
    def decorator(func):
        last_called = [0.0]
        
        async def wrapper(*args, **kwargs):
            now = time.time()
            time_since_last = now - last_called[0]
            min_interval = 1.0 / requests_per_second
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator