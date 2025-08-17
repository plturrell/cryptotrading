"""
Real Service Discovery for A2A Network
Uses blockchain and local discovery mechanisms
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    service_id: str
    agent_id: str
    host: str
    port: int
    protocol: str  # tcp, http, websocket, blockchain
    capabilities: List[str]
    status: ServiceStatus
    last_seen: datetime
    metadata: Dict[str, Any]
    health_score: float = 1.0  # 0.0 to 1.0

class ServiceDiscovery:
    """
    Real service discovery implementation
    Combines blockchain registry with local discovery
    """
    
    def __init__(self, agent_id: str, blockchain_client=None):
        self.agent_id = agent_id
        self.blockchain_client = blockchain_client
        self.logger = logging.getLogger(f"ServiceDiscovery-{agent_id}")
        
        # Local service registry
        self.local_services: Dict[str, ServiceEndpoint] = {}
        self.service_cache: Dict[str, ServiceEndpoint] = {}
        
        # Discovery mechanisms
        self.discovery_methods = []
        if blockchain_client:
            self.discovery_methods.append("blockchain")
        self.discovery_methods.extend(["multicast", "dns"])
        
        # Configuration
        self.cache_ttl = 300  # 5 minutes
        self.health_check_interval = 30  # 30 seconds
        self.discovery_interval = 60  # 1 minute
        
        # Background tasks
        self._tasks: Set[asyncio.Task] = set()
        self._running = False
        
        self.logger.info(f"Service discovery initialized with methods: {self.discovery_methods}")
    
    async def start(self):
        """Start service discovery"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        tasks = [
            self._periodic_discovery(),
            self._health_checker(),
            self._cache_cleaner()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        
        self.logger.info("Service discovery started")
    
    async def stop(self):
        """Stop service discovery"""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self.logger.info("Service discovery stopped")
    
    async def register_service(self, service_id: str, endpoint_info: Dict[str, Any]) -> bool:
        """Register a service"""
        try:
            endpoint = ServiceEndpoint(
                service_id=service_id,
                agent_id=endpoint_info.get("agent_id", self.agent_id),
                host=endpoint_info.get("host", "localhost"),
                port=endpoint_info.get("port", 0),
                protocol=endpoint_info.get("protocol", "tcp"),
                capabilities=endpoint_info.get("capabilities", []),
                status=ServiceStatus.ACTIVE,
                last_seen=datetime.utcnow(),
                metadata=endpoint_info.get("metadata", {}),
                health_score=1.0
            )
            
            # Register locally
            self.local_services[service_id] = endpoint
            self.service_cache[service_id] = endpoint
            
            # Register on blockchain if available
            if self.blockchain_client:
                success = await self.blockchain_client.register_agent(
                    service_id,
                    endpoint.capabilities,
                    f"{endpoint.host}:{endpoint.port}"
                )
                if not success:
                    self.logger.warning(f"Blockchain registration failed for {service_id}")
            
            self.logger.info(f"Registered service {service_id} at {endpoint.host}:{endpoint.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service {service_id}: {e}")
            return False
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister a service"""
        try:
            if service_id in self.local_services:
                del self.local_services[service_id]
            
            if service_id in self.service_cache:
                self.service_cache[service_id].status = ServiceStatus.INACTIVE
            
            self.logger.info(f"Unregistered service {service_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister service {service_id}: {e}")
            return False
    
    async def discover_services(self, capability: Optional[str] = None) -> List[ServiceEndpoint]:
        """Discover services by capability"""
        discovered_services = []
        
        # Search local cache first
        for service in self.service_cache.values():
            if self._is_service_valid(service) and self._matches_capability(service, capability):
                discovered_services.append(service)
        
        # Search via blockchain
        if self.blockchain_client:
            try:
                blockchain_services = await self._discover_via_blockchain(capability)
                for service in blockchain_services:
                    # Add to cache and results if not already present
                    if service.service_id not in self.service_cache:
                        self.service_cache[service.service_id] = service
                        discovered_services.append(service)
            except Exception as e:
                self.logger.error(f"Blockchain discovery failed: {e}")
        
        # Search via multicast
        try:
            multicast_services = await self._discover_via_multicast(capability)
            for service in multicast_services:
                if service.service_id not in self.service_cache:
                    self.service_cache[service.service_id] = service
                    discovered_services.append(service)
        except Exception as e:
            self.logger.debug(f"Multicast discovery failed: {e}")
        
        # Sort by health score
        discovered_services.sort(key=lambda s: s.health_score, reverse=True)
        
        self.logger.info(f"Discovered {len(discovered_services)} services for capability '{capability}'")
        return discovered_services
    
    async def get_service(self, service_id: str) -> Optional[ServiceEndpoint]:
        """Get specific service by ID"""
        # Check local cache first
        if service_id in self.service_cache:
            service = self.service_cache[service_id]
            if self._is_service_valid(service):
                return service
        
        # Try discovery
        services = await self.discover_services()
        for service in services:
            if service.service_id == service_id:
                return service
        
        return None
    
    async def list_services(self) -> List[Dict[str, Any]]:
        """List all known services"""
        services = []
        for service in self.service_cache.values():
            if self._is_service_valid(service):
                services.append(asdict(service))
        
        return services
    
    def _is_service_valid(self, service: ServiceEndpoint) -> bool:
        """Check if service is still valid"""
        # Check TTL
        age = (datetime.utcnow() - service.last_seen).total_seconds()
        if age > self.cache_ttl:
            return False
        
        # Check status
        if service.status == ServiceStatus.INACTIVE:
            return False
        
        return True
    
    def _matches_capability(self, service: ServiceEndpoint, capability: Optional[str]) -> bool:
        """Check if service matches capability requirement"""
        if not capability:
            return True
        
        return capability in service.capabilities
    
    async def _discover_via_blockchain(self, capability: Optional[str]) -> List[ServiceEndpoint]:
        """Discover services via blockchain"""
        services = []
        
        try:
            # Get agent list from blockchain
            agent_list = await self.blockchain_client.get_agent_list()
            
            for agent_id in agent_list:
                if agent_id == self.agent_id:
                    continue  # Skip self
                
                # Create service endpoint (simplified)
                service = ServiceEndpoint(
                    service_id=agent_id,
                    agent_id=agent_id,
                    host="blockchain",  # Special host for blockchain services
                    port=0,
                    protocol="blockchain",
                    capabilities=["blockchain_messaging", "consensus"],
                    status=ServiceStatus.ACTIVE,
                    last_seen=datetime.utcnow(),
                    metadata={"discovery_method": "blockchain"},
                    health_score=0.9  # Slightly lower than direct TCP
                )
                
                if self._matches_capability(service, capability):
                    services.append(service)
        
        except Exception as e:
            self.logger.error(f"Blockchain service discovery failed: {e}")
        
        return services
    
    async def _discover_via_multicast(self, capability: Optional[str]) -> List[ServiceEndpoint]:
        """Discover services via UDP multicast"""
        services = []
        
        try:
            # Send multicast discovery message
            discovery_message = {
                "type": "service_discovery",
                "agent_id": self.agent_id,
                "capability": capability,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Use UDP multicast on local network
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(5.0)  # 5 second timeout
            
            # Send discovery broadcast
            message = json.dumps(discovery_message).encode('utf-8')
            sock.sendto(message, ('255.255.255.255', 31337))  # Broadcast on port 31337
            
            # Listen for responses
            try:
                while True:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode('utf-8'))
                    
                    if response.get("type") == "service_response":
                        service = ServiceEndpoint(
                            service_id=response["service_id"],
                            agent_id=response["agent_id"],
                            host=addr[0],
                            port=response.get("port", 8080),
                            protocol=response.get("protocol", "tcp"),
                            capabilities=response.get("capabilities", []),
                            status=ServiceStatus.ACTIVE,
                            last_seen=datetime.utcnow(),
                            metadata={"discovery_method": "multicast"},
                            health_score=0.8
                        )
                        
                        if self._matches_capability(service, capability):
                            services.append(service)
            
            except socket.timeout:
                pass  # Normal timeout
            
            sock.close()
        
        except Exception as e:
            self.logger.debug(f"Multicast discovery error: {e}")
        
        return services
    
    async def _periodic_discovery(self):
        """Periodically discover new services"""
        while self._running:
            try:
                await self.discover_services()
                await asyncio.sleep(self.discovery_interval)
            except Exception as e:
                self.logger.error(f"Periodic discovery error: {e}")
                await asyncio.sleep(30)
    
    async def _health_checker(self):
        """Periodically check service health"""
        while self._running:
            try:
                for service_id, service in list(self.service_cache.items()):
                    if service.protocol == "blockchain":
                        continue  # Skip blockchain services
                    
                    # Simple TCP health check
                    health_score = await self._check_service_health(service)
                    service.health_score = health_score
                    
                    if health_score < 0.1:
                        service.status = ServiceStatus.INACTIVE
                    elif health_score < 0.5:
                        service.status = ServiceStatus.DEGRADED
                    else:
                        service.status = ServiceStatus.ACTIVE
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)
    
    async def _check_service_health(self, service: ServiceEndpoint) -> float:
        """Check health of a specific service"""
        try:
            if service.protocol == "tcp":
                # TCP connection test
                future = asyncio.open_connection(service.host, service.port)
                reader, writer = await asyncio.wait_for(future, timeout=5.0)
                writer.close()
                await writer.wait_closed()
                return 1.0
            
            elif service.protocol == "http":
                # HTTP health check
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    url = f"http://{service.host}:{service.port}/health"
                    async with session.get(url, timeout=5) as response:
                        return 1.0 if response.status == 200 else 0.5
            
            else:
                # Unknown protocol, assume healthy
                return 0.7
        
        except Exception:
            return 0.0
    
    async def _cache_cleaner(self):
        """Clean up expired services from cache"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                expired_services = []
                
                for service_id, service in self.service_cache.items():
                    age = (current_time - service.last_seen).total_seconds()
                    if age > self.cache_ttl * 2:  # Double TTL for cleanup
                        expired_services.append(service_id)
                
                for service_id in expired_services:
                    del self.service_cache[service_id]
                    self.logger.debug(f"Removed expired service: {service_id}")
                
                await asyncio.sleep(self.cache_ttl)  # Clean every TTL period
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service discovery status"""
        return {
            "running": self._running,
            "methods": self.discovery_methods,
            "local_services": len(self.local_services),
            "cached_services": len(self.service_cache),
            "active_services": len([s for s in self.service_cache.values() if s.status == ServiceStatus.ACTIVE])
        }