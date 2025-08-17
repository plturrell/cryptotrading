"""
Unified Service Discovery for Cryptotrading Platform
Supports local development and Vercel deployment for both A2A and MCP
"""
import asyncio
import json
import logging
import os
import time
import hashlib
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    LOCAL_DEVELOPMENT = "local_development"
    VERCEL_PRODUCTION = "vercel_production"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"

class ServiceType(Enum):
    """Types of services that can be discovered"""
    A2A_AGENT = "a2a_agent"          # Agent-to-agent communication
    MCP_SKILL = "mcp_skill"          # Intra-agent MCP skills
    DATA_SOURCE = "data_source"      # Data loading services
    ANALYTICS = "analytics"          # Analysis services
    STORAGE = "storage"              # Storage services
    GATEWAY = "gateway"              # API gateways

class DiscoveryMethod(Enum):
    """Service discovery methods"""
    BLOCKCHAIN = "blockchain"        # Anvil blockchain registry
    VERCEL_KV = "vercel_kv"         # Vercel KV store
    REDIS = "redis"                 # Redis registry
    DNS_SD = "dns_sd"               # DNS service discovery
    MULTICAST = "multicast"         # UDP multicast
    STATIC_CONFIG = "static_config" # Static configuration files
    HEALTH_PROBE = "health_probe"   # Active health probing

@dataclass
class ServiceCapability:
    """Service capability definition"""
    name: str
    version: str
    description: str
    input_types: List[str]
    output_types: List[str]
    requirements: Dict[str, Any]
    metadata: Dict[str, Any] = None

@dataclass 
class ServiceEndpoint:
    """Enhanced service endpoint with full metadata"""
    service_id: str
    service_type: ServiceType
    agent_id: str
    name: str
    description: str
    
    # Network information
    host: str
    port: int
    protocol: str  # http, https, tcp, websocket, blockchain
    base_path: str = "/"
    
    # Service metadata
    capabilities: List[ServiceCapability] = None
    tags: List[str] = None
    version: str = "1.0.0"
    
    # Health and status
    status: str = "active"
    health_score: float = 1.0
    last_seen: datetime = None
    response_time_ms: float = 0.0
    
    # Discovery metadata
    discovery_method: DiscoveryMethod = None
    ttl_seconds: int = 300
    environment: DeploymentEnvironment = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()

class UnifiedServiceDiscovery:
    """
    Production-grade unified service discovery
    Automatically adapts to deployment environment
    """
    
    def __init__(self, agent_id: str, service_type: ServiceType = ServiceType.A2A_AGENT):
        self.agent_id = agent_id
        self.service_type = service_type
        self.logger = logging.getLogger(f"ServiceDiscovery-{agent_id}")
        
        # Environment detection
        self.environment = self._detect_environment()
        self.logger.info(f"Detected environment: {self.environment.value}")
        
        # Discovery methods based on environment
        self.active_methods = self._configure_discovery_methods()
        self.logger.info(f"Active discovery methods: {[m.value for m in self.active_methods]}")
        
        # Service registry (local cache)
        self.services: Dict[str, ServiceEndpoint] = {}
        self.service_cache_file = Path("data/service_cache.json")
        
        # Discovery clients
        self.discovery_clients = {}
        
        # Configuration
        self.health_check_interval = 30
        self.cache_cleanup_interval = 300  # 5 minutes
        self.discovery_interval = 60
        
        # Background tasks
        self._tasks: Set[asyncio.Task] = set()
        self._running = False
        
    def _detect_environment(self) -> DeploymentEnvironment:
        """Detect deployment environment"""
        # Check for Vercel
        if os.getenv('VERCEL') or os.getenv('VERCEL_ENV'):
            return DeploymentEnvironment.VERCEL_PRODUCTION
        
        # Check for Kubernetes
        if os.path.exists('/var/run/secrets/kubernetes.io'):
            return DeploymentEnvironment.KUBERNETES
        
        # Check for Docker Compose
        if os.getenv('COMPOSE_PROJECT_NAME'):
            return DeploymentEnvironment.DOCKER_COMPOSE
        
        # Default to local development
        return DeploymentEnvironment.LOCAL_DEVELOPMENT
    
    def _configure_discovery_methods(self) -> List[DiscoveryMethod]:
        """Configure discovery methods based on environment"""
        methods = []
        
        if self.environment == DeploymentEnvironment.VERCEL_PRODUCTION:
            # Vercel: Use KV store, DNS, health probes
            methods.extend([
                DiscoveryMethod.VERCEL_KV,
                DiscoveryMethod.DNS_SD,
                DiscoveryMethod.HEALTH_PROBE,
                DiscoveryMethod.STATIC_CONFIG
            ])
        
        elif self.environment == DeploymentEnvironment.LOCAL_DEVELOPMENT:
            # Local: Use all methods for testing
            methods.extend([
                DiscoveryMethod.BLOCKCHAIN,
                DiscoveryMethod.REDIS,
                DiscoveryMethod.MULTICAST,
                DiscoveryMethod.DNS_SD,
                DiscoveryMethod.HEALTH_PROBE,
                DiscoveryMethod.STATIC_CONFIG
            ])
        
        elif self.environment == DeploymentEnvironment.KUBERNETES:
            # K8s: Use DNS service discovery and health probes
            methods.extend([
                DiscoveryMethod.DNS_SD,
                DiscoveryMethod.HEALTH_PROBE,
                DiscoveryMethod.STATIC_CONFIG
            ])
        
        else:
            # Docker Compose: Use DNS and multicast
            methods.extend([
                DiscoveryMethod.DNS_SD,
                DiscoveryMethod.MULTICAST,
                DiscoveryMethod.HEALTH_PROBE,
                DiscoveryMethod.STATIC_CONFIG
            ])
        
        return methods
    
    async def initialize(self) -> bool:
        """Initialize service discovery"""
        try:
            self.logger.info("Initializing unified service discovery")
            
            # Initialize discovery clients
            await self._initialize_discovery_clients()
            
            # Load cached services
            await self._load_service_cache()
            
            # Load static configuration
            await self._load_static_configuration()
            
            self.logger.info("Service discovery initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service discovery: {e}")
            return False
    
    async def start(self):
        """Start service discovery"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        tasks = [
            self._periodic_discovery(),
            self._health_monitor(),
            self._cache_manager()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        
        self.logger.info("Service discovery background tasks started")
    
    async def stop(self):
        """Stop service discovery"""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Save cache
        await self._save_service_cache()
        
        self.logger.info("Service discovery stopped")
    
    async def register_service(self, endpoint: ServiceEndpoint) -> bool:
        """Register a service with appropriate discovery methods"""
        try:
            endpoint.environment = self.environment
            endpoint.last_seen = datetime.utcnow()
            
            # Register locally
            self.services[endpoint.service_id] = endpoint
            
            # Register with active discovery methods
            registration_results = []
            
            for method in self.active_methods:
                try:
                    success = await self._register_with_method(endpoint, method)
                    registration_results.append(success)
                    if success:
                        self.logger.debug(f"Registered {endpoint.service_id} with {method.value}")
                except Exception as e:
                    self.logger.error(f"Registration failed for {method.value}: {e}")
                    registration_results.append(False)
            
            # Consider registration successful if at least one method worked
            overall_success = any(registration_results)
            
            if overall_success:
                self.logger.info(f"Registered service {endpoint.service_id} ({endpoint.service_type.value})")
                await self._save_service_cache()
            
            return overall_success
            
        except Exception as e:
            self.logger.error(f"Failed to register service {endpoint.service_id}: {e}")
            return False
    
    async def discover_services(self, 
                              service_type: Optional[ServiceType] = None,
                              capability: Optional[str] = None,
                              tags: Optional[List[str]] = None) -> List[ServiceEndpoint]:
        """Discover services with filtering"""
        all_services = {}
        
        # Discover from all active methods
        for method in self.active_methods:
            try:
                method_services = await self._discover_with_method(method, service_type, capability)
                for service in method_services:
                    # Use the most recent version of each service
                    if (service.service_id not in all_services or 
                        service.last_seen > all_services[service.service_id].last_seen):
                        all_services[service.service_id] = service
            except Exception as e:
                self.logger.error(f"Discovery failed for {method.value}: {e}")
        
        # Add MCP skills discovery if requested
        if service_type == ServiceType.MCP_SKILL or not service_type:
            try:
                mcp_services = await self._discover_mcp_skills(capability)
                for service in mcp_services:
                    if (service.service_id not in all_services or
                        service.last_seen > all_services[service.service_id].last_seen):
                        all_services[service.service_id] = service
            except Exception as e:
                self.logger.error(f"MCP skills discovery failed: {e}")
        
        # Add locally cached services
        for service in self.services.values():
            if self._is_service_valid(service):
                if (service.service_id not in all_services or
                    service.last_seen > all_services[service.service_id].last_seen):
                    all_services[service.service_id] = service
        
        # Apply filters
        filtered_services = []
        for service in all_services.values():
            if self._matches_filters(service, service_type, capability, tags):
                filtered_services.append(service)
        
        # Sort by health score and response time
        filtered_services.sort(key=lambda s: (s.health_score, -s.response_time_ms), reverse=True)
        
        self.logger.info(f"Discovered {len(filtered_services)} services (type={service_type}, capability={capability})")
        return filtered_services
    
    async def get_service(self, service_id: str) -> Optional[ServiceEndpoint]:
        """Get specific service by ID"""
        # Check local cache first
        if service_id in self.services and self._is_service_valid(self.services[service_id]):
            return self.services[service_id]
        
        # Try discovery
        services = await self.discover_services()
        for service in services:
            if service.service_id == service_id:
                return service
        
        return None
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister a service"""
        try:
            # Remove from local cache
            if service_id in self.services:
                endpoint = self.services[service_id]
                
                # Unregister from discovery methods
                for method in self.active_methods:
                    try:
                        await self._unregister_with_method(endpoint, method)
                    except Exception as e:
                        self.logger.error(f"Unregistration failed for {method.value}: {e}")
                
                del self.services[service_id]
                await self._save_service_cache()
                
                self.logger.info(f"Unregistered service {service_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unregister service {service_id}: {e}")
            return False
    
    # Discovery Method Implementations
    async def _register_with_method(self, endpoint: ServiceEndpoint, method: DiscoveryMethod) -> bool:
        """Register service with specific discovery method"""
        try:
            if method == DiscoveryMethod.VERCEL_KV:
                return await self._register_vercel_kv(endpoint)
            elif method == DiscoveryMethod.REDIS:
                return await self._register_redis(endpoint)
            elif method == DiscoveryMethod.BLOCKCHAIN:
                return await self._register_blockchain(endpoint)
            elif method == DiscoveryMethod.DNS_SD:
                return await self._register_dns_sd(endpoint)
            elif method == DiscoveryMethod.STATIC_CONFIG:
                return await self._register_static_config(endpoint)
            elif method == DiscoveryMethod.MULTICAST:
                return await self._register_multicast(endpoint)
            elif method == DiscoveryMethod.HEALTH_PROBE:
                # Health probe is discovery-only, no registration needed
                return True
            else:
                self.logger.warning(f"Registration method {method.value} not implemented")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error for {method.value}: {e}")
            return False
    
    async def _discover_with_method(self, method: DiscoveryMethod, 
                                   service_type: Optional[ServiceType] = None,
                                   capability: Optional[str] = None) -> List[ServiceEndpoint]:
        """Discover services with specific method"""
        try:
            if method == DiscoveryMethod.VERCEL_KV:
                return await self._discover_vercel_kv(service_type, capability)
            elif method == DiscoveryMethod.REDIS:
                return await self._discover_redis(service_type, capability)
            elif method == DiscoveryMethod.BLOCKCHAIN:
                return await self._discover_blockchain(service_type, capability)
            elif method == DiscoveryMethod.DNS_SD:
                return await self._discover_dns_sd(service_type, capability)
            elif method == DiscoveryMethod.MULTICAST:
                return await self._discover_multicast(service_type, capability)
            elif method == DiscoveryMethod.HEALTH_PROBE:
                return await self._discover_health_probe(service_type, capability)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Discovery error for {method.value}: {e}")
            return []
    
    # Vercel KV Implementation
    async def _register_vercel_kv(self, endpoint: ServiceEndpoint) -> bool:
        """Register service in Vercel KV store"""
        kv_url = os.getenv('KV_REST_API_URL')
        kv_token = os.getenv('KV_REST_API_TOKEN')
        
        if not kv_url or not kv_token:
            self.logger.warning("Vercel KV credentials not available")
            return False
        
        try:
            service_key = f"service:{endpoint.service_type.value}:{endpoint.service_id}"
            service_data = asdict(endpoint)
            
            # Convert datetime to string for JSON serialization
            service_data['last_seen'] = endpoint.last_seen.isoformat()
            
            headers = {
                'Authorization': f'Bearer {kv_token}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{kv_url}/set/{service_key}",
                    headers=headers,
                    json=service_data
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Vercel KV registration error: {e}")
            return False
    
    async def _discover_vercel_kv(self, service_type: Optional[ServiceType], 
                                 capability: Optional[str]) -> List[ServiceEndpoint]:
        """Discover services from Vercel KV store"""
        kv_url = os.getenv('KV_REST_API_URL')
        kv_token = os.getenv('KV_REST_API_TOKEN')
        
        if not kv_url or not kv_token:
            return []
        
        try:
            headers = {'Authorization': f'Bearer {kv_token}'}
            
            # Get all service keys
            pattern = f"service:{service_type.value if service_type else '*'}:*"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{kv_url}/scan/0",
                    headers=headers,
                    params={'match': pattern, 'count': 100}
                ) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    keys = data.get('result', [1, []])[1]  # Redis SCAN format
                    
                    services = []
                    for key in keys:
                        # Get service data
                        async with session.get(f"{kv_url}/get/{key}", headers=headers) as get_response:
                            if get_response.status == 200:
                                service_data = await get_response.json()
                                service_data['last_seen'] = datetime.fromisoformat(service_data['last_seen'])
                                
                                # Reconstruct ServiceEndpoint
                                service = ServiceEndpoint(**service_data)
                                if self._is_service_valid(service):
                                    services.append(service)
                    
                    return services
                    
        except Exception as e:
            self.logger.error(f"Vercel KV discovery error: {e}")
            return []
    
    # Redis Implementation  
    async def _register_redis(self, endpoint: ServiceEndpoint) -> bool:
        """Register service in Redis"""
        try:
            import aioredis
            
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            redis = aioredis.from_url(redis_url)
            
            service_key = f"service:{endpoint.service_type.value}:{endpoint.service_id}"
            service_data = asdict(endpoint)
            service_data['last_seen'] = endpoint.last_seen.isoformat()
            
            await redis.setex(service_key, endpoint.ttl_seconds, json.dumps(service_data))
            await redis.close()
            
            return True
            
        except ImportError:
            self.logger.debug("aioredis not available for Redis discovery")
            return False
        except Exception as e:
            self.logger.error(f"Redis registration error: {e}")
            return False
    
    async def _discover_redis(self, service_type: Optional[ServiceType], 
                            capability: Optional[str]) -> List[ServiceEndpoint]:
        """Discover services from Redis"""
        try:
            import aioredis
            
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            redis = aioredis.from_url(redis_url)
            
            pattern = f"service:{service_type.value if service_type else '*'}:*"
            keys = await redis.keys(pattern)
            
            services = []
            for key in keys:
                data = await redis.get(key)
                if data:
                    service_data = json.loads(data)
                    service_data['last_seen'] = datetime.fromisoformat(service_data['last_seen'])
                    
                    service = ServiceEndpoint(**service_data)
                    if self._is_service_valid(service):
                        services.append(service)
            
            await redis.close()
            return services
            
        except ImportError:
            return []
        except Exception as e:
            self.logger.error(f"Redis discovery error: {e}")
            return []
    
    # Health Probe Implementation
    async def _discover_health_probe(self, service_type: Optional[ServiceType], 
                                   capability: Optional[str]) -> List[ServiceEndpoint]:
        """Discover services via health probes"""
        discovered_services = []
        
        # Probe common ports and paths
        common_ports = [8080, 3000, 5000, 8000, 9000]
        health_paths = ['/health', '/ping', '/status', '/_health']
        
        for port in common_ports:
            for path in health_paths:
                try:
                    url = f"http://localhost:{port}{path}"
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status == 200:
                                # Try to extract service info from response
                                try:
                                    data = await response.json()
                                    service_id = data.get('service_id', f"discovered-{port}")
                                    
                                    service = ServiceEndpoint(
                                        service_id=service_id,
                                        service_type=service_type or ServiceType.A2A_AGENT,
                                        agent_id=data.get('agent_id', service_id),
                                        name=data.get('name', f"Service on port {port}"),
                                        description=data.get('description', 'Health probe discovered'),
                                        host='localhost',
                                        port=port,
                                        protocol='http',
                                        base_path=data.get('base_path', '/'),
                                        discovery_method=DiscoveryMethod.HEALTH_PROBE,
                                        response_time_ms=response.headers.get('x-response-time', 0)
                                    )
                                    
                                    discovered_services.append(service)
                                    
                                except json.JSONDecodeError:
                                    # Service doesn't return JSON, but it's alive
                                    service = ServiceEndpoint(
                                        service_id=f"http-service-{port}",
                                        service_type=ServiceType.A2A_AGENT,
                                        agent_id=f"agent-{port}",
                                        name=f"HTTP Service on port {port}",
                                        description="Discovered via health probe",
                                        host='localhost',
                                        port=port,
                                        protocol='http',
                                        discovery_method=DiscoveryMethod.HEALTH_PROBE
                                    )
                                    discovered_services.append(service)
                                    
                except Exception:
                    continue  # Port not responsive
        
        return discovered_services
    
    # Utility Methods
    async def _initialize_discovery_clients(self):
        """Initialize discovery method clients"""
        # Initialize blockchain client if needed
        if DiscoveryMethod.BLOCKCHAIN in self.active_methods:
            try:
                from ...blockchain.anvil_client import AnvilA2AClient
                self.blockchain_client = AnvilA2AClient()
                if not await self.blockchain_client.initialize():
                    self.logger.warning("Blockchain client initialization failed")
                    self.active_methods.remove(DiscoveryMethod.BLOCKCHAIN)
            except Exception as e:
                self.logger.error(f"Failed to initialize blockchain client: {e}")
                if DiscoveryMethod.BLOCKCHAIN in self.active_methods:
                    self.active_methods.remove(DiscoveryMethod.BLOCKCHAIN)
        
        # Initialize Redis client if needed
        if DiscoveryMethod.REDIS in self.active_methods:
            try:
                import aioredis
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
                self.redis_client = aioredis.from_url(redis_url)
                # Test connection
                await self.redis_client.ping()
            except Exception as e:
                self.logger.warning(f"Redis not available: {e}")
                if DiscoveryMethod.REDIS in self.active_methods:
                    self.active_methods.remove(DiscoveryMethod.REDIS)
                self.redis_client = None
    
    async def _load_service_cache(self):
        """Load services from local cache file"""
        try:
            if self.service_cache_file.exists():
                async with aiofiles.open(self.service_cache_file, 'r') as f:
                    data = json.loads(await f.read())
                    
                    for service_data in data.get('services', []):
                        service_data['last_seen'] = datetime.fromisoformat(service_data['last_seen'])
                        service = ServiceEndpoint(**service_data)
                        
                        if self._is_service_valid(service):
                            self.services[service.service_id] = service
                    
                    self.logger.info(f"Loaded {len(self.services)} services from cache")
                    
        except Exception as e:
            self.logger.error(f"Failed to load service cache: {e}")
    
    async def _save_service_cache(self):
        """Save services to local cache file"""
        try:
            self.service_cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'services': [],
                'last_updated': datetime.utcnow().isoformat()
            }
            
            for service in self.services.values():
                service_data = asdict(service)
                service_data['last_seen'] = service.last_seen.isoformat()
                cache_data['services'].append(service_data)
            
            async with aiofiles.open(self.service_cache_file, 'w') as f:
                await f.write(json.dumps(cache_data, indent=2))
                
        except Exception as e:
            self.logger.error(f"Failed to save service cache: {e}")
    
    async def _load_static_configuration(self):
        """Load static service configuration"""
        config_file = Path("config/services.json")
        
        try:
            if config_file.exists():
                async with aiofiles.open(config_file, 'r') as f:
                    config = json.loads(await f.read())
                    
                    for service_config in config.get('services', []):
                        service = ServiceEndpoint(**service_config)
                        service.discovery_method = DiscoveryMethod.STATIC_CONFIG
                        service.last_seen = datetime.utcnow()
                        
                        self.services[service.service_id] = service
                    
                    self.logger.info(f"Loaded {len(config.get('services', []))} static services")
                    
        except Exception as e:
            self.logger.debug(f"No static configuration found: {e}")
    
    def _is_service_valid(self, service: ServiceEndpoint) -> bool:
        """Check if service is still valid"""
        if service.status != "active":
            return False
        
        age = (datetime.utcnow() - service.last_seen).total_seconds()
        return age <= service.ttl_seconds
    
    def _matches_filters(self, service: ServiceEndpoint, 
                        service_type: Optional[ServiceType],
                        capability: Optional[str],
                        tags: Optional[List[str]]) -> bool:
        """Check if service matches filters"""
        if service_type and service.service_type != service_type:
            return False
        
        if capability:
            capability_names = [cap.name for cap in service.capabilities]
            if capability not in capability_names:
                return False
        
        if tags:
            if not all(tag in service.tags for tag in tags):
                return False
        
        return True
    
    # Background Tasks
    async def _periodic_discovery(self):
        """Periodically discover services"""
        while self._running:
            try:
                await self.discover_services()
                await asyncio.sleep(self.discovery_interval)
            except Exception as e:
                self.logger.error(f"Periodic discovery error: {e}")
                await asyncio.sleep(30)
    
    async def _health_monitor(self):
        """Monitor service health"""
        while self._running:
            try:
                for service in list(self.services.values()):
                    health_score = await self._check_service_health(service)
                    service.health_score = health_score
                    
                    if health_score < 0.1:
                        service.status = "inactive"
                    elif health_score < 0.5:
                        service.status = "degraded"
                    else:
                        service.status = "active"
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _cache_manager(self):
        """Manage service cache"""
        while self._running:
            try:
                # Clean expired services
                expired = []
                for service_id, service in self.services.items():
                    if not self._is_service_valid(service):
                        expired.append(service_id)
                
                for service_id in expired:
                    del self.services[service_id]
                
                if expired:
                    self.logger.info(f"Cleaned {len(expired)} expired services")
                    await self._save_service_cache()
                
                await asyncio.sleep(self.cache_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Cache manager error: {e}")
                await asyncio.sleep(60)
    
    async def _check_service_health(self, service: ServiceEndpoint) -> float:
        """Check health of a service"""
        try:
            if service.protocol in ['http', 'https']:
                url = f"{service.protocol}://{service.host}:{service.port}/health"
                
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response_time = (time.time() - start_time) * 1000
                        service.response_time_ms = response_time
                        
                        if response.status == 200:
                            return 1.0
                        elif response.status < 500:
                            return 0.7
                        else:
                            return 0.3
            
            elif service.protocol == 'tcp':
                # TCP health check
                future = asyncio.open_connection(service.host, service.port)
                reader, writer = await asyncio.wait_for(future, timeout=5.0)
                writer.close()
                await writer.wait_closed()
                return 1.0
            
            else:
                # Unknown protocol, assume healthy if recently seen
                age = (datetime.utcnow() - service.last_seen).total_seconds()
                return max(0.0, 1.0 - (age / service.ttl_seconds))
                
        except Exception:
            return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get service discovery status"""
        return {
            "agent_id": self.agent_id,
            "environment": self.environment.value,
            "service_type": self.service_type.value,
            "running": self._running,
            "active_methods": [m.value for m in self.active_methods],
            "total_services": len(self.services),
            "active_services": len([s for s in self.services.values() if s.status == "active"]),
            "service_types": {
                st.value: len([s for s in self.services.values() if s.service_type == st])
                for st in ServiceType
            }
        }

# Blockchain Discovery Implementation
    async def _register_blockchain(self, endpoint: ServiceEndpoint) -> bool:
        """Register with Anvil blockchain"""
        try:
            if not hasattr(self, 'blockchain_client') or not self.blockchain_client:
                from ...blockchain.anvil_client import AnvilA2AClient
                self.blockchain_client = AnvilA2AClient()
                await self.blockchain_client.initialize()
            
            success = await self.blockchain_client.register_agent(
                endpoint.agent_id,
                [cap.name for cap in endpoint.capabilities],
                f"{endpoint.host}:{endpoint.port}"
            )
            
            if success:
                self.logger.debug(f"Registered {endpoint.service_id} on blockchain")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Blockchain registration failed: {e}")
            return False
    
    async def _discover_blockchain(self, service_type, capability) -> List[ServiceEndpoint]:
        """Discover services from Anvil blockchain"""
        services = []
        
        try:
            if not hasattr(self, 'blockchain_client') or not self.blockchain_client:
                return services
            
            agent_list = await self.blockchain_client.get_agent_list()
            
            for agent_id in agent_list:
                if agent_id == self.agent_id:
                    continue
                
                # Create blockchain service endpoint
                service = ServiceEndpoint(
                    service_id=f"blockchain-{agent_id}",
                    service_type=service_type or ServiceType.A2A_AGENT,
                    agent_id=agent_id,
                    name=f"Blockchain Agent {agent_id}",
                    description="Agent accessible via blockchain",
                    host="blockchain",
                    port=0,
                    protocol="blockchain",
                    capabilities=[
                        ServiceCapability(
                            name="blockchain_messaging",
                            version="1.0",
                            description="Blockchain-based A2A messaging",
                            input_types=["a2a_message"],
                            output_types=["a2a_response"],
                            requirements={"blockchain": "anvil"}
                        )
                    ],
                    tags=["blockchain", "consensus", "distributed"],
                    discovery_method=DiscoveryMethod.BLOCKCHAIN,
                    health_score=0.9,
                    metadata={"blockchain_address": agent_id}
                )
                
                if self._matches_filters(service, service_type, capability, None):
                    services.append(service)
        
        except Exception as e:
            self.logger.error(f"Blockchain discovery failed: {e}")
        
        return services
    
    # DNS Service Discovery Implementation
    async def _register_dns_sd(self, endpoint: ServiceEndpoint) -> bool:
        """Register with DNS-SD (mDNS/Bonjour)"""
        try:
            if self.environment == DeploymentEnvironment.KUBERNETES:
                # K8s DNS registration
                return await self._register_k8s_dns(endpoint)
            elif self.environment == DeploymentEnvironment.DOCKER_COMPOSE:
                # Docker Compose DNS
                return await self._register_docker_dns(endpoint)
            else:
                # mDNS for local development
                return await self._register_mdns(endpoint)
                
        except Exception as e:
            self.logger.error(f"DNS-SD registration failed: {e}")
            return False
    
    async def _discover_dns_sd(self, service_type, capability) -> List[ServiceEndpoint]:
        """Discover services via DNS-SD"""
        services = []
        
        try:
            if self.environment == DeploymentEnvironment.KUBERNETES:
                services.extend(await self._discover_k8s_dns(service_type, capability))
            elif self.environment == DeploymentEnvironment.DOCKER_COMPOSE:
                services.extend(await self._discover_docker_dns(service_type, capability))
            else:
                services.extend(await self._discover_mdns(service_type, capability))
                
        except Exception as e:
            self.logger.error(f"DNS-SD discovery failed: {e}")
        
        return services
    
    # Multicast Discovery Implementation
    async def _discover_multicast(self, service_type, capability) -> List[ServiceEndpoint]:
        """Discover services via UDP multicast"""
        services = []
        
        try:
            import socket
            import asyncio
            
            # Send multicast discovery message
            discovery_message = {
                "type": "service_discovery",
                "agent_id": self.agent_id,
                "service_type": service_type.value if service_type else "any",
                "capability": capability,
                "timestamp": datetime.utcnow().isoformat(),
                "environment": self.environment.value
            }
            
            # Create UDP socket for multicast
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(3.0)  # 3 second timeout
            
            # Send discovery broadcast
            message = json.dumps(discovery_message).encode('utf-8')
            multicast_port = 31337
            sock.sendto(message, ('255.255.255.255', multicast_port))
            
            # Listen for responses
            start_time = time.time()
            while time.time() - start_time < 3.0:
                try:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode('utf-8'))
                    
                    if response.get("type") == "service_response":
                        # Parse service capabilities
                        capabilities = []
                        for cap_data in response.get("capabilities", []):
                            if isinstance(cap_data, dict):
                                capabilities.append(ServiceCapability(**cap_data))
                            else:
                                capabilities.append(ServiceCapability(
                                    name=str(cap_data),
                                    version="1.0",
                                    description=f"Capability: {cap_data}",
                                    input_types=["any"],
                                    output_types=["any"],
                                    requirements={}
                                ))
                        
                        service = ServiceEndpoint(
                            service_id=response.get("service_id", f"multicast-{addr[0]}:{response.get('port', 8080)}"),
                            service_type=ServiceType(response.get("service_type", "a2a_agent")),
                            agent_id=response.get("agent_id", f"agent-{addr[0]}"),
                            name=response.get("name", f"Multicast Service {addr[0]}"),
                            description=response.get("description", "Discovered via multicast"),
                            host=addr[0],
                            port=response.get("port", 8080),
                            protocol=response.get("protocol", "tcp"),
                            capabilities=capabilities,
                            tags=response.get("tags", ["multicast"]),
                            discovery_method=DiscoveryMethod.MULTICAST,
                            health_score=0.8,
                            metadata=response.get("metadata", {})
                        )
                        
                        if self._matches_filters(service, service_type, capability, None):
                            services.append(service)
                
                except socket.timeout:
                    break
                except Exception as e:
                    self.logger.debug(f"Multicast response error: {e}")
                    continue
            
            sock.close()
            
        except Exception as e:
            self.logger.debug(f"Multicast discovery error: {e}")
        
        return services
    
    # Static Configuration Implementation
    async def _register_static_config(self, endpoint: ServiceEndpoint) -> bool:
        """Register in static configuration file"""
        try:
            config_file = Path("config/services.json")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing config
            config = {"services": []}
            if config_file.exists():
                async with aiofiles.open(config_file, 'r') as f:
                    config = json.loads(await f.read())
            
            # Add/update service
            service_data = asdict(endpoint)
            service_data['last_seen'] = endpoint.last_seen.isoformat()
            
            # Remove existing entry for this service
            config['services'] = [s for s in config['services'] if s.get('service_id') != endpoint.service_id]
            config['services'].append(service_data)
            
            # Save config
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(json.dumps(config, indent=2))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Static config registration failed: {e}")
            return False
    
    # Unregister implementations
    async def _unregister_with_method(self, endpoint: ServiceEndpoint, method: DiscoveryMethod):
        """Unregister from specific discovery method"""
        try:
            if method == DiscoveryMethod.VERCEL_KV:
                await self._unregister_vercel_kv(endpoint)
            elif method == DiscoveryMethod.REDIS:
                await self._unregister_redis(endpoint)
            elif method == DiscoveryMethod.BLOCKCHAIN:
                await self._unregister_blockchain(endpoint)
            elif method == DiscoveryMethod.STATIC_CONFIG:
                await self._unregister_static_config(endpoint)
                
        except Exception as e:
            self.logger.error(f"Unregistration failed for {method.value}: {e}")
    
    async def _unregister_vercel_kv(self, endpoint: ServiceEndpoint):
        """Unregister from Vercel KV"""
        kv_url = os.getenv('KV_REST_API_URL')
        kv_token = os.getenv('KV_REST_API_TOKEN')
        
        if kv_url and kv_token:
            service_key = f"service:{endpoint.service_type.value}:{endpoint.service_id}"
            headers = {'Authorization': f'Bearer {kv_token}'}
            
            async with aiohttp.ClientSession() as session:
                await session.delete(f"{kv_url}/del/{service_key}", headers=headers)
    
    async def _unregister_redis(self, endpoint: ServiceEndpoint):
        """Unregister from Redis"""
        try:
            import aioredis
            redis = aioredis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            service_key = f"service:{endpoint.service_type.value}:{endpoint.service_id}"
            await redis.delete(service_key)
            await redis.close()
        except ImportError:
            pass
    
    async def _unregister_blockchain(self, endpoint: ServiceEndpoint):
        """Unregister from blockchain"""
        if hasattr(self, 'blockchain_client') and self.blockchain_client:
            try:
                # Mark agent as inactive on blockchain
                deactivation_data = {
                    'agent_id': endpoint.agent_id,
                    'active': False,
                    'deactivated_at': int(datetime.utcnow().timestamp())
                }
                
                tx_data = json.dumps(deactivation_data).encode('utf-8').hex()
                
                tx = {
                    'from': self.blockchain_client.address,
                    'to': self.blockchain_client.registry_contract_address,
                    'data': '0x' + tx_data,
                    'gas': 50000,
                    'gasPrice': self.blockchain_client.w3.to_wei(20, 'gwei'),
                    'nonce': self.blockchain_client.w3.eth.get_transaction_count(self.blockchain_client.address)
                }
                
                signed_tx = self.blockchain_client.account.sign_transaction(tx)
                tx_hash = self.blockchain_client.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = self.blockchain_client.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                self.logger.info(f"Unregistered {endpoint.agent_id} from blockchain, tx: {tx_hash.hex()}")
                
            except Exception as e:
                self.logger.error(f"Blockchain unregistration failed: {e}")
    
    async def _unregister_static_config(self, endpoint: ServiceEndpoint):
        """Remove from static configuration"""
        try:
            config_file = Path("config/services.json")
            if config_file.exists():
                async with aiofiles.open(config_file, 'r') as f:
                    config = json.loads(await f.read())
                
                config['services'] = [s for s in config['services'] if s.get('service_id') != endpoint.service_id]
                
                async with aiofiles.open(config_file, 'w') as f:
                    await f.write(json.dumps(config, indent=2))
        except Exception as e:
            self.logger.error(f"Static config unregistration failed: {e}")
    
    # DNS-SD Helper Methods
    async def _register_k8s_dns(self, endpoint: ServiceEndpoint) -> bool:
        """Register with Kubernetes DNS"""
        try:
            # Check if we're actually in Kubernetes
            if not os.path.exists('/var/run/secrets/kubernetes.io'):
                self.logger.error("Not running in Kubernetes environment")
                return False
            
            # Would need kubernetes client library
            try:
                from kubernetes import client, config
                config.load_incluster_config()
                v1 = client.CoreV1Api()
                
                # Create headless service for the agent
                service_manifest = {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": f"agent-{endpoint.service_id}",
                        "labels": {"app": "cryptotrading", "agent": endpoint.agent_id}
                    },
                    "spec": {
                        "clusterIP": "None",
                        "ports": [{"port": endpoint.port, "protocol": "TCP"}],
                        "selector": {"agent": endpoint.agent_id}
                    }
                }
                
                v1.create_namespaced_service(
                    namespace=os.getenv('K8S_NAMESPACE', 'default'),
                    body=service_manifest
                )
                return True
                
            except ImportError:
                self.logger.error("kubernetes library not installed")
                return False
                
        except Exception as e:
            self.logger.error(f"K8s DNS registration failed: {e}")
            return False
    
    async def _discover_k8s_dns(self, service_type, capability) -> List[ServiceEndpoint]:
        """Discover services via Kubernetes DNS"""
        services = []
        
        try:
            if not os.path.exists('/var/run/secrets/kubernetes.io'):
                return services
            
            from kubernetes import client, config
            config.load_incluster_config()
            v1 = client.CoreV1Api()
            
            # List services in namespace
            namespace = os.getenv('K8S_NAMESPACE', 'default')
            service_list = v1.list_namespaced_service(namespace)
            
            for svc in service_list.items:
                if 'cryptotrading' in svc.metadata.labels:
                    # Parse service into ServiceEndpoint
                    service = ServiceEndpoint(
                        service_id=svc.metadata.name,
                        service_type=service_type or ServiceType.A2A_AGENT,
                        agent_id=svc.metadata.labels.get('agent', 'unknown'),
                        name=svc.metadata.name,
                        description=f"K8s service {svc.metadata.name}",
                        host=f"{svc.metadata.name}.{namespace}.svc.cluster.local",
                        port=svc.spec.ports[0].port if svc.spec.ports else 8080,
                        protocol="tcp",
                        discovery_method=DiscoveryMethod.DNS_SD,
                        metadata={"k8s_labels": svc.metadata.labels}
                    )
                    
                    if self._matches_filters(service, service_type, capability, None):
                        services.append(service)
                        
        except ImportError:
            self.logger.debug("kubernetes library not available")
        except Exception as e:
            self.logger.error(f"K8s discovery failed: {e}")
            
        return services
    
    async def _register_docker_dns(self, endpoint: ServiceEndpoint) -> bool:
        """Register with Docker Compose DNS"""
        try:
            # Docker Compose DNS is automatic - we just need to verify we're in Docker
            if not os.getenv('COMPOSE_PROJECT_NAME'):
                self.logger.error("Not running in Docker Compose environment")
                return False
            
            # Write to shared volume for other containers to discover
            compose_services_dir = Path("/var/cryptotrading/services")
            if compose_services_dir.exists():
                service_file = compose_services_dir / f"{endpoint.service_id}.json"
                service_data = asdict(endpoint)
                service_data['last_seen'] = endpoint.last_seen.isoformat()
                
                async with aiofiles.open(service_file, 'w') as f:
                    await f.write(json.dumps(service_data, indent=2))
                
                self.logger.info(f"Registered {endpoint.service_id} in Docker Compose shared volume")
                return True
            else:
                self.logger.warning("Docker Compose shared volume not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Docker DNS registration failed: {e}")
            return False
    
    async def _discover_docker_dns(self, service_type, capability) -> List[ServiceEndpoint]:
        """Discover services via Docker Compose DNS"""
        services = []
        
        try:
            # Check shared volume for registered services
            compose_services_dir = Path("/var/cryptotrading/services")
            if compose_services_dir.exists():
                for service_file in compose_services_dir.glob("*.json"):
                    try:
                        async with aiofiles.open(service_file, 'r') as f:
                            service_data = json.loads(await f.read())
                            service_data['last_seen'] = datetime.fromisoformat(service_data['last_seen'])
                            
                            service = ServiceEndpoint(**service_data)
                            if self._matches_filters(service, service_type, capability, None):
                                services.append(service)
                    except Exception as e:
                        self.logger.debug(f"Failed to load service from {service_file}: {e}")
            
            # Also try Docker API if available
            try:
                import docker
                client = docker.from_env()
                
                # Get current compose project
                project_name = os.getenv('COMPOSE_PROJECT_NAME')
                if project_name:
                    containers = client.containers.list(
                        filters={"label": f"com.docker.compose.project={project_name}"}
                    )
                    
                    for container in containers:
                        if 'cryptotrading' in container.labels:
                            # Extract service info from container
                            service_name = container.labels.get('com.docker.compose.service')
                            if service_name:
                                service = ServiceEndpoint(
                                    service_id=f"docker-{service_name}",
                                    service_type=service_type or ServiceType.A2A_AGENT,
                                    agent_id=container.labels.get('agent_id', service_name),
                                    name=service_name,
                                    description=f"Docker Compose service {service_name}",
                                    host=service_name,  # Docker Compose DNS name
                                    port=int(container.labels.get('service_port', '8080')),
                                    protocol="tcp",
                                    discovery_method=DiscoveryMethod.DNS_SD,
                                    metadata={"container_id": container.id[:12]}
                                )
                                
                                if self._matches_filters(service, service_type, capability, None):
                                    services.append(service)
                                    
            except ImportError:
                self.logger.debug("docker library not available")
                
        except Exception as e:
            self.logger.error(f"Docker DNS discovery failed: {e}")
            
        return services
    
    async def _register_mdns(self, endpoint: ServiceEndpoint) -> bool:
        """Register with mDNS (Bonjour)"""
        try:
            try:
                from zeroconf import ServiceInfo, Zeroconf
                import socket
                
                # Create mDNS service info
                service_type = "_cryptotrading._tcp.local."
                service_name = f"{endpoint.service_id}.{service_type}"
                
                # Get local IP
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                
                info = ServiceInfo(
                    service_type,
                    service_name,
                    addresses=[socket.inet_aton(local_ip)],
                    port=endpoint.port,
                    properties={
                        "agent_id": endpoint.agent_id,
                        "service_type": endpoint.service_type.value,
                        "capabilities": json.dumps([cap.name for cap in endpoint.capabilities]),
                        "version": "1.0"
                    }
                )
                
                # Register service
                zeroconf = Zeroconf()
                zeroconf.register_service(info)
                
                # Store zeroconf instance for cleanup
                if not hasattr(self, '_mdns_instances'):
                    self._mdns_instances = {}
                self._mdns_instances[endpoint.service_id] = (zeroconf, info)
                
                self.logger.info(f"Registered {endpoint.service_id} via mDNS")
                return True
                
            except ImportError:
                # Fallback to basic file-based discovery for local dev
                local_services_dir = Path(".local/services")
                local_services_dir.mkdir(parents=True, exist_ok=True)
                
                service_file = local_services_dir / f"{endpoint.service_id}.json"
                service_data = asdict(endpoint)
                service_data['last_seen'] = endpoint.last_seen.isoformat()
                
                async with aiofiles.open(service_file, 'w') as f:
                    await f.write(json.dumps(service_data, indent=2))
                
                self.logger.info(f"Registered {endpoint.service_id} in local file registry")
                return True
                
        except Exception as e:
            self.logger.error(f"mDNS registration failed: {e}")
            return False
    
    async def _discover_mdns(self, service_type, capability) -> List[ServiceEndpoint]:
        """Discover services via mDNS"""
        services = []
        
        try:
            try:
                from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
                import threading
                
                discovered = []
                discovery_complete = threading.Event()
                
                class Listener(ServiceListener):
                    def add_service(self, zeroconf, service_type, name):
                        info = zeroconf.get_service_info(service_type, name)
                        if info:
                            properties = info.properties
                            discovered.append({
                                "name": name,
                                "host": socket.inet_ntoa(info.addresses[0]),
                                "port": info.port,
                                "properties": {
                                    k.decode('utf-8'): v.decode('utf-8') 
                                    for k, v in properties.items()
                                }
                            })
                    
                    def remove_service(self, zeroconf, service_type, name):
                        pass
                    
                    def update_service(self, zeroconf, service_type, name):
                        pass
                
                zeroconf = Zeroconf()
                listener = Listener()
                browser = ServiceBrowser(zeroconf, "_cryptotrading._tcp.local.", listener)
                
                # Wait for discovery (with timeout)
                await asyncio.sleep(2)  # Give 2 seconds for discovery
                
                zeroconf.close()
                
                # Process discovered services
                for disc in discovered:
                    props = disc['properties']
                    
                    # Parse capabilities
                    capabilities = []
                    if 'capabilities' in props:
                        cap_names = json.loads(props['capabilities'])
                        for cap_name in cap_names:
                            capabilities.append(ServiceCapability(
                                name=cap_name,
                                version="1.0",
                                description=f"mDNS discovered capability: {cap_name}",
                                input_types=["any"],
                                output_types=["any"],
                                requirements={}
                            ))
                    
                    service = ServiceEndpoint(
                        service_id=disc['name'].split('.')[0],
                        service_type=ServiceType(props.get('service_type', 'a2a_agent')),
                        agent_id=props.get('agent_id', 'unknown'),
                        name=disc['name'],
                        description="mDNS discovered service",
                        host=disc['host'],
                        port=disc['port'],
                        protocol="tcp",
                        capabilities=capabilities,
                        discovery_method=DiscoveryMethod.DNS_SD,
                        metadata={"mdns_properties": props}
                    )
                    
                    if self._matches_filters(service, service_type, capability, None):
                        services.append(service)
                        
            except ImportError:
                # Fallback to file-based discovery
                local_services_dir = Path(".local/services")
                if local_services_dir.exists():
                    for service_file in local_services_dir.glob("*.json"):
                        try:
                            async with aiofiles.open(service_file, 'r') as f:
                                service_data = json.loads(await f.read())
                                service_data['last_seen'] = datetime.fromisoformat(service_data['last_seen'])
                                
                                service = ServiceEndpoint(**service_data)
                                # Only include if still valid (not stale)
                                if self._is_service_valid(service):
                                    if self._matches_filters(service, service_type, capability, None):
                                        services.append(service)
                        except Exception as e:
                            self.logger.debug(f"Failed to load service from {service_file}: {e}")
                            
        except Exception as e:
            self.logger.error(f"mDNS discovery failed: {e}")
        
        return services
    
    # MCP Skills Discovery Implementation
    async def _discover_mcp_skills(self, capability: Optional[str] = None) -> List[ServiceEndpoint]:
        """Discover MCP intra-agent skills"""
        services = []
        
        try:
            # Load MCP skills from various sources
            
            # 1. Discover from MCP servers (localhost and configured endpoints)
            mcp_endpoints = await self._discover_mcp_servers()
            
            for endpoint_info in mcp_endpoints:
                try:
                    # Query MCP server for available tools/skills
                    skills = await self._query_mcp_server_skills(endpoint_info, capability)
                    services.extend(skills)
                except Exception as e:
                    self.logger.debug(f"Failed to query MCP server {endpoint_info}: {e}")
            
            # 2. Discover from static MCP configuration
            static_mcp_services = await self._load_static_mcp_skills(capability)
            services.extend(static_mcp_services)
            
            # 3. Discover from Vercel/cloud MCP registries
            if self.environment == DeploymentEnvironment.VERCEL_PRODUCTION:
                cloud_mcp_services = await self._discover_cloud_mcp_skills(capability)
                services.extend(cloud_mcp_services)
            
        except Exception as e:
            self.logger.error(f"MCP skills discovery failed: {e}")
        
        return services
    
    async def _discover_mcp_servers(self) -> List[Dict[str, Any]]:
        """Discover running MCP servers"""
        servers = []
        
        # Common MCP server ports and endpoints
        mcp_ports = [3000, 8000, 8080, 9000]
        mcp_paths = ['/mcp', '/api/mcp', '/skills', '/tools']
        
        for port in mcp_ports:
            for path in mcp_paths:
                try:
                    url = f"http://localhost:{port}{path}"
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{url}/info", timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status == 200:
                                server_info = await response.json()
                                servers.append({
                                    "url": url,
                                    "port": port,
                                    "path": path,
                                    "info": server_info
                                })
                                
                except Exception:
                    continue  # Server not responsive
        
        return servers
    
    async def _query_mcp_server_skills(self, endpoint_info: Dict[str, Any], capability: Optional[str]) -> List[ServiceEndpoint]:
        """Query MCP server for available skills/tools"""
        services = []
        
        try:
            base_url = endpoint_info["url"]
            
            async with aiohttp.ClientSession() as session:
                # Get tools list
                async with session.get(f"{base_url}/tools", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        tools_data = await response.json()
                        
                        for tool in tools_data.get("tools", []):
                            tool_name = tool.get("name", "unknown")
                            
                            # Filter by capability if specified
                            if capability and capability.lower() not in tool_name.lower():
                                continue
                            
                            # Create service capability
                            service_capability = ServiceCapability(
                                name=tool_name,
                                version=tool.get("version", "1.0"),
                                description=tool.get("description", f"MCP tool: {tool_name}"),
                                input_types=tool.get("inputSchema", {}).get("properties", {}).keys(),
                                output_types=["mcp_response"],
                                requirements={"mcp_server": base_url},
                                metadata=tool.get("metadata", {})
                            )
                            
                            # Create service endpoint
                            service = ServiceEndpoint(
                                service_id=f"mcp-{tool_name}-{endpoint_info['port']}",
                                service_type=ServiceType.MCP_SKILL,
                                agent_id=self.agent_id,
                                name=f"MCP {tool_name}",
                                description=tool.get("description", f"MCP skill: {tool_name}"),
                                host="localhost",
                                port=endpoint_info["port"],
                                protocol="http",
                                base_path=endpoint_info["path"],
                                capabilities=[service_capability],
                                tags=["mcp", "skill", "tool"] + tool.get("tags", []),
                                health_score=1.0,
                                metadata={
                                    "mcp_server": base_url,
                                    "tool_schema": tool.get("inputSchema", {}),
                                    "mcp_info": endpoint_info.get("info", {})
                                }
                            )
                            
                            services.append(service)
        
        except Exception as e:
            self.logger.debug(f"Failed to query MCP server {endpoint_info}: {e}")
        
        return services
    
    async def _load_static_mcp_skills(self, capability: Optional[str]) -> List[ServiceEndpoint]:
        """Load MCP skills from static configuration"""
        services = []
        
        try:
            mcp_config_file = Path("config/mcp_skills.json")
            
            if mcp_config_file.exists():
                async with aiofiles.open(mcp_config_file, 'r') as f:
                    config = json.loads(await f.read())
                    
                    for skill_config in config.get("skills", []):
                        skill_name = skill_config.get("name", "unknown")
                        
                        # Filter by capability
                        if capability and capability.lower() not in skill_name.lower():
                            continue
                        
                        # Parse capabilities
                        capabilities = []
                        for cap_data in skill_config.get("capabilities", []):
                            capabilities.append(ServiceCapability(**cap_data))
                        
                        # Create service endpoint
                        service = ServiceEndpoint(
                            service_id=skill_config.get("service_id", f"static-mcp-{skill_name}"),
                            service_type=ServiceType.MCP_SKILL,
                            agent_id=skill_config.get("agent_id", self.agent_id),
                            name=skill_config.get("name", f"Static MCP {skill_name}"),
                            description=skill_config.get("description", f"Static MCP skill: {skill_name}"),
                            host=skill_config.get("host", "localhost"),
                            port=skill_config.get("port", 8080),
                            protocol=skill_config.get("protocol", "http"),
                            capabilities=capabilities,
                            tags=skill_config.get("tags", ["mcp", "static"]),
                            metadata=skill_config.get("metadata", {})
                        )
                        
                        services.append(service)
        
        except Exception as e:
            self.logger.debug(f"Failed to load static MCP skills: {e}")
        
        return services
    
    async def _discover_cloud_mcp_skills(self, capability: Optional[str]) -> List[ServiceEndpoint]:
        """Discover MCP skills from cloud registries (Vercel/AWS)"""
        services = []
        
        try:
            # Query Vercel KV for MCP skills
            kv_url = os.getenv('KV_REST_API_URL')
            kv_token = os.getenv('KV_REST_API_TOKEN')
            
            if kv_url and kv_token:
                headers = {'Authorization': f'Bearer {kv_token}'}
                
                async with aiohttp.ClientSession() as session:
                    # Scan for MCP skill keys
                    async with session.get(
                        f"{kv_url}/scan/0",
                        headers=headers,
                        params={'match': 'mcp_skill:*', 'count': 100}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            keys = data.get('result', [1, []])[1]
                            
                            for key in keys:
                                try:
                                    # Get skill data
                                    async with session.get(f"{kv_url}/get/{key}", headers=headers) as get_response:
                                        if get_response.status == 200:
                                            skill_data = await get_response.json()
                                            
                                            # Filter by capability
                                            skill_name = skill_data.get("name", "unknown")
                                            if capability and capability.lower() not in skill_name.lower():
                                                continue
                                            
                                            # Parse capabilities
                                            capabilities = []
                                            for cap_data in skill_data.get("capabilities", []):
                                                capabilities.append(ServiceCapability(**cap_data))
                                            
                                            # Create service endpoint
                                            service = ServiceEndpoint(
                                                service_id=skill_data.get("service_id", f"cloud-mcp-{skill_name}"),
                                                service_type=ServiceType.MCP_SKILL,
                                                agent_id=skill_data.get("agent_id", "cloud"),
                                                name=skill_data.get("name", f"Cloud MCP {skill_name}"),
                                                description=skill_data.get("description", f"Cloud MCP skill: {skill_name}"),
                                                host=skill_data.get("host", "cloud"),
                                                port=skill_data.get("port", 443),
                                                protocol=skill_data.get("protocol", "https"),
                                                capabilities=capabilities,
                                                tags=skill_data.get("tags", ["mcp", "cloud"]),
                                                metadata=skill_data.get("metadata", {}),
                                                last_seen=datetime.fromisoformat(skill_data.get("last_seen", datetime.utcnow().isoformat()))
                                            )
                                            
                                            services.append(service)
                                            
                                except Exception as e:
                                    self.logger.debug(f"Failed to process cloud MCP skill {key}: {e}")
        
        except Exception as e:
            self.logger.debug(f"Failed to discover cloud MCP skills: {e}")
        
        return services
    
    async def _register_multicast(self, endpoint: ServiceEndpoint) -> bool:
        """Register for multicast discovery responses"""
        try:
            # Multicast registration means we respond to discovery requests
            if not hasattr(self, '_multicast_responder'):
                asyncio.create_task(self._start_multicast_responder(endpoint))
            return True
        except Exception as e:
            self.logger.error(f"Multicast registration failed: {e}")
            return False
    
    async def _start_multicast_responder(self, endpoint: ServiceEndpoint):
        """Respond to multicast discovery requests"""
        import socket
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', 31337))  # Listen on discovery port
            sock.settimeout(1.0)
            
            self._multicast_responder = sock
            self.logger.info("Started multicast responder on port 31337")
            
            while hasattr(self, '_multicast_responder'):
                try:
                    data, addr = sock.recvfrom(1024)
                    request = json.loads(data.decode('utf-8'))
                    
                    if request.get("type") == "service_discovery":
                        # Send response with our service info
                        response = {
                            "type": "service_response",
                            "service_id": endpoint.service_id,
                            "service_type": endpoint.service_type.value,
                            "agent_id": endpoint.agent_id,
                            "name": endpoint.name,
                            "description": endpoint.description,
                            "port": endpoint.port,
                            "protocol": endpoint.protocol,
                            "capabilities": [
                                {
                                    "name": cap.name,
                                    "version": cap.version,
                                    "description": cap.description,
                                    "input_types": list(cap.input_types),
                                    "output_types": list(cap.output_types),
                                    "requirements": cap.requirements
                                } for cap in endpoint.capabilities
                            ],
                            "tags": endpoint.tags,
                            "metadata": endpoint.metadata
                        }
                        
                        sock.sendto(json.dumps(response).encode('utf-8'), addr)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.debug(f"Multicast responder error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Multicast responder failed: {e}")
        finally:
            if hasattr(self, '_multicast_responder'):
                self._multicast_responder.close()
                del self._multicast_responder