"""
Distributed Service Discovery and Coordination
Provides enterprise-grade service discovery, load balancing, and coordination
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict
import aioredis
import socket

logger = logging.getLogger(__name__)

class ServiceStatus(str, Enum):
    """Service status states"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    host: str
    port: int
    protocol: str = "http"
    path: str = "/"
    weight: int = 100
    
    def to_url(self) -> str:
        """Convert to URL string"""
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"

@dataclass 
class ServiceInstance:
    """Service instance registration"""
    service_id: str
    service_name: str
    instance_id: str
    endpoints: List[ServiceEndpoint]
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    status: ServiceStatus = ServiceStatus.STARTING
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    health_check_url: Optional[str] = None
    ttl_seconds: int = 30
    
    def is_healthy(self) -> bool:
        """Check if service is considered healthy"""
        if self.status != ServiceStatus.HEALTHY:
            return False
            
        # Check heartbeat TTL
        age = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return age < self.ttl_seconds * 2  # Allow some grace period

@dataclass
class LoadBalancingPolicy:
    """Load balancing configuration"""
    strategy: str = "round_robin"  # round_robin, weighted, least_connections, consistent_hash
    health_check_enabled: bool = True
    failover_enabled: bool = True
    circuit_breaker_enabled: bool = True
    max_retries: int = 3

class ServiceRegistry:
    """Distributed service registry using Redis as backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self._services: Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        self._watchers: Dict[str, List[Callable]] = defaultdict(list)
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
    async def initialize(self):
        """Initialize Redis connection and start background tasks"""
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis service registry")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory registry: {e}")
            self.redis = None
            
        self._running = True
        
        # Start background tasks
        asyncio.create_task(self._cleanup_expired_services())
        asyncio.create_task(self._watch_service_changes())
        
    async def close(self):
        """Close registry and cleanup resources"""
        self._running = False
        
        # Cancel health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()
            
        # Close Redis connection
        if self.redis:
            await self.redis.close()
            
    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance"""
        try:
            # Store in local cache
            self._services[service.service_name][service.instance_id] = service
            
            # Store in Redis if available
            if self.redis:
                key = f"services:{service.service_name}:{service.instance_id}"
                value = json.dumps(asdict(service), default=str)
                
                await self.redis.setex(key, service.ttl_seconds, value)
                
                # Add to service set
                await self.redis.sadd(f"service_names", service.service_name)
                await self.redis.sadd(f"service_instances:{service.service_name}", service.instance_id)
                
            # Start health checks if enabled
            if service.health_check_url:
                self._start_health_check(service)
                
            # Notify watchers
            await self._notify_watchers(service.service_name, "registered", service)
            
            logger.info(f"Registered service {service.service_name}:{service.instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.service_name}: {e}")
            return False
            
    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance"""
        try:
            # Remove from local cache
            if service_name in self._services:
                service = self._services[service_name].pop(instance_id, None)
                
                if not self._services[service_name]:
                    del self._services[service_name]
                    
            # Remove from Redis
            if self.redis:
                key = f"services:{service_name}:{instance_id}"
                await self.redis.delete(key)
                await self.redis.srem(f"service_instances:{service_name}", instance_id)
                
                # Clean up empty service
                instance_count = await self.redis.scard(f"service_instances:{service_name}")
                if instance_count == 0:
                    await self.redis.srem("service_names", service_name)
                    await self.redis.delete(f"service_instances:{service_name}")
                    
            # Stop health checks
            task_key = f"{service_name}:{instance_id}"
            if task_key in self._health_check_tasks:
                self._health_check_tasks[task_key].cancel()
                del self._health_check_tasks[task_key]
                
            # Notify watchers
            if service:
                await self._notify_watchers(service_name, "deregistered", service)
                
            logger.info(f"Deregistered service {service_name}:{instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_name}: {e}")
            return False
            
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover all healthy instances of a service"""
        instances = []
        
        try:
            # Try Redis first
            if self.redis:
                instance_ids = await self.redis.smembers(f"service_instances:{service_name}")
                
                for instance_id in instance_ids:
                    key = f"services:{service_name}:{instance_id.decode()}"
                    data = await self.redis.get(key)
                    
                    if data:
                        service_data = json.loads(data)
                        # Convert back to ServiceInstance
                        service = self._dict_to_service(service_data)
                        if service and service.is_healthy():
                            instances.append(service)
                            
            # Fallback to local cache
            else:
                for instance in self._services.get(service_name, {}).values():
                    if instance.is_healthy():
                        instances.append(instance)
                        
        except Exception as e:
            logger.error(f"Failed to discover services for {service_name}: {e}")
            
        return instances
        
    async def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Send heartbeat for service instance"""
        try:
            # Update local cache
            if service_name in self._services and instance_id in self._services[service_name]:
                self._services[service_name][instance_id].last_heartbeat = datetime.utcnow()
                
            # Update Redis
            if self.redis:
                key = f"services:{service_name}:{instance_id}"
                
                # Get current data
                data = await self.redis.get(key)
                if data:
                    service_data = json.loads(data)
                    service_data["last_heartbeat"] = datetime.utcnow().isoformat()
                    
                    # Update with new TTL
                    service = self._dict_to_service(service_data)
                    if service:
                        value = json.dumps(asdict(service), default=str)
                        await self.redis.setex(key, service.ttl_seconds, value)
                        
            return True
            
        except Exception as e:
            logger.error(f"Heartbeat failed for {service_name}:{instance_id}: {e}")
            return False
            
    async def update_service_status(self, service_name: str, instance_id: str, 
                                  status: ServiceStatus) -> bool:
        """Update service status"""
        try:
            # Update local cache
            if service_name in self._services and instance_id in self._services[service_name]:
                self._services[service_name][instance_id].status = status
                
            # Update Redis
            if self.redis:
                key = f"services:{service_name}:{instance_id}"
                data = await self.redis.get(key)
                
                if data:
                    service_data = json.loads(data)
                    service_data["status"] = status.value
                    
                    service = self._dict_to_service(service_data)
                    if service:
                        value = json.dumps(asdict(service), default=str)
                        await self.redis.setex(key, service.ttl_seconds, value)
                        
                        # Notify watchers
                        await self._notify_watchers(service_name, "status_changed", service)
                        
            return True
            
        except Exception as e:
            logger.error(f"Failed to update status for {service_name}:{instance_id}: {e}")
            return False
            
    async def watch_service(self, service_name: str, callback: Callable):
        """Watch for changes to a service"""
        self._watchers[service_name].append(callback)
        
    def _dict_to_service(self, data: Dict[str, Any]) -> Optional[ServiceInstance]:
        """Convert dictionary to ServiceInstance"""
        try:
            # Handle datetime fields
            if isinstance(data.get("registered_at"), str):
                data["registered_at"] = datetime.fromisoformat(data["registered_at"])
            if isinstance(data.get("last_heartbeat"), str):
                data["last_heartbeat"] = datetime.fromisoformat(data["last_heartbeat"])
                
            # Handle enums
            if isinstance(data.get("status"), str):
                data["status"] = ServiceStatus(data["status"])
                
            # Handle endpoints
            if "endpoints" in data:
                data["endpoints"] = [
                    ServiceEndpoint(**ep) if isinstance(ep, dict) else ep
                    for ep in data["endpoints"]
                ]
                
            # Handle sets
            if "tags" in data and isinstance(data["tags"], list):
                data["tags"] = set(data["tags"])
                
            return ServiceInstance(**data)
            
        except Exception as e:
            logger.error(f"Failed to convert dict to ServiceInstance: {e}")
            return None
            
    def _start_health_check(self, service: ServiceInstance):
        """Start health check task for service"""
        task_key = f"{service.service_name}:{service.instance_id}"
        
        if task_key not in self._health_check_tasks:
            task = asyncio.create_task(self._health_check_loop(service))
            self._health_check_tasks[task_key] = task
            
    async def _health_check_loop(self, service: ServiceInstance):
        """Periodic health check for service"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    if service.health_check_url:
                        async with session.get(service.health_check_url, timeout=5) as response:
                            if response.status == 200:
                                await self.update_service_status(
                                    service.service_name, 
                                    service.instance_id,
                                    ServiceStatus.HEALTHY
                                )
                            else:
                                await self.update_service_status(
                                    service.service_name,
                                    service.instance_id, 
                                    ServiceStatus.UNHEALTHY
                                )
                                
                except Exception as e:
                    logger.warning(f"Health check failed for {service.service_name}: {e}")
                    await self.update_service_status(
                        service.service_name,
                        service.instance_id,
                        ServiceStatus.UNHEALTHY
                    )
                    
                # Use event-driven sleep that can be interrupted
                try:
                    await asyncio.sleep(10)  # Check every 10 seconds
                except asyncio.CancelledError:
                    logger.info(f"Health check loop cancelled for {service.service_name}")
                    break
                
    async def _cleanup_expired_services(self):
        """Remove expired services"""
        while self._running:
            try:
                # Use event-driven sleep that can be interrupted
                await asyncio.sleep(60)  # Run every minute
                
                if not self._running:
                    break
                    
                # Clean local cache
                expired_services = []
                
                for service_name, instances in self._services.items():
                    for instance_id, service in instances.items():
                        if not service.is_healthy():
                            expired_services.append((service_name, instance_id))
                            
                for service_name, instance_id in expired_services:
                    await self.deregister_service(service_name, instance_id)
                    
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                
    async def _watch_service_changes(self):
        """Watch for service changes in Redis"""
        if not self.redis:
            return
            
        try:
            # This would implement Redis keyspace notifications
            # For simplicity, we'll use periodic polling
            while self._running:
                try:
                    await asyncio.sleep(5)
                    if not self._running:
                        break
                    # Poll for changes and notify watchers
                except asyncio.CancelledError:
                    logger.info("Service watcher cancelled")
                    break
                
        except Exception as e:
            logger.error(f"Error watching service changes: {e}")
            
    async def _notify_watchers(self, service_name: str, event_type: str, service: ServiceInstance):
        """Notify service watchers of changes"""
        for callback in self._watchers.get(service_name, []):
            try:
                await callback(event_type, service)
            except Exception as e:
                logger.error(f"Watcher callback failed: {e}")

class LoadBalancer:
    """Load balancer for distributed services"""
    
    def __init__(self, service_registry: ServiceRegistry, policy: LoadBalancingPolicy):
        self.registry = service_registry
        self.policy = policy
        self._round_robin_counters: Dict[str, int] = defaultdict(int)
        self._connection_counts: Dict[str, int] = defaultdict(int)
        
    async def get_endpoint(self, service_name: str, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceEndpoint]:
        """Get endpoint using load balancing policy"""
        instances = await self.registry.discover_services(service_name)
        
        if not instances:
            return None
            
        # Filter healthy instances
        healthy_instances = [i for i in instances if i.is_healthy()]
        
        if not healthy_instances:
            if self.policy.failover_enabled:
                # Try unhealthy instances as last resort
                healthy_instances = instances
            else:
                return None
                
        # Apply load balancing strategy
        if self.policy.strategy == "round_robin":
            return self._round_robin_select(service_name, healthy_instances)
        elif self.policy.strategy == "weighted":
            return self._weighted_select(healthy_instances)
        elif self.policy.strategy == "least_connections":
            return self._least_connections_select(healthy_instances)
        elif self.policy.strategy == "consistent_hash":
            return self._consistent_hash_select(healthy_instances, request_context)
        else:
            # Default to round robin
            return self._round_robin_select(service_name, healthy_instances)
            
    def _round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceEndpoint:
        """Round robin selection"""
        counter = self._round_robin_counters[service_name]
        instance = instances[counter % len(instances)]
        self._round_robin_counters[service_name] = counter + 1
        
        return instance.endpoints[0] if instance.endpoints else None
        
    def _weighted_select(self, instances: List[ServiceInstance]) -> ServiceEndpoint:
        """Weighted random selection"""
        import random
        
        # Calculate total weight
        total_weight = sum(ep.weight for instance in instances for ep in instance.endpoints)
        
        if total_weight == 0:
            return random.choice(instances).endpoints[0]
            
        # Select based on weight
        target = random.randint(1, total_weight)
        current = 0
        
        for instance in instances:
            for endpoint in instance.endpoints:
                current += endpoint.weight
                if current >= target:
                    return endpoint
                    
        return instances[0].endpoints[0]
        
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceEndpoint:
        """Least connections selection"""
        min_connections = float('inf')
        selected_instance = None
        
        for instance in instances:
            key = f"{instance.service_name}:{instance.instance_id}"
            connections = self._connection_counts[key]
            
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance
                
        return selected_instance.endpoints[0] if selected_instance and selected_instance.endpoints else None
        
    def _consistent_hash_select(self, instances: List[ServiceInstance], 
                               context: Optional[Dict[str, Any]]) -> ServiceEndpoint:
        """Consistent hash selection"""
        if not context or "hash_key" not in context:
            # Fallback to round robin
            return self._round_robin_select("", instances)
            
        hash_key = str(context["hash_key"])
        hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        
        selected_instance = instances[hash_value % len(instances)]
        return selected_instance.endpoints[0] if selected_instance.endpoints else None
        
    def track_connection(self, service_name: str, instance_id: str):
        """Track new connection for least connections balancing"""
        key = f"{service_name}:{instance_id}"
        self._connection_counts[key] += 1
        
    def release_connection(self, service_name: str, instance_id: str):
        """Release connection for least connections balancing"""
        key = f"{service_name}:{instance_id}"
        if self._connection_counts[key] > 0:
            self._connection_counts[key] -= 1

class DistributedCoordinator:
    """Distributed coordination using Redis for leader election and locking"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.node_id = str(uuid.uuid4())
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Distributed coordinator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize coordinator: {e}")
            raise
            
    async def elect_leader(self, group: str, ttl: int = 30) -> bool:
        """Attempt to become leader for a group"""
        if not self.redis:
            return False
            
        try:
            key = f"leader:{group}"
            result = await self.redis.set(key, self.node_id, ex=ttl, nx=True)
            
            if result:
                logger.info(f"Became leader for group {group}")
                return True
            else:
                current_leader = await self.redis.get(key)
                logger.debug(f"Leader for {group} is {current_leader.decode() if current_leader else 'unknown'}")
                return False
                
        except Exception as e:
            logger.error(f"Leader election failed: {e}")
            return False
            
    async def is_leader(self, group: str) -> bool:
        """Check if this node is the leader"""
        if not self.redis:
            return False
            
        try:
            key = f"leader:{group}"
            current_leader = await self.redis.get(key)
            return current_leader and current_leader.decode() == self.node_id
        except Exception as e:
            logger.error(f"Leader check failed: {e}")
            return False
            
    async def acquire_lock(self, resource: str, timeout: int = 30) -> bool:
        """Acquire distributed lock"""
        if not self.redis:
            return False
            
        try:
            key = f"lock:{resource}"
            lock_id = f"{self.node_id}:{time.time()}"
            
            result = await self.redis.set(key, lock_id, ex=timeout, nx=True)
            
            if result:
                logger.debug(f"Acquired lock for {resource}")
                return True
            else:
                logger.debug(f"Failed to acquire lock for {resource}")
                return False
                
        except Exception as e:
            logger.error(f"Lock acquisition failed: {e}")
            return False
            
    async def release_lock(self, resource: str) -> bool:
        """Release distributed lock"""
        if not self.redis:
            return False
            
        try:
            key = f"lock:{resource}"
            lock_value = await self.redis.get(key)
            
            if lock_value and lock_value.decode().startswith(self.node_id):
                await self.redis.delete(key)
                logger.debug(f"Released lock for {resource}")
                return True
            else:
                logger.warning(f"Cannot release lock for {resource} - not owner")
                return False
                
        except Exception as e:
            logger.error(f"Lock release failed: {e}")
            return False

# Helper functions for easy service registration
def get_local_ip() -> str:
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

async def register_current_service(registry: ServiceRegistry, service_name: str, 
                                 port: int, health_path: str = "/health") -> ServiceInstance:
    """Register current service instance"""
    local_ip = get_local_ip()
    instance_id = f"{local_ip}:{port}"
    
    endpoints = [ServiceEndpoint(host=local_ip, port=port)]
    health_url = f"http://{local_ip}:{port}{health_path}"
    
    service = ServiceInstance(
        service_id=f"{service_name}-{instance_id}",
        service_name=service_name,
        instance_id=instance_id,
        endpoints=endpoints,
        health_check_url=health_url,
        status=ServiceStatus.HEALTHY
    )
    
    await registry.register_service(service)
    return service