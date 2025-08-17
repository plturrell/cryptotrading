"""
Distributed Strands Framework
Extends the Strands framework with distributed execution capabilities
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import aioredis
import aiohttp

from .strands import (
    StrandsAgent, WorkflowContext, WorkflowStatus, ToolStatus,
    ToolExecutionResult, WorkflowEngine, ToolRegistry
)
from ..config.production_config import StrandsConfig
from ...infrastructure.distributed.service_discovery import (
    ServiceRegistry, LoadBalancer, LoadBalancingPolicy, DistributedCoordinator,
    ServiceInstance, ServiceEndpoint, ServiceStatus, register_current_service
)

logger = logging.getLogger(__name__)

@dataclass
class DistributedWorkflowContext(WorkflowContext):
    """Extended workflow context for distributed execution"""
    assigned_nodes: Set[str] = None
    distribution_strategy: str = "load_balanced"  # load_balanced, broadcast, specific_nodes
    preferred_capabilities: List[str] = None
    resource_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.assigned_nodes is None:
            self.assigned_nodes = set()
        if self.preferred_capabilities is None:
            self.preferred_capabilities = []
        if self.resource_requirements is None:
            self.resource_requirements = {}

class DistributedToolRegistry(ToolRegistry):
    """Tool registry with distributed execution capabilities"""
    
    def __init__(self, service_registry: ServiceRegistry, load_balancer: LoadBalancer):
        super().__init__()
        self.service_registry = service_registry
        self.load_balancer = load_balancer
        self._remote_tools: Dict[str, Dict[str, Any]] = {}
        
    async def discover_remote_tools(self):
        """Discover tools available on remote nodes"""
        try:
            # Discover strands services
            strands_services = await self.service_registry.discover_services("strands-agent")
            
            for service in strands_services:
                await self._fetch_remote_tools(service)
                
        except Exception as e:
            logger.error(f"Failed to discover remote tools: {e}")
            
    async def _fetch_remote_tools(self, service: ServiceInstance):
        """Fetch tools from a remote service"""
        if not service.endpoints:
            return
            
        endpoint = service.endpoints[0]
        url = f"{endpoint.to_url()}/tools"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        tools = await response.json()
                        
                        for tool in tools:
                            tool_name = tool["name"]
                            self._remote_tools[tool_name] = {
                                "service": service,
                                "endpoint": endpoint,
                                "metadata": tool
                            }
                            
                        logger.debug(f"Discovered {len(tools)} tools from {service.instance_id}")
                        
        except Exception as e:
            logger.warning(f"Failed to fetch tools from {service.instance_id}: {e}")
            
    async def execute_remote(self, tool_name: str, parameters: Dict[str, Any]) -> ToolExecutionResult:
        """Execute tool on remote service"""
        if tool_name not in self._remote_tools:
            raise ValueError(f"Remote tool {tool_name} not found")
            
        remote_info = self._remote_tools[tool_name]
        service = remote_info["service"]
        endpoint = remote_info["endpoint"]
        
        # Select endpoint using load balancer
        selected_endpoint = await self.load_balancer.get_endpoint(
            service.service_name, 
            {"tool_name": tool_name}
        )
        
        if not selected_endpoint:
            raise Exception(f"No available endpoints for {service.service_name}")
            
        # Execute tool remotely
        url = f"{selected_endpoint.to_url()}/execute-tool"
        payload = {
            "tool_name": tool_name,
            "parameters": parameters
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return ToolExecutionResult(**result_data)
                    else:
                        error_text = await response.text()
                        raise Exception(f"Remote execution failed: {error_text}")
                        
        except Exception as e:
            logger.error(f"Remote tool execution failed: {e}")
            raise
            
    def list_all_tools(self) -> List[Dict[str, Any]]:
        """List both local and remote tools"""
        tools = super().list_tools()
        
        # Add remote tools
        for tool_name, info in self._remote_tools.items():
            tool_info = info["metadata"].copy()
            tool_info["remote"] = True
            tool_info["service"] = info["service"].instance_id
            tools.append(tool_info)
            
        return tools

class DistributedWorkflowEngine(WorkflowEngine):
    """Workflow engine with distributed execution capabilities"""
    
    def __init__(self, config: StrandsConfig, service_registry: ServiceRegistry, 
                 coordinator: DistributedCoordinator):
        super().__init__(config)
        self.service_registry = service_registry
        self.coordinator = coordinator
        self._distributed_workflows: Dict[str, DistributedWorkflowContext] = {}
        
    async def distribute_workflow(self, workflow: DistributedWorkflowContext) -> List[str]:
        """Distribute workflow across available nodes"""
        if workflow.distribution_strategy == "broadcast":
            return await self._broadcast_workflow(workflow)
        elif workflow.distribution_strategy == "specific_nodes":
            return await self._execute_on_specific_nodes(workflow)
        else:
            return await self._load_balanced_distribution(workflow)
            
    async def _broadcast_workflow(self, workflow: DistributedWorkflowContext) -> List[str]:
        """Broadcast workflow to all available nodes"""
        services = await self.service_registry.discover_services("strands-agent")
        execution_ids = []
        
        for service in services:
            if service.endpoints:
                try:
                    execution_id = await self._send_workflow_to_service(workflow, service)
                    if execution_id:
                        execution_ids.append(execution_id)
                        workflow.assigned_nodes.add(service.instance_id)
                except Exception as e:
                    logger.error(f"Failed to send workflow to {service.instance_id}: {e}")
                    
        return execution_ids
        
    async def _load_balanced_distribution(self, workflow: DistributedWorkflowContext) -> List[str]:
        """Distribute workflow using load balancing"""
        # For now, select one node
        services = await self.service_registry.discover_services("strands-agent")
        
        if not services:
            raise Exception("No Strands services available")
            
        # Filter by capabilities if specified
        if workflow.preferred_capabilities:
            filtered_services = []
            for service in services:
                service_caps = service.metadata.get("capabilities", [])
                if any(cap in service_caps for cap in workflow.preferred_capabilities):
                    filtered_services.append(service)
            services = filtered_services if filtered_services else services
            
        # Select best service
        selected_service = self._select_best_service(services, workflow)
        
        if selected_service:
            execution_id = await self._send_workflow_to_service(workflow, selected_service)
            if execution_id:
                workflow.assigned_nodes.add(selected_service.instance_id)
                return [execution_id]
                
        return []
        
    async def _execute_on_specific_nodes(self, workflow: DistributedWorkflowContext) -> List[str]:
        """Execute workflow on specific nodes"""
        if not workflow.assigned_nodes:
            return await self._load_balanced_distribution(workflow)
            
        services = await self.service_registry.discover_services("strands-agent")
        target_services = [s for s in services if s.instance_id in workflow.assigned_nodes]
        
        execution_ids = []
        for service in target_services:
            try:
                execution_id = await self._send_workflow_to_service(workflow, service)
                if execution_id:
                    execution_ids.append(execution_id)
            except Exception as e:
                logger.error(f"Failed to execute on {service.instance_id}: {e}")
                
        return execution_ids
        
    def _select_best_service(self, services: List[ServiceInstance], 
                           workflow: DistributedWorkflowContext) -> Optional[ServiceInstance]:
        """Select best service for workflow execution"""
        if not services:
            return None
            
        # Score services based on various factors
        best_service = None
        best_score = -1
        
        for service in services:
            score = 0
            
            # Health score
            if service.status == ServiceStatus.HEALTHY:
                score += 10
                
            # Capability match
            service_caps = service.metadata.get("capabilities", [])
            matching_caps = sum(1 for cap in workflow.preferred_capabilities if cap in service_caps)
            score += matching_caps * 5
            
            # Resource availability (if provided in metadata)
            cpu_usage = service.metadata.get("cpu_usage", 50)
            memory_usage = service.metadata.get("memory_usage", 50)
            score += max(0, 100 - cpu_usage - memory_usage) / 10
            
            # Load (concurrent workflows)
            active_workflows = service.metadata.get("active_workflows", 0)
            max_workflows = service.metadata.get("max_workflows", 10)
            if active_workflows < max_workflows:
                score += (max_workflows - active_workflows) * 2
                
            if score > best_score:
                best_score = score
                best_service = service
                
        return best_service
        
    async def _send_workflow_to_service(self, workflow: DistributedWorkflowContext, 
                                      service: ServiceInstance) -> Optional[str]:
        """Send workflow to a specific service"""
        if not service.endpoints:
            return None
            
        endpoint = service.endpoints[0]
        url = f"{endpoint.to_url()}/execute-workflow"
        
        # Convert workflow to serializable format
        payload = {
            "workflow_id": workflow.workflow_id,
            "inputs": workflow.inputs,
            "metadata": workflow.metadata,
            "distribution_context": {
                "original_node": self.coordinator.node_id,
                "distribution_strategy": workflow.distribution_strategy,
                "preferred_capabilities": workflow.preferred_capabilities,
                "resource_requirements": workflow.resource_requirements
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("execution_id")
                    else:
                        error_text = await response.text()
                        logger.error(f"Workflow submission failed: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to send workflow to {service.instance_id}: {e}")
            return None

class DistributedStrandsAgent(StrandsAgent):
    """Strands agent with distributed execution capabilities"""
    
    def __init__(self, agent_id: str, agent_type: str,
                 capabilities: Optional[List[str]] = None,
                 model_provider: str = "grok4",
                 config: Optional[StrandsConfig] = None,
                 redis_url: str = "redis://localhost:6379",
                 service_port: int = 8888,
                 **kwargs):
                 
        super().__init__(agent_id, agent_type, capabilities, model_provider, config, **kwargs)
        
        # Distributed components
        self.redis_url = redis_url
        self.service_port = service_port
        self.service_registry = ServiceRegistry(redis_url)
        self.coordinator = DistributedCoordinator(redis_url)
        
        # Load balancing
        lb_policy = LoadBalancingPolicy(
            strategy="weighted",
            health_check_enabled=True,
            failover_enabled=True,
            circuit_breaker_enabled=True
        )
        self.load_balancer = LoadBalancer(self.service_registry, lb_policy)
        
        # Replace components with distributed versions
        self.tool_registry = DistributedToolRegistry(self.service_registry, self.load_balancer)
        self.workflow_engine = DistributedWorkflowEngine(self.config, self.service_registry, self.coordinator)
        
        # Service instance
        self._service_instance: Optional[ServiceInstance] = None
        self._is_leader = False
        
    async def initialize_distributed(self):
        """Initialize distributed components"""
        try:
            # Initialize service registry and coordinator
            await self.service_registry.initialize()
            await self.coordinator.initialize()
            
            # Register this service
            self._service_instance = await register_current_service(
                self.service_registry,
                "strands-agent",
                self.service_port,
                "/health"
            )
            
            # Update service metadata
            if self._service_instance:
                self._service_instance.metadata.update({
                    "capabilities": self.capabilities,
                    "agent_type": self.agent_type,
                    "max_workflows": self.config.max_concurrent_workflows,
                    "active_workflows": 0,
                    "tools_count": len(self.tool_registry._tools)
                })
                
            # Discover remote tools
            await self.tool_registry.discover_remote_tools()
            
            # Start distributed background tasks
            asyncio.create_task(self._leader_election_loop())
            asyncio.create_task(self._service_heartbeat_loop())
            asyncio.create_task(self._tool_discovery_loop())
            
            logger.info(f"Distributed Strands agent {self.agent_id} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed components: {e}")
            raise
            
    async def execute_tool_distributed(self, tool_name: str, parameters: Dict[str, Any],
                                     prefer_local: bool = True) -> ToolExecutionResult:
        """Execute tool with distributed fallback"""
        # Try local execution first if preferred
        if prefer_local and tool_name in self.tool_registry._tools:
            try:
                return await self.execute_tool(tool_name, parameters)
            except Exception as e:
                logger.warning(f"Local execution failed, trying remote: {e}")
                
        # Try remote execution
        if tool_name in self.tool_registry._remote_tools:
            try:
                return await self.tool_registry.execute_remote(tool_name, parameters)
            except Exception as e:
                logger.error(f"Remote execution failed: {e}")
                
                # Fallback to local if available
                if tool_name in self.tool_registry._tools:
                    return await self.execute_tool(tool_name, parameters)
                else:
                    raise
                    
        # Tool not found anywhere
        raise ValueError(f"Tool {tool_name} not found locally or remotely")
        
    async def process_workflow_distributed(self, workflow_id: str, inputs: Dict[str, Any],
                                         distribution_strategy: str = "load_balanced",
                                         preferred_capabilities: List[str] = None) -> Dict[str, Any]:
        """Process workflow with distributed execution"""
        
        # Create distributed workflow context
        context = DistributedWorkflowContext(
            workflow_id=workflow_id,
            agent_id=self.agent_id,
            inputs=inputs,
            distribution_strategy=distribution_strategy,
            preferred_capabilities=preferred_capabilities or []
        )
        
        # Check if workflow should be distributed
        should_distribute = self._should_distribute_workflow(context)
        
        if should_distribute:
            # Distribute workflow
            execution_ids = await self.workflow_engine.distribute_workflow(context)
            
            if execution_ids:
                # Wait for distributed execution
                return await self._wait_for_distributed_results(context, execution_ids)
            else:
                logger.warning(f"Workflow distribution failed, executing locally")
                
        # Execute locally
        return await self.process_workflow(workflow_id, inputs)
        
    def _should_distribute_workflow(self, context: DistributedWorkflowContext) -> bool:
        """Determine if workflow should be distributed"""
        # Always distribute if specific strategy is requested
        if context.distribution_strategy in ["broadcast", "specific_nodes"]:
            return True
            
        # Distribute if workflow has specific capability requirements
        if context.preferred_capabilities:
            local_caps = set(self.capabilities)
            required_caps = set(context.preferred_capabilities)
            if not required_caps.issubset(local_caps):
                return True
                
        # Distribute if this node is overloaded
        current_workflows = len(self.workflow_engine._workflows)
        if current_workflows >= self.config.max_concurrent_workflows:
            return True
            
        # Distribute complex workflows
        workflow_def = context.inputs.get("workflow_definition", {})
        steps = workflow_def.get("steps", [])
        if len(steps) > 10:  # Arbitrary threshold
            return True
            
        return False
        
    async def _wait_for_distributed_results(self, context: DistributedWorkflowContext,
                                          execution_ids: List[str]) -> Dict[str, Any]:
        """Wait for distributed workflow results"""
        # This is a simplified implementation
        # In production, you'd implement proper result aggregation
        
        results = []
        
        for execution_id in execution_ids:
            try:
                # Poll for results (simplified)
                await asyncio.sleep(1)
                result = {
                    "execution_id": execution_id,
                    "status": WorkflowStatus.COMPLETED,
                    "result": f"Distributed execution {execution_id} completed"
                }
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to get result for {execution_id}: {e}")
                
        return {
            "workflow_id": context.workflow_id,
            "status": WorkflowStatus.COMPLETED,
            "distributed_results": results,
            "assigned_nodes": list(context.assigned_nodes)
        }
        
    async def _leader_election_loop(self):
        """Participate in leader election"""
        while True:
            try:
                self._is_leader = await self.coordinator.elect_leader("strands-cluster")
                
                if self._is_leader:
                    logger.info("Became cluster leader")
                    # Perform leader duties
                    await self._leader_duties()
                else:
                    logger.debug("Not cluster leader")
                    
                await asyncio.sleep(20)  # Re-elect every 20 seconds
                
            except Exception as e:
                logger.error(f"Leader election error: {e}")
                await asyncio.sleep(5)
                
    async def _leader_duties(self):
        """Perform duties as cluster leader"""
        try:
            # Coordinate global cleanup
            await self._cleanup_distributed_state()
            
            # Balance load across cluster
            await self._rebalance_cluster_load()
            
        except Exception as e:
            logger.error(f"Leader duties failed: {e}")
            
    async def _cleanup_distributed_state(self):
        """Clean up distributed state as leader"""
        # Clean up expired workflows, failed executions, etc.
        pass
        
    async def _rebalance_cluster_load(self):
        """Rebalance load across cluster"""
        # Implement load rebalancing logic
        pass
        
    async def _service_heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                if self._service_instance:
                    await self.service_registry.heartbeat(
                        self._service_instance.service_name,
                        self._service_instance.instance_id
                    )
                    
                    # Update metadata
                    current_workflows = len(self.workflow_engine._workflows)
                    await self._update_service_metadata({
                        "active_workflows": current_workflows,
                        "cpu_usage": 45,  # Would get from system
                        "memory_usage": 60,  # Would get from system
                        "is_leader": self._is_leader
                    })
                    
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(5)
                
    async def _tool_discovery_loop(self):
        """Periodically discover new tools"""
        while True:
            try:
                await asyncio.sleep(60)  # Discover every minute
                await self.tool_registry.discover_remote_tools()
            except Exception as e:
                logger.error(f"Tool discovery failed: {e}")
                
    async def _update_service_metadata(self, metadata: Dict[str, Any]):
        """Update service metadata"""
        if self._service_instance:
            self._service_instance.metadata.update(metadata)
            
    async def shutdown_distributed(self):
        """Shutdown distributed components"""
        try:
            if self._service_instance:
                await self.service_registry.deregister_service(
                    self._service_instance.service_name,
                    self._service_instance.instance_id
                )
                
            await self.service_registry.close()
            
        except Exception as e:
            logger.error(f"Distributed shutdown failed: {e}")
            
        await super().shutdown()