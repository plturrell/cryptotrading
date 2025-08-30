"""
MCP Agent Lifecycle Manager
Centralized lifecycle management for all MCP-enabled specialized agents
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import weakref

from .strands import StrandsAgent
from .memory import MemoryAgent
from .specialized.agent_manager import AgentManagerAgent
from .specialized.technical_analysis.technical_analysis_agent import TechnicalAnalysisAgent
from .specialized.ml_agent import MLAgent
from .specialized.strands_glean_agent import StrandsGleanAgent
from .specialized.mcts_calculation_agent import MCTSCalculationAgent
from .specialized.trading_algorithm_agent import TradingAlgorithmAgent
from .specialized.data_analysis_agent import DataAnalysisAgent
from .specialized.feature_store_agent import FeatureStoreAgent
from ..protocols.a2a.a2a_protocol import A2A_CAPABILITIES, MessageType, AgentStatus

logger = logging.getLogger(__name__)

class AgentLifecycleState(Enum):
    """Lifecycle states for MCP agents"""
    UNREGISTERED = "unregistered"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"

class AgentType(Enum):
    """Types of specialized agents"""
    AGENT_MANAGER = "agent_manager"
    TECHNICAL_ANALYSIS = "technical_analysis"
    ML_AGENT = "ml_agent"
    GLEAN_AGENT = "glean_agent"
    MCTS_CALCULATION = "mcts_calculation"
    HISTORICAL_DATA_LOADER = "historical_data_loader"
    DATABASE_MANAGER = "database_manager"
    TRADING_ALGORITHM = "trading_algorithm"
    DATA_ANALYSIS = "data_analysis"
    FEATURE_STORE = "feature_store"

@dataclass
class AgentHealthMetrics:
    """Health metrics for an agent"""
    agent_id: str
    agent_type: AgentType
    state: AgentLifecycleState
    health_score: float  # 0.0 to 1.0
    memory_usage_mb: float
    cache_hit_rate: float
    success_rate: float
    avg_response_time: float
    error_count: int
    last_activity: datetime
    uptime_seconds: float
    dependencies_healthy: bool
    memory_entries: int
    last_health_check: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AgentConfiguration:
    """Configuration for an agent"""
    agent_id: str
    agent_type: AgentType
    agent_class: type
    dependencies: List[str] = field(default_factory=list)
    auto_start: bool = True
    health_check_interval: float = 60.0
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    memory_limit_mb: float = 512.0
    initialization_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LifecycleEvent:
    """Lifecycle event for tracking agent state changes"""
    timestamp: datetime
    agent_id: str
    event_type: str
    from_state: Optional[AgentLifecycleState]
    to_state: AgentLifecycleState
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class MCPAgentLifecycleManager:
    """
    Centralized lifecycle manager for all MCP-enabled specialized agents
    
    Features:
    - Agent registration and discovery
    - Coordinated startup/shutdown sequences
    - Health monitoring and recovery
    - Dependency management
    - Resource management and cleanup
    - Performance tracking
    - Memory system coordination
    - Inter-agent communication
    """
    
    def __init__(self, health_check_interval: float = 30.0, 
                 cleanup_interval: float = 300.0,
                 max_event_history: int = 1000):
        self.health_check_interval = health_check_interval
        self.cleanup_interval = cleanup_interval
        self.max_event_history = max_event_history
        
        # Agent tracking
        self.agents: Dict[str, StrandsAgent] = {}
        self.agent_configs: Dict[str, AgentConfiguration] = {}
        self.agent_health: Dict[str, AgentHealthMetrics] = {}
        self.agent_refs: Dict[str, weakref.ref] = {}
        
        # State management
        self.manager_state = AgentLifecycleState.STOPPED
        self.shutdown_requested = False
        self.startup_sequence: List[str] = []
        self.shutdown_sequence: List[str] = []
        
        # Event tracking
        self.event_history: List[LifecycleEvent] = []
        self.restart_attempts: Dict[str, int] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Lifecycle hooks
        self.lifecycle_hooks: Dict[str, List[Callable]] = {
            "before_agent_start": [],
            "after_agent_start": [],
            "before_agent_stop": [],
            "after_agent_stop": [],
            "on_agent_error": [],
            "on_health_check": []
        }
        
        # Initialize default agent configurations
        self._initialize_default_configurations()
        
        logger.info("MCP Agent Lifecycle Manager initialized")
    
    def _initialize_default_configurations(self):
        """Initialize default configurations for specialized agents"""
        self.agent_configs = {
            "agent-manager-001": AgentConfiguration(
                agent_id="agent-manager-001",
                agent_type=AgentType.AGENT_MANAGER,
                agent_class=AgentManagerAgent,
                dependencies=[],
                auto_start=True,
                health_check_interval=30.0
            ),
            "technical_analysis_agent": AgentConfiguration(
                agent_id="technical_analysis_agent",
                agent_type=AgentType.TECHNICAL_ANALYSIS,
                agent_class=TechnicalAnalysisAgent,
                dependencies=["agent-manager-001"],
                auto_start=True,
                health_check_interval=60.0
            ),
            "ml_agent": AgentConfiguration(
                agent_id="ml_agent",
                agent_type=AgentType.ML_AGENT,
                agent_class=MLAgent,
                dependencies=["agent-manager-001"],
                auto_start=True,
                health_check_interval=45.0
            ),
            "strands_glean_agent": AgentConfiguration(
                agent_id="strands_glean_agent",
                agent_type=AgentType.GLEAN_AGENT,
                agent_class=StrandsGleanAgent,
                dependencies=["agent-manager-001"],
                auto_start=True,
                health_check_interval=90.0
            ),
            "mcts_calculation_agent": AgentConfiguration(
                agent_id="mcts_calculation_agent",
                agent_type=AgentType.MCTS_CALCULATION,
                agent_class=MCTSCalculationAgent,
                dependencies=["agent-manager-001", "ml_agent"],
                auto_start=True,
                health_check_interval=120.0
            ),
            "trading_algorithm_agent": AgentConfiguration(
                agent_id="trading_algorithm_agent",
                agent_type=AgentType.TRADING_ALGORITHM,
                agent_class=TradingAlgorithmAgent,
                dependencies=["agent-manager-001", "mcts_calculation_agent"],
                auto_start=False,  # Manual start required - analysis only
                health_check_interval=60.0
            ),
            "data_analysis_agent": AgentConfiguration(
                agent_id="data_analysis_agent",
                agent_type=AgentType.DATA_ANALYSIS,
                agent_class=DataAnalysisAgent,
                dependencies=["agent-manager-001"],
                auto_start=True,
                health_check_interval=60.0
            ),
            "feature_store_agent": AgentConfiguration(
                agent_id="feature_store_agent",
                agent_type=AgentType.FEATURE_STORE,
                agent_class=FeatureStoreAgent,
                dependencies=["agent-manager-001", "data_analysis_agent"],
                auto_start=True,
                health_check_interval=90.0
            )
        }
        
        # Calculate startup sequence based on dependencies
        self._calculate_startup_sequence()
    
    def _calculate_startup_sequence(self):
        """Calculate optimal startup sequence based on dependencies"""
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        sequence = []
        
        def visit(agent_id: str):
            if agent_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {agent_id}")
            if agent_id in visited:
                return
            
            temp_visited.add(agent_id)
            
            if agent_id in self.agent_configs:
                for dep_id in self.agent_configs[agent_id].dependencies:
                    visit(dep_id)
            
            temp_visited.remove(agent_id)
            visited.add(agent_id)
            sequence.append(agent_id)
        
        for agent_id in self.agent_configs.keys():
            if agent_id not in visited:
                visit(agent_id)
        
        self.startup_sequence = sequence
        self.shutdown_sequence = list(reversed(sequence))
        
        logger.info(f"Calculated startup sequence: {self.startup_sequence}")
    
    async def start(self):
        """Start the MCP Agent Lifecycle Manager"""
        logger.info("Starting MCP Agent Lifecycle Manager")
        
        await self._transition_manager_state(AgentLifecycleState.STARTING)
        
        try:
            # Start background monitoring tasks
            self._background_tasks = [
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._cleanup_loop()),
                asyncio.create_task(self._event_processor()),
                asyncio.create_task(self._memory_coordinator())
            ]
            
            # Start agents in dependency order
            await self._start_all_agents()
            
            await self._transition_manager_state(AgentLifecycleState.RUNNING)
            logger.info("MCP Agent Lifecycle Manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP Agent Lifecycle Manager: {e}")
            await self._transition_manager_state(AgentLifecycleState.ERROR)
            raise
    
    async def shutdown(self):
        """Graceful shutdown of all agents and manager"""
        logger.info("Shutting down MCP Agent Lifecycle Manager")
        
        await self._transition_manager_state(AgentLifecycleState.STOPPING)
        self.shutdown_requested = True
        
        try:
            # Execute before_stop hooks
            await self._execute_hooks("before_agent_stop")
            
            # Stop all agents in reverse dependency order
            await self._stop_all_agents()
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Execute after_stop hooks
            await self._execute_hooks("after_agent_stop")
            
            await self._transition_manager_state(AgentLifecycleState.STOPPED)
            logger.info("MCP Agent Lifecycle Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            await self._transition_manager_state(AgentLifecycleState.ERROR)
    
    async def register_agent(self, config: AgentConfiguration) -> bool:
        """Register an agent with the lifecycle manager"""
        try:
            agent_id = config.agent_id
            
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} already registered, updating configuration")
            
            # Store configuration
            self.agent_configs[agent_id] = config
            
            # Initialize health metrics
            self.agent_health[agent_id] = AgentHealthMetrics(
                agent_id=agent_id,
                agent_type=config.agent_type,
                state=AgentLifecycleState.UNREGISTERED,
                health_score=0.0,
                memory_usage_mb=0.0,
                cache_hit_rate=0.0,
                success_rate=0.0,
                avg_response_time=0.0,
                error_count=0,
                last_activity=datetime.utcnow(),
                uptime_seconds=0.0,
                dependencies_healthy=False,
                memory_entries=0
            )
            
            # Recalculate startup sequence
            self._calculate_startup_sequence()
            
            # Record event
            self._record_event(agent_id, "registered", None, AgentLifecycleState.UNREGISTERED)
            
            logger.info(f"Registered agent: {agent_id} ({config.agent_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {config.agent_id}: {e}")
            return False
    
    async def start_agent(self, agent_id: str) -> bool:
        """Start a specific agent"""
        if agent_id not in self.agent_configs:
            logger.error(f"Agent {agent_id} not registered")
            return False
        
        config = self.agent_configs[agent_id]
        
        try:
            await self._transition_agent_state(agent_id, AgentLifecycleState.INITIALIZING)
            
            # Check dependencies
            if not await self._check_agent_dependencies(agent_id):
                logger.error(f"Dependencies not met for agent {agent_id}")
                await self._transition_agent_state(agent_id, AgentLifecycleState.ERROR)
                return False
            
            # Create agent instance if not exists
            if agent_id not in self.agents:
                await self._create_agent_instance(agent_id, config)
            
            agent = self.agents[agent_id]
            
            await self._transition_agent_state(agent_id, AgentLifecycleState.STARTING)
            
            # Execute before_start hooks
            await self._execute_hooks("before_agent_start", agent)
            
            # Initialize agent if it has initialize method
            if hasattr(agent, 'initialize'):
                await agent.initialize()
            
            # Start agent if it has start method
            if hasattr(agent, 'start'):
                await agent.start()
            
            await self._transition_agent_state(agent_id, AgentLifecycleState.RUNNING)
            
            # Execute after_start hooks
            await self._execute_hooks("after_agent_start", agent)
            
            # Reset restart attempts
            self.restart_attempts[agent_id] = 0
            
            logger.info(f"Started agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent {agent_id}: {e}")
            await self._transition_agent_state(agent_id, AgentLifecycleState.ERROR)
            await self._handle_agent_error(agent_id, e)
            return False
    
    async def stop_agent(self, agent_id: str) -> bool:
        """Stop a specific agent"""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found or already stopped")
            return True
        
        try:
            await self._transition_agent_state(agent_id, AgentLifecycleState.STOPPING)
            
            agent = self.agents[agent_id]
            
            # Execute before_stop hooks
            await self._execute_hooks("before_agent_stop", agent)
            
            # Stop agent if it has stop method
            if hasattr(agent, 'stop'):
                await agent.stop()
            elif hasattr(agent, 'shutdown'):
                await agent.shutdown()
            
            # Clean up agent memory
            if isinstance(agent, MemoryAgent):
                await agent.clear_memory()
            
            await self._transition_agent_state(agent_id, AgentLifecycleState.STOPPED)
            
            # Execute after_stop hooks
            await self._execute_hooks("after_agent_stop", agent)
            
            # Remove from active agents
            del self.agents[agent_id]
            if agent_id in self.agent_refs:
                del self.agent_refs[agent_id]
            
            logger.info(f"Stopped agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}: {e}")
            await self._transition_agent_state(agent_id, AgentLifecycleState.ERROR)
            return False
    
    async def restart_agent(self, agent_id: str) -> bool:
        """Restart a specific agent"""
        logger.info(f"Restarting agent: {agent_id}")
        
        # Check restart attempts
        attempts = self.restart_attempts.get(agent_id, 0)
        max_attempts = self.agent_configs[agent_id].max_restart_attempts
        
        if attempts >= max_attempts:
            logger.error(f"Agent {agent_id} exceeded max restart attempts ({max_attempts})")
            return False
        
        self.restart_attempts[agent_id] = attempts + 1
        
        # Stop and start agent
        await self.stop_agent(agent_id)
        await asyncio.sleep(1)  # Brief pause
        return await self.start_agent(agent_id)
    
    async def get_agent_health(self, agent_id: str) -> Optional[AgentHealthMetrics]:
        """Get health metrics for a specific agent"""
        if agent_id not in self.agent_health:
            return None
        
        await self._update_agent_health(agent_id)
        return self.agent_health[agent_id]
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        total_agents = len(self.agent_configs)
        running_agents = sum(1 for h in self.agent_health.values() if h.state == AgentLifecycleState.RUNNING)
        error_agents = sum(1 for h in self.agent_health.values() if h.state == AgentLifecycleState.ERROR)
        
        overall_health = sum(h.health_score for h in self.agent_health.values()) / max(total_agents, 1)
        
        return {
            "manager_state": self.manager_state.value,
            "overall_health": overall_health,
            "total_agents": total_agents,
            "running_agents": running_agents,
            "error_agents": error_agents,
            "startup_sequence": self.startup_sequence,
            "agents": {
                agent_id: {
                    "type": health.agent_type.value,
                    "state": health.state.value,
                    "health_score": health.health_score,
                    "uptime_seconds": health.uptime_seconds,
                    "memory_entries": health.memory_entries,
                    "error_count": health.error_count,
                    "last_activity": health.last_activity.isoformat()
                }
                for agent_id, health in self.agent_health.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_agent_instance(self, agent_id: str, config: AgentConfiguration):
        """Create an agent instance"""
        try:
            # Create agent with configuration parameters
            agent = config.agent_class(
                agent_id=agent_id,
                **config.initialization_params
            )
            
            self.agents[agent_id] = agent
            self.agent_refs[agent_id] = weakref.ref(agent)
            
            logger.info(f"Created agent instance: {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to create agent instance {agent_id}: {e}")
            raise
    
    async def _start_all_agents(self):
        """Start all agents in dependency order"""
        for agent_id in self.startup_sequence:
            config = self.agent_configs[agent_id]
            if config.auto_start:
                success = await self.start_agent(agent_id)
                if not success:
                    logger.error(f"Failed to start agent {agent_id} during startup")
                    # Continue with other agents
    
    async def _stop_all_agents(self):
        """Stop all agents in reverse dependency order"""
        for agent_id in self.shutdown_sequence:
            if agent_id in self.agents:
                await self.stop_agent(agent_id)
    
    async def _check_agent_dependencies(self, agent_id: str) -> bool:
        """Check if agent dependencies are running"""
        config = self.agent_configs[agent_id]
        
        for dep_id in config.dependencies:
            if dep_id not in self.agent_health:
                logger.error(f"Dependency {dep_id} not found for agent {agent_id}")
                return False
            
            dep_health = self.agent_health[dep_id]
            if dep_health.state not in [AgentLifecycleState.RUNNING, AgentLifecycleState.DEGRADED]:
                logger.error(f"Dependency {dep_id} not running for agent {agent_id}")
                return False
        
        return True
    
    async def _update_agent_health(self, agent_id: str):
        """Update health metrics for an agent"""
        if agent_id not in self.agents or agent_id not in self.agent_health:
            return
        
        agent = self.agents[agent_id]
        health = self.agent_health[agent_id]
        
        try:
            # Update basic metrics
            health.last_health_check = datetime.utcnow()
            health.dependencies_healthy = await self._check_agent_dependencies(agent_id)
            
            # Get agent-specific health if available
            if hasattr(agent, 'get_health'):
                agent_health = await agent.get_health()
                if isinstance(agent_health, dict):
                    health.health_score = agent_health.get('health_score', 1.0)
                    health.memory_usage_mb = agent_health.get('memory_usage_mb', 0.0)
                    health.cache_hit_rate = agent_health.get('cache_hit_rate', 0.0)
                    health.success_rate = agent_health.get('success_rate', 1.0)
                    health.avg_response_time = agent_health.get('avg_response_time', 0.0)
                elif isinstance(agent_health, (int, float)):
                    health.health_score = float(agent_health)
            
            # Get memory metrics if agent has memory
            if isinstance(agent, MemoryAgent):
                try:
                    memory_stats = await agent.get_memory_stats()
                    if memory_stats:
                        health.memory_entries = memory_stats.get('total_entries', 0)
                except Exception:
                    pass
            
            # Update state based on health
            if health.health_score < 0.3:
                await self._transition_agent_state(agent_id, AgentLifecycleState.ERROR)
            elif health.health_score < 0.7:
                await self._transition_agent_state(agent_id, AgentLifecycleState.DEGRADED)
            elif health.state in [AgentLifecycleState.ERROR, AgentLifecycleState.DEGRADED]:
                await self._transition_agent_state(agent_id, AgentLifecycleState.RUNNING)
            
        except Exception as e:
            logger.error(f"Health check failed for agent {agent_id}: {e}")
            health.error_count += 1
            health.health_score = 0.0
            await self._transition_agent_state(agent_id, AgentLifecycleState.ERROR)
    
    async def _handle_agent_error(self, agent_id: str, error: Exception):
        """Handle agent error and attempt recovery"""
        logger.error(f"Agent {agent_id} error: {error}")
        
        # Execute error hooks
        if agent_id in self.agents:
            await self._execute_hooks("on_agent_error", self.agents[agent_id], error)
        
        # Attempt restart if configured
        config = self.agent_configs[agent_id]
        if config.restart_on_failure:
            logger.info(f"Attempting to restart agent {agent_id}")
            await self.restart_agent(agent_id)
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while not self.shutdown_requested:
            try:
                # Check all agents
                for agent_id in list(self.agents.keys()):
                    await self._update_agent_health(agent_id)
                
                # Execute health check hooks
                await self._execute_hooks("on_health_check")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self.shutdown_requested:
            try:
                # Clean up old events
                if len(self.event_history) > self.max_event_history:
                    self.event_history = self.event_history[-self.max_event_history:]
                
                # Clean up dead agent references
                dead_agents = []
                for agent_id, ref in self.agent_refs.items():
                    if ref() is None:
                        dead_agents.append(agent_id)
                
                for agent_id in dead_agents:
                    logger.warning(f"Agent {agent_id} was garbage collected")
                    if agent_id in self.agents:
                        del self.agents[agent_id]
                    del self.agent_refs[agent_id]
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    async def _event_processor(self):
        """Process lifecycle events"""
        while not self.shutdown_requested:
            try:
                # Process events (placeholder for future event processing)
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(1.0)
    
    async def _memory_coordinator(self):
        """Coordinate memory operations across agents"""
        while not self.shutdown_requested:
            try:
                # Coordinate memory cleanup across agents
                for agent_id, agent in self.agents.items():
                    if isinstance(agent, MemoryAgent):
                        try:
                            # Check memory usage and cleanup if needed
                            stats = await agent.get_memory_stats()
                            if stats and stats.get('total_entries', 0) > 1000:
                                logger.info(f"Triggering memory cleanup for agent {agent_id}")
                                # Could implement memory cleanup strategies here
                        except Exception as e:
                            logger.warning(f"Memory coordination failed for agent {agent_id}: {e}")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in memory coordinator: {e}")
                await asyncio.sleep(300)
    
    async def _transition_manager_state(self, new_state: AgentLifecycleState):
        """Transition manager state"""
        old_state = self.manager_state
        self.manager_state = new_state
        
        self._record_event("lifecycle_manager", "state_transition", old_state, new_state)
        logger.info(f"MCP Agent Lifecycle Manager: {old_state.value} -> {new_state.value}")
    
    async def _transition_agent_state(self, agent_id: str, new_state: AgentLifecycleState):
        """Transition agent state"""
        if agent_id not in self.agent_health:
            return
        
        health = self.agent_health[agent_id]
        old_state = health.state
        health.state = new_state
        
        self._record_event(agent_id, "state_transition", old_state, new_state)
        logger.debug(f"Agent {agent_id}: {old_state.value} -> {new_state.value}")
    
    def _record_event(self, agent_id: str, event_type: str, 
                     from_state: Optional[AgentLifecycleState], 
                     to_state: AgentLifecycleState,
                     metadata: Dict[str, Any] = None,
                     error_message: Optional[str] = None):
        """Record a lifecycle event"""
        event = LifecycleEvent(
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            metadata=metadata or {},
            error_message=error_message
        )
        
        self.event_history.append(event)
    
    async def _execute_hooks(self, hook_name: str, *args):
        """Execute lifecycle hooks"""
        if hook_name in self.lifecycle_hooks:
            for callback in self.lifecycle_hooks[hook_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args)
                    else:
                        callback(*args)
                except Exception as e:
                    logger.error(f"Error executing {hook_name} hook: {e}")
    
    def add_lifecycle_hook(self, hook_name: str, callback: Callable):
        """Add a lifecycle hook"""
        if hook_name in self.lifecycle_hooks:
            self.lifecycle_hooks[hook_name].append(callback)
            logger.debug(f"Added lifecycle hook: {hook_name}")
        else:
            logger.warning(f"Unknown hook name: {hook_name}")

# Global lifecycle manager instance
_mcp_agent_lifecycle_manager: Optional[MCPAgentLifecycleManager] = None

def get_mcp_agent_lifecycle_manager() -> MCPAgentLifecycleManager:
    """Get the global MCP agent lifecycle manager instance"""
    global _mcp_agent_lifecycle_manager
    if _mcp_agent_lifecycle_manager is None:
        _mcp_agent_lifecycle_manager = MCPAgentLifecycleManager()
    return _mcp_agent_lifecycle_manager

@asynccontextmanager
async def managed_mcp_agents():
    """Context manager for MCP agent lifecycle management"""
    manager = get_mcp_agent_lifecycle_manager()
    
    try:
        await manager.start()
        yield manager
    finally:
        await manager.shutdown()
