"""
Scalable A2A Workflow Engine
Handles multiple concurrent workflows with full A2A compliance
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from collections import defaultdict
import time

from ..protocols import A2AMessage, A2AProtocol, MessageType
from ..blockchain.workflow_instance_contract import WorkflowInstanceManager
from ..blockchain.blockchain_signatures import A2AMessageSigner
from ..registry.registry import agent_registry
from .state_manager import state_manager
from .message_queue import workflow_queue
from .distributed_lock import WorkflowLock
from .observability import observability, trace_step_execution

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    step_id: str
    agent_id: str
    action: str
    input_data: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    retry_count: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    message_template: Optional[Dict[str, Any]] = None  # A2A message template
    response_type: Optional[str] = None  # Expected response message type

@dataclass
class WorkflowDefinition:
    """Defines a reusable workflow template"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deprecated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [
                {
                    "step_id": s.step_id,
                    "agent_id": s.agent_id,
                    "action": s.action,
                    "depends_on": s.depends_on
                } for s in self.steps
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deprecated": self.deprecated
        }

class WorkflowEngine:
    """Scalable workflow execution engine with A2A compliance and blockchain signatures"""
    
    def __init__(self, max_concurrent_workflows: int = 10, w3=None, contract_address=None, deployer_key=None):
        self.max_concurrent = max_concurrent_workflows
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_registry: Dict[str, WorkflowDefinition] = {}
        self.execution_queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.message_router = None  # Will be injected
        self.metrics = defaultdict(int)
        
        # Blockchain components
        self.w3 = w3
        self.contract_address = contract_address
        self.workflow_instance_manager = None
        if w3 and contract_address and deployer_key:
            self.workflow_instance_manager = WorkflowInstanceManager(w3, contract_address, deployer_key)
        
    def register_workflow(self, workflow_def: WorkflowDefinition):
        """Register a reusable workflow definition"""
        self.workflow_registry[workflow_def.workflow_id] = workflow_def
        logger.info(f"Registered workflow: {workflow_def.name} ({workflow_def.workflow_id})")
    
    async def start(self):
        """Start workflow engine workers"""
        logger.info(f"Starting workflow engine with {self.max_concurrent} workers")
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._workflow_worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def stop(self):
        """Stop workflow engine"""
        logger.info("Stopping workflow engine")
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> str:
        """Execute a workflow asynchronously with blockchain instance deployment"""
        if workflow_id not in self.workflow_registry:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        # Create execution instance
        execution_id = execution_id or str(uuid.uuid4())
        workflow = self._create_workflow_instance(
            self.workflow_registry[workflow_id],
            input_data,
            execution_id
        )
        
        # Deploy workflow instance contract if blockchain is configured
        if self.workflow_instance_manager:
            try:
                instance_address = await self.workflow_instance_manager.deploy_instance(
                    workflow_id, execution_id
                )
                workflow.metadata["instance_address"] = instance_address
                logger.info(f"Deployed workflow instance contract at {instance_address}")
            except Exception as e:
                logger.error(f"Failed to deploy workflow instance contract: {e}")
        
        # Add to execution queue
        await self.execution_queue.put(workflow)
        self.metrics["workflows_queued"] += 1
        
        logger.info(f"Queued workflow {workflow.name} for execution (ID: {execution_id})")
        return execution_id
    
    async def _workflow_worker(self, worker_id: str):
        """Worker task that executes workflows"""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get workflow from queue
                workflow = await self.execution_queue.get()
                
                logger.info(f"Worker {worker_id} executing workflow {workflow.workflow_id}")
                self.metrics["workflows_started"] += 1
                
                # Execute workflow
                await self._execute_workflow_steps(workflow)
                
                self.metrics["workflows_completed"] += 1
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.metrics["workflows_failed"] += 1
    
    async def _execute_workflow_steps(self, workflow: WorkflowDefinition):
        """Execute all steps in a workflow with dependency management"""
        execution_id = workflow.metadata.get("execution_id", str(uuid.uuid4()))
        
        # Store workflow state
        await state_manager.set(
            f"workflow:state:{execution_id}",
            {"status": "running", "started_at": time.time()},
            ttl=3600
        )
        
        # Acquire workflow lock
        workflow_lock = await WorkflowLock.acquire_workflow_lock(
            workflow.workflow_id,
            execution_id,
            timeout=60
        )
        
        if not workflow_lock:
            raise RuntimeError(f"Failed to acquire lock for workflow {workflow.workflow_id}")
        
        try:
            with observability.trace_workflow(workflow.workflow_id, execution_id):
                completed_steps = set()
                
                while True:
                    # Find steps that can be executed
                    ready_steps = [
                        step for step in workflow.steps
                        if step.status == WorkflowStatus.PENDING
                        and all(dep in completed_steps for dep in step.depends_on)
                    ]
            
                    if not ready_steps:
                        # Check if all steps are complete
                        if all(s.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] 
                               for s in workflow.steps):
                            break
                        
                        # Wait for dependencies
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Execute ready steps in parallel
                    tasks = []
                    for step in ready_steps:
                        task = asyncio.create_task(self._execute_step(step, workflow))
                        tasks.append((step, task))
                    
                    # Wait for steps to complete
                    for step, task in tasks:
                        try:
                            await task
                            if step.status == WorkflowStatus.COMPLETED:
                                completed_steps.add(step.step_id)
                                # Update state in distributed store
                                await state_manager.set(
                                    f"workflow:step:{execution_id}:{step.step_id}",
                                    {"status": "completed", "completed_at": time.time()},
                                    ttl=3600
                                )
                        except Exception as e:
                            logger.error(f"Step {step.step_id} failed: {e}")
                            step.status = WorkflowStatus.FAILED
                            step.error = str(e)
                            observability.error_counter.add(1, {"step_id": step.step_id})
        
        finally:
            # Release workflow lock
            await workflow_lock.release()
            
            # Update final state
            await state_manager.set(
                f"workflow:state:{execution_id}",
                {
                    "status": "completed",
                    "completed_at": time.time(),
                    "duration_ms": (time.time() - workflow.metadata.get("started_at", time.time())) * 1000
                },
                ttl=86400  # 24 hours
            )
    
    async def _execute_step(self, step: WorkflowStep, workflow: WorkflowDefinition):
        """Execute a single workflow step"""
        step.status = WorkflowStatus.RUNNING
        step.started_at = datetime.now()
        
        for attempt in range(step.retry_count):
            try:
                # Create A2A message with workflow context
                message = A2AProtocol.create_message(
                    sender_id="workflow-engine",
                    receiver_id=step.agent_id,
                    message_type=MessageType.WORKFLOW_REQUEST,
                    payload={
                        "action": step.action,
                        "input": step.input_data
                    }
                )
                
                # Get workflow instance address if available
                instance_address = workflow.metadata.get("instance_address")
                
                # Add workflow context to message with instance address
                message.workflow_context = {
                    "workflow_id": workflow.workflow_id,
                    "workflow_name": workflow.name,
                    "step_id": step.step_id,
                    "step_action": step.action,
                    "execution_id": workflow.metadata.get("execution_id"),
                    "instance_address": instance_address,
                    "attempt": attempt + 1,
                    "total_steps": len(workflow.steps),
                    "dependencies": step.depends_on
                }
                
                # Add blockchain context if available
                if self.w3 and self.contract_address:
                    message.blockchain_context = {
                        "chain_id": self.w3.eth.chain_id,
                        "contract_address": self.contract_address,
                        "instance_address": instance_address
                    }
                
                # Route message to agent
                if self.message_router:
                    result = await self.message_router.route_message(message)
                else:
                    # Direct agent call fallback
                    result = await self._direct_agent_call(step)
                
                if result.get("success"):
                    step.status = WorkflowStatus.COMPLETED
                    step.result = result
                    step.completed_at = datetime.now()
                    return
                else:
                    raise Exception(result.get("error", "Unknown error"))
                    
            except asyncio.TimeoutError:
                logger.warning(f"Step {step.step_id} timeout (attempt {attempt + 1})")
                if attempt == step.retry_count - 1:
                    step.status = WorkflowStatus.FAILED
                    step.error = "Timeout"
            except Exception as e:
                logger.error(f"Step {step.step_id} error (attempt {attempt + 1}): {e}")
                if attempt == step.retry_count - 1:
                    step.status = WorkflowStatus.FAILED
                    step.error = str(e)
            
            # Wait before retry
            if attempt < step.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _direct_agent_call(self, step: WorkflowStep) -> Dict[str, Any]:
        """Fallback - should never be used in A2A compliant system"""
        # A2A compliance requires all communication through messages
        logger.error(f"A2A Compliance Violation: Attempted direct agent call for step {step.step_id}")
        return {
            "success": False, 
            "error": "A2A compliance violation: All agent communication must use A2A messages",
            "a2a_compliant": False
        }
    
    def _create_workflow_instance(
        self, 
        template: WorkflowDefinition,
        input_data: Dict[str, Any],
        execution_id: str
    ) -> WorkflowDefinition:
        """Create a workflow instance from template with input data"""
        import copy
        
        instance = copy.deepcopy(template)
        instance.workflow_id = execution_id
        
        # Inject input data into steps
        for step in instance.steps:
            # Replace placeholders in input_data
            step.input_data = self._inject_data(step.input_data, input_data)
        
        return instance
    
    def _inject_data(self, template: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject actual data into template placeholders"""
        result = {}
        
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Replace placeholder
                placeholder = value[2:-1]
                result[key] = data.get(placeholder, value)
            elif isinstance(value, dict):
                result[key] = self._inject_data(value, data)
            else:
                result[key] = value
        
        return result
    
    def get_metrics(self) -> Dict[str, int]:
        """Get workflow engine metrics"""
        return dict(self.metrics)

# Pre-defined workflow templates
def create_eth_loading_workflow() -> WorkflowDefinition:
    """Create ETH data loading workflow definition"""
    return WorkflowDefinition(
        workflow_id="eth-data-loading",
        name="ETH Data Loading Workflow",
        description="Load ETH data from Yahoo Finance into database with AI analysis",
        steps=[
            WorkflowStep(
                step_id="discover-structure",
                agent_id="data-management-001",
                action="discover_data_structure",
                input_data={
                    "source": "yahoo",
                    "config": {"symbol": "${symbol}"}
                }
            ),
            WorkflowStep(
                step_id="load-historical",
                agent_id="historical-loader-001",
                action="load_symbol_data",
                input_data={
                    "symbol": "${symbol}",
                    "days_back": "${days_back}"
                },
                depends_on=["discover-structure"]
            ),
            WorkflowStep(
                step_id="store-and-analyze",
                agent_id="database-001",
                action="store_historical_data",
                input_data={
                    "symbol": "${symbol}",
                    "storage_type": "sqlite",
                    "ai_analysis": True
                },
                depends_on=["load-historical"]
            )
        ]
    )

def create_multi_symbol_workflow() -> WorkflowDefinition:
    """Create multi-symbol parallel loading workflow"""
    return WorkflowDefinition(
        workflow_id="multi-symbol-loading",
        name="Multi-Symbol Parallel Loading",
        description="Load multiple symbols in parallel with analysis",
        steps=[
            # These steps would be dynamically generated based on symbols
            # This is just a template
        ]
    )