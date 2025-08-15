"""
A2A Orchestration Service
Provides high-level API for workflow execution with monitoring and scaling
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import json

from .workflow_engine import WorkflowEngine, WorkflowStatus
from .workflow_registry import workflow_registry
from ..agents.a2a_coordinator import A2ACoordinator
from ..protocols import MessageType, A2AProtocol
from .message_queue import workflow_queue, a2a_queue
from .state_manager import state_manager
from .observability import observability

logger = logging.getLogger(__name__)

@dataclass
class WorkflowExecution:
    """Tracks a workflow execution"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    input_data: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class OrchestrationService:
    """High-level orchestration service with monitoring and scaling"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.engine = WorkflowEngine(
            max_concurrent_workflows=self.config.get("max_concurrent", 10)
        )
        self.coordinator = A2ACoordinator()
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Metrics and monitoring
        self.metrics = defaultdict(int)
        self.performance_stats = defaultdict(list)
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Wire up the message router
        self.engine.message_router = self.coordinator
    
    async def start(self):
        """Start the orchestration service"""
        logger.info("Starting orchestration service")
        
        # Sync workflows from global registry to engine
        for workflow_id, workflow in workflow_registry.workflows.items():
            self.engine.register_workflow(workflow)
        
        # Start workflow engine
        await self.engine.start()
        
        # Start message queue processing
        await workflow_queue.start_processing()
        await a2a_queue.start_processing()
        
        # Register message processors
        workflow_queue.register_processor("workflow_request", self._process_workflow_request)
        a2a_queue.register_processor("a2a_message", self._process_a2a_message)
        
        # Start monitoring
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Orchestration service started")
    
    async def stop(self):
        """Stop the orchestration service"""
        logger.info("Stopping orchestration service")
        
        # Stop message queues
        await workflow_queue.stop_processing()
        await a2a_queue.stop_processing()
        
        # Cancel background tasks
        if self.monitor_task:
            self.monitor_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Stop engine
        await self.engine.stop()
        
        logger.info("Orchestration service stopped")
    
    async def _process_workflow_request(self, message: Dict[str, Any]):
        """Process workflow request from queue"""
        workflow_id = message.get("workflow_id")
        input_data = message.get("input_data", {})
        
        try:
            execution_id = await self.execute_workflow(workflow_id, input_data)
            logger.info(f"Started workflow {workflow_id} from queue (execution: {execution_id})")
        except Exception as e:
            logger.error(f"Failed to start workflow from queue: {e}")
            observability.error_counter.add(1, {"source": "queue", "error": str(e)})
    
    async def _process_a2a_message(self, message: Dict[str, Any]):
        """Process A2A message from queue"""
        try:
            # Route through coordinator
            a2a_msg = A2AProtocol.create_message(
                sender_id=message.get("sender_id"),
                receiver_id=message.get("receiver_id"),
                message_type=MessageType(message.get("message_type")),
                payload=message.get("payload", {})
            )
            
            result = await self.coordinator.route_message(a2a_msg)
            observability.message_counter.add(1, {"status": "processed"})
        except Exception as e:
            logger.error(f"Failed to process A2A message: {e}")
            observability.error_counter.add(1, {"source": "a2a", "error": str(e)})
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        priority: int = 5
    ) -> str:
        """Execute a workflow with the given input data"""
        # Validate workflow exists
        workflow = workflow_registry.get(workflow_id)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        # Create execution record
        execution_id = await self.engine.execute_workflow(workflow_id, input_data)
        
        self.executions[execution_id] = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            input_data=input_data,
            started_at=datetime.now()
        )
        
        self.metrics["workflows_submitted"] += 1
        
        logger.info(f"Submitted workflow {workflow_id} for execution (ID: {execution_id})")
        return execution_id
    
    async def execute_batch(
        self,
        workflow_id: str,
        batch_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Execute multiple instances of a workflow in batch"""
        execution_ids = []
        
        for data in batch_data:
            try:
                exec_id = await self.execute_workflow(workflow_id, data)
                execution_ids.append(exec_id)
            except Exception as e:
                logger.error(f"Failed to submit batch item: {e}")
                execution_ids.append(None)
        
        self.metrics["batch_submissions"] += 1
        self.metrics["batch_items_total"] += len(batch_data)
        
        return execution_ids
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow execution"""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration": (
                (execution.completed_at - execution.started_at).total_seconds()
                if execution.completed_at else None
            ),
            "result": execution.result,
            "error": execution.error
        }
    
    async def wait_for_completion(
        self,
        execution_id: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Wait for a workflow execution to complete"""
        start_time = datetime.now()
        
        while True:
            execution = self.executions.get(execution_id)
            if not execution:
                raise ValueError(f"Unknown execution: {execution_id}")
            
            if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                return self.get_execution_status(execution_id)
            
            # Check timeout
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Execution {execution_id} timed out")
            
            await asyncio.sleep(0.5)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        engine_metrics = self.engine.get_metrics()
        
        return {
            "service": dict(self.metrics),
            "engine": engine_metrics,
            "active_executions": len([
                e for e in self.executions.values()
                if e.status == WorkflowStatus.RUNNING
            ]),
            "performance": {
                "avg_duration": (
                    sum(self.performance_stats["duration"]) / len(self.performance_stats["duration"])
                    if self.performance_stats["duration"] else 0
                ),
                "success_rate": (
                    self.metrics["workflows_completed"] / self.metrics["workflows_submitted"]
                    if self.metrics["workflows_submitted"] > 0 else 0
                )
            }
        }
    
    async def _monitor_loop(self):
        """Background monitoring task"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Update execution statuses
                for execution in self.executions.values():
                    if execution.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                        # Check actual status from engine
                        try:
                            engine_status = await self._check_execution_status_from_engine(execution.execution_id)
                            if engine_status and engine_status != execution.status:
                                old_status = execution.status
                                execution.status = engine_status
                                execution.updated_at = datetime.utcnow()
                                
                                logger.info(f"Updated execution {execution.execution_id} status from {old_status} to {engine_status}")
                                
                                # If execution is completed, get results
                                if engine_status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                                    execution_results = await self._get_execution_results_from_engine(execution.execution_id)
                                    if execution_results:
                                        execution.result = execution_results
                                        
                        except Exception as e:
                            logger.warning(f"Failed to check execution status from engine: {e}")
                
                # Log metrics
                metrics = self.get_metrics()
                logger.debug(f"Orchestration metrics: {json.dumps(metrics, indent=2)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    
    async def _check_execution_status_from_engine(self, execution_id: str) -> Optional[WorkflowStatus]:
        """Check execution status from the workflow engine"""
        try:
            # Try to get status from persistent storage first
            from ...database.client import get_db
            from ...database.models import A2AWorkflowExecution
            
            db = get_db()
            with db.get_session() as session:
                db_execution = session.query(A2AWorkflowExecution).filter_by(
                    execution_id=execution_id
                ).first()
                
                if db_execution:
                    # Map database status to WorkflowStatus
                    status_mapping = {
                        'pending': WorkflowStatus.PENDING,
                        'running': WorkflowStatus.RUNNING,
                        'completed': WorkflowStatus.COMPLETED,
                        'failed': WorkflowStatus.FAILED
                    }
                    return status_mapping.get(db_execution.status)
            
            # If not in database, check with workflow engine directly
            if hasattr(self, 'workflow_engine') and self.workflow_engine:
                return await self.workflow_engine.get_execution_status(execution_id)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking execution status from engine: {e}")
            return None
    
    async def _get_execution_results_from_engine(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution results from the workflow engine"""
        try:
            # Try to get results from persistent storage first
            from ...database.client import get_db
            from ...database.models import A2AWorkflowExecution
            
            db = get_db()
            with db.get_session() as session:
                db_execution = session.query(A2AWorkflowExecution).filter_by(
                    execution_id=execution_id
                ).first()
                
                if db_execution and db_execution.result_data:
                    try:
                        return json.loads(db_execution.result_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in execution results: {execution_id}")
            
            # If not in database, check with workflow engine directly
            if hasattr(self, 'workflow_engine') and self.workflow_engine:
                return await self.workflow_engine.get_execution_results(execution_id)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting execution results from engine: {e}")
            return None
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Clean up old executions
                cutoff = datetime.now() - timedelta(hours=24)
                to_remove = []
                
                for exec_id, execution in self.executions.items():
                    if execution.completed_at and execution.completed_at < cutoff:
                        to_remove.append(exec_id)
                
                for exec_id in to_remove:
                    del self.executions[exec_id]
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old executions")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

# Convenience functions for common workflows
class WorkflowTemplates:
    """Pre-configured workflow templates"""
    
    @staticmethod
    async def load_eth_data(
        service: OrchestrationService,
        days_back: int = 30
    ) -> str:
        """Load ETH data using the standard workflow"""
        return await service.execute_workflow(
            "eth-data-loading",
            {
                "symbol": "ETH-USD",
                "days_back": days_back
            }
        )
    
    @staticmethod
    async def load_multiple_symbols(
        service: OrchestrationService,
        symbols: List[str],
        days_back: int = 30
    ) -> List[str]:
        """Load multiple symbols in parallel"""
        batch_data = [
            {"symbol": symbol, "days_back": days_back}
            for symbol in symbols
        ]
        
        return await service.execute_batch("eth-data-loading", batch_data)
    
    @staticmethod
    async def run_analysis_pipeline(
        service: OrchestrationService,
        symbol: str
    ) -> str:
        """Run multi-AI analysis pipeline"""
        return await service.execute_workflow(
            "analysis-pipeline",
            {"symbol": symbol}
        )
    
    @staticmethod
    async def discover_all_sources(
        service: OrchestrationService,
        symbol: str = "ETH-USD"
    ) -> str:
        """Discover all available data sources"""
        return await service.execute_workflow(
            "data-discovery",
            {
                "symbol": symbol,
                "exchange": "binance", 
                "pair": "ETHUSDT"
            }
        )