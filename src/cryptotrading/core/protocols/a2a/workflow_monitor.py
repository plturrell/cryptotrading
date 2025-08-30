"""
Workflow Monitoring with On-Chain Events
Real-time monitoring and analytics for A2A workflows
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from web3 import Web3
from web3.contract import Contract

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Workflow event types"""
    DATA_STORED = "DataStored"
    DATA_ACCESSED = "DataAccessed"
    DATA_DELETED = "DataDeleted"
    WORKFLOW_CREATED = "WorkflowCreated"
    WORKFLOW_COMPLETED = "WorkflowCompleted"
    DATA_ACCESS_GRANTED = "DataAccessGranted"


@dataclass
class WorkflowEvent:
    """Workflow event data"""
    event_type: EventType
    workflow_id: Optional[int]
    data_id: Optional[int]
    agent_id: str
    timestamp: datetime
    block_number: int
    transaction_hash: str
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowMetrics:
    """Workflow performance metrics"""
    workflow_id: int
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: Optional[timedelta] = None
    average_step_duration: Optional[timedelta] = None
    data_packets_created: int = 0
    data_packets_accessed: int = 0
    total_data_size: int = 0
    participating_agents: Set[str] = field(default_factory=set)
    gas_used: int = 0
    
    def calculate_metrics(self):
        """Calculate derived metrics"""
        if self.start_time and self.end_time:
            self.total_duration = self.end_time - self.start_time
            
            if self.completed_steps > 0:
                self.average_step_duration = self.total_duration / self.completed_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "workflow_id": self.workflow_id,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "completion_rate": f"{(self.completed_steps / self.total_steps * 100):.1f}%" if self.total_steps > 0 else "0%",
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration.total_seconds() if self.total_duration else None,
            "average_step_duration_seconds": self.average_step_duration.total_seconds() if self.average_step_duration else None,
            "data_packets_created": self.data_packets_created,
            "data_packets_accessed": self.data_packets_accessed,
            "total_data_size": self.total_data_size,
            "participating_agents": list(self.participating_agents),
            "gas_used": self.gas_used
        }


class WorkflowMonitor:
    """Monitor and analyze on-chain workflow events"""
    
    def __init__(
        self,
        web3_provider: str = "http://localhost:8545",
        registry_contract_address: Optional[str] = None,
        data_exchange_contract_address: Optional[str] = None
    ):
        """Initialize workflow monitor"""
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.registry_address = registry_contract_address
        self.data_exchange_address = data_exchange_contract_address
        
        self.registry_contract: Optional[Contract] = None
        self.data_exchange_contract: Optional[Contract] = None
        
        # Event tracking
        self.events: List[WorkflowEvent] = []
        self.workflow_metrics: Dict[int, WorkflowMetrics] = {}
        self.agent_activity: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Event handlers
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Monitoring state
        self.monitoring_active = False
        self.last_block_processed = 0
        
        logger.info("WorkflowMonitor initialized")
    
    async def initialize(
        self,
        registry_abi: Optional[List[Dict[str, Any]]] = None,
        data_exchange_abi: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Initialize contracts for monitoring"""
        try:
            if self.registry_address and registry_abi:
                self.registry_contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(self.registry_address),
                    abi=registry_abi
                )
                logger.info(f"Connected to Registry contract at {self.registry_address}")
            
            if self.data_exchange_address and data_exchange_abi:
                self.data_exchange_contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(self.data_exchange_address),
                    abi=data_exchange_abi
                )
                logger.info(f"Connected to DataExchange contract at {self.data_exchange_address}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize contracts: {e}")
            return False
    
    async def start_monitoring(self, from_block: Optional[int] = None):
        """Start monitoring blockchain events"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.last_block_processed = from_block or self.w3.eth.block_number
        
        logger.info(f"Starting workflow monitoring from block {self.last_block_processed}")
        
        # Start event monitoring loop
        asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop monitoring blockchain events"""
        self.monitoring_active = False
        logger.info("Workflow monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                current_block = self.w3.eth.block_number
                
                if current_block > self.last_block_processed:
                    # Process new blocks
                    await self._process_blocks(
                        self.last_block_processed + 1,
                        current_block
                    )
                    self.last_block_processed = current_block
                
                # Wait before next check
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_blocks(self, from_block: int, to_block: int):
        """Process events from a range of blocks"""
        try:
            # Process DataExchange events
            if self.data_exchange_contract:
                await self._process_data_exchange_events(from_block, to_block)
            
            # Process Registry events
            if self.registry_contract:
                await self._process_registry_events(from_block, to_block)
                
        except Exception as e:
            logger.error(f"Error processing blocks {from_block}-{to_block}: {e}")
    
    async def _process_data_exchange_events(self, from_block: int, to_block: int):
        """Process DataExchange contract events"""
        try:
            # DataStored events
            data_stored_filter = self.data_exchange_contract.events.DataStored.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            for event in data_stored_filter.get_all_entries():
                await self._handle_data_stored(event)
            
            # DataAccessed events
            data_accessed_filter = self.data_exchange_contract.events.DataAccessed.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            for event in data_accessed_filter.get_all_entries():
                await self._handle_data_accessed(event)
            
            # WorkflowCreated events
            workflow_created_filter = self.data_exchange_contract.events.WorkflowCreated.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            for event in workflow_created_filter.get_all_entries():
                await self._handle_workflow_created(event)
            
            # WorkflowCompleted events
            workflow_completed_filter = self.data_exchange_contract.events.WorkflowCompleted.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            for event in workflow_completed_filter.get_all_entries():
                await self._handle_workflow_completed(event)
                
        except Exception as e:
            logger.error(f"Error processing DataExchange events: {e}")
    
    async def _process_registry_events(self, from_block: int, to_block: int):
        """Process Registry contract events"""
        # Similar processing for agent registry events
        pass
    
    async def _handle_data_stored(self, event):
        """Handle DataStored event"""
        try:
            args = event['args']
            
            workflow_event = WorkflowEvent(
                event_type=EventType.DATA_STORED,
                workflow_id=None,  # Could be extracted from data
                data_id=args['dataId'],
                agent_id=args['senderAgentId'],
                timestamp=datetime.fromtimestamp(args['timestamp']),
                block_number=event['blockNumber'],
                transaction_hash=event['transactionHash'].hex(),
                additional_data={
                    'receiver': args['receiverAgentId'],
                    'data_type': args['dataType'],
                    'data_size': args['dataSize'],
                    'data_hash': args['dataHash']
                }
            )
            
            self.events.append(workflow_event)
            
            # Update agent activity
            self.agent_activity[args['senderAgentId']]['data_stored'] += 1
            
            # Trigger handlers
            await self._trigger_handlers(EventType.DATA_STORED, workflow_event)
            
            logger.debug(f"DataStored: ID {args['dataId']} from {args['senderAgentId']}")
            
        except Exception as e:
            logger.error(f"Error handling DataStored event: {e}")
    
    async def _handle_data_accessed(self, event):
        """Handle DataAccessed event"""
        try:
            args = event['args']
            
            workflow_event = WorkflowEvent(
                event_type=EventType.DATA_ACCESSED,
                workflow_id=None,
                data_id=args['dataId'],
                agent_id=args['agentId'],
                timestamp=datetime.fromtimestamp(args['timestamp']),
                block_number=event['blockNumber'],
                transaction_hash=event['transactionHash'].hex()
            )
            
            self.events.append(workflow_event)
            
            # Update agent activity
            self.agent_activity[args['agentId']]['data_accessed'] += 1
            
            # Trigger handlers
            await self._trigger_handlers(EventType.DATA_ACCESSED, workflow_event)
            
        except Exception as e:
            logger.error(f"Error handling DataAccessed event: {e}")
    
    async def _handle_workflow_created(self, event):
        """Handle WorkflowCreated event"""
        try:
            args = event['args']
            workflow_id = args['workflowId']
            
            # Initialize metrics for this workflow
            self.workflow_metrics[workflow_id] = WorkflowMetrics(
                workflow_id=workflow_id,
                start_time=datetime.fromtimestamp(args['timestamp']),
                participating_agents=set(args['participants'])
            )
            
            workflow_event = WorkflowEvent(
                event_type=EventType.WORKFLOW_CREATED,
                workflow_id=workflow_id,
                data_id=None,
                agent_id="orchestrator",
                timestamp=datetime.fromtimestamp(args['timestamp']),
                block_number=event['blockNumber'],
                transaction_hash=event['transactionHash'].hex(),
                additional_data={
                    'workflow_type': args['workflowType'],
                    'participants': args['participants']
                }
            )
            
            self.events.append(workflow_event)
            
            # Trigger handlers
            await self._trigger_handlers(EventType.WORKFLOW_CREATED, workflow_event)
            
            logger.info(f"Workflow created: ID {workflow_id}, Type: {args['workflowType']}")
            
        except Exception as e:
            logger.error(f"Error handling WorkflowCreated event: {e}")
    
    async def _handle_workflow_completed(self, event):
        """Handle WorkflowCompleted event"""
        try:
            args = event['args']
            workflow_id = args['workflowId']
            
            # Update metrics
            if workflow_id in self.workflow_metrics:
                metrics = self.workflow_metrics[workflow_id]
                metrics.end_time = datetime.fromtimestamp(args['timestamp'])
                metrics.calculate_metrics()
            
            workflow_event = WorkflowEvent(
                event_type=EventType.WORKFLOW_COMPLETED,
                workflow_id=workflow_id,
                data_id=None,
                agent_id="orchestrator",
                timestamp=datetime.fromtimestamp(args['timestamp']),
                block_number=event['blockNumber'],
                transaction_hash=event['transactionHash'].hex(),
                additional_data={
                    'result': args.get('result', b'').hex() if args.get('result') else None
                }
            )
            
            self.events.append(workflow_event)
            
            # Trigger handlers
            await self._trigger_handlers(EventType.WORKFLOW_COMPLETED, workflow_event)
            
            logger.info(f"Workflow completed: ID {workflow_id}")
            
        except Exception as e:
            logger.error(f"Error handling WorkflowCompleted event: {e}")
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register a custom event handler"""
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type.value}")
    
    async def _trigger_handlers(self, event_type: EventType, event: WorkflowEvent):
        """Trigger registered event handlers"""
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_workflow_metrics(self, workflow_id: int) -> Optional[WorkflowMetrics]:
        """Get metrics for a specific workflow"""
        return self.workflow_metrics.get(workflow_id)
    
    def get_agent_activity(self, agent_id: str) -> Dict[str, int]:
        """Get activity statistics for an agent"""
        return dict(self.agent_activity.get(agent_id, {}))
    
    def get_recent_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[WorkflowEvent]:
        """Get recent events, optionally filtered by type"""
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        total_workflows = len(self.workflow_metrics)
        completed_workflows = sum(
            1 for m in self.workflow_metrics.values()
            if m.end_time is not None
        )
        
        total_data_stored = sum(
            1 for e in self.events
            if e.event_type == EventType.DATA_STORED
        )
        
        total_data_accessed = sum(
            1 for e in self.events
            if e.event_type == EventType.DATA_ACCESSED
        )
        
        # Calculate average workflow duration
        durations = [
            m.total_duration.total_seconds()
            for m in self.workflow_metrics.values()
            if m.total_duration
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Most active agents
        agent_scores = {
            agent: sum(activity.values())
            for agent, activity in self.agent_activity.items()
        }
        top_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "summary": {
                "total_workflows": total_workflows,
                "completed_workflows": completed_workflows,
                "completion_rate": f"{(completed_workflows / total_workflows * 100):.1f}%" if total_workflows > 0 else "0%",
                "total_events": len(self.events),
                "total_data_stored": total_data_stored,
                "total_data_accessed": total_data_accessed,
                "average_workflow_duration_seconds": avg_duration
            },
            "top_agents": [
                {"agent_id": agent, "activity_score": score}
                for agent, score in top_agents
            ],
            "workflow_metrics": [
                metrics.to_dict()
                for metrics in self.workflow_metrics.values()
            ],
            "event_distribution": {
                event_type.value: sum(1 for e in self.events if e.event_type == event_type)
                for event_type in EventType
            }
        }


# Singleton instance
_monitor: Optional[WorkflowMonitor] = None


async def get_workflow_monitor() -> WorkflowMonitor:
    """Get or create the workflow monitor singleton"""
    global _monitor
    
    if _monitor is None:
        _monitor = WorkflowMonitor()
        await _monitor.initialize()
    
    return _monitor