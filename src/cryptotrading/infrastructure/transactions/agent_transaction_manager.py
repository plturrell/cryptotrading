"""
Agent Transaction Manager
Provides comprehensive transaction boundary management for agent operations
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
import json
import logging

logger = logging.getLogger(__name__)


class TransactionState(Enum):
    """Transaction states"""
    PENDING = "pending"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class TransactionIsolation(Enum):
    """Transaction isolation levels"""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


@dataclass
class TransactionOperation:
    """Represents a single operation within a transaction"""
    operation_id: str
    operation_type: str  # CREATE, UPDATE, DELETE, CALL_ACTION, etc.
    resource_type: str  # agent, data, model, etc.
    resource_id: str
    operation_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    rollback_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "operation_data": self.operation_data,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "rollback_data": self.rollback_data
        }


@dataclass
class AgentTransaction:
    """Represents a transaction for agent operations"""
    transaction_id: str
    agent_id: str
    transaction_type: str  # AGENT_OPERATION, CDS_BATCH, ML_PIPELINE, etc.
    isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED
    timeout_seconds: int = 300  # 5 minutes default
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    state: TransactionState = TransactionState.PENDING
    operations: List[TransactionOperation] = field(default_factory=list)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    rollback_reason: Optional[str] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return datetime.now() - self.started_at
        return None
    
    @property
    def is_expired(self) -> bool:
        if self.started_at:
            return datetime.now() > (self.started_at + timedelta(seconds=self.timeout_seconds))
        return False
    
    def add_operation(self, operation: TransactionOperation) -> None:
        """Add operation to transaction"""
        self.operations.append(operation)
        logger.debug(f"Added operation {operation.operation_id} to transaction {self.transaction_id}")
    
    def create_checkpoint(self, checkpoint_name: str, data: Dict[str, Any]) -> None:
        """Create a checkpoint in the transaction"""
        checkpoint = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "operation_count": len(self.operations),
            "data": data
        }
        self.checkpoints.append(checkpoint)
        logger.debug(f"Created checkpoint {checkpoint_name} in transaction {self.transaction_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "agent_id": self.agent_id,
            "transaction_type": self.transaction_type,
            "isolation_level": self.isolation_level.value,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "state": self.state.value,
            "operations": [op.to_dict() for op in self.operations],
            "checkpoints": self.checkpoints,
            "metadata": self.metadata,
            "error_details": self.error_details,
            "rollback_reason": self.rollback_reason,
            "duration_ms": self.duration.total_seconds() * 1000 if self.duration else None
        }


class TransactionManager:
    """Manages transactions for agent operations"""
    
    def __init__(self):
        self.active_transactions: Dict[str, AgentTransaction] = {}
        self.completed_transactions: List[AgentTransaction] = []
        self.max_completed_history = 1000
        self.cleanup_interval = 3600  # 1 hour
        self.rollback_handlers: Dict[str, Callable] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the transaction manager"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_transactions())
        logger.info("Transaction manager started")
    
    async def stop(self) -> None:
        """Stop the transaction manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Rollback all active transactions
        for transaction in list(self.active_transactions.values()):
            await self._rollback_transaction(transaction, "System shutdown")
        
        logger.info("Transaction manager stopped")
    
    def register_rollback_handler(self, operation_type: str, handler: Callable) -> None:
        """Register a rollback handler for specific operation types"""
        self.rollback_handlers[operation_type] = handler
        logger.debug(f"Registered rollback handler for operation type: {operation_type}")
    
    async def begin_transaction(
        self,
        agent_id: str,
        transaction_type: str = "AGENT_OPERATION",
        isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
        timeout_seconds: int = 300,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentTransaction:
        """Begin a new transaction"""
        async with self._lock:
            transaction_id = f"tx_{agent_id}_{uuid.uuid4().hex[:8]}"
            
            transaction = AgentTransaction(
                transaction_id=transaction_id,
                agent_id=agent_id,
                transaction_type=transaction_type,
                isolation_level=isolation_level,
                timeout_seconds=timeout_seconds,
                metadata=metadata or {}
            )
            
            transaction.state = TransactionState.ACTIVE
            transaction.started_at = datetime.now()
            
            self.active_transactions[transaction_id] = transaction
            
            logger.info(f"Started transaction {transaction_id} for agent {agent_id}")
            return transaction
    
    async def add_operation(
        self,
        transaction_id: str,
        operation_type: str,
        resource_type: str,
        resource_id: str,
        operation_data: Dict[str, Any],
        rollback_data: Optional[Dict[str, Any]] = None
    ) -> TransactionOperation:
        """Add an operation to a transaction"""
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found or not active")
            
            transaction = self.active_transactions[transaction_id]
            
            if transaction.is_expired:
                await self._rollback_transaction(transaction, "Transaction timeout")
                raise RuntimeError(f"Transaction {transaction_id} has expired")
            
            operation = TransactionOperation(
                operation_id=f"op_{uuid.uuid4().hex[:8]}",
                operation_type=operation_type,
                resource_type=resource_type,
                resource_id=resource_id,
                operation_data=operation_data,
                rollback_data=rollback_data
            )
            
            transaction.add_operation(operation)
            return operation
    
    async def create_checkpoint(
        self,
        transaction_id: str,
        checkpoint_name: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a checkpoint in a transaction"""
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found or not active")
            
            transaction = self.active_transactions[transaction_id]
            transaction.create_checkpoint(checkpoint_name, data or {})
    
    async def add_checkpoint(self, transaction_id: str, checkpoint_name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add a checkpoint to a transaction"""
        async with self._lock:
            if transaction_id in self.active_transactions:
                transaction = self.active_transactions[transaction_id]
                transaction.create_checkpoint(checkpoint_name, data or {})
                logger.debug(f"Added checkpoint '{checkpoint_name}' to transaction {transaction_id}")
            else:
                raise ValueError(f"Transaction {transaction_id} not found or not active")
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction"""
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found or not active")
            
            transaction = self.active_transactions[transaction_id]
            
            try:
                # Execute all pending operations
                for operation in transaction.operations:
                    if operation.status == "pending":
                        await self._execute_operation(operation)
                        operation.status = "completed"
                
                # Mark transaction as committed
                transaction.state = TransactionState.COMMITTED
                transaction.completed_at = datetime.now()
                
                # Move to completed transactions
                self.completed_transactions.append(transaction)
                del self.active_transactions[transaction_id]
                
                # Trim history if needed
                if len(self.completed_transactions) > self.max_completed_history:
                    self.completed_transactions = self.completed_transactions[-self.max_completed_history:]
                
                logger.info(f"Committed transaction {transaction_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to commit transaction {transaction_id}: {e}")
                await self._rollback_transaction(transaction, f"Commit failed: {str(e)}")
                return False
    
    async def rollback_transaction(self, transaction_id: str, reason: str = "Manual rollback") -> bool:
        """Rollback a transaction"""
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found or not active")
            
            transaction = self.active_transactions[transaction_id]
            return await self._rollback_transaction(transaction, reason)
    
    async def _rollback_transaction(self, transaction: AgentTransaction, reason: str) -> bool:
        """Internal rollback implementation"""
        try:
            # Rollback operations in reverse order
            for operation in reversed(transaction.operations):
                if operation.status == "completed":
                    await self._rollback_operation(operation)
            
            transaction.state = TransactionState.ROLLED_BACK
            transaction.completed_at = datetime.now()
            transaction.rollback_reason = reason
            
            # Move to completed transactions
            self.completed_transactions.append(transaction)
            if transaction.transaction_id in self.active_transactions:
                del self.active_transactions[transaction.transaction_id]
            
            logger.info(f"Rolled back transaction {transaction.transaction_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback transaction {transaction.transaction_id}: {e}")
            transaction.state = TransactionState.FAILED
            transaction.error_details = str(e)
            transaction.completed_at = datetime.now()
            
            # Move to completed transactions even if rollback failed
            self.completed_transactions.append(transaction)
            if transaction.transaction_id in self.active_transactions:
                del self.active_transactions[transaction.transaction_id]
            
            return False
    
    async def _execute_operation(self, operation: TransactionOperation) -> None:
        """Execute a transaction operation"""
        logger.debug(f"Executing operation {operation.operation_id}: {operation.operation_type}")
        
        # This is where actual operation execution would happen
        # For now, we'll just mark as completed
        # In a real implementation, this would call the appropriate handlers
        
        if operation.operation_type in ["CREATE", "UPDATE", "DELETE"]:
            # Simulate database operation
            await asyncio.sleep(0.01)  # Simulate I/O
        elif operation.operation_type == "CALL_ACTION":
            # Simulate CDS action call
            await asyncio.sleep(0.02)  # Simulate network call
        elif operation.operation_type == "ML_PREDICTION":
            # Simulate ML operation
            await asyncio.sleep(0.05)  # Simulate computation
        
        logger.debug(f"Completed operation {operation.operation_id}")
    
    async def _rollback_operation(self, operation: TransactionOperation) -> None:
        """Rollback a single operation"""
        logger.debug(f"Rolling back operation {operation.operation_id}")
        
        # Use registered rollback handler if available
        if operation.operation_type in self.rollback_handlers:
            handler = self.rollback_handlers[operation.operation_type]
            await handler(operation)
        else:
            # Default rollback logic based on operation type
            if operation.operation_type == "CREATE" and operation.rollback_data:
                # Delete created resource
                logger.debug(f"Rolling back CREATE by deleting {operation.resource_id}")
            elif operation.operation_type == "UPDATE" and operation.rollback_data:
                # Restore original data
                logger.debug(f"Rolling back UPDATE by restoring {operation.resource_id}")
            elif operation.operation_type == "DELETE" and operation.rollback_data:
                # Recreate deleted resource
                logger.debug(f"Rolling back DELETE by recreating {operation.resource_id}")
        
        operation.status = "rolled_back"
    
    async def _cleanup_expired_transactions(self) -> None:
        """Background task to clean up expired transactions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                expired_transactions = []
                async with self._lock:
                    for transaction_id, transaction in list(self.active_transactions.items()):
                        if transaction.is_expired:
                            expired_transactions.append(transaction)
                
                # Rollback expired transactions
                for transaction in expired_transactions:
                    await self._rollback_transaction(transaction, "Transaction timeout")
                
                if expired_transactions:
                    logger.info(f"Cleaned up {len(expired_transactions)} expired transactions")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during transaction cleanup: {e}")
    
    def get_transaction(self, transaction_id: str) -> Optional[AgentTransaction]:
        """Get transaction by ID"""
        return self.active_transactions.get(transaction_id)
    
    def get_agent_transactions(self, agent_id: str) -> List[AgentTransaction]:
        """Get all transactions for an agent"""
        active = [tx for tx in self.active_transactions.values() if tx.agent_id == agent_id]
        completed = [tx for tx in self.completed_transactions if tx.agent_id == agent_id]
        return active + completed
    
    def get_transaction_stats(self) -> Dict[str, Any]:
        """Get transaction statistics"""
        active_count = len(self.active_transactions)
        completed_count = len(self.completed_transactions)
        
        # Count by state
        state_counts = {}
        for tx in self.completed_transactions:
            state = tx.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Calculate average duration
        completed_with_duration = [tx for tx in self.completed_transactions if tx.duration]
        avg_duration = (
            sum(tx.duration.total_seconds() for tx in completed_with_duration) / 
            len(completed_with_duration)
        ) if completed_with_duration else 0
        
        return {
            "active_transactions": active_count,
            "completed_transactions": completed_count,
            "total_transactions": active_count + completed_count,
            "state_counts": state_counts,
            "average_duration_seconds": avg_duration,
            "expired_transactions": sum(1 for tx in self.active_transactions.values() if tx.is_expired)
        }


# Context manager for agent transactions
@asynccontextmanager
async def agent_transaction(
    agent_id: str,
    transaction_manager: TransactionManager,
    transaction_type: str = "AGENT_OPERATION",
    isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
    timeout_seconds: int = 300,
    metadata: Optional[Dict[str, Any]] = None
):
    """Context manager for agent transactions"""
    transaction = await transaction_manager.begin_transaction(
        agent_id=agent_id,
        transaction_type=transaction_type,
        isolation_level=isolation_level,
        timeout_seconds=timeout_seconds,
        metadata=metadata
    )
    
    try:
        yield transaction
        # Commit on successful completion
        await transaction_manager.commit_transaction(transaction.transaction_id)
    except Exception as e:
        # Rollback on error
        await transaction_manager.rollback_transaction(
            transaction.transaction_id, 
            f"Exception during transaction: {str(e)}"
        )
        raise


# Global transaction manager
_transaction_manager = None


def get_transaction_manager() -> TransactionManager:
    """Get the global transaction manager"""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = TransactionManager()
    return _transaction_manager


# Decorator for transactional operations
def transactional(
    transaction_type: str = "AGENT_OPERATION",
    isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
    timeout_seconds: int = 300
):
    """Decorator to make agent methods transactional"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            agent_id = getattr(self, 'agent_id', 'unknown')
            transaction_manager = get_transaction_manager()
            
            async with agent_transaction(
                agent_id=agent_id,
                transaction_manager=transaction_manager,
                transaction_type=transaction_type,
                isolation_level=isolation_level,
                timeout_seconds=timeout_seconds,
                metadata={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            ) as transaction:
                # Add the transaction to the method arguments
                if 'transaction' in func.__code__.co_varnames:
                    kwargs['transaction'] = transaction
                
                return await func(self, *args, **kwargs)
        
        return wrapper
    return decorator