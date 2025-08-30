"""
CDS Transactional Client
Enhanced CDS client with transaction boundary management
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.protocols.cds import CDSClient, CDSServiceConfig, CDSTransaction
from .agent_transaction_manager import (
    get_transaction_manager,
    AgentTransaction,
    TransactionOperation,
    TransactionIsolation
)
from ..monitoring.cds_integration_monitor import (
    get_cds_monitor,
    CDSOperationType,
    CDSIntegrationStatus
)
from ..monitoring.logger import get_logger

logger = get_logger(__name__)


class TransactionalCDSClient(CDSClient):
    """Enhanced CDS client with transaction boundary management"""
    
    def __init__(self, config: CDSServiceConfig):
        super().__init__(config)
        self.transaction_manager = get_transaction_manager()
        self.cds_monitor = get_cds_monitor()
        self.agent_transaction: Optional[AgentTransaction] = None
        
    async def connect(self, agent_id: str = None) -> None:
        """Connect to CDS service with monitoring"""
        actual_agent_id = agent_id or "unknown"
        self.cds_monitor.register_agent(actual_agent_id)
        self.cds_monitor.update_agent_status(actual_agent_id, CDSIntegrationStatus.CONNECTING)
        
        try:
            async with self.cds_monitor.track_operation(actual_agent_id, CDSOperationType.CONNECTION) as operation:
                await super().connect(agent_id)
                operation.method_used = "CDS"
                self.cds_monitor.update_agent_status(actual_agent_id, CDSIntegrationStatus.CONNECTED)
        except Exception as e:
            self.cds_monitor.update_agent_status(actual_agent_id, CDSIntegrationStatus.ERROR)
            raise
    
    @asynccontextmanager
    async def transactional_operation(
        self,
        agent_id: str,
        operation_type: str,
        transaction_type: str = "CDS_OPERATION",
        timeout_seconds: int = 300
    ):
        """Context manager for transactional CDS operations"""
        # Start agent transaction
        self.agent_transaction = await self.transaction_manager.begin_transaction(
            agent_id=agent_id,
            transaction_type=transaction_type,
            timeout_seconds=timeout_seconds,
            metadata={
                "operation_type": operation_type,
                "cds_service": True
            }
        )
        
        # Start CDS transaction
        cds_transaction = CDSTransaction(self)
        
        try:
            # Track the operation with monitoring
            async with self.cds_monitor.track_operation(agent_id, CDSOperationType(operation_type)) as monitor_op:
                async with cds_transaction:
                    monitor_op.method_used = "CDS"
                    monitor_op.transaction_id = self.agent_transaction.transaction_id
                    
                    yield {
                        "agent_transaction": self.agent_transaction,
                        "cds_transaction": cds_transaction,
                        "client": self
                    }
                    
                    # Commit agent transaction on success
                    await self.transaction_manager.commit_transaction(self.agent_transaction.transaction_id)
                    
        except Exception as e:
            # Rollback on error
            if self.agent_transaction:
                await self.transaction_manager.rollback_transaction(
                    self.agent_transaction.transaction_id,
                    f"CDS operation failed: {str(e)}"
                )
            raise
        finally:
            self.agent_transaction = None
    
    async def register_agent_transactional(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        capabilities: List[str]
    ) -> Dict[str, Any]:
        """Register agent with full transaction support"""
        async with self.transactional_operation(
            agent_id=agent_id,
            operation_type="agent_registration",
            transaction_type="AGENT_REGISTRATION"
        ) as tx_context:
            agent_tx = tx_context["agent_transaction"]
            
            # Add registration operation to transaction
            await self.transaction_manager.add_operation(
                transaction_id=agent_tx.transaction_id,
                operation_type="REGISTER_AGENT",
                resource_type="agent",
                resource_id=agent_id,
                operation_data={
                    "agent_name": agent_name,
                    "agent_type": agent_type,
                    "capabilities": capabilities
                },
                rollback_data={
                    "action": "unregister_agent",
                    "agent_id": agent_id
                }
            )
            
            # Execute the registration
            result = await self.register_agent(agent_name, agent_type, capabilities)
            
            # Create checkpoint
            await self.transaction_manager.create_checkpoint(
                transaction_id=agent_tx.transaction_id,
                checkpoint_name="agent_registered",
                data={"agent_id": agent_id, "result": result}
            )
            
            return result
    
    async def send_message_transactional(
        self,
        agent_id: str,
        from_agent_id: str,
        to_agent_id: str,
        message_type: str,
        payload: str,
        priority: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """Send message with transaction support"""
        async with self.transactional_operation(
            agent_id=agent_id,
            operation_type="message_send",
            transaction_type="MESSAGE_OPERATION"
        ) as tx_context:
            agent_tx = tx_context["agent_transaction"]
            
            # Add message operation to transaction
            message_operation = await self.transaction_manager.add_operation(
                transaction_id=agent_tx.transaction_id,
                operation_type="SEND_MESSAGE",
                resource_type="message",
                resource_id=f"{from_agent_id}_to_{to_agent_id}_{datetime.now().isoformat()}",
                operation_data={
                    "from_agent_id": from_agent_id,
                    "to_agent_id": to_agent_id,
                    "message_type": message_type,
                    "payload": payload,
                    "priority": priority
                },
                rollback_data={
                    "action": "mark_message_deleted",
                    "message_metadata": {
                        "from_agent_id": from_agent_id,
                        "to_agent_id": to_agent_id,
                        "message_type": message_type
                    }
                }
            )
            
            # Execute the message send
            result = await self.send_message(from_agent_id, to_agent_id, message_type, payload, priority)
            
            # Update operation with result
            if result.get('status') == 'SUCCESS':
                message_operation.rollback_data["message_id"] = result.get('messageId')
            
            return result
    
    async def call_action_transactional(
        self,
        agent_id: str,
        action_name: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Call CDS action with transaction support"""
        async with self.transactional_operation(
            agent_id=agent_id,
            operation_type="data_analysis" if "Analysis" in action_name else "ml_request",
            transaction_type="CDS_ACTION"
        ) as tx_context:
            agent_tx = tx_context["agent_transaction"]
            
            # Add action call to transaction
            await self.transaction_manager.add_operation(
                transaction_id=agent_tx.transaction_id,
                operation_type="CALL_ACTION",
                resource_type="cds_action",
                resource_id=action_name,
                operation_data={
                    "action_name": action_name,
                    "parameters": parameters or {}
                },
                rollback_data={
                    "action": "compensate_action",
                    "action_name": action_name,
                    "agent_id": agent_id
                }
            )
            
            # Execute the action
            result = await self.call_action(action_name, parameters)
            
            # Create checkpoint with result
            await self.transaction_manager.create_checkpoint(
                transaction_id=agent_tx.transaction_id,
                checkpoint_name=f"action_{action_name}_completed",
                data={"action_name": action_name, "result": result}
            )
            
            return result
    
    async def bulk_operations_transactional(
        self,
        agent_id: str,
        operations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple operations in a single transaction"""
        async with self.transactional_operation(
            agent_id=agent_id,
            operation_type="transaction",
            transaction_type="BULK_OPERATIONS",
            timeout_seconds=600  # 10 minutes for bulk operations
        ) as tx_context:
            agent_tx = tx_context["agent_transaction"]
            results = []
            
            for i, op in enumerate(operations):
                op_type = op.get("type", "UNKNOWN")
                
                # Add operation to transaction
                await self.transaction_manager.add_operation(
                    transaction_id=agent_tx.transaction_id,
                    operation_type=op_type,
                    resource_type=op.get("resource_type", "unknown"),
                    resource_id=op.get("resource_id", f"bulk_op_{i}"),
                    operation_data=op.get("data", {}),
                    rollback_data=op.get("rollback_data")
                )
                
                # Execute operation based on type
                try:
                    if op_type == "REGISTER_AGENT":
                        result = await self.register_agent(
                            op["data"]["agent_name"],
                            op["data"]["agent_type"],
                            op["data"]["capabilities"]
                        )
                    elif op_type == "SEND_MESSAGE":
                        data = op["data"]
                        result = await self.send_message(
                            data["from_agent_id"],
                            data["to_agent_id"],
                            data["message_type"],
                            data["payload"],
                            data.get("priority", "MEDIUM")
                        )
                    elif op_type == "CALL_ACTION":
                        result = await self.call_action(
                            op["data"]["action_name"],
                            op["data"].get("parameters")
                        )
                    else:
                        result = {"error": f"Unknown operation type: {op_type}"}
                    
                    results.append(result)
                    
                    # Create checkpoint for each successful operation
                    await self.transaction_manager.create_checkpoint(
                        transaction_id=agent_tx.transaction_id,
                        checkpoint_name=f"bulk_operation_{i}_completed",
                        data={"operation_index": i, "operation_type": op_type, "result": result}
                    )
                    
                except Exception as e:
                    logger.error(f"Bulk operation {i} failed: {e}")
                    results.append({"error": str(e), "operation_index": i})
                    # Continue with other operations
            
            return results
    
    async def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an agent transaction"""
        transaction = self.transaction_manager.get_transaction(transaction_id)
        if transaction:
            return transaction.to_dict()
        
        # Check completed transactions
        for tx in self.transaction_manager.completed_transactions:
            if tx.transaction_id == transaction_id:
                return tx.to_dict()
        
        return None
    
    async def list_agent_transactions(self, agent_id: str) -> List[Dict[str, Any]]:
        """List all transactions for an agent"""
        transactions = self.transaction_manager.get_agent_transactions(agent_id)
        return [tx.to_dict() for tx in transactions]


class CDSTransactionalMixin:
    """Mixin to add transactional CDS operations to agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transactional_cds_client: Optional[TransactionalCDSClient] = None
    
    async def initialize_transactional_cds(self, agent_id: str = None):
        """Initialize transactional CDS client"""
        try:
            # Get CDS config from the regular mixin
            config = getattr(self, 'cds_config', CDSServiceConfig())
            
            self._transactional_cds_client = TransactionalCDSClient(config)
            await self._transactional_cds_client.connect(agent_id or getattr(self, 'agent_id', 'unknown'))
            
            logger.info(f"Transactional CDS client initialized for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize transactional CDS client: {e}")
            return False
    
    @property
    def transactional_cds(self) -> Optional[TransactionalCDSClient]:
        """Get the transactional CDS client"""
        return self._transactional_cds_client
    
    async def register_agent_with_transaction(self, capabilities: List[str]) -> Dict[str, Any]:
        """Register agent using transactional CDS client"""
        if not self._transactional_cds_client:
            raise RuntimeError("Transactional CDS client not initialized")
        
        agent_id = getattr(self, 'agent_id', 'unknown')
        agent_type = getattr(self, 'agent_type', 'generic')
        
        return await self._transactional_cds_client.register_agent_transactional(
            agent_id=agent_id,
            agent_name=agent_id,
            agent_type=agent_type,
            capabilities=capabilities
        )
    
    async def send_message_with_transaction(
        self,
        to_agent_id: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """Send message using transactional CDS client"""
        if not self._transactional_cds_client:
            raise RuntimeError("Transactional CDS client not initialized")
        
        agent_id = getattr(self, 'agent_id', 'unknown')
        
        return await self._transactional_cds_client.send_message_transactional(
            agent_id=agent_id,
            from_agent_id=agent_id,
            to_agent_id=to_agent_id,
            message_type=message_type,
            payload=json.dumps(payload),
            priority=priority
        )
    
    async def call_action_with_transaction(
        self,
        action_name: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Call CDS action using transactional client"""
        if not self._transactional_cds_client:
            raise RuntimeError("Transactional CDS client not initialized")
        
        agent_id = getattr(self, 'agent_id', 'unknown')
        
        return await self._transactional_cds_client.call_action_transactional(
            agent_id=agent_id,
            action_name=action_name,
            parameters=parameters
        )
    
    async def cleanup_transactional_cds(self):
        """Cleanup transactional CDS client"""
        if self._transactional_cds_client:
            await self._transactional_cds_client.disconnect()
            self._transactional_cds_client = None


def create_transactional_cds_client(agent_id: str = None, config: CDSServiceConfig = None) -> TransactionalCDSClient:
    """Factory function to create transactional CDS client"""
    client = TransactionalCDSClient(config or CDSServiceConfig())
    return client