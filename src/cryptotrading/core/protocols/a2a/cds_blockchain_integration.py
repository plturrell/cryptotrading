"""
CDS-Blockchain Integration Layer
Bridges CAP Data Services with blockchain data exchange and workflows
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .blockchain_data_exchange import (
    BlockchainDataExchangeService,
    DataStatus,
    WorkflowStatus,
    get_data_exchange_service
)
from .workflow_orchestration import (
    WorkflowOrchestrator,
    WorkflowTemplate,
    WorkflowStep,
    WorkflowStepType,
    get_orchestrator
)
from .data_encryption import get_encryption_service

logger = logging.getLogger(__name__)


class CDSEntityType(Enum):
    """CDS entity types for mapping"""
    A2A_AGENT = "A2AAgents"
    A2A_CONNECTION = "A2AConnections"
    A2A_MESSAGE = "A2AMessages"
    A2A_WORKFLOW = "A2AWorkflows"
    A2A_WORKFLOW_EXECUTION = "A2AWorkflowExecutions"
    DATA_INGESTION_JOB = "DataIngestionJobs"
    MARKET_DATA_SOURCE = "MarketDataSources"
    AGGREGATED_MARKET_DATA = "AggregatedMarketData"
    ONCHAIN_DATA = "OnchainData"
    AI_ANALYSIS = "AIAnalyses"
    ML_MODEL_REGISTRY = "MLModelRegistry"


@dataclass
class CDSEntity:
    """Base class for CDS entities"""
    entity_type: CDSEntityType
    entity_id: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CDS storage"""
        return {
            "entityType": self.entity_type.value,
            "entityId": self.entity_id,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class CDSWorkflowMapping:
    """Maps CDS workflows to blockchain workflows"""
    cds_workflow_id: str
    blockchain_workflow_id: int
    workflow_type: str
    participants: List[str]
    status: str
    created_at: datetime
    data_packets: List[int]
    
    def to_cds_entity(self) -> Dict[str, Any]:
        """Convert to CDS A2AWorkflow entity"""
        return {
            "workflowId": self.cds_workflow_id,
            "blockchainId": self.blockchain_workflow_id,
            "workflowType": self.workflow_type,
            "participants": json.dumps(self.participants),
            "status": self.status,
            "createdAt": self.created_at.isoformat(),
            "dataPackets": json.dumps(self.data_packets)
        }


class CDSBlockchainAdapter:
    """Adapter for integrating CDS with blockchain services"""
    
    def __init__(self):
        """Initialize CDS-blockchain adapter"""
        self.data_exchange: Optional[BlockchainDataExchangeService] = None
        self.orchestrator: Optional[WorkflowOrchestrator] = None
        self.encryption_service = get_encryption_service()
        
        # Entity mappings
        self.workflow_mappings: Dict[str, CDSWorkflowMapping] = {}
        self.agent_mappings: Dict[str, str] = {}  # CDS ID -> blockchain ID
        self.data_mappings: Dict[str, int] = {}  # CDS data ID -> blockchain data ID
        
        logger.info("CDSBlockchainAdapter initialized")
    
    async def initialize(self) -> bool:
        """Initialize blockchain services"""
        try:
            self.data_exchange = await get_data_exchange_service()
            self.orchestrator = await get_orchestrator()
            
            logger.info("CDS-Blockchain integration initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CDS-blockchain integration: {e}")
            return False
    
    async def register_cds_agent(
        self,
        agent_name: str,
        agent_type: str,
        capabilities: str,
        cds_entity_id: str
    ) -> Optional[str]:
        """
        Register a CDS agent in blockchain
        
        Args:
            agent_name: Name of the agent
            agent_type: Type of agent
            capabilities: Agent capabilities (JSON string)
            cds_entity_id: CDS entity ID
        
        Returns:
            Blockchain agent ID if successful
        """
        try:
            # Generate blockchain agent ID
            blockchain_agent_id = f"{agent_type}-{agent_name}-{cds_entity_id[:8]}"
            
            # Store mapping
            self.agent_mappings[cds_entity_id] = blockchain_agent_id
            
            # Store agent metadata on-chain
            if self.data_exchange:
                agent_data = {
                    "cds_entity_id": cds_entity_id,
                    "agent_name": agent_name,
                    "agent_type": agent_type,
                    "capabilities": json.loads(capabilities) if isinstance(capabilities, str) else capabilities,
                    "registered_at": datetime.now().isoformat()
                }
                
                data_id = await self.data_exchange.store_data(
                    sender_agent_id="cds-adapter",
                    receiver_agent_id=blockchain_agent_id,
                    data=agent_data,
                    data_type="agent_registration",
                    is_encrypted=False
                )
                
                if data_id:
                    logger.info(f"Registered CDS agent {agent_name} with blockchain ID {blockchain_agent_id}")
                    return blockchain_agent_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to register CDS agent: {e}")
            return None
    
    async def send_cds_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        message_type: str,
        payload: str,
        priority: str = "normal"
    ) -> Optional[Dict[str, Any]]:
        """
        Send message between CDS agents via blockchain
        
        Args:
            from_agent_id: CDS sender agent ID
            to_agent_id: CDS receiver agent ID
            message_type: Type of message
            payload: Message payload (JSON string)
            priority: Message priority
        
        Returns:
            Message info if successful
        """
        try:
            # Map CDS IDs to blockchain IDs
            from_blockchain_id = self.agent_mappings.get(from_agent_id, from_agent_id)
            to_blockchain_id = self.agent_mappings.get(to_agent_id, to_agent_id)
            
            # Parse payload
            payload_data = json.loads(payload) if isinstance(payload, str) else payload
            
            # Encrypt if high priority
            is_encrypted = priority == "high"
            if is_encrypted:
                encrypted_data = self.encryption_service.encrypt_for_agents(
                    data=payload_data,
                    sender_agent_id=from_blockchain_id,
                    receiver_agent_ids=[to_blockchain_id]
                )
                data_to_store = encrypted_data.get(to_blockchain_id, payload_data)
            else:
                data_to_store = payload_data
            
            # Store message on-chain
            if self.data_exchange:
                data_id = await self.data_exchange.store_data(
                    sender_agent_id=from_blockchain_id,
                    receiver_agent_id=to_blockchain_id,
                    data=data_to_store,
                    data_type=f"cds_message:{message_type}",
                    is_encrypted=is_encrypted
                )
                
                if data_id:
                    return {
                        "messageId": f"msg-{data_id}",
                        "status": "delivered",
                        "deliveryTime": datetime.now().isoformat(),
                        "blockchainDataId": data_id
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to send CDS message: {e}")
            return None
    
    async def execute_cds_workflow(
        self,
        workflow_id: str,
        input_data: str
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a CDS workflow on blockchain
        
        Args:
            workflow_id: CDS workflow ID
            input_data: Input data for workflow (JSON string)
        
        Returns:
            Execution info if successful
        """
        try:
            if not self.orchestrator:
                logger.error("Orchestrator not initialized")
                return None
            
            # Parse input data
            data = json.loads(input_data) if isinstance(input_data, str) else input_data
            
            # Map CDS workflow to blockchain template
            template_id = self._map_cds_to_blockchain_workflow(workflow_id)
            
            # Create blockchain workflow
            blockchain_workflow_id = await self.orchestrator.create_workflow(
                template_id=template_id,
                parameters=data
            )
            
            if blockchain_workflow_id:
                # Create mapping
                mapping = CDSWorkflowMapping(
                    cds_workflow_id=workflow_id,
                    blockchain_workflow_id=int(blockchain_workflow_id.split('-')[-1]),
                    workflow_type=template_id,
                    participants=[],  # Will be populated from template
                    status="created",
                    created_at=datetime.now(),
                    data_packets=[]
                )
                
                self.workflow_mappings[workflow_id] = mapping
                
                # Execute workflow
                result = await self.orchestrator.execute_workflow(
                    workflow_id=blockchain_workflow_id,
                    initial_data=data
                )
                
                return {
                    "executionId": blockchain_workflow_id,
                    "status": "running" if result else "failed",
                    "estimatedTime": 300,  # seconds
                    "blockchainWorkflowId": mapping.blockchain_workflow_id
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to execute CDS workflow: {e}")
            return None
    
    async def sync_market_data_to_blockchain(
        self,
        source_id: str,
        symbols: List[str],
        data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Sync market data from CDS to blockchain
        
        Args:
            source_id: Market data source ID
            symbols: List of symbols
            data: Market data records
        
        Returns:
            Sync result if successful
        """
        try:
            if not self.data_exchange:
                return None
            
            stored_ids = []
            errors = 0
            
            for record in data:
                try:
                    # Store each record on-chain
                    data_id = await self.data_exchange.store_data(
                        sender_agent_id=f"cds-source-{source_id}",
                        receiver_agent_id="market-data-aggregator",
                        data=record,
                        data_type="market_data",
                        is_encrypted=False,
                        compress=True  # Compress market data
                    )
                    
                    if data_id:
                        stored_ids.append(data_id)
                        # Map CDS record ID to blockchain data ID
                        if 'id' in record:
                            self.data_mappings[record['id']] = data_id
                    else:
                        errors += 1
                        
                except Exception as e:
                    logger.error(f"Failed to store market data record: {e}")
                    errors += 1
            
            return {
                "recordsSynced": len(stored_ids),
                "errors": errors,
                "nextSync": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "blockchainDataIds": stored_ids
            }
            
        except Exception as e:
            logger.error(f"Failed to sync market data: {e}")
            return None
    
    async def store_ai_analysis_onchain(
        self,
        analysis_id: str,
        model_id: str,
        analysis_type: str,
        results: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[int]:
        """
        Store AI analysis results on blockchain
        
        Args:
            analysis_id: CDS analysis ID
            model_id: ML model ID
            analysis_type: Type of analysis
            results: Analysis results
            metadata: Additional metadata
        
        Returns:
            Blockchain data ID if successful
        """
        try:
            if not self.data_exchange:
                return None
            
            # Prepare analysis data
            analysis_data = {
                "cds_analysis_id": analysis_id,
                "model_id": model_id,
                "analysis_type": analysis_type,
                "results": results,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            # Encrypt sensitive analysis results
            encrypted_data = self.encryption_service.encrypt_for_agents(
                data=analysis_data,
                sender_agent_id=f"ml-model-{model_id}",
                receiver_agent_ids=["ai-analysis-consumer"]
            )
            
            # Store on-chain
            data_id = await self.data_exchange.store_data(
                sender_agent_id=f"ml-model-{model_id}",
                receiver_agent_id="ai-analysis-consumer",
                data=encrypted_data.get("ai-analysis-consumer", analysis_data),
                data_type=f"ai_analysis:{analysis_type}",
                is_encrypted=True,
                compress=True
            )
            
            if data_id:
                self.data_mappings[analysis_id] = data_id
                logger.info(f"Stored AI analysis {analysis_id} on-chain with ID {data_id}")
            
            return data_id
            
        except Exception as e:
            logger.error(f"Failed to store AI analysis on-chain: {e}")
            return None
    
    async def retrieve_blockchain_data_for_cds(
        self,
        data_id: int,
        requesting_agent: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve blockchain data for CDS consumption
        
        Args:
            data_id: Blockchain data ID
            requesting_agent: CDS agent requesting data
        
        Returns:
            Data if successful
        """
        try:
            if not self.data_exchange:
                return None
            
            # Map CDS agent to blockchain agent
            blockchain_agent = self.agent_mappings.get(requesting_agent, requesting_agent)
            
            # Retrieve data
            data = await self.data_exchange.retrieve_data(
                data_id=data_id,
                agent_id=blockchain_agent
            )
            
            if data:
                # Decrypt if needed
                if data.get('is_encrypted'):
                    decrypted = self.encryption_service.decrypt_agent_data(
                        encrypted_data=data['data'],
                        agent_id=blockchain_agent
                    )
                    data['data'] = decrypted
                
                # Add CDS metadata
                data['cds_metadata'] = {
                    'retrieved_at': datetime.now().isoformat(),
                    'requesting_agent': requesting_agent,
                    'blockchain_data_id': data_id
                }
                
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve blockchain data: {e}")
            return None
    
    def _map_cds_to_blockchain_workflow(self, cds_workflow_id: str) -> str:
        """Map CDS workflow ID to blockchain template ID"""
        # Simple mapping - in production this would be configurable
        mappings = {
            "market_analysis": "market_analysis_v1",
            "trading_signal": "trading_signal_v1",
            "portfolio_optimization": "portfolio_opt_v1",
            "risk_monitoring": "risk_monitoring_v1",
            "ml_training": "ml_training_v1"
        }
        
        # Extract workflow type from ID
        workflow_type = cds_workflow_id.split('-')[0] if '-' in cds_workflow_id else cds_workflow_id
        
        return mappings.get(workflow_type, "market_analysis_v1")
    
    async def sync_cds_entities_to_blockchain(
        self,
        entity_type: CDSEntityType,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Bulk sync CDS entities to blockchain
        
        Args:
            entity_type: Type of CDS entity
            entities: List of entity records
        
        Returns:
            Sync results
        """
        try:
            if not self.data_exchange:
                return {"success": 0, "failed": len(entities), "errors": ["Service not initialized"]}
            
            success_count = 0
            failed_count = 0
            stored_ids = []
            
            for entity in entities:
                try:
                    # Create CDSEntity
                    cds_entity = CDSEntity(
                        entity_type=entity_type,
                        entity_id=entity.get('id', str(uuid4())),
                        created_at=datetime.fromisoformat(entity.get('createdAt', datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(entity.get('updatedAt', datetime.now().isoformat())),
                        metadata=entity
                    )
                    
                    # Store on blockchain
                    data_id = await self.data_exchange.store_data(
                        sender_agent_id="cds-sync-service",
                        receiver_agent_id="cds-entity-store",
                        data=cds_entity.to_dict(),
                        data_type=f"cds_entity:{entity_type.value}",
                        is_encrypted=False,
                        compress=True
                    )
                    
                    if data_id:
                        success_count += 1
                        stored_ids.append(data_id)
                        self.data_mappings[cds_entity.entity_id] = data_id
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to sync entity: {e}")
                    failed_count += 1
            
            return {
                "success": success_count,
                "failed": failed_count,
                "total": len(entities),
                "stored_ids": stored_ids,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to sync CDS entities: {e}")
            return {
                "success": 0,
                "failed": len(entities),
                "errors": [str(e)]
            }
    
    async def get_workflow_status_for_cds(
        self,
        execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get blockchain workflow status for CDS
        
        Args:
            execution_id: Workflow execution ID
        
        Returns:
            Status information if available
        """
        try:
            if not self.orchestrator:
                return None
            
            # Get status from orchestrator
            status = self.orchestrator.get_workflow_status(execution_id)
            
            if status:
                # Enhance with CDS-specific info
                cds_workflow_id = None
                for cds_id, mapping in self.workflow_mappings.items():
                    if str(mapping.blockchain_workflow_id) in execution_id:
                        cds_workflow_id = cds_id
                        break
                
                status['cds_workflow_id'] = cds_workflow_id
                status['blockchain_execution_id'] = execution_id
                
                # Calculate progress
                if status.get('steps'):
                    completed = sum(1 for s in status['steps'] if s['status'] == 'completed')
                    total = len(status['steps'])
                    status['progress'] = (completed / total * 100) if total > 0 else 0
                
                return status
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get CDS-blockchain integration metrics"""
        metrics = {
            "mapped_agents": len(self.agent_mappings),
            "mapped_workflows": len(self.workflow_mappings),
            "mapped_data": len(self.data_mappings),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add blockchain metrics if available
        if self.data_exchange and hasattr(self.data_exchange, 'get_metrics'):
            metrics['blockchain_metrics'] = self.data_exchange.get_metrics()
        
        return metrics


# Singleton instance
_cds_adapter: Optional[CDSBlockchainAdapter] = None


async def get_cds_adapter() -> CDSBlockchainAdapter:
    """Get or create the CDS adapter singleton"""
    global _cds_adapter
    
    if _cds_adapter is None:
        _cds_adapter = CDSBlockchainAdapter()
        await _cds_adapter.initialize()
    
    return _cds_adapter


# CDS Service Implementation Functions
async def cds_register_agent(
    agent_name: str,
    agent_type: str,
    capabilities: str
) -> Dict[str, Any]:
    """CDS service action: registerAgent"""
    adapter = await get_cds_adapter()
    
    cds_entity_id = f"agent-{agent_name}-{datetime.now().timestamp()}"
    blockchain_id = await adapter.register_cds_agent(
        agent_name=agent_name,
        agent_type=agent_type,
        capabilities=capabilities,
        cds_entity_id=cds_entity_id
    )
    
    return {
        "agentId": cds_entity_id,
        "status": "registered" if blockchain_id else "failed",
        "message": f"Agent registered with blockchain ID: {blockchain_id}" if blockchain_id else "Registration failed"
    }


async def cds_send_message(
    from_agent_id: str,
    to_agent_id: str,
    message_type: str,
    payload: str,
    priority: str = "normal"
) -> Dict[str, Any]:
    """CDS service action: sendMessage"""
    adapter = await get_cds_adapter()
    
    result = await adapter.send_cds_message(
        from_agent_id=from_agent_id,
        to_agent_id=to_agent_id,
        message_type=message_type,
        payload=payload,
        priority=priority
    )
    
    return result or {
        "messageId": None,
        "status": "failed",
        "deliveryTime": None
    }


async def cds_execute_workflow(
    workflow_id: str,
    input_data: str
) -> Dict[str, Any]:
    """CDS service action: executeWorkflow"""
    adapter = await get_cds_adapter()
    
    result = await adapter.execute_cds_workflow(
        workflow_id=workflow_id,
        input_data=input_data
    )
    
    return result or {
        "executionId": None,
        "status": "failed",
        "estimatedTime": 0
    }


async def cds_sync_market_data(
    source_id: str,
    symbols: List[str],
    data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """CDS service action: syncMarketData"""
    adapter = await get_cds_adapter()
    
    result = await adapter.sync_market_data_to_blockchain(
        source_id=source_id,
        symbols=symbols,
        data=data
    )
    
    return result or {
        "recordsSynced": 0,
        "errors": len(data),
        "nextSync": None
    }