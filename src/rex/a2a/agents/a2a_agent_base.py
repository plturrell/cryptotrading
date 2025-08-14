"""
Base A2A Agent with Blockchain Signatures
All agents inherit from this base class to ensure blockchain compliance
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime

from ..protocols import A2AMessage, A2AProtocol, MessageType
from ..blockchain.blockchain_signatures import A2AMessageSigner, BlockchainDataStorage
from ..registry.registry import agent_registry

logger = logging.getLogger(__name__)

class A2AAgentBase(ABC):
    """Base class for all A2A agents with blockchain signature support"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        blockchain_address: str,
        private_key: str,
        contract_address: str,
        capabilities: List[str],
        w3=None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.blockchain_address = blockchain_address
        self.private_key = private_key
        self.contract_address = contract_address
        self.capabilities = capabilities
        self.w3 = w3
        
        # Initialize blockchain data storage
        if w3:
            self.data_storage = BlockchainDataStorage(w3, contract_address)
        else:
            self.data_storage = None
        
        # Register agent with blockchain address
        self._register_with_blockchain()
    
    def _register_with_blockchain(self):
        """Register agent with blockchain address in metadata"""
        metadata = {
            'blockchain_address': self.blockchain_address,
            'contract_address': self.contract_address,
            'capabilities': self.capabilities
        }
        agent_registry.register_agent(
            self.agent_id,
            self.agent_type,
            self.capabilities,
            metadata
        )
    
    async def send_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> A2AMessage:
        """Send A2A message with blockchain signature"""
        
        # Create message
        message = A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload
        )
        
        # Add blockchain fields
        message.sender_blockchain_address = self.blockchain_address
        if workflow_context:
            message.workflow_context = workflow_context
        
        # Sign message with blockchain signature
        signed_message_dict = A2AMessageSigner.sign_message(
            message.to_dict(),
            self.agent_id,
            self.blockchain_address,
            self.private_key,
            self.contract_address,
            workflow_context.get("instance_address") if workflow_context else None
        )
        
        # Update message with signature
        message.blockchain_signature = signed_message_dict["blockchain_signature"]
        message.blockchain_context = signed_message_dict["blockchain_context"]
        
        logger.info(f"Agent {self.agent_id} sending signed message to {receiver_id}")
        return message
    
    async def verify_message(self, message: A2AMessage) -> bool:
        """Verify incoming message blockchain signature"""
        message_dict = message.to_dict()
        
        if not A2AMessageSigner.verify_message(message_dict):
            logger.error(f"Failed to verify message from {message.sender_id}")
            return False
        
        logger.info(f"Successfully verified message from {message.sender_id}")
        return True
    
    async def store_data_with_signature(
        self,
        data: Dict[str, Any],
        message_id: str,
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store data with complete blockchain signature"""
        if not self.data_storage:
            logger.warning("No blockchain data storage available")
            return {"error": "No blockchain storage configured"}
        
        storage_receipt = self.data_storage.store_with_signature(
            data=data,
            agent_id=self.agent_id,
            agent_blockchain_address=self.blockchain_address,
            private_key=self.private_key,
            message_id=message_id,
            workflow_context=workflow_context
        )
        
        logger.info(f"Agent {self.agent_id} stored data with signature: {storage_receipt['signature_hash']}")
        return storage_receipt
    
    async def retrieve_verified_data(self, storage_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve and verify blockchain-signed data"""
        if not self.data_storage:
            return None
        
        data = self.data_storage.retrieve_and_verify(storage_key)
        if data:
            logger.info(f"Agent {self.agent_id} retrieved verified data")
        else:
            logger.error(f"Failed to retrieve/verify data for key: {storage_key}")
        
        return data
    
    @abstractmethod
    async def handle_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle incoming A2A message - must be implemented by subclasses"""
        pass
    
    async def process_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Process incoming message with signature verification"""
        # Verify message signature
        if not await self.verify_message(message):
            return {
                "success": False,
                "error": "Invalid message signature",
                "agent_id": self.agent_id
            }
        
        # Process message
        try:
            result = await self.handle_message(message)
            
            # Add agent blockchain context to result
            result["blockchain_context"] = {
                "agent_id": self.agent_id,
                "agent_blockchain_address": self.blockchain_address,
                "verified": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Get agent's blockchain information"""
        return {
            "agent_id": self.agent_id,
            "blockchain_address": self.blockchain_address,
            "contract_address": self.contract_address,
            "capabilities": self.capabilities,
            "chain_id": self.w3.eth.chain_id if self.w3 else None
        }