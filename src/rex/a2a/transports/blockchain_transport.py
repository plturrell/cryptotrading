"""
Blockchain Transport Implementation for A2A Messages
Provides blockchain-based message transport for A2A agents
"""

import logging
from typing import List, Optional
from datetime import datetime

from .base_transport import MessageTransport
from ..protocols.a2a_protocol import A2AMessage, MessageType
from ..blockchain.blockchain_registry import get_blockchain_registries
from ..blockchain.blockchain_signatures import A2AMessageSigner

logger = logging.getLogger(__name__)

class BlockchainTransport(MessageTransport):
    """Blockchain-based transport for A2A messages"""
    
    def __init__(self, private_key: str):
        """
        Initialize blockchain transport
        
        Args:
            private_key: Ethereum private key for blockchain operations
        """
        self.private_key = private_key
        self.blockchain_registry, self.blockchain_workflow_registry = get_blockchain_registries()
        self.agent_accounts = {}
        
        if not self.blockchain_registry:
            logger.error("No blockchain registry available")
            raise RuntimeError("Blockchain registry not available")
    
    async def send_message(self, message: A2AMessage) -> bool:
        """Send A2A message via blockchain with signatures"""
        try:
            # Get sender's blockchain address from registry
            sender_info = self.blockchain_registry.get_agent(message.sender_id)
            if not sender_info or 'blockchain_address' not in sender_info.get('metadata', {}):
                logger.error(f"No blockchain address found for sender {message.sender_id}")
                return False
            
            sender_blockchain_address = sender_info['metadata']['blockchain_address']
            
            # Sign message if not already signed
            if not message.blockchain_signature:
                # Get workflow instance address if available
                instance_address = message.workflow_context.get("instance_address") if message.workflow_context else None
                
                # Sign message
                signed_dict = A2AMessageSigner.sign_message(
                    message.to_dict(),
                    message.sender_id,
                    sender_blockchain_address,
                    self.private_key,
                    self.blockchain_registry.contract_address,
                    instance_address
                )
                
                # Update message with signature
                message.blockchain_signature = signed_dict["blockchain_signature"]
                message.blockchain_context = signed_dict["blockchain_context"]
            
            # Convert A2AMessage to blockchain format with all signature data
            blockchain_payload = {
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "message_type": message.message_type.value,
                "payload": message.payload,
                "message_id": message.message_id,
                "timestamp": message.timestamp,
                "protocol_version": message.protocol_version,
                "correlation_id": message.correlation_id,
                "priority": message.priority,
                "workflow_context": message.workflow_context,
                "sender_blockchain_address": sender_blockchain_address,
                "blockchain_signature": message.blockchain_signature,
                "blockchain_context": message.blockchain_context
            }
            
            # Send via blockchain
            tx_hash = self.blockchain_registry.send_blockchain_message(
                sender_id=message.sender_id,
                receiver_id=message.receiver_id,
                message_type=message.message_type,
                payload=blockchain_payload,
                priority=message.priority
            )
            
            if tx_hash:
                logger.debug(f"Message sent via blockchain: {message.message_id} (tx: {tx_hash})")
                return True
            else:
                logger.error(f"Failed to send message via blockchain: {message.message_id}")
                return False
                
        except Exception as e:
            logger.error(f"Blockchain send error: {e}")
            return False
    
    async def receive_messages(self, agent_id: str) -> List[A2AMessage]:
        """Receive pending messages from blockchain"""
        try:
            # Get pending blockchain messages
            blockchain_messages = self.blockchain_registry.get_pending_blockchain_messages(agent_id)
            
            # Convert to A2AMessage objects
            a2a_messages = []
            for msg in blockchain_messages:
                try:
                    # Parse message type
                    message_type = MessageType(msg['messageType'])
                    
                    # Create A2AMessage with blockchain signature fields
                    a2a_message = A2AMessage(
                        sender_id=msg['senderId'],
                        receiver_id=msg['receiverId'],
                        message_type=message_type,
                        payload=msg['payload'],
                        message_id=msg['messageId'],
                        timestamp=datetime.fromtimestamp(msg['timestamp']).isoformat(),
                        priority=msg['priority'],
                        protocol_version=msg.get('protocol_version', '1.0'),
                        correlation_id=msg.get('correlation_id'),
                        workflow_context=msg.get('workflow_context'),
                        sender_blockchain_address=msg.get('sender_blockchain_address'),
                        blockchain_signature=msg.get('blockchain_signature'),
                        blockchain_context=msg.get('blockchain_context')
                    )
                    
                    a2a_messages.append(a2a_message)
                    
                except Exception as e:
                    logger.error(f"Error parsing blockchain message: {e}")
                    continue
            
            logger.debug(f"Received {len(a2a_messages)} messages from blockchain for {agent_id}")
            return a2a_messages
            
        except Exception as e:
            logger.error(f"Blockchain receive error: {e}")
            return []
    
    async def acknowledge_message(self, agent_id: str, message_id: str) -> bool:
        """Acknowledge message processing on blockchain"""
        try:
            # Mark message as processed on blockchain
            success = self.blockchain_registry.mark_message_processed_blockchain(
                agent_id, message_id
            )
            
            if success:
                logger.debug(f"Message acknowledged on blockchain: {message_id}")
            else:
                logger.error(f"Failed to acknowledge message on blockchain: {message_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Blockchain acknowledge error: {e}")
            return False
    
    def register_agent_address(self, agent_id: str, address: str):
        """Register agent's blockchain address"""
        self.agent_accounts[agent_id] = {'address': address}
        logger.info(f"Registered blockchain address for {agent_id}: {address}")
    
    def get_agent_address(self, agent_id: str) -> Optional[str]:
        """Get agent's blockchain address"""
        return self.agent_accounts.get(agent_id, {}).get('address')
    
    def get_blockchain_info(self) -> dict:
        """Get blockchain connection information"""
        if self.blockchain_registry:
            return self.blockchain_registry.get_blockchain_info()
        return {"status": "disconnected"}