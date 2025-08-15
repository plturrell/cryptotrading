"""
Blockchain-integrated A2A Registry
Replaces the local registry with blockchain-backed agent and workflow management
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from web3 import Web3
from eth_account import Account
from pathlib import Path

from ..registry.registry import AgentRegistry
from ..orchestration.workflow_registry import WorkflowRegistry
from ..protocols.a2a_protocol import A2AMessage, MessageType, A2AProtocol
from .blockchain_signatures import BlockchainSignature, BlockchainDataStorage

logger = logging.getLogger(__name__)

class BlockchainA2ARegistry(AgentRegistry):
    """Blockchain-backed A2A Registry that integrates with smart contracts"""
    
    def __init__(self, contract_address: str = None, rpc_url: str = "http://127.0.0.1:8545"):
        super().__init__()
        
        # Blockchain connection
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract_address = contract_address
        self.contract = None
        
        # Load contract ABI and address
        self._load_contract()
        
        # Agent accounts mapping (address -> private_key)
        self.agent_accounts = {}
        
        # Sync with blockchain on initialization
        self._sync_from_blockchain()
    
    def _load_contract(self):
        """Load contract ABI and address from deployment artifacts"""
        artifacts_dir = Path(__file__).parent / "artifacts"
        deployment_file = artifacts_dir / "A2ANetwork_deployment.json"
        
        if deployment_file.exists():
            with open(deployment_file, 'r') as f:
                deployment = json.load(f)
                self.contract_address = deployment['address']
                contract_abi = deployment['abi']
                
                self.contract = self.w3.eth.contract(
                    address=self.contract_address,
                    abi=contract_abi
                )
                logger.info(f"Connected to A2A Network contract at {self.contract_address}")
        else:
            logger.warning("No contract deployment found. Run setup_local_blockchain.py first.")
    
    def _sync_from_blockchain(self):
        """Sync local state with blockchain"""
        if not self.contract:
            return
        
        try:
            # Get all registered agents from blockchain
            agent_ids = self.contract.functions.getAllAgents().call()
            
            for agent_id in agent_ids:
                agent_data = self.contract.functions.getAgent(agent_id).call()
                
                # Convert blockchain data to local format
                self.agents[agent_id] = {
                    'id': agent_data[0],  # agentId
                    'address': agent_data[1],  # agentAddress
                    'type': agent_data[2],  # agentType
                    'capabilities': list(agent_data[3]),  # capabilities
                    'status': self._status_from_blockchain(agent_data[4]),  # status
                    'registered_at': datetime.fromtimestamp(agent_data[5]).isoformat(),
                    'last_active_at': datetime.fromtimestamp(agent_data[6]).isoformat(),
                    'blockchain_synced': True
                }
                
                # Index capabilities
                for capability in agent_data[3]:
                    if capability not in self.capabilities:
                        self.capabilities[capability] = []
                    if agent_id not in self.capabilities[capability]:
                        self.capabilities[capability].append(agent_id)
            
            logger.info(f"Synced {len(agent_ids)} agents from blockchain")
            
        except Exception as e:
            logger.error(f"Failed to sync from blockchain: {e}")
    
    def _status_from_blockchain(self, status_int: int) -> str:
        """Convert blockchain status to string"""
        status_map = {0: "inactive", 1: "active", 2: "busy", 3: "error"}
        return status_map.get(status_int, "inactive")
    
    def _status_to_blockchain(self, status: str) -> int:
        """Convert string status to blockchain format"""
        status_map = {"inactive": 0, "active": 1, "busy": 2, "error": 3}
        return status_map.get(status, 0)
    
    def register_agent_blockchain(self, agent_id: str, agent_type: str, 
                                capabilities: List[str], private_key: str,
                                metadata: str = "") -> str:
        """Register agent on blockchain and locally"""
        
        # Create account from private key
        account = Account.from_key(private_key)
        
        try:
            # Register on blockchain
            tx = self.contract.functions.registerAgent(
                agent_id,
                agent_type,
                capabilities,
                metadata
            ).build_transaction({
                'from': account.address,
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Store account info
            self.agent_accounts[agent_id] = {
                'address': account.address,
                'private_key': private_key
            }
            
            # Register locally
            self.register_agent(agent_id, agent_type, capabilities, {
                'blockchain_address': account.address,
                'blockchain_synced': True,
                'transaction_hash': tx_hash.hex(),
                'block_number': receipt.blockNumber
            })
            
            logger.info(f"Agent {agent_id} registered on blockchain at {account.address}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id} on blockchain: {e}")
            raise
    
    def update_agent_status_blockchain(self, agent_id: str, status: str):
        """Update agent status on blockchain"""
        if agent_id not in self.agent_accounts:
            logger.error(f"No blockchain account found for agent {agent_id}")
            return
        
        account_info = self.agent_accounts[agent_id]
        
        try:
            tx = self.contract.functions.updateAgentStatus(
                self._status_to_blockchain(status)
            ).build_transaction({
                'from': account_info['address'],
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(account_info['address'])
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(
                tx, account_info['private_key']
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Update locally
            self.update_agent_status(agent_id, status)
            
            logger.info(f"Agent {agent_id} status updated to {status} on blockchain")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to update agent status on blockchain: {e}")
    
    def send_blockchain_message(self, sender_id: str, receiver_id: str, 
                               message_type: MessageType, payload: Dict[str, Any],
                               priority: int = 0) -> str:
        """Send A2A message via blockchain"""
        if sender_id not in self.agent_accounts:
            logger.error(f"No blockchain account found for sender {sender_id}")
            return None
        
        # Store payload using local storage
        payload_hash = self._store_payload_local(payload)
        
        account_info = self.agent_accounts[sender_id]
        
        try:
            # Convert MessageType to blockchain format
            message_type_int = self._message_type_to_blockchain(message_type)
            
            tx = self.contract.functions.sendMessage(
                receiver_id,
                message_type_int,
                payload_hash,
                priority
            ).build_transaction({
                'from': account_info['address'],
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(account_info['address'])
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(
                tx, account_info['private_key']
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Message sent from {sender_id} to {receiver_id} via blockchain")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to send blockchain message: {e}")
            return None
    
    def get_pending_blockchain_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get pending messages for an agent from blockchain"""
        if not self.contract:
            return []
        
        try:
            messages = self.contract.functions.getPendingMessages(agent_id).call()
            
            processed_messages = []
            for msg in messages:
                processed_msg = {
                    'messageId': msg[0],
                    'senderId': msg[1], 
                    'receiverId': msg[2],
                    'messageType': self._message_type_from_blockchain(msg[3]),
                    'payloadHash': msg[4],
                    'timestamp': msg[5],
                    'priority': msg[6],
                    'processed': msg[7],
                    'payload': self._load_payload_local(msg[4])  # Load actual payload
                }
                processed_messages.append(processed_msg)
            
            return processed_messages
            
        except Exception as e:
            logger.error(f"Failed to get pending messages: {e}")
            return []
    
    def mark_message_processed_blockchain(self, agent_id: str, message_id: str):
        """Mark message as processed on blockchain"""
        if agent_id not in self.agent_accounts:
            return
        
        account_info = self.agent_accounts[agent_id]
        
        try:
            tx = self.contract.functions.markMessageProcessed(message_id).build_transaction({
                'from': account_info['address'],
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(account_info['address'])
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(
                tx, account_info['private_key']
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Message {message_id} marked as processed")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to mark message as processed: {e}")
    
    def _message_type_to_blockchain(self, message_type: MessageType) -> int:
        """Convert MessageType to blockchain integer"""
        type_map = {
            MessageType.DATA_LOAD_REQUEST: 0,
            MessageType.DATA_LOAD_RESPONSE: 1,
            MessageType.ANALYSIS_REQUEST: 2,
            MessageType.ANALYSIS_RESPONSE: 3,
            MessageType.DATA_QUERY: 4,
            MessageType.DATA_QUERY_RESPONSE: 5,
            MessageType.TRADE_EXECUTION: 6,
            MessageType.TRADE_RESPONSE: 7,
            MessageType.WORKFLOW_REQUEST: 8,
            MessageType.WORKFLOW_RESPONSE: 9,
            MessageType.WORKFLOW_STATUS: 10,
            MessageType.HEARTBEAT: 11,
            MessageType.ERROR: 12
        }
        return type_map.get(message_type, 12)  # Default to ERROR
    
    def _message_type_from_blockchain(self, type_int: int) -> MessageType:
        """Convert blockchain integer to MessageType"""
        type_map = {
            0: MessageType.DATA_LOAD_REQUEST,
            1: MessageType.DATA_LOAD_RESPONSE,
            2: MessageType.ANALYSIS_REQUEST,
            3: MessageType.ANALYSIS_RESPONSE,
            4: MessageType.DATA_QUERY,
            5: MessageType.DATA_QUERY_RESPONSE,
            6: MessageType.TRADE_EXECUTION,
            7: MessageType.TRADE_RESPONSE,
            8: MessageType.WORKFLOW_REQUEST,
            9: MessageType.WORKFLOW_RESPONSE,
            10: MessageType.WORKFLOW_STATUS,
            11: MessageType.HEARTBEAT,
            12: MessageType.ERROR
        }
        return type_map.get(type_int, MessageType.ERROR)
    
    def _store_payload_local(self, payload: Dict[str, Any]) -> str:
        """Store payload using local database storage (honest implementation)"""
        # Create deterministic hash
        payload_json = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        
        # Store in local database/file system
        storage_dir = Path(__file__).parent / "payload_storage"
        storage_dir.mkdir(exist_ok=True)
        
        payload_file = storage_dir / f"{payload_hash}.json"
        with open(payload_file, 'w') as f:
            json.dump({
                "hash": payload_hash,
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
                "size": len(payload_json.encode())
            }, f, indent=2)
        
        logger.info(f"Payload stored locally: {payload_hash}")
        return payload_hash
    
    def _load_payload_local(self, payload_hash: str) -> Dict[str, Any]:
        """Load payload from local storage"""
        storage_dir = Path(__file__).parent / "payload_storage"
        payload_file = storage_dir / f"{payload_hash}.json"
        
        try:
            if payload_file.exists():
                with open(payload_file, 'r') as f:
                    stored_data = json.load(f)
                
                logger.info(f"Retrieved payload from local storage: {payload_hash}")
                return stored_data.get("payload", {})
            else:
                logger.warning(f"Payload not found in local storage: {payload_hash}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load payload from local storage: {e}")
            return {}
    
    def _retrieve_payload_from_blockchain(self, payload_hash: str) -> Dict[str, Any]:
        """Retrieve payload from blockchain events"""
        if not self.contract:
            logger.warning(f"No blockchain contract available, falling back to local storage for {payload_hash}")
            return self._load_payload_local(payload_hash)
            
        try:
            # Would retrieve from blockchain events/IPFS in production
            # Currently uses local storage as blockchain integration is pending
            return self._load_payload_local(payload_hash)
        except Exception as e:
            logger.error(f"Failed to retrieve from blockchain: {e}")
            return {}
    
    def _query_contract_storage_for_payload(self, payload_hash: str) -> Dict[str, Any]:
        """Query smart contract storage for payload"""
        if not self.contract:
            logger.warning(f"No contract available, falling back to local storage for {payload_hash}")
            return self._load_payload_local(payload_hash)
            
        # Contract storage integration pending
        return self._load_payload_local(payload_hash)
    
    def _retrieve_from_transaction_logs(self, payload_hash: str) -> Dict[str, Any]:
        """Retrieve payload from transaction logs"""
        if not self.w3 or not self.w3.is_connected():
            logger.warning(f"No blockchain connection, falling back to local storage for {payload_hash}")
            return self._load_payload_local(payload_hash)
            
        # Transaction log parsing integration pending
        return self._load_payload_local(payload_hash)
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Get blockchain connection information"""
        if not self.w3 or not self.w3.is_connected():
            return {"connected": False}
        
        return {
            "connected": True,
            "rpc_url": self.w3.provider.endpoint_uri,
            "chain_id": self.w3.eth.chain_id,
            "latest_block": self.w3.eth.block_number,
            "contract_address": self.contract_address,
            "registered_agents": len(self.agents),
            "gas_price": self.w3.eth.gas_price
        }


class BlockchainWorkflowRegistry(WorkflowRegistry):
    """Blockchain-backed Workflow Registry"""
    
    def __init__(self, blockchain_registry: BlockchainA2ARegistry, storage_path: Optional[Path] = None):
        super().__init__(storage_path)
        self.blockchain_registry = blockchain_registry
        self.contract = blockchain_registry.contract
        
    def register_workflow_blockchain(self, workflow, creator_private_key: str) -> str:
        """Register workflow on blockchain"""
        if not self.contract:
            logger.error("No blockchain contract available")
            return None
        
        # Create account from private key
        account = Account.from_key(creator_private_key)
        
        try:
            # Encode workflow steps for blockchain storage
            steps_data = self._encode_workflow_steps(workflow.steps)
            
            tx = self.contract.functions.registerWorkflow(
                workflow.workflow_id,
                workflow.name,
                workflow.description,
                steps_data
            ).build_transaction({
                'from': account.address,
                'gas': 1000000,
                'gasPrice': self.blockchain_registry.w3.eth.gas_price,
                'nonce': self.blockchain_registry.w3.eth.get_transaction_count(account.address)
            })
            
            signed_tx = self.blockchain_registry.w3.eth.account.sign_transaction(
                tx, creator_private_key
            )
            tx_hash = self.blockchain_registry.w3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Register locally
            self.register(workflow)
            
            logger.info(f"Workflow {workflow.workflow_id} registered on blockchain")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to register workflow on blockchain: {e}")
            return None
    
    def _encode_workflow_steps(self, steps) -> bytes:
        """Encode workflow steps for blockchain storage"""
        # Simplified encoding - in production use proper ABI encoding
        steps_data = []
        for step in steps:
            step_dict = {
                'stepId': step.step_id,
                'agentId': step.agent_id,
                'action': step.action,
                'inputData': step.input_data,
                'dependsOn': step.depends_on,
                'timeout': step.timeout,
                'retryCount': step.retry_count
            }
            steps_data.append(step_dict)
        
        steps_json = json.dumps(steps_data)
        return steps_json.encode('utf-8')


# Create global blockchain-integrated registries
blockchain_registry = None
blockchain_workflow_registry = None

def initialize_blockchain_registries(contract_address: str = None, rpc_url: str = "http://127.0.0.1:8545"):
    """Initialize blockchain-integrated registries"""
    global blockchain_registry, blockchain_workflow_registry
    
    blockchain_registry = BlockchainA2ARegistry(contract_address, rpc_url)
    blockchain_workflow_registry = BlockchainWorkflowRegistry(blockchain_registry)
    
    logger.info("Blockchain registries initialized")
    return blockchain_registry, blockchain_workflow_registry

def get_blockchain_registries():
    """Get the global blockchain registries"""
    return blockchain_registry, blockchain_workflow_registry