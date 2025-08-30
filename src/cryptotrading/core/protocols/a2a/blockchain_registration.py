"""
Blockchain Registration Service for A2A Agents
Integrates with the A2A Registry smart contract for on-chain agent registration
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from eth_account import Account
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError

from ...blockchain.anvil_client import AnvilA2AClient
from .a2a_protocol import AgentStatus, A2AAgentRegistry

logger = logging.getLogger(__name__)


@dataclass
class BlockchainAgentRegistration:
    """Blockchain agent registration structure"""
    
    agent_id: str
    wallet_address: str
    agent_type: str
    capabilities: List[str]
    mcp_tools: List[str]
    status: str
    registered_at: int
    ipfs_skill_card: str
    compliance_score: int
    transaction_hash: str
    block_number: int


class BlockchainRegistrationService:
    """Service for registering A2A agents on blockchain"""
    
    def __init__(
        self,
        anvil_url: str = "http://localhost:8545",
        private_key: Optional[str] = None,
        registry_contract_address: Optional[str] = None
    ):
        """Initialize blockchain registration service"""
        self.anvil_url = anvil_url
        self.w3 = Web3(Web3.HTTPProvider(anvil_url))
        
        # Set up account (Agent Manager)
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            # Use environment variable or create new account
            private_key = os.getenv("AGENT_MANAGER_PRIVATE_KEY")
            if private_key:
                self.account = Account.from_key(private_key)
            else:
                self.account = Account.create()
                logger.warning(f"Created new Agent Manager account: {self.account.address}")
                logger.warning(f"Private key: {self.account.key.hex()}")
        
        self.address = self.account.address
        self.w3.eth.default_account = self.address
        
        # Contract setup
        self.registry_contract_address = registry_contract_address
        self.registry_contract: Optional[Contract] = None
        
        # Load contract ABI
        self.registry_abi = self._load_registry_abi()
        
        logger.info(f"Blockchain Registration Service initialized with address: {self.address}")
    
    def _load_registry_abi(self) -> List[Dict]:
        """Load the A2A Registry contract ABI"""
        # This would normally be loaded from compiled contract artifacts
        # For now, we'll define the essential functions
        return [
            {
                "inputs": [
                    {"name": "agentId", "type": "string"},
                    {"name": "walletAddress", "type": "address"},
                    {"name": "agentType", "type": "string"},
                    {"name": "capabilities", "type": "string[]"},
                    {"name": "mcpTools", "type": "string[]"},
                    {"name": "ipfsSkillCard", "type": "string"}
                ],
                "name": "registerAgent",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "agentId", "type": "string"}],
                "name": "getAgent",
                "outputs": [
                    {
                        "components": [
                            {"name": "agentId", "type": "string"},
                            {"name": "walletAddress", "type": "address"},
                            {"name": "agentType", "type": "string"},
                            {"name": "capabilities", "type": "string[]"},
                            {"name": "mcpTools", "type": "string[]"},
                            {"name": "status", "type": "uint8"},
                            {"name": "registeredAt", "type": "uint256"},
                            {"name": "lastUpdated", "type": "uint256"},
                            {"name": "ipfsSkillCard", "type": "string"},
                            {"name": "complianceScore", "type": "uint256"}
                        ],
                        "name": "",
                        "type": "tuple"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "agentId", "type": "string"}],
                "name": "isAgentActive",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getAllAgentIds",
                "outputs": [{"name": "", "type": "string[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "agentId", "type": "string"},
                    {"name": "newStatus", "type": "uint8"}
                ],
                "name": "updateAgentStatus",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
    
    async def initialize_contract(self, contract_address: Optional[str] = None) -> bool:
        """Initialize the registry contract"""
        try:
            if contract_address:
                self.registry_contract_address = contract_address
            
            if not self.registry_contract_address:
                logger.error("No registry contract address provided")
                return False
            
            # Create contract instance
            self.registry_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.registry_contract_address),
                abi=self.registry_abi
            )
            
            # Test contract connection
            try:
                total_agents = self.registry_contract.functions.totalAgents().call()
                logger.info(f"Connected to A2A Registry contract. Total agents: {total_agents}")
                return True
            except Exception as e:
                logger.warning(f"Contract connection test failed: {e}")
                return True  # Continue anyway, might be a new deployment
            
        except Exception as e:
            logger.error(f"Failed to initialize contract: {e}")
            return False
    
    async def register_agent_on_blockchain(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        mcp_tools: List[str],
        wallet_address: Optional[str] = None,
        ipfs_skill_card: Optional[str] = None
    ) -> Optional[BlockchainAgentRegistration]:
        """Register an agent on the blockchain"""
        
        if not self.registry_contract:
            logger.error("Registry contract not initialized")
            return None
        
        # Generate wallet for agent if not provided
        if not wallet_address:
            agent_account = Account.create()
            wallet_address = agent_account.address
            logger.info(f"Generated wallet for agent {agent_id}: {wallet_address}")
        
        # Default IPFS skill card
        if not ipfs_skill_card:
            ipfs_skill_card = f"QmSkillCard{agent_id[:10]}"  # Mock IPFS hash
        
        try:
            # Check if agent already exists
            try:
                existing_agent = self.registry_contract.functions.getAgent(agent_id).call()
                if existing_agent[0]:  # agentId field
                    logger.warning(f"Agent {agent_id} already registered on blockchain")
                    return await self.get_agent_registration(agent_id)
            except ContractLogicError:
                # Agent doesn't exist, continue with registration
                pass
            
            # Build transaction
            tx_data = self.registry_contract.functions.registerAgent(
                agent_id,
                Web3.to_checksum_address(wallet_address),
                agent_type,
                capabilities,
                mcp_tools,
                ipfs_skill_card
            )
            
            # Get gas estimate
            try:
                gas_estimate = tx_data.estimate_gas({'from': self.address})
            except Exception as e:
                logger.error(f"Gas estimation failed: {e}")
                gas_estimate = 500000  # Default gas limit
            
            # Build transaction
            transaction = tx_data.build_transaction({
                'from': self.address,
                'gas': gas_estimate,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.address)
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            logger.info(f"Submitted blockchain registration for {agent_id}. TX: {tx_hash_hex}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"✅ Agent {agent_id} successfully registered on blockchain!")
                
                # Return registration details
                return BlockchainAgentRegistration(
                    agent_id=agent_id,
                    wallet_address=wallet_address,
                    agent_type=agent_type,
                    capabilities=capabilities,
                    mcp_tools=mcp_tools,
                    status="ACTIVE",
                    registered_at=int(datetime.now().timestamp()),
                    ipfs_skill_card=ipfs_skill_card,
                    compliance_score=100,
                    transaction_hash=tx_hash_hex,
                    block_number=receipt.blockNumber
                )
            else:
                logger.error(f"❌ Registration transaction failed for {agent_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id} on blockchain: {e}")
            return None
    
    async def get_agent_registration(self, agent_id: str) -> Optional[BlockchainAgentRegistration]:
        """Get agent registration from blockchain"""
        if not self.registry_contract:
            logger.error("Registry contract not initialized")
            return None
        
        try:
            agent_data = self.registry_contract.functions.getAgent(agent_id).call()
            
            return BlockchainAgentRegistration(
                agent_id=agent_data[0],
                wallet_address=agent_data[1],
                agent_type=agent_data[2],
                capabilities=list(agent_data[3]),
                mcp_tools=list(agent_data[4]),
                status=["INACTIVE", "ACTIVE", "SUSPENDED", "TERMINATED"][agent_data[5]],
                registered_at=agent_data[6],
                ipfs_skill_card=agent_data[8],
                compliance_score=agent_data[9],
                transaction_hash="",  # Not available from contract
                block_number=0  # Not available from contract
            )
            
        except ContractLogicError as e:
            logger.warning(f"Agent {agent_id} not found on blockchain: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get agent registration for {agent_id}: {e}")
            return None
    
    async def is_agent_active(self, agent_id: str) -> bool:
        """Check if agent is active on blockchain"""
        if not self.registry_contract:
            return False
        
        try:
            return self.registry_contract.functions.isAgentActive(agent_id).call()
        except Exception as e:
            logger.error(f"Failed to check agent status for {agent_id}: {e}")
            return False
    
    async def get_all_registered_agents(self) -> List[str]:
        """Get all registered agent IDs from blockchain"""
        if not self.registry_contract:
            return []
        
        try:
            return self.registry_contract.functions.getAllAgentIds().call()
        except Exception as e:
            logger.error(f"Failed to get all registered agents: {e}")
            return []
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status on blockchain"""
        if not self.registry_contract:
            logger.error("Registry contract not initialized")
            return False
        
        try:
            # Map status to contract enum
            status_map = {
                AgentStatus.INACTIVE: 0,
                AgentStatus.ACTIVE: 1,
                # Add more status mappings as needed
            }
            
            status_int = status_map.get(status, 1)  # Default to ACTIVE
            
            # Build and send transaction
            tx_data = self.registry_contract.functions.updateAgentStatus(agent_id, status_int)
            
            gas_estimate = tx_data.estimate_gas({'from': self.address})
            transaction = tx_data.build_transaction({
                'from': self.address,
                'gas': gas_estimate,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.address)
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"✅ Updated status for agent {agent_id} to {status}")
                return True
            else:
                logger.error(f"❌ Failed to update status for agent {agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update agent status for {agent_id}: {e}")
            return False


class EnhancedA2AAgentRegistry(A2AAgentRegistry):
    """Enhanced A2A Agent Registry with blockchain integration"""
    
    _blockchain_service: Optional[BlockchainRegistrationService] = None
    
    @classmethod
    def set_blockchain_service(cls, service: BlockchainRegistrationService):
        """Set the blockchain service for registry"""
        cls._blockchain_service = service
        logger.info("Blockchain service configured for A2A Agent Registry")
    
    @classmethod
    async def register_agent_with_blockchain(
        cls,
        agent_id: str,
        capabilities: List[str],
        agent_instance=None,
        agent_type: str = "unknown",
        mcp_tools: List[str] = None,
        status: AgentStatus = AgentStatus.ACTIVE,
    ):
        """Register agent both locally and on blockchain"""
        
        # Register locally first
        cls.register_agent(agent_id, capabilities, agent_instance, status)
        
        # Register on blockchain if service available
        if cls._blockchain_service:
            try:
                blockchain_registration = await cls._blockchain_service.register_agent_on_blockchain(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    capabilities=capabilities,
                    mcp_tools=mcp_tools or []
                )
                
                if blockchain_registration:
                    # Update local registry with blockchain info
                    if agent_id in cls._registered_agents:
                        cls._registered_agents[agent_id].update({
                            "blockchain_registered": True,
                            "wallet_address": blockchain_registration.wallet_address,
                            "transaction_hash": blockchain_registration.transaction_hash,
                            "block_number": blockchain_registration.block_number
                        })
                    
                    logger.info(f"🔗 Agent {agent_id} registered on blockchain and locally")
                    return True
                else:
                    logger.error(f"Failed to register {agent_id} on blockchain")
                    return False
            except Exception as e:
                logger.error(f"Blockchain registration failed for {agent_id}: {e}")
                return False
        else:
            logger.warning(f"No blockchain service configured - {agent_id} registered locally only")
            return True


# Global blockchain registration service
_blockchain_registration_service: Optional[BlockchainRegistrationService] = None


def get_blockchain_registration_service() -> Optional[BlockchainRegistrationService]:
    """Get or create blockchain registration service"""
    global _blockchain_registration_service
    if _blockchain_registration_service is None:
        try:
            _blockchain_registration_service = BlockchainRegistrationService()
            logger.info("Blockchain registration service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize blockchain service: {e}")
    return _blockchain_registration_service


async def initialize_blockchain_integration(
    anvil_url: str = "http://localhost:8545",
    registry_contract_address: Optional[str] = None
) -> bool:
    """Initialize blockchain integration for A2A agents"""
    try:
        service = BlockchainRegistrationService(anvil_url=anvil_url)
        
        if registry_contract_address:
            success = await service.initialize_contract(registry_contract_address)
            if not success:
                logger.error("Failed to initialize contract")
                return False
        
        # Set the service in the enhanced registry
        EnhancedA2AAgentRegistry.set_blockchain_service(service)
        
        global _blockchain_registration_service
        _blockchain_registration_service = service
        
        logger.info("🔗 Blockchain integration initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize blockchain integration: {e}")
        return False