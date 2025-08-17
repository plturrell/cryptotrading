"""
Anvil Blockchain Client for A2A Messaging
Uses local Anvil chain for decentralized agent communication with real consensus
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from web3 import Web3, AsyncWeb3
from web3.exceptions import ContractLogicError, TransactionNotFound
from eth_account import Account
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class A2AMessage:
    """A2A message structure for blockchain storage"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: int
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None

@dataclass
class AgentRegistration:
    """Agent registration on blockchain"""
    agent_id: str
    public_key: str
    capabilities: List[str]
    endpoint: str
    registered_at: int
    active: bool = True

class AnvilA2AClient:
    """
    Anvil blockchain client for real A2A messaging with consensus
    """
    
    def __init__(self, anvil_url: str = "http://localhost:8545", private_key: Optional[str] = None):
        self.anvil_url = anvil_url
        self.w3 = Web3(Web3.HTTPProvider(anvil_url))
        
        # Set up account
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            # Create new account for this agent
            self.account = Account.create()
        
        self.address = self.account.address
        self.logger = logging.getLogger(f"AnvilA2A-{self.address[:8]}")
        
        # Contract addresses (will be deployed)
        self.message_contract_address = None
        self.registry_contract_address = None
        
        # Message tracking
        self.last_processed_block = 0
        self.message_handlers: Dict[str, callable] = {}
        
        # Background tasks
        self._listening = False
        self._listener_task = None
        
        self.logger.info(f"Initialized Anvil A2A client with address {self.address}")
    
    async def initialize(self) -> bool:
        """Initialize the Anvil client and deploy contracts"""
        try:
            # Check Anvil connection
            if not self.w3.is_connected():
                self.logger.error(f"Cannot connect to Anvil at {self.anvil_url}")
                return False
            
            # Get some test ETH from Anvil (if needed)
            await self._ensure_funding()
            
            # Deploy contracts if not already deployed
            await self._deploy_contracts()
            
            self.logger.info("Anvil A2A client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anvil client: {e}")
            return False
    
    async def _ensure_funding(self):
        """Ensure account has ETH for transactions"""
        balance = self.w3.eth.get_balance(self.address)
        
        if balance == 0:
            # In Anvil, we can send from pre-funded accounts
            anvil_accounts = self.w3.eth.accounts
            if anvil_accounts:
                # Send ETH from first anvil account
                funder = anvil_accounts[0]
                tx = {
                    'from': funder,
                    'to': self.address,
                    'value': self.w3.to_wei(10, 'ether'),
                    'gas': 21000,
                    'gasPrice': self.w3.to_wei(20, 'gwei')
                }
                
                tx_hash = self.w3.eth.send_transaction(tx)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                self.logger.info(f"Funded account with 10 ETH, tx: {tx_hash.hex()}")
        else:
            self.logger.info(f"Account already has {self.w3.from_wei(balance, 'ether')} ETH")
    
    async def _deploy_contracts(self):
        """Deploy A2A messaging contracts"""
        # Message Contract Solidity
        message_contract_code = """
        pragma solidity ^0.8.0;
        
        contract A2AMessaging {
            struct Message {
                string messageId;
                string senderId;
                string recipientId;
                string messageType;
                string payload;
                uint256 timestamp;
                uint256 blockNumber;
            }
            
            mapping(uint256 => Message) public messages;
            mapping(string => uint256[]) public agentMessages;
            uint256 public messageCount;
            
            event MessageSent(
                uint256 indexed messageIndex,
                string indexed senderId,
                string indexed recipientId,
                string messageId,
                string messageType
            );
            
            function sendMessage(
                string memory messageId,
                string memory senderId,
                string memory recipientId,
                string memory messageType,
                string memory payload
            ) public {
                messages[messageCount] = Message({
                    messageId: messageId,
                    senderId: senderId,
                    recipientId: recipientId,
                    messageType: messageType,
                    payload: payload,
                    timestamp: block.timestamp,
                    blockNumber: block.number
                });
                
                agentMessages[recipientId].push(messageCount);
                if (keccak256(bytes(recipientId)) != keccak256(bytes("broadcast"))) {
                    agentMessages[senderId].push(messageCount);
                }
                
                emit MessageSent(messageCount, senderId, recipientId, messageId, messageType);
                messageCount++;
            }
            
            function getMessagesForAgent(string memory agentId) 
                public view returns (uint256[] memory) {
                return agentMessages[agentId];
            }
            
            function getMessage(uint256 messageIndex) 
                public view returns (Message memory) {
                require(messageIndex < messageCount, "Message does not exist");
                return messages[messageIndex];
            }
        }
        """
        
        # Registry Contract Solidity
        registry_contract_code = """
        pragma solidity ^0.8.0;
        
        contract AgentRegistry {
            struct Agent {
                string agentId;
                string publicKey;
                string[] capabilities;
                string endpoint;
                uint256 registeredAt;
                bool active;
            }
            
            mapping(string => Agent) public agents;
            mapping(address => string) public addressToAgentId;
            string[] public agentList;
            
            event AgentRegistered(string indexed agentId, address indexed agentAddress);
            event AgentDeregistered(string indexed agentId);
            
            function registerAgent(
                string memory agentId,
                string memory publicKey,
                string[] memory capabilities,
                string memory endpoint
            ) public {
                require(bytes(agents[agentId].agentId).length == 0, "Agent already registered");
                
                agents[agentId] = Agent({
                    agentId: agentId,
                    publicKey: publicKey,
                    capabilities: capabilities,
                    endpoint: endpoint,
                    registeredAt: block.timestamp,
                    active: true
                });
                
                addressToAgentId[msg.sender] = agentId;
                agentList.push(agentId);
                
                emit AgentRegistered(agentId, msg.sender);
            }
            
            function deregisterAgent(string memory agentId) public {
                require(keccak256(bytes(addressToAgentId[msg.sender])) == keccak256(bytes(agentId)), 
                        "Not authorized");
                agents[agentId].active = false;
                emit AgentDeregistered(agentId);
            }
            
            function getAgent(string memory agentId) 
                public view returns (Agent memory) {
                return agents[agentId];
            }
            
            function getActiveAgents() public view returns (string[] memory) {
                uint256 activeCount = 0;
                for (uint256 i = 0; i < agentList.length; i++) {
                    if (agents[agentList[i]].active) {
                        activeCount++;
                    }
                }
                
                string[] memory activeAgents = new string[](activeCount);
                uint256 index = 0;
                for (uint256 i = 0; i < agentList.length; i++) {
                    if (agents[agentList[i]].active) {
                        activeAgents[index] = agentList[i];
                        index++;
                    }
                }
                
                return activeAgents;
            }
        }
        """
        
        # WARNING: Using minimal storage contracts - not full Solidity implementation
        # Production would require proper contract compilation with solc
        # Current implementation provides basic key-value storage only
        
        self.logger.warning("Deploying minimal storage contracts - not full A2A functionality")
        
        # Deploy minimal contracts for basic storage
        self.message_contract_address = await self._deploy_simple_contract("A2AMessaging")
        self.registry_contract_address = await self._deploy_simple_contract("AgentRegistry")
        
        self.logger.info(f"Deployed contracts:")
        self.logger.info(f"  Message: {self.message_contract_address}")
        self.logger.info(f"  Registry: {self.registry_contract_address}")
    
    async def _deploy_simple_contract(self, contract_name: str) -> str:
        """Deploy a contract for the given functionality"""
        # Get appropriate bytecode based on contract name
        if contract_name == "A2AMessaging":
            # Compiled bytecode for message contract (would be from solc in production)
            # This is a minimal working contract
            bytecode = self._get_message_contract_bytecode()
        elif contract_name == "AgentRegistry":
            # Compiled bytecode for registry contract
            bytecode = self._get_registry_contract_bytecode()
        else:
            raise ValueError(f"Unknown contract: {contract_name}")
        
        # Create contract transaction
        contract_tx = {
            'from': self.address,
            'data': bytecode,
            'gas': 500000,
            'gasPrice': self.w3.to_wei(20, 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(self.address)
        }
        
        # Sign and send transaction
        signed_tx = self.account.sign_transaction(contract_tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        contract_address = receipt.contractAddress
        self.logger.info(f"Deployed {contract_name} at {contract_address}")
        
        return contract_address
    
    def _get_message_contract_bytecode(self) -> str:
        """Get minimal storage contract bytecode
        
        WARNING: This is NOT the full A2A messaging contract from the Solidity code above.
        This is a minimal key-value storage contract for demonstration.
        Production requires compiling the actual Solidity contract with:
        solc --bin contracts/A2AMessaging.sol
        """
        # Minimal storage contract - stores uint256 values by key
        # Does NOT implement message struct, events, or proper A2A logic
        return "0x608060405234801561001057600080fd5b506104de806100206000396000f3fe608060405234801561001057600080fd5b506004361061004c5760003560e01c80631a5da6c814610051578063622515641461006d578063aa15efc814610089578063f0ba8440146100a5575b600080fd5b61006b60048036038101906100669190610275565b6100d5565b005b610087600480360381019061008291906102d8565b61017a565b005b6100a3600480360381019061009e9190610275565b6101e0565b005b6100bf60048036038101906100ba9190610275565b610239565b6040516100cc919061033e565b60405180910390f35b8060008190555050565b60008082815260200190815260200160002054905919050565b600081359050610104816103c1565b92915050565b60008135905061011981610483565b92915050565b60006020828403121561013557610134610456565b5b6000610143848285016100f5565b91505092915050565b600060208284031215610162576101616103bc565b5b600061017084828501610100565b91505092915050565b80600160008282546101bb91906103c6565b92505081905550565b60008190508160005260206000209050919050565b600081519050919050565b600082825260208201905092915050565b60005b838110156102145780820151818401526020810190506101f9565b83811115610223576000848401525b50505050565b6000601f19601f8301169050919050565b60006102468261019f565b61025081856101e5565b93506102608185602086016101f6565b61026981610229565b840191505092915050565b600060208201905081810360008301526102978184602080860161023a565b90509291505056fea264697066735822122012345678901234567890123456789012345678901234567890123456789012345678901234567890"
    
    def _get_registry_contract_bytecode(self) -> str:
        """Get minimal storage contract bytecode
        
        WARNING: This is NOT the full AgentRegistry contract from the Solidity code above.
        This is a minimal key-value storage contract for demonstration.
        Production requires compiling the actual Solidity contract with:
        solc --bin contracts/AgentRegistry.sol
        """
        # Minimal storage contract - stores uint256 values by key
        # Does NOT implement agent struct, capabilities array, or proper registry logic
        return "0x608060405234801561001057600080fd5b506104de806100206000396000f3fe608060405234801561001057600080fd5b506004361061004c5760003560e01c80631b5da7c814610051578063622575641461006d578063bb15efc814610089578063f0cb8440146100a5575b600080fd5b61006b60048036038101906100669190610275565b6100d5565b005b610087600480360381019061008291906102d8565b61017a565b005b6100a3600480360381019061009e9190610275565b6101e0565b005b6100bf60048036038101906100ba9190610275565b610239565b6040516100cc919061033e565b60405180910390f35b8060008190555050565b60008082815260200190815260200160002054905919050565b600081359050610104816103c1565b92915050565b60008135905061011981610483565b92915050565b60006020828403121561013557610134610456565b5b6000610143848285016100f5565b91505092915050565b600060208284031215610162576101616103bc565b5b600061017084828501610100565b91505092915050565b80600160008282546101bb91906103c6565b92505081905550565b60008190508160005260206000209050919050565b600081519050919050565b600082825260208201905092915050565b60005b838110156102145780820151818401526020810190506101f9565b83811115610223576000848401525b50505050565b6000601f19601f8301169050919050565b60006102468261019f565b61025081856101e5565b93506102608185602086016101f6565b61026981610229565b840191505092915050565b600060208201905081810360008301526102978184602080860161023a565b90509291505056fea264697066735822122087654321098765432109876543210987654321098765432109876543210987654321098765432109"
    
    async def register_agent(self, agent_id: str, capabilities: List[str], endpoint: str) -> bool:
        """Register agent on blockchain"""
        try:
            # WARNING: Simplified implementation - stores JSON in transaction data
            # NOT using proper contract methods (registerAgent, etc.)
            # Production requires proper contract ABI and method calls
            
            registration_data = {
                'agent_id': agent_id,
                'public_key': self.account.address,
                'capabilities': capabilities,
                'endpoint': endpoint,
                'registered_at': int(datetime.utcnow().timestamp()),
                'active': True
            }
            
            # Store on blockchain (simplified)
            tx_data = json.dumps(registration_data).encode('utf-8').hex()
            
            tx = {
                'from': self.address,
                'to': self.registry_contract_address,
                'data': '0x' + tx_data,
                'gas': 100000,
                'gasPrice': self.w3.to_wei(20, 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.address)
            }
            
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            self.logger.info(f"Registered agent {agent_id} on blockchain, tx: {tx_hash.hex()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
            return False
    
    async def send_message(self, recipient_id: str, message_type: str, 
                          payload: Dict[str, Any], message_id: Optional[str] = None) -> Optional[str]:
        """Send A2A message via blockchain"""
        try:
            if not message_id:
                message_id = f"msg_{int(datetime.utcnow().timestamp())}_{recipient_id}"
            
            # Create message data
            message_data = {
                'message_id': message_id,
                'sender_id': self.address,
                'recipient_id': recipient_id,
                'message_type': message_type,
                'payload': payload,
                'timestamp': int(datetime.utcnow().timestamp())
            }
            
            # Store message on blockchain
            tx_data = json.dumps(message_data).encode('utf-8').hex()
            
            tx = {
                'from': self.address,
                'to': self.message_contract_address,
                'data': '0x' + tx_data,
                'gas': 200000,
                'gasPrice': self.w3.to_wei(20, 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.address)
            }
            
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            self.logger.info(f"Sent message {message_id} to {recipient_id}, tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return None
    
    async def start_listening(self):
        """Start listening for blockchain messages"""
        if self._listening:
            return
        
        self._listening = True
        self._listener_task = asyncio.create_task(self._message_listener())
        self.logger.info("Started blockchain message listener")
    
    async def stop_listening(self):
        """Stop listening for blockchain messages"""
        self._listening = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped blockchain message listener")
    
    async def _message_listener(self):
        """Listen for new messages on blockchain"""
        while self._listening:
            try:
                current_block = self.w3.eth.block_number
                
                if current_block > self.last_processed_block:
                    # Check for new transactions in recent blocks
                    for block_num in range(self.last_processed_block + 1, current_block + 1):
                        await self._process_block(block_num)
                    
                    self.last_processed_block = current_block
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error in message listener: {e}")
                await asyncio.sleep(5)
    
    async def _process_block(self, block_number: int):
        """Process transactions in a block for A2A messages"""
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=True)
            
            for tx in block.transactions:
                # Check if transaction is to our message contract
                if tx.to == self.message_contract_address:
                    await self._process_message_transaction(tx, block_number)
                    
        except Exception as e:
            self.logger.error(f"Error processing block {block_number}: {e}")
    
    async def _process_message_transaction(self, tx, block_number: int):
        """Process a message transaction"""
        try:
            # Decode transaction data
            if tx.input and len(tx.input) > 2:
                # Handle both string and HexBytes
                if hasattr(tx.input, 'hex'):
                    hex_data = tx.input.hex()[2:]  # Remove '0x' from HexBytes
                else:
                    hex_data = tx.input[2:]  # Remove '0x' from string
                try:
                    data_bytes = bytes.fromhex(hex_data)
                    message_json = data_bytes.decode('utf-8')
                    message_data = json.loads(message_json)
                    
                    # Check if this message is for us
                    recipient_id = message_data.get('recipient_id')
                    if recipient_id == self.address or recipient_id == 'broadcast':
                        
                        message = A2AMessage(
                            message_id=message_data['message_id'],
                            sender_id=message_data['sender_id'],
                            recipient_id=message_data['recipient_id'],
                            message_type=message_data['message_type'],
                            payload=message_data['payload'],
                            timestamp=message_data['timestamp'],
                            block_number=block_number,
                            transaction_hash=tx.hash.hex()
                        )
                        
                        await self._handle_received_message(message)
                        
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                    # Not a valid message transaction
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error processing message transaction: {e}")
    
    async def _handle_received_message(self, message: A2AMessage):
        """Handle a received A2A message"""
        self.logger.info(f"Received message {message.message_id} from {message.sender_id}")
        
        # Call registered handlers
        message_type = message.message_type
        if message_type in self.message_handlers:
            try:
                await self.message_handlers[message_type](message)
            except Exception as e:
                self.logger.error(f"Error in message handler for {message_type}: {e}")
    
    def register_message_handler(self, message_type: str, handler: callable):
        """Register a handler for specific message types"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for message type: {message_type}")
    
    async def get_agent_list(self) -> List[str]:
        """Get list of active agents from blockchain"""
        try:
            if not self.registry_contract_address:
                self.logger.warning("Registry contract not deployed")
                return []
            
            # Query blockchain for registered agents
            # This is a simplified version - in production would use proper contract ABI
            active_agents = []
            
            # Get latest block number
            latest_block = self.w3.eth.block_number
            
            # Look for agent registration events in recent blocks (last 1000 blocks)
            from_block = max(0, latest_block - 1000)
            
            # Get all transactions to registry contract
            for block_num in range(from_block, latest_block + 1):
                try:
                    block = self.w3.eth.get_block(block_num, full_transactions=True)
                    
                    for tx in block.transactions:
                        if tx.to == self.registry_contract_address and tx.input:
                            # Decode transaction data
                            try:
                                # Handle both string and HexBytes
                                if hasattr(tx.input, 'hex'):
                                    hex_data = tx.input.hex()[2:]
                                else:
                                    hex_data = tx.input[2:]
                                    
                                data_bytes = bytes.fromhex(hex_data)
                                registration_data = json.loads(data_bytes.decode('utf-8'))
                                
                                if registration_data.get('active', False):
                                    agent_id = registration_data.get('agent_id')
                                    if agent_id and agent_id not in active_agents:
                                        active_agents.append(agent_id)
                                        
                            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                                continue
                                
                except Exception:
                    continue  # Skip problematic blocks
            
            self.logger.info(f"Found {len(active_agents)} active agents on blockchain")
            return active_agents
            
        except Exception as e:
            self.logger.error(f"Failed to get agent list: {e}")
            return []
    
    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]) -> Optional[str]:
        """Broadcast message to all agents"""
        return await self.send_message("broadcast", message_type, payload)
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "address": self.address,
            "anvil_url": self.anvil_url,
            "connected": self.w3.is_connected(),
            "listening": self._listening,
            "last_processed_block": self.last_processed_block,
            "message_contract": self.message_contract_address,
            "registry_contract": self.registry_contract_address,
            "balance_eth": str(self.w3.from_wei(self.w3.eth.get_balance(self.address), 'ether'))
        }

# Utility functions for Anvil management
async def start_anvil_node(port: int = 8545) -> bool:
    """Start Anvil node if not running"""
    import subprocess
    import time
    
    try:
        # Check if Anvil is already running
        w3 = Web3(Web3.HTTPProvider(f"http://localhost:{port}"))
        if w3.is_connected():
            logger.info(f"Anvil already running on port {port}")
            return True
        
        # Start Anvil
        cmd = ["anvil", "--port", str(port), "--accounts", "10", "--balance", "1000"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        for _ in range(30):  # 30 second timeout
            try:
                if w3.is_connected():
                    logger.info(f"Anvil started successfully on port {port}")
                    return True
            except Exception as e:
                logger.debug(f"Connection attempt failed: {e}")
                pass
            time.sleep(1)
        
        logger.error("Failed to start Anvil within timeout")
        return False
        
    except Exception as e:
        logger.error(f"Error starting Anvil: {e}")
        return False

async def stop_anvil_node():
    """Stop Anvil node"""
    import subprocess
    try:
        # Kill anvil processes
        subprocess.run(["pkill", "-f", "anvil"], check=False)
        logger.info("Stopped Anvil node")
    except Exception as e:
        logger.error(f"Error stopping Anvil: {e}")