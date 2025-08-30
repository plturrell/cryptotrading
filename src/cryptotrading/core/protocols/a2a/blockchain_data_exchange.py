"""
Blockchain Data Exchange Service for A2A Agents
Provides on-chain data storage and exchange between agents
"""

import asyncio
import json
import logging
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import TransactionNotFound, BlockNotFound
from eth_account import Account
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.warning("tenacity not available - retry logic disabled")

logger = logging.getLogger(__name__)


class DataStatus(Enum):
    """Data status enum matching smart contract"""
    PENDING = 0
    AVAILABLE = 1
    PROCESSING = 2
    CONSUMED = 3
    EXPIRED = 4
    DELETED = 5


class WorkflowStatus(Enum):
    """Workflow status enum matching smart contract"""
    INITIATED = 0
    IN_PROGRESS = 1
    AWAITING_DATA = 2
    PROCESSING = 3
    COMPLETED = 4
    FAILED = 5
    CANCELLED = 6


class TransactionStatus(Enum):
    """Transaction status tracking"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Config:
    """Configuration for blockchain service"""
    # Network settings
    DEFAULT_RPC_URL: str = "http://localhost:8545"
    FALLBACK_RPC_URLS: List[str] = field(default_factory=lambda: [
        "http://localhost:8546"
    ])
    
    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_WAIT_MIN: int = 1  # seconds
    RETRY_WAIT_MAX: int = 10  # seconds
    
    # Transaction settings
    GAS_BUFFER_MULTIPLIER: float = 1.2
    MAX_GAS_PRICE_GWEI: int = 500
    TRANSACTION_TIMEOUT: int = 120  # seconds
    CONFIRMATION_BLOCKS: int = 2
    
    # Data settings
    MAX_DATA_SIZE: int = 256 * 1024  # 256KB
    COMPRESSION_THRESHOLD: int = 1024  # 1KB
    CACHE_TTL_MINUTES: int = 5
    MAX_CACHE_SIZE: int = 100
    
    # Monitoring
    METRICS_ENABLED: bool = True
    HEALTH_CHECK_INTERVAL: int = 60  # seconds


@dataclass
class TransactionMetrics:
    """Metrics for transaction monitoring"""
    total_transactions: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0
    total_gas_used: int = 0
    average_gas_price: float = 0.0
    average_confirmation_time: float = 0.0
    last_transaction_time: Optional[datetime] = None
    
    def update(self, success: bool, gas_used: int = 0, gas_price: int = 0, confirmation_time: float = 0):
        """Update metrics with new transaction data"""
        self.total_transactions += 1
        if success:
            self.successful_transactions += 1
        else:
            self.failed_transactions += 1
        
        if gas_used:
            self.total_gas_used += gas_used
        
        if gas_price and self.total_transactions > 0:
            self.average_gas_price = (
                (self.average_gas_price * (self.total_transactions - 1) + gas_price) /
                self.total_transactions
            )
        
        if confirmation_time and success:
            successful_count = self.successful_transactions
            if successful_count > 1:
                self.average_confirmation_time = (
                    (self.average_confirmation_time * (successful_count - 1) + confirmation_time) /
                    successful_count
                )
            else:
                self.average_confirmation_time = confirmation_time
        
        self.last_transaction_time = datetime.now()


@dataclass
class DataPacket:
    """On-chain data packet representation"""
    data_id: int
    sender_agent_id: str
    receiver_agent_id: str
    data: bytes
    data_type: str
    timestamp: int
    is_encrypted: bool
    data_hash: str
    status: DataStatus
    expires_at: int
    
    @classmethod
    def from_chain(cls, data: tuple) -> 'DataPacket':
        """Create DataPacket from smart contract response"""
        return cls(
            data_id=data[0],
            sender_agent_id=data[1],
            receiver_agent_id=data[2],
            data=data[3],
            data_type=data[4],
            timestamp=data[5],
            is_encrypted=data[6],
            data_hash=data[7],
            status=DataStatus(data[8]),
            expires_at=data[9]
        )


@dataclass
class WorkflowData:
    """On-chain workflow representation"""
    workflow_id: int
    workflow_type: str
    participant_agents: List[str]
    data_packet_ids: List[int]
    status: WorkflowStatus
    created_at: int
    result: bytes
    
    @classmethod
    def from_chain(cls, data: tuple) -> 'WorkflowData':
        """Create WorkflowData from smart contract response"""
        return cls(
            workflow_id=data[0],
            workflow_type=data[1],
            participant_agents=list(data[2]),
            data_packet_ids=list(data[3]),
            status=WorkflowStatus(data[4]),
            created_at=data[5],
            result=data[6]
        )


class ConnectionPool:
    """Web3 connection pool for reliability"""
    
    def __init__(self, primary_url: str, fallback_urls: List[str], max_connections: int = 10):
        """Initialize connection pool"""
        self.primary_url = primary_url
        self.fallback_urls = fallback_urls
        self.max_connections = max_connections
        self.connections: Deque[Web3] = deque(maxlen=max_connections)
        self.healthy_urls: List[str] = []
        self._lock = asyncio.Lock()
        
    async def get_connection(self) -> Optional[Web3]:
        """Get a healthy Web3 connection"""
        async with self._lock:
            # Try to reuse existing connection
            if self.connections:
                w3 = self.connections.popleft()
                if await self._check_connection(w3):
                    self.connections.append(w3)  # Put back at end
                    return w3
            
            # Create new connection
            for url in [self.primary_url] + self.fallback_urls:
                try:
                    w3 = Web3(Web3.HTTPProvider(url))
                    if await self._check_connection(w3):
                        if len(self.connections) < self.max_connections:
                            self.connections.append(w3)
                        return w3
                except Exception as e:
                    logger.warning(f"Failed to connect to {url}: {e}")
                    continue
            
            return None
    
    async def _check_connection(self, w3: Web3) -> bool:
        """Check if Web3 connection is healthy"""
        try:
            if w3.is_connected():
                block = w3.eth.get_block('latest')
                return block is not None
        except Exception:
            pass
        return False
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all configured URLs"""
        health = {}
        for url in [self.primary_url] + self.fallback_urls:
            try:
                w3 = Web3(Web3.HTTPProvider(url))
                health[url] = await self._check_connection(w3)
            except Exception:
                health[url] = False
        self.healthy_urls = [url for url, healthy in health.items() if healthy]
        return health


class BlockchainDataExchangeService:
    """Production-ready service for on-chain data storage and exchange between A2A agents"""
    
    def __init__(
        self,
        web3_provider: str = "http://localhost:8545",
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """Initialize blockchain data exchange service with production features"""
        self.config = config or Config()
        self.config.DEFAULT_RPC_URL = web3_provider
        self.contract_address = contract_address
        self.private_key = private_key
        self.account = Account.from_key(private_key) if private_key else None
        
        # Connection management
        self.connection_pool = ConnectionPool(
            self.config.DEFAULT_RPC_URL,
            self.config.FALLBACK_RPC_URLS,
            max_connections=10
        )
        
        # Primary connection for backwards compatibility
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.contract: Optional[Contract] = None
        self.contract_abi: Optional[List[Dict[str, Any]]] = None
        
        # Cache for recent data operations
        self._data_cache: Dict[int, Tuple[DataPacket, datetime]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Workflow tracking
        self._active_workflows: Dict[int, WorkflowData] = {}
        
        # Metrics
        self.metrics = TransactionMetrics()
        
        # Transaction tracking
        self._pending_transactions: Dict[str, Dict[str, Any]] = {}
        
        # Health monitoring
        self._last_health_check: Optional[datetime] = None
        self._is_healthy = True
        
        logger.info(f"BlockchainDataExchangeService initialized with provider: {web3_provider}")
    
    async def initialize(self, contract_abi: List[Dict[str, Any]]) -> bool:
        """Initialize connection to smart contract"""
        try:
            if not self.contract_address:
                logger.error("No contract address provided")
                return False
            
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address),
                abi=contract_abi
            )
            
            logger.info(f"Connected to A2ADataExchange contract at {self.contract_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize contract: {e}")
            return False
    
    def _compress_data(self, data: bytes) -> Tuple[bytes, bool]:
        """Compress data if it exceeds threshold"""
        if len(data) > self.COMPRESSION_THRESHOLD:
            compressed = zlib.compress(data, level=9)
            if len(compressed) < len(data):
                return compressed, True
        return data, False
    
    def _decompress_data(self, data: bytes, is_compressed: bool) -> bytes:
        """Decompress data if compressed"""
        if is_compressed:
            return zlib.decompress(data)
        return data
    
    async def store_data(
        self,
        sender_agent_id: str,
        receiver_agent_id: str,
        data: Any,
        data_type: str,
        is_encrypted: bool = False,
        ttl: int = 0,
        compress: bool = True
    ) -> Optional[int]:
        """
        Store data on-chain for agent-to-agent exchange
        
        Args:
            sender_agent_id: ID of sending agent
            receiver_agent_id: ID of receiving agent
            data: Data to store (will be serialized)
            data_type: Type of data (e.g., 'market_data', 'analysis_result')
            is_encrypted: Whether data is encrypted
            ttl: Time to live in seconds (0 for default)
            compress: Whether to compress data
        
        Returns:
            Data ID if successful, None otherwise
        """
        try:
            # Serialize data
            if isinstance(data, (dict, list)):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Compress if requested and beneficial
            if compress:
                compressed_data, was_compressed = self._compress_data(data_bytes)
                if was_compressed:
                    data_bytes = compressed_data
                    data_type = f"compressed:{data_type}"
            
            # Check size limit
            if len(data_bytes) > self.MAX_DATA_SIZE:
                logger.error(f"Data size {len(data_bytes)} exceeds maximum {self.MAX_DATA_SIZE}")
                return None
            
            # Build transaction
            if not self.contract or not self.account:
                logger.error("Contract or account not initialized")
                return None
            
            # Estimate gas and build transaction
            function = self.contract.functions.storeData(
                sender_agent_id,
                receiver_agent_id,
                data_bytes,
                data_type,
                is_encrypted,
                ttl
            )
            
            # Get transaction parameters
            tx = await self._build_transaction(function)
            
            # Send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                # Extract data ID from events
                data_stored_event = self.contract.events.DataStored().process_receipt(receipt)
                if data_stored_event:
                    data_id = data_stored_event[0]['args']['dataId']
                    logger.info(f"Data stored on-chain with ID: {data_id}")
                    return data_id
            
            logger.error("Transaction failed")
            return None
            
        except Exception as e:
            logger.error(f"Failed to store data on-chain: {e}")
            return None
    
    async def retrieve_data(
        self,
        data_id: int,
        agent_id: str,
        decompress: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from on-chain storage
        
        Args:
            data_id: ID of data packet
            agent_id: ID of requesting agent
            decompress: Whether to decompress data
        
        Returns:
            Data dictionary if successful, None otherwise
        """
        try:
            # Check cache first
            if data_id in self._data_cache:
                cache_time = self._cache_timestamps.get(data_id)
                if cache_time and datetime.now() - cache_time < self._cache_ttl:
                    packet = self._data_cache[data_id]
                    return self._process_retrieved_data(packet, decompress)
            
            # Call contract
            if not self.contract:
                logger.error("Contract not initialized")
                return None
            
            result = self.contract.functions.retrieveData(data_id, agent_id).call()
            
            if result:
                data_bytes, data_type, is_encrypted = result
                
                # Handle compression
                if decompress and data_type.startswith("compressed:"):
                    data_bytes = self._decompress_data(data_bytes, True)
                    data_type = data_type.replace("compressed:", "")
                
                # Deserialize based on type
                if data_type in ['json', 'dict', 'list', 'market_data', 'analysis_result']:
                    data = json.loads(data_bytes.decode('utf-8'))
                else:
                    data = data_bytes.decode('utf-8')
                
                return {
                    'data_id': data_id,
                    'data': data,
                    'data_type': data_type,
                    'is_encrypted': is_encrypted
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve data: {e}")
            return None
    
    async def create_workflow(
        self,
        workflow_type: str,
        participants: List[str]
    ) -> Optional[int]:
        """
        Create a new workflow involving multiple agents
        
        Args:
            workflow_type: Type of workflow
            participants: List of participating agent IDs
        
        Returns:
            Workflow ID if successful, None otherwise
        """
        try:
            if not self.contract or not self.account:
                logger.error("Contract or account not initialized")
                return None
            
            # Build transaction
            function = self.contract.functions.createWorkflow(
                workflow_type,
                participants
            )
            
            tx = await self._build_transaction(function)
            
            # Send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                # Extract workflow ID from events
                workflow_event = self.contract.events.WorkflowCreated().process_receipt(receipt)
                if workflow_event:
                    workflow_id = workflow_event[0]['args']['workflowId']
                    logger.info(f"Workflow created with ID: {workflow_id}")
                    return workflow_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return None
    
    async def add_data_to_workflow(
        self,
        workflow_id: int,
        data_id: int,
        agent_id: str
    ) -> bool:
        """Add data to an existing workflow"""
        try:
            if not self.contract or not self.account:
                return False
            
            function = self.contract.functions.addDataToWorkflow(
                workflow_id,
                data_id,
                agent_id
            )
            
            tx = await self._build_transaction(function)
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt['status'] == 1
            
        except Exception as e:
            logger.error(f"Failed to add data to workflow: {e}")
            return False
    
    async def complete_workflow(
        self,
        workflow_id: int,
        result: Any,
        agent_id: str
    ) -> bool:
        """Complete a workflow with results"""
        try:
            if not self.contract or not self.account:
                return False
            
            # Serialize result
            if isinstance(result, (dict, list)):
                result_bytes = json.dumps(result).encode('utf-8')
            else:
                result_bytes = str(result).encode('utf-8')
            
            function = self.contract.functions.completeWorkflow(
                workflow_id,
                result_bytes,
                agent_id
            )
            
            tx = await self._build_transaction(function)
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"Workflow {workflow_id} completed successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to complete workflow: {e}")
            return False
    
    async def get_workflow(self, workflow_id: int) -> Optional[WorkflowData]:
        """Get workflow details"""
        try:
            if not self.contract:
                return None
            
            result = self.contract.functions.getWorkflow(workflow_id).call()
            if result:
                return WorkflowData.from_chain(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow: {e}")
            return None
    
    async def grant_data_access(
        self,
        data_id: int,
        granter_agent_id: str,
        target_agent_id: str,
        can_read: bool = True,
        can_write: bool = False,
        duration: int = 3600
    ) -> bool:
        """Grant data access to another agent"""
        try:
            if not self.contract or not self.account:
                return False
            
            function = self.contract.functions.grantDataAccess(
                data_id,
                granter_agent_id,
                target_agent_id,
                can_read,
                can_write,
                duration
            )
            
            tx = await self._build_transaction(function)
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt['status'] == 1
            
        except Exception as e:
            logger.error(f"Failed to grant data access: {e}")
            return False
    
    async def delete_data(
        self,
        data_id: int,
        agent_id: str
    ) -> bool:
        """Delete data from on-chain storage"""
        try:
            if not self.contract or not self.account:
                return False
            
            function = self.contract.functions.deleteData(
                data_id,
                agent_id
            )
            
            tx = await self._build_transaction(function)
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                # Remove from cache if present
                self._data_cache.pop(data_id, None)
                self._cache_timestamps.pop(data_id, None)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete data: {e}")
            return False
    
    async def get_agent_data(self, agent_id: str) -> List[int]:
        """Get all data IDs for an agent"""
        try:
            if not self.contract:
                return []
            
            data_ids = self.contract.functions.getAgentData(agent_id).call()
            return list(data_ids)
            
        except Exception as e:
            logger.error(f"Failed to get agent data: {e}")
            return []
    
    async def _build_transaction(self, function) -> Dict[str, Any]:
        """Build transaction with gas estimation"""
        try:
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Estimate gas
            gas_estimate = function.estimate_gas({'from': self.account.address})
            
            # Build transaction
            tx = function.build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': int(gas_estimate * 1.2),  # Add 20% buffer
                'gasPrice': self.w3.eth.gas_price
            })
            
            return tx
            
        except Exception as e:
            logger.error(f"Failed to build transaction: {e}")
            raise
    
    def _process_retrieved_data(self, packet: DataPacket, decompress: bool) -> Dict[str, Any]:
        """Process retrieved data packet"""
        data_bytes = packet.data
        data_type = packet.data_type
        
        # Handle compression
        if decompress and data_type.startswith("compressed:"):
            data_bytes = self._decompress_data(data_bytes, True)
            data_type = data_type.replace("compressed:", "")
        
        # Deserialize
        if data_type in ['json', 'dict', 'list']:
            data = json.loads(data_bytes.decode('utf-8'))
        else:
            data = data_bytes.decode('utf-8')
        
        return {
            'data_id': packet.data_id,
            'data': data,
            'data_type': data_type,
            'is_encrypted': packet.is_encrypted,
            'sender': packet.sender_agent_id,
            'receiver': packet.receiver_agent_id,
            'timestamp': packet.timestamp,
            'expires_at': packet.expires_at
        }


# Singleton instance
_data_exchange_service: Optional[BlockchainDataExchangeService] = None


async def get_data_exchange_service() -> BlockchainDataExchangeService:
    """Get or create the data exchange service singleton"""
    global _data_exchange_service
    
    if _data_exchange_service is None:
        _data_exchange_service = BlockchainDataExchangeService()
        # Load ABI and initialize when needed
        # await _data_exchange_service.initialize(contract_abi)
    
    return _data_exchange_service


async def store_agent_data(
    sender: str,
    receiver: str,
    data: Any,
    data_type: str = "general"
) -> Optional[int]:
    """Convenience function to store data on-chain"""
    service = await get_data_exchange_service()
    return await service.store_data(sender, receiver, data, data_type)


async def retrieve_agent_data(
    data_id: int,
    agent_id: str
) -> Optional[Dict[str, Any]]:
    """Convenience function to retrieve data from chain"""
    service = await get_data_exchange_service()
    return await service.retrieve_data(data_id, agent_id)