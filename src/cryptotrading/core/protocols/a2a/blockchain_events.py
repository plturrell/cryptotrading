"""
Blockchain Event Listeners for A2A Agent Registry
Monitors on-chain events for agent registrations, status updates, and messaging
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from web3 import Web3
from web3.contract import Contract
from web3.exceptions import BlockNotFound, TransactionNotFound

from .blockchain_registration import BlockchainRegistrationService
from .a2a_protocol import AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class BlockchainEvent:
    """Blockchain event data structure"""
    
    event_name: str
    block_number: int
    transaction_hash: str
    log_index: int
    args: Dict[str, Any]
    timestamp: datetime
    processed: bool = False


class BlockchainEventListener:
    """Listens to blockchain events from A2A Registry contract"""
    
    def __init__(
        self,
        blockchain_service: BlockchainRegistrationService,
        start_block: Optional[int] = None
    ):
        """Initialize blockchain event listener"""
        self.blockchain_service = blockchain_service
        self.w3 = blockchain_service.w3
        self.contract = blockchain_service.registry_contract
        
        # Event tracking
        self.last_processed_block = start_block or self._get_current_block_number()
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.processed_events: Set[str] = set()  # Track processed events by unique ID
        
        # Listening state
        self.listening = False
        self.listen_task: Optional[asyncio.Task] = None
        self.poll_interval = 5  # seconds
        
        # Thread pool for async event processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Blockchain Event Listener initialized from block {self.last_processed_block}")
    
    def _get_current_block_number(self) -> int:
        """Get current blockchain block number"""
        try:
            return self.w3.eth.block_number
        except Exception as e:
            logger.error(f"Failed to get current block number: {e}")
            return 0
    
    def register_event_handler(self, event_name: str, handler: Callable):
        """Register a handler for specific blockchain events"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        
        self.event_handlers[event_name].append(handler)
        logger.info(f"Registered handler for {event_name} events")
    
    def remove_event_handler(self, event_name: str, handler: Callable):
        """Remove an event handler"""
        if event_name in self.event_handlers:
            try:
                self.event_handlers[event_name].remove(handler)
                logger.info(f"Removed handler for {event_name} events")
            except ValueError:
                logger.warning(f"Handler not found for {event_name} events")
    
    async def start_listening(self):
        """Start listening to blockchain events"""
        if self.listening:
            logger.warning("Event listener already running")
            return
        
        if not self.contract:
            logger.error("No contract available for event listening")
            return
        
        self.listening = True
        self.listen_task = asyncio.create_task(self._listen_loop())
        logger.info("🎧 Blockchain event listener started")
    
    async def stop_listening(self):
        """Stop listening to blockchain events"""
        if not self.listening:
            return
        
        self.listening = False
        
        if self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        logger.info("🔇 Blockchain event listener stopped")
    
    async def _listen_loop(self):
        """Main event listening loop"""
        try:
            while self.listening:
                await self._process_new_blocks()
                await asyncio.sleep(self.poll_interval)
                
        except asyncio.CancelledError:
            logger.info("Event listening cancelled")
        except Exception as e:
            logger.error(f"Error in event listening loop: {e}")
            self.listening = False
    
    async def _process_new_blocks(self):
        """Process new blocks for events"""
        try:
            current_block = self._get_current_block_number()
            
            if current_block <= self.last_processed_block:
                return  # No new blocks
            
            # Process blocks in chunks to avoid overwhelming the node
            chunk_size = 100
            from_block = self.last_processed_block + 1
            
            while from_block <= current_block:
                to_block = min(from_block + chunk_size - 1, current_block)
                
                # Get events from this block range
                events = await self._get_events_in_range(from_block, to_block)
                
                # Process events
                for event in events:
                    await self._process_event(event)
                
                from_block = to_block + 1
            
            self.last_processed_block = current_block
            logger.debug(f"Processed blocks up to {current_block}")
            
        except Exception as e:
            logger.error(f"Error processing new blocks: {e}")
    
    async def _get_events_in_range(self, from_block: int, to_block: int) -> List[BlockchainEvent]:
        """Get all relevant events in a block range"""
        events = []
        
        try:
            # Get AgentRegistered events
            agent_registered_events = self.contract.events.AgentRegistered.create_filter(
                fromBlock=from_block, toBlock=to_block
            ).get_all_entries()
            
            for event in agent_registered_events:
                events.append(self._convert_web3_event(event, "AgentRegistered"))
            
            # Get AgentStatusUpdated events
            status_updated_events = self.contract.events.AgentStatusUpdated.create_filter(
                fromBlock=from_block, toBlock=to_block
            ).get_all_entries()
            
            for event in status_updated_events:
                events.append(self._convert_web3_event(event, "AgentStatusUpdated"))
            
            # Get CapabilitiesUpdated events
            capabilities_updated_events = self.contract.events.CapabilitiesUpdated.create_filter(
                fromBlock=from_block, toBlock=to_block
            ).get_all_entries()
            
            for event in capabilities_updated_events:
                events.append(self._convert_web3_event(event, "CapabilitiesUpdated"))
            
            # Get SkillCardUpdated events
            skill_card_events = self.contract.events.SkillCardUpdated.create_filter(
                fromBlock=from_block, toBlock=to_block
            ).get_all_entries()
            
            for event in skill_card_events:
                events.append(self._convert_web3_event(event, "SkillCardUpdated"))
                
        except Exception as e:
            logger.error(f"Error getting events in range {from_block}-{to_block}: {e}")
        
        return events
    
    def _convert_web3_event(self, web3_event: Any, event_name: str) -> BlockchainEvent:
        """Convert Web3 event to our BlockchainEvent format"""
        return BlockchainEvent(
            event_name=event_name,
            block_number=web3_event.blockNumber,
            transaction_hash=web3_event.transactionHash.hex(),
            log_index=web3_event.logIndex,
            args=dict(web3_event.args),
            timestamp=datetime.now()  # Would ideally get block timestamp
        )
    
    async def _process_event(self, event: BlockchainEvent):
        """Process a blockchain event"""
        # Create unique event ID to avoid duplicate processing
        event_id = f"{event.transaction_hash}_{event.log_index}"
        
        if event_id in self.processed_events:
            return  # Already processed
        
        try:
            # Get handlers for this event type
            handlers = self.event_handlers.get(event.event_name, [])
            
            if handlers:
                logger.info(f"Processing {event.event_name} event from block {event.block_number}")
                
                # Execute handlers concurrently
                tasks = []
                for handler in handlers:
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor, handler, event
                    )
                    tasks.append(task)
                
                # Wait for all handlers to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                
            # Mark event as processed
            self.processed_events.add(event_id)
            event.processed = True
            
        except Exception as e:
            logger.error(f"Error processing {event.event_name} event: {e}")
    
    def get_listening_status(self) -> Dict[str, Any]:
        """Get current listening status"""
        return {
            "listening": self.listening,
            "last_processed_block": self.last_processed_block,
            "current_block": self._get_current_block_number(),
            "registered_handlers": {
                event_name: len(handlers)
                for event_name, handlers in self.event_handlers.items()
            },
            "processed_events_count": len(self.processed_events)
        }


class AgentRegistryEventHandler:
    """Default event handlers for A2A Agent Registry events"""
    
    def __init__(self):
        self.agent_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("Agent Registry Event Handler initialized")
    
    def handle_agent_registered(self, event: BlockchainEvent):
        """Handle AgentRegistered events"""
        try:
            args = event.args
            agent_id = args.get("agentId")
            wallet_address = args.get("walletAddress")
            agent_type = args.get("agentType")
            
            logger.info(f"🆕 New agent registered: {agent_id} ({agent_type}) at {wallet_address}")
            
            # Update local cache
            self.agent_cache[agent_id] = {
                "wallet_address": wallet_address,
                "agent_type": agent_type,
                "status": "ACTIVE",
                "registered_at": event.timestamp,
                "block_number": event.block_number,
                "transaction_hash": event.transaction_hash
            }
            
            # Could trigger additional actions here:
            # - Send welcome message to new agent
            # - Update discovery cache
            # - Notify other agents of new peer
            
        except Exception as e:
            logger.error(f"Error handling AgentRegistered event: {e}")
    
    def handle_agent_status_updated(self, event: BlockchainEvent):
        """Handle AgentStatusUpdated events"""
        try:
            args = event.args
            agent_id = args.get("agentId")
            old_status = args.get("oldStatus")
            new_status = args.get("newStatus")
            
            status_names = ["INACTIVE", "ACTIVE", "SUSPENDED", "TERMINATED"]
            old_status_name = status_names[old_status] if old_status < len(status_names) else "UNKNOWN"
            new_status_name = status_names[new_status] if new_status < len(status_names) else "UNKNOWN"
            
            logger.info(f"📊 Agent status changed: {agent_id} {old_status_name} → {new_status_name}")
            
            # Update local cache
            if agent_id in self.agent_cache:
                self.agent_cache[agent_id]["status"] = new_status_name
                self.agent_cache[agent_id]["last_updated"] = event.timestamp
            
        except Exception as e:
            logger.error(f"Error handling AgentStatusUpdated event: {e}")
    
    def handle_capabilities_updated(self, event: BlockchainEvent):
        """Handle CapabilitiesUpdated events"""
        try:
            args = event.args
            agent_id = args.get("agentId")
            new_capabilities = args.get("newCapabilities", [])
            
            logger.info(f"🔧 Agent capabilities updated: {agent_id} now has {len(new_capabilities)} capabilities")
            
            # Update local cache
            if agent_id in self.agent_cache:
                self.agent_cache[agent_id]["capabilities"] = new_capabilities
                self.agent_cache[agent_id]["last_updated"] = event.timestamp
            
        except Exception as e:
            logger.error(f"Error handling CapabilitiesUpdated event: {e}")
    
    def handle_skill_card_updated(self, event: BlockchainEvent):
        """Handle SkillCardUpdated events"""
        try:
            args = event.args
            agent_id = args.get("agentId")
            ipfs_hash = args.get("ipfsHash")
            compliance_score = args.get("complianceScore")
            
            logger.info(f"📋 Agent skill card updated: {agent_id} compliance={compliance_score}% IPFS={ipfs_hash}")
            
            # Update local cache
            if agent_id in self.agent_cache:
                self.agent_cache[agent_id]["ipfs_skill_card"] = ipfs_hash
                self.agent_cache[agent_id]["compliance_score"] = compliance_score
                self.agent_cache[agent_id]["last_updated"] = event.timestamp
            
        except Exception as e:
            logger.error(f"Error handling SkillCardUpdated event: {e}")
    
    def get_cached_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached agent information"""
        return self.agent_cache.copy()


# Global event listener and handler instances
_event_listener: Optional[BlockchainEventListener] = None
_event_handler: Optional[AgentRegistryEventHandler] = None


async def initialize_blockchain_event_system(
    blockchain_service: BlockchainRegistrationService,
    start_block: Optional[int] = None
) -> bool:
    """Initialize the blockchain event system"""
    global _event_listener, _event_handler
    
    try:
        # Create event listener
        _event_listener = BlockchainEventListener(blockchain_service, start_block)
        
        # Create default event handler
        _event_handler = AgentRegistryEventHandler()
        
        # Register default handlers
        _event_listener.register_event_handler("AgentRegistered", _event_handler.handle_agent_registered)
        _event_listener.register_event_handler("AgentStatusUpdated", _event_handler.handle_agent_status_updated)
        _event_listener.register_event_handler("CapabilitiesUpdated", _event_handler.handle_capabilities_updated)
        _event_listener.register_event_handler("SkillCardUpdated", _event_handler.handle_skill_card_updated)
        
        # Start listening
        await _event_listener.start_listening()
        
        logger.info("🎧 Blockchain event system initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize blockchain event system: {e}")
        return False


async def stop_blockchain_event_system():
    """Stop the blockchain event system"""
    global _event_listener
    
    if _event_listener:
        await _event_listener.stop_listening()
        _event_listener = None
        logger.info("🔇 Blockchain event system stopped")


def get_event_listener() -> Optional[BlockchainEventListener]:
    """Get the global event listener"""
    return _event_listener


def get_event_handler() -> Optional[AgentRegistryEventHandler]:
    """Get the global event handler"""
    return _event_handler