"""
Message Queue implementation for Vercel deployment
Uses Vercel KV for queue storage and Edge Functions for processing
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict

from .state_manager import state_manager

logger = logging.getLogger(__name__)

@dataclass
class QueueMessage:
    """Message in the queue"""
    id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: int = 0
    created_at: str = None
    attempts: int = 0
    max_attempts: int = 3
    visible_after: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

class MessageQueue:
    """Distributed message queue using Vercel KV"""
    
    def __init__(self, queue_name: str = "a2a-messages"):
        self.queue_name = queue_name
        self.processors: Dict[str, Callable] = {}
        self.processing = False
        self._process_task = None
    
    async def enqueue(
        self, 
        payload: Dict[str, Any], 
        priority: int = 0,
        delay_seconds: int = 0
    ) -> str:
        """Add message to queue"""
        message = QueueMessage(
            id=str(uuid.uuid4()),
            queue_name=self.queue_name,
            payload=payload,
            priority=priority,
            visible_after=(
                datetime.now().isoformat() if delay_seconds == 0 
                else (datetime.now() + timedelta(seconds=delay_seconds)).isoformat()
            )
        )
        
        # Store in sorted set by priority and timestamp
        score = f"{9-priority}.{datetime.now().timestamp()}"
        key = f"queue:{self.queue_name}:messages"
        
        await state_manager.set(
            f"queue:{self.queue_name}:msg:{message.id}",
            asdict(message),
            ttl=86400  # 24 hour TTL
        )
        
        # Add to queue index
        await self._add_to_index(message.id, score)
        
        logger.info(f"Enqueued message {message.id} to {self.queue_name}")
        return message.id
    
    async def dequeue(self, batch_size: int = 1) -> List[QueueMessage]:
        """Get messages from queue"""
        messages = []
        
        # Get message IDs from index
        message_ids = await self._get_from_index(batch_size)
        
        for msg_id in message_ids:
            msg_data = await state_manager.get(f"queue:{self.queue_name}:msg:{msg_id}")
            if msg_data:
                message = QueueMessage(**msg_data)
                
                # Check visibility
                if message.visible_after:
                    if datetime.fromisoformat(message.visible_after) > datetime.now():
                        continue
                
                messages.append(message)
                
                # Remove from index (will re-add if processing fails)
                await self._remove_from_index(msg_id)
        
        return messages
    
    async def ack(self, message_id: str):
        """Acknowledge message processing"""
        await state_manager.delete(f"queue:{self.queue_name}:msg:{message_id}")
        logger.debug(f"Acknowledged message {message_id}")
    
    async def nack(self, message_id: str, delay_seconds: int = 60):
        """Negative acknowledge - requeue message"""
        msg_data = await state_manager.get(f"queue:{self.queue_name}:msg:{message_id}")
        if msg_data:
            message = QueueMessage(**msg_data)
            message.attempts += 1
            
            if message.attempts >= message.max_attempts:
                # Move to DLQ
                await self._move_to_dlq(message)
            else:
                # Requeue with delay
                message.visible_after = (
                    datetime.now() + timedelta(seconds=delay_seconds)
                ).isoformat()
                
                await state_manager.set(
                    f"queue:{self.queue_name}:msg:{message_id}",
                    asdict(message)
                )
                
                # Re-add to index with lower priority
                score = f"{9-message.priority+1}.{datetime.now().timestamp()}"
                await self._add_to_index(message_id, score)
    
    def register_processor(self, message_type: str, processor: Callable):
        """Register a message processor"""
        self.processors[message_type] = processor
        logger.info(f"Registered processor for {message_type}")
    
    async def start_processing(self):
        """Start processing messages"""
        if self.processing:
            return
        
        self.processing = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"Started processing queue {self.queue_name}")
    
    async def stop_processing(self):
        """Stop processing messages"""
        self.processing = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped processing queue {self.queue_name}")
    
    async def _process_loop(self):
        """Main processing loop"""
        while self.processing:
            try:
                messages = await self.dequeue(batch_size=10)
                
                if messages:
                    # Process messages concurrently
                    tasks = []
                    for message in messages:
                        task = asyncio.create_task(self._process_message(message))
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # No messages, wait
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_message(self, message: QueueMessage):
        """Process a single message"""
        try:
            msg_type = message.payload.get("type", "unknown")
            processor = self.processors.get(msg_type)
            
            if processor:
                await processor(message.payload)
                await self.ack(message.id)
            else:
                logger.warning(f"No processor for message type: {msg_type}")
                await self.nack(message.id)
                
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            await self.nack(message.id)
    
    async def _add_to_index(self, message_id: str, score: str):
        """Add message to queue index"""
        key = f"queue:{self.queue_name}:index"
        index = await state_manager.get(key) or []
        index.append((score, message_id))
        index.sort(key=lambda x: x[0])
        await state_manager.set(key, index)
    
    async def _get_from_index(self, count: int) -> List[str]:
        """Get message IDs from index"""
        key = f"queue:{self.queue_name}:index"
        index = await state_manager.get(key) or []
        
        message_ids = []
        for i in range(min(count, len(index))):
            _, msg_id = index[i]
            message_ids.append(msg_id)
        
        # Update index
        if message_ids:
            index = index[len(message_ids):]
            await state_manager.set(key, index)
        
        return message_ids
    
    async def _remove_from_index(self, message_id: str):
        """Remove message from index"""
        key = f"queue:{self.queue_name}:index"
        index = await state_manager.get(key) or []
        index = [(s, m) for s, m in index if m != message_id]
        await state_manager.set(key, index)
    
    async def _move_to_dlq(self, message: QueueMessage):
        """Move message to dead letter queue"""
        dlq_key = f"queue:{self.queue_name}:dlq:{message.id}"
        await state_manager.set(dlq_key, asdict(message), ttl=604800)  # 7 days
        await state_manager.delete(f"queue:{self.queue_name}:msg:{message.id}")
        logger.warning(f"Moved message {message.id} to DLQ")
    
    async def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        key = f"queue:{self.queue_name}:index"
        index = await state_manager.get(key) or []
        
        dlq_keys = await state_manager.get_pattern(f"queue:{self.queue_name}:dlq:*")
        
        return {
            "pending": len(index),
            "dead_letter": len(dlq_keys),
            "processors": len(self.processors)
        }

# Global message queues
workflow_queue = MessageQueue("workflow-requests")
a2a_queue = MessageQueue("a2a-messages")