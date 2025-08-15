"""
Distributed Locking for Vercel deployment
Uses Vercel KV for lock coordination
"""

import asyncio
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager
import uuid

from .state_manager import state_manager

logger = logging.getLogger(__name__)

class DistributedLock:
    """Distributed lock implementation using Vercel KV"""
    
    def __init__(self, name: str, ttl: int = 30):
        self.name = name
        self.ttl = ttl
        self.lock_id = str(uuid.uuid4())
        self.key = f"lock:{name}"
    
    async def acquire(self, timeout: Optional[int] = None) -> bool:
        """Acquire the lock"""
        start_time = time.time()
        
        while True:
            # Try to set lock with NX (not exists) flag
            lock_data = {
                "lock_id": self.lock_id,
                "acquired_at": time.time(),
                "ttl": self.ttl
            }
            
            # Check if lock exists
            current_lock = await state_manager.get(self.key)
            
            if current_lock is None:
                # Try to acquire
                success = await state_manager.set(self.key, lock_data, ttl=self.ttl)
                if success:
                    logger.debug(f"Acquired lock {self.name}")
                    return True
            elif current_lock.get("lock_id") == self.lock_id:
                # We already own this lock, refresh it
                await state_manager.set(self.key, lock_data, ttl=self.ttl)
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                logger.warning(f"Failed to acquire lock {self.name} after {timeout}s")
                return False
            
            # Wait before retry
            await asyncio.sleep(0.1)
    
    async def release(self) -> bool:
        """Release the lock"""
        try:
            current_lock = await state_manager.get(self.key)
            
            if current_lock and current_lock.get("lock_id") == self.lock_id:
                await state_manager.delete(self.key)
                logger.debug(f"Released lock {self.name}")
                return True
            else:
                logger.warning(f"Cannot release lock {self.name} - not owned")
                return False
        except Exception as e:
            logger.error(f"Error releasing lock {self.name}: {e}")
            return False
    
    async def extend(self, additional_ttl: int = None) -> bool:
        """Extend lock TTL"""
        ttl = additional_ttl or self.ttl
        
        current_lock = await state_manager.get(self.key)
        if current_lock and current_lock.get("lock_id") == self.lock_id:
            lock_data = {
                "lock_id": self.lock_id,
                "acquired_at": current_lock.get("acquired_at"),
                "extended_at": time.time(),
                "ttl": ttl
            }
            return await state_manager.set(self.key, lock_data, ttl=ttl)
        return False
    
    @asynccontextmanager
    async def __aenter__(self):
        """Async context manager support"""
        acquired = await self.acquire()
        if not acquired:
            raise RuntimeError(f"Failed to acquire lock {self.name}")
        try:
            yield self
        finally:
            await self.release()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        pass

class WorkflowLock:
    """Specialized lock for workflow execution"""
    
    @staticmethod
    async def acquire_workflow_lock(
        workflow_id: str, 
        execution_id: str,
        timeout: int = 60
    ) -> Optional[DistributedLock]:
        """Acquire lock for workflow execution"""
        lock_name = f"workflow:{workflow_id}:{execution_id}"
        lock = DistributedLock(lock_name, ttl=300)  # 5 minute TTL
        
        if await lock.acquire(timeout=timeout):
            return lock
        return None
    
    @staticmethod
    async def acquire_step_lock(
        workflow_id: str,
        execution_id: str,
        step_id: str,
        timeout: int = 30
    ) -> Optional[DistributedLock]:
        """Acquire lock for workflow step"""
        lock_name = f"step:{workflow_id}:{execution_id}:{step_id}"
        lock = DistributedLock(lock_name, ttl=120)  # 2 minute TTL
        
        if await lock.acquire(timeout=timeout):
            return lock
        return None

# Utility functions
async def with_lock(lock_name: str, ttl: int = 30):
    """Decorator for functions that need distributed locking"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            lock = DistributedLock(lock_name, ttl=ttl)
            async with lock:
                return await func(*args, **kwargs)
        return wrapper
    return decorator