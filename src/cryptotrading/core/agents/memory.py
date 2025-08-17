"""
Memory-enabled agent that extends BaseAgent with memory capabilities.
Consolidates memory functionality from multiple previous implementations.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from .base import BaseAgent

class MemoryAgent(BaseAgent):
    """
    Memory-enabled agent that consolidates functionality from:
    - cryptotrading.core.agents.base_memory_agent.BaseMemoryAgent
    - cryptotrading.core.agents.memory_strands_agent.MemoryStrandsAgent
    """
    
    def __init__(self, agent_id: str, agent_type: str, memory_config: Optional[Dict] = None, **kwargs):
        super().__init__(agent_id, agent_type, **kwargs)
        self.memory_config = memory_config or {}
        self.memory_store = {}
        self._memory_lock = asyncio.Lock()  # Fix race condition
        self._setup_memory()
    
    def _setup_memory(self):
        """Setup memory storage and retrieval"""
        # Initialize memory components
        pass
    
    async def store_memory(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store information in agent memory"""
        async with self._memory_lock:  # Fix race condition
            self.memory_store[key] = {
                "value": value,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent memory"""
        async with self._memory_lock:  # Fix race condition
            if key in self.memory_store:
                return self.memory_store[key]["value"]
            return None
    
    async def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search agent memory"""
        async with self._memory_lock:  # Fix race condition
            results = []
            for key, entry in self.memory_store.items():
                # Simple substring search - can be enhanced
                if query.lower() in str(entry["value"]).lower() or query.lower() in key.lower():
                    results.append({
                        "key": key,
                        "value": entry["value"],
                        "metadata": entry["metadata"],
                        "timestamp": entry["timestamp"]
                    })
            return results
    
    async def clear_memory(self, key: Optional[str] = None):
        """Clear memory entries"""
        async with self._memory_lock:  # Fix race condition
            if key:
                self.memory_store.pop(key, None)
            else:
                self.memory_store.clear()
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        async with self._memory_lock:  # Fix race condition
            return {
                "total_entries": len(self.memory_store),
                "memory_keys": list(self.memory_store.keys()),
                "oldest_entry": min(
                    (entry["timestamp"] for entry in self.memory_store.values()),
                    default=None
                ),
                "newest_entry": max(
                    (entry["timestamp"] for entry in self.memory_store.values()),
                    default=None
                )
            }
    
    # Implement abstract methods from BaseAgent
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message for memory operations"""
        try:
            action = message.get("action")
            
            if action == "store":
                key = message.get("key")
                value = message.get("value")
                metadata = message.get("metadata")
                await self.store_memory(key, value, metadata)
                return {"status": "success", "action": "stored", "key": key}
            
            elif action == "retrieve":
                key = message.get("key")
                value = await self.retrieve_memory(key)
                return {"status": "success", "action": "retrieved", "key": key, "value": value}
            
            elif action == "search":
                query = message.get("query")
                results = await self.search_memory(query)
                return {"status": "success", "action": "searched", "results": results}
            
            elif action == "clear":
                key = message.get("key")
                await self.clear_memory(key)
                return {"status": "success", "action": "cleared", "key": key}
            
            elif action == "stats":
                stats = await self.get_memory_stats()
                return {"status": "success", "action": "stats", "stats": stats}
            
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _process_message_impl(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Internal message processing implementation"""
        # Delegate to the main process_message method
        return await self.process_message(message)
