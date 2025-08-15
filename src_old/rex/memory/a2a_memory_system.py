"""
A2A Memory System
Implements shared and private memory triggered through A2A messages
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from ..a2a.protocols.a2a_protocol import A2AMessage, MessageType, A2AProtocol
from .conversation_memory import ConversationMemoryManager
from .semantic_memory import SemanticMemoryManager

logger = logging.getLogger(__name__)

@dataclass
class SharedMemory:
    """Shared memory accessible to trusted agents within a process"""
    memory_id: str
    process_id: str
    creator_agent_id: str
    trusted_agents: List[str]  # List of agent IDs that can access this memory
    memory_type: str  # 'trade_analysis', 'market_insight', 'risk_assessment', etc.
    content: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    access_count: int = 0
    importance_score: float = 0.5

@dataclass 
class PrivateMemory:
    """Private memory hidden under A2A for specific agent"""
    memory_id: str
    agent_id: str
    session_id: str
    memory_type: str  # 'strategy', 'learning', 'personal_context', etc.
    content: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    encrypted: bool = False
    importance_score: float = 0.5

class A2AMemorySystem:
    """Manages A2A message-triggered memory system with shared and private tiers"""
    
    def __init__(self):
        self.conversation_memory = ConversationMemoryManager()
        self.semantic_memory = SemanticMemoryManager()
        
        # Memory stores
        self.shared_memories: Dict[str, SharedMemory] = {}
        self.private_memories: Dict[str, Dict[str, PrivateMemory]] = {}  # agent_id -> {memory_id -> memory}
        
        # Process trust relationships
        self.process_trust_map: Dict[str, List[str]] = {
            'trading_process': ['historical-loader-001', 'database-001', 'illuminate-001', 'execute-001', 'test-agent-001'],
            'analysis_process': ['database-001', 'illuminate-001', 'transform-001', 'test-agent-001'],
            'data_process': ['historical-loader-001', 'database-001', 'transform-001', 'test-agent-001']
        }
        
        # A2A message handlers
        self.message_handlers = {
            MessageType.MEMORY_SHARE: self._handle_memory_share,
            MessageType.MEMORY_REQUEST: self._handle_memory_request,
            MessageType.MEMORY_RESPONSE: self._handle_memory_response
        }
        
        logger.info("A2A Memory System initialized")
    
    async def process_a2a_memory_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Process A2A memory-related messages"""
        try:
            handler = self.message_handlers.get(message.message_type)
            if not handler:
                return {
                    "success": False,
                    "error": f"Unknown memory message type: {message.message_type}"
                }
            
            return await handler(message)
            
        except Exception as e:
            logger.error(f"Error processing A2A memory message: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_memory_share(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle MEMORY_SHARE messages to create shared or private memories"""
        try:
            payload = message.payload
            memory_scope = payload.get('scope', 'shared')  # 'shared' or 'private'
            
            if memory_scope == 'shared':
                return await self._create_shared_memory(message)
            else:
                return await self._create_private_memory(message)
                
        except Exception as e:
            logger.error(f"Error handling memory share: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_memory_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle MEMORY_REQUEST messages to retrieve memories"""
        try:
            payload = message.payload
            request_type = payload.get('request_type', 'query')  # 'query', 'recent', 'by_type'
            requester_agent = message.sender_id
            
            if request_type == 'query':
                query = payload.get('query', '')
                scope = payload.get('scope', 'both')  # 'shared', 'private', 'both'
                return await self._query_memories(requester_agent, query, scope)
                
            elif request_type == 'recent':
                limit = payload.get('limit', 10)
                scope = payload.get('scope', 'both')
                return await self._get_recent_memories(requester_agent, limit, scope)
                
            elif request_type == 'by_type':
                memory_type = payload.get('memory_type', '')
                scope = payload.get('scope', 'both')
                return await self._get_memories_by_type(requester_agent, memory_type, scope)
                
            else:
                return {"success": False, "error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            logger.error(f"Error handling memory request: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_memory_response(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle MEMORY_RESPONSE messages (acknowledgments)"""
        try:
            payload = message.payload
            original_memory_id = payload.get('memory_id')
            
            # Update access count for shared memories
            if original_memory_id in self.shared_memories:
                self.shared_memories[original_memory_id].access_count += 1
                logger.info(f"Updated access count for shared memory {original_memory_id}")
            
            return {"success": True, "acknowledged": True}
            
        except Exception as e:
            logger.error(f"Error handling memory response: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_shared_memory(self, message: A2AMessage) -> Dict[str, Any]:
        """Create shared memory accessible to trusted agents"""
        try:
            payload = message.payload
            creator_agent = message.sender_id
            process_id = payload.get('process_id', 'trading_process')
            
            # Verify agent is trusted for this process
            trusted_agents = self.process_trust_map.get(process_id, [])
            if creator_agent not in trusted_agents:
                return {
                    "success": False,
                    "error": f"Agent {creator_agent} not trusted for process {process_id}"
                }
            
            # Create shared memory
            memory_id = str(uuid.uuid4())
            shared_memory = SharedMemory(
                memory_id=memory_id,
                process_id=process_id,
                creator_agent_id=creator_agent,
                trusted_agents=trusted_agents,
                memory_type=payload.get('memory_type', 'general'),
                content=payload.get('content', ''),
                metadata=payload.get('metadata', {}),
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                importance_score=payload.get('importance_score', 0.5)
            )
            
            self.shared_memories[memory_id] = shared_memory
            
            # Store in semantic memory for searchability
            await self._store_in_semantic_memory(shared_memory, scope='shared')
            
            # Notify trusted agents
            await self._notify_trusted_agents(shared_memory)
            
            logger.info(f"Created shared memory {memory_id} for process {process_id}")
            
            return {
                "success": True,
                "memory_id": memory_id,
                "shared_with": trusted_agents,
                "process_id": process_id
            }
            
        except Exception as e:
            logger.error(f"Error creating shared memory: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_private_memory(self, message: A2AMessage) -> Dict[str, Any]:
        """Create private memory for specific agent"""
        try:
            payload = message.payload
            agent_id = message.sender_id
            session_id = payload.get('session_id', str(uuid.uuid4()))
            
            # Initialize agent's private memory store if needed
            if agent_id not in self.private_memories:
                self.private_memories[agent_id] = {}
            
            # Create private memory
            memory_id = str(uuid.uuid4())
            private_memory = PrivateMemory(
                memory_id=memory_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type=payload.get('memory_type', 'personal'),
                content=payload.get('content', ''),
                metadata=payload.get('metadata', {}),
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                encrypted=payload.get('encrypted', False),
                importance_score=payload.get('importance_score', 0.5)
            )
            
            self.private_memories[agent_id][memory_id] = private_memory
            
            # Store in semantic memory with agent restriction
            await self._store_in_semantic_memory(private_memory, scope='private')
            
            logger.info(f"Created private memory {memory_id} for agent {agent_id}")
            
            return {
                "success": True,
                "memory_id": memory_id,
                "agent_id": agent_id,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error creating private memory: {e}")
            return {"success": False, "error": str(e)}
    
    async def _store_in_semantic_memory(self, memory, scope: str):
        """Store memory in semantic memory system for search"""
        try:
            if isinstance(memory, SharedMemory):
                context = f"Shared memory for {memory.process_id} - Created by {memory.creator_agent_id}"
                keywords = [memory.process_id, memory.creator_agent_id, memory.memory_type]
            else:  # PrivateMemory
                context = f"Private memory for {memory.agent_id} - Session {memory.session_id}"
                keywords = [memory.agent_id, memory.session_id, memory.memory_type]
            
            self.semantic_memory.store_memory(
                user_id=1,  # System user for A2A memories
                memory_type="a2a_memory",
                content=memory.content,
                context=context,
                keywords=keywords,
                confidence=memory.importance_score
            )
            
        except Exception as e:
            logger.error(f"Error storing memory in semantic system: {e}")
    
    async def _notify_trusted_agents(self, shared_memory: SharedMemory):
        """Notify trusted agents about new shared memory"""
        try:
            for agent_id in shared_memory.trusted_agents:
                if agent_id != shared_memory.creator_agent_id:  # Don't notify creator
                    notification_message = A2AProtocol.create_message(
                        sender_id="memory-system",
                        receiver_id=agent_id,
                        message_type=MessageType.MEMORY_RESPONSE,
                        payload={
                            "notification_type": "shared_memory_created",
                            "memory_id": shared_memory.memory_id,
                            "process_id": shared_memory.process_id,
                            "creator": shared_memory.creator_agent_id,
                            "memory_type": shared_memory.memory_type,
                            "importance_score": shared_memory.importance_score
                        }
                    )
                    
                    # In a real system, this would be sent through the A2A coordinator
                    logger.info(f"Notified {agent_id} about shared memory {shared_memory.memory_id}")
                    
        except Exception as e:
            logger.error(f"Error notifying trusted agents: {e}")
    
    async def _query_memories(self, requester_agent: str, query: str, scope: str) -> Dict[str, Any]:
        """Query memories based on semantic search"""
        try:
            results = {
                "success": True,
                "query": query,
                "shared_memories": [],
                "private_memories": []
            }
            
            if scope in ['shared', 'both']:
                # Search shared memories
                for memory in self.shared_memories.values():
                    if requester_agent in memory.trusted_agents:
                        # Simple keyword matching (can be enhanced with semantic search)
                        if query.lower() in memory.content.lower() or \
                           query.lower() in memory.memory_type.lower():
                            results["shared_memories"].append({
                                "memory_id": memory.memory_id,
                                "content": memory.content,
                                "memory_type": memory.memory_type,
                                "creator": memory.creator_agent_id,
                                "created_at": memory.created_at,
                                "importance_score": memory.importance_score
                            })
            
            if scope in ['private', 'both']:
                # Search private memories for requester agent only
                agent_private_memories = self.private_memories.get(requester_agent, {})
                for memory in agent_private_memories.values():
                    if query.lower() in memory.content.lower() or \
                       query.lower() in memory.memory_type.lower():
                        results["private_memories"].append({
                            "memory_id": memory.memory_id,
                            "content": memory.content,
                            "memory_type": memory.memory_type,
                            "session_id": memory.session_id,
                            "created_at": memory.created_at,
                            "importance_score": memory.importance_score
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_recent_memories(self, requester_agent: str, limit: int, scope: str) -> Dict[str, Any]:
        """Get recent memories for requester agent"""
        try:
            results = {
                "success": True,
                "shared_memories": [],
                "private_memories": []
            }
            
            if scope in ['shared', 'both']:
                # Get recent shared memories accessible to requester
                accessible_shared = [
                    memory for memory in self.shared_memories.values()
                    if requester_agent in memory.trusted_agents
                ]
                # Sort by creation time, most recent first
                accessible_shared.sort(key=lambda m: m.created_at, reverse=True)
                
                for memory in accessible_shared[:limit]:
                    results["shared_memories"].append({
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "memory_type": memory.memory_type,
                        "creator": memory.creator_agent_id,
                        "created_at": memory.created_at
                    })
            
            if scope in ['private', 'both']:
                # Get recent private memories for requester agent
                agent_private_memories = self.private_memories.get(requester_agent, {})
                private_list = list(agent_private_memories.values())
                private_list.sort(key=lambda m: m.created_at, reverse=True)
                
                for memory in private_list[:limit]:
                    results["private_memories"].append({
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "memory_type": memory.memory_type,
                        "session_id": memory.session_id,
                        "created_at": memory.created_at
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_memories_by_type(self, requester_agent: str, memory_type: str, scope: str) -> Dict[str, Any]:
        """Get memories filtered by type"""
        try:
            results = {
                "success": True,
                "memory_type": memory_type,
                "shared_memories": [],
                "private_memories": []
            }
            
            if scope in ['shared', 'both']:
                # Filter shared memories by type
                for memory in self.shared_memories.values():
                    if (requester_agent in memory.trusted_agents and 
                        memory.memory_type == memory_type):
                        results["shared_memories"].append({
                            "memory_id": memory.memory_id,
                            "content": memory.content,
                            "creator": memory.creator_agent_id,
                            "created_at": memory.created_at,
                            "importance_score": memory.importance_score
                        })
            
            if scope in ['private', 'both']:
                # Filter private memories by type
                agent_private_memories = self.private_memories.get(requester_agent, {})
                for memory in agent_private_memories.values():
                    if memory.memory_type == memory_type:
                        results["private_memories"].append({
                            "memory_id": memory.memory_id,
                            "content": memory.content,
                            "session_id": memory.session_id,
                            "created_at": memory.created_at,
                            "importance_score": memory.importance_score
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting memories by type: {e}")
            return {"success": False, "error": str(e)}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about A2A memory system"""
        total_private = sum(len(agent_memories) for agent_memories in self.private_memories.values())
        
        return {
            "shared_memories": len(self.shared_memories),
            "private_memories": total_private,
            "processes": list(self.process_trust_map.keys()),
            "trusted_agents": sum(len(agents) for agents in self.process_trust_map.values()),
            "memory_types": {
                "shared": list(set(m.memory_type for m in self.shared_memories.values())),
                "private": list(set(
                    m.memory_type for agent_memories in self.private_memories.values()
                    for m in agent_memories.values()
                ))
            }
        }
    
    def export_memories(self, agent_id: str) -> Dict[str, Any]:
        """Export memories accessible to a specific agent"""
        export_data = {
            "agent_id": agent_id,
            "exported_at": datetime.utcnow().isoformat(),
            "shared_memories": [],
            "private_memories": []
        }
        
        # Export accessible shared memories
        for memory in self.shared_memories.values():
            if agent_id in memory.trusted_agents:
                export_data["shared_memories"].append(asdict(memory))
        
        # Export private memories for this agent
        agent_private_memories = self.private_memories.get(agent_id, {})
        for memory in agent_private_memories.values():
            export_data["private_memories"].append(asdict(memory))
        
        return export_data

# Global A2A memory system instance
a2a_memory_system = A2AMemorySystem()