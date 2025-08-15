"""
Memory-Enabled Strands Agent
Combines BaseStrandsAgent capabilities with BaseMemoryAgent memory functions
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .base_strands_agent import BaseStrandsAgent
from .base_memory_agent import BaseMemoryAgent

logger = logging.getLogger(__name__)

class MemoryStrandsAgent(BaseStrandsAgent):
    """
    Strands Agent with full memory capabilities
    Inherits from BaseStrandsAgent for Strands functionality
    Includes BaseMemoryAgent methods for memory operations
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        model_provider: str = "grok4",
        version: str = "1.0",
        user_id: int = 1
    ):
        # Initialize BaseStrandsAgent first
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities,
            model_provider=model_provider,
            version=version
        )
        
        # Initialize memory capabilities
        self.user_id = user_id
        self.session_id = None
        
        # Import memory components
        from ...memory import (
            ConversationMemoryManager, 
            AgentContextManager, 
            SemanticMemoryManager,
            MemoryRetrievalSystem
        )
        from ...memory.autonomous_memory_triggers import autonomous_memory
        from ...memory.a2a_memory_system import a2a_memory_system
        from ..protocols.a2a_protocol import A2AMessage, MessageType, A2AProtocol
        
        # Initialize memory systems
        self.conversation_memory = ConversationMemoryManager()
        self.agent_context = AgentContextManager()
        self.semantic_memory = SemanticMemoryManager()
        self.memory_retrieval = MemoryRetrievalSystem()
        self.a2a_memory_system = a2a_memory_system
        self.A2AProtocol = A2AProtocol
        self.MessageType = MessageType
        
        # Initialize memory context for the agent
        self._initialize_memory()
        
        # Register with autonomous memory system
        self._register_autonomous_triggers()
        
        logger.info(f"Initialized memory-enabled Strands agent: {self.agent_id}")
    
    def _initialize_memory(self):
        """Initialize memory context for the agent"""
        try:
            import uuid
            
            # Create or get session
            self.session_id = str(uuid.uuid4())
            self.conversation_memory.create_session(
                user_id=self.user_id,
                agent_type=self.agent_type,
                initial_context={'agent_id': self.agent_id}
            )
            
            # Initialize agent context
            initial_context = {
                'working_memory': {},
                'goals': [],
                'knowledge_base': {},
                'state': 'initialized',
                'capabilities': self.capabilities,
                'preferences': {},
                'learning_data': {}
            }
            
            self.agent_context.create_agent_context(
                session_id=self.session_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                initial_context=initial_context
            )
            
            logger.info(f"Memory initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error initializing memory for agent {self.agent_id}: {e}")
    
    def _register_autonomous_triggers(self):
        """Register agent with autonomous memory system"""
        try:
            from ...memory.autonomous_memory_triggers import autonomous_memory
            
            # Register event handlers for this agent
            autonomous_memory.register_event_handler(
                'agent_response', 
                self._handle_agent_response_event
            )
            
            autonomous_memory.register_event_handler(
                'trade_executed',
                self._handle_trade_event
            )
            
            autonomous_memory.register_event_handler(
                'market_data_update',
                self._handle_market_event
            )
            
            logger.info(f"Registered autonomous memory triggers for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error registering autonomous triggers: {e}")
    
    async def _handle_agent_response_event(self, event_data: Dict[str, Any]):
        """Handle agent response events for autonomous memory"""
        if event_data.get('agent_id') == self.agent_id:
            response_content = event_data.get('response', '')
            importance = event_data.get('importance_score', 0.6)
            
            if importance > 0.7:
                self.conversation_memory.add_message(
                    session_id=self.session_id,
                    role="assistant",
                    content=response_content,
                    metadata={
                        'autonomous_trigger': True,
                        'agent_id': self.agent_id,
                        'event_type': 'agent_response'
                    },
                    importance_score=importance
                )
    
    async def _handle_trade_event(self, event_data: Dict[str, Any]):
        """Handle trade execution events"""
        trade_summary = f"Trade Alert: {event_data.get('action', 'Unknown')} " \
                       f"{event_data.get('quantity', 0)} {event_data.get('symbol', 'UNKNOWN')} " \
                       f"at ${event_data.get('price', 0):.2f}"
        
        self.conversation_memory.add_message(
            session_id=self.session_id,
            role="system",
            content=trade_summary,
            metadata={'event_type': 'trade_execution', 'autonomous': True},
            importance_score=0.9
        )
        
        # Update agent working memory with trade context
        trade_context = {
            'last_trade': event_data,
            'last_trade_time': datetime.utcnow().isoformat()
        }
        
        current_memory = self.agent_context.get_working_memory(self.session_id, self.agent_id)
        current_memory.update(trade_context)
        
        self.agent_context.save_working_memory(
            session_id=self.session_id,
            agent_id=self.agent_id,
            memory_data=current_memory
        )
    
    async def _handle_market_event(self, event_data: Dict[str, Any]):
        """Handle market data update events"""
        symbol = event_data.get('symbol', 'UNKNOWN')
        price = event_data.get('price', 0)
        change = event_data.get('price_change_percent', 0)
        
        if abs(change) > 2:  # More than 2% change
            market_update = f"Market Update: {symbol} ${price:.2f} ({change:+.1f}%)"
            
            self.conversation_memory.add_message(
                session_id=self.session_id,
                role="system",
                content=market_update,
                metadata={'event_type': 'market_update', 'autonomous': True},
                importance_score=0.7 if abs(change) > 5 else 0.5
            )
    
    # Memory Methods - Direct implementation from BaseMemoryAgent
    
    async def create_shared_memory(self, content: str, memory_type: str, 
                                 process_id: str = 'trading_process', 
                                 importance_score: float = 0.7) -> Dict[str, Any]:
        """Create shared memory via A2A message"""
        memory_message = self.A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id="memory-system",
            message_type=self.MessageType.MEMORY_SHARE,
            payload={
                'scope': 'shared',
                'process_id': process_id,
                'memory_type': memory_type,
                'content': content,
                'importance_score': importance_score,
                'metadata': {
                    'created_by_agent_type': self.agent_type,
                    'session_id': self.session_id
                }
            }
        )
        
        return await self.a2a_memory_system.process_a2a_memory_message(memory_message)
    
    async def create_private_memory(self, content: str, memory_type: str,
                                  importance_score: float = 0.5) -> Dict[str, Any]:
        """Create private memory via A2A message"""
        memory_message = self.A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id="memory-system",
            message_type=self.MessageType.MEMORY_SHARE,
            payload={
                'scope': 'private',
                'session_id': self.session_id,
                'memory_type': memory_type,
                'content': content,
                'importance_score': importance_score,
                'metadata': {
                    'agent_type': self.agent_type,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        )
        
        return await self.a2a_memory_system.process_a2a_memory_message(memory_message)
    
    async def query_memories(self, query: str, scope: str = 'both') -> Dict[str, Any]:
        """Query memories via A2A message"""
        memory_message = self.A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id="memory-system",
            message_type=self.MessageType.MEMORY_REQUEST,
            payload={
                'request_type': 'query',
                'query': query,
                'scope': scope
            }
        )
        
        return await self.a2a_memory_system.process_a2a_memory_message(memory_message)
    
    async def get_recent_memories(self, limit: int = 10, scope: str = 'both') -> Dict[str, Any]:
        """Get recent memories via A2A message"""
        memory_message = self.A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id="memory-system",
            message_type=self.MessageType.MEMORY_REQUEST,
            payload={
                'request_type': 'recent',
                'limit': limit,
                'scope': scope
            }
        )
        
        return await self.a2a_memory_system.process_a2a_memory_message(memory_message)
    
    async def store_trade_memory(self, trade_data: Dict[str, Any], 
                               is_shared: bool = True) -> Dict[str, Any]:
        """Store trade-related memory with appropriate scope"""
        content = f"Trade executed: {trade_data.get('action', 'unknown')} " \
                 f"{trade_data.get('quantity', 0)} {trade_data.get('symbol', 'UNKNOWN')} " \
                 f"at ${trade_data.get('price', 0):.2f}"
        
        if is_shared:
            return await self.create_shared_memory(
                content=content,
                memory_type='trade_execution',
                importance_score=0.9
            )
        else:
            return await self.create_private_memory(
                content=content,
                memory_type='trade_execution',
                importance_score=0.9
            )
    
    async def store_analysis_memory(self, analysis_data: Dict[str, Any],
                                  is_shared: bool = True) -> Dict[str, Any]:
        """Store analysis-related memory with appropriate scope"""
        symbol = analysis_data.get('symbol', 'UNKNOWN')
        signal = analysis_data.get('signal', 'HOLD')
        confidence = analysis_data.get('confidence', 0.5)
        
        content = f"Analysis for {symbol}: {signal} signal with {confidence:.0%} confidence. " \
                 f"{analysis_data.get('reasoning', 'Technical analysis')}"
        
        if is_shared:
            return await self.create_shared_memory(
                content=content,
                memory_type='market_analysis',
                process_id='analysis_process',
                importance_score=confidence
            )
        else:
            return await self.create_private_memory(
                content=content,
                memory_type='market_analysis',
                importance_score=confidence
            )
    
    def search_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """Search relevant memories"""
        return self.semantic_memory.search_similar_memories(
            user_id=self.user_id,
            query=query,
            limit=limit
        )
    
    def get_conversation_summary(self) -> str:
        """Get conversation summary"""
        return self.conversation_memory.get_conversation_summary(self.session_id) or ""
    
    def save_knowledge(self, key: str, knowledge: Any) -> bool:
        """Save knowledge to agent's knowledge base"""
        return self.agent_context.save_knowledge(
            session_id=self.session_id,
            agent_id=self.agent_id,
            knowledge_key=key,
            knowledge_data=knowledge
        )
    
    def get_knowledge(self, key: str) -> Any:
        """Get knowledge from agent's knowledge base"""
        return self.agent_context.get_knowledge(
            session_id=self.session_id,
            agent_id=self.agent_id,
            knowledge_key=key
        )
    
    def export_memory(self) -> Dict:
        """Export complete memory context"""
        return {
            'agent_info': {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'user_id': self.user_id,
                'session_id': self.session_id
            },
            'conversation': self.conversation_memory.export_conversation(self.session_id),
            'agent_context': self.agent_context.get_agent_context(self.session_id, self.agent_id),
            'context_history': self.agent_context.get_context_history(self.session_id, self.agent_id),
            'a2a_memories': self.a2a_memory_system.export_memories(self.agent_id)
        }
    
    def add_goal(self, goal_description: str, priority: str = "medium") -> bool:
        """Add goal to agent's goals"""
        try:
            goal = {
                'description': goal_description,
                'priority': priority,
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            context = self.agent_context.get_agent_context(self.session_id, self.agent_id)
            if context:
                goals = context.get('goals', [])
                goals.append(goal)
                return self.agent_context.update_context(
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    context_updates={'goals': goals}
                )
            return False
        except Exception as e:
            logger.error(f"Error adding goal for agent {self.agent_id}: {e}")
            return False
    
    def set_state(self, state: str, state_data: Optional[Dict] = None):
        """Set agent state"""
        try:
            state_info = {
                'state': state,
                'state_data': state_data or {},
                'updated_at': datetime.now().isoformat()
            }
            
            return self.agent_context.update_context(
                session_id=self.session_id,
                agent_id=self.agent_id,
                context_updates={'state': state_info}
            )
        except Exception as e:
            logger.error(f"Error setting state for agent {self.agent_id}: {e}")
            return False