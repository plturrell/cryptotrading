"""
Base Memory-Enabled Agent
Provides memory capabilities to all A2A agents
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from ...memory import (
    ConversationMemoryManager, 
    AgentContextManager, 
    SemanticMemoryManager,
    MemoryRetrievalSystem
)
from ...memory.autonomous_memory_triggers import autonomous_memory
from ...memory.a2a_memory_system import a2a_memory_system
from ..protocols.a2a_protocol import A2AMessage, MessageType, A2AProtocol

logger = logging.getLogger(__name__)

class BaseMemoryAgent:
    """Base class for memory-enabled A2A agents"""
    
    def __init__(self, agent_id: str, agent_type: str, user_id: int = 1):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.user_id = user_id
        self.session_id = None
        
        # Initialize memory systems
        self.conversation_memory = ConversationMemoryManager()
        self.agent_context = AgentContextManager()
        self.semantic_memory = SemanticMemoryManager()
        self.memory_retrieval = MemoryRetrievalSystem()
        
        # Agent-specific initialization
        self._initialize_memory()
        
        # Register with autonomous memory system
        self._register_autonomous_triggers()
        
    def _initialize_memory(self):
        """Initialize memory context for the agent"""
        try:
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
                'capabilities': self._get_agent_capabilities(),
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
            # Store significant agent responses in memory
            response_content = event_data.get('response', '')
            importance = event_data.get('importance_score', 0.6)
            
            if importance > 0.7:  # Only store important responses
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
        # Generate contextual conversation about the trade
        trade_summary = f"Trade Alert: {event_data.get('action', 'Unknown')} " \
                       f"{event_data.get('quantity', 0)} {event_data.get('symbol', 'UNKNOWN')} " \
                       f"at ${event_data.get('price', 0):.2f}"
        
        # Store as both conversation and semantic memory
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
        
        # Only store significant market movements
        if abs(change) > 2:  # More than 2% change
            market_update = f"Market Update: {symbol} ${price:.2f} ({change:+.1f}%)"
            
            self.conversation_memory.add_message(
                session_id=self.session_id,
                role="system",
                content=market_update,
                metadata={'event_type': 'market_update', 'autonomous': True},
                importance_score=0.7 if abs(change) > 5 else 0.5
            )
    
    def _get_agent_capabilities(self) -> List[str]:
        """Override in subclasses to define agent-specific capabilities"""
        return ['memory_enabled', 'context_aware', 'autonomous_memory']
    
    async def process_with_memory(self, request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process request with full memory context"""
        try:
            # Store user request in conversation memory
            self.conversation_memory.add_message(
                session_id=self.session_id,
                role="user",
                content=request,
                metadata=context or {},
                importance_score=0.6
            )
            
            # Retrieve relevant memory context
            memory_context = self.memory_retrieval.retrieve_contextual_memory(
                user_id=self.user_id,
                session_id=self.session_id,
                agent_id=self.agent_id,
                query=request
            )
            
            # Get personalized context
            personalized_context = self.memory_retrieval.get_personalized_context(
                user_id=self.user_id,
                symbol=context.get('symbol') if context else None
            )
            
            # Process the request with memory context
            response = await self._process_request_with_context(
                request, memory_context, personalized_context, context
            )
            
            # Store agent response
            self.conversation_memory.add_message(
                session_id=self.session_id,
                role="assistant",
                content=str(response),
                metadata={'agent_id': self.agent_id, 'agent_type': self.agent_type},
                importance_score=0.7
            )
            
            # Update working memory
            self._update_working_memory(request, response, memory_context)
            
            # Learn from interaction
            self._learn_from_interaction(request, response, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request with memory: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_request_with_context(self, request: str, memory_context: Dict,
                                          personalized_context: Dict, 
                                          original_context: Optional[Dict]) -> Dict[str, Any]:
        """Override in subclasses to implement agent-specific logic"""
        return {
            'success': True,
            'message': 'Base agent response',
            'memory_context_used': len(memory_context.get('semantic_memories', [])),
            'personalization_applied': bool(personalized_context.get('user_preferences'))
        }
    
    def _update_working_memory(self, request: str, response: Dict, memory_context: Dict):
        """Update agent's working memory"""
        try:
            current_memory = self.agent_context.get_working_memory(self.session_id, self.agent_id)
            
            # Update with latest interaction
            current_memory['last_request'] = request
            current_memory['last_response'] = response
            current_memory['last_interaction_time'] = datetime.utcnow().isoformat()
            current_memory['memory_context_size'] = len(memory_context.get('semantic_memories', []))
            
            # Track recent requests
            if 'recent_requests' not in current_memory:
                current_memory['recent_requests'] = []
            
            current_memory['recent_requests'].append({
                'request': request,
                'timestamp': datetime.utcnow().isoformat(),
                'success': response.get('success', False)
            })
            
            # Keep only last 10 requests
            current_memory['recent_requests'] = current_memory['recent_requests'][-10:]
            
            self.agent_context.save_working_memory(
                session_id=self.session_id,
                agent_id=self.agent_id,
                memory_data=current_memory
            )
            
        except Exception as e:
            logger.error(f"Error updating working memory: {e}")
    
    def _learn_from_interaction(self, request: str, response: Dict, context: Optional[Dict]):
        """Learn from the current interaction"""
        try:
            learning_data = {
                'request_type': self._classify_request(request),
                'request': request,
                'response_success': response.get('success', False),
                'context_available': bool(context),
                'response_quality': self._assess_response_quality(response),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.agent_context.learn_from_interaction(
                session_id=self.session_id,
                agent_id=self.agent_id,
                interaction_data=learning_data
            )
            
            # Store important interactions in semantic memory
            if learning_data['response_quality'] > 0.7:
                self.semantic_memory.store_memory(
                    user_id=self.user_id,
                    memory_type="episodic",
                    content=f"Request: {request}\nResponse: {str(response)}",
                    context=f"Successful interaction with {self.agent_type}",
                    confidence=learning_data['response_quality']
                )
                
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    def _classify_request(self, request: str) -> str:
        """Classify the type of request"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['load', 'fetch', 'get', 'retrieve']):
            return 'data_request'
        elif any(word in request_lower for word in ['analyze', 'analysis', 'insight']):
            return 'analysis_request'
        elif any(word in request_lower for word in ['store', 'save', 'update']):
            return 'storage_request'
        elif any(word in request_lower for word in ['a2a', 'message', 'protocol']):
            return 'a2a_communication'
        else:
            return 'general'
    
    def _assess_response_quality(self, response: Dict) -> float:
        """Assess the quality of the response"""
        quality = 0.5  # Base quality
        
        if response.get('success'):
            quality += 0.3
        
        if response.get('data') or response.get('result'):
            quality += 0.2
        
        if response.get('error'):
            quality -= 0.3
        
        return max(0.0, min(1.0, quality))
    
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
    
    def add_goal(self, goal_description: str, priority: str = "medium") -> bool:
        """Add a goal to the agent"""
        goal = {
            'description': goal_description,
            'priority': priority,
            'added_by': 'agent',
            'context': self.agent_type
        }
        
        return self.agent_context.add_goal(
            session_id=self.session_id,
            agent_id=self.agent_id,
            goal=goal
        )
    
    def get_active_goals(self) -> List[Dict]:
        """Get agent's active goals"""
        return self.agent_context.get_active_goals(
            session_id=self.session_id,
            agent_id=self.agent_id
        )
    
    def set_state(self, state: str, state_data: Optional[Dict] = None):
        """Set agent's current state"""
        return self.agent_context.set_agent_state(
            session_id=self.session_id,
            agent_id=self.agent_id,
            state=state,
            state_data=state_data
        )
    
    def get_state(self) -> Dict:
        """Get agent's current state"""
        return self.agent_context.get_agent_state(
            session_id=self.session_id,
            agent_id=self.agent_id
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
    
    def update_preferences(self, preferences: Dict) -> bool:
        """Update user preferences based on interaction"""
        return self.conversation_memory.update_user_preferences(
            session_id=self.session_id,
            preferences=preferences
        )
    
    def get_memory_insights(self) -> Dict:
        """Get insights about memory usage"""
        return self.memory_retrieval.get_memory_insights(self.user_id)
    
    def transfer_context_to_session(self, new_session_id: str) -> bool:
        """Transfer agent context to a new session"""
        return self.agent_context.transfer_context_to_session(
            from_session_id=self.session_id,
            to_session_id=new_session_id,
            agent_id=self.agent_id
        )
    
    async def create_shared_memory(self, content: str, memory_type: str, 
                                 process_id: str = 'trading_process', 
                                 importance_score: float = 0.7) -> Dict[str, Any]:
        """Create shared memory via A2A message"""
        memory_message = A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id="memory-system",
            message_type=MessageType.MEMORY_SHARE,
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
        
        return await a2a_memory_system.process_a2a_memory_message(memory_message)
    
    async def create_private_memory(self, content: str, memory_type: str,
                                  importance_score: float = 0.5) -> Dict[str, Any]:
        """Create private memory via A2A message"""
        memory_message = A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id="memory-system",
            message_type=MessageType.MEMORY_SHARE,
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
        
        return await a2a_memory_system.process_a2a_memory_message(memory_message)
    
    async def query_memories(self, query: str, scope: str = 'both') -> Dict[str, Any]:
        """Query memories via A2A message"""
        memory_message = A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id="memory-system",
            message_type=MessageType.MEMORY_REQUEST,
            payload={
                'request_type': 'query',
                'query': query,
                'scope': scope
            }
        )
        
        return await a2a_memory_system.process_a2a_memory_message(memory_message)
    
    async def get_recent_memories(self, limit: int = 10, scope: str = 'both') -> Dict[str, Any]:
        """Get recent memories via A2A message"""
        memory_message = A2AProtocol.create_message(
            sender_id=self.agent_id,
            receiver_id="memory-system",
            message_type=MessageType.MEMORY_REQUEST,
            payload={
                'request_type': 'recent',
                'limit': limit,
                'scope': scope
            }
        )
        
        return await a2a_memory_system.process_a2a_memory_message(memory_message)
    
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
            'memory_insights': self.get_memory_insights(),
            'a2a_memories': a2a_memory_system.export_memories(self.agent_id)
        }