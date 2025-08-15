"""
Memory Retrieval System
Orchestrates memory retrieval across conversation, agent context, and semantic memory systems
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from .conversation_memory import ConversationMemoryManager
from .agent_context import AgentContextManager
from .semantic_memory import SemanticMemoryManager

logger = logging.getLogger(__name__)

class MemoryRetrievalSystem:
    """Unified memory retrieval system for AI agents"""
    
    def __init__(self):
        self.conversation_memory = ConversationMemoryManager()
        self.agent_context = AgentContextManager()
        self.semantic_memory = SemanticMemoryManager()
        
    def initialize_agent_memory(self, user_id: int, session_id: str, agent_id: str, 
                              agent_type: str) -> Dict:
        """Initialize complete memory context for an agent"""
        try:
            # Get or create conversation session
            conv_session = self.conversation_memory.get_session(session_id)
            if not conv_session:
                session_id = self.conversation_memory.create_session(
                    user_id=user_id,
                    agent_type=agent_type
                )
            
            # Get or create agent context
            agent_context_data = self.agent_context.get_agent_context(session_id, agent_id)
            if not agent_context_data:
                self.agent_context.create_agent_context(
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_type=agent_type
                )
                agent_context_data = self.agent_context.get_agent_context(session_id, agent_id)
            
            # Get user preferences
            preferences = self.conversation_memory.get_user_preferences(user_id)
            
            # Get relevant semantic memories
            recent_memories = self.semantic_memory.search_similar_memories(
                user_id=user_id,
                query=f"trading {agent_type}",
                limit=10
            )
            
            memory_context = {
                'session_id': session_id,
                'agent_context': agent_context_data,
                'user_preferences': preferences,
                'recent_memories': recent_memories,
                'conversation_summary': self.conversation_memory.get_conversation_summary(session_id),
                'initialized_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Initialized memory context for agent {agent_id}")
            return memory_context
            
        except Exception as e:
            logger.error(f"Error initializing agent memory: {e}")
            return {}
    
    def retrieve_contextual_memory(self, user_id: int, session_id: str, agent_id: str,
                                 query: str, context_type: str = "general") -> Dict:
        """Retrieve relevant memories based on current context"""
        try:
            # Get conversation history
            conversation_history = self.conversation_memory.get_conversation_history(
                session_id=session_id,
                limit=20
            )
            
            # Get similar conversations
            similar_conversations = self.conversation_memory.find_similar_conversations(
                user_id=user_id,
                query=query,
                limit=5
            )
            
            # Get semantic memories
            semantic_results = self.semantic_memory.search_similar_memories(
                user_id=user_id,
                query=query,
                limit=10
            )
            
            # Get memory fragments
            fragment_results = self.semantic_memory.search_memory_fragments(
                user_id=user_id,
                query=query,
                limit=10
            )
            
            # Get agent's working memory
            working_memory = self.agent_context.get_working_memory(session_id, agent_id)
            
            # Get active goals
            active_goals = self.agent_context.get_active_goals(session_id, agent_id)
            
            # Compile contextual memory
            contextual_memory = {
                'query': query,
                'context_type': context_type,
                'conversation_history': conversation_history,
                'similar_conversations': similar_conversations,
                'semantic_memories': semantic_results,
                'memory_fragments': fragment_results,
                'working_memory': working_memory,
                'active_goals': active_goals,
                'retrieved_at': datetime.utcnow().isoformat(),
                'relevance_scores': self._calculate_relevance_scores(
                    query, semantic_results, fragment_results
                )
            }
            
            return contextual_memory
            
        except Exception as e:
            logger.error(f"Error retrieving contextual memory: {e}")
            return {}
    
    def store_interaction_memory(self, user_id: int, session_id: str, agent_id: str,
                               user_message: str, agent_response: str, 
                               interaction_metadata: Optional[Dict] = None) -> bool:
        """Store complete interaction in memory systems"""
        try:
            # Store in conversation memory
            self.conversation_memory.add_message(
                session_id=session_id,
                role="user",
                content=user_message,
                metadata=interaction_metadata,
                importance_score=0.6
            )
            
            self.conversation_memory.add_message(
                session_id=session_id,
                role="assistant", 
                content=agent_response,
                metadata=interaction_metadata,
                importance_score=0.6
            )
            
            # Extract key information for semantic memory
            combined_content = f"User: {user_message}\nAgent: {agent_response}"
            
            # Store important interactions in semantic memory
            if interaction_metadata and interaction_metadata.get('important', False):
                self.semantic_memory.store_memory(
                    user_id=user_id,
                    memory_type="episodic",
                    content=combined_content,
                    context=f"Trading interaction with {agent_id}",
                    confidence=0.7
                )
            
            # Update agent learning data
            learning_data = {
                'interaction_type': interaction_metadata.get('type', 'general') if interaction_metadata else 'general',
                'user_query': user_message,
                'agent_response': agent_response,
                'success_metrics': interaction_metadata.get('success_metrics', {}) if interaction_metadata else {},
                'user_feedback': interaction_metadata.get('feedback') if interaction_metadata else None
            }
            
            self.agent_context.learn_from_interaction(session_id, agent_id, learning_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing interaction memory: {e}")
            return False
    
    def get_personalized_context(self, user_id: int, symbol: Optional[str] = None) -> Dict:
        """Get personalized context for trading decisions"""
        try:
            # Get user preferences
            preferences = self.conversation_memory.get_user_preferences(user_id)
            
            # Get symbol-specific memories if provided
            symbol_memories = []
            if symbol:
                symbol_memories = self.semantic_memory.get_memories_by_symbol(
                    user_id=user_id,
                    symbol=symbol,
                    limit=10
                )
            
            # Get recent trading patterns
            trading_patterns = self.semantic_memory.search_similar_memories(
                user_id=user_id,
                query="trading decision pattern outcome",
                memory_type="procedural",
                limit=10
            )
            
            # Get risk preferences
            risk_memories = self.semantic_memory.search_memory_fragments(
                user_id=user_id,
                query="risk tolerance preference",
                fragment_type="preference",
                limit=5
            )
            
            personalized_context = {
                'user_preferences': preferences,
                'symbol_specific_memories': symbol_memories,
                'trading_patterns': trading_patterns,
                'risk_preferences': risk_memories,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return personalized_context
            
        except Exception as e:
            logger.error(f"Error getting personalized context: {e}")
            return {}
    
    def update_memory_from_outcome(self, user_id: int, session_id: str, agent_id: str,
                                 decision_context: Dict, outcome: Dict) -> bool:
        """Update memories based on trading outcomes"""
        try:
            # Store outcome in semantic memory
            outcome_content = f"Decision: {decision_context.get('decision', 'unknown')}, " \
                            f"Symbol: {decision_context.get('symbol', 'unknown')}, " \
                            f"Outcome: {outcome.get('result', 'unknown')}, " \
                            f"PnL: {outcome.get('pnl', 'unknown')}"
            
            confidence = 0.8 if outcome.get('result') == 'profit' else 0.6
            
            self.semantic_memory.store_memory(
                user_id=user_id,
                memory_type="procedural",
                content=outcome_content,
                context="Trading outcome",
                keywords=['trading', 'outcome', decision_context.get('strategy', 'unknown')],
                associated_symbols=[decision_context.get('symbol', 'unknown')],
                confidence=confidence
            )
            
            # Update agent learning
            learning_data = {
                'decision_context': decision_context,
                'outcome': outcome,
                'success': outcome.get('result') == 'profit',
                'learned_from': 'trading_outcome'
            }
            
            self.agent_context.learn_from_interaction(session_id, agent_id, learning_data)
            
            # Reinforce or weaken related memories based on outcome
            if outcome.get('result') == 'profit':
                # Find and reinforce similar successful patterns
                similar_memories = self.semantic_memory.search_similar_memories(
                    user_id=user_id,
                    query=outcome_content,
                    limit=5
                )
                
                for memory in similar_memories:
                    if memory['similarity_score'] > 0.7:
                        self.semantic_memory.reinforce_memory(
                            memory_id=memory['id'],
                            confidence_boost=0.1
                        )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory from outcome: {e}")
            return False
    
    def get_memory_insights(self, user_id: int, timeframe_days: int = 30) -> Dict:
        """Get insights from memory data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
            
            # Get conversation statistics
            with self.conversation_memory.db.get_session() as session:
                from ..database import ConversationSession, ConversationMessage
                
                sessions = session.query(ConversationSession).filter(
                    ConversationSession.user_id == user_id,
                    ConversationSession.updated_at >= cutoff_date
                ).all()
                
                total_messages = 0
                agent_types_used = set()
                
                for conv_session in sessions:
                    agent_types_used.add(conv_session.agent_type)
                    message_count = session.query(ConversationMessage).filter(
                        ConversationMessage.session_id == conv_session.session_id
                    ).count()
                    total_messages += message_count
            
            # Get semantic memory insights
            from ..database import SemanticMemory
            with self.semantic_memory.db.get_session() as session:
                recent_memories = session.query(SemanticMemory).filter(
                    SemanticMemory.user_id == user_id,
                    SemanticMemory.created_at >= cutoff_date
                ).all()
                
                memory_types = {}
                confidence_avg = 0
                symbols_mentioned = set()
                
                for memory in recent_memories:
                    memory_types[memory.memory_type] = memory_types.get(memory.memory_type, 0) + 1
                    confidence_avg += memory.confidence
                    
                    if memory.associated_symbols:
                        symbols_mentioned.update(memory.associated_symbols.split())
                
                if recent_memories:
                    confidence_avg /= len(recent_memories)
            
            insights = {
                'timeframe_days': timeframe_days,
                'conversation_stats': {
                    'total_sessions': len(sessions),
                    'total_messages': total_messages,
                    'agent_types_used': list(agent_types_used),
                    'avg_messages_per_session': total_messages / max(1, len(sessions))
                },
                'memory_stats': {
                    'total_memories': len(recent_memories),
                    'memory_types_distribution': memory_types,
                    'average_confidence': confidence_avg,
                    'symbols_discussed': list(symbols_mentioned)
                },
                'learning_indicators': {
                    'memory_retention_rate': min(1.0, len(recent_memories) / max(1, timeframe_days)),
                    'conversation_engagement': min(1.0, total_messages / max(1, timeframe_days * 2)),
                    'knowledge_diversity': len(memory_types) / max(1, len(recent_memories)) if recent_memories else 0
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting memory insights: {e}")
            return {}
    
    def _calculate_relevance_scores(self, query: str, semantic_results: List[Dict], 
                                  fragment_results: List[Dict]) -> Dict:
        """Calculate relevance scores for retrieved memories"""
        try:
            # Combine all results
            all_results = []
            all_results.extend([(r, 'semantic') for r in semantic_results])
            all_results.extend([(r, 'fragment') for r in fragment_results])
            
            if not all_results:
                return {'average_relevance': 0.0, 'total_results': 0}
            
            # Calculate scores
            total_score = 0
            for result, result_type in all_results:
                similarity = result.get('similarity_score', 0.0)
                confidence = result.get('confidence', result.get('relevance_score', 0.5))
                
                # Weight by type and recency
                weight = 1.0
                if result_type == 'semantic':
                    weight = 1.2  # Slight preference for semantic memories
                
                score = (similarity * 0.6 + confidence * 0.4) * weight
                total_score += score
            
            return {
                'average_relevance': total_score / len(all_results),
                'total_results': len(all_results),
                'semantic_count': len(semantic_results),
                'fragment_count': len(fragment_results)
            }
            
        except Exception as e:
            logger.error(f"Error calculating relevance scores: {e}")
            return {'average_relevance': 0.0, 'total_results': 0}
    
    def cleanup_all_memories(self, user_id: int) -> Dict:
        """Run cleanup across all memory systems"""
        try:
            # Cleanup conversations
            conv_cleaned = self.conversation_memory.cleanup_old_conversations(days_to_keep=30)
            
            # Cleanup agent contexts
            context_cleaned = self.agent_context.cleanup_expired_contexts(hours_to_keep=24)
            
            # Apply memory decay
            decay_applied = self.semantic_memory.decay_memories(user_id=user_id)
            
            # Consolidate similar memories
            consolidated = self.semantic_memory.consolidate_memories(user_id=user_id)
            
            cleanup_results = {
                'conversations_cleaned': conv_cleaned,
                'contexts_cleaned': context_cleaned,
                'memories_decayed': decay_applied,
                'memories_consolidated': consolidated,
                'cleaned_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Memory cleanup completed for user {user_id}: {cleanup_results}")
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return {}