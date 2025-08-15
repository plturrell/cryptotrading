"""
Memory Optimization and Cleanup System
Handles memory maintenance, optimization, and performance tuning
"""

import asyncio
import logging
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..database import (
    get_db, ConversationSession, ConversationMessage, AgentContext,
    MemoryFragment, SemanticMemory, User
)
from .conversation_memory import ConversationMemoryManager
from .agent_context import AgentContextManager
from .semantic_memory import SemanticMemoryManager

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Advanced memory optimization and maintenance system"""
    
    def __init__(self):
        self.db = get_db()
        self.conversation_memory = ConversationMemoryManager()
        self.agent_context = AgentContextManager()
        self.semantic_memory = SemanticMemoryManager()
        
        # Optimization settings
        self.optimization_config = {
            'max_conversations_per_user': 100,
            'max_messages_per_conversation': 1000,
            'memory_decay_days': 30,
            'consolidation_similarity_threshold': 0.85,
            'importance_threshold_for_retention': 0.3,
            'context_retention_hours': 72,
            'fragment_access_threshold': 5
        }
        
    async def run_full_optimization(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Run complete memory optimization suite"""
        optimization_start = datetime.utcnow()
        results = {
            'started_at': optimization_start.isoformat(),
            'user_id': user_id,
            'optimization_steps': {}
        }
        
        try:
            logger.info(f"Starting full memory optimization for user {user_id or 'all users'}")
            
            # Step 1: Conversation cleanup
            conv_results = await self._optimize_conversations(user_id)
            results['optimization_steps']['conversation_cleanup'] = conv_results
            
            # Step 2: Agent context optimization
            context_results = await self._optimize_agent_contexts(user_id)
            results['optimization_steps']['context_optimization'] = context_results
            
            # Step 3: Semantic memory consolidation
            semantic_results = await self._optimize_semantic_memory(user_id)
            results['optimization_steps']['semantic_optimization'] = semantic_results
            
            # Step 4: Memory fragment cleanup
            fragment_results = await self._optimize_memory_fragments(user_id)
            results['optimization_steps']['fragment_optimization'] = fragment_results
            
            # Step 5: Performance analysis
            performance_results = await self._analyze_memory_performance(user_id)
            results['optimization_steps']['performance_analysis'] = performance_results
            
            # Step 6: Recommendation generation
            recommendations = await self._generate_optimization_recommendations(user_id)
            results['optimization_steps']['recommendations'] = recommendations
            
            results['completed_at'] = datetime.utcnow().isoformat()
            results['duration_seconds'] = (datetime.utcnow() - optimization_start).total_seconds()
            results['overall_success'] = True
            
            logger.info(f"Memory optimization completed in {results['duration_seconds']:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")
            results['error'] = str(e)
            results['overall_success'] = False
            results['completed_at'] = datetime.utcnow().isoformat()
            return results
    
    async def _optimize_conversations(self, user_id: Optional[int]) -> Dict[str, Any]:
        """Optimize conversation data"""
        try:
            with self.db.get_session() as session:
                # Base query
                query = session.query(ConversationSession)
                if user_id:
                    query = query.filter(ConversationSession.user_id == user_id)
                
                # Get conversation statistics
                total_sessions = query.count()
                active_sessions = query.filter(ConversationSession.active == True).count()
                
                # Identify old conversations for deactivation
                cutoff_date = datetime.utcnow() - timedelta(days=self.optimization_config['memory_decay_days'])
                old_sessions = query.filter(
                    ConversationSession.updated_at < cutoff_date,
                    ConversationSession.active == True
                ).all()
                
                deactivated_count = 0
                for conv_session in old_sessions:
                    conv_session.active = False
                    deactivated_count += 1
                
                # Cleanup oversized conversations
                oversized_cleanup = 0
                all_sessions = query.filter(ConversationSession.active == True).all()
                
                for conv_session in all_sessions:
                    message_count = session.query(ConversationMessage).filter(
                        ConversationMessage.session_id == conv_session.session_id
                    ).count()
                    
                    if message_count > self.optimization_config['max_messages_per_conversation']:
                        # Remove oldest messages, keeping only the most recent and important ones
                        messages_to_remove = session.query(ConversationMessage).filter(
                            ConversationMessage.session_id == conv_session.session_id,
                            ConversationMessage.importance_score < self.optimization_config['importance_threshold_for_retention']
                        ).order_by(ConversationMessage.created_at.asc()).limit(
                            message_count - self.optimization_config['max_messages_per_conversation']
                        ).all()
                        
                        for msg in messages_to_remove:
                            session.delete(msg)
                            oversized_cleanup += 1
                
                session.commit()
                
                return {
                    'total_sessions': total_sessions,
                    'active_sessions_before': active_sessions,
                    'sessions_deactivated': deactivated_count,
                    'messages_cleaned': oversized_cleanup,
                    'active_sessions_after': active_sessions - deactivated_count
                }
                
        except Exception as e:
            logger.error(f"Error optimizing conversations: {e}")
            return {'error': str(e)}
    
    async def _optimize_agent_contexts(self, user_id: Optional[int]) -> Dict[str, Any]:
        """Optimize agent context data"""
        try:
            with self.db.get_session() as session:
                # Base query
                query = session.query(AgentContext)
                if user_id:
                    # Filter by user_id through session relationship
                    query = query.join(ConversationSession).filter(
                        ConversationSession.user_id == user_id
                    )
                
                total_contexts = query.count()
                
                # Cleanup expired contexts
                cutoff_time = datetime.utcnow() - timedelta(
                    hours=self.optimization_config['context_retention_hours']
                )
                
                expired_contexts = query.filter(
                    or_(
                        AgentContext.expires_at < datetime.utcnow(),
                        and_(
                            AgentContext.updated_at < cutoff_time,
                            AgentContext.active == True
                        )
                    )
                ).all()
                
                expired_count = 0
                for context in expired_contexts:
                    context.active = False
                    expired_count += 1
                
                # Consolidate old versions
                version_cleanup = 0
                active_contexts = query.filter(AgentContext.active == True).all()
                
                # Group by session_id and agent_id
                context_groups = {}
                for context in active_contexts:
                    key = f"{context.session_id}:{context.agent_id}"
                    if key not in context_groups:
                        context_groups[key] = []
                    context_groups[key].append(context)
                
                # Keep only the latest version for each agent
                for key, contexts in context_groups.items():
                    if len(contexts) > 1:
                        # Sort by version, keep the latest
                        contexts.sort(key=lambda x: x.version, reverse=True)
                        for old_context in contexts[1:]:  # All except the first (latest)
                            old_context.active = False
                            version_cleanup += 1
                
                session.commit()
                
                return {
                    'total_contexts': total_contexts,
                    'expired_contexts_deactivated': expired_count,
                    'old_versions_cleaned': version_cleanup,
                    'optimization_completed': True
                }
                
        except Exception as e:
            logger.error(f"Error optimizing agent contexts: {e}")
            return {'error': str(e)}
    
    async def _optimize_semantic_memory(self, user_id: Optional[int]) -> Dict[str, Any]:
        """Optimize semantic memory through consolidation and decay"""
        try:
            # Apply memory decay
            decay_results = 0
            if user_id:
                decay_results = self.semantic_memory.decay_memories(user_id)
            else:
                # Apply to all users
                with self.db.get_session() as session:
                    users = session.query(User).all()
                    for user in users:
                        decay_results += self.semantic_memory.decay_memories(user.id)
            
            # Consolidate similar memories
            consolidation_results = 0
            if user_id:
                consolidation_results = self.semantic_memory.consolidate_memories(
                    user_id, min_similarity=self.optimization_config['consolidation_similarity_threshold']
                )
            else:
                with self.db.get_session() as session:
                    users = session.query(User).all()
                    for user in users:
                        consolidation_results += self.semantic_memory.consolidate_memories(
                            user.id, min_similarity=self.optimization_config['consolidation_similarity_threshold']
                        )
            
            # Cleanup low-confidence memories
            with self.db.get_session() as session:
                query = session.query(SemanticMemory).filter(
                    SemanticMemory.confidence < 0.2,
                    SemanticMemory.reinforcement_count < 2
                )
                
                if user_id:
                    query = query.filter(SemanticMemory.user_id == user_id)
                
                low_confidence_memories = query.all()
                cleanup_count = len(low_confidence_memories)
                
                for memory in low_confidence_memories:
                    session.delete(memory)
                
                session.commit()
            
            return {
                'memories_decayed': decay_results,
                'memories_consolidated': consolidation_results,
                'low_confidence_cleaned': cleanup_count,
                'optimization_completed': True
            }
            
        except Exception as e:
            logger.error(f"Error optimizing semantic memory: {e}")
            return {'error': str(e)}
    
    async def _optimize_memory_fragments(self, user_id: Optional[int]) -> Dict[str, Any]:
        """Optimize memory fragments"""
        try:
            with self.db.get_session() as session:
                query = session.query(MemoryFragment)
                if user_id:
                    query = query.filter(MemoryFragment.user_id == user_id)
                
                total_fragments = query.count()
                
                # Remove rarely accessed fragments
                unused_fragments = query.filter(
                    or_(
                        MemoryFragment.access_count < self.optimization_config['fragment_access_threshold'],
                        MemoryFragment.last_accessed < datetime.utcnow() - timedelta(days=60)
                    )
                ).all()
                
                removed_count = len(unused_fragments)
                for fragment in unused_fragments:
                    session.delete(fragment)
                
                # Boost frequently accessed fragments
                popular_fragments = query.filter(
                    MemoryFragment.access_count >= 10
                ).all()
                
                boosted_count = 0
                for fragment in popular_fragments:
                    if fragment.relevance_score < 0.8:
                        fragment.relevance_score = min(1.0, fragment.relevance_score + 0.1)
                        boosted_count += 1
                
                session.commit()
                
                return {
                    'total_fragments': total_fragments,
                    'unused_fragments_removed': removed_count,
                    'popular_fragments_boosted': boosted_count,
                    'optimization_completed': True
                }
                
        except Exception as e:
            logger.error(f"Error optimizing memory fragments: {e}")
            return {'error': str(e)}
    
    async def _analyze_memory_performance(self, user_id: Optional[int]) -> Dict[str, Any]:
        """Analyze memory system performance"""
        try:
            with self.db.get_session() as session:
                performance_metrics = {}
                
                # Conversation metrics
                conv_query = session.query(ConversationSession)
                if user_id:
                    conv_query = conv_query.filter(ConversationSession.user_id == user_id)
                
                active_conversations = conv_query.filter(ConversationSession.active == True).count()
                avg_messages = session.query(func.avg(
                    session.query(ConversationMessage).filter(
                        ConversationMessage.session_id == ConversationSession.session_id
                    ).count()
                )).scalar() or 0
                
                # Memory metrics
                memory_query = session.query(SemanticMemory)
                if user_id:
                    memory_query = memory_query.filter(SemanticMemory.user_id == user_id)
                
                total_memories = memory_query.count()
                avg_confidence = memory_query.with_entities(
                    func.avg(SemanticMemory.confidence)
                ).scalar() or 0
                
                # Context metrics
                context_query = session.query(AgentContext)
                if user_id:
                    context_query = context_query.join(ConversationSession).filter(
                        ConversationSession.user_id == user_id
                    )
                
                active_contexts = context_query.filter(AgentContext.active == True).count()
                
                performance_metrics = {
                    'active_conversations': active_conversations,
                    'avg_messages_per_conversation': float(avg_messages),
                    'total_semantic_memories': total_memories,
                    'avg_memory_confidence': float(avg_confidence),
                    'active_agent_contexts': active_contexts,
                    'memory_efficiency_score': self._calculate_efficiency_score({
                        'conversations': active_conversations,
                        'memories': total_memories,
                        'contexts': active_contexts,
                        'confidence': avg_confidence
                    })
                }
                
                return performance_metrics
                
        except Exception as e:
            logger.error(f"Error analyzing memory performance: {e}")
            return {'error': str(e)}
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall memory system efficiency score"""
        try:
            # Normalize metrics to 0-1 scale
            conversation_score = min(1.0, metrics['conversations'] / 50)  # Max 50 conversations
            memory_score = min(1.0, metrics['memories'] / 1000)  # Max 1000 memories
            context_score = min(1.0, metrics['contexts'] / 20)  # Max 20 contexts
            confidence_score = metrics['confidence']  # Already 0-1
            
            # Weighted average
            weights = {
                'conversation': 0.2,
                'memory': 0.3,
                'context': 0.2,
                'confidence': 0.3
            }
            
            efficiency = (
                conversation_score * weights['conversation'] +
                memory_score * weights['memory'] +
                context_score * weights['context'] +
                confidence_score * weights['confidence']
            )
            
            return float(efficiency)
            
        except Exception as e:
            logger.error(f"Error calculating efficiency score: {e}")
            return 0.5
    
    async def _generate_optimization_recommendations(self, user_id: Optional[int]) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze current state
            with self.db.get_session() as session:
                if user_id:
                    # User-specific recommendations
                    conv_count = session.query(ConversationSession).filter(
                        ConversationSession.user_id == user_id,
                        ConversationSession.active == True
                    ).count()
                    
                    memory_count = session.query(SemanticMemory).filter(
                        SemanticMemory.user_id == user_id
                    ).count()
                    
                    avg_confidence = session.query(func.avg(SemanticMemory.confidence)).filter(
                        SemanticMemory.user_id == user_id
                    ).scalar() or 0
                    
                    if conv_count > 80:
                        recommendations.append({
                            'type': 'conversation_cleanup',
                            'priority': 'high',
                            'description': 'Consider archiving old conversations to improve performance'
                        })
                    
                    if memory_count > 800:
                        recommendations.append({
                            'type': 'memory_consolidation',
                            'priority': 'medium',
                            'description': 'Run memory consolidation to reduce redundancy'
                        })
                    
                    if avg_confidence < 0.6:
                        recommendations.append({
                            'type': 'memory_quality',
                            'priority': 'high',
                            'description': 'Focus on reinforcing high-quality memories'
                        })
                
                else:
                    # System-wide recommendations
                    total_sessions = session.query(ConversationSession).count()
                    total_memories = session.query(SemanticMemory).count()
                    
                    if total_sessions > 1000:
                        recommendations.append({
                            'type': 'system_cleanup',
                            'priority': 'high',
                            'description': 'System-wide cleanup recommended for optimal performance'
                        })
                    
                    if total_memories > 10000:
                        recommendations.append({
                            'type': 'memory_archiving',
                            'priority': 'medium',
                            'description': 'Consider implementing memory archiving strategy'
                        })
            
            # Always recommend regular optimization
            recommendations.append({
                'type': 'regular_maintenance',
                'priority': 'low',
                'description': 'Schedule regular memory optimization for best performance'
            })
            
            return {
                'recommendations': recommendations,
                'next_optimization_suggested': (
                    datetime.utcnow() + timedelta(days=7)
                ).isoformat(),
                'optimization_frequency': 'weekly'
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'error': str(e)}
    
    async def schedule_periodic_optimization(self, interval_hours: int = 24):
        """Schedule periodic memory optimization"""
        while True:
            try:
                logger.info("Running scheduled memory optimization")
                await self.run_full_optimization()
                await asyncio.sleep(interval_hours * 3600)  # Convert hours to seconds
                
            except Exception as e:
                logger.error(f"Error in scheduled optimization: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    def update_optimization_config(self, new_config: Dict[str, Any]) -> bool:
        """Update optimization configuration"""
        try:
            self.optimization_config.update(new_config)
            logger.info(f"Updated optimization config: {new_config}")
            return True
        except Exception as e:
            logger.error(f"Error updating optimization config: {e}")
            return False