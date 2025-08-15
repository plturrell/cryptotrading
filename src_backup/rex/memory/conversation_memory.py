"""
Conversation Memory Manager
Handles persistent storage and retrieval of conversational context across sessions
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session

from ..database import get_db, ConversationSession, ConversationMessage, User
from .semantic_memory import SemanticMemoryManager

logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    """Manages persistent conversation memory across sessions"""
    
    def __init__(self):
        self.db = get_db()
        self.semantic_memory = SemanticMemoryManager()
        
    def create_session(self, user_id: int, agent_type: str, initial_context: Optional[Dict] = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        
        with self.db.get_session() as session:
            conv_session = ConversationSession(
                user_id=user_id,
                session_id=session_id,
                agent_type=agent_type,
                context_summary=json.dumps(initial_context or {}),
                preferences=json.dumps({})
            )
            session.add(conv_session)
            session.commit()
            
        logger.info(f"Created conversation session {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session by ID"""
        with self.db.get_session() as session:
            return session.query(ConversationSession).filter(
                ConversationSession.session_id == session_id
            ).first()
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Optional[Dict] = None, importance_score: float = 0.5) -> bool:
        """Add a message to the conversation"""
        try:
            # Generate embedding for semantic search
            embedding = self.semantic_memory.generate_embedding(content)
            
            with self.db.get_session() as session:
                message = ConversationMessage(
                    session_id=session_id,
                    role=role,
                    content=content,
                    message_metadata=json.dumps(metadata or {}),
                    embedding=json.dumps(embedding),
                    importance_score=importance_score,
                    token_count=len(content.split())
                )
                session.add(message)
                
                # Update session timestamp
                conv_session = session.query(ConversationSession).filter(
                    ConversationSession.session_id == session_id
                ).first()
                if conv_session:
                    conv_session.updated_at = datetime.utcnow()
                
                session.commit()
                
            # Store important messages in semantic memory
            if importance_score > 0.7:
                conv_session = self.get_session(session_id)
                if conv_session:
                    self.semantic_memory.store_memory(
                        user_id=conv_session.user_id,
                        memory_type="episodic",
                        content=content,
                        context=f"Conversation with {conv_session.agent_type}",
                        confidence=importance_score
                    )
                    
            return True
            
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 50, 
                               min_importance: float = 0.0) -> List[Dict]:
        """Get conversation history for a session"""
        with self.db.get_session() as session:
            messages = session.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id,
                ConversationMessage.importance_score >= min_importance
            ).order_by(ConversationMessage.created_at.desc()).limit(limit).all()
            
            return [{
                'id': msg.id,
                'role': msg.role,
                'content': msg.content,
                'metadata': json.loads(msg.message_metadata or '{}'),
                'importance_score': msg.importance_score,
                'created_at': msg.created_at.isoformat()
            } for msg in reversed(messages)]
    
    def get_conversation_summary(self, session_id: str) -> Optional[str]:
        """Get conversation summary"""
        conv_session = self.get_session(session_id)
        if conv_session and conv_session.context_summary:
            return json.loads(conv_session.context_summary).get('summary')
        return None
    
    def update_conversation_summary(self, session_id: str, summary: str) -> bool:
        """Update conversation summary"""
        try:
            with self.db.get_session() as session:
                conv_session = session.query(ConversationSession).filter(
                    ConversationSession.session_id == session_id
                ).first()
                
                if conv_session:
                    context = json.loads(conv_session.context_summary or '{}')
                    context['summary'] = summary
                    context['updated_at'] = datetime.utcnow().isoformat()
                    conv_session.context_summary = json.dumps(context)
                    conv_session.updated_at = datetime.utcnow()
                    session.commit()
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error updating conversation summary for session {session_id}: {e}")
            return False
    
    def find_similar_conversations(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        """Find similar conversations using semantic search"""
        try:
            query_embedding = self.semantic_memory.generate_embedding(query)
            
            with self.db.get_session() as session:
                # Get recent sessions for the user
                sessions = session.query(ConversationSession).filter(
                    ConversationSession.user_id == user_id,
                    ConversationSession.active == True
                ).order_by(ConversationSession.updated_at.desc()).limit(20).all()
                
                similar_conversations = []
                
                for conv_session in sessions:
                    # Get messages from this session
                    messages = session.query(ConversationMessage).filter(
                        ConversationMessage.session_id == conv_session.session_id
                    ).all()
                    
                    # Calculate similarity scores
                    for message in messages:
                        if message.embedding:
                            msg_embedding = json.loads(message.embedding)
                            similarity = self.semantic_memory.calculate_similarity(
                                query_embedding, msg_embedding
                            )
                            
                            if similarity > 0.7:  # High similarity threshold
                                similar_conversations.append({
                                    'session_id': conv_session.session_id,
                                    'agent_type': conv_session.agent_type,
                                    'message_content': message.content[:200] + "...",
                                    'similarity_score': similarity,
                                    'created_at': message.created_at.isoformat()
                                })
                
                # Sort by similarity and return top results
                similar_conversations.sort(key=lambda x: x['similarity_score'], reverse=True)
                return similar_conversations[:limit]
                
        except Exception as e:
            logger.error(f"Error finding similar conversations: {e}")
            return []
    
    def get_user_preferences(self, user_id: int) -> Dict:
        """Get aggregated user preferences from conversations"""
        with self.db.get_session() as session:
            sessions = session.query(ConversationSession).filter(
                ConversationSession.user_id == user_id
            ).all()
            
            aggregated_preferences = {
                'trading_style': 'moderate',
                'risk_tolerance': 'medium',
                'preferred_symbols': [],
                'communication_style': 'detailed',
                'notification_preferences': {}
            }
            
            for conv_session in sessions:
                if conv_session.preferences:
                    prefs = json.loads(conv_session.preferences)
                    # Merge preferences (implement preference aggregation logic)
                    for key, value in prefs.items():
                        if key in aggregated_preferences:
                            if isinstance(value, list):
                                aggregated_preferences[key].extend(value)
                            else:
                                aggregated_preferences[key] = value
                                
            return aggregated_preferences
    
    def update_user_preferences(self, session_id: str, preferences: Dict) -> bool:
        """Update user preferences for a session"""
        try:
            with self.db.get_session() as session:
                conv_session = session.query(ConversationSession).filter(
                    ConversationSession.session_id == session_id
                ).first()
                
                if conv_session:
                    current_prefs = json.loads(conv_session.preferences or '{}')
                    current_prefs.update(preferences)
                    conv_session.preferences = json.dumps(current_prefs)
                    conv_session.updated_at = datetime.utcnow()
                    session.commit()
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error updating preferences for session {session_id}: {e}")
            return False
    
    def cleanup_old_conversations(self, days_to_keep: int = 30) -> int:
        """Clean up old conversations beyond retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            with self.db.get_session() as session:
                # Mark old sessions as inactive instead of deleting
                old_sessions = session.query(ConversationSession).filter(
                    ConversationSession.updated_at < cutoff_date,
                    ConversationSession.active == True
                ).all()
                
                count = len(old_sessions)
                
                for conv_session in old_sessions:
                    conv_session.active = False
                    
                session.commit()
                
            logger.info(f"Marked {count} old conversation sessions as inactive")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {e}")
            return 0
    
    def export_conversation(self, session_id: str) -> Optional[Dict]:
        """Export complete conversation data"""
        try:
            conv_session = self.get_session(session_id)
            if not conv_session:
                return None
                
            messages = self.get_conversation_history(session_id, limit=1000)
            
            return {
                'session_info': {
                    'session_id': conv_session.session_id,
                    'user_id': conv_session.user_id,
                    'agent_type': conv_session.agent_type,
                    'created_at': conv_session.created_at.isoformat(),
                    'updated_at': conv_session.updated_at.isoformat(),
                    'context_summary': json.loads(conv_session.context_summary or '{}'),
                    'preferences': json.loads(conv_session.preferences or '{}')
                },
                'messages': messages,
                'total_messages': len(messages)
            }
            
        except Exception as e:
            logger.error(f"Error exporting conversation {session_id}: {e}")
            return None