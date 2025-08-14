"""
Agent Context Manager
Handles persistent agent context and state across sessions for seamless continuation
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from sqlalchemy.orm import Session

from ..database import get_db, AgentContext, ConversationSession

logger = logging.getLogger(__name__)

class AgentContextManager:
    """Manages persistent agent context across sessions"""
    
    def __init__(self):
        self.db = get_db()
        self._context_cache = {}  # In-memory cache for active contexts
        
    def create_agent_context(self, session_id: str, agent_id: str, agent_type: str,
                           initial_context: Optional[Dict] = None) -> bool:
        """Create initial context for an agent"""
        try:
            context_data = initial_context or {
                'working_memory': {},
                'goals': [],
                'knowledge_base': {},
                'state': 'initialized',
                'capabilities': [],
                'preferences': {},
                'learning_data': {}
            }
            
            with self.db.get_session() as session:
                agent_context = AgentContext(
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_type=agent_type,
                    context_type='complete',
                    context_data=json.dumps(context_data),
                    version=1,
                    active=True
                )
                session.add(agent_context)
                session.commit()
                
            # Cache the context
            self._context_cache[f"{session_id}:{agent_id}"] = context_data
            
            logger.info(f"Created context for agent {agent_id} in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating agent context: {e}")
            return False
    
    def get_agent_context(self, session_id: str, agent_id: str) -> Optional[Dict]:
        """Get current agent context"""
        cache_key = f"{session_id}:{agent_id}"
        
        # Check cache first
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
            
        try:
            with self.db.get_session() as session:
                context = session.query(AgentContext).filter(
                    AgentContext.session_id == session_id,
                    AgentContext.agent_id == agent_id,
                    AgentContext.active == True
                ).order_by(AgentContext.version.desc()).first()
                
                if context:
                    context_data = json.loads(context.context_data)
                    # Cache for future use
                    self._context_cache[cache_key] = context_data
                    return context_data
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting agent context: {e}")
            return None
    
    def update_agent_context(self, session_id: str, agent_id: str, 
                           context_updates: Dict, create_version: bool = False) -> bool:
        """Update agent context with new data"""
        try:
            current_context = self.get_agent_context(session_id, agent_id)
            if current_context is None:
                # Create initial context if it doesn't exist
                return self.create_agent_context(session_id, agent_id, "unknown", context_updates)
            
            # Merge updates into current context
            updated_context = self._merge_context(current_context, context_updates)
            
            with self.db.get_session() as session:
                if create_version:
                    # Create new version
                    latest_context = session.query(AgentContext).filter(
                        AgentContext.session_id == session_id,
                        AgentContext.agent_id == agent_id,
                        AgentContext.active == True
                    ).order_by(AgentContext.version.desc()).first()
                    
                    new_version = (latest_context.version + 1) if latest_context else 1
                    
                    new_context = AgentContext(
                        session_id=session_id,
                        agent_id=agent_id,
                        agent_type=latest_context.agent_type if latest_context else "unknown",
                        context_type='complete',
                        context_data=json.dumps(updated_context),
                        version=new_version,
                        active=True
                    )
                    session.add(new_context)
                else:
                    # Update existing context
                    context = session.query(AgentContext).filter(
                        AgentContext.session_id == session_id,
                        AgentContext.agent_id == agent_id,
                        AgentContext.active == True
                    ).order_by(AgentContext.version.desc()).first()
                    
                    if context:
                        context.context_data = json.dumps(updated_context)
                        context.updated_at = datetime.utcnow()
                
                session.commit()
                
            # Update cache
            cache_key = f"{session_id}:{agent_id}"
            self._context_cache[cache_key] = updated_context
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating agent context: {e}")
            return False
    
    def save_working_memory(self, session_id: str, agent_id: str, memory_data: Dict) -> bool:
        """Save agent's working memory"""
        return self.update_agent_context(
            session_id, agent_id, 
            {'working_memory': memory_data}
        )
    
    def get_working_memory(self, session_id: str, agent_id: str) -> Dict:
        """Get agent's working memory"""
        context = self.get_agent_context(session_id, agent_id)
        return context.get('working_memory', {}) if context else {}
    
    def add_goal(self, session_id: str, agent_id: str, goal: Dict) -> bool:
        """Add a goal to agent's goal list"""
        context = self.get_agent_context(session_id, agent_id)
        if context:
            goals = context.get('goals', [])
            goal['id'] = str(uuid.uuid4())
            goal['created_at'] = datetime.utcnow().isoformat()
            goal['status'] = 'active'
            goals.append(goal)
            return self.update_agent_context(session_id, agent_id, {'goals': goals})
        return False
    
    def update_goal_status(self, session_id: str, agent_id: str, goal_id: str, status: str) -> bool:
        """Update goal status"""
        context = self.get_agent_context(session_id, agent_id)
        if context:
            goals = context.get('goals', [])
            for goal in goals:
                if goal.get('id') == goal_id:
                    goal['status'] = status
                    goal['updated_at'] = datetime.utcnow().isoformat()
                    break
            return self.update_agent_context(session_id, agent_id, {'goals': goals})
        return False
    
    def get_active_goals(self, session_id: str, agent_id: str) -> List[Dict]:
        """Get agent's active goals"""
        context = self.get_agent_context(session_id, agent_id)
        if context:
            goals = context.get('goals', [])
            return [goal for goal in goals if goal.get('status') == 'active']
        return []
    
    def save_knowledge(self, session_id: str, agent_id: str, knowledge_key: str, knowledge_data: Any) -> bool:
        """Save knowledge to agent's knowledge base"""
        context = self.get_agent_context(session_id, agent_id)
        if context:
            knowledge_base = context.get('knowledge_base', {})
            knowledge_base[knowledge_key] = {
                'data': knowledge_data,
                'saved_at': datetime.utcnow().isoformat(),
                'access_count': knowledge_base.get(knowledge_key, {}).get('access_count', 0)
            }
            return self.update_agent_context(session_id, agent_id, {'knowledge_base': knowledge_base})
        return False
    
    def get_knowledge(self, session_id: str, agent_id: str, knowledge_key: str) -> Any:
        """Get knowledge from agent's knowledge base"""
        context = self.get_agent_context(session_id, agent_id)
        if context:
            knowledge_base = context.get('knowledge_base', {})
            if knowledge_key in knowledge_base:
                # Increment access count
                knowledge_base[knowledge_key]['access_count'] += 1
                knowledge_base[knowledge_key]['last_accessed'] = datetime.utcnow().isoformat()
                self.update_agent_context(session_id, agent_id, {'knowledge_base': knowledge_base})
                return knowledge_base[knowledge_key]['data']
        return None
    
    def set_agent_state(self, session_id: str, agent_id: str, state: str, state_data: Optional[Dict] = None) -> bool:
        """Set agent's current state"""
        state_update = {
            'state': state,
            'state_data': state_data or {},
            'state_updated_at': datetime.utcnow().isoformat()
        }
        return self.update_agent_context(session_id, agent_id, state_update)
    
    def get_agent_state(self, session_id: str, agent_id: str) -> Dict:
        """Get agent's current state"""
        context = self.get_agent_context(session_id, agent_id)
        if context:
            return {
                'state': context.get('state', 'unknown'),
                'state_data': context.get('state_data', {}),
                'state_updated_at': context.get('state_updated_at')
            }
        return {'state': 'unknown', 'state_data': {}, 'state_updated_at': None}
    
    def learn_from_interaction(self, session_id: str, agent_id: str, interaction_data: Dict) -> bool:
        """Store learning data from interactions"""
        context = self.get_agent_context(session_id, agent_id)
        if context:
            learning_data = context.get('learning_data', {})
            interaction_id = str(uuid.uuid4())
            learning_data[interaction_id] = {
                **interaction_data,
                'learned_at': datetime.utcnow().isoformat()
            }
            
            # Keep only recent learning data (last 100 interactions)
            if len(learning_data) > 100:
                oldest_keys = sorted(learning_data.keys(), 
                                   key=lambda k: learning_data[k]['learned_at'])[:len(learning_data)-100]
                for key in oldest_keys:
                    del learning_data[key]
                    
            return self.update_agent_context(session_id, agent_id, {'learning_data': learning_data})
        return False
    
    def get_learning_patterns(self, session_id: str, agent_id: str) -> List[Dict]:
        """Get patterns from agent's learning data"""
        context = self.get_agent_context(session_id, agent_id)
        if context:
            learning_data = context.get('learning_data', {})
            return list(learning_data.values())
        return []
    
    def transfer_context_to_session(self, from_session_id: str, to_session_id: str, agent_id: str) -> bool:
        """Transfer agent context from one session to another"""
        try:
            source_context = self.get_agent_context(from_session_id, agent_id)
            if source_context:
                # Create context in new session
                with self.db.get_session() as session:
                    new_context = AgentContext(
                        session_id=to_session_id,
                        agent_id=agent_id,
                        agent_type=source_context.get('agent_type', 'unknown'),
                        context_type='transferred',
                        context_data=json.dumps(source_context),
                        version=1,
                        active=True
                    )
                    session.add(new_context)
                    session.commit()
                
                # Update cache
                cache_key = f"{to_session_id}:{agent_id}"
                self._context_cache[cache_key] = source_context
                
                logger.info(f"Transferred context for agent {agent_id} from {from_session_id} to {to_session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error transferring agent context: {e}")
            
        return False
    
    def get_context_history(self, session_id: str, agent_id: str) -> List[Dict]:
        """Get version history of agent context"""
        try:
            with self.db.get_session() as session:
                contexts = session.query(AgentContext).filter(
                    AgentContext.session_id == session_id,
                    AgentContext.agent_id == agent_id
                ).order_by(AgentContext.version.asc()).all()
                
                return [{
                    'version': ctx.version,
                    'context_type': ctx.context_type,
                    'active': ctx.active,
                    'created_at': ctx.created_at.isoformat(),
                    'updated_at': ctx.updated_at.isoformat(),
                    'context_summary': self._summarize_context(json.loads(ctx.context_data))
                } for ctx in contexts]
                
        except Exception as e:
            logger.error(f"Error getting context history: {e}")
            return []
    
    def cleanup_expired_contexts(self, hours_to_keep: int = 24) -> int:
        """Clean up expired agent contexts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_to_keep)
            
            with self.db.get_session() as session:
                expired_contexts = session.query(AgentContext).filter(
                    AgentContext.expires_at < cutoff_time,
                    AgentContext.active == True
                ).all()
                
                count = len(expired_contexts)
                
                for context in expired_contexts:
                    context.active = False
                    
                session.commit()
                
            # Clear cache for expired contexts
            expired_keys = []
            for cache_key in self._context_cache:
                session_id, agent_id = cache_key.split(':', 1)
                if any(ctx.session_id == session_id and ctx.agent_id == agent_id 
                      for ctx in expired_contexts):
                    expired_keys.append(cache_key)
                    
            for key in expired_keys:
                del self._context_cache[key]
                
            logger.info(f"Cleaned up {count} expired agent contexts")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired contexts: {e}")
            return 0
    
    def _merge_context(self, current: Dict, updates: Dict) -> Dict:
        """Merge context updates into current context"""
        merged = current.copy()
        
        for key, value in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = merged[key] + value
            else:
                merged[key] = value
                
        merged['last_updated'] = datetime.utcnow().isoformat()
        return merged
    
    def _summarize_context(self, context_data: Dict) -> Dict:
        """Create a summary of context data for history"""
        return {
            'state': context_data.get('state', 'unknown'),
            'goals_count': len(context_data.get('goals', [])),
            'active_goals': len([g for g in context_data.get('goals', []) if g.get('status') == 'active']),
            'knowledge_items': len(context_data.get('knowledge_base', {})),
            'learning_entries': len(context_data.get('learning_data', {})),
            'capabilities': context_data.get('capabilities', [])
        }