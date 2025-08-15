"""
Agent Initialization with Persistent Memory
Handles agent startup, memory loading, and context restoration
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
from sqlalchemy.orm import Session

from ..database import get_db, ConversationSession, User
from .conversation_memory import ConversationMemoryManager
from .agent_context import AgentContextManager
from .semantic_memory import SemanticMemoryManager
from .memory_retrieval import MemoryRetrievalSystem

logger = logging.getLogger(__name__)

class AgentMemoryInitializer:
    """Manages agent initialization with persistent memory context"""
    
    def __init__(self):
        self.db = get_db()
        self.conversation_memory = ConversationMemoryManager()
        self.agent_context = AgentContextManager()
        self.semantic_memory = SemanticMemoryManager()
        self.memory_retrieval = MemoryRetrievalSystem()
        
        # Registry of initialized agents
        self.initialized_agents = {}
        
    async def initialize_agent_with_memory(self, agent_id: str, agent_type: str, 
                                         user_id: int, 
                                         restore_previous_session: bool = True) -> Dict[str, Any]:
        """Initialize agent with full memory context"""
        try:
            initialization_start = datetime.utcnow()
            
            logger.info(f"Initializing agent {agent_id} with memory for user {user_id}")
            
            # Step 1: Create or restore session
            session_id = await self._get_or_create_session(
                user_id, agent_type, restore_previous_session
            )
            
            # Step 2: Initialize agent context
            context_initialized = await self._initialize_agent_context(
                session_id, agent_id, agent_type, restore_previous_session
            )
            
            # Step 3: Load relevant memories
            memory_context = await self._load_relevant_memories(
                user_id, session_id, agent_id, agent_type
            )
            
            # Step 4: Set up agent goals and state
            goals_state = await self._setup_agent_goals_and_state(
                session_id, agent_id, agent_type, memory_context
            )
            
            # Step 5: Generate initialization summary
            initialization_summary = await self._generate_initialization_summary(
                agent_id, agent_type, session_id, memory_context, goals_state
            )
            
            # Register the initialized agent
            self.initialized_agents[agent_id] = {
                'agent_type': agent_type,
                'user_id': user_id,
                'session_id': session_id,
                'initialized_at': initialization_start.isoformat(),
                'memory_context': memory_context,
                'initialization_summary': initialization_summary
            }
            
            logger.info(f"Agent {agent_id} initialized successfully in {(datetime.utcnow() - initialization_start).total_seconds():.2f}s")
            
            return {
                'success': True,
                'agent_id': agent_id,
                'session_id': session_id,
                'memory_context': memory_context,
                'initialization_summary': initialization_summary,
                'context_initialized': context_initialized,
                'goals_and_state': goals_state,
                'initialization_time_seconds': (datetime.utcnow() - initialization_start).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error initializing agent {agent_id}: {e}")
            return {
                'success': False,
                'agent_id': agent_id,
                'error': str(e)
            }
    
    async def _get_or_create_session(self, user_id: int, agent_type: str, 
                                   restore_previous: bool) -> str:
        """Get existing session or create new one"""
        if restore_previous:
            # Try to find the most recent active session for this agent type
            with self.db.get_session() as session:
                recent_session = session.query(ConversationSession).filter(
                    ConversationSession.user_id == user_id,
                    ConversationSession.agent_type == agent_type,
                    ConversationSession.active == True
                ).order_by(ConversationSession.updated_at.desc()).first()
                
                if recent_session:
                    logger.info(f"Restoring previous session {recent_session.session_id}")
                    return recent_session.session_id
        
        # Create new session
        session_id = self.conversation_memory.create_session(
            user_id=user_id,
            agent_type=agent_type,
            initial_context={'restored': False, 'initialization_type': 'new'}
        )
        
        logger.info(f"Created new session {session_id}")
        return session_id
    
    async def _initialize_agent_context(self, session_id: str, agent_id: str, 
                                      agent_type: str, restore_previous: bool) -> bool:
        """Initialize or restore agent context"""
        try:
            # Check if context already exists
            existing_context = self.agent_context.get_agent_context(session_id, agent_id)
            
            if existing_context and restore_previous:
                logger.info(f"Restored existing context for agent {agent_id}")
                return True
            
            # Create new context
            initial_context = {
                'working_memory': {
                    'initialization_time': datetime.utcnow().isoformat(),
                    'session_type': 'restored' if restore_previous else 'new'
                },
                'goals': [],
                'knowledge_base': {},
                'state': 'initializing',
                'capabilities': self._get_agent_capabilities(agent_type),
                'preferences': {},
                'learning_data': {}
            }
            
            success = self.agent_context.create_agent_context(
                session_id=session_id,
                agent_id=agent_id,
                agent_type=agent_type,
                initial_context=initial_context
            )
            
            if success:
                logger.info(f"Created new context for agent {agent_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing agent context: {e}")
            return False
    
    def _get_agent_capabilities(self, agent_type: str) -> List[str]:
        """Get capabilities for agent type"""
        base_capabilities = ['memory_enabled', 'context_aware', 'learning_capable']
        
        type_specific_capabilities = {
            'a2a_coordinator': ['message_routing', 'workflow_orchestration', 'agent_management'],
            'historical_loader': ['data_retrieval', 'market_data', 'historical_analysis'],
            'database_agent': ['data_storage', 'query_processing', 'data_validation'],
            'data_management': ['data_processing', 'analytics', 'reporting'],
            'trading_agent': ['trade_execution', 'risk_management', 'portfolio_analysis'],
            'blockchain_agent': ['blockchain_interaction', 'smart_contracts', 'transaction_processing']
        }
        
        specific_caps = type_specific_capabilities.get(agent_type, [])
        return base_capabilities + specific_caps
    
    async def _load_relevant_memories(self, user_id: int, session_id: str, 
                                    agent_id: str, agent_type: str) -> Dict[str, Any]:
        """Load relevant memories for agent initialization"""
        try:
            # Get recent conversation history
            conversation_history = self.conversation_memory.get_conversation_history(
                session_id=session_id,
                limit=10
            )
            
            # Get user preferences
            user_preferences = self.conversation_memory.get_user_preferences(user_id)
            
            # Get agent-specific memories
            agent_memories = self.semantic_memory.search_similar_memories(
                user_id=user_id,
                query=f"{agent_type} agent interactions",
                limit=10
            )
            
            # Get procedural memories for this agent type
            procedural_memories = self.semantic_memory.search_similar_memories(
                user_id=user_id,
                query=f"{agent_type} procedures workflows",
                memory_type="procedural",
                limit=10
            )
            
            # Get recent learning patterns
            learning_patterns = []
            if session_id:
                learning_patterns = self.agent_context.get_learning_patterns(session_id, agent_id)
            
            memory_context = {
                'conversation_history': conversation_history,
                'user_preferences': user_preferences,
                'agent_specific_memories': agent_memories,
                'procedural_memories': procedural_memories,
                'learning_patterns': learning_patterns[-20:],  # Last 20 patterns
                'memory_loaded_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Loaded {len(agent_memories)} agent memories and {len(procedural_memories)} procedural memories")
            return memory_context
            
        except Exception as e:
            logger.error(f"Error loading relevant memories: {e}")
            return {}
    
    async def _setup_agent_goals_and_state(self, session_id: str, agent_id: str, 
                                         agent_type: str, memory_context: Dict) -> Dict[str, Any]:
        """Set up agent goals and initial state based on memory"""
        try:
            # Set agent state to active
            self.agent_context.set_agent_state(
                session_id=session_id,
                agent_id=agent_id,
                state='active',
                state_data={
                    'initialized_at': datetime.utcnow().isoformat(),
                    'memory_context_loaded': bool(memory_context),
                    'previous_interactions': len(memory_context.get('conversation_history', []))
                }
            )
            
            # Add type-specific goals
            goals_added = []
            type_goals = {
                'a2a_coordinator': [
                    {'description': 'Facilitate efficient A2A communication', 'priority': 'high'},
                    {'description': 'Maintain message routing accuracy', 'priority': 'high'},
                    {'description': 'Monitor agent performance', 'priority': 'medium'}
                ],
                'historical_loader': [
                    {'description': 'Provide accurate historical market data', 'priority': 'high'},
                    {'description': 'Optimize data loading performance', 'priority': 'medium'}
                ],
                'database_agent': [
                    {'description': 'Ensure data integrity and consistency', 'priority': 'high'},
                    {'description': 'Optimize query performance', 'priority': 'medium'}
                ],
                'trading_agent': [
                    {'description': 'Execute profitable trades', 'priority': 'high'},
                    {'description': 'Manage risk effectively', 'priority': 'high'},
                    {'description': 'Learn from trading outcomes', 'priority': 'medium'}
                ]
            }
            
            agent_goals = type_goals.get(agent_type, [
                {'description': f'Perform {agent_type} duties effectively', 'priority': 'high'}
            ])
            
            for goal in agent_goals:
                success = self.agent_context.add_goal(
                    session_id=session_id,
                    agent_id=agent_id,
                    goal=goal
                )
                if success:
                    goals_added.append(goal['description'])
            
            return {
                'state_set': 'active',
                'goals_added': goals_added,
                'goals_count': len(goals_added)
            }
            
        except Exception as e:
            logger.error(f"Error setting up agent goals and state: {e}")
            return {'error': str(e)}
    
    async def _generate_initialization_summary(self, agent_id: str, agent_type: str, 
                                             session_id: str, memory_context: Dict,
                                             goals_state: Dict) -> str:
        """Generate a summary of the initialization process"""
        try:
            conversation_count = len(memory_context.get('conversation_history', []))
            memory_count = len(memory_context.get('agent_specific_memories', []))
            procedural_count = len(memory_context.get('procedural_memories', []))
            learning_count = len(memory_context.get('learning_patterns', []))
            goals_count = goals_state.get('goals_count', 0)
            
            summary = f"""Agent {agent_id} ({agent_type}) initialized successfully:
            
Memory Context Loaded:
- {conversation_count} recent conversation messages
- {memory_count} agent-specific memories  
- {procedural_count} procedural memories
- {learning_count} learning patterns from previous interactions

Agent Configuration:
- {goals_count} goals established
- State: {goals_state.get('state_set', 'unknown')}
- Session: {session_id}

Capabilities: {', '.join(self._get_agent_capabilities(agent_type))}

The agent is now ready to operate with full memory context and can leverage past experiences for improved performance."""
            
            # Store the initialization in memory
            self.semantic_memory.store_memory(
                user_id=1,  # System user
                memory_type="episodic",
                content=f"Agent {agent_id} initialized with {memory_count} memories and {goals_count} goals",
                context=f"Agent initialization for {agent_type}",
                confidence=0.9
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating initialization summary: {e}")
            return f"Agent {agent_id} initialized with limited summary due to error: {str(e)}"
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an initialized agent"""
        return self.initialized_agents.get(agent_id)
    
    def list_initialized_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all initialized agents"""
        return self.initialized_agents.copy()
    
    async def shutdown_agent(self, agent_id: str, save_context: bool = True) -> bool:
        """Shutdown agent and optionally save context"""
        try:
            if agent_id not in self.initialized_agents:
                logger.warning(f"Agent {agent_id} not found in initialized agents")
                return False
            
            agent_info = self.initialized_agents[agent_id]
            
            if save_context:
                # Update agent state to shutdown
                self.agent_context.set_agent_state(
                    session_id=agent_info['session_id'],
                    agent_id=agent_id,
                    state='shutdown',
                    state_data={
                        'shutdown_at': datetime.utcnow().isoformat(),
                        'context_saved': True
                    }
                )
                
                # Store shutdown summary
                self.semantic_memory.store_memory(
                    user_id=agent_info['user_id'],
                    memory_type="episodic", 
                    content=f"Agent {agent_id} shutdown gracefully with context saved",
                    context=f"Agent shutdown for {agent_info['agent_type']}",
                    confidence=0.8
                )
            
            # Remove from registry
            del self.initialized_agents[agent_id]
            
            logger.info(f"Agent {agent_id} shutdown {'with' if save_context else 'without'} context saving")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down agent {agent_id}: {e}")
            return False
    
    async def transfer_agent_context(self, agent_id: str, new_session_id: str) -> bool:
        """Transfer agent to a new session"""
        try:
            if agent_id not in self.initialized_agents:
                return False
            
            agent_info = self.initialized_agents[agent_id]
            old_session_id = agent_info['session_id']
            
            # Transfer context
            success = self.agent_context.transfer_context_to_session(
                from_session_id=old_session_id,
                to_session_id=new_session_id,
                agent_id=agent_id
            )
            
            if success:
                # Update registry
                agent_info['session_id'] = new_session_id
                agent_info['transferred_at'] = datetime.utcnow().isoformat()
                
                logger.info(f"Transferred agent {agent_id} from {old_session_id} to {new_session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error transferring agent context: {e}")
            return False

# Global instance - initialized lazily to avoid circular imports
agent_memory_initializer = None

def get_agent_memory_initializer():
    """Get or create global agent memory initializer"""
    global agent_memory_initializer
    if agent_memory_initializer is None:
        agent_memory_initializer = AgentMemoryInitializer()
    return agent_memory_initializer