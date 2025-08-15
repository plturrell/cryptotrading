"""
Autonomous Memory Triggers
Handles automatic population of conversation history and memory from various sources
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .conversation_memory import ConversationMemoryManager
from .agent_context import AgentContextManager
from .semantic_memory import SemanticMemoryManager

logger = logging.getLogger(__name__)

@dataclass
class MemoryTrigger:
    """Configuration for memory trigger"""
    trigger_id: str
    source: str  # 'market_data', 'trade_execution', 'user_interaction', 'agent_communication'
    condition: Dict[str, Any]  # Trigger conditions
    action: str  # 'store_conversation', 'update_context', 'create_memory'
    importance_score: float = 0.5
    enabled: bool = True

class AutonomousMemorySystem:
    """Manages autonomous memory population and triggers"""
    
    def __init__(self):
        self.conversation_memory = ConversationMemoryManager()
        self.agent_context = AgentContextManager()
        self.semantic_memory = SemanticMemoryManager()
        
        # Active triggers
        self.triggers: Dict[str, MemoryTrigger] = {}
        self.running = False
        self.monitor_task = None
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'market_data_update': [],
            'trade_executed': [],
            'user_message': [],
            'agent_response': [],
            'price_alert': [],
            'technical_signal': [],
            'error_occurred': [],
            'workflow_completed': []
        }
        
        # Initialize default triggers
        self._setup_default_triggers()
    
    def _setup_default_triggers(self):
        """Setup default memory triggers for common trading scenarios"""
        
        # 1. Trade Execution Trigger
        self.add_trigger(MemoryTrigger(
            trigger_id="trade_execution",
            source="trade_execution",
            condition={"event_type": "trade_completed"},
            action="store_conversation",
            importance_score=0.9
        ))
        
        # 2. Significant Price Movement Trigger
        self.add_trigger(MemoryTrigger(
            trigger_id="price_movement",
            source="market_data",
            condition={"price_change_percent": ">5"},
            action="create_memory",
            importance_score=0.8
        ))
        
        # 3. AI Analysis Trigger
        self.add_trigger(MemoryTrigger(
            trigger_id="ai_analysis",
            source="ai_analysis",
            condition={"confidence": ">0.8"},
            action="store_conversation",
            importance_score=0.85
        ))
        
        # 4. User Query Trigger
        self.add_trigger(MemoryTrigger(
            trigger_id="user_query",
            source="user_interaction",
            condition={"type": "query"},
            action="store_conversation",
            importance_score=0.7
        ))
        
        # 5. Agent Communication Trigger
        self.add_trigger(MemoryTrigger(
            trigger_id="agent_communication",
            source="agent_communication",
            condition={"message_type": "important"},
            action="update_context",
            importance_score=0.6
        ))
    
    def add_trigger(self, trigger: MemoryTrigger):
        """Add a memory trigger"""
        self.triggers[trigger.trigger_id] = trigger
        logger.info(f"Added memory trigger: {trigger.trigger_id}")
    
    def remove_trigger(self, trigger_id: str):
        """Remove a memory trigger"""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            logger.info(f"Removed memory trigger: {trigger_id}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event that may trigger memory actions"""
        try:
            # Process through triggers
            await self._process_triggers(event_type, event_data)
            
            # Call registered handlers
            for handler in self.event_handlers.get(event_type, []):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing event {event_type}: {e}")
    
    async def _process_triggers(self, event_type: str, event_data: Dict[str, Any]):
        """Process event through all matching triggers"""
        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue
                
            # Check if trigger matches event
            if await self._trigger_matches(trigger, event_type, event_data):
                await self._execute_trigger_action(trigger, event_data)
    
    async def _trigger_matches(self, trigger: MemoryTrigger, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Check if trigger condition matches the event"""
        try:
            # Simple condition matching (can be extended)
            for condition_key, condition_value in trigger.condition.items():
                if condition_key not in event_data:
                    return False
                
                event_value = event_data[condition_key]
                
                # Handle different condition types
                if isinstance(condition_value, str) and condition_value.startswith('>'):
                    threshold = float(condition_value[1:])
                    if not (isinstance(event_value, (int, float)) and event_value > threshold):
                        return False
                elif isinstance(condition_value, str) and condition_value.startswith('<'):
                    threshold = float(condition_value[1:])
                    if not (isinstance(event_value, (int, float)) and event_value < threshold):
                        return False
                elif event_value != condition_value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trigger condition: {e}")
            return False
    
    async def _execute_trigger_action(self, trigger: MemoryTrigger, event_data: Dict[str, Any]):
        """Execute the trigger action"""
        try:
            if trigger.action == "store_conversation":
                await self._store_conversation_from_event(trigger, event_data)
            elif trigger.action == "update_context":
                await self._update_context_from_event(trigger, event_data)
            elif trigger.action == "create_memory":
                await self._create_memory_from_event(trigger, event_data)
            else:
                logger.warning(f"Unknown trigger action: {trigger.action}")
                
        except Exception as e:
            logger.error(f"Error executing trigger action {trigger.action}: {e}")
    
    async def _store_conversation_from_event(self, trigger: MemoryTrigger, event_data: Dict[str, Any]):
        """Store conversation message from event"""
        user_id = event_data.get('user_id', 1)
        agent_type = event_data.get('agent_type', trigger.source)
        
        # Create or get session
        session_id = await self._get_or_create_session(user_id, agent_type)
        
        # Generate conversation content based on event
        if trigger.source == "trade_execution":
            content = self._generate_trade_conversation(event_data)
        elif trigger.source == "market_data":
            content = self._generate_market_conversation(event_data)
        elif trigger.source == "ai_analysis":
            content = self._generate_analysis_conversation(event_data)
        else:
            content = f"Event: {json.dumps(event_data, default=str)}"
        
        # Store in conversation memory
        if content:
            self.conversation_memory.add_message(
                session_id=session_id,
                role="assistant",
                content=content,
                metadata={"trigger_id": trigger.trigger_id, "source": trigger.source},
                importance_score=trigger.importance_score
            )
    
    async def _update_context_from_event(self, trigger: MemoryTrigger, event_data: Dict[str, Any]):
        """Update agent context from event"""
        agent_id = event_data.get('agent_id', f"{trigger.source}_agent")
        session_id = event_data.get('session_id')
        
        if not session_id:
            user_id = event_data.get('user_id', 1)
            session_id = await self._get_or_create_session(user_id, trigger.source)
        
        # Update context based on event
        context_update = {
            'working_memory': {
                f'last_{trigger.source}_event': event_data,
                f'last_{trigger.source}_time': datetime.utcnow().isoformat()
            }
        }
        
        self.agent_context.update_agent_context(
            session_id=session_id,
            agent_id=agent_id,
            context_updates=context_update
        )
    
    async def _create_memory_from_event(self, trigger: MemoryTrigger, event_data: Dict[str, Any]):
        """Create semantic memory from event"""
        user_id = event_data.get('user_id', 1)
        
        # Generate memory content
        if trigger.source == "market_data":
            content = self._generate_market_memory(event_data)
        elif trigger.source == "trade_execution":
            content = self._generate_trade_memory(event_data)
        else:
            content = f"Event recorded: {trigger.source} - {json.dumps(event_data, default=str)}"
        
        # Extract keywords and symbols
        keywords = self._extract_keywords_from_event(event_data)
        symbols = self._extract_symbols_from_event(event_data)
        
        # Store in semantic memory
        self.semantic_memory.store_memory(
            user_id=user_id,
            memory_type="episodic",
            content=content,
            context=f"Autonomous trigger: {trigger.trigger_id}",
            keywords=keywords,
            associated_symbols=symbols,
            confidence=trigger.importance_score
        )
    
    def _generate_trade_conversation(self, event_data: Dict[str, Any]) -> str:
        """Generate realistic trading conversation from event"""
        symbol = event_data.get('symbol', 'UNKNOWN')
        action = event_data.get('action', 'trade')
        quantity = event_data.get('quantity', 0)
        price = event_data.get('price', 0)
        pnl = event_data.get('pnl', 0)
        
        if action == 'buy':
            return f"Executed buy order for {quantity} {symbol} at ${price:.2f}. Position opened successfully."
        elif action == 'sell':
            pnl_text = f"with P&L of ${pnl:.2f}" if pnl != 0 else ""
            return f"Executed sell order for {quantity} {symbol} at ${price:.2f} {pnl_text}. Position closed."
        else:
            return f"Trade executed: {action} {quantity} {symbol} at ${price:.2f}"
    
    def _generate_market_conversation(self, event_data: Dict[str, Any]) -> str:
        """Generate market analysis conversation from event"""
        symbol = event_data.get('symbol', 'UNKNOWN')
        price = event_data.get('price', 0)
        change = event_data.get('price_change_percent', 0)
        
        if change > 5:
            return f"{symbol} showing strong bullish momentum, up {change:.1f}% to ${price:.2f}. Technical indicators suggest continued upward pressure."
        elif change < -5:
            return f"{symbol} experiencing significant bearish pressure, down {change:.1f}% to ${price:.2f}. May present buying opportunity if support holds."
        else:
            return f"{symbol} price update: ${price:.2f}, change {change:+.1f}%. Market showing consolidation pattern."
    
    def _generate_analysis_conversation(self, event_data: Dict[str, Any]) -> str:
        """Generate AI analysis conversation from event"""
        symbol = event_data.get('symbol', 'UNKNOWN')
        signal = event_data.get('signal', 'HOLD')
        confidence = event_data.get('confidence', 0.5)
        reasoning = event_data.get('reasoning', 'Technical analysis')
        
        return f"AI Analysis for {symbol}: {signal} signal with {confidence:.0%} confidence. {reasoning}. Recommend monitoring for entry/exit opportunities."
    
    def _generate_market_memory(self, event_data: Dict[str, Any]) -> str:
        """Generate market memory content"""
        symbol = event_data.get('symbol', 'UNKNOWN')
        price = event_data.get('price', 0)
        volume = event_data.get('volume', 0)
        
        return f"{symbol} market data: Price ${price:.2f}, Volume {volume:,.0f}. Timestamp: {datetime.utcnow().isoformat()}"
    
    def _generate_trade_memory(self, event_data: Dict[str, Any]) -> str:
        """Generate trade memory content"""
        symbol = event_data.get('symbol', 'UNKNOWN')
        action = event_data.get('action', 'trade')
        outcome = event_data.get('outcome', 'completed')
        
        return f"Trade outcome: {action} {symbol} - {outcome}. Strategy performance recorded for future optimization."
    
    def _extract_keywords_from_event(self, event_data: Dict[str, Any]) -> List[str]:
        """Extract keywords from event data"""
        keywords = []
        
        if 'symbol' in event_data:
            keywords.append(event_data['symbol'].lower())
        if 'action' in event_data:
            keywords.append(event_data['action'].lower())
        if 'signal' in event_data:
            keywords.append(event_data['signal'].lower())
        
        # Add contextual keywords
        keywords.extend(['trading', 'market', 'analysis'])
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_symbols_from_event(self, event_data: Dict[str, Any]) -> List[str]:
        """Extract trading symbols from event data"""
        symbols = []
        
        if 'symbol' in event_data:
            symbols.append(event_data['symbol'].upper())
        if 'symbols' in event_data:
            symbols.extend([s.upper() for s in event_data['symbols']])
        
        return list(set(symbols))  # Remove duplicates
    
    async def _get_or_create_session(self, user_id: int, agent_type: str) -> str:
        """Get existing session or create new one"""
        # Try to find existing active session
        from ..database import get_db, ConversationSession
        
        with get_db().get_session() as session:
            existing_session = session.query(ConversationSession).filter(
                ConversationSession.user_id == user_id,
                ConversationSession.agent_type == agent_type,
                ConversationSession.active == True
            ).order_by(ConversationSession.updated_at.desc()).first()
            
            if existing_session:
                return existing_session.session_id
        
        # Create new session
        return self.conversation_memory.create_session(
            user_id=user_id,
            agent_type=agent_type,
            initial_context={'created_by': 'autonomous_trigger'}
        )
    
    async def start_monitoring(self):
        """Start autonomous memory monitoring"""
        if self.running:
            return
        
        self.running = True
        logger.info("Started autonomous memory monitoring")
        
        # Start monitoring task (can be extended to watch external systems)
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop autonomous memory monitoring"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped autonomous memory monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop (placeholder for future external system integration)"""
        while self.running:
            try:
                # This is where you would integrate with external systems:
                # - Market data feeds
                # - Trading platform APIs
                # - Database change events
                # - File system watchers
                # - Web hooks
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5)
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get statistics about memory triggers"""
        enabled_triggers = [t for t in self.triggers.values() if t.enabled]
        
        return {
            'total_triggers': len(self.triggers),
            'enabled_triggers': len(enabled_triggers),
            'trigger_sources': list(set(t.source for t in enabled_triggers)),
            'running': self.running,
            'event_handlers': {k: len(v) for k, v in self.event_handlers.items()}
        }

# Global instance
autonomous_memory = AutonomousMemorySystem()