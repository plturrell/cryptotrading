"""
MCP Event Streaming System
Lightweight event streaming for real-time updates in serverless environments
"""
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .auth import AuthContext
from .metrics import mcp_metrics

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for MCP streaming"""

    TOOL_EXECUTION = "tool_execution"
    RESOURCE_UPDATE = "resource_update"
    MARKET_DATA = "market_data"
    PORTFOLIO_UPDATE = "portfolio_update"
    TRADE_SIGNAL = "trade_signal"
    SYSTEM_STATUS = "system_status"
    USER_ACTION = "user_action"
    ERROR = "error"


@dataclass
class MCPEvent:
    """MCP event structure"""

    event_type: EventType
    data: Dict[str, Any]
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source": self.source,
        }

    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPEvent":
        """Create event from dictionary"""
        return cls(
            event_type=EventType(data["event_type"]),
            data=data["data"],
            timestamp=data["timestamp"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            source=data.get("source"),
        )


class EventFilter:
    """Filter for event subscriptions"""

    def __init__(
        self, event_types: List[EventType] = None, user_id: str = None, source: str = None
    ):
        self.event_types = set(event_types) if event_types else set()
        self.user_id = user_id
        self.source = source

    def matches(self, event: MCPEvent) -> bool:
        """Check if event matches filter"""
        if self.event_types and event.event_type not in self.event_types:
            return False

        if self.user_id and event.user_id != self.user_id:
            return False

        if self.source and event.source != self.source:
            return False

        return True


class EventSubscription:
    """Event subscription for streaming"""

    def __init__(
        self, subscription_id: str, filter: EventFilter, callback: Callable[[MCPEvent], None]
    ):
        self.subscription_id = subscription_id
        self.filter = filter
        self.callback = callback
        self.created_at = time.time()
        self.last_event_at = None
        self.event_count = 0
        self.active = True

    async def deliver_event(self, event: MCPEvent):
        """Deliver event to subscription"""
        if not self.active or not self.filter.matches(event):
            return

        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(event)
            else:
                self.callback(event)

            self.last_event_at = time.time()
            self.event_count += 1

        except Exception as e:
            logger.error(f"Error delivering event to subscription {self.subscription_id}: {e}")


class EventBuffer:
    """Lightweight event buffer for serverless environments"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.events: List[MCPEvent] = []
        self._lock = asyncio.Lock()

    async def add_event(self, event: MCPEvent):
        """Add event to buffer"""
        async with self._lock:
            # Remove expired events
            current_time = time.time()
            self.events = [e for e in self.events if current_time - e.timestamp < self.ttl_seconds]

            # Add new event
            self.events.append(event)

            # Maintain max size
            if len(self.events) > self.max_size:
                self.events = self.events[-self.max_size :]

    async def get_events(
        self, filter: EventFilter = None, since: float = None, limit: int = 100
    ) -> List[MCPEvent]:
        """Get events from buffer"""
        async with self._lock:
            events = self.events.copy()

        # Apply time filter
        if since:
            events = [e for e in events if e.timestamp > since]

        # Apply event filter
        if filter:
            events = [e for e in events if filter.matches(e)]

        # Apply limit
        if limit:
            events = events[-limit:]

        return events

    async def clear_expired(self):
        """Clear expired events"""
        current_time = time.time()
        async with self._lock:
            self.events = [e for e in self.events if current_time - e.timestamp < self.ttl_seconds]


class EventStreamer:
    """Lightweight event streaming for serverless MCP"""

    def __init__(self):
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_buffer = EventBuffer()
        self._subscription_counter = 0
        self._lock = asyncio.Lock()

    async def publish_event(self, event: MCPEvent):
        """Publish event to all subscribers"""
        # Add to buffer
        await self.event_buffer.add_event(event)

        # Deliver to active subscriptions
        async with self._lock:
            subscriptions = list(self.subscriptions.values())

        # Deliver events concurrently
        delivery_tasks = [sub.deliver_event(event) for sub in subscriptions if sub.active]

        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

        # Record metrics
        mcp_metrics.collector.counter(
            "mcp.events.published", tags={"event_type": event.event_type.value}
        )

    async def subscribe(self, filter: EventFilter, callback: Callable[[MCPEvent], None]) -> str:
        """Subscribe to events"""
        async with self._lock:
            self._subscription_counter += 1
            subscription_id = f"sub_{self._subscription_counter}_{int(time.time())}"

            subscription = EventSubscription(subscription_id, filter, callback)
            self.subscriptions[subscription_id] = subscription

        logger.info(f"Created event subscription: {subscription_id}")

        mcp_metrics.collector.counter("mcp.events.subscriptions_created")

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        async with self._lock:
            if subscription_id in self.subscriptions:
                self.subscriptions[subscription_id].active = False
                del self.subscriptions[subscription_id]
                logger.info(f"Removed event subscription: {subscription_id}")
                mcp_metrics.collector.counter("mcp.events.subscriptions_removed")
                return True

        return False

    async def get_subscription_stats(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Get subscription statistics"""
        async with self._lock:
            if subscription_id not in self.subscriptions:
                return None

            sub = self.subscriptions[subscription_id]
            return {
                "subscription_id": sub.subscription_id,
                "created_at": sub.created_at,
                "last_event_at": sub.last_event_at,
                "event_count": sub.event_count,
                "active": sub.active,
                "filter": {
                    "event_types": [et.value for et in sub.filter.event_types],
                    "user_id": sub.filter.user_id,
                    "source": sub.filter.source,
                },
            }

    async def get_events(
        self, filter: EventFilter = None, since: float = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get events from buffer"""
        events = await self.event_buffer.get_events(filter, since, limit)
        return [event.to_dict() for event in events]

    async def cleanup_subscriptions(self, max_age_seconds: int = 3600):
        """Clean up old inactive subscriptions"""
        current_time = time.time()

        async with self._lock:
            expired_subs = [
                sub_id
                for sub_id, sub in self.subscriptions.items()
                if not sub.active or (current_time - sub.created_at) > max_age_seconds
            ]

            for sub_id in expired_subs:
                del self.subscriptions[sub_id]

        if expired_subs:
            logger.info(f"Cleaned up {len(expired_subs)} expired subscriptions")

    async def get_stats(self) -> Dict[str, Any]:
        """Get event streaming statistics"""
        async with self._lock:
            active_subs = sum(1 for sub in self.subscriptions.values() if sub.active)
            total_events = sum(sub.event_count for sub in self.subscriptions.values())

        return {
            "active_subscriptions": active_subs,
            "total_subscriptions": len(self.subscriptions),
            "total_events_delivered": total_events,
            "buffer_size": len(self.event_buffer.events),
        }


class MCPEventPublisher:
    """Helper class for publishing MCP events"""

    def __init__(self, streamer: EventStreamer, auth_context: AuthContext = None):
        self.streamer = streamer
        self.auth_context = auth_context

    async def publish_tool_execution(
        self, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any], success: bool
    ):
        """Publish tool execution event"""
        event = MCPEvent(
            event_type=EventType.TOOL_EXECUTION,
            data={
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "success": success,
                "execution_time": time.time(),
            },
            timestamp=time.time(),
            user_id=self.auth_context.user_id if self.auth_context else None,
            source="mcp_server",
        )

        await self.streamer.publish_event(event)

    async def publish_market_data(self, symbol: str, price: float, volume: float):
        """Publish market data update"""
        event = MCPEvent(
            event_type=EventType.MARKET_DATA,
            data={
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": datetime.utcnow().isoformat(),
            },
            timestamp=time.time(),
            source="market_data",
        )

        await self.streamer.publish_event(event)

    async def publish_portfolio_update(self, portfolio_id: str, changes: Dict[str, Any]):
        """Publish portfolio update"""
        event = MCPEvent(
            event_type=EventType.PORTFOLIO_UPDATE,
            data={
                "portfolio_id": portfolio_id,
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat(),
            },
            timestamp=time.time(),
            user_id=self.auth_context.user_id if self.auth_context else None,
            source="portfolio_manager",
        )

        await self.streamer.publish_event(event)

    async def publish_trade_signal(
        self, signal_type: str, symbol: str, confidence: float, details: Dict[str, Any]
    ):
        """Publish trade signal"""
        event = MCPEvent(
            event_type=EventType.TRADE_SIGNAL,
            data={
                "signal_type": signal_type,
                "symbol": symbol,
                "confidence": confidence,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
            },
            timestamp=time.time(),
            source="trading_signals",
        )

        await self.streamer.publish_event(event)

    async def publish_system_status(
        self, component: str, status: str, details: Dict[str, Any] = None
    ):
        """Publish system status update"""
        event = MCPEvent(
            event_type=EventType.SYSTEM_STATUS,
            data={
                "component": component,
                "status": status,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
            timestamp=time.time(),
            source="system_monitor",
        )

        await self.streamer.publish_event(event)

    async def publish_error(self, error_type: str, message: str, context: Dict[str, Any] = None):
        """Publish error event"""
        event = MCPEvent(
            event_type=EventType.ERROR,
            data={
                "error_type": error_type,
                "message": message,
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
            timestamp=time.time(),
            user_id=self.auth_context.user_id if self.auth_context else None,
            source="error_handler",
        )

        await self.streamer.publish_event(event)


# Global event streamer
global_event_streamer = EventStreamer()


def get_event_streamer() -> EventStreamer:
    """Get global event streamer"""
    return global_event_streamer


def create_event_publisher(auth_context: AuthContext = None) -> MCPEventPublisher:
    """Create event publisher"""
    return MCPEventPublisher(global_event_streamer, auth_context)


async def subscribe_to_events(
    event_types: List[EventType],
    callback: Callable[[MCPEvent], None],
    user_id: str = None,
    source: str = None,
) -> str:
    """Subscribe to events"""
    filter = EventFilter(event_types, user_id, source)
    return await global_event_streamer.subscribe(filter, callback)


async def unsubscribe_from_events(subscription_id: str) -> bool:
    """Unsubscribe from events"""
    return await global_event_streamer.unsubscribe(subscription_id)


async def get_recent_events(
    event_types: List[EventType] = None, since: float = None, limit: int = 100
) -> List[Dict[str, Any]]:
    """Get recent events"""
    filter = EventFilter(event_types) if event_types else None
    return await global_event_streamer.get_events(filter, since, limit)
