"""
Communication Interface Definitions
Abstract interfaces for communication components to prevent circular dependencies
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, AsyncIterator
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """Message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class IMessage(ABC):
    """Message interface"""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Message ID"""
        pass
    
    @property
    @abstractmethod
    def sender_id(self) -> str:
        """Sender ID"""
        pass
    
    @property
    @abstractmethod
    def recipient_id(self) -> Optional[str]:
        """Recipient ID (None for broadcast)"""
        pass
    
    @property
    @abstractmethod
    def message_type(self) -> MessageType:
        """Message type"""
        pass
    
    @property
    @abstractmethod
    def payload(self) -> Dict[str, Any]:
        """Message payload"""
        pass
    
    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Message timestamp"""
        pass


class IMessageHandler(ABC):
    """Message handler interface"""
    
    @abstractmethod
    async def handle_message(self, message: IMessage) -> Optional[IMessage]:
        """Handle incoming message"""
        pass
    
    @abstractmethod
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if handler can handle message type"""
        pass


class ITransport(ABC):
    """Transport layer interface"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect transport"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect transport"""
        pass
    
    @abstractmethod
    async def send_message(self, message: IMessage) -> bool:
        """Send message"""
        pass
    
    @abstractmethod
    async def receive_messages(self) -> AsyncIterator[IMessage]:
        """Receive messages"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected"""
        pass


class ICommunicationManager(ABC):
    """Communication manager interface"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize communication manager"""
        pass
    
    @abstractmethod
    async def register_handler(self, message_type: MessageType, 
                             handler: IMessageHandler) -> bool:
        """Register message handler"""
        pass
    
    @abstractmethod
    async def send_message(self, recipient_id: str, payload: Dict[str, Any],
                          message_type: MessageType = MessageType.REQUEST,
                          priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Send message to specific recipient"""
        pass
    
    @abstractmethod
    async def broadcast_message(self, payload: Dict[str, Any],
                              message_type: MessageType = MessageType.BROADCAST) -> int:
        """Broadcast message to all connected agents"""
        pass
    
    @abstractmethod
    async def subscribe_to_events(self, event_types: List[str]) -> bool:
        """Subscribe to event types"""
        pass
    
    @abstractmethod
    async def unsubscribe_from_events(self, event_types: List[str]) -> bool:
        """Unsubscribe from event types"""
        pass
    
    @abstractmethod
    def get_connected_agents(self) -> List[str]:
        """Get list of connected agent IDs"""
        pass


class IProtocolHandler(ABC):
    """Protocol handler interface"""
    
    @abstractmethod
    async def encode_message(self, message: IMessage) -> bytes:
        """Encode message for transmission"""
        pass
    
    @abstractmethod
    async def decode_message(self, data: bytes) -> IMessage:
        """Decode received message"""
        pass
    
    @abstractmethod
    def get_protocol_version(self) -> str:
        """Get protocol version"""
        pass
    
    @abstractmethod
    def supports_compression(self) -> bool:
        """Check if protocol supports compression"""
        pass


class IServiceDiscovery(ABC):
    """Service discovery interface"""
    
    @abstractmethod
    async def register_service(self, service_id: str, service_info: Dict[str, Any]) -> bool:
        """Register service"""
        pass
    
    @abstractmethod
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister service"""
        pass
    
    @abstractmethod
    async def discover_services(self, service_type: str = None) -> List[Dict[str, Any]]:
        """Discover available services"""
        pass
    
    @abstractmethod
    async def get_service_info(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get service information"""
        pass
    
    @abstractmethod
    async def health_check_service(self, service_id: str) -> bool:
        """Check service health"""
        pass


class ILoadBalancer(ABC):
    """Load balancer interface"""
    
    @abstractmethod
    async def select_target(self, service_type: str, 
                          strategy: str = "round_robin") -> Optional[str]:
        """Select target service for load balancing"""
        pass
    
    @abstractmethod
    async def report_health(self, service_id: str, is_healthy: bool):
        """Report service health status"""
        pass
    
    @abstractmethod
    async def get_target_weights(self, service_type: str) -> Dict[str, float]:
        """Get target weights for weighted load balancing"""
        pass


class ICircuitBreaker(ABC):
    """Circuit breaker interface"""
    
    @abstractmethod
    async def call_service(self, service_id: str, call_func: Callable) -> Any:
        """Call service with circuit breaker protection"""
        pass
    
    @abstractmethod
    async def record_success(self, service_id: str):
        """Record successful service call"""
        pass
    
    @abstractmethod
    async def record_failure(self, service_id: str):
        """Record failed service call"""
        pass
    
    @abstractmethod
    def get_circuit_state(self, service_id: str) -> str:
        """Get circuit breaker state (open/closed/half-open)"""
        pass


class IEventBus(ABC):
    """Event bus interface"""
    
    @abstractmethod
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """Publish event"""
        pass
    
    @abstractmethod
    async def subscribe_to_event(self, event_type: str, 
                               handler: Callable[[Dict[str, Any]], None]):
        """Subscribe to event type"""
        pass
    
    @abstractmethod
    async def unsubscribe_from_event(self, event_type: str, handler: Callable):
        """Unsubscribe from event type"""
        pass
    
    @abstractmethod
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        pass