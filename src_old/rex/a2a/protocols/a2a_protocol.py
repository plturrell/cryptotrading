"""
A2A Protocol Definitions for rex.com
Ensures 100% A2A compliance across all agents
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    """A2A Message Types"""
    DATA_LOAD_REQUEST = "DATA_LOAD_REQUEST"
    DATA_LOAD_RESPONSE = "DATA_LOAD_RESPONSE"
    ANALYSIS_REQUEST = "ANALYSIS_REQUEST"
    ANALYSIS_RESPONSE = "ANALYSIS_RESPONSE"
    DATA_QUERY = "DATA_QUERY"
    DATA_QUERY_RESPONSE = "DATA_QUERY_RESPONSE"
    TRADE_EXECUTION = "TRADE_EXECUTION"
    TRADE_RESPONSE = "TRADE_RESPONSE"
    WORKFLOW_REQUEST = "WORKFLOW_REQUEST"
    WORKFLOW_RESPONSE = "WORKFLOW_RESPONSE"
    WORKFLOW_STATUS = "WORKFLOW_STATUS"
    HEARTBEAT = "HEARTBEAT"
    ERROR = "ERROR"
    MEMORY_SHARE = "MEMORY_SHARE"
    MEMORY_REQUEST = "MEMORY_REQUEST"
    MEMORY_RESPONSE = "MEMORY_RESPONSE"

class AgentStatus(Enum):
    """Agent Status States"""
    ACTIVE = "active"
    BUSY = "busy"
    INACTIVE = "inactive"
    ERROR = "error"

@dataclass
class A2AMessage:
    """Standard A2A Message Structure"""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    message_id: str
    timestamp: str
    protocol_version: str = "1.0"
    correlation_id: Optional[str] = None
    priority: int = 0  # 0=low, 1=normal, 2=high, 3=critical
    workflow_context: Optional[Dict[str, Any]] = None  # Workflow ID, step ID, execution ID, instance address
    
    # Blockchain signature fields
    sender_blockchain_address: Optional[str] = None
    blockchain_signature: Optional[Dict[str, Any]] = None
    blockchain_context: Optional[Dict[str, Any]] = None  # chain_id, contract_address, instance_address
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "type": self.message_type.value,
            "payload": self.payload,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "protocol_version": self.protocol_version,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "workflow_context": self.workflow_context,
            "sender_blockchain_address": self.sender_blockchain_address,
            "blockchain_signature": self.blockchain_signature,
            "blockchain_context": self.blockchain_context
        }

@dataclass
class A2AResponse:
    """Standard A2A Response Structure"""
    success: bool
    message_id: str
    sender_id: str
    receiver_id: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    correlation_id: Optional[str] = None

class A2AProtocol:
    """A2A Protocol Handler"""
    
    @staticmethod
    def create_message(sender_id: str, receiver_id: str, message_type: MessageType, 
                      payload: Dict[str, Any], priority: int = 0) -> A2AMessage:
        """Create standardized A2A message"""
        message_id = f"{sender_id}_{datetime.now().timestamp()}"
        timestamp = datetime.now().isoformat()
        
        return A2AMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            message_id=message_id,
            timestamp=timestamp,
            priority=priority
        )
    
    @staticmethod
    def create_response(message: A2AMessage, success: bool, 
                       data: Optional[Dict[str, Any]] = None, 
                       error: Optional[str] = None) -> A2AResponse:
        """Create standardized A2A response"""
        return A2AResponse(
            success=success,
            message_id=message.message_id,
            sender_id=message.receiver_id,  # Response sender is original receiver
            receiver_id=message.sender_id,  # Response receiver is original sender
            timestamp=datetime.now().isoformat(),
            data=data,
            error=error,
            correlation_id=message.correlation_id
        )
    
    @staticmethod
    def validate_message(message_dict: Dict[str, Any]) -> bool:
        """Validate A2A message structure"""
        required_fields = [
            "sender_id", "receiver_id", "type", "payload", 
            "message_id", "timestamp", "protocol_version"
        ]
        
        for field in required_fields:
            if field not in message_dict:
                return False
        
        # Validate message type
        try:
            MessageType(message_dict["type"])
        except ValueError:
            return False
        
        return True

# A2A Capability Registry
A2A_CAPABILITIES = {
    'historical-loader-001': [
        'data_loading', 'historical_data', 'technical_indicators', 
        'bulk_processing', 'multi_source_aggregation'
    ],
    'database-001': [
        'data_storage', 'data_retrieval', 'bulk_insert', 'ai_analysis_storage',
        'portfolio_management', 'trade_history', 'multi_ai_analysis'
    ],
    'transform-001': [
        'data_preprocessing', 'format_conversion', 'data_validation',
        'technical_analysis', 'feature_engineering'
    ],
    'illuminate-001': [
        'market_analysis', 'pattern_recognition', 'insight_generation',
        'multi_ai_integration', 'sentiment_analysis'
    ],
    'execute-001': [
        'trade_execution', 'order_management', 'risk_control',
        'portfolio_optimization', 'execution_analytics'
    ]
}

# Message routing table
A2A_ROUTING = {
    MessageType.DATA_LOAD_REQUEST: ['database-001'],
    MessageType.ANALYSIS_REQUEST: ['illuminate-001', 'database-001'],
    MessageType.DATA_QUERY: ['database-001'],
    MessageType.TRADE_EXECUTION: ['execute-001', 'database-001']
}