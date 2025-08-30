"""
A2A Protocol Definitions for cryptotrading.com
Ensures 100% A2A compliance across all agents
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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
    workflow_context: Optional[
        Dict[str, Any]
    ] = None  # Workflow ID, step ID, execution ID, instance address

    # Blockchain signature fields
    sender_blockchain_address: Optional[str] = None
    blockchain_signature: Optional[Dict[str, Any]] = None
    blockchain_context: Optional[
        Dict[str, Any]
    ] = None  # chain_id, contract_address, instance_address

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
            "blockchain_context": self.blockchain_context,
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
    def create_message(
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 0,
    ) -> A2AMessage:
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
            priority=priority,
        )

    @staticmethod
    def create_response(
        message: A2AMessage,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> A2AResponse:
        """Create standardized A2A response"""
        return A2AResponse(
            success=success,
            message_id=message.message_id,
            sender_id=message.receiver_id,  # Response sender is original receiver
            receiver_id=message.sender_id,  # Response receiver is original sender
            timestamp=datetime.now().isoformat(),
            data=data,
            error=error,
            correlation_id=message.correlation_id,
        )

    @staticmethod
    def validate_message(message_dict: Dict[str, Any]) -> bool:
        """Validate A2A message structure"""
        required_fields = [
            "sender_id",
            "receiver_id",
            "type",
            "payload",
            "message_id",
            "timestamp",
            "protocol_version",
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


# A2A Capability Registry - Updated with Specialized Agents
A2A_CAPABILITIES = {
    # Specialized STRAND Agents
    "news_intelligence_agent": [
        "news_collection",
        "sentiment_analysis",
        "news_translation",
        "market_correlation",
        "alert_generation",
        "news_summarization",
        "insight_extraction",
        "sentiment_monitoring",
        "trending_topics",
        "source_credibility",
        "fake_news_detection",
        "narrative_tracking",
    ],
    "trading_algorithm_agent": [
        "grid_trading",
        "dollar_cost_averaging",
        "arbitrage",
        "momentum_trend",
        "mean_reversion",
        "scalping",
        "market_making",
        "breakout_trading",
        "ml_predictive",
        "multi_strategy_hybrid",
        "signal_generation",
        "strategy_analysis",
    ],
    "technical_analysis_agent": [
        "technical_indicators",
        "momentum_analysis",
        "volume_analysis",
        "pattern_recognition",
        "trend_analysis",
        "oscillator_analysis",
        "support_resistance",
        "market_sentiment",
    ],
    "mcts_calculation_agent": [
        "general_optimization",
        "algorithm_performance",
        "monte_carlo_simulation",
        "calculation_metrics",
        "convergence_analysis",
    ],
    "ml_agent": [
        "price_prediction",
        "model_training",
        "feature_engineering",
        "batch_prediction",
        "model_evaluation",
        "hyperparameter_optimization",
        "ml_inference",
    ],
    "agent_manager_agent": [
        "agent_registration",
        "compliance_monitoring",
        "skill_card_management",
        "mcp_segregation",
        "system_health",
        "agent_orchestration",
        "capability_discovery",
        "status_monitoring",
    ],
    "strands_glean_agent": [
        "code_analysis",
        "dependency_mapping",
        "symbol_search",
        "code_navigation",
        "insight_generation",
        "coverage_validation",
        "change_monitoring",
        "code_quality",
    ],
    "historical_data_loader_agent": [
        "data_loading",
        "historical_data",
        "multi_source_aggregation",
        "temporal_alignment",
        "data_validation",
        "catalog_management",
        "yahoo_finance",
        "fred_data",
        "cboe_data",
        "defillama_data",
    ],
    "database_agent": [
        "data_storage",
        "data_retrieval",
        "bulk_insert",
        "ai_analysis_storage",
        "portfolio_management",
        "trade_history",
        "database_health",
        "query_optimization",
        "data_cleanup",
    ],
    "feature_store_agent": [
        "compute_features",
        "get_feature_vector",
        "get_training_features",
        "get_feature_definitions",
        "get_feature_importance",
        "feature_engineering",
        "ml_features",
        "technical_indicators",
    ],
    "data_analysis_agent": [
        "validate_data_quality",
        "analyze_data_distribution",
        "compute_correlation_matrix",
        "detect_outliers",
        "compute_rolling_statistics",
        "statistical_analysis",
        "data_validation",
        "quality_assessment",
    ],
    "clrs_algorithms_agent": [
        "binary_search",
        "linear_search",
        "quick_select",
        "find_minimum",
        "find_maximum",
        "insertion_sort",
        "merge_sort",
        "quick_sort",
        "algorithmic_calculations",
        "search_algorithms",
        "sorting_algorithms",
        "clrs_algorithms",
    ],
    "technical_analysis_skills_agent": [
        "calculate_momentum_indicators",
        "calculate_momentum_volatility",
        "analyze_volume_patterns",
        "identify_support_resistance",
        "detect_chart_patterns",
        "comprehensive_analysis",
        "technical_indicators",
        "momentum_analysis",
        "volume_analysis",
        "pattern_recognition",
        "support_resistance",
        "chart_patterns",
        "ta_calculations",
    ],
    "ml_models_agent": [
        "train_model",
        "predict_prices",
        "evaluate_model",
        "optimize_hyperparameters",
        "ensemble_predict",
        "feature_importance",
        "ml_training",
        "model_evaluation",
        "hyperparameter_optimization",
        "ensemble_methods",
        "ml_calculations",
    ],
    "code_quality_agent": [
        "analyze_code_quality",
        "calculate_complexity_metrics",
        "detect_code_smells",
        "analyze_dependencies",
        "calculate_impact_analysis",
        "generate_quality_report",
        "code_analysis",
        "quality_metrics",
        "code_smells",
        "dependency_analysis",
        "impact_analysis",
        "quality_reporting",
    ],
    "trading_algorithm_agent": [
        "grid_trading",
        "dollar_cost_averaging",
        "arbitrage_detection",
        "momentum_trading",
        "mean_reversion",
        "scalping",
        "market_making",
        "breakout_trading",
        "ml_predictions",
        "multi_strategy_management",
        "risk_management",
        "portfolio_optimization",
        "signal_generation",
        "strategy_analysis",
        "backtesting",
    ],
    "data_analysis_agent": [
        "data_processing",
        "statistical_analysis",
        "pattern_recognition",
        "anomaly_detection",
        "correlation_analysis",
        "trend_analysis",
        "data_quality_assessment",
        "feature_extraction",
        "data_visualization",
        "report_generation",
    ],
    "feature_store_agent": [
        "feature_storage",
        "feature_retrieval",
        "feature_versioning",
        "feature_validation",
        "feature_transformation",
        "feature_serving",
        "metadata_management",
        "lineage_tracking",
        "feature_monitoring",
        "feature_discovery",
    ],
}

# Message routing table - Updated with Specialized Agents
A2A_ROUTING = {
    MessageType.DATA_LOAD_REQUEST: [
        "historical_data_loader_agent",
        "database_agent",
        "feature_store_agent",
    ],
    MessageType.ANALYSIS_REQUEST: [
        "technical_analysis_agent",
        "ml_agent",
        "strands_glean_agent",
        "feature_store_agent",
        "data_analysis_agent",
        "technical_analysis_skills_agent",
        "ml_models_agent",
        "code_quality_agent",
        "clrs_algorithms_agent",
        "trading_algorithm_agent",
    ],
    MessageType.DATA_QUERY: ["database_agent", "feature_store_agent"],
    MessageType.TRADE_EXECUTION: [
        "trading_algorithm_agent"
    ],  # Note: Generates signals only, no actual execution
    MessageType.WORKFLOW_REQUEST: ["agent_manager_agent", "trading_algorithm_agent"],
    MessageType.WORKFLOW_STATUS: ["agent_manager_agent"],
    MessageType.HEARTBEAT: ["agent_manager_agent"],
    # Legacy routing (for backward compatibility)
    "legacy_data_load": ["database-001"],
    "legacy_analysis": ["illuminate-001", "database-001"],
    "legacy_query": ["database-001"],
    "legacy_trade": ["execute-001", "database-001"],
}


# Agent registration helper
class A2AAgentRegistry:
    """A2A Agent Registration and Discovery"""

    _registered_agents = {}

    @classmethod
    def register_agent(
        cls,
        agent_id: str,
        capabilities: List[str],
        agent_instance=None,
        status: AgentStatus = AgentStatus.ACTIVE,
    ):
        """Register agent with A2A protocol"""
        cls._registered_agents[agent_id] = {
            "capabilities": capabilities,
            "status": status,
            "instance": agent_instance,
            "registered_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat(),
        }
        return True

    @classmethod
    def get_agent_capabilities(cls, agent_id: str) -> List[str]:
        """Get capabilities for specific agent"""
        return cls._registered_agents.get(agent_id, {}).get("capabilities", [])

    @classmethod
    def find_agents_by_capability(cls, capability: str) -> List[str]:
        """Find all agents that support a specific capability"""
        agents = []
        for agent_id, info in cls._registered_agents.items():
            if capability in info.get("capabilities", []):
                agents.append(agent_id)
        return agents

    @classmethod
    def get_all_agents(cls) -> Dict[str, Any]:
        """Get all registered agents"""
        return cls._registered_agents.copy()

    @classmethod
    def update_agent_status(cls, agent_id: str, status: AgentStatus):
        """Update agent status"""
        if agent_id in cls._registered_agents:
            cls._registered_agents[agent_id]["status"] = status
            cls._registered_agents[agent_id]["last_heartbeat"] = datetime.now().isoformat()
            return True
        return False
