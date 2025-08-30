"""
Enhanced A2A Message Types for Professional Trading
Advanced MCP message types for comprehensive indicators and institutional strategies
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class EnhancedMessageType(Enum):
    """Enhanced message types for professional trading operations"""

    # Comprehensive Indicators
    COMPREHENSIVE_INDICATORS_REQUEST = "comprehensive_indicators_request"
    COMPREHENSIVE_INDICATORS_RESPONSE = "comprehensive_indicators_response"
    COMPREHENSIVE_INDICATORS_STREAM = "comprehensive_indicators_stream"

    # Institutional Strategies
    INSTITUTIONAL_STRATEGY_REQUEST = "institutional_strategy_request"
    INSTITUTIONAL_STRATEGY_RESPONSE = "institutional_strategy_response"
    INSTITUTIONAL_STRATEGY_STREAM = "institutional_strategy_stream"

    # Market Regime Detection
    REGIME_DETECTION_REQUEST = "regime_detection_request"
    REGIME_DETECTION_RESPONSE = "regime_detection_response"
    REGIME_DETECTION_ALERT = "regime_detection_alert"
    REGIME_CHANGE_NOTIFICATION = "regime_change_notification"

    # Portfolio Optimization
    PORTFOLIO_OPTIMIZATION_REQUEST = "portfolio_optimization_request"
    PORTFOLIO_OPTIMIZATION_RESPONSE = "portfolio_optimization_response"
    PORTFOLIO_REBALANCE_SIGNAL = "portfolio_rebalance_signal"

    # Real-time Alerts
    THRESHOLD_ALERT = "threshold_alert"
    CORRELATION_ALERT = "correlation_alert"
    VOLATILITY_ALERT = "volatility_alert"
    RISK_MANAGEMENT_ALERT = "risk_management_alert"

    # Streaming Data
    REAL_TIME_INDICATORS_STREAM = "real_time_indicators_stream"
    CORRELATION_MATRIX_STREAM = "correlation_matrix_stream"
    POSITION_SIZING_STREAM = "position_sizing_stream"

    # Professional Analytics
    OPTIONS_ANALYTICS_REQUEST = "options_analytics_request"
    OPTIONS_ANALYTICS_RESPONSE = "options_analytics_response"
    ENSEMBLE_CORRELATION_REQUEST = "ensemble_correlation_request"
    ENSEMBLE_CORRELATION_RESPONSE = "ensemble_correlation_response"

    # Protocol Management
    PROTOCOL_VERSION_REQUEST = "protocol_version_request"
    PROTOCOL_VERSION_RESPONSE = "protocol_version_response"
    CAPABILITY_DISCOVERY_REQUEST = "capability_discovery_request"
    CAPABILITY_DISCOVERY_RESPONSE = "capability_discovery_response"

    # Interactive Help
    HELP_REQUEST = "help_request"
    HELP_RESPONSE = "help_response"
    TOOL_DOCUMENTATION_REQUEST = "tool_documentation_request"
    TOOL_DOCUMENTATION_RESPONSE = "tool_documentation_response"


@dataclass
class ProtocolVersion:
    """Protocol version information"""

    major: int
    minor: int
    patch: int
    features: List[str]

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def version_string(self) -> str:
        return self.__str__()


@dataclass
class StreamingConfig:
    """Configuration for streaming responses"""

    batch_size: int = 100
    frequency_ms: int = 1000
    buffer_size: int = 10000
    compression: bool = True
    format: str = "json"  # json, msgpack, protobuf


@dataclass
class ComprehensiveIndicatorsRequest:
    """Request for comprehensive indicators data"""

    symbols: List[str]
    days_back: int = 365
    interval: str = "1d"
    include_metadata: bool = True
    streaming: bool = False
    streaming_config: Optional[StreamingConfig] = None


@dataclass
class InstitutionalStrategyRequest:
    """Request for institutional strategy data"""

    strategy_name: str  # two_sigma, deribit, jump_trading, etc.
    days_back: int = 365
    interval: str = "1d"
    include_weights: bool = True
    calculate_signals: bool = True
    streaming: bool = False


@dataclass
class RegimeDetectionRequest:
    """Request for market regime detection"""

    regime_type: Optional[str] = None  # risk_on, risk_off, neutral, crisis, euphoria
    lookback_days: int = 90
    include_probabilities: bool = True
    alert_threshold: float = 0.7


@dataclass
class PortfolioOptimizationRequest:
    """Request for portfolio optimization"""

    assets: List[str]
    optimization_method: str = "mean_variance"  # mean_variance, risk_parity, black_litterman
    constraints: Dict[str, Any] = None
    risk_tolerance: float = 0.02
    rebalance_frequency: str = "monthly"


@dataclass
class ThresholdAlert:
    """Threshold breach alert"""

    symbol: str
    indicator_name: str
    current_value: float
    threshold_value: float
    threshold_type: str  # warning, critical, extreme
    severity: str  # low, medium, high, critical
    action_required: str
    timestamp: datetime
    agent_id: str


@dataclass
class RegimeChangeNotification:
    """Market regime change notification"""

    previous_regime: str
    new_regime: str
    confidence: float
    trigger_indicators: List[str]
    regime_duration: int  # days in previous regime
    timestamp: datetime
    implications: List[str]


@dataclass
class CorrelationAlert:
    """Correlation breakdown/strengthening alert"""

    asset_pair: tuple
    correlation_type: str  # ensemble, rolling, regime_based
    previous_correlation: float
    current_correlation: float
    change_magnitude: float
    significance_level: float
    timestamp: datetime


@dataclass
class VolatilityAlert:
    """Volatility regime change alert"""

    symbol: str
    volatility_measure: str  # vix, realized, implied
    current_level: float
    percentile: float
    regime: str  # low_vol, normal, high_vol, extreme
    timestamp: datetime
    cross_asset_implications: List[str]


@dataclass
class RiskManagementAlert:
    """Risk management alert"""

    alert_type: str  # position_size, correlation_risk, regime_change, threshold_breach
    severity: str  # low, medium, high, critical
    affected_positions: List[str]
    recommended_actions: List[str]
    risk_metrics: Dict[str, float]
    timestamp: datetime


@dataclass
class CapabilityInfo:
    """Agent capability information"""

    agent_id: str
    agent_type: str
    capabilities: List[str]
    tools: List[Dict[str, Any]]
    protocols_supported: List[str]
    version: ProtocolVersion
    performance_metrics: Dict[str, Any]


@dataclass
class ToolDocumentation:
    """Tool documentation information"""

    tool_name: str
    description: str
    parameters: Dict[str, Any]
    examples: List[Dict[str, Any]]
    performance_notes: str
    institutional_usage: str
    related_tools: List[str]


# Protocol version constants
CURRENT_PROTOCOL_VERSION = ProtocolVersion(
    major=2,
    minor=1,
    patch=0,
    features=[
        "comprehensive_indicators",
        "institutional_strategies",
        "regime_detection",
        "portfolio_optimization",
        "real_time_alerts",
        "streaming_responses",
        "options_analytics",
        "ensemble_correlations",
        "interactive_help",
    ],
)

# Message type to dataclass mapping
MESSAGE_DATACLASS_MAPPING = {
    EnhancedMessageType.COMPREHENSIVE_INDICATORS_REQUEST: ComprehensiveIndicatorsRequest,
    EnhancedMessageType.INSTITUTIONAL_STRATEGY_REQUEST: InstitutionalStrategyRequest,
    EnhancedMessageType.REGIME_DETECTION_REQUEST: RegimeDetectionRequest,
    EnhancedMessageType.PORTFOLIO_OPTIMIZATION_REQUEST: PortfolioOptimizationRequest,
    EnhancedMessageType.THRESHOLD_ALERT: ThresholdAlert,
    EnhancedMessageType.REGIME_CHANGE_NOTIFICATION: RegimeChangeNotification,
    EnhancedMessageType.CORRELATION_ALERT: CorrelationAlert,
    EnhancedMessageType.VOLATILITY_ALERT: VolatilityAlert,
    EnhancedMessageType.RISK_MANAGEMENT_ALERT: RiskManagementAlert,
}
