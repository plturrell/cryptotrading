"""
Feature Store Agent - STRANDS Integration
Specialized agent for ML feature engineering and computation

NOTE: This agent is fully MCP-compliant. ALL functionality must be accessed through MCP tools via the process_mcp_request() method.
Direct method calls to business logic methods are not supported - use MCP tool calls instead.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ...infrastructure.mcp.feature_store_mcp_tools import feature_store_mcp_tools
from ...ml.feature_store import feature_store
from ...protocols.a2a.a2a_messaging import A2AMessagingClient
from ...protocols.cds.cds_client import A2AAgentCDSMixin
from ..strands import StrandsAgent

# Enhanced CDS integration imports
try:
    from ...infrastructure.monitoring.cds_integration_monitor import get_cds_monitor, CDSOperationType
    from ...infrastructure.transactions.cds_transactional_client import CDSTransactionalMixin
    from ...infrastructure.transactions.agent_transaction_manager import transactional, TransactionIsolation
    CDS_ENHANCED_FEATURES = True
except ImportError:
    # Fallback classes for compatibility
    class CDSTransactionalMixin:
        pass
    def transactional(transaction_type=None, isolation_level=None):
        def decorator(func):
            return func
        return decorator
    class TransactionIsolation:
        READ_COMMITTED = "READ_COMMITTED"
    get_cds_monitor = None
    CDSOperationType = None
    CDS_ENHANCED_FEATURES = False

logger = logging.getLogger(__name__)


class FeatureStoreAgent(StrandsAgent, A2AAgentCDSMixin, CDSTransactionalMixin):
    """STRANDS agent for feature store operations"""

    def __init__(self, agent_id: str = "feature_store_agent", **kwargs):
        """Initialize Feature Store Agent with CDS integration"""
        super().__init__(
            agent_id=agent_id,
            agent_type="feature_store",
            capabilities=[
                "compute_features",
                "get_feature_vector",
                "get_training_features",
                "get_feature_definitions",
                "get_feature_importance",
            ],
            **kwargs
        )

        self.feature_store = feature_store
        self.mcp_tools = feature_store_mcp_tools

        # Initialize CDS monitoring if available
        if CDS_ENHANCED_FEATURES and get_cds_monitor:
            self._cds_monitor = get_cds_monitor()
        else:
            self._cds_monitor = None

        # Initialize A2A messaging for cross-agent communication
        self.a2a_messaging = A2AMessagingClient(agent_id=self.agent_id)

        # Register with A2A protocol
        from ...protocols.a2a.a2a_protocol import A2A_CAPABILITIES, A2AAgentRegistry

        capabilities = A2A_CAPABILITIES.get(agent_id, [])
        A2AAgentRegistry.register_agent(agent_id, capabilities, self)

        # Initialize MCP handlers
        self.mcp_handlers = {
            "compute_features": self._mcp_compute_features,
            "get_feature_vector": self._mcp_get_feature_vector,
            "get_training_features": self._mcp_prepare_training_data,
            "get_feature_definitions": self._mcp_get_feature_metadata,
            "get_feature_importance": self._mcp_analyze_feature_importance,
            # A2A feature integration tools
            "a2a_provide_features": self._mcp_a2a_provide_features,
            "a2a_engineer_custom_features": self._mcp_a2a_engineer_custom_features,
            "a2a_validate_feature_quality": self._mcp_a2a_validate_feature_quality,
        }

        # Register MCP tools as STRANDS tools
        self._register_strands_tools()

        logger.info(f"Feature Store Agent {agent_id} initialized")

    def _register_strands_tools(self):
        """Register MCP tools as STRANDS tools"""
        for tool_def in self.mcp_tools.tools:
            tool_name = tool_def["name"]

            # Create STRANDS tool wrapper
            async def tool_wrapper(tool_name=tool_name, **kwargs):
                return await self.mcp_tools.handle_tool_call(tool_name, kwargs)

            # Register with STRANDS
            self.register_tool(
                name=tool_name,
                description=tool_def["description"],
                func=tool_wrapper,
                input_schema=tool_def["inputSchema"],
            )

    async def initialize(self) -> bool:
        """Initialize the Feature Store Agent with CDS integration"""
        try:
            logger.info(f"Initializing Feature Store Agent {self.agent_id} with CDS")

            # Initialize CDS connection
            await self.initialize_cds()

            # Test feature store connectivity
            available_features = list(self.feature_store.features.keys())
            logger.info(f"Feature Store has {len(available_features)} features available")

            # Quick validation test with CDS monitoring
            if self._cds_monitor and CDSOperationType:
                async with self._cds_monitor.track_operation(self.agent_id, CDSOperationType.DATA_ACCESS):
                    try:
                        importance = self.feature_store.get_feature_importance()
                        logger.info(f"Feature importance loaded: {len(importance)} features")
                    except Exception as e:
                        logger.warning(f"Feature importance test failed: {e}")
            else:
                # Fallback without monitoring
                try:
                    importance = self.feature_store.get_feature_importance()
                    logger.info(f"Feature importance loaded: {len(importance)} features")
                except Exception as e:
                    logger.warning(f"Feature importance test failed: {e}")

            logger.info(f"Feature Store Agent {self.agent_id} initialized successfully with CDS")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Feature Store Agent {self.agent_id}: {e}")
            return False

    async def start(self) -> bool:
        """Start the Feature Store Agent"""
        try:
            logger.info(f"Starting Feature Store Agent {self.agent_id}")

            # Feature store is primarily request-driven
            # No background processes needed

            logger.info(f"Feature Store Agent {self.agent_id} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start Feature Store Agent {self.agent_id}: {e}")
            return False

    async def process_mcp_request(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        MAIN ENTRY POINT for all Feature Store Agent functionality.

        This is the ONLY way to access agent functionality - all operations must go through MCP tools.

        Args:
            tool_name: Name of the MCP tool to execute
            arguments: Tool arguments

        Returns:
            Dict containing tool execution results
        """
        try:
            if tool_name not in self.mcp_handlers:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}. Available tools: {list(self.mcp_handlers.keys())}",
                }

            handler = self.mcp_handlers[tool_name]
            result = await handler(arguments)

            return {"success": True, "result": result, "tool_name": tool_name}

        except Exception as e:
            logger.error(f"Error processing MCP request for tool {tool_name}: {e}")
            return {"success": False, "error": str(e), "tool_name": tool_name}

    @transactional(transaction_type="FEATURE_COMPUTATION", isolation_level=TransactionIsolation.READ_COMMITTED)
    async def _mcp_compute_features(self, arguments: Dict[str, Any], transaction=None) -> Dict[str, Any]:
        """MCP handler: Compute features for a specific symbol with transaction support"""
        try:
            symbol = arguments.get("symbol")
            features = arguments.get("features")

            if not symbol:
                raise ValueError("Symbol is required")

            # Track operation with CDS monitoring
            if self._cds_monitor and CDSOperationType:
                async with self._cds_monitor.track_operation(self.agent_id, CDSOperationType.COMPUTATION):
                    # Register agent if CDS connection available
                    if hasattr(self, '_cds_client') and self._cds_client:
                        await self.register_with_cds(capabilities={"feature_computation": True})
                    
                    result = await self.execute_tool(
                        "compute_features", {"symbol": symbol, "features": features}
                    )
            else:
                # Fallback without monitoring
                result = await self.execute_tool(
                    "compute_features", {"symbol": symbol, "features": features}
                )

            return result

        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            raise

    async def _mcp_get_feature_vector(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP handler: Get latest feature vector for a symbol"""
        try:
            symbol = arguments.get("symbol")

            if not symbol:
                raise ValueError("Symbol is required")

            result = await self.execute_tool("get_feature_vector", {"symbol": symbol})

            return result

        except Exception as e:
            logger.error(f"Error getting feature vector for {symbol}: {e}")
            raise

    @transactional(transaction_type="TRAINING_DATA_PREPARATION", isolation_level=TransactionIsolation.READ_COMMITTED)
    async def _mcp_prepare_training_data(self, arguments: Dict[str, Any], transaction=None) -> Dict[str, Any]:
        """MCP handler: Prepare training data for multiple symbols with transaction support"""
        try:
            symbols = arguments.get("symbols", [])
            start_date = arguments.get("start_date")
            end_date = arguments.get("end_date")

            if not symbols:
                raise ValueError("Symbols list is required")
            if not start_date:
                raise ValueError("Start date is required")
            if not end_date:
                raise ValueError("End date is required")

            # Track operation with CDS monitoring
            if self._cds_monitor and CDSOperationType:
                async with self._cds_monitor.track_operation(self.agent_id, CDSOperationType.DATA_ACCESS):
                    result = await self.execute_tool(
                        "get_training_features",
                        {"symbols": symbols, "start_date": start_date, "end_date": end_date},
                    )
            else:
                # Fallback without monitoring
                result = await self.execute_tool(
                    "get_training_features",
                    {"symbols": symbols, "start_date": start_date, "end_date": end_date},
                )

            return result

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    async def _mcp_get_feature_metadata(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP handler: Get feature definitions and metadata"""
        try:
            feature_names = arguments.get("feature_names")

            result = await self.execute_tool(
                "get_feature_definitions", {"feature_names": feature_names}
            )

            return result

        except Exception as e:
            logger.error(f"Error getting feature metadata: {e}")
            raise

    async def _mcp_analyze_feature_importance(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP handler: Get feature importance analysis"""
        try:
            result = await self.execute_tool("get_feature_importance", {})

            return result

        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            raise

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming messages through MCP tools.

        DEPRECATED: All functionality should use process_mcp_request() instead.
        This method is maintained for backward compatibility only.
        """
        try:
            message_type = message.get("type", "unknown")

            # Route to MCP handlers
            if message_type == "compute_features":
                return await self.process_mcp_request(
                    "compute_features",
                    {"symbol": message.get("symbol"), "features": message.get("features")},
                )

            elif message_type == "get_feature_vector":
                return await self.process_mcp_request(
                    "get_feature_vector", {"symbol": message.get("symbol")}
                )

            elif message_type == "prepare_training":
                return await self.process_mcp_request(
                    "get_training_features",
                    {
                        "symbols": message.get("symbols", []),
                        "start_date": message.get("start_date"),
                        "end_date": message.get("end_date"),
                    },
                )

            elif message_type == "feature_metadata":
                return await self.process_mcp_request(
                    "get_feature_definitions", {"feature_names": message.get("feature_names")}
                )

            elif message_type == "feature_importance":
                return await self.process_mcp_request("get_feature_importance", {})

            else:
                return await super().process_message(message)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"success": False, "error": str(e)}

    # ============= A2A Feature Integration MCP Tools =============

    async def _mcp_a2a_provide_features(self, feature_request: Dict[str, Any]) -> Dict[str, Any]:
        """Provide engineered features to requesting ML/strategy agents via A2A messaging"""
        try:
            requesting_agent = feature_request.get("requesting_agent")
            symbols = feature_request.get("symbols", [])
            feature_types = feature_request.get("feature_types", ["technical", "fundamental"])
            timeframe = feature_request.get("timeframe", "1h")
            lookback_days = feature_request.get("lookback_days", 30)

            logger.info(f"Processing A2A feature request from {requesting_agent}")

            # Compute requested features
            feature_result = await self._mcp_compute_features({
                "symbols": symbols,
                "feature_types": feature_types,
                "timeframe": timeframe,
                "lookback_days": lookback_days
            })

            if feature_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": f"Feature computation failed: {feature_result.get('error')}"
                }

            # Prepare feature metadata
            metadata = await self._mcp_get_feature_metadata({
                "feature_names": list(feature_result.get("features", {}).keys())
            })

            return {
                "status": "success",
                "features_provided": feature_result.get("features", {}),
                "feature_metadata": metadata.get("feature_definitions", {}),
                "symbols_covered": symbols,
                "timeframe": timeframe,
                "feature_count": len(feature_result.get("features", {})),
                "requesting_agent": requesting_agent,
                "computed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"A2A feature provision failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _mcp_a2a_engineer_custom_features(self, engineering_request: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer custom features based on specific agent requirements via A2A messaging"""
        try:
            requesting_agent = engineering_request.get("requesting_agent")
            custom_requirements = engineering_request.get("requirements", {})
            base_data = engineering_request.get("base_data", {})
            target_variable = engineering_request.get("target_variable")

            logger.info(f"Engineering custom features for {requesting_agent}")

            # Extract requirements
            feature_window = custom_requirements.get("window_size", 20)
            interaction_features = custom_requirements.get("create_interactions", False)
            polynomial_degree = custom_requirements.get("polynomial_degree", 1)
            lag_features = custom_requirements.get("lag_periods", [1, 5, 10])

            # Generate custom feature specifications
            custom_features = {}

            # Technical indicator variations
            if "technical_variations" in custom_requirements:
                for symbol in base_data.keys():
                    custom_features[f"{symbol}_custom_momentum"] = {
                        "formula": f"(price_close - sma_{feature_window}) / atr_{feature_window}",
                        "type": "momentum_normalized",
                        "lookback": feature_window
                    }
                    
                    custom_features[f"{symbol}_volatility_regime"] = {
                        "formula": f"rolling_std_{feature_window} / rolling_std_{feature_window*4}",
                        "type": "volatility_regime",
                        "lookback": feature_window * 4
                    }

            # Interaction features
            if interaction_features and len(base_data) > 1:
                symbols = list(base_data.keys())
                for i, sym1 in enumerate(symbols):
                    for sym2 in symbols[i+1:]:
                        custom_features[f"correlation_{sym1}_{sym2}"] = {
                            "formula": f"rolling_correlation({sym1}_returns, {sym2}_returns, {feature_window})",
                            "type": "cross_asset_correlation",
                            "lookback": feature_window
                        }

            # Lag features
            for lag in lag_features:
                for symbol in base_data.keys():
                    custom_features[f"{symbol}_return_lag_{lag}"] = {
                        "formula": f"{symbol}_returns.shift({lag})",
                        "type": "lagged_return",
                        "lookback": lag
                    }

            return {
                "status": "success",
                "custom_features": custom_features,
                "feature_count": len(custom_features),
                "engineering_specs": {
                    "window_size": feature_window,
                    "interactions": interaction_features,
                    "polynomial_degree": polynomial_degree,
                    "lag_periods": lag_features
                },
                "requesting_agent": requesting_agent,
                "estimated_computation_time": len(custom_features) * 0.1  # seconds
            }

        except Exception as e:
            logger.error(f"A2A custom feature engineering failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _mcp_a2a_validate_feature_quality(self, validation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature quality for ML/strategy agents via A2A messaging"""
        try:
            requesting_agent = validation_request.get("requesting_agent")
            features = validation_request.get("features", {})
            target_variable = validation_request.get("target_variable")
            validation_metrics = validation_request.get("metrics", ["correlation", "mutual_information", "stability"])

            logger.info(f"Validating feature quality for {requesting_agent}")

            validation_results = {
                "feature_validations": {},
                "overall_quality": {},
                "recommendations": [],
                "flags": []
            }

            # Validate each feature
            for feature_name, feature_data in features.items():
                feature_validation = {
                    "name": feature_name,
                    "quality_scores": {},
                    "issues": [],
                    "statistics": {}
                }

                # Check for missing values
                if hasattr(feature_data, 'isnull'):
                    missing_pct = feature_data.isnull().mean() if len(feature_data) > 0 else 1.0
                else:
                    missing_pct = 0.0

                feature_validation["quality_scores"]["completeness"] = 1.0 - missing_pct
                if missing_pct > 0.1:
                    feature_validation["issues"].append(f"High missing values: {missing_pct:.2%}")
                    validation_results["recommendations"].append(f"Consider imputation for {feature_name}")

                # Check feature stability
                if "stability" in validation_metrics:
                    stability_score = self._calculate_feature_stability(feature_data)
                    feature_validation["quality_scores"]["stability"] = stability_score
                    if stability_score < 0.7:
                        feature_validation["issues"].append("Low stability across time periods")

                # Check predictive power (if target provided)
                if target_variable and "correlation" in validation_metrics:
                    correlation = self._calculate_feature_target_correlation(feature_data, target_variable)
                    feature_validation["quality_scores"]["correlation"] = abs(correlation)
                    if abs(correlation) < 0.05:
                        feature_validation["issues"].append("Low correlation with target variable")
                        validation_results["flags"].append(f"Consider removing {feature_name}")

                validation_results["feature_validations"][feature_name] = feature_validation

            # Calculate overall quality scores
            all_scores = []
            for fv in validation_results["feature_validations"].values():
                all_scores.extend(fv["quality_scores"].values())

            if all_scores:
                validation_results["overall_quality"] = {
                    "average_score": sum(all_scores) / len(all_scores),
                    "features_validated": len(features),
                    "high_quality_features": sum(1 for fv in validation_results["feature_validations"].values() 
                                                if all(score >= 0.7 for score in fv["quality_scores"].values())),
                    "quality_grade": "A" if sum(all_scores) / len(all_scores) >= 0.8 else "B"
                }

            return {
                "status": "success",
                "validation_results": validation_results,
                "requesting_agent": requesting_agent,
                "quality_approved": validation_results["overall_quality"].get("average_score", 0) >= 0.7
            }

        except Exception as e:
            logger.error(f"A2A feature quality validation failed: {e}")
            return {"status": "error", "error": str(e)}

    def _calculate_feature_stability(self, feature_data) -> float:
        """Calculate stability score for a feature across time periods"""
        try:
            if not hasattr(feature_data, 'rolling'):
                return 0.8  # Default for non-pandas data
            
            # Check stability using rolling statistics
            window_size = min(100, len(feature_data) // 4)
            if window_size < 10:
                return 0.8  # Not enough data
                
            rolling_mean = feature_data.rolling(window_size).mean()
            rolling_std = feature_data.rolling(window_size).std()
            
            # Coefficient of variation of rolling statistics
            mean_stability = 1.0 - (rolling_mean.std() / (abs(rolling_mean.mean()) + 1e-8))
            std_stability = 1.0 - (rolling_std.std() / (rolling_std.mean() + 1e-8))
            
            return max(0.0, min(1.0, (mean_stability + std_stability) / 2))
        except:
            return 0.5  # Conservative default

    def _calculate_feature_target_correlation(self, feature_data, target_data) -> float:
        """Calculate correlation between feature and target"""
        try:
            if hasattr(feature_data, 'corr'):
                return feature_data.corr(target_data)
            else:
                # Fallback for non-pandas data
                import numpy as np
                return np.corrcoef(feature_data, target_data)[0, 1] if len(feature_data) > 1 else 0.0
        except:
            return 0.0


# Global agent instance
feature_store_agent = FeatureStoreAgent()
