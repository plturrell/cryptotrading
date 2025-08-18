"""
Feature Store Agent - STRANDS Integration
Specialized agent for ML feature engineering and computation
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from ..base_agent import BaseAgent, AgentConfig
from ...ml.feature_store import feature_store
from ...infrastructure.mcp.feature_store_mcp_tools import feature_store_mcp_tools

logger = logging.getLogger(__name__)

class FeatureStoreAgent(BaseAgent):
    """STRANDS agent for feature store operations"""
    
    def __init__(self, agent_id: str = "feature_store_agent", **kwargs):
        """Initialize Feature Store Agent"""
        config = AgentConfig(
            agent_id=agent_id,
            agent_type="feature_store",
            description="ML feature engineering and computation agent",
            capabilities=[
                "compute_features", "get_feature_vector", "get_training_features",
                "get_feature_definitions", "get_feature_importance"
            ],
            max_concurrent_tools=3,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=30
        )
        
        super().__init__(
            agent_id=agent_id,
            agent_type="feature_store",
            config=config,
            **kwargs
        )
        
        self.feature_store = feature_store
        self.mcp_tools = feature_store_mcp_tools
        
        # Register with A2A protocol
        from ...protocols.a2a.a2a_protocol import A2A_CAPABILITIES, A2AAgentRegistry
        capabilities = A2A_CAPABILITIES.get(agent_id, [])
        A2AAgentRegistry.register_agent(agent_id, capabilities, self)
        
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
                input_schema=tool_def["inputSchema"]
            )
    
    async def initialize(self) -> bool:
        """Initialize the Feature Store Agent"""
        try:
            logger.info(f"Initializing Feature Store Agent {self.agent_id}")
            
            # Test feature store connectivity
            available_features = list(self.feature_store.features.keys())
            logger.info(f"Feature Store has {len(available_features)} features available")
            
            # Quick validation test
            try:
                importance = self.feature_store.get_feature_importance()
                logger.info(f"Feature importance loaded: {len(importance)} features")
            except Exception as e:
                logger.warning(f"Feature importance test failed: {e}")
            
            logger.info(f"Feature Store Agent {self.agent_id} initialized successfully")
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
    
    async def compute_symbol_features(self, symbol: str, features: List[str] = None) -> Dict[str, Any]:
        """Compute features for a specific symbol"""
        try:
            result = await self.execute_tool("compute_features", {
                "symbol": symbol,
                "features": features
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_latest_feature_vector(self, symbol: str) -> Dict[str, Any]:
        """Get latest feature vector for a symbol"""
        try:
            result = await self.execute_tool("get_feature_vector", {
                "symbol": symbol
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting feature vector for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    async def prepare_training_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Prepare training data for multiple symbols"""
        try:
            result = await self.execute_tool("get_training_features", {
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_feature_metadata(self, feature_names: List[str] = None) -> Dict[str, Any]:
        """Get feature definitions and metadata"""
        try:
            result = await self.execute_tool("get_feature_definitions", {
                "feature_names": feature_names
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting feature metadata: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance analysis"""
        try:
            result = await self.execute_tool("get_feature_importance", {})
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages with feature store operations"""
        try:
            message_type = message.get("type", "unknown")
            
            if message_type == "compute_features":
                symbol = message.get("symbol")
                features = message.get("features")
                return await self.compute_symbol_features(symbol, features)
                
            elif message_type == "get_feature_vector":
                symbol = message.get("symbol")
                return await self.get_latest_feature_vector(symbol)
                
            elif message_type == "prepare_training":
                symbols = message.get("symbols", [])
                start_date = message.get("start_date")
                end_date = message.get("end_date")
                return await self.prepare_training_data(symbols, start_date, end_date)
                
            elif message_type == "feature_metadata":
                feature_names = message.get("feature_names")
                return await self.get_feature_metadata(feature_names)
                
            elif message_type == "feature_importance":
                return await self.analyze_feature_importance()
                
            else:
                return await super().process_message(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"success": False, "error": str(e)}

# Global agent instance
feature_store_agent = FeatureStoreAgent()
