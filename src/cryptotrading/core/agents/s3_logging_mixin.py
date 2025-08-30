"""
S3 Logging Mixin for All Strands Agents
Provides automatic S3 logging capabilities to any agent that inherits from it
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, Union
from abc import ABC

# Import S3 storage tools
try:
    from ...infrastructure.mcp.s3_storage_mcp_tools import S3StorageMCPTools
    from ...infrastructure.analysis.mcp_agent_segregation import AgentContext, ResourceType
    S3_AVAILABLE = True
except ImportError as e:
    S3_AVAILABLE = False

logger = logging.getLogger(__name__)

class S3LoggingMixin(ABC):
    """
    Mixin class that provides S3 logging capabilities to Strands agents
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._s3_logger = None
        self._agent_context = None
        self._init_s3_logging()
    
    def _init_s3_logging(self):
        """Initialize S3 logging if available"""
        if S3_AVAILABLE:
            try:
                self._s3_logger = S3StorageMCPTools()
                logger.info(f"S3 logging initialized for agent: {getattr(self, 'agent_id', 'unknown')}")
            except Exception as e:
                logger.warning(f"S3 logging initialization failed: {e}")
                self._s3_logger = None
        else:
            logger.info("S3 logging not available - storage tools not imported")
    
    def set_agent_context(self, agent_context: 'AgentContext'):
        """Set agent context for S3 logging"""
        self._agent_context = agent_context
    
    async def log_to_s3(
        self,
        message: str,
        level: str = "info",
        activity_type: str = "general",
        data: Dict[str, Any] = None,
        force_sync: bool = False
    ):
        """
        Log message to S3 storage
        
        Args:
            message: Log message
            level: Log level (debug, info, warning, error)
            activity_type: Type of activity being logged
            data: Additional structured data
            force_sync: Force synchronous logging (default: async)
        """
        if not self._s3_logger:
            return
        
        try:
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            
            log_params = {
                "agent_id": agent_id,
                "activity_type": activity_type,
                "message": message,
                "level": level,
                "data": data or {}
            }
            
            if force_sync:
                # Synchronous logging
                result = await self._s3_logger.log_agent_activity(
                    parameters=log_params,
                    agent_context=self._agent_context
                )
            else:
                # Asynchronous logging (fire and forget)
                asyncio.create_task(
                    self._s3_logger.log_agent_activity(
                        parameters=log_params,
                        agent_context=self._agent_context
                    )
                )
                result = {"success": True, "async": True}
            
            if not result.get("success"):
                logger.warning(f"S3 logging failed for {agent_id}: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"S3 logging error: {e}")
    
    async def store_calculation_result(
        self,
        calculation_type: str,
        input_parameters: Dict[str, Any],
        result: Dict[str, Any],
        execution_time: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """
        Store calculation results in S3
        
        Args:
            calculation_type: Type of calculation (mcts, technical_analysis, etc.)
            input_parameters: Parameters used for calculation
            result: Calculation result
            execution_time: Time taken in milliseconds
            confidence: Confidence score
        """
        if not self._s3_logger:
            return
        
        try:
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            
            params = {
                "agent_id": agent_id,
                "calculation_type": calculation_type,
                "input_parameters": input_parameters,
                "result": result,
                "execution_time": execution_time,
                "confidence": confidence
            }
            
            # Store calculation result asynchronously
            asyncio.create_task(
                self._s3_logger.store_calculation_result(
                    parameters=params,
                    agent_context=self._agent_context
                )
            )
            
        except Exception as e:
            logger.error(f"S3 calculation storage error: {e}")
    
    async def store_market_analysis(
        self,
        symbol: str,
        analysis_type: str,
        indicators: Dict[str, Any] = None,
        signals: Dict[str, Any] = None,
        recommendation: str = None,
        timeframe: str = "1h",
        confidence: float = None
    ):
        """
        Store market analysis results in S3
        
        Args:
            symbol: Trading symbol analyzed
            analysis_type: Type of analysis
            indicators: Technical indicators
            signals: Trading signals
            recommendation: Trading recommendation
            timeframe: Analysis timeframe
            confidence: Confidence score
        """
        if not self._s3_logger:
            return
        
        try:
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            
            params = {
                "agent_id": agent_id,
                "symbol": symbol,
                "analysis_type": analysis_type,
                "indicators": indicators or {},
                "signals": signals or {},
                "recommendation": recommendation,
                "timeframe": timeframe,
                "confidence": confidence
            }
            
            # Store analysis result asynchronously
            asyncio.create_task(
                self._s3_logger.store_market_analysis(
                    parameters=params,
                    agent_context=self._agent_context
                )
            )
            
        except Exception as e:
            logger.error(f"S3 market analysis storage error: {e}")
    
    async def backup_agent_state(
        self,
        state_data: Dict[str, Any],
        configuration: Dict[str, Any] = None,
        version: str = "1.0"
    ):
        """
        Backup agent state to S3
        
        Args:
            state_data: Current agent state
            configuration: Agent configuration
            version: State version
        """
        if not self._s3_logger:
            return
        
        try:
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            
            params = {
                "agent_id": agent_id,
                "state_data": state_data,
                "configuration": configuration or {},
                "version": version
            }
            
            result = await self._s3_logger.backup_agent_state(
                parameters=params,
                agent_context=self._agent_context
            )
            
            if result.get("success"):
                await self.log_to_s3(
                    f"Agent state backed up successfully",
                    level="info",
                    activity_type="backup",
                    data={"backup_version": version}
                )
            
            return result
            
        except Exception as e:
            logger.error(f"S3 agent state backup error: {e}")
            return {"success": False, "error": str(e)}
    
    async def log_agent_startup(self, configuration: Dict[str, Any] = None):
        """Log agent startup to S3"""
        await self.log_to_s3(
            "Agent started up successfully",
            level="info",
            activity_type="startup",
            data={
                "configuration": configuration or {},
                "startup_time": datetime.utcnow().isoformat()
            }
        )
    
    async def log_agent_shutdown(self, reason: str = "normal"):
        """Log agent shutdown to S3"""
        await self.log_to_s3(
            f"Agent shutting down: {reason}",
            level="info",
            activity_type="shutdown",
            data={
                "shutdown_reason": reason,
                "shutdown_time": datetime.utcnow().isoformat()
            },
            force_sync=True  # Force sync for shutdown logs
        )
    
    async def log_error(self, error: Exception, context: str = ""):
        """Log error to S3 with full context"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add traceback if available
        import traceback
        if hasattr(error, '__traceback__'):
            error_data["traceback"] = traceback.format_tb(error.__traceback__)
        
        await self.log_to_s3(
            f"Error in {context}: {str(error)}",
            level="error",
            activity_type="error",
            data=error_data
        )
    
    async def log_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        additional_metrics: Dict[str, Any] = None
    ):
        """Log performance metrics to S3"""
        perf_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat(),
            **(additional_metrics or {})
        }
        
        await self.log_to_s3(
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            level="info",
            activity_type="performance",
            data=perf_data
        )
    
    async def log_decision(
        self,
        decision: str,
        reasoning: str,
        confidence: float = None,
        input_data: Dict[str, Any] = None
    ):
        """Log agent decisions to S3"""
        decision_data = {
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence,
            "input_data": input_data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.log_to_s3(
            f"Decision made: {decision}",
            level="info",
            activity_type="decision",
            data=decision_data
        )
    
    async def log_interaction(
        self,
        interaction_type: str,
        with_agent: str = None,
        with_user: str = None,
        message: str = "",
        data: Dict[str, Any] = None
    ):
        """Log agent interactions to S3"""
        interaction_data = {
            "interaction_type": interaction_type,
            "with_agent": with_agent,
            "with_user": with_user,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.log_to_s3(
            f"Interaction: {interaction_type} with {with_agent or with_user or 'unknown'}",
            level="info",
            activity_type="interaction",
            data=interaction_data
        )


class S3LoggingAgent(S3LoggingMixin):
    """
    Base agent class with S3 logging capabilities
    Agents can inherit from this to get automatic S3 logging
    """
    
    def __init__(self, agent_id: str, **kwargs):
        self.agent_id = agent_id
        super().__init__(**kwargs)
        
        # Log initialization
        asyncio.create_task(self.log_agent_startup())
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            # Attempt to log shutdown (may not work if event loop is closed)
            if hasattr(self, '_s3_logger') and self._s3_logger:
                asyncio.create_task(self.log_agent_shutdown("destructor"))
        except:
            pass  # Ignore errors during cleanup


# Decorator for automatically logging method calls
def log_agent_method(activity_type: str = None, log_params: bool = False, log_result: bool = False):
    """
    Decorator to automatically log agent method calls to S3
    
    Args:
        activity_type: Override activity type for logging
        log_params: Whether to log method parameters
        log_result: Whether to log method result
    """
    def decorator(func):
        async def async_wrapper(self, *args, **kwargs):
            if not hasattr(self, 'log_to_s3'):
                # Not an S3 logging agent, just execute normally
                return await func(self, *args, **kwargs)
            
            method_name = func.__name__
            activity = activity_type or method_name
            start_time = datetime.utcnow()
            
            log_data = {
                "method": method_name,
                "start_time": start_time.isoformat()
            }
            
            if log_params:
                log_data["parameters"] = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()}
                }
            
            try:
                # Log method start
                await self.log_to_s3(
                    f"Method {method_name} started",
                    level="debug",
                    activity_type=activity,
                    data=log_data
                )
                
                # Execute method
                result = await func(self, *args, **kwargs)
                
                # Log method completion
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds() * 1000
                
                completion_data = {
                    **log_data,
                    "end_time": end_time.isoformat(),
                    "duration_ms": duration,
                    "status": "completed"
                }
                
                if log_result:
                    completion_data["result"] = str(result)
                
                await self.log_to_s3(
                    f"Method {method_name} completed in {duration:.2f}ms",
                    level="debug", 
                    activity_type=activity,
                    data=completion_data
                )
                
                return result
                
            except Exception as e:
                # Log method error
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds() * 1000
                
                error_data = {
                    **log_data,
                    "end_time": end_time.isoformat(),
                    "duration_ms": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
                await self.log_to_s3(
                    f"Method {method_name} failed: {str(e)}",
                    level="error",
                    activity_type=activity,
                    data=error_data
                )
                
                raise
        
        def sync_wrapper(self, *args, **kwargs):
            # For sync methods, just execute normally
            # Could add sync logging here if needed
            return func(self, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator