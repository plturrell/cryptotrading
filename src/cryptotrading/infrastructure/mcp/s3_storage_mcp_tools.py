"""
S3 Storage MCP Tools for All Strands Agents
Provides comprehensive S3 storage capabilities for logging, data storage, and agent operations
"""

import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import traceback

# Import our S3 services
from ..storage.s3_storage_service import S3StorageService
from ..storage.crypto_data_manager import CryptoDataManager
from ..analysis.mcp_agent_segregation import (
    AgentContext,
    ResourceType,
    get_segregation_manager,
    require_agent_auth,
)

logger = logging.getLogger(__name__)

class S3StorageMCPTools:
    """
    Comprehensive S3 storage MCP tools for all agents
    """
    
    def __init__(self):
        self.name = "s3_storage_tools"
        self.description = "S3 storage and logging tools for Strands agents"
        self._s3_service = None
        self._data_manager = None
    
    def _get_s3_service(self) -> Optional[S3StorageService]:
        """Get or initialize S3 service"""
        if self._s3_service is None:
            try:
                self._s3_service = S3StorageService()
                logger.info("S3 service initialized for MCP tools")
            except Exception as e:
                logger.error(f"Failed to initialize S3 service: {e}")
                return None
        return self._s3_service
    
    def _get_data_manager(self) -> Optional[CryptoDataManager]:
        """Get or initialize crypto data manager"""
        if self._data_manager is None:
            s3_service = self._get_s3_service()
            if s3_service:
                self._data_manager = CryptoDataManager(s3_service)
                logger.info("Crypto data manager initialized for MCP tools")
        return self._data_manager
    
    @require_agent_auth(ResourceType.STORAGE)
    async def log_agent_activity(
        self, parameters: Dict[str, Any], agent_context: AgentContext = None
    ) -> Dict[str, Any]:
        """
        Log agent activity to S3
        
        Parameters:
        - agent_id: Agent identifier
        - activity_type: Type of activity (calculation, analysis, trade, etc.)
        - message: Log message
        - level: Log level (info, warning, error, debug)
        - data: Additional structured data
        """
        try:
            s3_service = self._get_s3_service()
            if not s3_service:
                return {"success": False, "error": "S3 service not available"}
            
            agent_id = parameters.get("agent_id", "unknown")
            activity_type = parameters.get("activity_type", "general")
            message = parameters.get("message", "")
            level = parameters.get("level", "info")
            additional_data = parameters.get("data", {})
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "agent_context": {
                    "tenant_id": agent_context.tenant_id if agent_context else "default",
                    "session_id": agent_context.session_id if agent_context else None
                },
                "activity_type": activity_type,
                "level": level,
                "message": message,
                "data": additional_data
            }
            
            # Organize by agent and date
            date_str = datetime.utcnow().strftime('%Y/%m/%d')
            s3_key = f"agent-logs/{agent_id}/{activity_type}/{date_str}/{datetime.utcnow().isoformat()}.json"
            
            # Add metadata
            metadata = {
                'agent_id': agent_id,
                'activity_type': activity_type,
                'level': level,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            success = s3_service.upload_data(
                data=json.dumps(log_entry, indent=2),
                s3_key=s3_key,
                metadata=metadata,
                content_type='application/json'
            )
            
            return {
                "success": success,
                "s3_key": s3_key,
                "message": "Activity logged successfully" if success else "Failed to log activity"
            }
            
        except Exception as e:
            logger.error(f"Agent activity logging failed: {e}")
            return {"success": False, "error": str(e)}
    
    @require_agent_auth(ResourceType.STORAGE)
    async def store_agent_data(
        self, parameters: Dict[str, Any], agent_context: AgentContext = None
    ) -> Dict[str, Any]:
        """
        Store structured agent data in S3
        
        Parameters:
        - agent_id: Agent identifier
        - data_type: Type of data (calculation, analysis, model, etc.)
        - data: Data to store
        - metadata: Additional metadata
        """
        try:
            s3_service = self._get_s3_service()
            if not s3_service:
                return {"success": False, "error": "S3 service not available"}
            
            agent_id = parameters.get("agent_id", "unknown")
            data_type = parameters.get("data_type", "general")
            data = parameters.get("data", {})
            metadata = parameters.get("metadata", {})
            
            # Create data payload
            data_payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "data_type": data_type,
                "agent_context": {
                    "tenant_id": agent_context.tenant_id if agent_context else "default",
                    "session_id": agent_context.session_id if agent_context else None
                },
                "data": data,
                "metadata": metadata
            }
            
            # Organize by agent and data type
            date_str = datetime.utcnow().strftime('%Y/%m/%d')
            s3_key = f"agent-data/{agent_id}/{data_type}/{date_str}/{datetime.utcnow().isoformat()}.json"
            
            # Enhanced metadata
            s3_metadata = {
                'agent_id': agent_id,
                'data_type': data_type,
                'timestamp': datetime.utcnow().isoformat(),
                **metadata
            }
            
            success = s3_service.upload_data(
                data=json.dumps(data_payload, indent=2, default=str),
                s3_key=s3_key,
                metadata=s3_metadata,
                content_type='application/json'
            )
            
            return {
                "success": success,
                "s3_key": s3_key,
                "message": "Data stored successfully" if success else "Failed to store data"
            }
            
        except Exception as e:
            logger.error(f"Agent data storage failed: {e}")
            return {"success": False, "error": str(e)}
    
    @require_agent_auth(ResourceType.STORAGE)
    async def store_calculation_result(
        self, parameters: Dict[str, Any], agent_context: AgentContext = None
    ) -> Dict[str, Any]:
        """
        Store calculation results from MCTS or other calculation agents
        
        Parameters:
        - agent_id: Calculation agent ID
        - calculation_type: Type of calculation (mcts, technical_analysis, ml_model, etc.)
        - input_parameters: Input parameters used
        - result: Calculation result
        - execution_time: Time taken for calculation
        - confidence: Confidence score if applicable
        """
        try:
            data_manager = self._get_data_manager()
            if not data_manager:
                return {"success": False, "error": "Data manager not available"}
            
            calculation_data = {
                "agent_id": parameters.get("agent_id"),
                "calculation_type": parameters.get("calculation_type"),
                "input_parameters": parameters.get("input_parameters", {}),
                "result": parameters.get("result", {}),
                "execution_time_ms": parameters.get("execution_time"),
                "confidence_score": parameters.get("confidence"),
                "timestamp": datetime.utcnow().isoformat(),
                "session_context": {
                    "tenant_id": agent_context.tenant_id if agent_context else "default",
                    "session_id": agent_context.session_id if agent_context else None
                }
            }
            
            success = data_manager.save_analytics_report(
                report_type=f"calculation_{parameters.get('calculation_type', 'general')}",
                analysis_data=calculation_data,
                timestamp=datetime.utcnow()
            )
            
            return {
                "success": success,
                "message": "Calculation result stored" if success else "Failed to store calculation"
            }
            
        except Exception as e:
            logger.error(f"Calculation result storage failed: {e}")
            return {"success": False, "error": str(e)}
    
    @require_agent_auth(ResourceType.STORAGE)
    async def store_market_analysis(
        self, parameters: Dict[str, Any], agent_context: AgentContext = None
    ) -> Dict[str, Any]:
        """
        Store market analysis results
        
        Parameters:
        - agent_id: Analysis agent ID
        - symbol: Trading symbol analyzed
        - analysis_type: Type of analysis (technical, fundamental, sentiment)
        - indicators: Technical indicators if applicable
        - signals: Trading signals generated
        - recommendation: Trading recommendation
        - timeframe: Analysis timeframe
        """
        try:
            data_manager = self._get_data_manager()
            if not data_manager:
                return {"success": False, "error": "Data manager not available"}
            
            analysis_data = {
                "agent_id": parameters.get("agent_id"),
                "symbol": parameters.get("symbol"),
                "analysis_type": parameters.get("analysis_type"),
                "timeframe": parameters.get("timeframe", "1h"),
                "indicators": parameters.get("indicators", {}),
                "signals": parameters.get("signals", {}),
                "recommendation": parameters.get("recommendation"),
                "confidence": parameters.get("confidence"),
                "timestamp": datetime.utcnow().isoformat(),
                "agent_context": {
                    "tenant_id": agent_context.tenant_id if agent_context else "default"
                }
            }
            
            success = data_manager.save_analytics_report(
                report_type=f"market_analysis_{parameters.get('analysis_type', 'general')}",
                analysis_data=analysis_data,
                timestamp=datetime.utcnow()
            )
            
            return {
                "success": success,
                "message": "Market analysis stored" if success else "Failed to store analysis"
            }
            
        except Exception as e:
            logger.error(f"Market analysis storage failed: {e}")
            return {"success": False, "error": str(e)}
    
    @require_agent_auth(ResourceType.STORAGE)
    async def backup_agent_state(
        self, parameters: Dict[str, Any], agent_context: AgentContext = None
    ) -> Dict[str, Any]:
        """
        Backup agent state and configuration
        
        Parameters:
        - agent_id: Agent identifier
        - state_data: Agent state to backup
        - configuration: Agent configuration
        - version: State version
        """
        try:
            s3_service = self._get_s3_service()
            if not s3_service:
                return {"success": False, "error": "S3 service not available"}
            
            agent_id = parameters.get("agent_id")
            state_data = parameters.get("state_data", {})
            configuration = parameters.get("configuration", {})
            version = parameters.get("version", "1.0")
            
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "version": version,
                "state_data": state_data,
                "configuration": configuration,
                "backup_type": "agent_state"
            }
            
            # Store in backup folder with versioning
            date_str = datetime.utcnow().strftime('%Y/%m/%d')
            s3_key = f"backups/agents/{agent_id}/{date_str}/state_v{version}_{datetime.utcnow().isoformat()}.json"
            
            metadata = {
                'agent_id': agent_id,
                'backup_type': 'agent_state',
                'version': version,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            success = s3_service.upload_data(
                data=json.dumps(backup_data, indent=2, default=str),
                s3_key=s3_key,
                metadata=metadata,
                content_type='application/json'
            )
            
            return {
                "success": success,
                "s3_key": s3_key,
                "message": "Agent state backed up" if success else "Backup failed"
            }
            
        except Exception as e:
            logger.error(f"Agent state backup failed: {e}")
            return {"success": False, "error": str(e)}
    
    @require_agent_auth(ResourceType.STORAGE)
    async def retrieve_agent_logs(
        self, parameters: Dict[str, Any], agent_context: AgentContext = None
    ) -> Dict[str, Any]:
        """
        Retrieve agent logs from S3
        
        Parameters:
        - agent_id: Agent identifier
        - activity_type: Filter by activity type (optional)
        - start_date: Start date for logs (YYYY-MM-DD)
        - end_date: End date for logs (YYYY-MM-DD)  
        - limit: Maximum number of logs to retrieve
        """
        try:
            s3_service = self._get_s3_service()
            if not s3_service:
                return {"success": False, "error": "S3 service not available"}
            
            agent_id = parameters.get("agent_id")
            activity_type = parameters.get("activity_type", "")
            limit = parameters.get("limit", 100)
            
            # Build search prefix
            if activity_type:
                prefix = f"agent-logs/{agent_id}/{activity_type}/"
            else:
                prefix = f"agent-logs/{agent_id}/"
            
            # List objects
            objects = s3_service.list_objects(prefix=prefix, max_keys=limit)
            
            logs = []
            for obj in objects:
                try:
                    log_data = s3_service.get_object_data(obj['key'])
                    if log_data:
                        log_entry = json.loads(log_data.decode('utf-8'))
                        logs.append({
                            "s3_key": obj['key'],
                            "size": obj['size'],
                            "last_modified": obj['last_modified'].isoformat(),
                            "log_data": log_entry
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse log {obj['key']}: {e}")
            
            return {
                "success": True,
                "logs": logs,
                "total_found": len(logs)
            }
            
        except Exception as e:
            logger.error(f"Agent log retrieval failed: {e}")
            return {"success": False, "error": str(e)}
    
    @require_agent_auth(ResourceType.STORAGE)
    async def get_storage_stats(
        self, parameters: Dict[str, Any], agent_context: AgentContext = None
    ) -> Dict[str, Any]:
        """
        Get storage statistics for agent data
        
        Parameters:
        - agent_id: Specific agent ID (optional, if not provided gets all stats)
        """
        try:
            data_manager = self._get_data_manager()
            if not data_manager:
                return {"success": False, "error": "Data manager not available"}
            
            agent_id = parameters.get("agent_id")
            
            # Get overall storage stats
            stats = data_manager.get_storage_stats()
            
            if agent_id:
                # Filter stats for specific agent
                s3_service = self._get_s3_service()
                agent_objects = s3_service.list_objects(prefix=f"agent-logs/{agent_id}/")
                agent_data_objects = s3_service.list_objects(prefix=f"agent-data/{agent_id}/")
                
                agent_stats = {
                    "agent_id": agent_id,
                    "log_files": len(agent_objects),
                    "data_files": len(agent_data_objects),
                    "total_files": len(agent_objects) + len(agent_data_objects),
                    "log_size_bytes": sum(obj['size'] for obj in agent_objects),
                    "data_size_bytes": sum(obj['size'] for obj in agent_data_objects)
                }
                
                return {"success": True, "agent_stats": agent_stats, "overall_stats": stats}
            else:
                return {"success": True, "overall_stats": stats}
                
        except Exception as e:
            logger.error(f"Storage stats retrieval failed: {e}")
            return {"success": False, "error": str(e)}

# Tool registration for MCP
def get_s3_storage_tools() -> List[Dict[str, Any]]:
    """Get all S3 storage MCP tools"""
    
    s3_tools = S3StorageMCPTools()
    
    return [
        {
            "name": "log_agent_activity",
            "description": "Log agent activity and operations to S3 storage",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent identifier"},
                    "activity_type": {"type": "string", "description": "Type of activity"},
                    "message": {"type": "string", "description": "Log message"},
                    "level": {"type": "string", "description": "Log level", "enum": ["debug", "info", "warning", "error"]},
                    "data": {"type": "object", "description": "Additional structured data"}
                },
                "required": ["agent_id", "message"]
            },
            "handler": s3_tools.log_agent_activity
        },
        {
            "name": "store_agent_data",
            "description": "Store structured agent data in S3",
            "inputSchema": {
                "type": "object", 
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent identifier"},
                    "data_type": {"type": "string", "description": "Type of data"},
                    "data": {"type": "object", "description": "Data to store"},
                    "metadata": {"type": "object", "description": "Additional metadata"}
                },
                "required": ["agent_id", "data_type", "data"]
            },
            "handler": s3_tools.store_agent_data
        },
        {
            "name": "store_calculation_result",
            "description": "Store calculation results from MCTS or other calculation agents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Calculation agent ID"},
                    "calculation_type": {"type": "string", "description": "Type of calculation"},
                    "input_parameters": {"type": "object", "description": "Input parameters used"},
                    "result": {"type": "object", "description": "Calculation result"},
                    "execution_time": {"type": "number", "description": "Execution time in milliseconds"},
                    "confidence": {"type": "number", "description": "Confidence score"}
                },
                "required": ["agent_id", "calculation_type", "result"]
            },
            "handler": s3_tools.store_calculation_result
        },
        {
            "name": "store_market_analysis",
            "description": "Store market analysis results",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Analysis agent ID"},
                    "symbol": {"type": "string", "description": "Trading symbol"},
                    "analysis_type": {"type": "string", "description": "Type of analysis"},
                    "indicators": {"type": "object", "description": "Technical indicators"},
                    "signals": {"type": "object", "description": "Trading signals"},
                    "recommendation": {"type": "string", "description": "Trading recommendation"},
                    "timeframe": {"type": "string", "description": "Analysis timeframe"}
                },
                "required": ["agent_id", "symbol", "analysis_type"]
            },
            "handler": s3_tools.store_market_analysis
        },
        {
            "name": "backup_agent_state",
            "description": "Backup agent state and configuration",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent identifier"},
                    "state_data": {"type": "object", "description": "Agent state data"},
                    "configuration": {"type": "object", "description": "Agent configuration"},
                    "version": {"type": "string", "description": "State version"}
                },
                "required": ["agent_id", "state_data"]
            },
            "handler": s3_tools.backup_agent_state
        },
        {
            "name": "retrieve_agent_logs",
            "description": "Retrieve agent logs from S3 storage", 
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent identifier"},
                    "activity_type": {"type": "string", "description": "Filter by activity type"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                    "limit": {"type": "integer", "description": "Max logs to retrieve"}
                },
                "required": ["agent_id"]
            },
            "handler": s3_tools.retrieve_agent_logs
        },
        {
            "name": "get_storage_stats",
            "description": "Get storage statistics for agent data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Specific agent ID (optional)"}
                }
            },
            "handler": s3_tools.get_storage_stats
        }
    ]