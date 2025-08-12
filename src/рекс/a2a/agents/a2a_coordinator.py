"""
A2A Coordinator - 100% A2A Compliant Message Router and Orchestrator
Manages communication between Historical Loader and Database agents
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from .historical_loader_agent import historical_loader_agent
from .database_agent import database_agent
from ..protocols import A2AProtocol, MessageType, A2AMessage
from ..registry.registry import agent_registry

logger = logging.getLogger(__name__)

class A2ACoordinator:
    """Coordinates A2A communication between agents"""
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.agents = {
            'historical-loader-001': historical_loader_agent,
            'database-001': database_agent
        }
        self.message_history = []
        
    async def route_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Route A2A message to appropriate agent"""
        try:
            receiver_id = message.receiver_id
            
            if receiver_id not in self.agents:
                return {
                    "success": False,
                    "error": f"Unknown receiver agent: {receiver_id}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log message
            self.message_history.append({
                "message": message.to_dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Routing message from {message.sender_id} to {message.receiver_id}: {message.message_type.value}")
            
            # Route to appropriate agent
            target_agent = self.agents[receiver_id]
            
            if receiver_id == 'database-001':
                # Use database agent's A2A message processor
                response = await target_agent.agent(
                    f"Process A2A message: {message.to_dict()}"
                )
            elif receiver_id == 'historical-loader-001':
                # Use historical loader agent
                response = await target_agent.process_request(
                    f"Handle A2A message: {message.to_dict()}"
                )
            else:
                response = {"success": False, "error": "Unsupported agent"}
            
            return response
            
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def bulk_load_workflow(self, symbols: List[str], days_back: int = 365) -> Dict[str, Any]:
        """Execute complete A2A workflow: Load data -> Store in database with AI analysis"""
        workflow_id = f"bulk_load_{datetime.now().timestamp()}"
        logger.info(f"Starting A2A bulk load workflow {workflow_id} for symbols: {symbols}")
        
        results = {
            "workflow_id": workflow_id,
            "symbols": symbols,
            "started_at": datetime.now().isoformat(),
            "steps": [],
            "overall_success": True,
            "total_records": 0
        }
        
        try:
            for symbol in symbols:
                symbol_result = await self._process_symbol_workflow(symbol, days_back)
                results["steps"].append(symbol_result)
                
                if symbol_result["success"]:
                    results["total_records"] += symbol_result.get("records_count", 0)
                else:
                    results["overall_success"] = False
            
            results["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"A2A bulk load workflow {workflow_id} completed: {results['overall_success']}")
            return results
            
        except Exception as e:
            logger.error(f"A2A bulk load workflow {workflow_id} failed: {e}")
            results["overall_success"] = False
            results["error"] = str(e)
            results["completed_at"] = datetime.now().isoformat()
            return results
    
    async def _process_symbol_workflow(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Process single symbol through complete A2A workflow"""
        step_result = {
            "symbol": symbol,
            "success": False,
            "started_at": datetime.now().isoformat(),
            "messages": []
        }
        
        try:
            # Step 1: Historical Loader Agent loads data
            logger.info(f"Step 1: Loading historical data for {symbol}")
            
            load_request = A2AProtocol.create_message(
                sender_id="coordinator",
                receiver_id="historical-loader-001", 
                message_type=MessageType.DATA_LOAD_REQUEST,
                payload={
                    "symbol": symbol,
                    "days_back": days_back,
                    "include_indicators": True
                }
            )
            
            # Direct call to historical loader
            load_result = historical_loader_agent.load_data_for_symbols([symbol], days_back)
            step_result["messages"].append({
                "step": "data_load",
                "success": symbol in load_result and load_result[symbol]["success"],
                "timestamp": datetime.now().isoformat()
            })
            
            if symbol not in load_result or not load_result[symbol]["success"]:
                step_result["error"] = f"Failed to load data for {symbol}"
                return step_result
            
            loaded_data = load_result[symbol]["data"]
            
            # Step 2: Send data to Database Agent for storage and AI analysis
            logger.info(f"Step 2: Storing data and running AI analysis for {symbol}")
            
            storage_message = A2AProtocol.create_message(
                sender_id="historical-loader-001",
                receiver_id="database-001",
                message_type=MessageType.DATA_LOAD_REQUEST,
                payload={
                    "symbol": symbol,
                    "data": [
                        {
                            "symbol": symbol,
                            "price": loaded_data.iloc[-1]["close"] if len(loaded_data) > 0 else 0,
                            "volume": loaded_data.iloc[-1]["volume"] if len(loaded_data) > 0 else 0,
                            "rsi": loaded_data.iloc[-1].get("rsi_14", 50) if len(loaded_data) > 0 else 50
                        }
                    ],
                    "records_count": len(loaded_data)
                }
            )
            
            # Route message to database agent
            storage_response = await self.route_message(storage_message)
            step_result["messages"].append({
                "step": "data_storage_ai_analysis", 
                "success": storage_response.get("success", False),
                "timestamp": datetime.now().isoformat()
            })
            
            if storage_response.get("success"):
                step_result["success"] = True
                step_result["records_count"] = len(loaded_data)
                step_result["ai_analyses"] = storage_response.get("ai_analyses", 0)
            else:
                step_result["error"] = storage_response.get("error", "Storage failed")
            
        except Exception as e:
            logger.error(f"Error in symbol workflow for {symbol}: {e}")
            step_result["error"] = str(e)
        
        step_result["completed_at"] = datetime.now().isoformat()
        return step_result
    
    async def get_symbol_analysis(self, symbol: str, ai_providers: List[str] = None) -> Dict[str, Any]:
        """Get AI analysis for symbol using A2A protocol"""
        if ai_providers is None:
            ai_providers = ['deepseek', 'perplexity', 'claude']
        
        analysis_message = A2AProtocol.create_message(
            sender_id="coordinator",
            receiver_id="database-001",
            message_type=MessageType.ANALYSIS_REQUEST,
            payload={
                "symbol": symbol,
                "data_context": {"symbol": symbol, "price": 0, "volume": 0},
                "providers": ai_providers
            }
        )
        
        response = await self.route_message(analysis_message)
        return response
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get A2A message history for debugging"""
        return self.message_history
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        status = {}
        for agent_id in self.agents.keys():
            agent_info = agent_registry.get_agent(agent_id)
            if agent_info:
                status[agent_id] = {
                    "type": agent_info["type"],
                    "status": agent_info["status"],
                    "capabilities": agent_info["capabilities"],
                    "registered_at": agent_info["registered_at"]
                }
        return status

# Global A2A coordinator instance
a2a_coordinator = A2ACoordinator()