"""
A2A Coordination Sub-skill for Calculation Agent

Handles coordination with other A2A agents for distributed calculations:
- Task distribution and workload balancing
- Result aggregation from multiple agents
- Fault tolerance and retry mechanisms
- Performance optimization through agent selection

This sub-skill is designed for future expansion when multiple calculation
agents are deployed across the A2A network.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncio

from src.strands.tools import tool
from .types import CoordinationTask, CalculationResult


logger = logging.getLogger(__name__)


class CoordinationSkill:
    """A2A coordination sub-skill for distributed calculations"""
    
    def __init__(self):
        self.active_coordination_sessions: Dict[str, Dict] = {}
        self.agent_performance_history: Dict[str, Dict] = {}
        self.coordination_stats = {
            "total_sessions": 0,
            "successful_coordinations": 0,
            "failed_coordinations": 0,
            "average_coordination_time": 0.0,
            "agents_coordinated_with": set()
        }
    
    @tool
    def create_coordination_session(self, calculation_request: Dict[str, Any], 
                                   target_agents: List[str],
                                   strategy: str = "redundant") -> Dict[str, Any]:
        """
        Create a new coordination session for distributed calculation
        
        Args:
            calculation_request: The calculation to distribute
            target_agents: List of agent IDs to coordinate with
            strategy: Coordination strategy (parallel, sequential, redundant)
            
        Returns:
            Dict with coordination session details
        """
        session_id = f"coord_{datetime.now().isoformat()}"
        start_time = time.time()
        
        try:
            session_data = {
                "session_id": session_id,
                "calculation_request": calculation_request,
                "target_agents": target_agents,
                "strategy": strategy,
                "status": "created",
                "tasks": [],
                "results": {},
                "start_time": start_time,
                "created_at": datetime.now().isoformat()
            }
            
            # Create tasks based on strategy
            if strategy == "parallel":
                # All agents work on same problem simultaneously
                for agent_id in target_agents:
                    task = CoordinationTask(
                        task_id=f"{session_id}_task_{agent_id}",
                        expression=calculation_request.get("expression", ""),
                        assigned_agent=agent_id,
                        task_type="full_calculation",
                        created_at=datetime.now()
                    )
                    session_data["tasks"].append(task)
                    
            elif strategy == "sequential":
                # Agents work in sequence, each building on previous result
                for i, agent_id in enumerate(target_agents):
                    task = CoordinationTask(
                        task_id=f"{session_id}_step_{i}_{agent_id}",
                        expression=calculation_request.get("expression", ""),
                        assigned_agent=agent_id,
                        task_type="sequential_step",
                        dependencies=[f"{session_id}_step_{i-1}"] if i > 0 else [],
                        created_at=datetime.now()
                    )
                    session_data["tasks"].append(task)
                    
            elif strategy == "redundant":
                # Multiple agents work on same problem for verification
                for agent_id in target_agents:
                    task = CoordinationTask(
                        task_id=f"{session_id}_redundant_{agent_id}",
                        expression=calculation_request.get("expression", ""),
                        assigned_agent=agent_id,
                        task_type="verification_calculation",
                        created_at=datetime.now()
                    )
                    session_data["tasks"].append(task)
            
            self.active_coordination_sessions[session_id] = session_data
            self.coordination_stats["total_sessions"] += 1
            self.coordination_stats["agents_coordinated_with"].update(target_agents)
            
            logger.info(f"Created coordination session {session_id} with {len(target_agents)} agents using {strategy} strategy")
            
            return {
                "success": True,
                "session_id": session_id,
                "strategy": strategy,
                "target_agents": target_agents,
                "tasks_created": len(session_data["tasks"]),
                "status": "ready_for_execution"
            }
            
        except Exception as e:
            logger.error(f"Coordination session creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    @tool
    def execute_coordination_session(self, session_id: str) -> Dict[str, Any]:
        """
        Execute a coordination session by distributing tasks to agents
        
        Args:
            session_id: ID of the coordination session to execute
            
        Returns:
            Dict with execution results
        """
        start_time = time.time()
        
        try:
            if session_id not in self.active_coordination_sessions:
                return {
                    "success": False,
                    "error": f"Coordination session {session_id} not found"
                }
            
            session = self.active_coordination_sessions[session_id]
            session["status"] = "executing"
            session["execution_start_time"] = start_time
            
            # This is a placeholder implementation
            # In a full A2A system, this would:
            # 1. Send A2A messages to target agents with task assignments
            # 2. Monitor task progress and handle agent responses
            # 3. Implement retry logic for failed tasks
            # 4. Aggregate results based on coordination strategy
            
            execution_results = {
                "session_id": session_id,
                "strategy": session["strategy"],
                "tasks_executed": len(session["tasks"]),
                "agents_contacted": session["target_agents"],
                "status": "simulation_mode",
                "note": "This is a placeholder - full A2A message coordination not implemented"
            }
            
            # Simulate task distribution and results collection
            simulated_results = self._simulate_task_execution(session)
            execution_results.update(simulated_results)
            
            # Update session status
            execution_time = time.time() - start_time
            session["status"] = "completed"
            session["execution_time"] = execution_time
            session["results"] = simulated_results
            
            # Update coordination statistics
            self._update_coordination_stats(True, execution_time)
            
            logger.info(f"Coordination session {session_id} execution completed")
            
            return {
                "success": True,
                "execution_time": execution_time,
                **execution_results
            }
            
        except Exception as e:
            logger.error(f"Coordination session execution failed: {e}")
            
            # Update failed coordination stats
            self._update_coordination_stats(False, time.time() - start_time)
            
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "execution_time": time.time() - start_time
            }
    
    @tool
    def get_coordination_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get status of a coordination session
        
        Args:
            session_id: ID of the coordination session
            
        Returns:
            Dict with session status and progress
        """
        try:
            if session_id not in self.active_coordination_sessions:
                return {
                    "success": False,
                    "error": f"Coordination session {session_id} not found"
                }
            
            session = self.active_coordination_sessions[session_id]
            
            # Calculate progress
            total_tasks = len(session["tasks"])
            completed_tasks = sum(1 for task in session["tasks"] if hasattr(task, 'status') and task.status == "completed")
            progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            status_info = {
                "session_id": session_id,
                "status": session["status"],
                "strategy": session["strategy"],
                "target_agents": session["target_agents"],
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "progress_percentage": progress_percentage,
                "created_at": session["created_at"],
                "execution_time": session.get("execution_time")
            }
            
            return {
                "success": True,
                **status_info
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    def aggregate_coordination_results(self, session_id: str, 
                                     aggregation_method: str = "consensus") -> Dict[str, Any]:
        """
        Aggregate results from coordinated agents
        
        Args:
            session_id: ID of the coordination session
            aggregation_method: How to combine results (consensus, average, best_confidence)
            
        Returns:
            Dict with aggregated result
        """
        try:
            if session_id not in self.active_coordination_sessions:
                return {
                    "success": False,
                    "error": f"Coordination session {session_id} not found"
                }
            
            session = self.active_coordination_sessions[session_id]
            results = session.get("results", {})
            
            if not results:
                return {
                    "success": False,
                    "error": "No results available for aggregation"
                }
            
            # Implement different aggregation methods
            if aggregation_method == "consensus":
                aggregated = self._aggregate_by_consensus(results)
            elif aggregation_method == "average":
                aggregated = self._aggregate_by_average(results)
            elif aggregation_method == "best_confidence":
                aggregated = self._aggregate_by_confidence(results)
            else:
                return {
                    "success": False,
                    "error": f"Unknown aggregation method: {aggregation_method}"
                }
            
            return {
                "success": True,
                "session_id": session_id,
                "aggregation_method": aggregation_method,
                "agents_contributed": len(results),
                "aggregated_result": aggregated,
                "individual_results": results
            }
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination performance statistics"""
        try:
            stats = self.coordination_stats.copy()
            stats["agents_coordinated_with"] = list(stats["agents_coordinated_with"])
            
            # Add agent performance summaries
            agent_performance = {}
            for agent_id, perf_data in self.agent_performance_history.items():
                agent_performance[agent_id] = {
                    "tasks_assigned": perf_data.get("tasks_assigned", 0),
                    "tasks_completed": perf_data.get("tasks_completed", 0),
                    "average_response_time": perf_data.get("average_response_time", 0.0),
                    "success_rate": perf_data.get("success_rate", 0.0)
                }
            
            stats["agent_performance"] = agent_performance
            
            return {
                "success": True,
                "coordination_statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _simulate_task_execution(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate task execution for demonstration purposes"""
        strategy = session["strategy"]
        tasks = session["tasks"]
        
        # Simulate results from different agents
        simulated_results = {}
        
        for task in tasks:
            agent_id = task.assigned_agent
            
            # Simulate calculation result from agent
            simulated_result = {
                "agent_id": agent_id,
                "task_id": task.task_id,
                "result": f"simulated_result_from_{agent_id}",
                "success": True,
                "computation_time": 0.1,
                "confidence": 0.9,
                "method": "simulated"
            }
            
            simulated_results[agent_id] = simulated_result
        
        return {
            "individual_results": simulated_results,
            "results_count": len(simulated_results),
            "all_successful": all(r["success"] for r in simulated_results.values())
        }
    
    def _aggregate_by_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results by consensus (majority vote)"""
        # Simplified consensus - in practice would need more sophisticated logic
        result_values = [r.get("result") for r in results.values() if r.get("success")]
        
        if not result_values:
            return {"error": "No successful results for consensus"}
        
        # Simple majority selection (could be enhanced)
        consensus_result = max(set(result_values), key=result_values.count) if result_values else None
        consensus_count = result_values.count(consensus_result) if consensus_result else 0
        
        return {
            "consensus_result": consensus_result,
            "consensus_count": consensus_count,
            "total_results": len(result_values),
            "consensus_percentage": (consensus_count / len(result_values) * 100) if result_values else 0
        }
    
    def _aggregate_by_average(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results by averaging (for numeric results)"""
        numeric_results = []
        
        for result in results.values():
            if result.get("success"):
                try:
                    # Try to convert result to float
                    numeric_val = float(result.get("result", 0))
                    numeric_results.append(numeric_val)
                except (ValueError, TypeError):
                    continue
        
        if not numeric_results:
            return {"error": "No numeric results available for averaging"}
        
        average_result = sum(numeric_results) / len(numeric_results)
        
        return {
            "average_result": average_result,
            "individual_values": numeric_results,
            "count": len(numeric_results),
            "std_deviation": (sum((x - average_result) ** 2 for x in numeric_results) / len(numeric_results)) ** 0.5
        }
    
    def _aggregate_by_confidence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Select result with highest confidence score"""
        best_result = None
        best_confidence = -1
        
        for agent_id, result in results.items():
            if result.get("success"):
                confidence = result.get("confidence", 0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = result
                    best_result["selected_agent"] = agent_id
        
        if not best_result:
            return {"error": "No successful results with confidence scores"}
        
        return {
            "best_result": best_result.get("result"),
            "confidence": best_confidence,
            "selected_agent": best_result.get("selected_agent"),
            "selection_reason": "highest_confidence"
        }
    
    def _update_coordination_stats(self, success: bool, execution_time: float):
        """Update coordination performance statistics"""
        if success:
            self.coordination_stats["successful_coordinations"] += 1
        else:
            self.coordination_stats["failed_coordinations"] += 1
        
        # Update average coordination time
        total_sessions = self.coordination_stats["total_sessions"]
        current_avg = self.coordination_stats["average_coordination_time"]
        new_avg = (current_avg * (total_sessions - 1) + execution_time) / total_sessions
        self.coordination_stats["average_coordination_time"] = new_avg