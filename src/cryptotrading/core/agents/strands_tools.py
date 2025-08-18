"""
Comprehensive Strands Native Tools Ecosystem
Production-grade tools with advanced capabilities for crypto trading platform.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .strands_orchestrator import strand_tool, ToolPriority, EnhancedStrandsAgent

class StrandsToolsAgent(EnhancedStrandsAgent):
    """
    Comprehensive Strands Tools Agent with 50+ native tools
    Covers: Trading, Analysis, Risk Management, Portfolio Management, 
    Data Processing, Communication, Monitoring, and System Operations
    """

    # TRADING TOOLS
    @strand_tool(
        name="advanced_market_scanner",
        description="Advanced market scanning with multiple criteria",
        priority=ToolPriority.HIGH,
        timeout=45.0,
        tags=["trading", "market", "scanning"]
    )
    async def advanced_market_scanner(self, criteria: Dict[str, Any] = None, 
                                    markets: List[str] = None) -> Dict[str, Any]:
        """Comprehensive market scanning with advanced filtering"""
        criteria = criteria or {"min_volume": 1000000, "volatility_range": [0.02, 0.15]}
        markets = markets or ["BTC", "ETH", "ADA", "DOT", "LINK"]
        
        scan_results = []
        for symbol in markets:
            # Get market data
            market_data = await self.execute_tool("get_market_data", {"symbol": symbol})
            
            if market_data.get("success"):
                data = market_data["result"]
                
                # Apply criteria
                meets_criteria = True
                if data.get("volume_24h", 0) < criteria.get("min_volume", 0):
                    meets_criteria = False
                
                if meets_criteria:
                    scan_results.append({
                        "symbol": symbol,
                        "price": data.get("current_price", 0),
                        "volume": data.get("volume_24h", 0),
                        "change_24h": data.get("change_24h", 0),
                        "score": self._calculate_opportunity_score(data)
                    })
        
        # Sort by score
        scan_results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "scan_results": scan_results,
            "criteria_used": criteria,
            "markets_scanned": len(markets),
            "opportunities_found": len(scan_results)
        }
    
    
    @strand_tool(
        name="multi_timeframe_analysis",
        description="Analyze trends across multiple timeframes",
        priority=ToolPriority.HIGH,
        timeout=60.0,
        tags=["analysis", "technical", "trends"]
    )
    async def multi_timeframe_analysis(self, symbol: str, 
                                     timeframes: List[str] = None) -> Dict[str, Any]:
        """Comprehensive multi-timeframe trend analysis"""
        timeframes = timeframes or ["1h", "4h", "1d", "1w"]
        analysis_results = {}
        
        for tf in timeframes:
            market_data = await self.execute_tool("get_market_data", {
                "symbol": symbol, 
                "timeframe": tf
            })
            
            if market_data.get("success"):
                data = market_data["result"]
                
                # Simple trend analysis
                trend = "neutral"
                if data.get("change_24h", 0) > 2:
                    trend = "bullish"
                elif data.get("change_24h", 0) < -2:
                    trend = "bearish"
                
                analysis_results[tf] = {
                    "trend": trend,
                    "price": data.get("current_price", 0),
                    "change": data.get("change_24h", 0),
                    "volume": data.get("volume_24h", 0)
                }
        
        # Overall consensus
        bullish_count = sum(1 for r in analysis_results.values() if r["trend"] == "bullish")
        bearish_count = sum(1 for r in analysis_results.values() if r["trend"] == "bearish")
        
        overall_trend = "neutral"
        if bullish_count > bearish_count:
            overall_trend = "bullish"
        elif bearish_count > bullish_count:
            overall_trend = "bearish"
        
        return {
            "symbol": symbol,
            "timeframe_analysis": analysis_results,
            "overall_trend": overall_trend,
            "trend_strength": abs(bullish_count - bearish_count) / len(timeframes),
            "consensus_level": max(bullish_count, bearish_count) / len(timeframes)
        }
    
    
    
    # DATA PROCESSING TOOLS
    @strand_tool(
        name="data_aggregation_engine",
        description="Aggregate and process data from multiple sources",
        priority=ToolPriority.NORMAL,
        timeout=120.0,
        tags=["data", "aggregation", "processing"]
    )
    async def data_aggregation_engine(self, symbols: List[str], 
                                    data_types: List[str] = None,
                                    time_range: str = "24h") -> Dict[str, Any]:
        """Comprehensive data aggregation from multiple sources"""
        data_types = data_types or ["market_data", "sentiment", "volume"]
        aggregated_data = {}
        
        for symbol in symbols:
            symbol_data = {}
            
            # Gather different types of data
            for data_type in data_types:
                try:
                    if data_type == "market_data":
                        result = await self.execute_tool("get_market_data", {"symbol": symbol})
                    elif data_type == "sentiment":
                        result = await self.execute_tool("analyze_sentiment", {"symbol": symbol})
                    elif data_type == "volume":
                        result = await self.execute_tool("get_market_data", {"symbol": symbol})
                        # Extract volume-specific data
                        if result.get("success"):
                            result["result"] = {"volume_24h": result["result"].get("volume_24h", 0)}
                    
                    if result.get("success"):
                        symbol_data[data_type] = result["result"]
                    else:
                        symbol_data[data_type] = {"error": "Data not available"}
                        
                except Exception as e:
                    symbol_data[data_type] = {"error": str(e)}
            
            aggregated_data[symbol] = symbol_data
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(aggregated_data)
        
        return {
            "aggregated_data": aggregated_data,
            "aggregate_metrics": aggregate_metrics,
            "data_sources": data_types,
            "symbols_processed": len(symbols),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # COMMUNICATION AND COORDINATION TOOLS
    @strand_tool(
        name="broadcast_to_network",
        description="Broadcast information to connected agent network",
        priority=ToolPriority.NORMAL,
        dependencies=["coordinate_agents"],
        tags=["communication", "broadcast", "network"]
    )
    async def broadcast_to_network(self, message_type: str, data: Dict[str, Any],
                                 priority: str = "normal") -> Dict[str, Any]:
        """Broadcast information to all connected agents"""
        if not self.enable_a2a or not self.connected_agents:
            return {"error": "No agents connected or A2A disabled"}
        
        broadcast_results = {}
        broadcast_id = str(uuid.uuid4())
        
        broadcast_message = {
            "broadcast_id": broadcast_id,
            "message_type": message_type,
            "data": data,
            "priority": priority,
            "sender": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all connected agents
        for agent_id in self.connected_agents.keys():
            try:
                result = await self.send_message_to_agent(agent_id, "broadcast", broadcast_message)
                broadcast_results[agent_id] = {"status": "delivered", "response": result}
            except Exception as e:
                broadcast_results[agent_id] = {"status": "failed", "error": str(e)}
        
        return {
            "broadcast_id": broadcast_id,
            "message_type": message_type,
            "agents_contacted": len(self.connected_agents),
            "delivery_results": broadcast_results,
            "success_rate": len([r for r in broadcast_results.values() if r["status"] == "delivered"]) / len(broadcast_results) if broadcast_results else 0
        }
    
    # MONITORING AND SYSTEM TOOLS
    @strand_tool(
        name="system_health_monitor",
        description="Comprehensive system health monitoring",
        priority=ToolPriority.NORMAL,
        tags=["monitoring", "health", "system"]
    )
    async def system_health_monitor(self) -> Dict[str, Any]:
        """Comprehensive system health monitoring"""
        health_data = {
            "agent_status": await self.health_check(),
            "strands_metrics": await self.get_strands_metrics(),
            "tool_performance": self._analyze_tool_performance(),
            "memory_usage": self._get_memory_usage(),
            "error_rates": self._calculate_error_rates(),
            "connectivity": self._check_connectivity()
        }
        
        # Overall health score
        health_score = self._calculate_health_score(health_data)
        health_data["overall_health_score"] = health_score
        health_data["health_status"] = self._get_health_status(health_score)
        
        return health_data
    
    # Helper methods
    async def _calculate_opportunity_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate opportunity score using MCP tools"""
        try:
            # Delegate to data analysis MCP tools
            from ...infrastructure.mcp.data_analysis_mcp_tools import data_analysis_mcp_tools
            
            result = await data_analysis_mcp_tools.handle_tool_call(
                "calculate_opportunity_score",
                {"market_data": market_data}
            )
            
            if result.get("success", False):
                return result["result"]
            else:
                # Fallback calculation
                return self._fallback_opportunity_score(market_data)
        except Exception:
            return self._fallback_opportunity_score(market_data)
    
    def _fallback_opportunity_score(self, market_data: Dict[str, Any]) -> float:
        """Fallback opportunity score calculation"""
        score = 0.0
        volume = market_data.get("volume_24h", 0)
        if volume > 10000000:
            score += 30
        elif volume > 1000000:
            score += 20
        elif volume > 100000:
            score += 10
        
        change = abs(market_data.get("change_24h", 0))
        if 2 <= change <= 8:
            score += 40
        elif change < 2:
            score += 20
        
        if market_data.get("change_24h", 0) > 0:
            score += 30
        
        return min(score, 100)
    
    def _calculate_risk_score(self, risk_data: Dict[str, Any]) -> float:
        """Calculate comprehensive risk score"""
        score = 50  # Base neutral score
        
        # Adjust based on VaR
        var = abs(risk_data.get("var_95", 0))
        if var > 10:
            score += 30  # High risk
        elif var > 5:
            score += 15  # Medium risk
        
        # Adjust based on Sharpe ratio
        sharpe = risk_data.get("sharpe_ratio", 0)
        if sharpe > 2:
            score -= 20  # Good risk-adjusted returns
        elif sharpe < 0.5:
            score += 20  # Poor risk-adjusted returns
        
        # Adjust based on max drawdown
        drawdown = abs(risk_data.get("max_drawdown", 0))
        if drawdown > 25:
            score += 25  # High drawdown risk
        elif drawdown > 15:
            score += 10  # Medium drawdown risk
        
        return max(0, min(100, score))
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 80:
            return "very_high"
        elif risk_score >= 65:
            return "high"
        elif risk_score >= 50:
            return "medium"
        elif risk_score >= 35:
            return "low"
        else:
            return "very_low"
    
    async def _perform_stress_test(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform portfolio stress testing"""
        scenarios = {
            "market_crash_20": {"market_change": -0.20},
            "market_crash_50": {"market_change": -0.50},
            "crypto_winter": {"market_change": -0.80},
            "flash_crash": {"market_change": -0.30}
        }
        
        stress_results = {}
        current_value = portfolio_data.get("total_value_usd", 0)
        
        for scenario_name, scenario in scenarios.items():
            market_change = scenario["market_change"]
            projected_value = current_value * (1 + market_change)
            loss_amount = current_value - projected_value
            
            stress_results[scenario_name] = {
                "projected_value": projected_value,
                "loss_amount": loss_amount,
                "loss_percentage": abs(market_change) * 100
            }
        
        return stress_results
    
    def _generate_risk_recommendations(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        risk_level = risk_assessment.get("risk_level", "medium")
        
        if risk_level in ["high", "very_high"]:
            recommendations.extend([
                "Consider reducing position sizes",
                "Implement stop-loss orders",
                "Diversify portfolio further",
                "Review correlation between assets"
            ])
        
        if risk_assessment.get("sharpe_ratio", 0) < 1:
            recommendations.append("Optimize risk-adjusted returns")
        
        if abs(risk_assessment.get("max_drawdown", 0)) > 20:
            recommendations.append("Implement stricter drawdown controls")
        
        return recommendations
    
    def _calculate_aggregate_metrics(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics across aggregated data"""
        all_prices = []
        all_volumes = []
        sentiment_scores = []
        
        for symbol_data in aggregated_data.values():
            market_data = symbol_data.get("market_data", {})
            if "current_price" in market_data:
                all_prices.append(market_data["current_price"])
            if "volume_24h" in market_data:
                all_volumes.append(market_data["volume_24h"])
            
            sentiment_data = symbol_data.get("sentiment", {})
            if "sentiment_score" in sentiment_data:
                sentiment_scores.append(sentiment_data["sentiment_score"])
        
        return {
            "average_price": np.mean(all_prices) if all_prices else 0,
            "total_volume": sum(all_volumes),
            "average_sentiment": np.mean(sentiment_scores) if sentiment_scores else 0.5,
            "data_completeness": len(all_prices) / len(aggregated_data) if aggregated_data else 0
        }
    
    def _analyze_tool_performance(self) -> Dict[str, Any]:
        """Analyze tool execution performance"""
        executions = self.context.tool_executions
        if not executions:
            return {"total_executions": 0}
        
        success_count = sum(1 for exec in executions if exec.get("result", {}).get("success", False))
        total_duration = sum(exec.get("duration", 0) for exec in executions)
        
        return {
            "total_executions": len(executions),
            "success_rate": success_count / len(executions),
            "average_duration": total_duration / len(executions),
            "total_duration": total_duration
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "conversation_entries": len(self.context.conversation_history),
            "tool_executions": len(self.context.tool_executions),
            "shared_memory_items": len(self.context.shared_memory),
            "workflow_state_size": len(self.context.workflow_state)
        }
    
    def _calculate_error_rates(self) -> Dict[str, Any]:
        """Calculate error rates across different categories"""
        executions = self.context.tool_executions
        if not executions:
            return {"overall_error_rate": 0}
        
        total_errors = sum(1 for exec in executions if not exec.get("result", {}).get("success", True))
        
        return {
            "overall_error_rate": total_errors / len(executions),
            "total_errors": total_errors,
            "total_executions": len(executions)
        }
    
    def _check_connectivity(self) -> Dict[str, Any]:
        """Check connectivity status"""
        return {
            "connected_agents": len(self.connected_agents),
            "a2a_enabled": self.enable_a2a,
            "mcp_tools_available": len(self.mcp_tools),
            "strands_tools_available": len(self.tool_registry)
        }
    
    def _calculate_health_score(self, health_data: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        score = 100
        
        # Penalize for high error rates
        error_rate = health_data.get("error_rates", {}).get("overall_error_rate", 0)
        score -= error_rate * 50
        
        # Penalize for poor tool performance
        tool_perf = health_data.get("tool_performance", {})
        success_rate = tool_perf.get("success_rate", 1)
        score -= (1 - success_rate) * 30
        
        # Reward for good connectivity
        connectivity = health_data.get("connectivity", {})
        if connectivity.get("a2a_enabled") and connectivity.get("connected_agents", 0) > 0:
            score += 10
        
        return max(0, min(100, score))
    
    def _get_health_status(self, health_score: float) -> str:
        """Convert health score to status"""
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 60:
            return "fair"
        elif health_score >= 40:
            return "poor"
        else:
            return "critical"