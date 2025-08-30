"""
Integration module for Trading Algorithm Agent with A2A Network.

Handles agent registration, communication, and signal generation.

IMPORTANT: This module provides trading ANALYSIS and SIGNALS only.
It does NOT execute real trades or manage actual portfolios.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from .trading_algorithm_agent import TradingAlgorithmAgent, TradingStrategy
from ..base import AgentStatus
from ...protocols.a2a.a2a_protocol import A2AMessage, MessageType
from ....data.providers.real_only_provider import RealOnlyDataProvider

logger = logging.getLogger(__name__)


class TradingAgentIntegration:
    """Integrates Trading Algorithm Agent with A2A network and blockchain."""
    
    def __init__(self):
        self.trading_agent = TradingAlgorithmAgent()
        self.data_provider = RealOnlyDataProvider()
        self.registered = False
        self.agent_registry = {}
        
    async def register_with_a2a_network(self) -> Dict[str, Any]:
        """Register the trading agent with the A2A network."""
        # Create skill card for registration
        skill_card = {
            "agent_id": self.trading_agent.agent_id,
            "name": "Trading Algorithm Agent",
            "version": "1.0.0",
            "capabilities": [
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
                "portfolio_optimization"
            ],
            "mcp_tools": list(self.trading_agent.mcp_tools.keys()),
            "supported_markets": ["crypto"],
            "supported_exchanges": ["binance", "coinbase", "kraken"],
            "compliance": {
                "a2a_compliant": True,
                "mcp_segregated": True,
                "security_verified": True
            },
            "performance_metrics": {
                "average_latency_ms": 50,
                "uptime_percentage": 99.9,
                "max_concurrent_strategies": 10
            }
        }
        
        # Send registration message
        registration_message = A2AMessage(
            type=MessageType.AGENT_REGISTRATION,
            sender=self.trading_agent.agent_id,
            receiver="agent_manager",
            payload=skill_card
        )
        
        # Process registration
        try:
            response = await self._send_to_network(registration_message)
            
            if response and response.payload.get("status") == "registered":
                self.registered = True
                logger.info(f"Trading agent registered successfully: {self.trading_agent.agent_id}")
                
                # Store registration details
                self.agent_registry[self.trading_agent.agent_id] = {
                    "skill_card": skill_card,
                    "registration_time": datetime.now(),
                    "network_id": response.payload.get("network_id")
                }
                
                return {
                    "status": "success",
                    "agent_id": self.trading_agent.agent_id,
                    "network_id": response.payload.get("network_id"),
                    "capabilities": skill_card["capabilities"]
                }
            else:
                return {
                    "status": "failed",
                    "reason": response.payload.get("reason", "Unknown") if response else "No response"
                }
                
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def start_trading_services(self) -> None:
        """Start all trading services and strategy monitors."""
        if not self.registered:
            await self.register_with_a2a_network()
        
        # Start message handler
        asyncio.create_task(self._handle_incoming_messages())
        
        # Start strategy monitors
        asyncio.create_task(self._monitor_strategies())
        
        # Start risk monitor
        asyncio.create_task(self._monitor_risk())
        
        # Start performance tracker
        asyncio.create_task(self._track_performance())
        
        logger.info("Trading services started")
    
    async def _handle_incoming_messages(self) -> None:
        """Handle incoming A2A messages."""
        while True:
            try:
                # Check for incoming messages (would connect to actual message queue)
                message = await self._receive_from_network()
                
                if message:
                    # Route message based on type
                    if message.type == MessageType.ANALYSIS_REQUEST:
                        await self._handle_analysis_request(message)
                    elif message.type == MessageType.TRADE_EXECUTION:
                        await self._handle_trade_execution(message)
                    elif message.type == MessageType.WORKFLOW_REQUEST:
                        await self._handle_workflow_request(message)
                    elif message.type == MessageType.DATA_LOAD_REQUEST:
                        await self._handle_data_request(message)
                    
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await asyncio.sleep(1)
    
    async def _handle_analysis_request(self, message: A2AMessage) -> None:
        """Handle analysis request from other agents."""
        payload = message.payload
        strategy = payload.get("strategy")
        symbols = payload.get("symbols", [])
        
        result = None
        
        if strategy == "grid_trading":
            result = await self.trading_agent._mcp_grid_create(
                symbols[0] if symbols else "BTC/USDT",
                Decimal(payload.get("investment", "1000"))
            )
        elif strategy == "momentum":
            signals = await self.trading_agent._mcp_momentum_scan(symbols)
            result = {"signals": [s.__dict__ for s in signals]}
        elif strategy == "arbitrage":
            opportunities = await self.trading_agent._mcp_arbitrage_scan(symbols)
            result = {"opportunities": [o.__dict__ for o in opportunities]}
        elif strategy == "mean_reversion":
            signals = await self.trading_agent._mcp_mean_reversion_identify(symbols)
            result = {"signals": [s.__dict__ for s in signals]}
        elif strategy == "ml_predict":
            predictions = []
            for symbol in symbols:
                pred = await self.trading_agent._mcp_ml_predict(symbol)
                predictions.append(pred)
            result = {"predictions": predictions}
        else:
            result = {"error": f"Unknown strategy: {strategy}"}
        
        # Send response
        response = A2AMessage(
            type=MessageType.ANALYSIS_RESPONSE,
            sender=self.trading_agent.agent_id,
            receiver=message.sender,
            payload=result
        )
        
        await self._send_to_network(response)
    
    async def _handle_trade_execution(self, message: A2AMessage) -> None:
        """Handle trade signal generation request (NO ACTUAL EXECUTION)."""
        payload = message.payload
        
        # Generate trade signal based on request
        signal = payload.get("signal")
        if not signal:
            response_payload = {"error": "No signal provided"}
        else:
            # Determine strategy and execute
            strategy = TradingStrategy(signal.get("strategy", "momentum"))
            
            if strategy == TradingStrategy.MOMENTUM:
                result = await self.trading_agent._mcp_momentum_enter(signal)
            elif strategy == TradingStrategy.MEAN_REVERSION:
                result = await self.trading_agent._mcp_mean_reversion_trade(signal)
            elif strategy == TradingStrategy.BREAKOUT:
                result = await self.trading_agent._mcp_breakout_trade(signal)
            else:
                result = {"error": f"Unsupported strategy for execution: {strategy}"}
            
            response_payload = result
        
        # Send response
        response = A2AMessage(
            type=MessageType.TRADE_RESPONSE,
            sender=self.trading_agent.agent_id,
            receiver=message.sender,
            payload=response_payload
        )
        
        await self._send_to_network(response)
    
    async def _handle_workflow_request(self, message: A2AMessage) -> None:
        """Handle complex workflow requests."""
        workflow = message.payload.get("workflow")
        
        if workflow == "full_market_analysis":
            result = await self._execute_full_market_analysis()
        elif workflow == "portfolio_rebalance":
            result = await self._execute_portfolio_rebalance()
        elif workflow == "strategy_optimization":
            result = await self._execute_strategy_optimization()
        else:
            result = {"error": f"Unknown workflow: {workflow}"}
        
        # Send response
        response = A2AMessage(
            type=MessageType.WORKFLOW_RESPONSE,
            sender=self.trading_agent.agent_id,
            receiver=message.sender,
            payload=result
        )
        
        await self._send_to_network(response)
    
    async def _handle_data_request(self, message: A2AMessage) -> None:
        """Handle data request from other agents."""
        data_type = message.payload.get("data_type")
        
        if data_type == "performance_metrics":
            data = self.trading_agent.performance_metrics
        elif data_type == "active_positions":
            data = self.trading_agent.positions
        elif data_type == "strategy_status":
            data = self.trading_agent.active_strategies
        elif data_type == "risk_metrics":
            portfolio = {"positions": self.trading_agent.positions}
            data = await self.trading_agent._mcp_risk_calculate(portfolio)
        else:
            data = {"error": f"Unknown data type: {data_type}"}
        
        # Send response
        response = A2AMessage(
            type=MessageType.DATA_LOAD_RESPONSE,
            sender=self.trading_agent.agent_id,
            receiver=message.sender,
            payload=data
        )
        
        await self._send_to_network(response)
    
    async def _monitor_strategies(self) -> None:
        """Monitor active strategies and trigger actions."""
        while True:
            try:
                for strategy, is_active in self.trading_agent.active_strategies.items():
                    if not is_active:
                        continue
                    
                    if strategy == TradingStrategy.GRID_TRADING:
                        # Check grid rebalancing
                        for symbol in self.trading_agent.order_book.keys():
                            await self.trading_agent._mcp_grid_rebalance(symbol)
                    
                    elif strategy == TradingStrategy.DCA:
                        # DCA is handled by scheduled tasks
                        pass
                    
                    elif strategy == TradingStrategy.ARBITRAGE:
                        # Continuous arbitrage scanning
                        symbols = await self._get_top_symbols()
                        opportunities = await self.trading_agent._mcp_arbitrage_scan(symbols)
                        
                        if opportunities:
                            # Execute best opportunity
                            await self.trading_agent._mcp_arbitrage_execute(opportunities[0])
                    
                    elif strategy == TradingStrategy.MOMENTUM:
                        # Update trailing stops
                        for symbol in self.trading_agent.positions.keys():
                            await self.trading_agent._mcp_momentum_trail_stop(symbol)
                    
                    elif strategy == TradingStrategy.SCALPING:
                        # High-frequency monitoring
                        symbols = await self._get_top_symbols()
                        opportunities = await self.trading_agent._mcp_scalping_scan(symbols)
                        
                        for opp in opportunities[:3]:  # Limit concurrent scalps
                            await self.trading_agent._mcp_scalping_execute(opp)
                    
                    elif strategy == TradingStrategy.MARKET_MAKING:
                        # Update quotes
                        for symbol in self.trading_agent.positions.keys():
                            await self.trading_agent._mcp_market_making_quote(symbol)
                            await self.trading_agent._mcp_inventory_manage(symbol)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Strategy monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_risk(self) -> None:
        """Monitor portfolio risk and trigger protective actions."""
        while True:
            try:
                # Calculate current risk metrics
                portfolio = {"positions": self.trading_agent.positions}
                risk_metrics = await self.trading_agent._mcp_risk_calculate(portfolio)
                
                # Check risk limits
                if risk_metrics.get("max_drawdown", 0) < -self.trading_agent.max_drawdown:
                    # Drawdown limit exceeded - reduce positions
                    await self._reduce_risk_exposure()
                    
                    # Notify
                    alert = A2AMessage(
                        type=MessageType.ALERT,
                        sender=self.trading_agent.agent_id,
                        receiver="risk_manager",
                        payload={
                            "alert_type": "max_drawdown_exceeded",
                            "drawdown": risk_metrics["max_drawdown"],
                            "action_taken": "positions_reduced"
                        }
                    )
                    await self._send_to_network(alert)
                
                # Check correlation risk
                if risk_metrics.get("correlation_risk", 0) > 0.9:
                    # High correlation - diversify
                    await self._diversify_portfolio()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _track_performance(self) -> None:
        """Track and report performance metrics."""
        while True:
            try:
                # Aggregate performance across strategies
                performance = {}
                
                for strategy in TradingStrategy:
                    if strategy == TradingStrategy.MULTI_STRATEGY:
                        continue
                    
                    strategy_perf = await self._calculate_strategy_performance(strategy)
                    performance[strategy.value] = strategy_perf
                
                # Calculate overall metrics
                total_trades = sum(p.get("total_trades", 0) for p in performance.values())
                total_profit = sum(p.get("total_profit", 0) for p in performance.values())
                avg_win_rate = np.mean([p.get("win_rate", 0) for p in performance.values() if p.get("win_rate")])
                
                overall_performance = {
                    "timestamp": datetime.now(),
                    "total_trades": total_trades,
                    "total_profit": total_profit,
                    "average_win_rate": avg_win_rate,
                    "strategy_performance": performance,
                    "active_strategies": [s.value for s, active in self.trading_agent.active_strategies.items() if active]
                }
                
                # Store performance
                self.trading_agent.performance_metrics["latest"] = overall_performance
                
                # Broadcast performance update
                update = A2AMessage(
                    type=MessageType.PERFORMANCE_UPDATE,
                    sender=self.trading_agent.agent_id,
                    receiver="broadcast",
                    payload=overall_performance
                )
                await self._send_to_network(update)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(600)
    
    # Workflow implementations
    
    async def _execute_full_market_analysis(self) -> Dict[str, Any]:
        """Execute comprehensive market analysis workflow."""
        symbols = await self._get_top_symbols(50)
        
        analysis = {
            "timestamp": datetime.now(),
            "symbols_analyzed": len(symbols),
            "opportunities": {}
        }
        
        # Run all strategy scans
        analysis["opportunities"]["momentum"] = await self.trading_agent._mcp_momentum_scan(symbols)
        analysis["opportunities"]["mean_reversion"] = await self.trading_agent._mcp_mean_reversion_identify(symbols)
        analysis["opportunities"]["breakout"] = await self.trading_agent._mcp_breakout_detect(symbols)
        analysis["opportunities"]["arbitrage"] = await self.trading_agent._mcp_arbitrage_scan(symbols)
        
        # ML predictions for top opportunities
        ml_predictions = []
        for symbol in symbols[:10]:
            pred = await self.trading_agent._mcp_ml_predict(symbol)
            ml_predictions.append(pred)
        analysis["ml_predictions"] = ml_predictions
        
        # Market condition assessment
        analysis["market_condition"] = await self._assess_market_condition()
        
        # Risk assessment
        analysis["risk_assessment"] = await self.trading_agent._mcp_risk_calculate(
            {"positions": self.trading_agent.positions}
        )
        
        return analysis
    
    async def _execute_portfolio_rebalance(self) -> Dict[str, Any]:
        """Execute portfolio rebalancing workflow."""
        # Get current positions
        current_positions = list(self.trading_agent.positions.keys())
        
        # Optimize allocation
        optimization = await self.trading_agent._mcp_portfolio_optimize(
            current_positions,
            target_return=0.15
        )
        
        # Calculate rebalancing trades
        rebalancing_trades = []
        
        for symbol, target_weight in optimization["allocations"].items():
            current_value = self.trading_agent.positions.get(symbol, {}).get("value", 0)
            portfolio_value = await self.trading_agent._get_portfolio_value()
            target_value = portfolio_value * Decimal(str(target_weight))
            
            difference = target_value - current_value
            
            if abs(difference) > portfolio_value * Decimal("0.01"):  # 1% threshold
                if difference > 0:
                    action = "BUY"
                    quantity = difference / await self.trading_agent._get_current_price(symbol)
                else:
                    action = "SELL"
                    quantity = abs(difference) / await self.trading_agent._get_current_price(symbol)
                
                rebalancing_trades.append({
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "current_weight": float(current_value / portfolio_value),
                    "target_weight": target_weight
                })
        
        # Generate rebalancing signals
        rebalancing_signals = []
        for trade in rebalancing_trades:
            signal = await self.trading_agent._generate_trade_signal(
                symbol=trade["symbol"],
                side=trade["action"],
                quantity=trade["quantity"],
                price=await self.trading_agent._get_current_price(trade["symbol"]),
                order_type="MARKET"
            )
            rebalancing_signals.append(signal)
        
        return {
            "optimization": optimization,
            "rebalancing_trades": rebalancing_trades,
            "rebalancing_signals": rebalancing_signals,
            "timestamp": datetime.now()
        }
    
    async def _execute_strategy_optimization(self) -> Dict[str, Any]:
        """Execute strategy optimization workflow."""
        # Optimize each active strategy
        optimization_results = await self.trading_agent._mcp_strategy_optimize()
        
        # Adjust strategy allocation
        allocation_results = await self.trading_agent._mcp_strategy_allocate()
        
        # Assess market condition and switch strategies if needed
        market_condition = await self._assess_market_condition()
        switch_results = await self.trading_agent._mcp_strategy_switch(market_condition)
        
        return {
            "optimization": optimization_results,
            "allocation": allocation_results,
            "strategy_switch": switch_results,
            "market_condition": market_condition,
            "timestamp": datetime.now()
        }
    
    # Helper methods
    
    async def _get_top_symbols(self, limit: int = 20) -> List[str]:
        """Get top trading symbols by volume."""
        # This would connect to actual data source
        return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"][:limit]
    
    async def _assess_market_condition(self) -> str:
        """Assess overall market condition."""
        # Analyze market indicators
        btc_price = await self.trading_agent._get_current_price("BTC/USDT")
        btc_history = await self.trading_agent._get_price_history("BTC/USDT", 100)
        
        # Calculate trend
        ma_20 = np.mean(btc_history[-20:])
        ma_50 = np.mean(btc_history[-50:])
        
        # Calculate volatility
        volatility = np.std(btc_history[-20:]) / ma_20
        
        # Determine condition
        if ma_20 > ma_50 * 1.02:
            if volatility > 0.05:
                return "high_volatility"
            return "trending_up"
        elif ma_20 < ma_50 * 0.98:
            return "trending_down"
        else:
            if volatility > 0.05:
                return "high_volatility"
            elif volatility < 0.02:
                return "low_volatility"
            return "ranging"
    
    async def _reduce_risk_exposure(self) -> None:
        """Reduce risk by closing positions."""
        # Close losing positions first
        for symbol, position in self.trading_agent.positions.items():
            if position.get("unrealized_pnl", 0) < 0:
                await self.trading_agent._generate_trade_signal(
                    symbol=symbol,
                    side="SELL" if position["side"] == "BUY" else "BUY",
                    quantity=position["quantity"],
                    price=await self.trading_agent._get_current_price(symbol),
                    order_type="MARKET"
                )
    
    async def _diversify_portfolio(self) -> None:
        """Diversify portfolio to reduce correlation risk."""
        # Get uncorrelated assets
        current_symbols = list(self.trading_agent.positions.keys())
        all_symbols = await self._get_top_symbols(50)
        
        # Find low correlation symbols
        for symbol in all_symbols:
            if symbol not in current_symbols:
                # Add small position in uncorrelated asset
                await self.trading_agent._generate_trade_signal(
                    symbol=symbol,
                    side="BUY",
                    quantity=Decimal("10"),  # Small position
                    price=await self.trading_agent._get_current_price(symbol),
                    order_type="MARKET"
                )
                break
    
    async def _calculate_strategy_performance(self, strategy: TradingStrategy) -> Dict[str, Any]:
        """Calculate performance metrics for a specific strategy."""
        # This would aggregate actual trade data
        return {
            "total_trades": 0,
            "total_profit": 0,
            "win_rate": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0
        }
    
    async def _send_to_network(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Send message to A2A network."""
        # This would connect to actual network
        logger.debug(f"Sending message: {message.type} from {message.sender} to {message.receiver}")
        return None
    
    async def _receive_from_network(self) -> Optional[A2AMessage]:
        """Receive message from A2A network."""
        # This would connect to actual message queue
        return None


# Main entry point
async def main():
    """Main function to start the trading agent."""
    integration = TradingAgentIntegration()
    
    # Register with network
    registration = await integration.register_with_a2a_network()
    logger.info(f"Registration result: {registration}")
    
    # Start services
    await integration.start_trading_services()
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down trading agent")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())