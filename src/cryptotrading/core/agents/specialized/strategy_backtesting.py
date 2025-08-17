"""
Strategy Backtesting Tools for MCTS Trading Agent
Provides comprehensive backtesting capabilities with performance metrics
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    TREND_FOLLOWING = "trend_following"
    PAIRS_TRADING = "pairs_trading"
    MARKET_MAKING = "market_making"


@dataclass
class BacktestConfig:
    """Configuration for strategy backtesting"""
    strategy_type: StrategyType
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    position_size: float = 0.1  # 10% of capital per position
    stop_loss: float = 0.05    # 5% stop loss
    take_profit: float = 0.15  # 15% take profit
    transaction_cost: float = 0.001  # 0.1% transaction cost
    slippage: float = 0.001    # 0.1% slippage
    max_positions: int = 5
    rebalance_frequency: str = "daily"  # daily, weekly, monthly


@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # BUY or SELL
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    calmar_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float


@dataclass
class BacktestResult:
    """Complete backtesting results"""
    config: BacktestConfig
    performance: PerformanceMetrics
    trades: List[Trade]
    portfolio_value: List[Tuple[datetime, float]]
    drawdown_series: List[Tuple[datetime, float]]
    monthly_returns: Dict[str, float]
    insights: Dict[str, Any]
    confidence: float


class StrategyBacktester:
    """
    Comprehensive strategy backtesting engine
    """
    
    def __init__(self, market_data_provider=None):
        """
        Initialize backtester
        
        Args:
            market_data_provider: Provider for historical market data
        """
        self.market_data_provider = market_data_provider
        self.trade_id_counter = 0
    
    async def backtest_strategy(self, config: BacktestConfig,
                              strategy_logic: Optional[Dict[str, Any]] = None) -> BacktestResult:
        """
        Run comprehensive strategy backtest
        
        Args:
            config: Backtesting configuration
            strategy_logic: Custom strategy implementation
            
        Returns:
            Complete backtesting results
        """
        logger.info(f"Starting backtest for {config.strategy_type.value} strategy")
        
        # Initialize tracking variables
        portfolio_value = config.initial_capital
        trades = []
        open_positions = {}
        portfolio_history = []
        daily_returns = []
        
        # Get historical data
        historical_data = await self._get_historical_data(config)
        
        # Simulate trading
        for date, market_data in historical_data.items():
            # Check for exit signals on open positions
            await self._check_exit_signals(open_positions, market_data, trades)
            
            # Generate entry signals
            entry_signals = await self._generate_entry_signals(
                config, market_data, open_positions, strategy_logic
            )
            
            # Execute trades
            for signal in entry_signals:
                if len(open_positions) < config.max_positions:
                    trade = await self._execute_trade(signal, market_data, config)
                    if trade:
                        trades.append(trade)
                        open_positions[trade.symbol] = trade
            
            # Calculate portfolio value
            current_value = await self._calculate_portfolio_value(
                open_positions, market_data, config.initial_capital
            )
            portfolio_history.append((date, current_value))
            
            # Calculate daily return
            if len(portfolio_history) > 1:
                prev_value = portfolio_history[-2][1]
                daily_return = (current_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        # Close remaining positions
        final_market_data = list(historical_data.values())[-1]
        for position in open_positions.values():
            if position.symbol in final_market_data:
                await self._close_position(position, final_market_data[position.symbol])
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            portfolio_history, trades, daily_returns, config
        )
        
        # Generate insights
        insights = await self._generate_insights(trades, performance, config)
        
        # Calculate drawdown series
        drawdown_series = self._calculate_drawdown_series(portfolio_history)
        
        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(portfolio_history)
        
        return BacktestResult(
            config=config,
            performance=performance,
            trades=trades,
            portfolio_value=portfolio_history,
            drawdown_series=drawdown_series,
            monthly_returns=monthly_returns,
            insights=insights,
            confidence=self._calculate_confidence(performance, len(trades))
        )
    
    async def compare_strategies(self, configs: List[BacktestConfig]) -> Dict[str, Any]:
        """
        Compare multiple strategies
        
        Args:
            configs: List of strategy configurations to compare
            
        Returns:
            Comparison results with rankings
        """
        results = []
        
        for config in configs:
            result = await self.backtest_strategy(config)
            results.append(result)
        
        # Rank strategies by Sharpe ratio
        ranked_results = sorted(results, key=lambda x: x.performance.sharpe_ratio, reverse=True)
        
        comparison = {
            'rankings': [
                {
                    'rank': i + 1,
                    'strategy': result.config.strategy_type.value,
                    'sharpe_ratio': result.performance.sharpe_ratio,
                    'total_return': result.performance.total_return,
                    'max_drawdown': result.performance.max_drawdown,
                    'win_rate': result.performance.win_rate
                }
                for i, result in enumerate(ranked_results)
            ],
            'best_strategy': ranked_results[0].config.strategy_type.value,
            'performance_summary': {
                'highest_return': max(results, key=lambda x: x.performance.total_return),
                'lowest_risk': min(results, key=lambda x: x.performance.max_drawdown),
                'best_sharpe': max(results, key=lambda x: x.performance.sharpe_ratio)
            }
        }
        
        return comparison
    
    async def _get_historical_data(self, config: BacktestConfig) -> Dict[datetime, Dict[str, Any]]:
        """Get historical market data for backtesting"""
        # Mock implementation for testing
        # In production, this would fetch real historical data
        
        import random
        from datetime import datetime, timedelta
        
        start_date = datetime.strptime(config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
        
        data = {}
        current_date = start_date
        
        # Initialize prices
        prices = {symbol: 100.0 for symbol in config.symbols}
        
        while current_date <= end_date:
            # Simulate price movements
            market_data = {}
            for symbol in config.symbols:
                # Random walk with some trend
                change = random.normalvariate(0.001, 0.02)  # 0.1% average, 2% volatility
                prices[symbol] *= (1 + change)
                
                market_data[symbol] = {
                    'price': prices[symbol],
                    'volume': random.randint(1000000, 10000000),
                    'high': prices[symbol] * (1 + abs(random.normalvariate(0, 0.01))),
                    'low': prices[symbol] * (1 - abs(random.normalvariate(0, 0.01))),
                    'volatility': abs(random.normalvariate(0.02, 0.005))
                }
            
            data[current_date] = market_data
            current_date += timedelta(days=1)
        
        return data
    
    async def _generate_entry_signals(self, config: BacktestConfig, 
                                    market_data: Dict[str, Any],
                                    open_positions: Dict[str, Trade],
                                    strategy_logic: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate entry signals based on strategy type"""
        signals = []
        
        for symbol in config.symbols:
            if symbol in open_positions:
                continue  # Already have position
            
            symbol_data = market_data[symbol]
            
            # Strategy-specific logic
            if config.strategy_type == StrategyType.MOMENTUM:
                # Simple momentum strategy
                if symbol_data['volatility'] > 0.025:  # High volatility
                    signals.append({
                        'symbol': symbol,
                        'side': 'BUY',
                        'confidence': min(symbol_data['volatility'] * 20, 1.0)
                    })
            
            elif config.strategy_type == StrategyType.MEAN_REVERSION:
                # Simple mean reversion
                if symbol_data['volatility'] < 0.015:  # Low volatility
                    signals.append({
                        'symbol': symbol,
                        'side': 'BUY',
                        'confidence': 1.0 - symbol_data['volatility'] * 30
                    })
            
            elif config.strategy_type == StrategyType.TREND_FOLLOWING:
                # Simple trend following
                price_change = (symbol_data['high'] - symbol_data['low']) / symbol_data['price']
                if price_change > 0.02:  # Strong upward movement
                    signals.append({
                        'symbol': symbol,
                        'side': 'BUY',
                        'confidence': min(price_change * 25, 1.0)
                    })
        
        return signals
    
    async def _check_exit_signals(self, open_positions: Dict[str, Trade],
                                market_data: Dict[str, Any], trades: List[Trade]):
        """Check for exit signals on open positions"""
        positions_to_close = []
        
        for symbol, position in open_positions.items():
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['price']
            
            # Calculate current P&L
            if position.side == 'BUY':
                pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:  # SELL
                pnl_pct = (position.entry_price - current_price) / position.entry_price
            
            # Check stop loss
            if pnl_pct <= -0.05:  # 5% stop loss
                await self._close_position(position, market_data[symbol])
                position.status = "STOPPED"
                positions_to_close.append(symbol)
            
            # Check take profit
            elif pnl_pct >= 0.15:  # 15% take profit
                await self._close_position(position, market_data[symbol])
                position.status = "CLOSED"
                positions_to_close.append(symbol)
        
        # Remove closed positions
        for symbol in positions_to_close:
            del open_positions[symbol]
    
    async def _execute_trade(self, signal: Dict[str, Any], 
                           market_data: Dict[str, Any],
                           config: BacktestConfig) -> Optional[Trade]:
        """Execute a trade based on signal"""
        symbol = signal['symbol']
        symbol_data = market_data[symbol]
        
        # Calculate position size
        price = symbol_data['price']
        position_value = config.initial_capital * config.position_size
        quantity = position_value / price
        
        # Apply transaction costs and slippage
        effective_price = price * (1 + config.transaction_cost + config.slippage)
        
        trade = Trade(
            symbol=symbol,
            entry_time=datetime.now(),  # Would use actual market time
            exit_time=None,
            entry_price=effective_price,
            exit_price=None,
            quantity=quantity,
            side=signal['side'],
            status="OPEN"
        )
        
        self.trade_id_counter += 1
        
        return trade
    
    async def _close_position(self, position: Trade, market_data: Dict[str, Any]):
        """Close an open position"""
        current_price = market_data['price']
        
        # Apply transaction costs and slippage
        effective_price = current_price * (1 - 0.001 - 0.001)  # costs + slippage
        
        position.exit_time = datetime.now()
        position.exit_price = effective_price
        
        # Calculate P&L
        if position.side == 'BUY':
            position.pnl = (position.exit_price - position.entry_price) * position.quantity
            position.return_pct = (position.exit_price - position.entry_price) / position.entry_price
        else:  # SELL
            position.pnl = (position.entry_price - position.exit_price) * position.quantity
            position.return_pct = (position.entry_price - position.exit_price) / position.entry_price
    
    async def _calculate_portfolio_value(self, open_positions: Dict[str, Trade],
                                       market_data: Dict[str, Any], 
                                       initial_capital: float) -> float:
        """Calculate current portfolio value"""
        total_value = initial_capital
        
        for symbol, position in open_positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                if position.side == 'BUY':
                    position_value = current_price * position.quantity
                else:
                    position_value = position.entry_price * position.quantity - (current_price - position.entry_price) * position.quantity
                
                # Subtract the initial investment and add current value
                total_value += position_value - (position.entry_price * position.quantity)
        
        return total_value
    
    def _calculate_performance_metrics(self, portfolio_history: List[Tuple[datetime, float]],
                                     trades: List[Trade], daily_returns: List[float],
                                     config: BacktestConfig) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not portfolio_history or not trades:
            return PerformanceMetrics(
                total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0, average_win=0,
                average_loss=0, total_trades=0, winning_trades=0, losing_trades=0,
                calmar_ratio=0, sortino_ratio=0, var_95=0, expected_shortfall=0
            )
        
        # Calculate returns
        initial_value = portfolio_history[0][1]
        final_value = portfolio_history[-1][1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate annualized return
        days = len(portfolio_history)
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Calculate volatility
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown([v for _, v in portfolio_history])
        
        # Trade statistics
        closed_trades = [t for t in trades if t.pnl is not None]
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        losing_trades = len([t for t in closed_trades if t.pnl < 0])
        total_trades = len(closed_trades)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in closed_trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in closed_trades if t.pnl < 0]
        
        average_win = np.mean(wins) if wins else 0
        average_loss = np.mean(losses) if losses else 0
        
        total_wins = sum(wins)
        total_losses = sum(losses)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate Sortino ratio
        negative_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate VaR and Expected Shortfall
        var_95 = np.percentile(daily_returns, 5) if daily_returns else 0
        worst_5_percent = [r for r in daily_returns if r <= var_95]
        expected_shortfall = np.mean(worst_5_percent) if worst_5_percent else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            expected_shortfall=expected_shortfall
        )
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not values:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _calculate_drawdown_series(self, portfolio_history: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float]]:
        """Calculate drawdown time series"""
        if not portfolio_history:
            return []
        
        drawdown_series = []
        peak = portfolio_history[0][1]
        
        for date, value in portfolio_history:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            drawdown_series.append((date, drawdown))
        
        return drawdown_series
    
    def _calculate_monthly_returns(self, portfolio_history: List[Tuple[datetime, float]]) -> Dict[str, float]:
        """Calculate monthly returns"""
        if len(portfolio_history) < 30:
            return {}
        
        monthly_returns = {}
        current_month = None
        month_start_value = None
        
        for date, value in portfolio_history:
            month_key = f"{date.year}-{date.month:02d}"
            
            if month_key != current_month:
                if current_month is not None and month_start_value is not None:
                    # Calculate return for previous month
                    prev_value = portfolio_history[portfolio_history.index((date, value)) - 1][1]
                    monthly_return = (prev_value - month_start_value) / month_start_value
                    monthly_returns[current_month] = monthly_return
                
                current_month = month_key
                month_start_value = value
        
        return monthly_returns
    
    async def _generate_insights(self, trades: List[Trade], 
                               performance: PerformanceMetrics,
                               config: BacktestConfig) -> Dict[str, Any]:
        """Generate trading insights and recommendations"""
        insights = {
            'strategy_effectiveness': 'Good' if performance.sharpe_ratio > 1.0 else 'Moderate' if performance.sharpe_ratio > 0.5 else 'Poor',
            'risk_assessment': 'Low' if performance.max_drawdown < 0.1 else 'Medium' if performance.max_drawdown < 0.2 else 'High',
            'consistency': 'High' if performance.win_rate > 0.6 else 'Medium' if performance.win_rate > 0.4 else 'Low',
            'recommendations': [],
            'key_observations': []
        }
        
        # Generate recommendations based on performance
        if performance.sharpe_ratio < 1.0:
            insights['recommendations'].append("Consider adjusting strategy parameters to improve risk-adjusted returns")
        
        if performance.max_drawdown > 0.15:
            insights['recommendations'].append("Implement stricter risk management to reduce maximum drawdown")
        
        if performance.win_rate < 0.5:
            insights['recommendations'].append("Analyze losing trades to improve entry/exit criteria")
        
        # Key observations
        if performance.profit_factor > 2.0:
            insights['key_observations'].append("Strong profit factor indicates good trade selection")
        
        if len(trades) < 20:
            insights['key_observations'].append("Limited sample size - consider longer backtesting period")
        
        return insights
    
    def _calculate_confidence(self, performance: PerformanceMetrics, num_trades: int) -> float:
        """Calculate confidence score for backtest results"""
        # Base confidence on multiple factors
        trade_count_factor = min(num_trades / 100, 1.0)  # More trades = higher confidence
        consistency_factor = performance.win_rate
        risk_factor = 1.0 - min(performance.max_drawdown, 1.0)
        
        confidence = (trade_count_factor + consistency_factor + risk_factor) / 3
        return round(confidence, 2)