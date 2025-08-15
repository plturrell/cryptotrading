"""
Financial Analysis Sub-skill for Calculation Agent

Provides specialized financial computation capabilities through Strands MCP:
- Multi-asset correlation analysis for FX and crypto pairs
- Time series analysis with lead/lag relationships
- Risk metrics (Sharpe ratio, maximum drawdown, volatility)
- Real-time signal processing and early warning generation
- Portfolio analysis and performance attribution

This sub-skill integrates with the main calculation agent via Strands tools
and leverages GROK intelligence for financial decision making.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.strands.tools import tool
from .types import CalculationResult, ComputationMethod, CalculationType
from .utils import format_result_for_display, validate_numeric_result


logger = logging.getLogger(__name__)


class FinancialAnalysisSkill:
    """Financial analysis sub-skill for calculation agent"""
    
    def __init__(self):
        self.correlation_cache = {}
        self.signal_history = []
        self.performance_tracking = {
            "signal_accuracy": {},
            "correlation_stability": {},
            "risk_metrics_computed": 0,
            "backtests_performed": 0
        }
        
        # Financial constants
        self.trading_days_per_year = 252
        self.hours_per_trading_day = 24  # Crypto trades 24/7
        self.risk_free_rates = {
            "USD": 0.05,  # Current fed funds rate
            "EUR": 0.04,
            "JPY": 0.001,
            "GBP": 0.045
        }
    
    # CORRELATION ANALYSIS TOOLS
    
    @tool
    def calculate_correlation_matrix(self, asset_data: Dict[str, List[float]], 
                                   method: str = "pearson") -> Dict[str, Any]:
        """
        Calculate correlation matrix between multiple FX and crypto pairs
        
        Args:
            asset_data: Dict with asset names as keys and price data as values
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Dict with correlation matrix and analysis
        """
        start_time = time.time()
        
        try:
            # Validate input data
            if not asset_data or len(asset_data) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 assets for correlation analysis"
                }
            
            # Convert to pandas DataFrame for easier analysis
            df_data = {}
            min_length = float('inf')
            
            for asset, prices in asset_data.items():
                if not prices or len(prices) < 10:
                    return {
                        "success": False,
                        "error": f"Insufficient data for {asset} (need at least 10 points)"
                    }
                df_data[asset] = prices
                min_length = min(min_length, len(prices))
            
            # Align data lengths
            for asset in df_data:
                df_data[asset] = df_data[asset][-min_length:]
            
            df = pd.DataFrame(df_data)
            
            # Calculate correlation matrix
            if method == "pearson":
                corr_matrix = df.corr(method='pearson')
            elif method == "spearman":
                corr_matrix = df.corr(method='spearman')
            elif method == "kendall":
                corr_matrix = df.corr(method='kendall')
            else:
                return {
                    "success": False,
                    "error": f"Unknown correlation method: {method}"
                }
            
            # Calculate p-values for statistical significance
            n = len(df)
            p_values = {}
            for asset1 in df.columns:
                p_values[asset1] = {}
                for asset2 in df.columns:
                    if asset1 != asset2:
                        corr_coef = corr_matrix.loc[asset1, asset2]
                        # Calculate t-statistic and p-value
                        t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2))
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        p_values[asset1][asset2] = p_val
                    else:
                        p_values[asset1][asset2] = 0.0
            
            # Identify significant correlations (p < 0.05)
            significant_pairs = []
            for asset1 in df.columns:
                for asset2 in df.columns:
                    if asset1 < asset2:  # Avoid duplicates
                        corr = corr_matrix.loc[asset1, asset2]
                        p_val = p_values[asset1][asset2]
                        if p_val < 0.05:
                            significant_pairs.append({
                                "pair": f"{asset1}/{asset2}",
                                "correlation": float(corr),
                                "p_value": float(p_val),
                                "strength": self._interpret_correlation_strength(abs(corr))
                            })
            
            # Sort by absolute correlation strength
            significant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "correlation_matrix",
                "computation_time": computation_time,
                "correlation_matrix": corr_matrix.to_dict(),
                "p_value_matrix": p_values,
                "significant_correlations": significant_pairs,
                "sample_size": n,
                "assets_analyzed": list(df.columns),
                "strongest_correlation": significant_pairs[0] if significant_pairs else None,
                "summary": f"Analyzed {len(df.columns)} assets with {n} data points, found {len(significant_pairs)} significant correlations"
            }
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            return {
                "success": False,
                "method": "correlation_matrix",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    @tool
    def compute_rolling_correlations(self, asset1_data: List[float], asset2_data: List[float],
                                   asset1_name: str, asset2_name: str,
                                   window_size: int = 24, step_size: int = 1) -> Dict[str, Any]:
        """
        Compute rolling correlations between two assets over time
        
        Args:
            asset1_data: Price data for first asset (e.g., USD/JPY)
            asset2_data: Price data for second asset (e.g., BTC/USDT)
            asset1_name: Name of first asset
            asset2_name: Name of second asset
            window_size: Size of rolling window (hours/periods)
            step_size: Step size for rolling window
            
        Returns:
            Dict with rolling correlation time series and analysis
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if len(asset1_data) != len(asset2_data):
                return {
                    "success": False,
                    "error": "Asset data arrays must have same length"
                }
            
            if len(asset1_data) < window_size:
                return {
                    "success": False,
                    "error": f"Need at least {window_size} data points for rolling correlation"
                }
            
            # Convert to numpy arrays
            data1 = np.array(asset1_data)
            data2 = np.array(asset2_data)
            
            # Calculate rolling correlations
            rolling_corrs = []
            timestamps = []
            
            for i in range(0, len(data1) - window_size + 1, step_size):
                window_data1 = data1[i:i + window_size]
                window_data2 = data2[i:i + window_size]
                
                # Calculate correlation for this window
                corr = np.corrcoef(window_data1, window_data2)[0, 1]
                rolling_corrs.append(corr)
                timestamps.append(i + window_size - 1)  # End of window
            
            rolling_corrs = np.array(rolling_corrs)
            
            # Identify correlation regime changes
            correlation_changes = self._detect_correlation_regimes(rolling_corrs)
            
            # Calculate statistics
            stats_analysis = {
                "mean_correlation": float(np.mean(rolling_corrs[~np.isnan(rolling_corrs)])),
                "std_correlation": float(np.std(rolling_corrs[~np.isnan(rolling_corrs)])),
                "min_correlation": float(np.nanmin(rolling_corrs)),
                "max_correlation": float(np.nanmax(rolling_corrs)),
                "correlation_range": float(np.nanmax(rolling_corrs) - np.nanmin(rolling_corrs)),
                "stable_periods": len([c for c in correlation_changes if c["type"] == "stable"]),
                "breakdown_periods": len([c for c in correlation_changes if c["type"] == "breakdown"])
            }
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "rolling_correlations",
                "computation_time": computation_time,
                "asset_pair": f"{asset1_name}/{asset2_name}",
                "window_size": window_size,
                "step_size": step_size,
                "rolling_correlations": rolling_corrs.tolist(),
                "timestamps": timestamps,
                "correlation_statistics": stats_analysis,
                "regime_changes": correlation_changes,
                "current_correlation": float(rolling_corrs[-1]) if len(rolling_corrs) > 0 else None,
                "correlation_trend": self._analyze_correlation_trend(rolling_corrs[-10:] if len(rolling_corrs) >= 10 else rolling_corrs),
                "summary": f"Rolling correlation analysis: {asset1_name} vs {asset2_name}, {len(rolling_corrs)} windows analyzed"
            }
            
        except Exception as e:
            logger.error(f"Rolling correlation calculation failed: {e}")
            return {
                "success": False,
                "method": "rolling_correlations",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    @tool
    def analyze_lead_lag_relationships(self, series1_data: List[float], series2_data: List[float],
                                     series1_name: str, series2_name: str,
                                     max_lag: int = 8) -> Dict[str, Any]:
        """
        Analyze lead-lag relationships between two time series (e.g., FX leading crypto)
        
        Args:
            series1_data: First time series (potential leader)
            series2_data: Second time series (potential follower)  
            series1_name: Name of first series
            series2_name: Name of second series
            max_lag: Maximum lag to test (hours/periods)
            
        Returns:
            Dict with lead-lag analysis and optimal lag identification
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if len(series1_data) != len(series2_data):
                return {
                    "success": False,
                    "error": "Time series must have same length"
                }
            
            if len(series1_data) < max_lag * 2:
                return {
                    "success": False,
                    "error": f"Need at least {max_lag * 2} data points for lag analysis"
                }
            
            # Convert to numpy arrays and calculate returns
            prices1 = np.array(series1_data)
            prices2 = np.array(series2_data)
            
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]
            
            # Test different lags
            lag_correlations = {}
            
            # Test negative lags (series2 leads series1)
            for lag in range(-max_lag, 0):
                if abs(lag) < len(returns1):
                    if lag < 0:
                        corr = np.corrcoef(returns1[-lag:], returns2[:len(returns1)+lag])[0, 1]
                    else:
                        corr = np.corrcoef(returns1[:-lag], returns2[lag:])[0, 1]
                    lag_correlations[lag] = corr
            
            # Test zero lag (contemporaneous)
            corr = np.corrcoef(returns1, returns2)[0, 1]
            lag_correlations[0] = corr
            
            # Test positive lags (series1 leads series2)
            for lag in range(1, max_lag + 1):
                if lag < len(returns1):
                    corr = np.corrcoef(returns1[:-lag], returns2[lag:])[0, 1]
                    lag_correlations[lag] = corr
            
            # Find optimal lag (highest absolute correlation)
            valid_lags = {k: v for k, v in lag_correlations.items() if not np.isnan(v)}
            if not valid_lags:
                return {
                    "success": False,
                    "error": "No valid correlations found across lags"
                }
            
            optimal_lag = max(valid_lags.keys(), key=lambda k: abs(valid_lags[k]))
            optimal_correlation = valid_lags[optimal_lag]
            
            # Interpret results
            if optimal_lag > 0:
                leader = series1_name
                follower = series2_name
                lead_time = optimal_lag
            elif optimal_lag < 0:
                leader = series2_name
                follower = series1_name
                lead_time = abs(optimal_lag)
            else:
                leader = "Contemporaneous"
                follower = "Contemporaneous"
                lead_time = 0
            
            # Calculate statistical significance
            n = len(returns1) - abs(optimal_lag)
            t_stat = optimal_correlation * np.sqrt((n - 2) / (1 - optimal_correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "lead_lag_analysis",
                "computation_time": computation_time,
                "series_pair": f"{series1_name} vs {series2_name}",
                "max_lag_tested": max_lag,
                "lag_correlations": valid_lags,
                "optimal_lag": optimal_lag,
                "optimal_correlation": float(optimal_correlation),
                "p_value": float(p_value),
                "statistically_significant": p_value < 0.05,
                "leader": leader,
                "follower": follower,
                "lead_time_periods": lead_time,
                "relationship_strength": self._interpret_correlation_strength(abs(optimal_correlation)),
                "early_warning_capability": lead_time > 0 and abs(optimal_correlation) > 0.3,
                "summary": f"Lead-lag analysis: {leader} leads {follower} by {lead_time} periods (r={optimal_correlation:.3f})"
            }
            
        except Exception as e:
            logger.error(f"Lead-lag analysis failed: {e}")
            return {
                "success": False,
                "method": "lead_lag_analysis",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    # FINANCIAL RISK METRICS TOOLS
    
    @tool
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = None,
                             periods_per_year: int = None) -> Dict[str, Any]:
        """
        Calculate Sharpe ratio for a return series
        
        Args:
            returns: List of period returns
            risk_free_rate: Annual risk-free rate (defaults to USD rate)
            periods_per_year: Number of periods per year for annualization
            
        Returns:
            Dict with Sharpe ratio and related metrics
        """
        start_time = time.time()
        
        try:
            if not returns or len(returns) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 return observations"
                }
            
            returns_array = np.array(returns)
            risk_free_rate = risk_free_rate or self.risk_free_rates["USD"]
            periods_per_year = periods_per_year or (self.trading_days_per_year * self.hours_per_trading_day)
            
            # Calculate metrics
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)
            
            # Annualize
            annualized_return = mean_return * periods_per_year
            annualized_volatility = std_return * np.sqrt(periods_per_year)
            
            # Calculate Sharpe ratio
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Calculate other risk-adjusted metrics
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0
            annualized_downside_deviation = downside_deviation * np.sqrt(periods_per_year)
            
            sortino_ratio = excess_return / annualized_downside_deviation if annualized_downside_deviation > 0 else 0
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "sharpe_ratio",
                "computation_time": computation_time,
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "annualized_return": float(annualized_return),
                "annualized_volatility": float(annualized_volatility),
                "risk_free_rate": risk_free_rate,
                "excess_return": float(excess_return),
                "downside_deviation": float(annualized_downside_deviation),
                "periods_analyzed": len(returns),
                "periods_per_year": periods_per_year,
                "risk_adjustment_rating": self._rate_risk_adjusted_performance(sharpe_ratio, sortino_ratio),
                "summary": f"Sharpe: {sharpe_ratio:.3f}, Sortino: {sortino_ratio:.3f}, Ann. Return: {annualized_return:.2%}"
            }
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation failed: {e}")
            return {
                "success": False,
                "method": "sharpe_ratio",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    @tool
    def compute_maximum_drawdown(self, price_series: List[float]) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and related risk metrics
        
        Args:
            price_series: Price time series data
            
        Returns:
            Dict with maximum drawdown analysis
        """
        start_time = time.time()
        
        try:
            if not price_series or len(price_series) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 price observations"
                }
            
            prices = np.array(price_series)
            
            # Calculate cumulative returns (normalized to start at 1)
            cumulative_returns = prices / prices[0]
            
            # Calculate running maximum (peak values)
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # Find maximum drawdown
            max_drawdown = np.min(drawdowns)
            max_drawdown_idx = np.argmin(drawdowns)
            
            # Find the peak before max drawdown
            peak_idx = np.argmax(running_max[:max_drawdown_idx + 1])
            
            # Calculate drawdown duration (periods underwater)
            underwater = drawdowns < -0.01  # More than 1% drawdown
            if np.any(underwater):
                # Find longest underwater period
                underwater_periods = []
                start_idx = None
                
                for i, is_underwater in enumerate(underwater):
                    if is_underwater and start_idx is None:
                        start_idx = i
                    elif not is_underwater and start_idx is not None:
                        underwater_periods.append(i - start_idx)
                        start_idx = None
                
                # Handle case where series ends underwater
                if start_idx is not None:
                    underwater_periods.append(len(underwater) - start_idx)
                
                max_underwater_duration = max(underwater_periods) if underwater_periods else 0
                avg_underwater_duration = np.mean(underwater_periods) if underwater_periods else 0
            else:
                max_underwater_duration = 0
                avg_underwater_duration = 0
            
            # Calculate recovery information
            if max_drawdown_idx < len(prices) - 1:
                # Find recovery point (when price exceeds previous peak)
                recovery_prices = prices[max_drawdown_idx:]
                recovery_threshold = prices[peak_idx]
                recovery_idx = None
                
                for i, price in enumerate(recovery_prices):
                    if price >= recovery_threshold:
                        recovery_idx = max_drawdown_idx + i
                        break
                
                recovery_time = recovery_idx - max_drawdown_idx if recovery_idx else None
            else:
                recovery_time = None
            
            # Calculate additional risk metrics
            calmar_ratio = self._calculate_calmar_ratio(prices, max_drawdown)
            pain_index = np.mean(np.abs(drawdowns))  # Average drawdown magnitude
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "maximum_drawdown",
                "computation_time": computation_time,
                "max_drawdown": float(max_drawdown),
                "max_drawdown_percent": float(max_drawdown * 100),
                "max_drawdown_start_idx": int(peak_idx),
                "max_drawdown_end_idx": int(max_drawdown_idx),
                "max_underwater_duration": int(max_underwater_duration),
                "avg_underwater_duration": float(avg_underwater_duration),
                "recovery_time": int(recovery_time) if recovery_time else None,
                "calmar_ratio": float(calmar_ratio),
                "pain_index": float(pain_index),
                "current_drawdown": float(drawdowns[-1]),
                "periods_analyzed": len(prices),
                "drawdown_series": drawdowns.tolist(),
                "risk_assessment": self._assess_drawdown_risk(max_drawdown, max_underwater_duration),
                "summary": f"Max DD: {max_drawdown:.2%}, Duration: {max_underwater_duration} periods, Current DD: {drawdowns[-1]:.2%}"
            }
            
        except Exception as e:
            logger.error(f"Maximum drawdown calculation failed: {e}")
            return {
                "success": False,
                "method": "maximum_drawdown", 
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    @tool
    def calculate_volatility_metrics(self, returns: List[float], window: int = 20) -> Dict[str, Any]:
        """
        Calculate practical volatility metrics for trading systems
        
        Args:
            returns: Return series data
            window: Rolling window for volatility calculation
            
        Returns:
            Dict with volatility analysis
        """
        start_time = time.time()
        
        try:
            if not returns or len(returns) < window:
                return {
                    "success": False,
                    "error": f"Need at least {window} return observations for volatility analysis"
                }
            
            returns_array = np.array(returns)
            
            # Calculate different volatility measures
            historical_vol = np.std(returns_array, ddof=1) * np.sqrt(252)  # Annualized
            rolling_vol = []
            
            # Rolling volatility
            for i in range(window-1, len(returns_array)):
                window_returns = returns_array[i-window+1:i+1]
                vol = np.std(window_returns, ddof=1) * np.sqrt(252)
                rolling_vol.append(vol)
            
            current_vol = rolling_vol[-1] if rolling_vol else historical_vol
            
            # Volatility statistics
            vol_percentiles = {
                "25th": float(np.percentile(rolling_vol, 25)) if rolling_vol else historical_vol,
                "50th": float(np.percentile(rolling_vol, 50)) if rolling_vol else historical_vol,
                "75th": float(np.percentile(rolling_vol, 75)) if rolling_vol else historical_vol,
                "95th": float(np.percentile(rolling_vol, 95)) if rolling_vol else historical_vol
            }
            
            # Volatility regime assessment
            regime = "low" if current_vol < vol_percentiles["50th"] else "high"
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "volatility_metrics",
                "computation_time": computation_time,
                "historical_volatility": float(historical_vol),
                "current_volatility": float(current_vol),
                "rolling_volatility": rolling_vol,
                "volatility_percentiles": vol_percentiles,
                "volatility_regime": regime,
                "periods_analyzed": len(returns),
                "summary": f"Current volatility: {current_vol:.2%} (annualized), Regime: {regime}"
            }
            
        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            return {
                "success": False,
                "method": "volatility_metrics",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    @tool
    def calculate_information_ratio(self, portfolio_returns: List[float], benchmark_returns: List[float]) -> Dict[str, Any]:
        """
        Calculate Information Ratio - key metric for active trading performance
        
        Args:
            portfolio_returns: Portfolio/strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dict with information ratio analysis
        """
        start_time = time.time()
        
        try:
            if len(portfolio_returns) != len(benchmark_returns):
                return {
                    "success": False,
                    "error": "Portfolio and benchmark return series must have same length"
                }
            
            if len(portfolio_returns) < 10:
                return {
                    "success": False,
                    "error": "Need at least 10 return observations"
                }
            
            portfolio = np.array(portfolio_returns)
            benchmark = np.array(benchmark_returns)
            
            # Calculate excess returns
            excess_returns = portfolio - benchmark
            
            # Calculate Information Ratio components
            excess_return_mean = np.mean(excess_returns)
            tracking_error = np.std(excess_returns, ddof=1)
            
            # Information Ratio
            info_ratio = excess_return_mean / tracking_error if tracking_error > 0 else 0
            
            # Additional statistics
            hit_rate = np.sum(excess_returns > 0) / len(excess_returns)
            max_excess = np.max(excess_returns)
            min_excess = np.min(excess_returns)
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "information_ratio",
                "computation_time": computation_time,
                "information_ratio": float(info_ratio),
                "excess_return": float(excess_return_mean),
                "tracking_error": float(tracking_error),
                "hit_rate": float(hit_rate),
                "max_excess_return": float(max_excess),
                "min_excess_return": float(min_excess),
                "periods_analyzed": len(portfolio_returns),
                "summary": f"Information Ratio: {info_ratio:.3f}, Hit Rate: {hit_rate:.1%}"
            }
            
        except Exception as e:
            logger.error(f"Information ratio calculation failed: {e}")
            return {
                "success": False,
                "method": "information_ratio",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    @tool
    def calculate_calmar_ratio(self, returns: List[float]) -> Dict[str, Any]:
        """
        Calculate Calmar Ratio - annualized return divided by maximum drawdown
        
        Args:
            returns: Return series data
            
        Returns:
            Dict with Calmar ratio analysis
        """
        start_time = time.time()
        
        try:
            if not returns or len(returns) < 20:
                return {
                    "success": False,
                    "error": "Need at least 20 return observations for Calmar ratio"
                }
            
            # Calculate maximum drawdown
            dd_result = self.calculate_maximum_drawdown(returns)
            if not dd_result.get("success"):
                return dd_result
            
            max_drawdown = dd_result["max_drawdown"]
            
            # Calculate annualized return
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            annualized_return = (1 + mean_return) ** 252 - 1  # Assuming daily returns
            
            # Calmar Ratio
            calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "calmar_ratio",
                "computation_time": computation_time,
                "calmar_ratio": float(calmar_ratio),
                "annualized_return": float(annualized_return),
                "max_drawdown": float(max_drawdown),
                "periods_analyzed": len(returns),
                "summary": f"Calmar Ratio: {calmar_ratio:.3f} (Return/Drawdown risk-adjusted)"
            }
            
        except Exception as e:
            logger.error(f"Calmar ratio calculation failed: {e}")
            return {
                "success": False,
                "method": "calmar_ratio",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    # TIME SERIES ANALYSIS TOOLS
    
    @tool
    def detect_correlation_regimes(self, correlation_series: List[float], 
                                 threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect correlation regime changes in time series
        
        Args:
            correlation_series: Time series of correlation values
            threshold: Threshold for regime change detection
            
        Returns:
            Dict with regime analysis
        """
        start_time = time.time()
        
        try:
            if not correlation_series or len(correlation_series) < 10:
                return {
                    "success": False,
                    "error": "Need at least 10 correlation observations"
                }
            
            corrs = np.array(correlation_series)
            regimes = []
            regime_changes = []
            
            # Identify regime changes
            current_regime = "stable" if abs(corrs[0]) > threshold else "breakdown"
            regime_start = 0
            
            for i in range(1, len(corrs)):
                if abs(corrs[i]) > threshold:
                    new_regime = "stable"
                else:
                    new_regime = "breakdown"
                
                if new_regime != current_regime:
                    # End current regime
                    regimes.append({
                        "regime": current_regime,
                        "start_idx": regime_start,
                        "end_idx": i - 1,
                        "duration": i - regime_start,
                        "avg_correlation": float(np.mean(np.abs(corrs[regime_start:i])))
                    })
                    
                    regime_changes.append({
                        "change_idx": i,
                        "from_regime": current_regime,
                        "to_regime": new_regime,
                        "correlation_value": float(corrs[i])
                    })
                    
                    current_regime = new_regime
                    regime_start = i
            
            # Add final regime
            regimes.append({
                "regime": current_regime,
                "start_idx": regime_start,
                "end_idx": len(corrs) - 1,
                "duration": len(corrs) - regime_start,
                "avg_correlation": float(np.mean(np.abs(corrs[regime_start:])))
            })
            
            # Calculate regime statistics
            stable_periods = [r for r in regimes if r["regime"] == "stable"]
            breakdown_periods = [r for r in regimes if r["regime"] == "breakdown"]
            
            avg_stable_duration = np.mean([r["duration"] for r in stable_periods]) if stable_periods else 0
            avg_breakdown_duration = np.mean([r["duration"] for r in breakdown_periods]) if breakdown_periods else 0
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "correlation_regime_detection",
                "computation_time": computation_time,
                "threshold": threshold,
                "regimes": regimes,
                "regime_changes": regime_changes,
                "total_regimes": len(regimes),
                "stable_regimes": len(stable_periods),
                "breakdown_regimes": len(breakdown_periods),
                "avg_stable_duration": float(avg_stable_duration),
                "avg_breakdown_duration": float(avg_breakdown_duration),
                "current_regime": regimes[-1]["regime"] if regimes else "unknown",
                "regime_stability": len(stable_periods) / len(regimes) if regimes else 0,
                "summary": f"Detected {len(regimes)} correlation regimes, currently in {regimes[-1]['regime'] if regimes else 'unknown'} regime"
            }
            
        except Exception as e:
            logger.error(f"Correlation regime detection failed: {e}")
            return {
                "success": False,
                "method": "correlation_regime_detection",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    # SIGNAL PROCESSING AND EARLY WARNING TOOLS
    
    @tool
    def generate_early_warning_signals(self, correlation_data: Dict[str, Any],
                                     price_data: Dict[str, List[float]],
                                     signal_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate early warning signals based on correlation analysis
        
        Args:
            correlation_data: Results from correlation analysis
            price_data: Current price data for assets
            signal_thresholds: Custom thresholds for signal generation
            
        Returns:
            Dict with early warning signals and recommendations
        """
        start_time = time.time()
        
        try:
            # Default signal thresholds
            default_thresholds = {
                "correlation_breakdown": -0.3,  # When correlation drops below this
                "correlation_spike": 0.7,       # When correlation spikes above this
                "volatility_threshold": 0.05,   # 5% price movement threshold
                "lead_lag_confidence": 0.5      # Minimum correlation for lead-lag signals
            }
            
            thresholds = {**default_thresholds, **(signal_thresholds or {})}
            signals = []
            
            # Extract current correlations if available
            current_correlations = correlation_data.get("correlation_matrix", {})
            lead_lag_data = correlation_data.get("lead_lag_analysis", {})
            
            # Check for correlation breakdown signals
            for asset_pair, corr_value in current_correlations.items():
                if isinstance(corr_value, dict):
                    for other_asset, corr in corr_value.items():
                        if isinstance(corr, (int, float)) and corr < thresholds["correlation_breakdown"]:
                            signals.append({
                                "signal_type": "correlation_breakdown",
                                "asset_pair": f"{asset_pair}/{other_asset}",
                                "correlation_value": float(corr),
                                "severity": "high" if corr < -0.5 else "medium",
                                "action": "reduce_exposure",
                                "description": f"Correlation breakdown detected between {asset_pair} and {other_asset}"
                            })
            
            # Check for correlation spike signals  
            for asset_pair, corr_value in current_correlations.items():
                if isinstance(corr_value, dict):
                    for other_asset, corr in corr_value.items():
                        if isinstance(corr, (int, float)) and corr > thresholds["correlation_spike"]:
                            signals.append({
                                "signal_type": "correlation_spike",
                                "asset_pair": f"{asset_pair}/{other_asset}",
                                "correlation_value": float(corr),
                                "severity": "medium",
                                "action": "diversification_risk",
                                "description": f"High correlation spike between {asset_pair} and {other_asset}"
                            })
            
            # Check for early warning based on lead-lag relationships
            if lead_lag_data.get("early_warning_capability"):
                leader = lead_lag_data.get("leader", "")
                follower = lead_lag_data.get("follower", "")
                lead_time = lead_lag_data.get("lead_time_periods", 0)
                
                if leader in price_data and len(price_data[leader]) >= 2:
                    # Check recent price movement in leader
                    recent_change = (price_data[leader][-1] - price_data[leader][-2]) / price_data[leader][-2]
                    
                    if abs(recent_change) > thresholds["volatility_threshold"]:
                        signals.append({
                            "signal_type": "early_warning",
                            "leader_asset": leader,
                            "follower_asset": follower,
                            "lead_time_periods": lead_time,
                            "leader_price_change": float(recent_change),
                            "severity": "high" if abs(recent_change) > 0.1 else "medium",
                            "action": "prepare_for_follow_through",
                            "description": f"{leader} moved {recent_change:.2%}, expect {follower} to follow in {lead_time} periods"
                        })
            
            # Prioritize signals by severity and recency
            high_severity_signals = [s for s in signals if s.get("severity") == "high"]
            medium_severity_signals = [s for s in signals if s.get("severity") == "medium"]
            low_severity_signals = [s for s in signals if s.get("severity") == "low"]
            
            # Generate overall risk assessment
            if high_severity_signals:
                overall_risk = "high"
                recommendation = "Immediate action required - review positions and risk exposure"
            elif medium_severity_signals:
                overall_risk = "medium"  
                recommendation = "Monitor closely - consider position adjustments"
            elif low_severity_signals:
                overall_risk = "low"
                recommendation = "Normal market conditions - maintain current strategy"
            else:
                overall_risk = "minimal"
                recommendation = "No significant signals detected"
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "early_warning_signals",
                "computation_time": computation_time,
                "signal_thresholds": thresholds,
                "total_signals": len(signals),
                "high_severity_signals": len(high_severity_signals),
                "medium_severity_signals": len(medium_severity_signals),
                "low_severity_signals": len(low_severity_signals),
                "signals": signals,
                "overall_risk_level": overall_risk,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat(),
                "summary": f"Generated {len(signals)} signals, risk level: {overall_risk}"
            }
            
        except Exception as e:
            logger.error(f"Early warning signal generation failed: {e}")
            return {
                "success": False,
                "method": "early_warning_signals",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    # BACKTESTING FRAMEWORK
    
    @tool
    def backtest_correlation_strategy(self, fx_data: List[float], crypto_data: List[float],
                                    strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtest FX-crypto correlation trading strategy
        
        Args:
            fx_data: FX pair price data
            crypto_data: Crypto price data (same timeframe)
            strategy_config: Strategy parameters
            
        Returns:
            Dict with backtest results and performance metrics
        """
        start_time = time.time()
        
        try:
            # Default strategy configuration
            config = {
                "correlation_window": 20,
                "signal_threshold": 0.6,
                "stop_loss": -0.02,  # -2%
                "take_profit": 0.04,  # +4%
                "position_size": 0.01,  # 1% of capital per trade
                "initial_capital": 100000,
                **strategy_config
            }
            
            if len(fx_data) != len(crypto_data) or len(fx_data) < config["correlation_window"] * 2:
                return {
                    "success": False,
                    "error": "Insufficient or mismatched data for backtesting"
                }
            
            # Convert to returns
            fx_returns = np.diff(fx_data) / fx_data[:-1]
            crypto_returns = np.diff(crypto_data) / crypto_data[:-1]
            
            # Backtest simulation
            positions = []  # Track all positions
            current_position = None
            capital = config["initial_capital"]
            equity_curve = [capital]
            
            # Calculate rolling correlations for signals
            for i in range(config["correlation_window"], len(fx_returns)):
                # Calculate correlation for signal generation
                window_fx = fx_returns[i-config["correlation_window"]:i]
                window_crypto = crypto_returns[i-config["correlation_window"]:i]
                correlation = np.corrcoef(window_fx, window_crypto)[0, 1]
                
                # Generate trading signals
                if abs(correlation) > config["signal_threshold"]:
                    # Strong correlation detected - expect crypto to follow FX
                    fx_signal = 1 if fx_returns[i-1] > 0 else -1  # Previous FX move direction
                    
                    # Enter position if no current position
                    if current_position is None:
                        position_value = capital * config["position_size"]
                        current_position = {
                            "entry_index": i,
                            "entry_price": crypto_data[i],
                            "direction": fx_signal,  # Long/short crypto based on FX signal
                            "position_size": position_value / crypto_data[i],
                            "stop_loss": crypto_data[i] * (1 + fx_signal * config["stop_loss"]),
                            "take_profit": crypto_data[i] * (1 + fx_signal * config["take_profit"])
                        }
                
                # Check exit conditions for current position
                if current_position is not None and i < len(crypto_data) - 1:
                    current_price = crypto_data[i+1]
                    
                    # Calculate unrealized P&L
                    pnl = current_position["position_size"] * (current_price - current_position["entry_price"]) * current_position["direction"]
                    
                    # Exit conditions
                    exit_position = False
                    exit_reason = ""
                    
                    if current_position["direction"] == 1:  # Long position
                        if current_price <= current_position["stop_loss"]:
                            exit_position = True
                            exit_reason = "stop_loss"
                        elif current_price >= current_position["take_profit"]:
                            exit_position = True
                            exit_reason = "take_profit"
                    else:  # Short position
                        if current_price >= current_position["stop_loss"]:
                            exit_position = True
                            exit_reason = "stop_loss"
                        elif current_price <= current_position["take_profit"]:
                            exit_position = True
                            exit_reason = "take_profit"
                    
                    # Force exit at end of data
                    if i >= len(crypto_data) - 2:
                        exit_position = True
                        exit_reason = "end_of_data"
                    
                    if exit_position:
                        # Close position
                        realized_pnl = pnl
                        capital += realized_pnl
                        
                        positions.append({
                            "entry_index": current_position["entry_index"],
                            "exit_index": i + 1,
                            "entry_price": current_position["entry_price"],
                            "exit_price": current_price,
                            "direction": current_position["direction"],
                            "pnl": realized_pnl,
                            "pnl_percent": realized_pnl / (current_position["position_size"] * current_position["entry_price"]),
                            "exit_reason": exit_reason,
                            "duration": (i + 1) - current_position["entry_index"]
                        })
                        
                        current_position = None
                
                equity_curve.append(capital)
            
            # Calculate performance metrics
            total_trades = len(positions)
            winning_trades = len([p for p in positions if p["pnl"] > 0])
            losing_trades = total_trades - winning_trades
            
            if total_trades > 0:
                win_rate = winning_trades / total_trades
                avg_win = np.mean([p["pnl"] for p in positions if p["pnl"] > 0]) if winning_trades > 0 else 0
                avg_loss = np.mean([p["pnl"] for p in positions if p["pnl"] < 0]) if losing_trades > 0 else 0
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
                
                # Calculate returns-based metrics
                equity_returns = np.diff(equity_curve) / equity_curve[:-1]
                sharpe_ratio = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if np.std(equity_returns) > 0 else 0
                
                # Maximum drawdown
                running_max = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - running_max) / running_max
                max_drawdown = np.min(drawdown)
                
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            total_return = (capital - config["initial_capital"]) / config["initial_capital"]
            
            computation_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "correlation_strategy_backtest",
                "computation_time": computation_time,
                "strategy_config": config,
                "performance_summary": {
                    "total_return": float(total_return),
                    "total_return_percent": float(total_return * 100),
                    "final_capital": float(capital),
                    "max_drawdown": float(max_drawdown),
                    "sharpe_ratio": float(sharpe_ratio),
                    "profit_factor": float(profit_factor) if profit_factor != float('inf') else 999.0
                },
                "trade_statistics": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": float(win_rate),
                    "average_win": float(avg_win),
                    "average_loss": float(avg_loss),
                    "avg_trade_duration": float(np.mean([p["duration"] for p in positions])) if positions else 0
                },
                "equity_curve": equity_curve,
                "individual_trades": positions[-10:] if len(positions) > 10 else positions,  # Last 10 trades
                "periods_tested": len(fx_data),
                "summary": f"Backtest completed: {total_return:.2%} return, {total_trades} trades, {win_rate:.1%} win rate"
            }
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            return {
                "success": False,
                "method": "correlation_strategy_backtest",
                "error": str(e),
                "computation_time": time.time() - start_time
            }
    
    # HELPER METHODS
    
    def _interpret_correlation_strength(self, corr_abs: float) -> str:
        """Interpret correlation strength"""
        if corr_abs >= 0.8:
            return "very_strong"
        elif corr_abs >= 0.6:
            return "strong"
        elif corr_abs >= 0.4:
            return "moderate"
        elif corr_abs >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _detect_correlation_regimes(self, rolling_corrs: np.ndarray, threshold: float = 0.3) -> List[Dict]:
        """Detect correlation regime changes"""
        regimes = []
        if len(rolling_corrs) < 3:
            return regimes
        
        # Simple regime detection based on correlation stability
        for i in range(1, len(rolling_corrs) - 1):
            if abs(rolling_corrs[i] - rolling_corrs[i-1]) > threshold:
                regimes.append({
                    "type": "breakdown",
                    "index": i,
                    "correlation_change": float(rolling_corrs[i] - rolling_corrs[i-1])
                })
            else:
                regimes.append({
                    "type": "stable",
                    "index": i,
                    "correlation": float(rolling_corrs[i])
                })
        
        return regimes
    
    def _analyze_correlation_trend(self, recent_corrs: np.ndarray) -> str:
        """Analyze recent correlation trend"""
        if len(recent_corrs) < 3:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(recent_corrs))
        slope, _, _, _, _ = stats.linregress(x, recent_corrs)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_calmar_ratio(self, prices: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(prices) < 2 or max_drawdown == 0:
            return 0
        
        total_return = (prices[-1] / prices[0]) - 1
        periods = len(prices)
        annual_periods = self.trading_days_per_year * self.hours_per_trading_day
        annualized_return = (1 + total_return) ** (annual_periods / periods) - 1
        
        return annualized_return / abs(max_drawdown)
    
    def _rate_risk_adjusted_performance(self, sharpe: float, sortino: float) -> str:
        """Rate risk-adjusted performance"""
        if sharpe > 2 and sortino > 2:
            return "excellent"
        elif sharpe > 1 and sortino > 1:
            return "good"
        elif sharpe > 0.5 and sortino > 0.5:
            return "fair"
        else:
            return "poor"
    
    def _assess_drawdown_risk(self, max_dd: float, max_duration: int) -> str:
        """Assess drawdown risk level"""
        if max_dd < -0.5 or max_duration > 100:
            return "high_risk"
        elif max_dd < -0.2 or max_duration > 50:
            return "medium_risk"
        else:
            return "low_risk"
    
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for financial analysis skill"""
        return {
            "correlations_computed": len(self.correlation_cache),
            "signals_generated": len(self.signal_history),
            "performance_tracking": self.performance_tracking.copy()
        }