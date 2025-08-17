"""
Comprehensive Quality Validation Rules for 58 Crypto Factors

Implements specific validation rules for each factor category with
statistical outlier detection, consistency checks, and real-time monitoring.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..factors import Factor, FactorCategory, ALL_FACTORS, get_factor_by_name
from ...data.database.models import DataSourceEnum

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of factor validation"""
    passed: bool
    quality_score: float
    failed_rules: List[str]
    warnings: List[str]
    recommendations: List[str]
    outlier_score: Optional[float] = None
    statistical_metrics: Optional[Dict[str, float]] = None


class FactorQualityValidator:
    """
    Comprehensive quality validator for all 58 crypto factors
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.factor_ranges = self._initialize_factor_ranges()
        self.correlation_thresholds = self._initialize_correlation_thresholds()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for each factor"""
        
        rules = {}
        
        # Price Factor Rules (1-10)
        rules.update({
            "spot_price": {
                "min_value": 0.0001,
                "max_change_percent": 50.0,  # 50% max change in 1 period
                "staleness_seconds": 60,
                "required_exchanges": 2,
                "price_deviation_percent": 5.0
            },
            "price_return_1h": {
                "min_value": -100.0,
                "max_value": 200.0,
                "outlier_threshold": 3.0,  # 3 standard deviations
                "min_data_points": 12
            },
            "price_return_24h": {
                "min_value": -100.0,
                "max_value": 500.0,
                "outlier_threshold": 3.5,
                "min_data_points": 24
            },
            "price_return_7d": {
                "min_value": -100.0,
                "max_value": 1000.0,
                "outlier_threshold": 4.0,
                "min_data_points": 7
            },
            "price_return_30d": {
                "min_value": -100.0,
                "max_value": 2000.0,
                "outlier_threshold": 4.0,
                "min_data_points": 30
            },
            "log_return_1h": {
                "min_value": -2.0,
                "max_value": 2.0,
                "outlier_threshold": 3.0
            },
            "vwap_1h": {
                "min_value": 0.0,
                "max_deviation_from_spot": 0.1,  # 10%
                "min_volume": 1000.0,
                "min_trades": 100
            },
            "twap_1h": {
                "min_value": 0.0,
                "min_data_points": 60
            },
            "price_vs_ma_50": {
                "min_value": 0.5,
                "max_value": 2.0,
                "min_data_points": 45
            },
            "price_vs_ma_200": {
                "min_value": 0.2,
                "max_value": 5.0,
                "min_data_points": 180
            }
        })
        
        # Volume Factor Rules (11-18)
        rules.update({
            "spot_volume": {
                "min_value": 0.0,
                "max_spike_ratio": 100.0,
                "min_exchanges": 2
            },
            "volume_24h": {
                "min_value": 0.0,
                "min_data_completeness": 0.9
            },
            "volume_ratio_1h_24h": {
                "min_value": 0.0,
                "max_value": 50.0,
                "min_24h_volume": 10000.0
            },
            "buy_sell_ratio": {
                "min_value": 0.1,
                "max_value": 10.0,
                "min_trades": 50
            },
            "large_trade_volume": {
                "min_value": 0.0,
                "large_trade_threshold": 100000.0,
                "min_sample_trades": 1000
            },
            "volume_momentum": {
                "min_value": -10.0,
                "max_value": 10.0,
                "min_periods": 20
            },
            "obv": {
                "allow_negative": True,
                "min_periods": 100
            },
            "volume_profile": {
                "price_bins": 100,
                "min_volume_per_bin": 0.0,
                "min_total_volume": 100000.0
            }
        })
        
        # Technical Factor Rules (19-28)
        rules.update({
            "rsi_14": {
                "min_value": 0.0,
                "max_value": 100.0,
                "min_periods": 14,
                "overbought_threshold": 70.0,
                "oversold_threshold": 30.0
            },
            "macd_signal": {
                "min_periods": 35,
                "outlier_threshold": 3.0
            },
            "bollinger_position": {
                "min_value": -0.5,
                "max_value": 1.5,
                "min_periods": 20
            },
            "stochastic_k": {
                "min_value": 0.0,
                "max_value": 100.0,
                "min_periods": 14
            },
            "williams_r": {
                "min_value": -100.0,
                "max_value": 0.0,
                "min_periods": 14
            },
            "adx": {
                "min_value": 0.0,
                "max_value": 100.0,
                "min_periods": 28
            },
            "cci": {
                "min_periods": 20,
                "outlier_threshold": 4.0  # CCI can have extreme values
            },
            "mfi": {
                "min_value": 0.0,
                "max_value": 100.0,
                "min_periods": 14,
                "min_volume": 1000.0
            },
            "ichimoku_cloud": {
                "min_periods": 52
            },
            "parabolic_sar": {
                "min_periods": 50
            }
        })
        
        # Volatility Factor Rules (29-35)
        rules.update({
            "volatility_1h": {
                "min_value": 0.0,
                "max_value": 10.0,
                "min_returns": 30
            },
            "volatility_24h": {
                "min_value": 0.0,
                "max_value": 5.0,
                "min_returns": 200
            },
            "garch_volatility": {
                "min_value": 0.0,
                "max_value": 5.0,
                "min_periods": 500
            },
            "volatility_ratio": {
                "min_value": 0.1,
                "max_value": 10.0,
                "min_long_term_vol": 0.01
            },
            "parkinson_volatility": {
                "min_value": 0.0,
                "max_value": 5.0,
                "min_periods": 50
            },
            "atr": {
                "min_value": 0.0,
                "min_periods": 14
            },
            "volatility_skew": {
                "min_value": -3.0,
                "max_value": 3.0,
                "min_returns": 100
            }
        })
        
        # Market Structure Factor Rules (36-42)
        rules.update({
            "bid_ask_spread": {
                "min_value": 0.0,
                "max_value": 0.1,  # 10% max spread
                "max_staleness_ms": 1000
            },
            "order_book_imbalance": {
                "min_value": -1.0,
                "max_value": 1.0,
                "depth_levels": 20,
                "min_depth_usd": 10000.0
            },
            "market_depth_ratio": {
                "min_value": 0.0,
                "max_value": 1.0,
                "min_depth_usd": 50000.0
            },
            "trade_size_distribution": {
                "size_buckets": [100, 1000, 10000, 100000],
                "min_trades": 100,
                "min_sample_size": 500
            },
            "price_impact": {
                "min_value": 0.0,
                "max_value": 0.05,  # 5% max impact
                "test_size_usd": 100000.0,
                "min_depth_levels": 20
            },
            "exchange_flow": {
                "min_exchanges": 2,
                "min_volume_usd": 10000.0
            },
            "liquidation_levels": {
                "price_range_percent": 0.1,
                "min_position_size": 100000.0,
                "data_freshness_minutes": 5
            }
        })
        
        # On-Chain Factor Rules (43-48)
        rules.update({
            "network_hashrate": {
                "min_value": 0.0,
                "max_change_percent": 50.0,
                "applicable_to": ["BTC", "ETH", "LTC"]
            },
            "active_addresses": {
                "min_value": 0,
                "min_addresses": 1000
            },
            "transaction_volume": {
                "min_value": 0.0,
                "min_transactions": 100
            },
            "exchange_balance": {
                "min_value": 0.0,
                "tracked_exchanges": 10
            },
            "nvt_ratio": {
                "min_value": 0.0,
                "max_value": 1000.0,
                "min_tx_volume": 1000000.0
            },
            "whale_movements": {
                "whale_threshold_usd": 1000000.0,
                "confirmation_blocks": 1
            }
        })
        
        # Sentiment Factor Rules (49-52)
        rules.update({
            "social_volume": {
                "min_value": 0,
                "min_mentions": 100,
                "platforms": ["twitter", "reddit", "telegram"]
            },
            "social_sentiment": {
                "min_value": -1.0,
                "max_value": 1.0,
                "min_scored_posts": 50
            },
            "fear_greed_index": {
                "min_value": 0,
                "max_value": 100,
                "components": ["volatility", "momentum", "social", "surveys"]
            },
            "reddit_sentiment": {
                "min_value": -1.0,
                "max_value": 1.0,
                "min_comments": 100,
                "subreddits": ["cryptocurrency", "bitcoin", "ethtrader"]
            }
        })
        
        # Macro Factor Rules (53-55)
        rules.update({
            "dxy_correlation": {
                "min_value": -1.0,
                "max_value": 1.0,
                "correlation_window": 30,
                "min_observations": 20
            },
            "gold_correlation": {
                "min_value": -1.0,
                "max_value": 1.0,
                "min_observations": 20
            },
            "spy_correlation": {
                "min_value": -1.0,
                "max_value": 1.0,
                "min_observations": 20
            }
        })
        
        # DeFi Factor Rules (56-58)
        rules.update({
            "tvl_ratio": {
                "min_value": 0.0,
                "max_value": 10.0,
                "min_tvl": 1000000.0,
                "applicable_to": ["ETH", "SOL", "AVAX", "MATIC"]
            },
            "staking_ratio": {
                "min_value": 0.0,
                "max_value": 1.0,
                "data_freshness_hours": 24,
                "applicable_to": ["ETH", "SOL", "ADA", "DOT"]
            },
            "defi_dominance": {
                "min_value": 0.0,
                "max_value": 1.0,
                "tracked_protocols": 100
            }
        })
        
        return rules
    
    def _initialize_factor_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Initialize expected value ranges for factors"""
        
        ranges = {
            # Price factors - wide ranges for crypto volatility
            "spot_price": (0.0001, 1000000.0),
            "price_return_1h": (-50.0, 50.0),
            "price_return_24h": (-80.0, 200.0),
            "price_return_7d": (-90.0, 500.0),
            "price_return_30d": (-95.0, 1000.0),
            
            # Volume factors
            "volume_ratio_1h_24h": (0.01, 20.0),
            "buy_sell_ratio": (0.2, 5.0),
            
            # Technical indicators
            "rsi_14": (0.0, 100.0),
            "bollinger_position": (0.0, 1.0),
            "stochastic_k": (0.0, 100.0),
            "williams_r": (-100.0, 0.0),
            "adx": (0.0, 100.0),
            "mfi": (0.0, 100.0),
            
            # Volatility measures
            "volatility_1h": (0.0, 3.0),
            "volatility_24h": (0.0, 2.0),
            "volatility_ratio": (0.3, 3.0),
            "atr": (0.0, 1000.0),
            
            # Market structure
            "bid_ask_spread": (0.0, 0.05),
            "order_book_imbalance": (-1.0, 1.0),
            "market_depth_ratio": (0.0, 0.5),
            "price_impact": (0.0, 0.02),
            
            # Sentiment
            "social_sentiment": (-1.0, 1.0),
            "fear_greed_index": (0.0, 100.0),
            "reddit_sentiment": (-1.0, 1.0),
            
            # Correlations
            "dxy_correlation": (-1.0, 1.0),
            "gold_correlation": (-1.0, 1.0),
            "spy_correlation": (-1.0, 1.0),
            
            # DeFi
            "tvl_ratio": (0.0, 5.0),
            "staking_ratio": (0.0, 1.0),
            "defi_dominance": (0.0, 0.5),
        }
        
        return ranges
    
    def _initialize_correlation_thresholds(self) -> Dict[str, float]:
        """Initialize correlation thresholds for related factors"""
        
        return {
            # Price returns should be correlated across timeframes
            "price_return_correlation": 0.3,
            
            # Volatility measures should be positively correlated
            "volatility_correlation": 0.5,
            
            # Technical indicators in same category
            "momentum_correlation": 0.4,
            "oscillator_correlation": 0.3,
            
            # Volume measures
            "volume_correlation": 0.6,
        }
    
    def validate_factor(
        self, 
        factor_name: str, 
        values: pd.Series, 
        symbol: str,
        source: DataSourceEnum,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Comprehensive validation of a single factor
        
        Args:
            factor_name: Name of the factor to validate
            values: Time series of factor values
            symbol: Trading symbol
            source: Data source
            additional_data: Additional context data
            
        Returns:
            ValidationResult with detailed validation information
        """
        
        try:
            factor = get_factor_by_name(factor_name)
            if not factor:
                return ValidationResult(
                    passed=False,
                    quality_score=0.0,
                    failed_rules=[f"Unknown factor: {factor_name}"],
                    warnings=[],
                    recommendations=["Check factor name spelling"]
                )
            
            rules = self.validation_rules.get(factor_name, {})
            failed_rules = []
            warnings = []
            recommendations = []
            quality_scores = []
            
            # 1. Basic value range validation
            range_result = self._validate_value_ranges(factor_name, values, rules)
            quality_scores.append(range_result['score'])
            failed_rules.extend(range_result['failed_rules'])
            warnings.extend(range_result['warnings'])
            
            # 2. Statistical outlier detection
            outlier_result = self._detect_statistical_outliers(values, rules)
            quality_scores.append(outlier_result['score'])
            if outlier_result['outliers_detected']:
                warnings.append(f"Detected {outlier_result['outlier_count']} statistical outliers")
            
            # 3. Data completeness validation
            completeness_result = self._validate_data_completeness(values, rules)
            quality_scores.append(completeness_result['score'])
            if completeness_result['score'] < 0.9:
                failed_rules.append(f"Data completeness too low: {completeness_result['score']:.2f}")
            
            # 4. Time series consistency checks
            consistency_result = self._validate_time_series_consistency(values, factor)
            quality_scores.append(consistency_result['score'])
            failed_rules.extend(consistency_result['failed_rules'])
            warnings.extend(consistency_result['warnings'])
            
            # 5. Factor-specific validation
            specific_result = self._validate_factor_specific_rules(
                factor, values, symbol, rules, additional_data
            )
            quality_scores.append(specific_result['score'])
            failed_rules.extend(specific_result['failed_rules'])
            warnings.extend(specific_result['warnings'])
            recommendations.extend(specific_result['recommendations'])
            
            # 6. Cross-factor consistency (if additional data provided)
            if additional_data:
                cross_result = self._validate_cross_factor_consistency(
                    factor_name, values, additional_data
                )
                quality_scores.append(cross_result['score'])
                warnings.extend(cross_result['warnings'])
            
            # Calculate overall quality score
            overall_quality = np.mean(quality_scores) if quality_scores else 0.0
            
            # Generate recommendations
            if overall_quality < 0.8:
                recommendations.append("Consider re-collecting data from alternative sources")
            if len(failed_rules) > 3:
                recommendations.append("Multiple validation failures - check data pipeline")
            
            # Determine if validation passed
            passed = len(failed_rules) == 0 and overall_quality >= 0.8
            
            return ValidationResult(
                passed=passed,
                quality_score=overall_quality,
                failed_rules=failed_rules,
                warnings=warnings,
                recommendations=recommendations,
                outlier_score=outlier_result.get('outlier_score'),
                statistical_metrics={
                    'mean': values.mean() if len(values) > 0 else 0.0,
                    'std': values.std() if len(values) > 0 else 0.0,
                    'min': values.min() if len(values) > 0 else 0.0,
                    'max': values.max() if len(values) > 0 else 0.0,
                    'count': len(values),
                    'null_count': values.isnull().sum()
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating factor {factor_name}: {str(e)}")
            return ValidationResult(
                passed=False,
                quality_score=0.0,
                failed_rules=[f"Validation error: {str(e)}"],
                warnings=[],
                recommendations=["Check factor calculation implementation"]
            )
    
    def _validate_value_ranges(
        self, 
        factor_name: str, 
        values: pd.Series, 
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate factor values are within expected ranges"""
        
        failed_rules = []
        warnings = []
        
        # Check minimum value
        if 'min_value' in rules:
            min_violations = (values < rules['min_value']).sum()
            if min_violations > 0:
                failed_rules.append(f"Values below minimum: {min_violations} points")
        
        # Check maximum value
        if 'max_value' in rules:
            max_violations = (values > rules['max_value']).sum()
            if max_violations > 0:
                failed_rules.append(f"Values above maximum: {max_violations} points")
        
        # Check expected range if defined
        if factor_name in self.factor_ranges:
            expected_min, expected_max = self.factor_ranges[factor_name]
            out_of_range = ((values < expected_min) | (values > expected_max)).sum()
            out_of_range_pct = out_of_range / len(values) * 100
            
            if out_of_range_pct > 5:  # More than 5% out of expected range
                warnings.append(f"{out_of_range_pct:.1f}% of values outside expected range")
        
        # Calculate score based on violations
        total_violations = len(failed_rules)
        score = max(0.0, 1.0 - (total_violations * 0.2))
        
        return {
            'score': score,
            'failed_rules': failed_rules,
            'warnings': warnings
        }
    
    def _detect_statistical_outliers(
        self, 
        values: pd.Series, 
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect statistical outliers using multiple methods"""
        
        if len(values) < 10:
            return {'score': 1.0, 'outliers_detected': False, 'outlier_count': 0}
        
        clean_values = values.dropna()
        if len(clean_values) == 0:
            return {'score': 0.0, 'outliers_detected': False, 'outlier_count': 0}
        
        outlier_threshold = rules.get('outlier_threshold', 3.0)
        
        # Z-score method
        z_scores = np.abs((clean_values - clean_values.mean()) / clean_values.std())
        z_outliers = (z_scores > outlier_threshold).sum()
        
        # IQR method
        q1 = clean_values.quantile(0.25)
        q3 = clean_values.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_outliers = ((clean_values < iqr_lower) | (clean_values > iqr_upper)).sum()
        
        # Modified Z-score (using median)
        median = clean_values.median()
        mad = np.median(np.abs(clean_values - median))
        modified_z_scores = 0.6745 * (clean_values - median) / mad if mad > 0 else 0
        modified_z_outliers = (np.abs(modified_z_scores) > outlier_threshold).sum()
        
        # Take the most conservative estimate
        total_outliers = max(z_outliers, iqr_outliers, modified_z_outliers)
        outlier_percentage = total_outliers / len(clean_values) * 100
        
        # Calculate outlier score
        outlier_score = min(outlier_percentage / 100.0, 1.0)
        
        # Quality score decreases with outlier percentage
        quality_score = max(0.0, 1.0 - (outlier_percentage / 20.0))  # 20% outliers = 0 score
        
        return {
            'score': quality_score,
            'outliers_detected': total_outliers > 0,
            'outlier_count': total_outliers,
            'outlier_score': outlier_score,
            'outlier_percentage': outlier_percentage
        }
    
    def _validate_data_completeness(
        self, 
        values: pd.Series, 
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data completeness and gaps"""
        
        total_points = len(values)
        null_points = values.isnull().sum()
        completeness = 1.0 - (null_points / total_points) if total_points > 0 else 0.0
        
        min_completeness = rules.get('min_data_completeness', 0.9)
        min_data_points = rules.get('min_data_points', 10)
        
        score = 1.0
        if completeness < min_completeness:
            score *= completeness / min_completeness
        
        if total_points < min_data_points:
            score *= total_points / min_data_points
        
        return {
            'score': min(score, 1.0),
            'completeness': completeness,
            'total_points': total_points,
            'null_points': null_points
        }
    
    def _validate_time_series_consistency(
        self, 
        values: pd.Series, 
        factor: Factor
    ) -> Dict[str, Any]:
        """Validate time series consistency and patterns"""
        
        failed_rules = []
        warnings = []
        
        if len(values) < 5:
            return {'score': 0.5, 'failed_rules': [], 'warnings': ['Insufficient data for consistency check']}
        
        # Check for unrealistic jumps
        if factor.category in [FactorCategory.PRICE, FactorCategory.VOLUME]:
            diff_pct = values.pct_change().abs()
            extreme_changes = (diff_pct > 0.5).sum()  # >50% change
            if extreme_changes > len(values) * 0.05:  # More than 5% of points
                warnings.append(f"Detected {extreme_changes} extreme value changes")
        
        # Check for flat periods (might indicate stale data)
        if factor.category != FactorCategory.TECHNICAL:  # Technical indicators can be flat
            consecutive_same = 0
            max_consecutive = 0
            prev_val = None
            
            for val in values.dropna():
                if prev_val is not None and abs(val - prev_val) < 1e-8:
                    consecutive_same += 1
                    max_consecutive = max(max_consecutive, consecutive_same)
                else:
                    consecutive_same = 0
                prev_val = val
            
            if max_consecutive > len(values) * 0.1:  # More than 10% consecutive same values
                warnings.append(f"Detected {max_consecutive} consecutive identical values")
        
        # Check for monotonic trends (might indicate calculation error)
        if len(values) > 20:
            is_monotonic_increasing = values.is_monotonic_increasing
            is_monotonic_decreasing = values.is_monotonic_decreasing
            
            if is_monotonic_increasing or is_monotonic_decreasing:
                warnings.append("Values are strictly monotonic - check calculation")
        
        # Calculate consistency score
        score = 1.0 - (len(warnings) * 0.1)  # Each warning reduces score by 10%
        score = max(0.0, score)
        
        return {
            'score': score,
            'failed_rules': failed_rules,
            'warnings': warnings
        }
    
    def _validate_factor_specific_rules(
        self,
        factor: Factor,
        values: pd.Series,
        symbol: str,
        rules: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply factor-specific validation rules"""
        
        failed_rules = []
        warnings = []
        recommendations = []
        
        # Category-specific validations
        if factor.category == FactorCategory.TECHNICAL:
            result = self._validate_technical_indicator(factor.name, values, rules)
        elif factor.category == FactorCategory.VOLUME:
            result = self._validate_volume_factor(factor.name, values, rules, additional_data)
        elif factor.category == FactorCategory.VOLATILITY:
            result = self._validate_volatility_factor(factor.name, values, rules)
        elif factor.category == FactorCategory.ONCHAIN:
            result = self._validate_onchain_factor(factor.name, values, symbol, rules)
        elif factor.category == FactorCategory.SENTIMENT:
            result = self._validate_sentiment_factor(factor.name, values, rules)
        elif factor.category == FactorCategory.MACRO:
            result = self._validate_macro_factor(factor.name, values, rules)
        elif factor.category == FactorCategory.DEFI:
            result = self._validate_defi_factor(factor.name, values, symbol, rules)
        else:
            result = {'score': 1.0, 'failed_rules': [], 'warnings': [], 'recommendations': []}
        
        return result
    
    def _validate_technical_indicator(
        self, 
        factor_name: str, 
        values: pd.Series, 
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate technical indicators"""
        
        failed_rules = []
        warnings = []
        recommendations = []
        
        # RSI-specific validation
        if factor_name == "rsi_14":
            overbought = (values > rules.get('overbought_threshold', 70)).sum()
            oversold = (values < rules.get('oversold_threshold', 30)).sum()
            
            if overbought == 0 and oversold == 0:
                warnings.append("RSI never reaches overbought/oversold levels")
            
            # RSI should not be constant
            if values.std() < 5:
                warnings.append("RSI has very low variability")
        
        # Bollinger Bands position validation
        elif factor_name == "bollinger_position":
            outside_bands = ((values < 0) | (values > 1)).sum()
            if outside_bands > len(values) * 0.05:  # More than 5% outside bands
                warnings.append(f"{outside_bands} values outside Bollinger Bands")
        
        # MACD validation
        elif factor_name == "macd_signal":
            # MACD should oscillate around zero
            if abs(values.mean()) > values.std():
                warnings.append("MACD signal may have drift")
        
        score = 1.0 - (len(failed_rules) * 0.3) - (len(warnings) * 0.1)
        return {
            'score': max(0.0, score),
            'failed_rules': failed_rules,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _validate_volume_factor(
        self, 
        factor_name: str, 
        values: pd.Series, 
        rules: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate volume-based factors"""
        
        failed_rules = []
        warnings = []
        
        # Volume should be non-negative
        negative_volume = (values < 0).sum()
        if negative_volume > 0:
            failed_rules.append(f"Negative volume detected: {negative_volume} points")
        
        # Check for volume spikes
        if factor_name == "spot_volume":
            max_spike_ratio = rules.get('max_spike_ratio', 100.0)
            median_volume = values.median()
            
            if median_volume > 0:
                spikes = (values > median_volume * max_spike_ratio).sum()
                if spikes > 0:
                    warnings.append(f"Detected {spikes} volume spikes > {max_spike_ratio}x median")
        
        # Buy-sell ratio validation
        elif factor_name == "buy_sell_ratio":
            extreme_ratios = ((values < 0.1) | (values > 10.0)).sum()
            if extreme_ratios > len(values) * 0.1:
                warnings.append(f"Extreme buy-sell ratios detected: {extreme_ratios} points")
        
        score = 1.0 - (len(failed_rules) * 0.4) - (len(warnings) * 0.1)
        return {
            'score': max(0.0, score),
            'failed_rules': failed_rules,
            'warnings': warnings,
            'recommendations': []
        }
    
    def _validate_volatility_factor(
        self, 
        factor_name: str, 
        values: pd.Series, 
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate volatility measures"""
        
        failed_rules = []
        warnings = []
        
        # Volatility should be non-negative
        negative_vol = (values < 0).sum()
        if negative_vol > 0:
            failed_rules.append(f"Negative volatility detected: {negative_vol} points")
        
        # Check for unrealistic volatility levels
        if factor_name in ["volatility_1h", "volatility_24h"]:
            extreme_vol = (values > 2.0).sum()  # >200% annualized
            if extreme_vol > len(values) * 0.01:  # More than 1% of points
                warnings.append(f"Extreme volatility levels detected: {extreme_vol} points")
        
        # GARCH volatility should be smooth
        elif factor_name == "garch_volatility":
            if len(values) > 5:
                vol_changes = values.diff().abs()
                large_changes = (vol_changes > vol_changes.quantile(0.95)).sum()
                if large_changes > len(values) * 0.1:
                    warnings.append("GARCH volatility shows high variability")
        
        score = 1.0 - (len(failed_rules) * 0.4) - (len(warnings) * 0.1)
        return {
            'score': max(0.0, score),
            'failed_rules': failed_rules,
            'warnings': warnings,
            'recommendations': []
        }
    
    def _validate_onchain_factor(
        self, 
        factor_name: str, 
        values: pd.Series, 
        symbol: str,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate on-chain metrics"""
        
        failed_rules = []
        warnings = []
        
        # Check if factor is applicable to this symbol
        applicable_to = rules.get('applicable_to', [])
        if applicable_to and not any(symbol.startswith(s) for s in applicable_to):
            warnings.append(f"Factor {factor_name} may not be applicable to {symbol}")
        
        # Network hashrate validation
        if factor_name == "network_hashrate":
            if symbol.startswith("BTC") or symbol.startswith("ETH"):
                # Hashrate should be relatively stable
                if len(values) > 10:
                    hashrate_changes = values.pct_change().abs()
                    large_changes = (hashrate_changes > 0.2).sum()  # >20% change
                    if large_changes > len(values) * 0.1:
                        warnings.append("High hashrate volatility detected")
        
        # Active addresses should be positive integers
        elif factor_name == "active_addresses":
            non_integer = (values != values.round()).sum()
            if non_integer > 0:
                warnings.append("Non-integer active address counts detected")
        
        score = 1.0 - (len(failed_rules) * 0.3) - (len(warnings) * 0.1)
        return {
            'score': max(0.0, score),
            'failed_rules': failed_rules,
            'warnings': warnings,
            'recommendations': []
        }
    
    def _validate_sentiment_factor(
        self, 
        factor_name: str, 
        values: pd.Series, 
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate sentiment metrics"""
        
        failed_rules = []
        warnings = []
        
        # Sentiment scores should be bounded
        if factor_name in ["social_sentiment", "reddit_sentiment"]:
            out_of_bounds = ((values < -1.0) | (values > 1.0)).sum()
            if out_of_bounds > 0:
                failed_rules.append(f"Sentiment values out of [-1, 1] range: {out_of_bounds}")
        
        # Fear & Greed Index validation
        elif factor_name == "fear_greed_index":
            out_of_bounds = ((values < 0) | (values > 100)).sum()
            if out_of_bounds > 0:
                failed_rules.append(f"Fear & Greed values out of [0, 100] range: {out_of_bounds}")
            
            # Should show variability (not constant)
            if values.std() < 5:
                warnings.append("Fear & Greed Index shows low variability")
        
        score = 1.0 - (len(failed_rules) * 0.4) - (len(warnings) * 0.1)
        return {
            'score': max(0.0, score),
            'failed_rules': failed_rules,
            'warnings': warnings,
            'recommendations': []
        }
    
    def _validate_macro_factor(
        self, 
        factor_name: str, 
        values: pd.Series, 
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate macroeconomic correlations"""
        
        failed_rules = []
        warnings = []
        
        # Correlations should be bounded [-1, 1]
        if "correlation" in factor_name:
            out_of_bounds = ((values < -1.0) | (values > 1.0)).sum()
            if out_of_bounds > 0:
                failed_rules.append(f"Correlation values out of [-1, 1] range: {out_of_bounds}")
            
            # Correlations should show some variability
            if values.std() < 0.1:
                warnings.append("Correlation shows very low variability")
        
        score = 1.0 - (len(failed_rules) * 0.4) - (len(warnings) * 0.1)
        return {
            'score': max(0.0, score),
            'failed_rules': failed_rules,
            'warnings': warnings,
            'recommendations': []
        }
    
    def _validate_defi_factor(
        self, 
        factor_name: str, 
        values: pd.Series, 
        symbol: str,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate DeFi-specific metrics"""
        
        failed_rules = []
        warnings = []
        
        # Check if factor is applicable to this symbol
        applicable_to = rules.get('applicable_to', [])
        if applicable_to and not any(symbol.startswith(s) for s in applicable_to):
            warnings.append(f"DeFi factor {factor_name} may not be applicable to {symbol}")
        
        # Ratios should be non-negative
        if factor_name in ["tvl_ratio", "staking_ratio", "defi_dominance"]:
            negative_values = (values < 0).sum()
            if negative_values > 0:
                failed_rules.append(f"Negative ratio values detected: {negative_values}")
        
        # Staking ratio should be ≤ 1
        if factor_name == "staking_ratio":
            above_one = (values > 1.0).sum()
            if above_one > 0:
                failed_rules.append(f"Staking ratio > 100% detected: {above_one} points")
        
        score = 1.0 - (len(failed_rules) * 0.4) - (len(warnings) * 0.1)
        return {
            'score': max(0.0, score),
            'failed_rules': failed_rules,
            'warnings': warnings,
            'recommendations': []
        }
    
    def _validate_cross_factor_consistency(
        self, 
        factor_name: str, 
        values: pd.Series, 
        additional_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate consistency across related factors"""
        
        warnings = []
        
        # Check price return correlations across timeframes
        if factor_name == "price_return_1h" and "price_return_24h" in additional_data:
            corr = values.corr(additional_data["price_return_24h"])
            if abs(corr) < self.correlation_thresholds["price_return_correlation"]:
                warnings.append(f"Low correlation between 1h and 24h returns: {corr:.3f}")
        
        # Check volatility measure consistency
        if factor_name == "volatility_1h" and "volatility_24h" in additional_data:
            corr = values.corr(additional_data["volatility_24h"])
            if corr < self.correlation_thresholds["volatility_correlation"]:
                warnings.append(f"Low correlation between volatility measures: {corr:.3f}")
        
        # Check technical indicator consistency
        if factor_name == "rsi_14" and "stochastic_k" in additional_data:
            corr = values.corr(additional_data["stochastic_k"])
            if corr < self.correlation_thresholds["oscillator_correlation"]:
                warnings.append(f"Low correlation between momentum oscillators: {corr:.3f}")
        
        score = 1.0 - (len(warnings) * 0.1)
        return {
            'score': max(0.0, score),
            'warnings': warnings
        }
    
    def validate_factor_batch(
        self, 
        factor_data: Dict[str, pd.Series], 
        symbol: str,
        source: DataSourceEnum
    ) -> Dict[str, ValidationResult]:
        """Validate multiple factors together for cross-validation"""
        
        results = {}
        
        # Validate each factor individually first
        for factor_name, values in factor_data.items():
            results[factor_name] = self.validate_factor(
                factor_name, values, symbol, source, factor_data
            )
        
        # Add batch-specific validations
        self._validate_factor_batch_consistency(factor_data, results)
        
        return results
    
    def _validate_factor_batch_consistency(
        self, 
        factor_data: Dict[str, pd.Series], 
        results: Dict[str, ValidationResult]
    ):
        """Add batch-level consistency checks"""
        
        # Check for systematic issues across multiple factors
        low_quality_factors = [
            name for name, result in results.items() 
            if result.quality_score < 0.8
        ]
        
        if len(low_quality_factors) > len(results) * 0.3:  # >30% low quality
            for name in results:
                results[name].warnings.append(
                    f"Systematic quality issues detected across {len(low_quality_factors)} factors"
                )
                results[name].recommendations.append(
                    "Check data source connectivity and calculation pipeline"
                )