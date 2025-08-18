"""
ML Feature Store for efficient feature engineering and data access
Production-ready feature management system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pickle
import json
import hashlib
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Feature store should use database data, not direct market data clients
from ..memory.optimized_cache import get_cache_manager
from ..config.environment import get_processing_config, get_feature_flags
from ..processing.parallel_executor import get_parallel_executor
from .feature_cache import FeatureCachePersistence

logger = logging.getLogger(__name__)

# Simple metrics for Vercel compatibility
class SimpleMetrics:
    def counter(self, name: str, value: int = 1, labels: dict = None):
        pass

business_metrics = SimpleMetrics()


class FeatureDefinition:
    """Definition of a feature with metadata"""
    
    def __init__(self, name: str, dtype: str, description: str, 
                 compute_func: callable, dependencies: List[str] = None):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.compute_func = compute_func
        self.dependencies = dependencies or []
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }


class FeatureStore:
    """Centralized feature store for ML pipelines"""
    
    def __init__(self):
        self.features = {}
        self.cache_manager = get_cache_manager()
        # Use database for feature data instead of direct market client
        from ...infrastructure.database.unified_database import UnifiedDatabase
        self.database = UnifiedDatabase()
        self.config = get_processing_config()
        self.flags = get_feature_flags()
        self.cache_ttl = self.flags.cache_ttl_seconds
        
        # Initialize feature cache persistence
        self.feature_cache = FeatureCachePersistence(self.database)
        
        # Use environment-aware parallel executor
        if self.config['parallel_processing']:
            self.executor = get_parallel_executor()
        else:
            self.executor = None
            
        self._register_features()
        logger.info(f"Feature Store initialized: parallel={self.config['parallel_processing']}, cache_ttl={self.cache_ttl}s")
    
    def _register_features(self):
        """Register all available features"""
        # Price features
        self.register_feature(FeatureDefinition(
            name="price_change_1h",
            dtype="float",
            description="Price change in last 1 hour",
            compute_func=self._compute_price_change_1h,
            dependencies=["close"]
        ))
        
        self.register_feature(FeatureDefinition(
            name="price_change_24h",
            dtype="float",
            description="Price change in last 24 hours",
            compute_func=self._compute_price_change_24h,
            dependencies=["close"]
        ))
        
        self.register_feature(FeatureDefinition(
            name="price_change_7d",
            dtype="float",
            description="Price change in last 7 days",
            compute_func=self._compute_price_change_7d,
            dependencies=["close"]
        ))
        
        # Technical indicators
        self.register_feature(FeatureDefinition(
            name="rsi_14",
            dtype="float",
            description="14-period RSI",
            compute_func=self._compute_rsi_14,
            dependencies=["close"]
        ))
        
        self.register_feature(FeatureDefinition(
            name="macd_signal",
            dtype="float",
            description="MACD signal line",
            compute_func=self._compute_macd_signal,
            dependencies=["close"]
        ))
        
        self.register_feature(FeatureDefinition(
            name="bb_position",
            dtype="float",
            description="Position within Bollinger Bands",
            compute_func=self._compute_bb_position,
            dependencies=["close"]
        ))
        
        # Volume features
        self.register_feature(FeatureDefinition(
            name="volume_ratio_20",
            dtype="float",
            description="Volume ratio to 20-period average",
            compute_func=self._compute_volume_ratio,
            dependencies=["volume"]
        ))
        
        self.register_feature(FeatureDefinition(
            name="price_volume_correlation",
            dtype="float",
            description="30-day price-volume correlation",
            compute_func=self._compute_price_volume_corr,
            dependencies=["close", "volume"]
        ))
        
        # Volatility features
        self.register_feature(FeatureDefinition(
            name="volatility_20",
            dtype="float",
            description="20-day volatility",
            compute_func=self._compute_volatility_20,
            dependencies=["close"]
        ))
        
        self.register_feature(FeatureDefinition(
            name="volatility_ratio",
            dtype="float",
            description="Current vs historical volatility ratio",
            compute_func=self._compute_volatility_ratio,
            dependencies=["close"]
        ))
        
        # Market structure
        self.register_feature(FeatureDefinition(
            name="support_level",
            dtype="float",
            description="Recent support level",
            compute_func=self._compute_support_level,
            dependencies=["low"]
        ))
        
        self.register_feature(FeatureDefinition(
            name="resistance_level",
            dtype="float",
            description="Recent resistance level",
            compute_func=self._compute_resistance_level,
            dependencies=["high"]
        ))
        
        # Trend features
        self.register_feature(FeatureDefinition(
            name="trend_strength",
            dtype="float",
            description="Trend strength indicator",
            compute_func=self._compute_trend_strength,
            dependencies=["close"]
        ))
        
        self.register_feature(FeatureDefinition(
            name="momentum_10",
            dtype="float",
            description="10-period momentum",
            compute_func=self._compute_momentum,
            dependencies=["close"]
        ))
        
        logger.info(f"Registered {len(self.features)} features")
    
    def register_feature(self, feature_def: FeatureDefinition):
        """Register a new feature"""
        self.features[feature_def.name] = feature_def
        business_metrics.counter("feature_store.features_registered", 1)
    
    async def compute_features(self, symbol: str, features: List[str] = None) -> pd.DataFrame:
        """Compute requested features for a symbol"""
        try:
            # Get base data
            data = await self._get_base_data(symbol)
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return pd.DataFrame()
            
            # Select features to compute
            if features is None:
                features = list(self.features.keys())
            
            # Check persistent feature cache first
            timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()
            cached_features = await self.feature_cache.get_features(
                symbol, features, timestamp, tolerance_hours=1
            )
            
            if cached_features and len(cached_features) == len(features):
                business_metrics.counter("feature_store.db_cache_hit", 1)
                # Convert to DataFrame
                result_df = pd.DataFrame([cached_features], index=[timestamp])
                return result_df
            
            # Check in-memory cache
            cached_features = await self.cache_manager.get(
                "features", symbol, features=sorted(features)
            )
            if cached_features is not None:
                business_metrics.counter("feature_store.cache_hit", 1)
                return cached_features
                
            # Compute features with parallel processing if available
            if self.executor and self.config['parallel_processing'] and len(features) > 3:
                feature_data = await self._compute_features_parallel(data, features)
            else:
                feature_data = await self._compute_features_sequential(data, features)
            
            # Create DataFrame
            result_df = pd.DataFrame(feature_data, index=data.index)
            
            # Store in persistent feature cache
            if not result_df.empty:
                latest_features = result_df.iloc[-1].to_dict()
                await self.feature_cache.store_features(
                    symbol, latest_features, timestamp
                )
            
            # Cache results in memory
            await self.cache_manager.set(
                "features", symbol, result_df, 
                ttl_seconds=self.cache_ttl, features=sorted(features)
            )
            
            # Track metrics
            business_metrics.counter(
                "feature_store.features_computed",
                len(features),
                {"symbol": symbol}
            )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            business_metrics.counter("feature_store.compute_error", 1)
            return pd.DataFrame()
    
    async def get_feature_vector(self, symbol: str, timestamp: datetime = None) -> Dict[str, float]:
        """Get feature vector for a specific point in time"""
        # Compute all features
        features_df = await self.compute_features(symbol)
        
        if features_df.empty:
            return {}
        
        # Get specific timestamp or latest
        if timestamp:
            # Find closest timestamp
            idx = features_df.index.get_loc(timestamp, method='nearest')
            feature_vector = features_df.iloc[idx].to_dict()
        else:
            # Get latest
            feature_vector = features_df.iloc[-1].to_dict()
        
        # Clean NaN values
        feature_vector = {k: v for k, v in feature_vector.items() if not pd.isna(v)}
        
        return feature_vector
    
    async def _compute_features_sequential(self, data: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Compute features sequentially"""
        feature_data = {}
        
        for feature_name in features:
            if feature_name not in self.features:
                logger.warning(f"Unknown feature: {feature_name}")
                continue
            
            feature_def = self.features[feature_name]
            
            try:
                # Compute feature
                feature_values = feature_def.compute_func(data)
                feature_data[feature_name] = feature_values
                
            except Exception as e:
                logger.error(f"Error computing feature {feature_name}: {e}")
                feature_data[feature_name] = np.nan
        
        return feature_data
    
    async def _compute_features_parallel(self, data: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Compute features in parallel using the parallel executor"""
        from ..processing.parallel_executor import OptimizedFactorComputer
        
        # Prepare factor functions
        factor_functions = {}
        for feature_name in features:
            if feature_name in self.features:
                factor_functions[feature_name] = self.features[feature_name].compute_func
        
        # Use optimized factor computer
        factor_computer = OptimizedFactorComputer()
        
        # Compute factors in parallel
        results = await factor_computer.compute_factors_parallel(
            ["data"],  # Single data key
            factor_functions,
            {"data": data}
        )
        
        # Extract results
        feature_data = results.get("data", {})
        
        # Handle missing features
        for feature_name in features:
            if feature_name not in feature_data:
                logger.warning(f"Feature {feature_name} not computed")
                feature_data[feature_name] = np.nan
        
        return feature_data
    
    async def get_training_features(self, symbols: List[str], 
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """Get features for multiple symbols for training"""
        all_features = []
        
        for symbol in symbols:
            features_df = await self.compute_features(symbol)
            
            if not features_df.empty:
                # Filter by date range
                mask = (features_df.index >= start_date) & (features_df.index <= end_date)
                symbol_features = features_df.loc[mask].copy()
                symbol_features['symbol'] = symbol
                all_features.append(symbol_features)
        
        if all_features:
            return pd.concat(all_features, axis=0)
        
        return pd.DataFrame()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from latest models"""
        # This would integrate with the model registry
        # For now, return static importance
        return {
            "rsi_14": 0.15,
            "macd_signal": 0.12,
            "volatility_20": 0.10,
            "price_change_24h": 0.10,
            "volume_ratio_20": 0.08,
            "bb_position": 0.08,
            "trend_strength": 0.07,
            "momentum_10": 0.06,
            "price_volume_correlation": 0.05,
            "volatility_ratio": 0.05,
            "support_level": 0.04,
            "resistance_level": 0.04,
            "price_change_1h": 0.03,
            "price_change_7d": 0.03
        }
    
    # Feature computation functions
    def _compute_price_change_1h(self, data: pd.DataFrame) -> pd.Series:
        """Compute 1-hour price change percentage"""
        return data['close'].pct_change(1) * 100
    
    def _compute_price_change_24h(self, data: pd.DataFrame) -> pd.Series:
        """Compute 24-hour price change percentage"""
        return data['close'].pct_change(24) * 100
    
    def _compute_price_change_7d(self, data: pd.DataFrame) -> pd.Series:
        """Compute 7-day price change percentage"""
        return data['close'].pct_change(168) * 100  # 7 * 24 hours
    
    def _compute_rsi_14(self, data: pd.DataFrame) -> pd.Series:
        """Compute 14-period RSI with Wilder's smoothing"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / 14
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_macd_signal(self, data: pd.DataFrame) -> pd.Series:
        """Compute MACD signal line with proper EMA initialization"""
        # Proper EMA calculation with correct alpha values
        alpha_12 = 2.0 / (12 + 1)
        alpha_26 = 2.0 / (26 + 1)
        alpha_9 = 2.0 / (9 + 1)
        
        ema_12 = data['close'].ewm(alpha=alpha_12, adjust=False).mean()
        ema_26 = data['close'].ewm(alpha=alpha_26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(alpha=alpha_9, adjust=False).mean()
        return signal
    
    def _compute_bb_position(self, data: pd.DataFrame) -> pd.Series:
        """Compute position within Bollinger Bands (0-1)"""
        bb_middle = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_width = bb_upper - bb_lower
        bb_position = (data['close'] - bb_lower) / bb_width
        return bb_position.clip(0, 1)
    
    def _compute_volume_ratio(self, data: pd.DataFrame) -> pd.Series:
        """Compute volume ratio to 20-period average"""
        volume_ma = data['volume'].rolling(window=20).mean()
        return data['volume'] / volume_ma
    
    def _compute_price_volume_corr(self, data: pd.DataFrame) -> pd.Series:
        """Compute 30-day rolling correlation between price and volume"""
        price_returns = data['close'].pct_change()
        volume_changes = data['volume'].pct_change()
        correlation = price_returns.rolling(window=30).corr(volume_changes)
        return correlation
    
    def _compute_volatility_20(self, data: pd.DataFrame) -> pd.Series:
        """Compute 20-day volatility"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(24 * 365)  # Annualized
        return volatility
    
    def _compute_volatility_ratio(self, data: pd.DataFrame) -> pd.Series:
        """Compute ratio of current to historical volatility"""
        returns = data['close'].pct_change()
        current_vol = returns.rolling(window=20).std()
        historical_vol = returns.rolling(window=100).std()
        return current_vol / historical_vol
    
    def _compute_support_level(self, data: pd.DataFrame) -> pd.Series:
        """Compute rolling support level"""
        return data['low'].rolling(window=20).min()
    
    def _compute_resistance_level(self, data: pd.DataFrame) -> pd.Series:
        """Compute rolling resistance level"""
        return data['high'].rolling(window=20).max()
    
    def _compute_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Compute trend strength using ADX-like calculation"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Simplified trend strength
        price_change = data['close'] - data['close'].shift(14)
        trend_strength = (price_change.abs() / atr).rolling(window=14).mean()
        
        return trend_strength
    
    def _compute_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Compute 10-period momentum"""
        return data['close'] - data['close'].shift(10)
    
    # Helper methods
    async def _get_base_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get base OHLCV data for feature computation"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months for feature computation
            
            data = await self.database.get_historical_data(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1h'
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching base data: {e}")
            return None
    
    # Cache methods now handled by self.cache_client


# Global feature store instance
feature_store = FeatureStore()