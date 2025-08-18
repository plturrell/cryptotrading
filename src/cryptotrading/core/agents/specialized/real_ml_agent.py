"""
Real ML Agent with Actual Model Training and Evaluation
NO RANDOM METRICS - Real machine learning only
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class RealMLConfig:
    """Configuration for real ML agent"""
    # Model parameters
    model_types: List[str] = field(default_factory=lambda: ['random_forest', 'gradient_boosting', 'ensemble'])
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    
    # Training parameters
    test_size: float = 0.2
    validation_splits: int = 5  # For time series cross-validation
    
    # Feature engineering
    feature_window: int = 20  # Days of historical data for features
    target_horizon: int = 1   # Days ahead to predict
    
    # Model persistence
    save_models: bool = True
    model_path: str = "models/"


@dataclass
class RealModelMetrics:
    """Real model evaluation metrics"""
    # Regression metrics
    mse: float
    rmse: float
    mae: float
    r2: float
    
    # Trading-specific metrics
    directional_accuracy: float  # % of correct up/down predictions
    profit_factor: float         # Gross profit / Gross loss
    sharpe_ratio: float         # Risk-adjusted returns
    max_drawdown: float         # Maximum peak-to-trough decline
    
    # Cross-validation scores
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Feature importance
    feature_importance: Dict[str, float]
    
    # Prediction analysis
    prediction_distribution: Dict[str, Any]
    residual_analysis: Dict[str, Any]


class RealFeatureEngineer:
    """Create real features from market data"""
    
    def __init__(self, config: RealMLConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create real technical features from price data"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            features[f'sma_{period}_ratio'] = data['close'] / features[f'sma_{period}']
        
        # Volatility features
        features['volatility_20'] = features['returns'].rolling(window=20).std()
        features['volatility_ratio'] = features['volatility_20'] / features['volatility_20'].rolling(window=50).mean()
        
        # Volume features
        features['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma_20']
        features['volume_price'] = data['volume'] * data['close']
        
        # Technical indicators
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_sma = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        features['bb_upper'] = bb_sma + (bb_std * 2)
        features['bb_lower'] = bb_sma - (bb_std * 2)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (data['close'] - features['bb_lower']) / features['bb_width']
        
        # Price patterns
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'returns_mean_{window}'] = features['returns'].rolling(window=window).mean()
            features[f'returns_std_{window}'] = features['returns'].rolling(window=window).std()
            features[f'returns_skew_{window}'] = features['returns'].rolling(window=window).skew()
            features[f'returns_kurt_{window}'] = features['returns'].rolling(window=window).kurt()
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features
    
    def create_target(self, data: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Create target variable (future returns)"""
        # Predict future returns
        target = data['close'].shift(-horizon) / data['close'] - 1
        return target
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Remove NaN values
        valid_idx = features.notna().all(axis=1) & target.notna()
        X = features[valid_idx].values
        y = target[valid_idx].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y


class RealMLAgent:
    """Real ML agent with actual model training and evaluation"""
    
    def __init__(self, config: RealMLConfig):
        self.config = config
        self.feature_engineer = RealFeatureEngineer(config)
        self.models = {}
        self.metrics = {}
        self.is_trained = False
        
    async def train_models(self, historical_data: pd.DataFrame) -> Dict[str, RealModelMetrics]:
        """Train real ML models on historical data"""
        logger.info("Starting real ML model training")
        
        # Create features
        features = self.feature_engineer.create_features(historical_data)
        target = self.feature_engineer.create_target(historical_data, self.config.target_horizon)
        
        # Prepare data
        X, y = self.feature_engineer.prepare_data(features, target)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data for training: {len(X)} samples")
        
        # Train different models
        results = {}
        
        if 'random_forest' in self.config.model_types:
            logger.info("Training Random Forest model")
            rf_model, rf_metrics = await self._train_random_forest(X, y)
            self.models['random_forest'] = rf_model
            results['random_forest'] = rf_metrics
        
        if 'gradient_boosting' in self.config.model_types:
            logger.info("Training Gradient Boosting model")
            gb_model, gb_metrics = await self._train_gradient_boosting(X, y)
            self.models['gradient_boosting'] = gb_model
            results['gradient_boosting'] = gb_metrics
        
        if 'ensemble' in self.config.model_types and len(self.models) > 1:
            logger.info("Creating ensemble model")
            ensemble_metrics = await self._create_ensemble(X, y)
            results['ensemble'] = ensemble_metrics
        
        self.metrics = results
        self.is_trained = True
        
        # Save models if configured
        if self.config.save_models:
            await self._save_models()
        
        return results
    
    async def _train_random_forest(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestRegressor, RealModelMetrics]:
        """Train Random Forest with real evaluation"""
        model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.validation_splits)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        cv_scores = -cv_scores  # Convert to positive MSE
        
        # Train on full data
        model.fit(X, y)
        
        # Get predictions for evaluation
        predictions = model.predict(X)
        
        # Calculate real metrics
        metrics = self._calculate_real_metrics(y, predictions, cv_scores)
        
        # Add feature importance
        feature_importance = dict(zip(
            self.feature_engineer.feature_names,
            model.feature_importances_
        ))
        metrics.feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])  # Top 20 features
        
        return model, metrics
    
    async def _train_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> Tuple[GradientBoostingRegressor, RealModelMetrics]:
        """Train Gradient Boosting with real evaluation"""
        model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            learning_rate=0.1,
            random_state=42
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.validation_splits)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        cv_scores = -cv_scores
        
        # Train on full data
        model.fit(X, y)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Calculate metrics
        metrics = self._calculate_real_metrics(y, predictions, cv_scores)
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_engineer.feature_names,
            model.feature_importances_
        ))
        metrics.feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])
        
        return model, metrics
    
    async def _create_ensemble(self, X: np.ndarray, y: np.ndarray) -> RealModelMetrics:
        """Create ensemble predictions and evaluate"""
        # Get predictions from all models
        all_predictions = []
        for name, model in self.models.items():
            if name != 'ensemble':
                all_predictions.append(model.predict(X))
        
        # Simple average ensemble
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        # Evaluate ensemble
        # For CV scores, average the CV scores of individual models
        cv_scores = np.mean([
            self.metrics[name].cv_scores 
            for name in self.models.keys() 
            if name != 'ensemble'
        ], axis=0)
        
        metrics = self._calculate_real_metrics(y, ensemble_predictions, cv_scores)
        
        # Average feature importance from all models
        all_importances = {}
        for name, model_metrics in self.metrics.items():
            if name != 'ensemble':
                for feature, importance in model_metrics.feature_importance.items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
        
        avg_importance = {
            feature: np.mean(scores) 
            for feature, scores in all_importances.items()
        }
        metrics.feature_importance = dict(sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])
        
        return metrics
    
    def _calculate_real_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, cv_scores: np.ndarray) -> RealModelMetrics:
        """Calculate real model evaluation metrics"""
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        actual_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        # Trading simulation for profit factor and Sharpe
        # Simulate trading based on predictions
        returns = []
        for i in range(len(y_true)):
            if y_pred[i] > 0:  # Predicted positive return
                returns.append(y_true[i])  # Take the actual return
            else:
                returns.append(0)  # Stay out of market
        
        returns = np.array(returns)
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = -np.sum(returns[returns < 0])
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Sharpe ratio (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Prediction distribution analysis
        prediction_distribution = {
            'mean': float(np.mean(y_pred)),
            'std': float(np.std(y_pred)),
            'min': float(np.min(y_pred)),
            'max': float(np.max(y_pred)),
            'percentiles': {
                '25': float(np.percentile(y_pred, 25)),
                '50': float(np.percentile(y_pred, 50)),
                '75': float(np.percentile(y_pred, 75))
            }
        }
        
        # Residual analysis
        residuals = y_true - y_pred
        residual_analysis = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skew': float(self._calculate_skew(residuals)),
            'kurtosis': float(self._calculate_kurtosis(residuals)),
            'autocorrelation': float(self._calculate_autocorrelation(residuals))
        }
        
        return RealModelMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            cv_scores=cv_scores.tolist(),
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            feature_importance={},  # Set later by specific model
            prediction_distribution=prediction_distribution,
            residual_analysis=residual_analysis
        )
    
    def _calculate_skew(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        n = len(data)
        if n < 3:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        
        return n / ((n-1) * (n-2)) * np.sum(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        n = len(data)
        if n < 4:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        
        return n * (n+1) / ((n-1) * (n-2) * (n-3)) * np.sum(((data - mean) / std) ** 4) - 3 * (n-1)**2 / ((n-2) * (n-3))
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag"""
        if len(data) <= lag:
            return 0
        
        return np.corrcoef(data[:-lag], data[lag:])[0, 1]
    
    async def predict(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with trained models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        X = self.feature_engineer.scaler.transform(features[self.feature_engineer.feature_names].values)
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name != 'ensemble':
                predictions[name] = model.predict(X)
        
        # Ensemble prediction
        if 'ensemble' in self.config.model_types and len(predictions) > 1:
            predictions['ensemble'] = np.mean(list(predictions.values()), axis=0)
        
        return predictions
    
    async def evaluate_real_time_performance(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on new real data"""
        # Create features
        features = self.feature_engineer.create_features(new_data)
        target = self.feature_engineer.create_target(new_data, self.config.target_horizon)
        
        # Prepare data
        X, y = self.feature_engineer.prepare_data(features, target)
        
        if len(X) == 0:
            return {"error": "No valid data for evaluation"}
        
        # Evaluate each model
        results = {}
        for name, model in self.models.items():
            if name != 'ensemble':
                predictions = model.predict(X)
                
                results[name] = {
                    'rmse': np.sqrt(mean_squared_error(y, predictions)),
                    'mae': mean_absolute_error(y, predictions),
                    'directional_accuracy': np.mean(np.sign(y) == np.sign(predictions)),
                    'correlation': np.corrcoef(y, predictions)[0, 1]
                }
        
        # Ensemble evaluation
        if 'ensemble' in self.config.model_types:
            ensemble_pred = np.mean([
                self.models[name].predict(X) 
                for name in self.models if name != 'ensemble'
            ], axis=0)
            
            results['ensemble'] = {
                'rmse': np.sqrt(mean_squared_error(y, ensemble_pred)),
                'mae': mean_absolute_error(y, ensemble_pred),
                'directional_accuracy': np.mean(np.sign(y) == np.sign(ensemble_pred)),
                'correlation': np.corrcoef(y, ensemble_pred)[0, 1]
            }
        
        return results
    
    async def _save_models(self):
        """Save trained models to disk"""
        import os
        os.makedirs(self.config.model_path, exist_ok=True)
        
        for name, model in self.models.items():
            if name != 'ensemble':  # Ensemble is not a separate model
                model_file = os.path.join(self.config.model_path, f"{name}_model.pkl")
                joblib.dump(model, model_file)
                logger.info(f"Saved {name} model to {model_file}")
        
        # Save feature scaler
        scaler_file = os.path.join(self.config.model_path, "feature_scaler.pkl")
        joblib.dump(self.feature_engineer.scaler, scaler_file)
        
        # Save feature names
        features_file = os.path.join(self.config.model_path, "feature_names.pkl")
        joblib.dump(self.feature_engineer.feature_names, features_file)
        
        # Save metrics
        metrics_file = os.path.join(self.config.model_path, "model_metrics.pkl")
        joblib.dump(self.metrics, metrics_file)
    
    async def load_models(self):
        """Load pre-trained models from disk"""
        import os
        
        if not os.path.exists(self.config.model_path):
            raise ValueError(f"Model path does not exist: {self.config.model_path}")
        
        # Load models
        for model_type in self.config.model_types:
            if model_type != 'ensemble':
                model_file = os.path.join(self.config.model_path, f"{model_type}_model.pkl")
                if os.path.exists(model_file):
                    self.models[model_type] = joblib.load(model_file)
                    logger.info(f"Loaded {model_type} model from {model_file}")
        
        # Load feature scaler
        scaler_file = os.path.join(self.config.model_path, "feature_scaler.pkl")
        if os.path.exists(scaler_file):
            self.feature_engineer.scaler = joblib.load(scaler_file)
        
        # Load feature names
        features_file = os.path.join(self.config.model_path, "feature_names.pkl")
        if os.path.exists(features_file):
            self.feature_engineer.feature_names = joblib.load(features_file)
        
        # Load metrics
        metrics_file = os.path.join(self.config.model_path, "model_metrics.pkl")
        if os.path.exists(metrics_file):
            self.metrics = joblib.load(metrics_file)
        
        self.is_trained = len(self.models) > 0


# Export the real implementation
__all__ = ['RealMLAgent', 'RealMLConfig', 'RealModelMetrics', 'RealFeatureEngineer']