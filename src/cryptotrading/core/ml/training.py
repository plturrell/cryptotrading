"""
ML model training pipeline for cryptocurrency predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor

# Optional imports for local development
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("Schedule not available - automatic retraining disabled")

from .models import CryptoPricePredictor, model_registry
# Training should use database data, not direct market data clients

# Use unified monitoring
from ..monitoring import get_monitor
from ..storage import get_sync_storage

monitor = get_monitor("ml-training")
storage = get_sync_storage()
logger = logging.getLogger("ml.training")


class ModelTrainingPipeline:
    """Automated model training pipeline"""
    
    def __init__(self):
        # Use database for training data instead of direct market client
        from ...data.database.unified_database import UnifiedDatabase
        self.database = UnifiedDatabase()
        self.training_config = {
            'symbols': ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LINK', 'UNI'],
            'lookback_days': 365,
            'model_types': ['ensemble', 'neural_network', 'random_forest'],
            'prediction_horizons': [1, 24, 168],  # 1h, 24h, 7d
            'retrain_interval_hours': 24,
            'min_data_points': 1000
        }
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.performance_history = []
        
    async def fetch_training_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for training"""
        with monitor.span(f"fetch_training_data_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                
                # Get historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.training_config['lookback_days'])
                
                data = await self.database.get_historical_data(
                    symbol=symbol, 
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    interval='1h'
                )
                
                if data is None or len(data) < self.training_config['min_data_points']:
                    logger.warning(f"Insufficient data for {symbol}: {len(data) if data else 0} points")
                    return None
                
                monitor.increment_counter(
                    "ml.training.data_fetched",
                    1,
                    {"symbol": symbol, "points": str(len(data))}
                )
                
                return data
                
            except Exception as e:
                logger.error(f"Error fetching training data for {symbol}: {e}")
                monitor.increment_counter(
                    "ml.training.fetch_error",
                    1,
                    {"symbol": symbol, "error": str(e)}
                )
                return None
    
    async def train_model(self, symbol: str, model_type: str, horizon_hours: int) -> Optional[Dict[str, Any]]:
        """Train a single model"""
        with monitor.span(f"train_model_{symbol}_{model_type}_{horizon_hours}h") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("model_type", model_type)
                span.set_attribute("horizon_hours", horizon_hours)
                
                # Fetch data
                data = await self.fetch_training_data(symbol)
                if data is None:
                    return None
                
                # Create model
                version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model = CryptoPricePredictor(model_type=model_type, version=version)
                
                # Train model
                logger.info(f"Training {model_type} model for {symbol} with {horizon_hours}h horizon")
                start_time = time.time()
                
                metrics = model.train(data, target_hours=horizon_hours)
                
                training_time = time.time() - start_time
                
                # Register model
                model_name = f"{symbol}_{model_type}_{horizon_hours}h"
                model_registry.register_model(model_name, model)
                
                # Record performance
                performance = {
                    'symbol': symbol,
                    'model_type': model_type,
                    'horizon_hours': horizon_hours,
                    'version': version,
                    'metrics': metrics,
                    'training_time': training_time,
                    'trained_at': datetime.now().isoformat()
                }
                
                self.performance_history.append(performance)
                
                # Track metrics
                monitor.increment_counter(
                    "ml.training.model_trained",
                    1,
                    {"symbol": symbol, "model_type": model_type, "horizon": f"{horizon_hours}h"}
                )
                
                monitor.set_gauge(
                    "ml.training.model_accuracy",
                    metrics['r2'],
                    {"symbol": symbol, "model_type": model_type}
                )
                
                monitor.record_histogram(
                    "ml.training.time_seconds",
                    training_time,
                    {"model_type": model_type}
                )
                
                logger.info(f"Model trained successfully: {model_name} - R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")
                
                return performance
                
            except Exception as e:
                logger.error(f"Error training model: {e}")
                monitor.increment_counter(
                    "ml.training.error",
                    1,
                    {"symbol": symbol, "model_type": model_type, "error": str(e)}
                )
                return None
    
    async def train_all_models(self):
        """Train all configured models"""
        logger.info("Starting comprehensive model training")
        
        tasks = []
        
        # Create training tasks
        for symbol in self.training_config['symbols']:
            for model_type in self.training_config['model_types']:
                for horizon in self.training_config['prediction_horizons']:
                    task = self.train_model(symbol, model_type, horizon)
                    tasks.append(task)
        
        # Run training in parallel
        results = await asyncio.gather(*tasks)
        
        # Filter successful results
        successful = [r for r in results if r is not None]
        
        # Save performance history
        self._save_performance_history()
        
        logger.info(f"Training completed: {len(successful)}/{len(tasks)} models trained successfully")
        
        return successful
    
    def _save_performance_history(self):
        """Save performance history to disk"""
        history_file = self.models_dir / "performance_history.json"
        
        # Keep only last 100 entries per model
        unique_models = {}
        for perf in self.performance_history:
            key = f"{perf['symbol']}_{perf['model_type']}_{perf['horizon_hours']}h"
            if key not in unique_models:
                unique_models[key] = []
            unique_models[key].append(perf)
        
        # Keep only recent entries
        filtered_history = []
        for model_perfs in unique_models.values():
            sorted_perfs = sorted(model_perfs, key=lambda x: x['trained_at'], reverse=True)
            filtered_history.extend(sorted_perfs[:10])
        
        self.performance_history = filtered_history
        
        # Save to file
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def get_best_model(self, symbol: str, horizon_hours: int) -> Optional[str]:
        """Get the best performing model for a symbol and horizon"""
        best_score = -float('inf')
        best_model = None
        
        for perf in self.performance_history:
            if perf['symbol'] == symbol and perf['horizon_hours'] == horizon_hours:
                if perf['metrics']['r2'] > best_score:
                    best_score = perf['metrics']['r2']
                    best_model = f"{symbol}_{perf['model_type']}_{horizon_hours}h"
        
        return best_model
    
    def schedule_retraining(self):
        """Schedule periodic model retraining"""
        interval_hours = self.training_config['retrain_interval_hours']
        
        # Schedule training
        schedule.every(interval_hours).hours.do(lambda: asyncio.run(self.train_all_models()))
        
        logger.info(f"Model retraining scheduled every {interval_hours} hours")
        
        # Use async scheduler instead of blocking thread
        self._scheduler_task = None
        self._running = True
        
    async def start_scheduler(self):
        """Start the async training scheduler"""
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("Training scheduler started")
        
    async def stop_scheduler(self):
        """Stop the training scheduler gracefully"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Training scheduler stopped")
        
    async def _run_scheduler(self):
        """Run scheduled training tasks asynchronously"""
        while self._running:
            try:
                # Check and run pending scheduled tasks
                schedule.run_pending()
                # Use async sleep instead of blocking sleep
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)  # Continue after error


class ModelEvaluator:
    """Evaluate and compare model performance"""
    
    def __init__(self):
        # Use database for evaluation data instead of direct market client
        from ...data.database.unified_database import UnifiedDatabase
        self.database = UnifiedDatabase()
        
    async def backtest_model(self, model_name: str, test_days: int = 30) -> Dict[str, Any]:
        """Backtest a model on recent data"""
        try:
            # Get model
            model = model_registry.get_model(model_name)
            
            # Parse model name
            parts = model_name.split('_')
            symbol = parts[0]
            horizon_hours = int(parts[-1].replace('h', ''))
            
            # Get test data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=test_days + 30)  # Extra for feature engineering
            
            data = await self.database.get_historical_data(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1h'
            )
            
            if data is None or len(data) < 100:
                return {'error': 'Insufficient data for backtesting'}
            
            # Make predictions
            predictions = []
            actuals = []
            
            # Use last 30 days for testing
            test_start_idx = len(data) - (test_days * 24)
            
            for i in range(test_start_idx, len(data) - horizon_hours):
                # Get data up to this point
                historical = data.iloc[:i]
                
                # Make prediction
                pred = model.predict(historical)
                predictions.append(pred['predicted_price'])
                
                # Get actual price
                actual = data.iloc[i + horizon_hours]['close']
                actuals.append(actual)
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            errors = actuals - predictions
            pct_errors = errors / actuals * 100
            
            metrics = {
                'model_name': model_name,
                'test_days': test_days,
                'predictions_made': len(predictions),
                'mae': float(np.mean(np.abs(errors))),
                'rmse': float(np.sqrt(np.mean(errors ** 2))),
                'mape': float(np.mean(np.abs(pct_errors))),
                'direction_accuracy': float(np.mean((predictions[1:] > predictions[:-1]) == (actuals[1:] > actuals[:-1]))),
                'profit_factor': self._calculate_profit_factor(predictions, actuals),
                'sharpe_ratio': self._calculate_sharpe_ratio(predictions, actuals)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error backtesting model {model_name}: {e}")
            return {'error': str(e)}
    
    def _calculate_profit_factor(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate profit factor from predictions"""
        # Simple trading strategy: buy when predicted > current
        returns = []
        
        for i in range(1, len(predictions)):
            if predictions[i] > actuals[i-1]:  # Predicted to go up
                returns.append((actuals[i] - actuals[i-1]) / actuals[i-1])
            else:  # Predicted to go down
                returns.append(0)  # Stay out of market
        
        returns = np.array(returns)
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = -np.sum(returns[returns < 0])
        
        if gross_loss == 0:
            return float(gross_profit) if gross_profit > 0 else 0.0
        
        return float(gross_profit / gross_loss)
    
    def _calculate_sharpe_ratio(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Sharpe ratio from predictions"""
        # Calculate returns from trading on predictions
        returns = []
        
        for i in range(1, len(predictions)):
            if predictions[i] > actuals[i-1]:  # Predicted to go up
                returns.append((actuals[i] - actuals[i-1]) / actuals[i-1])
            else:
                returns.append(0)
        
        returns = np.array(returns)
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming hourly data)
        return float(np.mean(returns) / np.std(returns) * np.sqrt(24 * 365))


# Global instances
training_pipeline = ModelTrainingPipeline()
model_evaluator = ModelEvaluator()