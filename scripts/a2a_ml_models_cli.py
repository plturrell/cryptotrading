#!/usr/bin/env python3
"""
A2A ML Models CLI - Advanced ML model management and deployment
Real implementation with model training, serving, and ensemble capabilities
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import click

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set environment variables for development
os.environ["ENVIRONMENT"] = "development"
os.environ["SKIP_DB_INIT"] = "true"

try:
    from src.cryptotrading.core.ml.ensemble import EnsembleManager
    from src.cryptotrading.core.ml.inference import PredictionRequest, PredictionResponse
    from src.cryptotrading.core.ml.models import CryptoPricePredictor, model_registry
    from src.cryptotrading.core.ml.training import ModelTrainingPipeline
    from src.cryptotrading.infrastructure.mcp.ml_models_mcp_tools import MLModelsMCPTools

    REAL_IMPLEMENTATION = True
except ImportError as e:
    print(f"⚠️ Using fallback implementation: {e}")
    REAL_IMPLEMENTATION = False


class MLModelsAgent:
    """ML Models Agent with advanced model management capabilities"""

    def __init__(self):
        self.agent_id = "ml_models_agent"
        self.capabilities = [
            "train_model",
            "predict_prices",
            "evaluate_model",
            "optimize_hyperparameters",
            "ensemble_predict",
            "feature_importance",
        ]

        if REAL_IMPLEMENTATION:
            self.mcp_tools = MLModelsMCPTools()
            self.predictor = CryptoPricePredictor()
            self.training_pipeline = ModelTrainingPipeline()
            self.ensemble_manager = EnsembleManager()

        # Mock model registry
        self.models = {
            "xgboost_btc_v1": {"type": "xgboost", "accuracy": 0.87, "status": "trained"},
            "lstm_eth_v2": {"type": "lstm", "accuracy": 0.83, "status": "trained"},
            "rf_ensemble_v1": {"type": "random_forest", "accuracy": 0.85, "status": "trained"},
            "transformer_multi": {"type": "transformer", "accuracy": 0.89, "status": "training"},
        }

    async def train_model(
        self,
        model_type: str,
        symbol: str,
        epochs: int = 100,
        features: List[str] = None,
        hyperparams: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Train ML model on symbol data"""
        if not REAL_IMPLEMENTATION:
            return self._mock_train_model(model_type, symbol, epochs, features)

        try:
            training_config = {
                "model_type": model_type,
                "symbol": symbol,
                "epochs": epochs,
                "features": features or ["price", "volume", "rsi", "macd"],
                "hyperparameters": hyperparams or {},
            }

            result = await self.training_pipeline.train_model(training_config)

            return {
                "success": True,
                "model_id": result.get("model_id"),
                "model_type": model_type,
                "symbol": symbol,
                "training_accuracy": result.get("training_accuracy"),
                "validation_accuracy": result.get("validation_accuracy"),
                "test_accuracy": result.get("test_accuracy"),
                "epochs_completed": result.get("epochs_completed"),
                "training_time": result.get("training_time"),
                "model_path": result.get("model_path"),
                "features_used": result.get("features_used"),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Model training failed: {str(e)}"}

    def _mock_train_model(
        self, model_type: str, symbol: str, epochs: int, features: List[str]
    ) -> Dict[str, Any]:
        """Mock model training"""
        import random

        model_id = f"{model_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "success": True,
            "model_id": model_id,
            "model_type": model_type,
            "symbol": symbol,
            "training_accuracy": round(random.uniform(0.80, 0.95), 3),
            "validation_accuracy": round(random.uniform(0.75, 0.90), 3),
            "test_accuracy": round(random.uniform(0.70, 0.88), 3),
            "epochs_completed": epochs,
            "training_time": f"{random.randint(5, 30)}m {random.randint(10, 59)}s",
            "model_path": f"models/{model_id}.pkl",
            "features_used": features or ["price", "volume", "rsi", "macd", "bb_upper", "bb_lower"],
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def predict_prices(
        self, model_id: str, symbol: str, horizon: int = 5, confidence_interval: bool = True
    ) -> Dict[str, Any]:
        """Generate price predictions using trained model"""
        if not REAL_IMPLEMENTATION:
            return self._mock_predict_prices(model_id, symbol, horizon, confidence_interval)

        try:
            prediction_request = PredictionRequest(
                model_id=model_id,
                symbol=symbol,
                horizon=horizon,
                include_confidence=confidence_interval,
            )

            result = await self.predictor.predict(prediction_request)

            return {
                "success": True,
                "model_id": model_id,
                "symbol": symbol,
                "predictions": result.predictions,
                "confidence_intervals": result.confidence_intervals
                if confidence_interval
                else None,
                "model_accuracy": result.model_accuracy,
                "prediction_confidence": result.overall_confidence,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Price prediction failed: {str(e)}"}

    def _mock_predict_prices(
        self, model_id: str, symbol: str, horizon: int, confidence_interval: bool
    ) -> Dict[str, Any]:
        """Mock price predictions"""
        import random

        base_price = 50000.0 if symbol.startswith("BTC") else 3000.0
        predictions = []
        confidence_intervals = [] if confidence_interval else None

        for i in range(horizon):
            # Random walk with slight upward bias
            change = random.uniform(-0.03, 0.05)
            predicted_price = base_price * (1 + change * (i + 1))

            predictions.append(
                {
                    "step": i + 1,
                    "predicted_price": round(predicted_price, 2),
                    "confidence": round(random.uniform(0.65, 0.90), 3),
                }
            )

            if confidence_interval:
                confidence_intervals.append(
                    {
                        "step": i + 1,
                        "lower_bound": round(predicted_price * 0.95, 2),
                        "upper_bound": round(predicted_price * 1.05, 2),
                        "confidence_level": 0.95,
                    }
                )

        return {
            "success": True,
            "model_id": model_id,
            "symbol": symbol,
            "predictions": predictions,
            "confidence_intervals": confidence_intervals,
            "model_accuracy": round(random.uniform(0.75, 0.92), 3),
            "prediction_confidence": round(random.uniform(0.70, 0.85), 3),
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def evaluate_model(
        self, model_id: str, test_data_path: str = None, metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        if not REAL_IMPLEMENTATION:
            return self._mock_evaluate_model(
                model_id, metrics or ["accuracy", "precision", "recall", "f1"]
            )

        try:
            evaluation_config = {
                "model_id": model_id,
                "test_data_path": test_data_path,
                "metrics": metrics or ["mse", "mae", "r2", "directional_accuracy"],
            }

            result = await self.training_pipeline.evaluate_model(evaluation_config)

            return {
                "success": True,
                "model_id": model_id,
                "evaluation_metrics": result.get("metrics", {}),
                "performance_summary": result.get("summary", {}),
                "test_predictions": result.get("predictions", [])[:10],  # First 10
                "evaluation_date": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Model evaluation failed: {str(e)}"}

    def _mock_evaluate_model(self, model_id: str, metrics: List[str]) -> Dict[str, Any]:
        """Mock model evaluation"""
        import random

        metric_results = {}
        for metric in metrics:
            if metric == "accuracy":
                metric_results[metric] = round(random.uniform(0.75, 0.92), 3)
            elif metric == "precision":
                metric_results[metric] = round(random.uniform(0.70, 0.88), 3)
            elif metric == "recall":
                metric_results[metric] = round(random.uniform(0.72, 0.90), 3)
            elif metric == "f1":
                metric_results[metric] = round(random.uniform(0.71, 0.89), 3)
            elif metric == "mse":
                metric_results[metric] = round(random.uniform(100, 1000), 2)
            elif metric == "mae":
                metric_results[metric] = round(random.uniform(50, 500), 2)
            elif metric == "r2":
                metric_results[metric] = round(random.uniform(0.65, 0.85), 3)
            elif metric == "directional_accuracy":
                metric_results[metric] = round(random.uniform(0.68, 0.82), 3)

        return {
            "success": True,
            "model_id": model_id,
            "evaluation_metrics": metric_results,
            "performance_summary": {
                "overall_score": round(random.uniform(0.75, 0.90), 3),
                "recommendation": "Model performs well on test data",
                "strengths": ["Good directional accuracy", "Stable predictions"],
                "weaknesses": ["Slightly high volatility in predictions"],
            },
            "test_predictions": [
                {"actual": 51234.56, "predicted": 51456.78, "error": 222.22},
                {"actual": 52123.45, "predicted": 51987.65, "error": -135.8},
            ],
            "mock": True,
            "evaluation_date": datetime.now().isoformat(),
        }

    async def optimize_hyperparameters(
        self,
        model_type: str,
        symbol: str,
        search_space: Dict[str, Any] = None,
        max_trials: int = 50,
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters"""
        if not REAL_IMPLEMENTATION:
            return self._mock_optimize_hyperparameters(model_type, symbol, max_trials)

        try:
            optimization_config = {
                "model_type": model_type,
                "symbol": symbol,
                "search_space": search_space or self._get_default_search_space(model_type),
                "max_trials": max_trials,
                "objective": "val_accuracy",
            }

            result = await self.training_pipeline.optimize_hyperparameters(optimization_config)

            return {
                "success": True,
                "model_type": model_type,
                "symbol": symbol,
                "best_score": result.get("best_score"),
                "best_parameters": result.get("best_params"),
                "trials_completed": result.get("trials_completed"),
                "optimization_time": result.get("optimization_time"),
                "trial_history": result.get("trial_history", [])[-10:],  # Last 10 trials
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Hyperparameter optimization failed: {str(e)}"}

    def _mock_optimize_hyperparameters(
        self, model_type: str, symbol: str, max_trials: int
    ) -> Dict[str, Any]:
        """Mock hyperparameter optimization"""
        import random

        # Model-specific best parameters
        if model_type == "xgboost":
            best_params = {
                "learning_rate": 0.01,
                "max_depth": 6,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        elif model_type == "lstm":
            best_params = {
                "hidden_units": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
            }
        elif model_type == "random_forest":
            best_params = {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            }
        else:
            best_params = {"param1": 0.5, "param2": 10, "param3": "auto"}

        return {
            "success": True,
            "model_type": model_type,
            "symbol": symbol,
            "best_score": round(random.uniform(0.82, 0.94), 3),
            "best_parameters": best_params,
            "trials_completed": max_trials,
            "optimization_time": f"{random.randint(15, 45)}m {random.randint(10, 59)}s",
            "trial_history": [
                {"trial": i, "score": round(random.uniform(0.70, 0.92), 3)}
                for i in range(max(1, max_trials - 9), max_trials + 1)
            ],
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_default_search_space(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameter search space"""
        if model_type == "xgboost":
            return {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 6, 9, 12],
                "n_estimators": [50, 100, 200, 300],
            }
        elif model_type == "lstm":
            return {
                "hidden_units": [64, 128, 256],
                "num_layers": [1, 2, 3],
                "dropout": [0.1, 0.2, 0.3],
            }
        else:
            return {"param1": [0.1, 0.5, 1.0], "param2": [5, 10, 20]}

    async def ensemble_predict(
        self, model_ids: List[str], symbol: str, horizon: int = 5, weights: List[float] = None
    ) -> Dict[str, Any]:
        """Generate ensemble predictions from multiple models"""
        if not REAL_IMPLEMENTATION:
            return self._mock_ensemble_predict(model_ids, symbol, horizon, weights)

        try:
            ensemble_config = {
                "model_ids": model_ids,
                "symbol": symbol,
                "horizon": horizon,
                "weights": weights or [1.0 / len(model_ids)] * len(model_ids),
            }

            result = await self.ensemble_manager.predict(ensemble_config)

            return {
                "success": True,
                "ensemble_models": model_ids,
                "model_weights": ensemble_config["weights"],
                "symbol": symbol,
                "ensemble_predictions": result.get("predictions"),
                "individual_predictions": result.get("individual_predictions"),
                "ensemble_confidence": result.get("confidence"),
                "model_contributions": result.get("contributions"),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Ensemble prediction failed: {str(e)}"}

    def _mock_ensemble_predict(
        self, model_ids: List[str], symbol: str, horizon: int, weights: List[float]
    ) -> Dict[str, Any]:
        """Mock ensemble predictions"""
        import random

        base_price = 50000.0 if symbol.startswith("BTC") else 3000.0
        weights = weights or [1.0 / len(model_ids)] * len(model_ids)

        # Individual model predictions
        individual_preds = {}
        for model_id in model_ids:
            preds = []
            for i in range(horizon):
                change = random.uniform(-0.02, 0.04)
                price = base_price * (1 + change * (i + 1))
                preds.append(
                    {
                        "step": i + 1,
                        "predicted_price": round(price, 2),
                        "confidence": round(random.uniform(0.65, 0.85), 3),
                    }
                )
            individual_preds[model_id] = preds

        # Ensemble predictions (weighted average)
        ensemble_preds = []
        for i in range(horizon):
            weighted_price = 0
            weighted_conf = 0
            for j, model_id in enumerate(model_ids):
                weighted_price += individual_preds[model_id][i]["predicted_price"] * weights[j]
                weighted_conf += individual_preds[model_id][i]["confidence"] * weights[j]

            ensemble_preds.append(
                {
                    "step": i + 1,
                    "predicted_price": round(weighted_price, 2),
                    "confidence": round(weighted_conf, 3),
                }
            )

        return {
            "success": True,
            "ensemble_models": model_ids,
            "model_weights": weights,
            "symbol": symbol,
            "ensemble_predictions": ensemble_preds,
            "individual_predictions": individual_preds,
            "ensemble_confidence": round(random.uniform(0.78, 0.92), 3),
            "model_contributions": {model_id: weights[i] for i, model_id in enumerate(model_ids)},
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def feature_importance(
        self, model_id: str, analysis_type: str = "shap"
    ) -> Dict[str, Any]:
        """Analyze feature importance for trained model"""
        if not REAL_IMPLEMENTATION:
            return self._mock_feature_importance(model_id, analysis_type)

        try:
            analysis_config = {"model_id": model_id, "analysis_type": analysis_type, "top_n": 20}

            result = await self.predictor.analyze_feature_importance(analysis_config)

            return {
                "success": True,
                "model_id": model_id,
                "analysis_type": analysis_type,
                "feature_importance": result.get("importance_scores"),
                "feature_ranking": result.get("ranking"),
                "cumulative_importance": result.get("cumulative_importance"),
                "interpretation": result.get("interpretation"),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Feature importance analysis failed: {str(e)}"}

    def _mock_feature_importance(self, model_id: str, analysis_type: str) -> Dict[str, Any]:
        """Mock feature importance analysis"""
        import random

        features = [
            "price_sma_20",
            "price_sma_50",
            "rsi_14",
            "macd_signal",
            "volume_sma_10",
            "bb_upper",
            "bb_lower",
            "price_ema_12",
            "price_ema_26",
            "volume_ratio",
            "volatility_20",
            "momentum_10",
            "atr_14",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "cci_20",
            "adx_14",
            "aroon_up",
            "aroon_down",
        ]

        # Generate importance scores
        importance_scores = {}
        for feature in features:
            importance_scores[feature] = round(random.uniform(0.01, 0.25), 4)

        # Sort by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        feature_ranking = [{"feature": f, "importance": imp} for f, imp in sorted_features]

        # Calculate cumulative importance
        total_importance = sum(importance_scores.values())
        cumulative = 0
        cumulative_importance = []
        for feature, importance in sorted_features:
            cumulative += importance / total_importance
            cumulative_importance.append(
                {"feature": feature, "cumulative_importance": round(cumulative, 3)}
            )

        return {
            "success": True,
            "model_id": model_id,
            "analysis_type": analysis_type,
            "feature_importance": importance_scores,
            "feature_ranking": feature_ranking[:10],  # Top 10
            "cumulative_importance": cumulative_importance[:10],
            "interpretation": {
                "most_important": sorted_features[0][0],
                "least_important": sorted_features[-1][0],
                "top_5_contribution": round(
                    sum([imp for _, imp in sorted_features[:5]]) / total_importance, 3
                ),
            },
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }


# Global agent instance
agent = MLModelsAgent()


def async_command(f):
    """Decorator to run async commands"""

    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """A2A ML Models CLI - Advanced ML model management"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if not REAL_IMPLEMENTATION:
        click.echo("⚠️ Running in fallback mode - using mock ML operations")


@cli.command()
@click.argument(
    "model-type", type=click.Choice(["xgboost", "lstm", "random_forest", "transformer"])
)
@click.argument("symbol")
@click.option("--epochs", default=100, help="Number of training epochs")
@click.option("--features", help="Comma-separated feature list")
@click.option("--hyperparams", help="JSON string with hyperparameters")
@click.pass_context
@async_command
async def train(ctx, model_type, symbol, epochs, features, hyperparams):
    """Train ML model on symbol data"""
    try:
        feature_list = features.split(",") if features else None
        hyperparam_dict = json.loads(hyperparams) if hyperparams else None

        result = await agent.train_model(model_type, symbol, epochs, feature_list, hyperparam_dict)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"🎯 Model Training Complete - {model_type.upper()}")
        click.echo("=" * 60)
        click.echo(f"Model ID: {result.get('model_id')}")
        click.echo(f"Symbol: {result.get('symbol')}")
        click.echo(f"Training Accuracy: {result.get('training_accuracy'):.3f}")
        click.echo(f"Validation Accuracy: {result.get('validation_accuracy'):.3f}")
        click.echo(f"Test Accuracy: {result.get('test_accuracy'):.3f}")
        click.echo(f"Epochs Completed: {result.get('epochs_completed')}")
        click.echo(f"Training Time: {result.get('training_time')}")
        click.echo(f"Features Used: {', '.join(result.get('features_used', []))}")

        if result.get("mock"):
            click.echo("🔄 Mock training - enable real implementation for actual model training")

        if ctx.obj["verbose"]:
            click.echo(f"\nModel Path: {result.get('model_path')}")
            click.echo(f"Timestamp: {result.get('timestamp')}")

    except Exception as e:
        click.echo(f"Error training model: {e}", err=True)


@cli.command()
@click.argument("model-id")
@click.argument("symbol")
@click.option("--horizon", default=5, help="Prediction horizon (steps)")
@click.option("--no-confidence", is_flag=True, help="Exclude confidence intervals")
@click.pass_context
@async_command
async def predict(ctx, model_id, symbol, horizon, no_confidence):
    """Generate price predictions using trained model"""
    try:
        result = await agent.predict_prices(model_id, symbol, horizon, not no_confidence)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"📈 Price Predictions - {symbol.upper()}")
        click.echo("=" * 50)
        click.echo(f"Model: {result.get('model_id')}")
        click.echo(f"Model Accuracy: {result.get('model_accuracy'):.3f}")
        click.echo(f"Overall Confidence: {result.get('prediction_confidence'):.3f}")
        click.echo()

        predictions = result.get("predictions", [])
        confidence_intervals = result.get("confidence_intervals", [])

        for i, pred in enumerate(predictions):
            click.echo(
                f"Step {pred['step']}: ${pred['predicted_price']:,.2f} (conf: {pred['confidence']:.3f})"
            )

            if confidence_intervals and i < len(confidence_intervals):
                ci = confidence_intervals[i]
                click.echo(f"  95% CI: ${ci['lower_bound']:,.2f} - ${ci['upper_bound']:,.2f}")

        if result.get("mock"):
            click.echo(
                "\n🔄 Mock predictions - enable real implementation for actual model predictions"
            )

        if ctx.obj["verbose"]:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")

    except Exception as e:
        click.echo(f"Error generating predictions: {e}", err=True)


@cli.command()
@click.argument("model-id")
@click.option("--test-data", help="Path to test data file")
@click.option("--metrics", help="Comma-separated metrics list")
@click.pass_context
@async_command
async def evaluate(ctx, model_id, test_data, metrics):
    """Evaluate model performance"""
    try:
        metric_list = metrics.split(",") if metrics else None

        result = await agent.evaluate_model(model_id, test_data, metric_list)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"📊 Model Evaluation - {model_id}")
        click.echo("=" * 50)

        metrics_data = result.get("evaluation_metrics", {})
        for metric, value in metrics_data.items():
            click.echo(f"{metric.upper()}: {value}")

        click.echo()

        summary = result.get("performance_summary", {})
        if summary:
            click.echo(f"Overall Score: {summary.get('overall_score', 'N/A')}")
            click.echo(f"Recommendation: {summary.get('recommendation', 'N/A')}")

            strengths = summary.get("strengths", [])
            if strengths:
                click.echo(f"Strengths: {', '.join(strengths)}")

            weaknesses = summary.get("weaknesses", [])
            if weaknesses:
                click.echo(f"Weaknesses: {', '.join(weaknesses)}")

        if result.get("mock"):
            click.echo(
                "\n🔄 Mock evaluation - enable real implementation for actual model evaluation"
            )

        if ctx.obj["verbose"]:
            test_preds = result.get("test_predictions", [])
            if test_preds:
                click.echo("\nSample Predictions:")
                for pred in test_preds[:3]:
                    click.echo(
                        f"  Actual: {pred.get('actual')}, Predicted: {pred.get('predicted')}, Error: {pred.get('error')}"
                    )

            click.echo(f"\nEvaluation Date: {result.get('evaluation_date')}")

    except Exception as e:
        click.echo(f"Error evaluating model: {e}", err=True)


@cli.command()
@click.argument(
    "model-type", type=click.Choice(["xgboost", "lstm", "random_forest", "transformer"])
)
@click.argument("symbol")
@click.option("--max-trials", default=50, help="Maximum optimization trials")
@click.option("--search-space", help="JSON string with parameter search space")
@click.pass_context
@async_command
async def optimize(ctx, model_type, symbol, max_trials, search_space):
    """Optimize model hyperparameters"""
    try:
        search_dict = json.loads(search_space) if search_space else None

        result = await agent.optimize_hyperparameters(model_type, symbol, search_dict, max_trials)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"⚡ Hyperparameter Optimization - {model_type.upper()}")
        click.echo("=" * 60)
        click.echo(f"Symbol: {result.get('symbol')}")
        click.echo(f"Best Score: {result.get('best_score'):.3f}")
        click.echo(f"Trials Completed: {result.get('trials_completed')}")
        click.echo(f"Optimization Time: {result.get('optimization_time')}")
        click.echo()

        best_params = result.get("best_parameters", {})
        if best_params:
            click.echo("Best Parameters:")
            for param, value in best_params.items():
                click.echo(f"  {param}: {value}")

        if result.get("mock"):
            click.echo(
                "\n🔄 Mock optimization - enable real implementation for actual hyperparameter tuning"
            )

        if ctx.obj["verbose"]:
            trial_history = result.get("trial_history", [])
            if trial_history:
                click.echo(f"\nRecent Trials:")
                for trial in trial_history[-5:]:
                    click.echo(f"  Trial {trial['trial']}: {trial['score']:.3f}")

            click.echo(f"\nTimestamp: {result.get('timestamp')}")

    except Exception as e:
        click.echo(f"Error optimizing hyperparameters: {e}", err=True)


@cli.command()
@click.argument("model-ids", nargs=-1, required=True)
@click.argument("symbol")
@click.option("--horizon", default=5, help="Prediction horizon")
@click.option("--weights", help="Comma-separated model weights")
@click.pass_context
@async_command
async def ensemble(ctx, model_ids, symbol, horizon, weights):
    """Generate ensemble predictions from multiple models"""
    try:
        weight_list = [float(w) for w in weights.split(",")] if weights else None

        result = await agent.ensemble_predict(list(model_ids), symbol, horizon, weight_list)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"🔮 Ensemble Predictions - {symbol.upper()}")
        click.echo("=" * 60)
        click.echo(f"Models: {', '.join(result.get('ensemble_models', []))}")
        click.echo(f"Weights: {result.get('model_weights')}")
        click.echo(f"Ensemble Confidence: {result.get('ensemble_confidence'):.3f}")
        click.echo()

        ensemble_preds = result.get("ensemble_predictions", [])
        for pred in ensemble_preds:
            click.echo(
                f"Step {pred['step']}: ${pred['predicted_price']:,.2f} (conf: {pred['confidence']:.3f})"
            )

        if result.get("mock"):
            click.echo(
                "\n🔄 Mock ensemble - enable real implementation for actual ensemble predictions"
            )

        if ctx.obj["verbose"]:
            contributions = result.get("model_contributions", {})
            if contributions:
                click.echo(f"\nModel Contributions:")
                for model, contrib in contributions.items():
                    click.echo(f"  {model}: {contrib:.1%}")

            click.echo(f"\nTimestamp: {result.get('timestamp')}")

    except Exception as e:
        click.echo(f"Error generating ensemble predictions: {e}", err=True)


@cli.command()
@click.argument("model-id")
@click.option(
    "--analysis-type",
    default="shap",
    type=click.Choice(["shap", "permutation", "gain"]),
    help="Feature importance analysis type",
)
@click.pass_context
@async_command
async def importance(ctx, model_id, analysis_type):
    """Analyze feature importance for trained model"""
    try:
        result = await agent.feature_importance(model_id, analysis_type)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"🔍 Feature Importance Analysis - {model_id}")
        click.echo("=" * 60)
        click.echo(f"Analysis Type: {analysis_type.upper()}")
        click.echo()

        feature_ranking = result.get("feature_ranking", [])
        if feature_ranking:
            click.echo("Top Features:")
            for i, feature_info in enumerate(feature_ranking[:10], 1):
                click.echo(f"{i:2d}. {feature_info['feature']}: {feature_info['importance']:.4f}")
            click.echo()

        interpretation = result.get("interpretation", {})
        if interpretation:
            click.echo(f"Most Important: {interpretation.get('most_important')}")
            click.echo(f"Top 5 Contribution: {interpretation.get('top_5_contribution', 0):.1%}")

        if result.get("mock"):
            click.echo(
                "\n🔄 Mock analysis - enable real implementation for actual feature importance"
            )

        if ctx.obj["verbose"]:
            cumulative = result.get("cumulative_importance", [])
            if cumulative:
                click.echo(f"\nCumulative Importance:")
                for feature_info in cumulative[:5]:
                    click.echo(
                        f"  {feature_info['feature']}: {feature_info['cumulative_importance']:.1%}"
                    )

            click.echo(f"\nTimestamp: {result.get('timestamp')}")

    except Exception as e:
        click.echo(f"Error analyzing feature importance: {e}", err=True)


@cli.command()
@click.pass_context
def models(ctx):
    """List available models"""
    click.echo("🤖 Available ML Models:")
    click.echo()
    for i, (model_id, info) in enumerate(agent.models.items(), 1):
        status_emoji = "✅" if info["status"] == "trained" else "🔄"
        click.echo(f"{i:2d}. {status_emoji} {model_id}")
        click.echo(f"    Type: {info['type']}")
        click.echo(f"    Status: {info['status']}")
        click.echo(f"    Accuracy: {info.get('accuracy', 'N/A')}")
        click.echo()


@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    click.echo("🔧 ML Models Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")


@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    click.echo("🏥 ML Models Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo(f"Available Models: {len(agent.models)}")
    click.echo(f"Implementation: {'Real' if REAL_IMPLEMENTATION else 'Fallback'}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    cli()
