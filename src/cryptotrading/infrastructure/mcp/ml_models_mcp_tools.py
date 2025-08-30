"""
ML Models MCP Tools - All machine learning model calculations
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...core.ml.models import MLModels
from ...core.ml.production_models import ProductionMLModels
from ...core.ml.training import MLTraining

logger = logging.getLogger(__name__)


class MLModelsMCPTools:
    """MCP tools for all ML model calculations"""

    def __init__(self):
        self.production_models = ProductionMLModels()
        self.ml_models = MLModels()
        self.ml_training = MLTraining()
        self.tools = self._create_tools()

    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions for ML models"""
        return [
            {
                "name": "train_model",
                "description": "Train ML model with specified parameters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "enum": ["lstm", "transformer", "ensemble"],
                            "description": "Type of model to train",
                        },
                        "training_data": {
                            "type": "string",
                            "description": "JSON string containing training data",
                        },
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Feature columns to use",
                        },
                        "target": {"type": "string", "description": "Target column name"},
                        "hyperparameters": {
                            "type": "object",
                            "description": "Model hyperparameters",
                        },
                    },
                    "required": ["model_type", "training_data", "features", "target"],
                },
            },
            {
                "name": "predict_prices",
                "description": "Generate price predictions using trained models",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string", "description": "ID of trained model to use"},
                        "input_data": {
                            "type": "string",
                            "description": "JSON string containing input data",
                        },
                        "prediction_horizon": {
                            "type": "integer",
                            "default": 1,
                            "description": "Number of periods to predict",
                        },
                        "confidence_interval": {"type": "boolean", "default": True},
                    },
                    "required": ["model_id", "input_data"],
                },
            },
            {
                "name": "evaluate_model",
                "description": "Evaluate model performance on test data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string", "description": "ID of model to evaluate"},
                        "test_data": {
                            "type": "string",
                            "description": "JSON string containing test data",
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["mse", "mae", "r2"],
                        },
                    },
                    "required": ["model_id", "test_data"],
                },
            },
            {
                "name": "optimize_hyperparameters",
                "description": "Optimize model hyperparameters using grid/random search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "enum": ["lstm", "transformer", "ensemble"],
                        },
                        "training_data": {
                            "type": "string",
                            "description": "JSON string containing training data",
                        },
                        "param_grid": {
                            "type": "object",
                            "description": "Parameter grid for optimization",
                        },
                        "optimization_method": {
                            "type": "string",
                            "enum": ["grid", "random", "bayesian"],
                            "default": "grid",
                        },
                    },
                    "required": ["model_type", "training_data", "param_grid"],
                },
            },
            {
                "name": "ensemble_predict",
                "description": "Generate ensemble predictions from multiple models",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of model IDs to ensemble",
                        },
                        "input_data": {
                            "type": "string",
                            "description": "JSON string containing input data",
                        },
                        "ensemble_method": {
                            "type": "string",
                            "enum": ["average", "weighted", "voting"],
                            "default": "average",
                        },
                        "weights": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Weights for weighted ensemble",
                        },
                    },
                    "required": ["model_ids", "input_data"],
                },
            },
            {
                "name": "feature_importance",
                "description": "Calculate feature importance for trained model",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string", "description": "ID of trained model"},
                        "method": {
                            "type": "string",
                            "enum": ["permutation", "shap", "built_in"],
                            "default": "permutation",
                        },
                    },
                    "required": ["model_id"],
                },
            },
        ]

    def register_tools(self, server):
        """Register all ML model tools with MCP server"""
        for tool_def in self.tools:
            tool_name = tool_def["name"]

            @server.call_tool()
            async def handle_tool(name: str, arguments: dict) -> dict:
                if name == tool_name:
                    return await self.handle_tool_call(tool_name, arguments)
                return {"error": f"Unknown tool: {name}"}

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls for ML models"""
        try:
            if tool_name == "train_model":
                return await self._handle_train_model(arguments)
            elif tool_name == "predict_prices":
                return await self._handle_predict_prices(arguments)
            elif tool_name == "evaluate_model":
                return await self._handle_evaluate_model(arguments)
            elif tool_name == "optimize_hyperparameters":
                return await self._handle_optimize_hyperparameters(arguments)
            elif tool_name == "ensemble_predict":
                return await self._handle_ensemble_predict(arguments)
            elif tool_name == "feature_importance":
                return await self._handle_feature_importance(arguments)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error in ML model tool {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_train_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model training requests"""
        try:
            model_type = args["model_type"]
            training_data = pd.read_json(args["training_data"])
            features = args["features"]
            target = args["target"]
            hyperparameters = args.get("hyperparameters", {})

            # Prepare training data
            X = training_data[features]
            y = training_data[target]

            # Train model
            model_result = await self.ml_training.train_model(model_type, X, y, hyperparameters)

            return {
                "success": True,
                "model_id": model_result["model_id"],
                "training_metrics": model_result["metrics"],
                "model_type": model_type,
                "features": features,
                "target": target,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_predict_prices(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle price prediction requests"""
        try:
            model_id = args["model_id"]
            input_data = pd.read_json(args["input_data"])
            prediction_horizon = args.get("prediction_horizon", 1)
            confidence_interval = args.get("confidence_interval", True)

            # Generate predictions
            predictions = await self.production_models.predict(
                model_id, input_data, prediction_horizon, confidence_interval
            )

            return {
                "success": True,
                "predictions": predictions["values"],
                "confidence_intervals": predictions.get("confidence_intervals")
                if confidence_interval
                else None,
                "model_id": model_id,
                "prediction_horizon": prediction_horizon,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_evaluate_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model evaluation requests"""
        try:
            model_id = args["model_id"]
            test_data = pd.read_json(args["test_data"])
            metrics = args.get("metrics", ["mse", "mae", "r2"])

            # Evaluate model
            evaluation_results = await self.production_models.evaluate_model(
                model_id, test_data, metrics
            )

            return {
                "success": True,
                "evaluation_metrics": evaluation_results,
                "model_id": model_id,
                "test_samples": len(test_data),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_optimize_hyperparameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hyperparameter optimization requests"""
        try:
            model_type = args["model_type"]
            training_data = pd.read_json(args["training_data"])
            param_grid = args["param_grid"]
            optimization_method = args.get("optimization_method", "grid")

            # Optimize hyperparameters
            optimization_results = await self.ml_training.optimize_hyperparameters(
                model_type, training_data, param_grid, optimization_method
            )

            return {
                "success": True,
                "best_parameters": optimization_results["best_params"],
                "best_score": optimization_results["best_score"],
                "optimization_method": optimization_method,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_ensemble_predict(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ensemble prediction requests"""
        try:
            model_ids = args["model_ids"]
            input_data = pd.read_json(args["input_data"])
            ensemble_method = args.get("ensemble_method", "average")
            weights = args.get("weights")

            # Generate ensemble predictions
            ensemble_predictions = await self.production_models.ensemble_predict(
                model_ids, input_data, ensemble_method, weights
            )

            return {
                "success": True,
                "ensemble_predictions": ensemble_predictions["values"],
                "individual_predictions": ensemble_predictions["individual"],
                "ensemble_method": ensemble_method,
                "model_ids": model_ids,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_feature_importance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature importance calculation requests"""
        try:
            model_id = args["model_id"]
            method = args.get("method", "permutation")

            # Calculate feature importance
            importance_results = await self.production_models.calculate_feature_importance(
                model_id, method
            )

            return {
                "success": True,
                "feature_importance": importance_results["importance"],
                "method": method,
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
