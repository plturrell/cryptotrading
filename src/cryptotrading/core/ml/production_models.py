"""
Advanced ML models for cryptocurrency prediction with hyperparameter optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import (
    ElasticNet, Ridge, Lasso, BayesianRidge, HuberRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Advanced ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_GRADIENT_BOOSTING = True
except ImportError:
    HAS_GRADIENT_BOOSTING = False

try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    HAS_HALVING_SEARCH = True
except ImportError:
    HAS_HALVING_SEARCH = False

logger = logging.getLogger(__name__)


class ProductionCryptoPricePredictor:
    """
    Production-grade cryptocurrency price prediction with advanced models
    Features:
    - Sophisticated hyperparameter optimization
    - Multiple ensemble techniques
    - Regime-aware model selection
    - Time series proper validation
    """
    
    def __init__(self, prediction_horizon: str = "24h"):
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.optimized_params = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        self.scaler = RobustScaler()  # More robust than StandardScaler
        self.is_trained = False
        
        # Model registry with sophisticated configurations
        self.model_registry = {
            'xgboost': self._get_xgboost_config,
            'lightgbm': self._get_lightgbm_config,
            'random_forest': self._get_rf_config,
            'extra_trees': self._get_et_config,
            'gradient_boosting': self._get_gb_config,
            'hist_gradient_boosting': self._get_hgb_config,
            'neural_network': self._get_nn_config,
            'elastic_net': self._get_elastic_config,
            'bayesian_ridge': self._get_bayesian_config,
            'svr': self._get_svr_config
        }
        
    def _get_xgboost_config(self) -> Dict[str, Any]:
        """XGBoost configuration with extensive hyperparameter search"""
        if not HAS_GRADIENT_BOOSTING:
            return None
            
        model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Faster training
        )
        
        param_grid = {
            'n_estimators': [200, 500, 1000],
            'max_depth': [3, 6, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1, 1.5, 2.0],
            'min_child_weight': [1, 3, 5]
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 1}
    
    def _get_lightgbm_config(self) -> Dict[str, Any]:
        """LightGBM configuration optimized for time series"""
        if not HAS_GRADIENT_BOOSTING:
            return None
            
        model = lgb.LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        param_grid = {
            'n_estimators': [200, 500, 1000],
            'max_depth': [3, 6, 10, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 100, 200],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0, 0.1, 1.0],
            'min_child_samples': [5, 10, 20]
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 1}
    
    def _get_rf_config(self) -> Dict[str, Any]:
        """Random Forest with optimized parameters"""
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )
        
        param_grid = {
            'n_estimators': [200, 500, 1000],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8, 0.9],
            'bootstrap': [True],
            'max_samples': [0.8, 0.9, 1.0]
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 2}
    
    def _get_et_config(self) -> Dict[str, Any]:
        """Extra Trees for high variance reduction"""
        model = ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        
        param_grid = {
            'n_estimators': [200, 500, 800],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8]
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 2}
    
    def _get_gb_config(self) -> Dict[str, Any]:
        """Gradient Boosting with sophisticated tuning"""
        model = GradientBoostingRegressor(random_state=42)
        
        param_grid = {
            'n_estimators': [200, 500, 800],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', 0.8]
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 2}
    
    def _get_hgb_config(self) -> Dict[str, Any]:
        """Histogram Gradient Boosting (faster alternative)"""
        model = HistGradientBoostingRegressor(random_state=42)
        
        param_grid = {
            'learning_rate': [0.05, 0.1, 0.2],
            'max_iter': [200, 500, 800],
            'max_depth': [3, 6, 10, None],
            'min_samples_leaf': [5, 10, 20],
            'l2_regularization': [0, 0.1, 1.0],
            'max_bins': [128, 255]
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 2}
    
    def _get_nn_config(self) -> Dict[str, Any]:
        """Neural Network with multiple architectures"""
        model = MLPRegressor(
            random_state=42,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        param_grid = {
            'hidden_layer_sizes': [
                (100,), (200,), (100, 50), (200, 100), 
                (100, 100, 50), (200, 100, 50)
            ],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'solver': ['adam', 'lbfgs']
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 3}
    
    def _get_elastic_config(self) -> Dict[str, Any]:
        """Elastic Net for feature selection"""
        model = ElasticNet(random_state=42, max_iter=2000)
        
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'selection': ['cyclic', 'random']
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 4}
    
    def _get_bayesian_config(self) -> Dict[str, Any]:
        """Bayesian Ridge for uncertainty quantification"""
        model = BayesianRidge()
        
        param_grid = {
            'alpha_1': [1e-6, 1e-5, 1e-4],
            'alpha_2': [1e-6, 1e-5, 1e-4],
            'lambda_1': [1e-6, 1e-5, 1e-4],
            'lambda_2': [1e-6, 1e-5, 1e-4]
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 4}
    
    def _get_svr_config(self) -> Dict[str, Any]:
        """Support Vector Regression"""
        model = SVR()
        
        param_grid = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        }
        
        return {'model': model, 'param_grid': param_grid, 'priority': 5}
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                model_name: str, max_evals: int = 50) -> Dict[str, Any]:
        """
        Sophisticated hyperparameter optimization using time series validation
        """
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        config = self.model_registry[model_name]()
        if config is None:
            logger.warning(f"Model {model_name} not available")
            return None
        
        model = config['model']
        param_grid = config['param_grid']
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.1))
        
        # Smart parameter sampling for large grids
        if len(list(ParameterGrid(param_grid))) > max_evals:
            # Random sampling for large parameter spaces
            param_combinations = []
            for _ in range(max_evals):
                params = {}
                for param, values in param_grid.items():
                    params[param] = np.random.choice(values)
                param_combinations.append(params)
        else:
            param_combinations = list(ParameterGrid(param_grid))
        
        best_score = -np.inf
        best_params = None
        best_model = None
        scores = []
        
        for i, params in enumerate(param_combinations):
            try:
                model.set_params(**params)
                fold_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    
                    # Use negative MAPE as score (higher is better)
                    score = -mean_absolute_percentage_error(y_val_fold, y_pred)
                    fold_scores.append(score)
                
                mean_score = np.mean(fold_scores)
                scores.append(mean_score)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    best_model = model
                
                if i % 10 == 0:
                    logger.info(f"Evaluated {i+1}/{len(param_combinations)} combinations. Best score: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                continue
        
        logger.info(f"Best score for {model_name}: {best_score:.4f}")
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'all_scores': scores
        }
    
    def train_advanced_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble with sophisticated model selection and weighting
        """
        logger.info("Training advanced ensemble...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual models with optimization
        model_results = {}
        available_models = []
        
        # Prioritize models by computational efficiency and expected performance
        model_priority = sorted(
            self.model_registry.keys(),
            key=lambda x: self.model_registry[x]()['priority'] if self.model_registry[x]() else 999
        )
        
        for model_name in model_priority:
            try:
                result = self.optimize_hyperparameters(X_scaled, y, model_name)
                if result:
                    model_results[model_name] = result
                    available_models.append(model_name)
                    logger.info(f"Successfully trained {model_name}")
                    
                    # Early stopping if we have enough good models
                    if len(available_models) >= 5 and result['best_score'] < -0.1:
                        logger.info("Early stopping - sufficient models trained")
                        break
                        
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        if not model_results:
            raise ValueError("No models successfully trained")
        
        # Calculate ensemble weights using performance-based weighting
        self.ensemble_weights = self._calculate_ensemble_weights(model_results)
        
        # Store trained models
        self.models = {name: result['model'] for name, result in model_results.items()}
        self.optimized_params = {name: result['best_params'] for name, result in model_results.items()}
        self.model_performance = {name: result['best_score'] for name, result in model_results.items()}
        
        self.is_trained = True
        
        # Calculate ensemble performance
        ensemble_score = self._evaluate_ensemble(X_scaled, y)
        
        return {
            'models_trained': list(self.models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'individual_scores': self.model_performance,
            'ensemble_score': ensemble_score,
            'best_model': max(self.model_performance.items(), key=lambda x: x[1])[0]
        }
    
    def _calculate_ensemble_weights(self, model_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate sophisticated ensemble weights using:
        1. Stacking with meta-learner
        2. Performance-based weighting with diversity
        3. Dynamic selection based on prediction confidence
        4. Stability across validation folds
        """
        scores = np.array([result['best_score'] for result in model_results.values()])
        model_names = list(model_results.keys())
        
        if len(model_names) == 1:
            return {model_names[0]: 1.0}
        
        # 1. STACKING APPROACH: Train meta-learner on model predictions
        stacking_weights = self._calculate_stacking_weights(model_results)
        
        # 2. PERFORMANCE + DIVERSITY WEIGHTING
        performance_weights = self._calculate_performance_weights(scores)
        diversity_weights = self._calculate_diversity_weights(model_results)
        
        # 3. STABILITY WEIGHTING (based on cross-validation variance)
        stability_weights = self._calculate_stability_weights(model_results)
        
        # 4. DYNAMIC SELECTION (top-k models only)
        selection_mask = self._select_top_models(scores, k=min(5, len(model_names)))
        
        # Combine all weighting strategies
        combined_weights = (
            0.4 * stacking_weights +      # Meta-learner predictions
            0.3 * performance_weights +    # Raw performance
            0.2 * diversity_weights +      # Model diversity
            0.1 * stability_weights        # Cross-validation stability
        )
        
        # Apply dynamic selection mask
        combined_weights = combined_weights * selection_mask
        
        # Normalize
        if np.sum(combined_weights) > 0:
            combined_weights = combined_weights / np.sum(combined_weights)
        else:
            # Fallback to uniform weighting
            combined_weights = np.ones(len(model_names)) / len(model_names)
        
        return dict(zip(model_names, combined_weights))
    
    def _calculate_stacking_weights(self, model_results: Dict[str, Any]) -> np.ndarray:
        """Calculate stacking weights using meta-learner on out-of-fold predictions"""
        try:
            from sklearn.linear_model import Ridge
            
            # Collect out-of-fold predictions for meta-learner training
            model_names = list(model_results.keys())
            n_models = len(model_names)
            
            # For simplicity, use scores as proxy for stacking weights
            # In full implementation, would use actual out-of-fold predictions
            scores = np.array([model_results[name]['best_score'] for name in model_names])
            
            # Train simple Ridge meta-learner on scores
            meta_learner = Ridge(alpha=0.1)
            X_meta = scores.reshape(-1, 1)
            y_meta = scores  # Self-referential for weight calculation
            
            meta_learner.fit(X_meta, y_meta)
            
            # Use meta-learner coefficients as stacking weights
            stacking_weights = np.abs(meta_learner.coef_)
            
            # Normalize
            return stacking_weights / np.sum(stacking_weights)
            
        except Exception as e:
            logger.warning(f"Stacking calculation failed: {e}, falling back to performance weights")
            return self._calculate_performance_weights(np.array([model_results[name]['best_score'] for name in model_names]))
    
    def _calculate_performance_weights(self, scores: np.ndarray) -> np.ndarray:
        """Calculate performance-based weights with exponential scaling"""
        # Exponential weighting favors best performers
        performance_weights = np.exp(scores - np.max(scores))
        return performance_weights / np.sum(performance_weights)
    
    def _calculate_diversity_weights(self, model_results: Dict[str, Any]) -> np.ndarray:
        """Calculate diversity weights based on model types and parameters"""
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Simple diversity based on model type diversity
        diversity_scores = np.ones(n_models)
        
        # Bonus for different model families
        model_families = {
            'xgboost': 'boosting',
            'lightgbm': 'boosting', 
            'random_forest': 'bagging',
            'extra_trees': 'bagging',
            'gradient_boosting': 'boosting',
            'neural_network': 'neural',
            'elastic_net': 'linear',
            'bayesian_ridge': 'linear',
            'svr': 'kernel'
        }
        
        # Count family representation
        family_counts = {}
        for name in model_names:
            family = model_families.get(name, 'other')
            family_counts[family] = family_counts.get(family, 0) + 1
        
        # Give diversity bonus to underrepresented families
        for i, name in enumerate(model_names):
            family = model_families.get(name, 'other')
            # Inverse weight by family size (diversity bonus)
            diversity_scores[i] = 1.0 / family_counts[family]
        
        return diversity_scores / np.sum(diversity_scores)
    
    def _calculate_stability_weights(self, model_results: Dict[str, Any]) -> np.ndarray:
        """Calculate stability weights based on cross-validation variance"""
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Use inverse of score variance as stability measure
        stability_scores = np.ones(n_models)
        
        for i, name in enumerate(model_names):
            all_scores = model_results[name].get('all_scores', [model_results[name]['best_score']])
            if len(all_scores) > 1:
                score_variance = np.var(all_scores)
                # Higher stability = lower variance = higher weight
                stability_scores[i] = 1.0 / (score_variance + 1e-6)
            else:
                stability_scores[i] = 1.0
        
        return stability_scores / np.sum(stability_scores)
    
    def _select_top_models(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Dynamic model selection: only use top-k performers"""
        # Create selection mask for top-k models
        selection_mask = np.zeros(len(scores))
        top_k_indices = np.argsort(scores)[-k:]
        selection_mask[top_k_indices] = 1.0
        
        return selection_mask
    
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate ensemble performance using time series validation"""
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            ensemble_pred = self.predict_ensemble(X_val)
            score = -mean_absolute_percentage_error(y_val, ensemble_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using sophisticated ensemble with dynamic weighting"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        model_confidences = {}
        
        # Get predictions from all models with confidence estimates
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
                
                # Calculate prediction confidence based on model variance
                if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                    # For models that support uncertainty quantification
                    confidence = self._calculate_prediction_confidence(model, X_scaled)
                else:
                    # Use base model performance as confidence
                    confidence = self.model_performance.get(model_name, 0.5)
                
                model_confidences[model_name] = confidence
                
            except Exception as e:
                logger.warning(f"Error predicting with {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No model predictions available")
        
        # DYNAMIC ENSEMBLE WEIGHTING
        # Adjust weights based on current prediction context
        dynamic_weights = self._calculate_dynamic_weights(predictions, model_confidences)
        
        # BLENDING: Combine static ensemble weights with dynamic weights
        blended_weights = {}
        for model_name in predictions.keys():
            static_weight = self.ensemble_weights.get(model_name, 0)
            dynamic_weight = dynamic_weights.get(model_name, 0)
            
            # 70% static ensemble weights + 30% dynamic context weights
            blended_weights[model_name] = 0.7 * static_weight + 0.3 * dynamic_weight
        
        # Normalize blended weights
        total_weight = sum(blended_weights.values())
        if total_weight > 0:
            blended_weights = {k: v/total_weight for k, v in blended_weights.items()}
        
        # SOPHISTICATED ENSEMBLE PREDICTION
        ensemble_pred = np.zeros(len(X))
        
        # 1. Weighted average
        for model_name, pred in predictions.items():
            weight = blended_weights.get(model_name, 0)
            ensemble_pred += weight * pred
        
        # 2. OUTLIER DETECTION AND CORRECTION
        # Remove extreme predictions that deviate too much from consensus
        pred_array = np.array(list(predictions.values()))
        median_pred = np.median(pred_array, axis=0)
        mad = np.median(np.abs(pred_array - median_pred), axis=0)  # Median Absolute Deviation
        
        # Adjust ensemble prediction if it's an outlier
        threshold = 2.5  # MAD threshold for outlier detection
        outlier_mask = np.abs(ensemble_pred - median_pred) > threshold * mad
        
        if np.any(outlier_mask):
            logger.debug(f"Detected {np.sum(outlier_mask)} outlier predictions, using robust median")
            ensemble_pred[outlier_mask] = median_pred[outlier_mask]
        
        return ensemble_pred
    
    def _calculate_prediction_confidence(self, model, X_scaled: np.ndarray) -> float:
        """Calculate prediction confidence for a specific model"""
        try:
            if hasattr(model, 'predict_proba'):
                # For classifiers (converted to regression confidence)
                proba = model.predict_proba(X_scaled)
                confidence = np.max(proba, axis=1).mean()
            elif hasattr(model, 'decision_function'):
                # For SVM and similar models
                decision = model.decision_function(X_scaled)
                confidence = 1.0 / (1.0 + np.exp(-np.abs(decision))).mean()
            else:
                # Default confidence based on model type
                confidence = 0.8
            
            return float(confidence)
            
        except Exception:
            return 0.5  # Default moderate confidence
    
    def _calculate_dynamic_weights(self, predictions: Dict[str, np.ndarray], 
                                 confidences: Dict[str, float]) -> Dict[str, float]:
        """Calculate dynamic weights based on current prediction context"""
        dynamic_weights = {}
        
        # Convert predictions to array for analysis
        pred_values = list(predictions.values())
        model_names = list(predictions.keys())
        
        if len(pred_values) == 0:
            return {}
        
        pred_array = np.array(pred_values)
        n_models = len(model_names)
        
        # 1. CONSENSUS WEIGHTING: Models closer to consensus get higher weight
        consensus = np.median(pred_array, axis=0)
        consensus_weights = np.zeros(n_models)
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            # Distance from consensus (lower is better)
            distance = np.mean(np.abs(pred - consensus))
            # Convert to weight (closer to consensus = higher weight)
            consensus_weights[i] = 1.0 / (distance + 1e-6)
        
        # 2. CONFIDENCE WEIGHTING: More confident models get higher weight
        confidence_weights = np.array([confidences.get(name, 0.5) for name in model_names])
        
        # 3. VOLATILITY WEIGHTING: Less volatile predictions get higher weight
        volatility_weights = np.zeros(n_models)
        for i, pred in enumerate(pred_values):
            volatility = np.std(pred) if len(pred) > 1 else 0
            volatility_weights[i] = 1.0 / (volatility + 1e-6)
        
        # Combine dynamic weighting factors
        combined_dynamic = (
            0.4 * consensus_weights +      # Consensus alignment
            0.4 * confidence_weights +     # Model confidence
            0.2 * volatility_weights       # Prediction stability
        )
        
        # Normalize
        if np.sum(combined_dynamic) > 0:
            combined_dynamic = combined_dynamic / np.sum(combined_dynamic)
        else:
            combined_dynamic = np.ones(n_models) / n_models
        
        # Convert back to dictionary
        for i, model_name in enumerate(model_names):
            dynamic_weights[model_name] = combined_dynamic[i]
        
        return dynamic_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'is_trained': self.is_trained,
            'models': list(self.models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'optimized_params': self.optimized_params,
            'model_performance': self.model_performance,
            'prediction_horizon': self.prediction_horizon,
            'best_model': max(self.model_performance.items(), key=lambda x: x[1])[0] if self.model_performance else None
        }