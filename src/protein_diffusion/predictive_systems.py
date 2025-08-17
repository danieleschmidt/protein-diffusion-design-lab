"""
Predictive Systems & Autonomous Optimization
Generation 4: Self-Adaptive Intelligence

Advanced predictive systems with time-series forecasting, adaptive optimization,
and autonomous decision-making for protein design evolution.
"""

import asyncio
import json
import time
import uuid
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import itertools
from collections import defaultdict, deque
import math
import random
import copy
from statistics import mean, stdev

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.optimize import minimize, differential_evolution
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PredictionModel(Enum):
    """Types of prediction models."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    TIME_SERIES_ARIMA = "time_series_arima"
    GAUSSIAN_PROCESS = "gaussian_process"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"


class OptimizationStrategy(Enum):
    """Types of optimization strategies."""
    GREEDY = "greedy"
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    SIMULATED_ANNEALING = "simulated_annealing"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    SHORT_TERM = "short_term"      # 1-5 steps ahead
    MEDIUM_TERM = "medium_term"    # 5-20 steps ahead
    LONG_TERM = "long_term"        # 20+ steps ahead
    ADAPTIVE = "adaptive"          # Dynamic horizon


@dataclass
class PredictionRequest:
    """Request for making predictions."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    features: List[float] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    confidence_required: float = 0.95
    target_metrics: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PredictionResult:
    """Result of prediction."""
    request_id: str
    predictions: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    model_used: str
    prediction_accuracy: float
    uncertainty: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_time: float = 0.0
    model_version: str = "1.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OptimizationObjective:
    """Optimization objective definition."""
    name: str
    target_value: float
    weight: float = 1.0
    direction: str = "maximize"  # "maximize" or "minimize"
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    tolerance: float = 0.01


class TimeSeries:
    """Time series data management and forecasting."""
    
    def __init__(self, max_history: int = 1000):
        self.data: deque = deque(maxlen=max_history)
        self.timestamps: deque = deque(maxlen=max_history)
        self.features: List[str] = []
        self.models: Dict[str, Any] = {}
        
    def add_data_point(self, timestamp: datetime, values: Dict[str, float]):
        """Add new data point to time series."""
        self.timestamps.append(timestamp)
        self.data.append(values)
        
        # Update feature list
        self.features = list(set(self.features) | set(values.keys()))
    
    def get_feature_series(self, feature: str, window: int = None) -> List[float]:
        """Get time series for specific feature."""
        series = []
        for data_point in self.data:
            if feature in data_point:
                series.append(data_point[feature])
            else:
                series.append(0.0)  # Default value for missing data
        
        if window and len(series) > window:
            return series[-window:]
        
        return series
    
    def smooth_series(self, feature: str, window: int = 5) -> List[float]:
        """Apply smoothing to time series."""
        series = self.get_feature_series(feature)
        
        if len(series) < window:
            return series
        
        if SCIPY_AVAILABLE and len(series) > window:
            # Use Savitzky-Golay filter for smoothing
            try:
                smoothed = savgol_filter(series, window_length=min(window, len(series)//2*2-1), polyorder=2)
                return smoothed.tolist()
            except:
                pass
        
        # Simple moving average fallback
        smoothed = []
        for i in range(len(series)):
            start = max(0, i - window//2)
            end = min(len(series), i + window//2 + 1)
            avg = np.mean(series[start:end])
            smoothed.append(avg)
        
        return smoothed
    
    def detect_trends(self, feature: str, window: int = 20) -> Dict[str, Any]:
        """Detect trends in time series."""
        series = self.get_feature_series(feature, window)
        
        if len(series) < 3:
            return {"trend": "insufficient_data", "strength": 0.0}
        
        # Calculate linear trend
        x = np.arange(len(series))
        
        if SKLEARN_AVAILABLE:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), series)
            slope = model.coef_[0]
            r2 = model.score(x.reshape(-1, 1), series)
        else:
            # Simple linear regression
            n = len(series)
            sum_x = np.sum(x)
            sum_y = np.sum(series)
            sum_xy = np.sum(x * series)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            r2 = 0.5  # Mock R² value
        
        # Determine trend direction and strength
        if abs(slope) < 1e-6:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        strength = min(1.0, abs(slope) * 10)  # Normalize strength
        
        return {
            "trend": trend,
            "slope": slope,
            "strength": strength,
            "r2": r2,
            "volatility": np.std(series) / max(1e-8, np.mean(series))
        }
    
    def forecast(
        self, 
        feature: str, 
        steps_ahead: int = 5, 
        model_type: str = "linear"
    ) -> Dict[str, Any]:
        """Forecast future values."""
        series = self.get_feature_series(feature)
        
        if len(series) < 3:
            return {
                "forecasts": [series[-1] if series else 0.0] * steps_ahead,
                "confidence_intervals": [(series[-1] if series else 0.0, series[-1] if series else 0.0)] * steps_ahead,
                "model": "last_value",
                "accuracy": 0.0
            }
        
        x = np.arange(len(series))
        forecast_x = np.arange(len(series), len(series) + steps_ahead)
        
        if SKLEARN_AVAILABLE and model_type == "linear":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), series)
            forecasts = model.predict(forecast_x.reshape(-1, 1))
            accuracy = model.score(x.reshape(-1, 1), series)
        else:
            # Simple trend extrapolation
            trend = self.detect_trends(feature)
            slope = trend["slope"]
            last_value = series[-1]
            
            forecasts = [last_value + slope * (i + 1) for i in range(steps_ahead)]
            accuracy = trend["r2"]
        
        # Calculate confidence intervals (simplified)
        std_error = np.std(series) / np.sqrt(len(series))
        confidence_intervals = [
            (forecast - 1.96 * std_error, forecast + 1.96 * std_error)
            for forecast in forecasts
        ]
        
        return {
            "forecasts": forecasts.tolist() if hasattr(forecasts, 'tolist') else forecasts,
            "confidence_intervals": confidence_intervals,
            "model": model_type,
            "accuracy": accuracy
        }


class PredictiveModel:
    """Predictive model for optimization metrics."""
    
    def __init__(self, model_type: PredictionModel = PredictionModel.RANDOM_FOREST):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.training_history: List[Dict[str, Any]] = []
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.model_version = "1.0"
        
        if SKLEARN_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the predictive model."""
        if not SKLEARN_AVAILABLE:
            return
        
        if self.model_type == PredictionModel.LINEAR_REGRESSION:
            self.model = LinearRegression()
        elif self.model_type == PredictionModel.RANDOM_FOREST:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == PredictionModel.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == PredictionModel.NEURAL_NETWORK:
            self.model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.scaler = StandardScaler()
    
    async def train(
        self, 
        training_data: List[Dict[str, Any]], 
        target_metrics: List[str]
    ) -> Dict[str, Any]:
        """Train the predictive model."""
        
        if not SKLEARN_AVAILABLE or not training_data:
            return {"status": "training_failed", "reason": "insufficient_data_or_dependencies"}
        
        logger.info(f"Training {self.model_type.value} model with {len(training_data)} samples")
        
        start_time = time.time()
        
        # Prepare training data
        X, y_dict = self._prepare_training_data(training_data, target_metrics)
        
        if len(X) < 5:  # Minimum samples for training
            return {"status": "training_failed", "reason": "insufficient_samples"}
        
        training_results = {}
        
        for metric in target_metrics:
            if metric not in y_dict:
                continue
            
            y = y_dict[metric]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model_copy = copy.deepcopy(self.model)
            model_copy.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model_copy.score(X_train_scaled, y_train)
            test_score = model_copy.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model_copy, X_train_scaled, y_train, cv=min(5, len(X_train)//2))
            
            # Predictions for error metrics
            y_pred = model_copy.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            training_results[metric] = {
                "train_score": train_score,
                "test_score": test_score,
                "cv_mean": np.mean(cv_scores),
                "cv_std": np.std(cv_scores),
                "mse": mse,
                "mae": mae,
                "feature_importance": self._get_feature_importance(model_copy)
            }
        
        # Update model with best target
        best_metric = max(training_results.keys(), key=lambda m: training_results[m]["test_score"])
        y_best = y_dict[best_metric]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_best)
        
        training_time = time.time() - start_time
        
        training_result = {
            "status": "training_completed",
            "model_type": self.model_type.value,
            "num_samples": len(training_data),
            "num_features": len(self.feature_names),
            "target_metrics": target_metrics,
            "best_metric": best_metric,
            "training_results": training_results,
            "training_time": training_time,
            "model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.training_history.append(training_result)
        
        logger.info(f"Model training completed in {training_time:.2f}s")
        logger.info(f"Best metric: {best_metric} (test score: {training_results[best_metric]['test_score']:.4f})")
        
        return training_result
    
    def _prepare_training_data(
        self, 
        training_data: List[Dict[str, Any]], 
        target_metrics: List[str]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data for model."""
        
        # Extract features
        feature_data = []
        target_data = {metric: [] for metric in target_metrics}
        
        for sample in training_data:
            # Extract features (excluding target metrics)
            features = {}
            for key, value in sample.items():
                if key not in target_metrics and isinstance(value, (int, float)):
                    features[key] = value
            
            if features:
                feature_data.append(features)
                
                # Extract targets
                for metric in target_metrics:
                    target_data[metric].append(sample.get(metric, 0.0))
        
        if not feature_data:
            return np.array([]), {}
        
        # Get consistent feature names
        self.feature_names = sorted(set().union(*[f.keys() for f in feature_data]))
        
        # Convert to arrays
        X = []
        for features in feature_data:
            row = [features.get(name, 0.0) for name in self.feature_names]
            X.append(row)
        
        X = np.array(X)
        y_dict = {metric: np.array(values) for metric, values in target_data.items()}
        
        return X, y_dict
    
    def _get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model."""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return {}
        
        importance_dict = {}
        for i, importance in enumerate(importances):
            if i < len(self.feature_names):
                importance_dict[self.feature_names[i]] = float(importance)
        
        return importance_dict
    
    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """Make predictions based on request."""
        
        start_time = time.time()
        
        # Check cache
        cache_key = f"{hash(str(request.features))}_{request.horizon.value}"
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            # Return if cache is fresh (less than 1 minute old)
            if (datetime.now(timezone.utc) - cached_result.created_at).seconds < 60:
                return cached_result
        
        if not SKLEARN_AVAILABLE or self.model is None:
            # Return mock predictions
            return self._mock_prediction(request, time.time() - start_time)
        
        # Prepare input features
        X = self._prepare_prediction_input(request.features)
        
        if X is None:
            return self._mock_prediction(request, time.time() - start_time)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Calculate uncertainty (simplified)
        uncertainty = self._calculate_uncertainty(X, prediction)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            prediction, uncertainty, request.confidence_required
        )
        
        # Get feature importance for this prediction
        feature_importance = self._get_prediction_feature_importance(X)
        
        prediction_time = time.time() - start_time
        
        result = PredictionResult(
            request_id=request.request_id,
            predictions={"primary_metric": prediction},
            confidence_intervals={"primary_metric": confidence_intervals},
            model_used=self.model_type.value,
            prediction_accuracy=self._estimate_prediction_accuracy(),
            uncertainty=uncertainty,
            feature_importance=feature_importance,
            prediction_time=prediction_time,
            model_version=self.model_version
        )
        
        # Cache result
        self.prediction_cache[cache_key] = result
        
        return result
    
    def _prepare_prediction_input(self, features: List[float]) -> Optional[np.ndarray]:
        """Prepare input features for prediction."""
        
        if len(features) != len(self.feature_names):
            logger.warning(f"Feature dimension mismatch: expected {len(self.feature_names)}, got {len(features)}")
            return None
        
        X = np.array(features).reshape(1, -1)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X
    
    def _calculate_uncertainty(self, X: np.ndarray, prediction: float) -> float:
        """Calculate prediction uncertainty."""
        
        # Simplified uncertainty calculation
        # In practice, would use model-specific uncertainty quantification
        
        if hasattr(self.model, 'predict_proba'):
            # For classifiers with probability
            return 0.1
        elif hasattr(self.model, 'estimators_'):
            # For ensemble methods, use prediction variance
            predictions = np.array([estimator.predict(X)[0] for estimator in self.model.estimators_])
            return np.std(predictions)
        else:
            # Default uncertainty based on training error
            return 0.1 * abs(prediction)
    
    def _calculate_confidence_intervals(
        self, 
        prediction: float, 
        uncertainty: float, 
        confidence: float
    ) -> Tuple[float, float]:
        """Calculate confidence intervals for prediction."""
        
        if SCIPY_AVAILABLE:
            # Use t-distribution for confidence intervals
            alpha = 1 - confidence
            t_score = stats.t.ppf(1 - alpha/2, df=max(1, len(self.training_history)))
            margin = t_score * uncertainty
        else:
            # Simple approximation
            margin = 1.96 * uncertainty  # 95% confidence interval
        
        return (prediction - margin, prediction + margin)
    
    def _get_prediction_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get feature importance for specific prediction."""
        
        # Simplified feature importance for prediction
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {
                name: float(importance) 
                for name, importance in zip(self.feature_names, importances)
            }
        
        return {}
    
    def _estimate_prediction_accuracy(self) -> float:
        """Estimate prediction accuracy based on training history."""
        
        if not self.training_history:
            return 0.5
        
        latest_training = self.training_history[-1]
        training_results = latest_training.get("training_results", {})
        
        if training_results:
            test_scores = [result["test_score"] for result in training_results.values()]
            return np.mean(test_scores)
        
        return 0.5
    
    def _mock_prediction(self, request: PredictionRequest, prediction_time: float) -> PredictionResult:
        """Generate mock prediction when model is not available."""
        
        # Generate mock prediction based on input features
        if request.features:
            mock_prediction = np.mean(request.features) + np.random.normal(0, 0.1)
        else:
            mock_prediction = np.random.uniform(0.5, 0.9)
        
        mock_uncertainty = 0.1
        mock_confidence = (mock_prediction - mock_uncertainty, mock_prediction + mock_uncertainty)
        
        return PredictionResult(
            request_id=request.request_id,
            predictions={"primary_metric": mock_prediction},
            confidence_intervals={"primary_metric": mock_confidence},
            model_used="mock_model",
            prediction_accuracy=0.7,
            uncertainty=mock_uncertainty,
            feature_importance={},
            prediction_time=prediction_time,
            model_version="mock_1.0"
        )


class AdaptiveOptimizer:
    """Adaptive optimization system with predictive capabilities."""
    
    def __init__(self):
        self.predictive_model = PredictiveModel()
        self.time_series = TimeSeries()
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_strategy = OptimizationStrategy.ADAPTIVE_HYBRID
        self.performance_metrics: Dict[str, float] = {}
        self.adaptation_threshold = 0.05
        
    async def optimize_with_prediction(
        self,
        objectives: List[OptimizationObjective],
        parameter_space: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    ) -> Dict[str, Any]:
        """Optimize objectives using predictive guidance."""
        
        logger.info(f"Starting adaptive optimization with {len(objectives)} objectives")
        
        optimization_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize optimization state
        current_parameters = self._initialize_parameters(parameter_space)
        current_values = await self._evaluate_objectives(objectives, current_parameters)
        
        best_parameters = current_parameters.copy()
        best_values = current_values.copy()
        best_score = self._calculate_composite_score(objectives, best_values)
        
        iteration_history = []
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            # Record current state in time series
            timestamp = datetime.now(timezone.utc)
            self.time_series.add_data_point(timestamp, {
                "iteration": iteration,
                "composite_score": self._calculate_composite_score(objectives, current_values),
                **current_values,
                **{f"param_{k}": v for k, v in current_parameters.items()}
            })
            
            # Make predictions for next steps
            predictions = await self._predict_optimization_trajectory(
                objectives, current_parameters, current_values, prediction_horizon
            )
            
            # Adapt optimization strategy based on predictions and performance
            await self._adapt_strategy(iteration, predictions)
            
            # Generate candidate parameters using current strategy
            candidate_parameters = await self._generate_candidates(
                current_parameters, parameter_space, predictions
            )
            
            # Evaluate candidates
            candidate_values = await self._evaluate_objectives(objectives, candidate_parameters)
            candidate_score = self._calculate_composite_score(objectives, candidate_values)
            
            # Update if improvement found
            if candidate_score > best_score:
                best_parameters = candidate_parameters.copy()
                best_values = candidate_values.copy()
                best_score = candidate_score
                current_parameters = candidate_parameters
                current_values = candidate_values
            else:
                # Apply acceptance probability for non-improving moves
                acceptance_prob = self._calculate_acceptance_probability(
                    current_values, candidate_values, iteration, max_iterations
                )
                
                if np.random.random() < acceptance_prob:
                    current_parameters = candidate_parameters
                    current_values = candidate_values
            
            # Record iteration statistics
            iteration_time = time.time() - iteration_start
            iteration_stats = {
                "iteration": iteration,
                "current_score": self._calculate_composite_score(objectives, current_values),
                "best_score": best_score,
                "strategy": self.current_strategy.value,
                "prediction_accuracy": predictions.prediction_accuracy if predictions else 0.0,
                "acceptance_probability": acceptance_prob if 'acceptance_prob' in locals() else 1.0,
                "iteration_time": iteration_time,
                "timestamp": timestamp.isoformat()
            }
            iteration_history.append(iteration_stats)
            
            # Check convergence
            if self._check_convergence(iteration_history):
                logger.info(f"Optimization converged at iteration {iteration}")
                break
            
            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}: Best score = {best_score:.6f}")
        
        total_time = time.time() - start_time
        
        # Train predictive model with optimization data
        if len(iteration_history) > 10:
            await self._update_predictive_model(iteration_history, objectives)
        
        optimization_result = {
            "optimization_id": optimization_id,
            "objectives": [{"name": obj.name, "target": obj.target_value, "weight": obj.weight} for obj in objectives],
            "best_parameters": best_parameters,
            "best_values": best_values,
            "best_score": best_score,
            "total_iterations": len(iteration_history),
            "convergence_iteration": len(iteration_history),
            "final_strategy": self.current_strategy.value,
            "execution_time": total_time,
            "iteration_history": iteration_history,
            "prediction_performance": self._analyze_prediction_performance(iteration_history),
            "optimization_efficiency": best_score / max(1, total_time),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Optimization completed in {total_time:.2f}s, best score: {best_score:.6f}")
        
        return optimization_result
    
    def _initialize_parameters(self, parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Initialize optimization parameters."""
        parameters = {}
        
        for param_name, (low, high) in parameter_space.items():
            parameters[param_name] = np.random.uniform(low, high)
        
        return parameters
    
    async def _evaluate_objectives(
        self, 
        objectives: List[OptimizationObjective], 
        parameters: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate objective functions."""
        
        values = {}
        
        for objective in objectives:
            # Mock objective evaluation based on parameters
            param_values = list(parameters.values())
            
            if objective.name == "stability":
                # Stability objective: prefer moderate parameter values
                value = np.exp(-0.5 * np.sum([(v - 0.5)**2 for v in param_values]))
            elif objective.name == "binding_affinity":
                # Binding affinity: multi-modal with peaks
                value = (0.8 * np.exp(-0.5 * np.sum([(v - 0.3)**2 for v in param_values])) +
                        0.6 * np.exp(-0.5 * np.sum([(v - 0.7)**2 for v in param_values])))
            elif objective.name == "solubility":
                # Solubility: linear combination with noise
                value = 0.5 + 0.3 * np.mean(param_values) + 0.1 * np.random.normal(0, 0.1)
            else:
                # Generic objective
                value = np.random.uniform(0.3, 0.9)
            
            # Apply direction and constraints
            if objective.direction == "minimize":
                value = 1.0 - value  # Convert to maximization
            
            values[objective.name] = np.clip(value, 0.0, 1.0)
        
        # Simulate evaluation time
        await asyncio.sleep(0.01)
        
        return values
    
    def _calculate_composite_score(
        self, 
        objectives: List[OptimizationObjective], 
        values: Dict[str, float]
    ) -> float:
        """Calculate weighted composite score."""
        
        total_score = 0.0
        total_weight = 0.0
        
        for objective in objectives:
            if objective.name in values:
                score = values[objective.name]
                
                # Apply target preference
                if objective.target_value is not None:
                    # Penalty for deviation from target
                    deviation = abs(score - objective.target_value)
                    score = score * (1.0 - deviation * 0.5)
                
                total_score += score * objective.weight
                total_weight += objective.weight
        
        return total_score / max(1e-8, total_weight)
    
    async def _predict_optimization_trajectory(
        self,
        objectives: List[OptimizationObjective],
        current_parameters: Dict[str, float],
        current_values: Dict[str, float],
        horizon: PredictionHorizon
    ) -> Optional[PredictionResult]:
        """Predict optimization trajectory."""
        
        # Prepare features for prediction
        features = []
        features.extend(current_parameters.values())
        features.extend(current_values.values())
        
        # Add time series features
        for objective in objectives:
            series = self.time_series.get_feature_series(objective.name, window=20)
            if series:
                features.append(np.mean(series))
                features.append(np.std(series))
        
        # Create prediction request
        steps_ahead = {
            PredictionHorizon.SHORT_TERM: 3,
            PredictionHorizon.MEDIUM_TERM: 10,
            PredictionHorizon.LONG_TERM: 25,
            PredictionHorizon.ADAPTIVE: 5
        }.get(horizon, 5)
        
        request = PredictionRequest(
            features=features,
            horizon=horizon,
            target_metrics=[obj.name for obj in objectives]
        )
        
        try:
            return await self.predictive_model.predict(request)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None
    
    async def _adapt_strategy(self, iteration: int, predictions: Optional[PredictionResult]):
        """Adapt optimization strategy based on performance and predictions."""
        
        # Analyze recent performance
        if len(self.optimization_history) > 0:
            recent_performance = self._analyze_recent_performance()
        else:
            recent_performance = {"improvement_rate": 0.5}
        
        # Determine strategy based on performance and predictions
        improvement_rate = recent_performance.get("improvement_rate", 0.5)
        
        if predictions and predictions.uncertainty > 0.3:
            # High uncertainty - use more exploratory strategy
            if self.current_strategy != OptimizationStrategy.EVOLUTIONARY:
                self.current_strategy = OptimizationStrategy.EVOLUTIONARY
                logger.debug(f"Switched to evolutionary strategy due to high uncertainty")
        elif improvement_rate < 0.1:
            # Slow improvement - try different strategy
            if self.current_strategy == OptimizationStrategy.GRADIENT_BASED:
                self.current_strategy = OptimizationStrategy.SIMULATED_ANNEALING
            elif self.current_strategy == OptimizationStrategy.SIMULATED_ANNEALING:
                self.current_strategy = OptimizationStrategy.EVOLUTIONARY
            else:
                self.current_strategy = OptimizationStrategy.GRADIENT_BASED
            
            logger.debug(f"Switched to {self.current_strategy.value} due to slow improvement")
        elif improvement_rate > 0.5:
            # Fast improvement - use exploitative strategy
            if self.current_strategy != OptimizationStrategy.GRADIENT_BASED:
                self.current_strategy = OptimizationStrategy.GRADIENT_BASED
                logger.debug(f"Switched to gradient-based strategy due to fast improvement")
    
    def _analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent optimization performance."""
        
        if not self.optimization_history:
            return {"improvement_rate": 0.5}
        
        recent_optimization = self.optimization_history[-1]
        iteration_history = recent_optimization.get("iteration_history", [])
        
        if len(iteration_history) < 10:
            return {"improvement_rate": 0.5}
        
        # Calculate improvement rate over last 10 iterations
        recent_scores = [iteration["best_score"] for iteration in iteration_history[-10:]]
        
        if len(recent_scores) > 1:
            improvement_rate = (recent_scores[-1] - recent_scores[0]) / max(1e-8, recent_scores[0])
        else:
            improvement_rate = 0.0
        
        return {
            "improvement_rate": max(0.0, improvement_rate),
            "score_variance": np.var(recent_scores),
            "convergence_trend": 1.0 if len(set(recent_scores[-5:])) == 1 else 0.0
        }
    
    async def _generate_candidates(
        self,
        current_parameters: Dict[str, float],
        parameter_space: Dict[str, Tuple[float, float]],
        predictions: Optional[PredictionResult]
    ) -> Dict[str, float]:
        """Generate candidate parameters based on current strategy."""
        
        if self.current_strategy == OptimizationStrategy.GRADIENT_BASED:
            return self._gradient_based_candidate(current_parameters, parameter_space, predictions)
        elif self.current_strategy == OptimizationStrategy.EVOLUTIONARY:
            return self._evolutionary_candidate(current_parameters, parameter_space)
        elif self.current_strategy == OptimizationStrategy.SIMULATED_ANNEALING:
            return self._simulated_annealing_candidate(current_parameters, parameter_space)
        else:
            return self._random_candidate(parameter_space)
    
    def _gradient_based_candidate(
        self,
        current_parameters: Dict[str, float],
        parameter_space: Dict[str, Tuple[float, float]],
        predictions: Optional[PredictionResult]
    ) -> Dict[str, float]:
        """Generate candidate using gradient-based approach."""
        
        candidate = current_parameters.copy()
        
        # Use feature importance from predictions as gradient approximation
        if predictions and predictions.feature_importance:
            step_size = 0.1
            
            for param_name in candidate.keys():
                if param_name in predictions.feature_importance:
                    importance = predictions.feature_importance[param_name]
                    gradient = importance * step_size
                    
                    # Update parameter
                    low, high = parameter_space[param_name]
                    candidate[param_name] = np.clip(
                        candidate[param_name] + gradient,
                        low, high
                    )
        else:
            # Random gradient if no predictions available
            for param_name in candidate.keys():
                low, high = parameter_space[param_name]
                gradient = np.random.normal(0, 0.1)
                candidate[param_name] = np.clip(
                    candidate[param_name] + gradient,
                    low, high
                )
        
        return candidate
    
    def _evolutionary_candidate(
        self,
        current_parameters: Dict[str, float],
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """Generate candidate using evolutionary approach."""
        
        candidate = current_parameters.copy()
        
        # Apply mutations
        mutation_rate = 0.3
        mutation_strength = 0.2
        
        for param_name in candidate.keys():
            if np.random.random() < mutation_rate:
                low, high = parameter_space[param_name]
                range_size = high - low
                mutation = np.random.normal(0, mutation_strength * range_size)
                
                candidate[param_name] = np.clip(
                    candidate[param_name] + mutation,
                    low, high
                )
        
        return candidate
    
    def _simulated_annealing_candidate(
        self,
        current_parameters: Dict[str, float],
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """Generate candidate using simulated annealing approach."""
        
        candidate = current_parameters.copy()
        
        # Temperature-based perturbation
        temperature = 0.1  # Would normally decrease over time
        
        for param_name in candidate.keys():
            low, high = parameter_space[param_name]
            range_size = high - low
            perturbation = np.random.normal(0, temperature * range_size)
            
            candidate[param_name] = np.clip(
                candidate[param_name] + perturbation,
                low, high
            )
        
        return candidate
    
    def _random_candidate(self, parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Generate random candidate parameters."""
        
        candidate = {}
        
        for param_name, (low, high) in parameter_space.items():
            candidate[param_name] = np.random.uniform(low, high)
        
        return candidate
    
    def _calculate_acceptance_probability(
        self,
        current_values: Dict[str, float],
        candidate_values: Dict[str, float],
        iteration: int,
        max_iterations: int
    ) -> float:
        """Calculate acceptance probability for non-improving moves."""
        
        # Simple simulated annealing acceptance
        current_sum = sum(current_values.values())
        candidate_sum = sum(candidate_values.values())
        
        if candidate_sum >= current_sum:
            return 1.0
        
        # Temperature decreases over time
        temperature = 1.0 * (1.0 - iteration / max_iterations)
        delta = candidate_sum - current_sum
        
        return np.exp(delta / max(1e-8, temperature))
    
    def _check_convergence(self, iteration_history: List[Dict[str, Any]], patience: int = 20) -> bool:
        """Check if optimization has converged."""
        
        if len(iteration_history) < patience:
            return False
        
        recent_scores = [iteration["best_score"] for iteration in iteration_history[-patience:]]
        
        # Check if improvement is negligible
        score_range = max(recent_scores) - min(recent_scores)
        return score_range < self.adaptation_threshold
    
    async def _update_predictive_model(
        self,
        iteration_history: List[Dict[str, Any]],
        objectives: List[OptimizationObjective]
    ):
        """Update predictive model with optimization data."""
        
        # Prepare training data from iteration history
        training_data = []
        
        for iteration in iteration_history:
            data_point = {
                "iteration": iteration["iteration"],
                "current_score": iteration["current_score"],
                "strategy": hash(iteration["strategy"]) % 100,  # Categorical encoding
                "iteration_time": iteration["iteration_time"]
            }
            
            # Add target metrics
            for objective in objectives:
                data_point[objective.name] = iteration.get(objective.name, 0.5)
            
            training_data.append(data_point)
        
        # Train model
        target_metrics = [obj.name for obj in objectives]
        await self.predictive_model.train(training_data, target_metrics)
        
        logger.debug(f"Updated predictive model with {len(training_data)} samples")
    
    def _analyze_prediction_performance(self, iteration_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze predictive model performance during optimization."""
        
        prediction_accuracies = [
            iteration.get("prediction_accuracy", 0.5) 
            for iteration in iteration_history
        ]
        
        return {
            "mean_accuracy": np.mean(prediction_accuracies),
            "min_accuracy": np.min(prediction_accuracies),
            "max_accuracy": np.max(prediction_accuracies),
            "accuracy_improvement": prediction_accuracies[-1] - prediction_accuracies[0] if len(prediction_accuracies) > 1 else 0.0
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance."""
        
        if not self.optimization_history:
            return {"status": "no_optimizations_performed"}
        
        best_scores = [opt["best_score"] for opt in self.optimization_history]
        execution_times = [opt["execution_time"] for opt in self.optimization_history]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "best_overall_score": max(best_scores),
            "average_score": np.mean(best_scores),
            "score_improvement_trend": best_scores[-1] - best_scores[0] if len(best_scores) > 1 else 0.0,
            "average_execution_time": np.mean(execution_times),
            "optimization_efficiency": max(best_scores) / max(execution_times),
            "strategies_used": list(set([
                iteration["strategy"] 
                for opt in self.optimization_history 
                for iteration in opt.get("iteration_history", [])
            ])),
            "current_strategy": self.current_strategy.value,
            "predictive_model_status": {
                "model_type": self.predictive_model.model_type.value,
                "training_history": len(self.predictive_model.training_history),
                "model_version": self.predictive_model.model_version
            },
            "last_optimization": self.optimization_history[-1]["timestamp"] if self.optimization_history else None
        }


# Global adaptive optimizer instance
adaptive_optimizer = AdaptiveOptimizer()


# Example usage
async def example_predictive_optimization():
    """Example predictive optimization for protein design."""
    
    print("🔮 Predictive Systems & Autonomous Optimization Demo")
    print("=" * 60)
    
    # Define optimization objectives
    objectives = [
        OptimizationObjective("stability", target_value=0.85, weight=0.4, direction="maximize"),
        OptimizationObjective("binding_affinity", target_value=0.8, weight=0.3, direction="maximize"),
        OptimizationObjective("solubility", target_value=0.75, weight=0.2, direction="maximize"),
        OptimizationObjective("novelty", target_value=0.7, weight=0.1, direction="maximize")
    ]
    
    # Define parameter space for protein design
    parameter_space = {
        "hydrophobicity": (0.0, 1.0),
        "charge_distribution": (0.0, 1.0),
        "secondary_structure": (0.0, 1.0),
        "flexibility": (0.0, 1.0),
        "molecular_weight": (0.0, 1.0)
    }
    
    # Run adaptive optimization
    result = await adaptive_optimizer.optimize_with_prediction(
        objectives=objectives,
        parameter_space=parameter_space,
        max_iterations=50,
        prediction_horizon=PredictionHorizon.MEDIUM_TERM
    )
    
    print(f"\n✅ Optimization Results:")
    print(f"   Best Score: {result['best_score']:.6f}")
    print(f"   Total Iterations: {result['total_iterations']}")
    print(f"   Final Strategy: {result['final_strategy']}")
    print(f"   Execution Time: {result['execution_time']:.2f}s")
    print(f"   Optimization Efficiency: {result['optimization_efficiency']:.4f}")
    
    print(f"\n🎯 Best Parameters:")
    for param, value in result["best_parameters"].items():
        print(f"   {param}: {value:.4f}")
    
    print(f"\n📊 Best Objective Values:")
    for objective, value in result["best_values"].items():
        print(f"   {objective}: {value:.4f}")
    
    print(f"\n🔮 Prediction Performance:")
    pred_perf = result["prediction_performance"]
    print(f"   Mean Accuracy: {pred_perf['mean_accuracy']:.4f}")
    print(f"   Max Accuracy: {pred_perf['max_accuracy']:.4f}")
    print(f"   Accuracy Improvement: {pred_perf['accuracy_improvement']:.4f}")
    
    # Show optimization summary
    summary = adaptive_optimizer.get_optimization_summary()
    print(f"\n📈 System Summary:")
    print(f"   Total Optimizations: {summary['total_optimizations']}")
    print(f"   Best Overall Score: {summary['best_overall_score']:.6f}")
    print(f"   Current Strategy: {summary['current_strategy']}")
    print(f"   Strategies Used: {', '.join(summary['strategies_used'])}")
    
    return result


if __name__ == "__main__":
    asyncio.run(example_predictive_optimization())