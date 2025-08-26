"""
Advanced Predictive Monitoring System

Final component of autonomous SDLC featuring:
- Real-time system health monitoring
- Predictive performance analytics
- Anomaly detection and prevention
- Self-healing system responses
- Future state prediction
- Resource optimization forecasting
- Quality trend analysis
- Automated alert systems
"""

import asyncio
import json
import time
import uuid
import logging
import numpy as np
import statistics
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import itertools
from collections import defaultdict, deque
import random
import math
import warnings

logger = logging.getLogger(__name__)


class MonitoringLevel(Enum):
    """Levels of monitoring intensity."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PREDICTIVE = "predictive"
    AUTONOMOUS = "autonomous"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class SystemHealth(Enum):
    """System health states."""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"


@dataclass
class MetricPoint:
    """Single metric measurement point."""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source_metric: str
    threshold_value: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    auto_resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "source_metric": self.source_metric,
            "threshold_value": self.threshold_value,
            "actual_value": self.actual_value,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None,
            "auto_resolved": self.auto_resolved
        }


@dataclass
class Prediction:
    """Future state prediction."""
    prediction_id: str
    metric_name: str
    predicted_value: float
    prediction_time: datetime
    confidence: float
    prediction_horizon: timedelta
    model_used: str
    contributing_factors: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "metric_name": self.metric_name,
            "predicted_value": self.predicted_value,
            "prediction_time": self.prediction_time.isoformat(),
            "confidence": self.confidence,
            "prediction_horizon_seconds": self.prediction_horizon.total_seconds(),
            "model_used": self.model_used,
            "contributing_factors": self.contributing_factors,
            "created_at": self.created_at.isoformat()
        }


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics_buffer = deque(maxlen=10000)
        self.metric_definitions = {}
        self.collection_active = False
        self.collection_task = None
        
    def register_metric(self, name: str, description: str, unit: str, tags: Dict[str, str] = None):
        """Register a metric for collection."""
        self.metric_definitions[name] = {
            "description": description,
            "unit": unit,
            "tags": tags or {},
            "collection_count": 0,
            "last_value": None,
            "min_value": float('inf'),
            "max_value": float('-inf'),
            "sum_values": 0.0,
            "sum_squares": 0.0
        }
        
        logger.debug(f"Registered metric: {name}")
    
    async def start_collection(self):
        """Start metric collection."""
        if self.collection_active:
            logger.warning("Metrics collection already active")
            return
        
        self.collection_active = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop metric collection."""
        self.collection_active = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self.collection_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        current_time = datetime.now(timezone.utc)
        
        # CPU Usage (mock)
        cpu_usage = 20.0 + 30.0 * np.sin(time.time() / 60) + np.random.normal(0, 5)
        cpu_usage = max(0.0, min(100.0, cpu_usage))
        await self.record_metric("cpu_usage_percent", cpu_usage, current_time, {"system": "monitoring"})
        
        # Memory Usage (mock)
        memory_usage = 40.0 + 20.0 * np.sin(time.time() / 120) + np.random.normal(0, 3)
        memory_usage = max(0.0, min(100.0, memory_usage))
        await self.record_metric("memory_usage_percent", memory_usage, current_time, {"system": "monitoring"})
        
        # Request Rate (mock)
        base_rate = 50.0
        time_of_day_factor = 1.0 + 0.5 * np.sin((time.time() % 86400) / 86400 * 2 * np.pi)
        request_rate = base_rate * time_of_day_factor + np.random.normal(0, 5)
        request_rate = max(0.0, request_rate)
        await self.record_metric("requests_per_second", request_rate, current_time, {"endpoint": "protein_generation"})
        
        # Response Time (mock)
        base_response_time = 0.2
        load_factor = 1.0 + (request_rate / 100.0)
        response_time = base_response_time * load_factor + np.random.exponential(0.05)
        await self.record_metric("response_time_seconds", response_time, current_time, {"endpoint": "protein_generation"})
        
        # Error Rate (mock)
        error_rate = 0.01 + 0.02 * max(0, (cpu_usage - 80) / 20) + np.random.uniform(0, 0.005)
        error_rate = max(0.0, min(1.0, error_rate))
        await self.record_metric("error_rate_percent", error_rate * 100, current_time, {"system": "api"})
        
        # Protein Generation Quality (mock)
        quality_trend = 0.8 + 0.1 * np.sin(time.time() / 300)  # Slow trend
        quality_noise = np.random.normal(0, 0.05)
        protein_quality = max(0.0, min(1.0, quality_trend + quality_noise))
        await self.record_metric("protein_generation_quality", protein_quality, current_time, {"model": "quantum_neural"})
        
        # Queue Length (mock)
        queue_length = max(0, int(request_rate / 10 - 3 + np.random.normal(0, 2)))
        await self.record_metric("processing_queue_length", queue_length, current_time, {"queue": "generation"})
        
        # GPU Utilization (mock)
        gpu_util = 60.0 + 25.0 * (request_rate / 100.0) + np.random.normal(0, 8)
        gpu_util = max(0.0, min(100.0, gpu_util))
        await self.record_metric("gpu_utilization_percent", gpu_util, current_time, {"device": "gpu_0"})
        
        # Cache Hit Rate (mock)
        cache_hit_rate = 85.0 + 10.0 * np.random.beta(2, 1) + np.random.normal(0, 2)
        cache_hit_rate = max(0.0, min(100.0, cache_hit_rate))
        await self.record_metric("cache_hit_rate_percent", cache_hit_rate, current_time, {"cache": "protein_cache"})
        
        # Database Connection Pool (mock)
        db_connections = max(1, int(10 + request_rate / 20 + np.random.normal(0, 2)))
        await self.record_metric("database_connections_active", db_connections, current_time, {"database": "protein_db"})
    
    async def record_metric(self, name: str, value: float, timestamp: datetime = None, tags: Dict[str, str] = None):
        """Record a metric value."""
        timestamp = timestamp or datetime.now(timezone.utc)
        tags = tags or {}
        
        metric_point = MetricPoint(
            metric_name=name,
            value=value,
            timestamp=timestamp,
            tags=tags
        )
        
        self.metrics_buffer.append(metric_point)
        
        # Update metric definition statistics
        if name in self.metric_definitions:
            definition = self.metric_definitions[name]
            definition["collection_count"] += 1
            definition["last_value"] = value
            definition["min_value"] = min(definition["min_value"], value)
            definition["max_value"] = max(definition["max_value"], value)
            definition["sum_values"] += value
            definition["sum_squares"] += value * value
    
    def get_recent_metrics(self, metric_name: str, duration: timedelta) -> List[MetricPoint]:
        """Get recent metrics for a specific metric name."""
        cutoff_time = datetime.now(timezone.utc) - duration
        
        return [
            metric for metric in self.metrics_buffer
            if metric.metric_name == metric_name and metric.timestamp >= cutoff_time
        ]
    
    def get_metric_statistics(self, metric_name: str, duration: timedelta = None) -> Dict[str, Any]:
        """Get statistics for a metric."""
        if duration:
            metrics = self.get_recent_metrics(metric_name, duration)
            values = [m.value for m in metrics]
        else:
            # Use stored statistics
            if metric_name not in self.metric_definitions:
                return {}
            
            definition = self.metric_definitions[metric_name]
            n = definition["collection_count"]
            
            if n == 0:
                return {}
            
            return {
                "count": n,
                "min": definition["min_value"],
                "max": definition["max_value"],
                "mean": definition["sum_values"] / n,
                "std": math.sqrt(max(0, (definition["sum_squares"] / n) - (definition["sum_values"] / n) ** 2)),
                "last_value": definition["last_value"]
            }
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "last_value": values[-1] if values else None
        }


class AnomalyDetector:
    """Detects anomalies in system metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.metric_models = {}
        self.anomaly_history = deque(maxlen=1000)
        
    def train_model(self, metric_name: str, historical_data: List[MetricPoint]):
        """Train anomaly detection model for a metric."""
        if len(historical_data) < 10:
            logger.warning(f"Insufficient data to train model for {metric_name}")
            return
        
        values = [point.value for point in historical_data]
        
        # Simple statistical model
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Time-based patterns
        timestamps = [point.timestamp for point in historical_data]
        time_diffs = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Detect periodic patterns
        periodic_score = self._detect_periodicity(values)
        
        self.metric_models[metric_name] = {
            "mean": mean_value,
            "std": std_value,
            "min_expected": mean_value - 3 * std_value,
            "max_expected": mean_value + 3 * std_value,
            "periodic_score": periodic_score,
            "training_samples": len(values),
            "last_trained": datetime.now(timezone.utc)
        }
        
        logger.info(f"Trained anomaly model for {metric_name}")
    
    def _detect_periodicity(self, values: List[float]) -> float:
        """Detect periodic patterns in values."""
        if len(values) < 20:
            return 0.0
        
        # Simple autocorrelation-based periodicity detection
        n = len(values)
        mean_val = statistics.mean(values)
        centered_values = [v - mean_val for v in values]
        
        max_correlation = 0.0
        
        for lag in range(2, min(n // 4, 50)):
            correlation = 0.0
            count = 0
            
            for i in range(n - lag):
                correlation += centered_values[i] * centered_values[i + lag]
                count += 1
            
            if count > 0:
                correlation = abs(correlation / count)
                max_correlation = max(max_correlation, correlation)
        
        return max_correlation / (statistics.stdev(values) ** 2 + 0.001)
    
    def detect_anomaly(self, metric_point: MetricPoint) -> Optional[Dict[str, Any]]:
        """Detect if a metric point is anomalous."""
        metric_name = metric_point.metric_name
        
        if metric_name not in self.metric_models:
            return None
        
        model = self.metric_models[metric_name]
        value = metric_point.value
        
        # Statistical anomaly detection
        z_score = abs((value - model["mean"]) / (model["std"] + 0.001))
        is_statistical_anomaly = z_score > self.sensitivity
        
        # Range-based detection
        is_range_anomaly = value < model["min_expected"] or value > model["max_expected"]
        
        # Rate of change detection
        rate_anomaly_score = 0.0
        if len(self.anomaly_history) > 0:
            last_point = self.anomaly_history[-1]
            if last_point.get("metric_name") == metric_name:
                time_diff = (metric_point.timestamp - last_point["timestamp"]).total_seconds()
                if time_diff > 0:
                    rate_of_change = abs(value - last_point["value"]) / time_diff
                    expected_rate = model["std"] / 60.0  # Expected change per second
                    rate_anomaly_score = rate_of_change / (expected_rate + 0.001)
        
        is_rate_anomaly = rate_anomaly_score > self.sensitivity
        
        # Overall anomaly score
        anomaly_score = max(z_score / self.sensitivity, rate_anomaly_score / self.sensitivity)
        is_anomaly = is_statistical_anomaly or is_range_anomaly or is_rate_anomaly
        
        if is_anomaly:
            anomaly_info = {
                "metric_name": metric_name,
                "value": value,
                "timestamp": metric_point.timestamp,
                "z_score": z_score,
                "rate_anomaly_score": rate_anomaly_score,
                "anomaly_score": anomaly_score,
                "expected_range": [model["min_expected"], model["max_expected"]],
                "model_mean": model["mean"],
                "model_std": model["std"]
            }
            
            self.anomaly_history.append(anomaly_info)
            return anomaly_info
        
        return None


class PredictiveAnalytics:
    """Provides predictive analytics for system metrics."""
    
    def __init__(self, prediction_horizons: List[timedelta] = None):
        self.prediction_horizons = prediction_horizons or [
            timedelta(minutes=5),
            timedelta(minutes=15),
            timedelta(hours=1),
            timedelta(hours=4)
        ]
        self.prediction_models = {}
        self.prediction_history = deque(maxlen=1000)
        
    def train_prediction_model(self, metric_name: str, historical_data: List[MetricPoint]):
        """Train prediction model for a metric."""
        if len(historical_data) < 50:
            logger.warning(f"Insufficient data for prediction model: {metric_name}")
            return
        
        # Sort by timestamp
        historical_data.sort(key=lambda x: x.timestamp)
        
        values = [point.value for point in historical_data]
        timestamps = [point.timestamp for point in historical_data]
        
        # Convert timestamps to seconds since first timestamp
        time_series = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Simple linear trend model
        trend_model = self._fit_trend_model(time_series, values)
        
        # Seasonal model
        seasonal_model = self._fit_seasonal_model(time_series, values)
        
        # Moving average model
        ma_model = self._fit_moving_average_model(values)
        
        self.prediction_models[metric_name] = {
            "trend_model": trend_model,
            "seasonal_model": seasonal_model,
            "moving_average_model": ma_model,
            "last_timestamp": timestamps[-1],
            "last_value": values[-1],
            "training_samples": len(values),
            "model_accuracy": self._calculate_model_accuracy(time_series, values, trend_model, seasonal_model)
        }
        
        logger.info(f"Trained prediction model for {metric_name}")
    
    def _fit_trend_model(self, time_series: List[float], values: List[float]) -> Dict[str, float]:
        """Fit linear trend model."""
        n = len(values)
        
        if n < 2:
            return {"slope": 0.0, "intercept": values[0] if values else 0.0}
        
        # Simple linear regression
        x_mean = statistics.mean(time_series)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(time_series, values))
        denominator = sum((x - x_mean) ** 2 for x in time_series)
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        return {"slope": slope, "intercept": intercept}
    
    def _fit_seasonal_model(self, time_series: List[float], values: List[float]) -> Dict[str, Any]:
        """Fit seasonal model."""
        if len(values) < 20:
            return {"amplitude": 0.0, "period": 3600.0, "phase": 0.0}
        
        # Detect dominant frequency using simple peak detection
        # This is a simplified approach - in practice, you'd use FFT
        
        # Try different periods
        periods = [60, 300, 900, 3600, 7200]  # 1min, 5min, 15min, 1hr, 2hr
        best_period = 3600.0
        best_amplitude = 0.0
        best_phase = 0.0
        best_correlation = 0.0
        
        for period in periods:
            if period > time_series[-1] - time_series[0]:
                continue
            
            # Fit sine wave
            amplitudes = []
            phases = []
            correlations = []
            
            for phase_offset in np.linspace(0, 2 * np.pi, 8):
                sine_values = [np.sin(2 * np.pi * t / period + phase_offset) for t in time_series]
                
                # Calculate correlation
                correlation = np.corrcoef(values, sine_values)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
                    
                    # Estimate amplitude
                    amplitude = np.std(values) * abs(correlation)
                    amplitudes.append(amplitude)
                    phases.append(phase_offset)
            
            if correlations:
                max_corr_idx = np.argmax(correlations)
                if correlations[max_corr_idx] > best_correlation:
                    best_correlation = correlations[max_corr_idx]
                    best_period = period
                    best_amplitude = amplitudes[max_corr_idx]
                    best_phase = phases[max_corr_idx]
        
        return {
            "amplitude": best_amplitude,
            "period": best_period,
            "phase": best_phase,
            "correlation": best_correlation
        }
    
    def _fit_moving_average_model(self, values: List[float]) -> Dict[str, Any]:
        """Fit moving average model."""
        window_sizes = [5, 10, 20]
        best_window = 10
        best_error = float('inf')
        
        for window in window_sizes:
            if window >= len(values):
                continue
            
            errors = []
            for i in range(window, len(values)):
                ma_value = statistics.mean(values[i-window:i])
                error = abs(values[i] - ma_value)
                errors.append(error)
            
            if errors:
                avg_error = statistics.mean(errors)
                if avg_error < best_error:
                    best_error = avg_error
                    best_window = window
        
        # Calculate recent moving average
        recent_values = values[-best_window:] if len(values) >= best_window else values
        recent_ma = statistics.mean(recent_values) if recent_values else 0.0
        
        return {
            "window_size": best_window,
            "recent_average": recent_ma,
            "average_error": best_error
        }
    
    def _calculate_model_accuracy(self, time_series: List[float], values: List[float], 
                                  trend_model: Dict[str, float], seasonal_model: Dict[str, Any]) -> float:
        """Calculate model accuracy on historical data."""
        if len(values) < 10:
            return 0.5
        
        predictions = []
        actuals = []
        
        # Test on last 20% of data
        test_start_idx = int(len(values) * 0.8)
        
        for i in range(test_start_idx, len(values)):
            t = time_series[i]
            
            # Trend component
            trend_pred = trend_model["slope"] * t + trend_model["intercept"]
            
            # Seasonal component
            seasonal_pred = seasonal_model["amplitude"] * np.sin(
                2 * np.pi * t / seasonal_model["period"] + seasonal_model["phase"]
            )
            
            combined_pred = trend_pred + seasonal_pred
            predictions.append(combined_pred)
            actuals.append(values[i])
        
        if not predictions:
            return 0.5
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = statistics.mean([
            abs((actual - pred) / (actual + 0.001)) 
            for actual, pred in zip(actuals, predictions)
        ])
        
        # Convert to accuracy (0-1 scale)
        accuracy = max(0.0, 1.0 - mape)
        return accuracy
    
    def predict_metric(self, metric_name: str, prediction_time: datetime, 
                      horizon: timedelta) -> Optional[Prediction]:
        """Predict metric value at future time."""
        if metric_name not in self.prediction_models:
            return None
        
        model = self.prediction_models[metric_name]
        
        # Calculate time difference from last known point
        time_delta = (prediction_time - model["last_timestamp"]).total_seconds()
        
        # Trend prediction
        trend_pred = model["trend_model"]["slope"] * time_delta + model["last_value"]
        
        # Seasonal prediction
        seasonal_model = model["seasonal_model"]
        seasonal_pred = seasonal_model["amplitude"] * np.sin(
            2 * np.pi * time_delta / seasonal_model["period"] + seasonal_model["phase"]
        )
        
        # Moving average influence (for short-term stability)
        ma_influence = 0.3 if horizon.total_seconds() < 3600 else 0.1  # Higher influence for short-term
        ma_pred = model["moving_average_model"]["recent_average"]
        
        # Combined prediction
        combined_pred = (
            trend_pred * 0.4 + 
            (trend_pred + seasonal_pred) * 0.4 + 
            ma_pred * ma_influence +
            model["last_value"] * (0.2 - ma_influence)
        )
        
        # Confidence calculation
        model_accuracy = model["model_accuracy"]
        time_decay = max(0.1, 1.0 - (horizon.total_seconds() / 86400))  # Decay over 24 hours
        confidence = model_accuracy * time_decay
        
        # Contributing factors
        factors = []
        if abs(model["trend_model"]["slope"]) > 0.001:
            factors.append("trend")
        if seasonal_model["correlation"] > 0.1:
            factors.append("seasonal_pattern")
        if ma_influence > 0.1:
            factors.append("recent_average")
        
        prediction = Prediction(
            prediction_id=str(uuid.uuid4()),
            metric_name=metric_name,
            predicted_value=combined_pred,
            prediction_time=prediction_time,
            confidence=confidence,
            prediction_horizon=horizon,
            model_used="trend_seasonal_ma_hybrid",
            contributing_factors=factors
        )
        
        self.prediction_history.append(prediction)
        return prediction
    
    def get_predictions(self, metric_name: str, base_time: datetime = None) -> List[Prediction]:
        """Get predictions for all configured horizons."""
        base_time = base_time or datetime.now(timezone.utc)
        predictions = []
        
        for horizon in self.prediction_horizons:
            prediction_time = base_time + horizon
            prediction = self.predict_metric(metric_name, prediction_time, horizon)
            
            if prediction:
                predictions.append(prediction)
        
        return predictions


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_rules = {}
        self.suppression_rules = {}
        
    def register_alert_rule(self, rule_id: str, metric_name: str, condition: str, 
                           threshold: float, severity: AlertSeverity, 
                           description: str = "", cooldown_minutes: int = 5):
        """Register an alert rule."""
        self.alert_rules[rule_id] = {
            "metric_name": metric_name,
            "condition": condition,  # "greater_than", "less_than", "equals", "anomaly"
            "threshold": threshold,
            "severity": severity,
            "description": description,
            "cooldown_minutes": cooldown_minutes,
            "last_triggered": None
        }
        
        logger.info(f"Registered alert rule: {rule_id}")
    
    def check_alert_conditions(self, metric_point: MetricPoint, anomaly_info: Dict[str, Any] = None) -> List[Alert]:
        """Check if metric triggers any alert conditions."""
        triggered_alerts = []
        current_time = datetime.now(timezone.utc)
        
        for rule_id, rule in self.alert_rules.items():
            if rule["metric_name"] != metric_point.metric_name:
                continue
            
            # Check cooldown
            if rule["last_triggered"]:
                cooldown_delta = timedelta(minutes=rule["cooldown_minutes"])
                if current_time - rule["last_triggered"] < cooldown_delta:
                    continue
            
            # Check condition
            triggered = False
            
            if rule["condition"] == "greater_than":
                triggered = metric_point.value > rule["threshold"]
            elif rule["condition"] == "less_than":
                triggered = metric_point.value < rule["threshold"]
            elif rule["condition"] == "equals":
                triggered = abs(metric_point.value - rule["threshold"]) < 0.001
            elif rule["condition"] == "anomaly":
                triggered = anomaly_info is not None and anomaly_info.get("anomaly_score", 0) > rule["threshold"]
            
            if triggered:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    title=f"Alert: {rule['metric_name']} {rule['condition']} {rule['threshold']}",
                    description=rule["description"] or f"Metric {rule['metric_name']} triggered condition",
                    severity=rule["severity"],
                    source_metric=metric_point.metric_name,
                    threshold_value=rule["threshold"],
                    actual_value=metric_point.value,
                    timestamp=current_time
                )
                
                triggered_alerts.append(alert)
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                rule["last_triggered"] = current_time
                
                logger.warning(f"Alert triggered: {alert.title}")
        
        return triggered_alerts
    
    def resolve_alert(self, alert_id: str, auto_resolved: bool = False) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_timestamp = datetime.now(timezone.utc)
            alert.auto_resolved = auto_resolved
            
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.title} ({'auto' if auto_resolved else 'manual'})")
            return True
        
        return False
    
    def get_active_alerts(self, severity_filter: AlertSeverity = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_alerts = list(self.active_alerts.values())
        
        severity_counts = {severity: 0 for severity in AlertSeverity}
        for alert in active_alerts:
            severity_counts[alert.severity] += 1
        
        recent_history = [alert for alert in self.alert_history 
                         if alert.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)]
        
        return {
            "active_alerts_count": len(active_alerts),
            "severity_distribution": {sev.value: count for sev, count in severity_counts.items()},
            "alerts_last_24h": len(recent_history),
            "most_recent_alert": max(active_alerts, key=lambda x: x.timestamp).timestamp.isoformat() if active_alerts else None,
            "alert_rules_count": len(self.alert_rules)
        }


class SelfHealingSystem:
    """Implements self-healing responses to system issues."""
    
    def __init__(self):
        self.healing_actions = {}
        self.healing_history = deque(maxlen=500)
        self.healing_enabled = True
        
    def register_healing_action(self, trigger_condition: str, action_name: str, 
                               action_function: Callable, cooldown_minutes: int = 15):
        """Register a self-healing action."""
        self.healing_actions[trigger_condition] = {
            "action_name": action_name,
            "action_function": action_function,
            "cooldown_minutes": cooldown_minutes,
            "last_executed": None,
            "execution_count": 0,
            "success_count": 0
        }
        
        logger.info(f"Registered healing action: {action_name} for {trigger_condition}")
    
    async def process_alert(self, alert: Alert) -> Optional[Dict[str, Any]]:
        """Process alert and trigger healing actions if applicable."""
        if not self.healing_enabled:
            return None
        
        trigger_key = f"{alert.source_metric}_{alert.severity.value}"
        
        if trigger_key in self.healing_actions:
            action_info = self.healing_actions[trigger_key]
            
            # Check cooldown
            current_time = datetime.now(timezone.utc)
            if action_info["last_executed"]:
                cooldown_delta = timedelta(minutes=action_info["cooldown_minutes"])
                if current_time - action_info["last_executed"] < cooldown_delta:
                    return None
            
            # Execute healing action
            healing_result = await self._execute_healing_action(alert, action_info)
            
            self.healing_history.append({
                "alert_id": alert.alert_id,
                "trigger": trigger_key,
                "action_name": action_info["action_name"],
                "result": healing_result,
                "timestamp": current_time
            })
            
            return healing_result
        
        return None
    
    async def _execute_healing_action(self, alert: Alert, action_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a healing action."""
        action_info["execution_count"] += 1
        action_info["last_executed"] = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Executing healing action: {action_info['action_name']}")
            
            # Execute the action function
            result = await action_info["action_function"](alert)
            
            action_info["success_count"] += 1
            
            return {
                "success": True,
                "action": action_info["action_name"],
                "result": result,
                "execution_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Healing action failed: {action_info['action_name']}: {e}")
            
            return {
                "success": False,
                "action": action_info["action_name"],
                "error": str(e),
                "execution_time": datetime.now(timezone.utc).isoformat()
            }
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get statistics about healing actions."""
        total_executions = sum(action["execution_count"] for action in self.healing_actions.values())
        total_successes = sum(action["success_count"] for action in self.healing_actions.values())
        
        success_rate = (total_successes / total_executions) if total_executions > 0 else 0.0
        
        recent_actions = [
            action for action in self.healing_history
            if datetime.fromisoformat(action["timestamp"].replace('Z', '+00:00')) > 
               datetime.now(timezone.utc) - timedelta(hours=24)
        ]
        
        return {
            "healing_enabled": self.healing_enabled,
            "registered_actions": len(self.healing_actions),
            "total_executions": total_executions,
            "total_successes": total_successes,
            "success_rate": success_rate,
            "recent_actions_24h": len(recent_actions),
            "action_details": {
                trigger: {
                    "name": info["action_name"],
                    "executions": info["execution_count"],
                    "successes": info["success_count"],
                    "success_rate": info["success_count"] / max(1, info["execution_count"])
                }
                for trigger, info in self.healing_actions.items()
            }
        }


class AdvancedMonitoringSystem:
    """Main monitoring system orchestrating all components."""
    
    def __init__(self, monitoring_level: MonitoringLevel = MonitoringLevel.COMPREHENSIVE):
        self.monitoring_level = monitoring_level
        self.metrics_collector = MetricsCollector(collection_interval=1.0)
        self.anomaly_detector = AnomalyDetector(sensitivity=2.0)
        self.predictive_analytics = PredictiveAnalytics()
        self.alert_manager = AlertManager()
        self.self_healing = SelfHealingSystem()
        
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Setup default metrics
        self._register_default_metrics()
        
        # Setup default alert rules
        self._register_default_alert_rules()
        
        # Setup default healing actions
        self._register_default_healing_actions()
    
    def _register_default_metrics(self):
        """Register default system metrics."""
        metrics = [
            ("cpu_usage_percent", "CPU utilization percentage", "%"),
            ("memory_usage_percent", "Memory utilization percentage", "%"),
            ("requests_per_second", "Request rate", "req/s"),
            ("response_time_seconds", "Response time", "seconds"),
            ("error_rate_percent", "Error rate percentage", "%"),
            ("protein_generation_quality", "Quality of generated proteins", "score"),
            ("processing_queue_length", "Processing queue length", "items"),
            ("gpu_utilization_percent", "GPU utilization percentage", "%"),
            ("cache_hit_rate_percent", "Cache hit rate percentage", "%"),
            ("database_connections_active", "Active database connections", "connections")
        ]
        
        for name, description, unit in metrics:
            self.metrics_collector.register_metric(name, description, unit)
    
    def _register_default_alert_rules(self):
        """Register default alert rules."""
        rules = [
            ("high_cpu", "cpu_usage_percent", "greater_than", 90.0, AlertSeverity.ERROR),
            ("critical_cpu", "cpu_usage_percent", "greater_than", 95.0, AlertSeverity.CRITICAL),
            ("high_memory", "memory_usage_percent", "greater_than", 85.0, AlertSeverity.WARNING),
            ("critical_memory", "memory_usage_percent", "greater_than", 95.0, AlertSeverity.CRITICAL),
            ("high_error_rate", "error_rate_percent", "greater_than", 5.0, AlertSeverity.WARNING),
            ("critical_error_rate", "error_rate_percent", "greater_than", 15.0, AlertSeverity.CRITICAL),
            ("slow_response", "response_time_seconds", "greater_than", 2.0, AlertSeverity.WARNING),
            ("very_slow_response", "response_time_seconds", "greater_than", 5.0, AlertSeverity.ERROR),
            ("low_quality", "protein_generation_quality", "less_than", 0.5, AlertSeverity.WARNING),
            ("very_low_quality", "protein_generation_quality", "less_than", 0.3, AlertSeverity.ERROR),
            ("large_queue", "processing_queue_length", "greater_than", 50, AlertSeverity.WARNING),
            ("critical_queue", "processing_queue_length", "greater_than", 100, AlertSeverity.CRITICAL)
        ]
        
        for rule_id, metric, condition, threshold, severity in rules:
            self.alert_manager.register_alert_rule(
                rule_id, metric, condition, threshold, severity,
                f"Alert for {metric} {condition} {threshold}"
            )
    
    def _register_default_healing_actions(self):
        """Register default healing actions."""
        
        async def restart_service_action(alert: Alert) -> str:
            """Mock service restart action."""
            await asyncio.sleep(2)  # Simulate restart time
            return f"Service restarted due to {alert.source_metric} alert"
        
        async def scale_resources_action(alert: Alert) -> str:
            """Mock resource scaling action."""
            await asyncio.sleep(1)  # Simulate scaling time
            return f"Resources scaled up due to {alert.source_metric} alert"
        
        async def clear_cache_action(alert: Alert) -> str:
            """Mock cache clearing action."""
            await asyncio.sleep(0.5)  # Simulate cache clear time
            return f"Cache cleared due to {alert.source_metric} alert"
        
        async def reduce_load_action(alert: Alert) -> str:
            """Mock load reduction action."""
            await asyncio.sleep(1)  # Simulate load reduction time
            return f"Load reduced due to {alert.source_metric} alert"
        
        healing_actions = [
            ("cpu_usage_percent_critical", "restart_overloaded_service", restart_service_action),
            ("memory_usage_percent_critical", "scale_memory_resources", scale_resources_action),
            ("error_rate_percent_critical", "restart_error_prone_service", restart_service_action),
            ("response_time_seconds_error", "clear_performance_cache", clear_cache_action),
            ("processing_queue_length_critical", "reduce_incoming_load", reduce_load_action)
        ]
        
        for trigger, action_name, action_func in healing_actions:
            self.self_healing.register_healing_action(trigger, action_name, action_func)
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        logger.info(f"Starting monitoring system (level: {self.monitoring_level.value})")
        
        self.monitoring_active = True
        
        # Start metrics collection
        await self.metrics_collector.start_collection()
        
        # Start main monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Monitoring system started successfully")
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        logger.info("Stopping monitoring system")
        
        self.monitoring_active = False
        
        # Stop metrics collection
        await self.metrics_collector.stop_collection()
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        training_interval = timedelta(minutes=10)
        last_training = datetime.now(timezone.utc) - training_interval
        
        while self.monitoring_active:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Process recent metrics
                await self._process_recent_metrics()
                
                # Retrain models periodically
                if current_time - last_training >= training_interval:
                    await self._retrain_models()
                    last_training = current_time
                
                # Generate predictions
                if self.monitoring_level in [MonitoringLevel.PREDICTIVE, MonitoringLevel.AUTONOMOUS]:
                    await self._generate_predictions()
                
                # Auto-resolve alerts that are no longer valid
                await self._auto_resolve_alerts()
                
                await asyncio.sleep(5.0)  # Main loop interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_recent_metrics(self):
        """Process recently collected metrics."""
        recent_window = timedelta(minutes=2)
        
        for metric_name in self.metrics_collector.metric_definitions:
            recent_metrics = self.metrics_collector.get_recent_metrics(metric_name, recent_window)
            
            for metric_point in recent_metrics[-5:]:  # Process last 5 points
                # Anomaly detection
                anomaly_info = self.anomaly_detector.detect_anomaly(metric_point)
                
                # Check alert conditions
                triggered_alerts = self.alert_manager.check_alert_conditions(metric_point, anomaly_info)
                
                # Process alerts with self-healing
                for alert in triggered_alerts:
                    if self.monitoring_level == MonitoringLevel.AUTONOMOUS:
                        healing_result = await self.self_healing.process_alert(alert)
                        if healing_result and healing_result["success"]:
                            # Auto-resolve alert if healing was successful
                            self.alert_manager.resolve_alert(alert.alert_id, auto_resolved=True)
    
    async def _retrain_models(self):
        """Retrain anomaly detection and prediction models."""
        logger.debug("Retraining models...")
        
        training_window = timedelta(hours=2)
        
        for metric_name in self.metrics_collector.metric_definitions:
            historical_data = self.metrics_collector.get_recent_metrics(metric_name, training_window)
            
            if len(historical_data) >= 50:
                # Retrain anomaly detection
                self.anomaly_detector.train_model(metric_name, historical_data)
                
                # Retrain prediction model
                if self.monitoring_level in [MonitoringLevel.PREDICTIVE, MonitoringLevel.AUTONOMOUS]:
                    self.predictive_analytics.train_prediction_model(metric_name, historical_data)
        
        logger.debug("Model retraining completed")
    
    async def _generate_predictions(self):
        """Generate predictions for all metrics."""
        for metric_name in self.metrics_collector.metric_definitions:
            predictions = self.predictive_analytics.get_predictions(metric_name)
            
            # Check predictions for potential issues
            for prediction in predictions:
                if prediction.confidence > 0.7:  # High confidence predictions
                    await self._evaluate_prediction_alerts(prediction)
    
    async def _evaluate_prediction_alerts(self, prediction: Prediction):
        """Evaluate if prediction warrants a proactive alert."""
        metric_name = prediction.metric_name
        predicted_value = prediction.predicted_value
        
        # Check if predicted value would trigger existing alert rules
        mock_metric = MetricPoint(
            metric_name=metric_name,
            value=predicted_value,
            timestamp=prediction.prediction_time
        )
        
        # This would trigger alerts based on predictions
        potential_alerts = self.alert_manager.check_alert_conditions(mock_metric)
        
        if potential_alerts and prediction.confidence > 0.8:
            # Create predictive alert
            predictive_alert = Alert(
                alert_id=str(uuid.uuid4()),
                title=f"Predictive Alert: {metric_name} expected to reach {predicted_value:.2f}",
                description=f"Model predicts {metric_name} will reach {predicted_value:.2f} at {prediction.prediction_time} (confidence: {prediction.confidence:.1%})",
                severity=AlertSeverity.INFO,
                source_metric=metric_name,
                threshold_value=predicted_value,
                actual_value=predicted_value,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.alert_manager.active_alerts[predictive_alert.alert_id] = predictive_alert
            logger.info(f"Generated predictive alert: {predictive_alert.title}")
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts that are no longer valid."""
        resolution_window = timedelta(minutes=5)
        current_time = datetime.now(timezone.utc)
        
        alerts_to_resolve = []
        
        for alert_id, alert in self.alert_manager.active_alerts.items():
            # Check if alert condition is no longer met
            if current_time - alert.timestamp > resolution_window:
                recent_metrics = self.metrics_collector.get_recent_metrics(
                    alert.source_metric, 
                    timedelta(minutes=2)
                )
                
                if recent_metrics:
                    recent_values = [m.value for m in recent_metrics[-5:]]  # Last 5 values
                    avg_recent_value = statistics.mean(recent_values)
                    
                    # Check if condition is resolved
                    condition_resolved = False
                    
                    if "greater_than" in alert.title and avg_recent_value < alert.threshold_value * 0.9:
                        condition_resolved = True
                    elif "less_than" in alert.title and avg_recent_value > alert.threshold_value * 1.1:
                        condition_resolved = True
                    
                    if condition_resolved:
                        alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.alert_manager.resolve_alert(alert_id, auto_resolved=True)
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health status."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Count alerts by severity
        critical_count = len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
        error_count = len([a for a in active_alerts if a.severity == AlertSeverity.ERROR])
        warning_count = len([a for a in active_alerts if a.severity == AlertSeverity.WARNING])
        
        # Determine health based on alerts
        if critical_count > 0:
            return SystemHealth.FAILING
        elif error_count > 2:
            return SystemHealth.CRITICAL
        elif error_count > 0 or warning_count > 5:
            return SystemHealth.DEGRADED
        elif warning_count > 0:
            return SystemHealth.GOOD
        else:
            return SystemHealth.OPTIMAL
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        current_time = datetime.now(timezone.utc)
        
        # Recent metrics summary
        metrics_summary = {}
        for metric_name in self.metrics_collector.metric_definitions:
            recent_stats = self.metrics_collector.get_metric_statistics(
                metric_name, timedelta(minutes=10)
            )
            metrics_summary[metric_name] = recent_stats
        
        # Predictions summary
        predictions_summary = {}
        if self.monitoring_level in [MonitoringLevel.PREDICTIVE, MonitoringLevel.AUTONOMOUS]:
            for metric_name in list(self.metrics_collector.metric_definitions.keys())[:3]:  # Top 3 metrics
                predictions = self.predictive_analytics.get_predictions(metric_name)
                if predictions:
                    predictions_summary[metric_name] = [pred.to_dict() for pred in predictions]
        
        return {
            "system_health": self.get_system_health().value,
            "monitoring_level": self.monitoring_level.value,
            "monitoring_active": self.monitoring_active,
            "current_time": current_time.isoformat(),
            "metrics_summary": metrics_summary,
            "alert_summary": self.alert_manager.get_alert_summary(),
            "healing_statistics": self.self_healing.get_healing_statistics(),
            "predictions_summary": predictions_summary,
            "anomaly_detection_status": {
                "trained_models": len(self.anomaly_detector.metric_models),
                "recent_anomalies": len([
                    a for a in self.anomaly_detector.anomaly_history
                    if a["timestamp"] > current_time - timedelta(hours=1)
                ])
            }
        }


# Global monitoring system instance
advanced_monitoring = None


async def run_monitoring_example():
    """Example of advanced predictive monitoring."""
    
    print("📡 Advanced Predictive Monitoring System Demo")
    print("=" * 60)
    
    # Create monitoring system
    monitoring = AdvancedMonitoringSystem(MonitoringLevel.AUTONOMOUS)
    
    print(f"\n🎯 Starting monitoring system...")
    await monitoring.start_monitoring()
    
    print(f"✅ Monitoring active with {len(monitoring.metrics_collector.metric_definitions)} metrics")
    
    # Let it run for a bit to collect data
    print(f"\n📊 Collecting metrics for 30 seconds...")
    await asyncio.sleep(30)
    
    # Get dashboard data
    dashboard = monitoring.get_monitoring_dashboard()
    
    print(f"\n🏥 System Health: {dashboard['system_health'].upper()}")
    print(f"📈 Active Alerts: {dashboard['alert_summary']['active_alerts_count']}")
    print(f"🔧 Healing Actions Available: {dashboard['healing_statistics']['registered_actions']}")
    
    # Show some metrics
    print(f"\n📊 Recent Metrics (last 10 minutes):")
    for metric_name, stats in list(dashboard['metrics_summary'].items())[:5]:
        if stats:
            print(f"   {metric_name}: {stats.get('mean', 0):.2f} ± {stats.get('std', 0):.2f}")
    
    # Show predictions if available
    if dashboard['predictions_summary']:
        print(f"\n🔮 Predictions Available:")
        for metric_name, predictions in list(dashboard['predictions_summary'].items())[:3]:
            print(f"   {metric_name}: {len(predictions)} predictions")
    
    # Show alerts
    active_alerts = monitoring.alert_manager.get_active_alerts()
    if active_alerts:
        print(f"\n🚨 Active Alerts:")
        for alert in active_alerts[:3]:
            print(f"   {alert.severity.value.upper()}: {alert.title}")
    
    print(f"\n🎉 Monitoring demonstration running successfully!")
    print(f"   System will continue monitoring and auto-healing...")
    
    # Let it run a bit more to show auto-healing
    await asyncio.sleep(15)
    
    # Final status
    final_dashboard = monitoring.get_monitoring_dashboard()
    healing_stats = final_dashboard['healing_statistics']
    
    print(f"\n🔧 Self-Healing Statistics:")
    print(f"   Total Executions: {healing_stats['total_executions']}")
    print(f"   Success Rate: {healing_stats['success_rate']:.1%}")
    print(f"   Recent Actions (24h): {healing_stats['recent_actions_24h']}")
    
    # Cleanup
    await monitoring.stop_monitoring()
    print(f"\n✅ Monitoring system stopped gracefully")
    
    return dashboard, final_dashboard


if __name__ == "__main__":
    # Run monitoring example
    results = asyncio.run(run_monitoring_example())