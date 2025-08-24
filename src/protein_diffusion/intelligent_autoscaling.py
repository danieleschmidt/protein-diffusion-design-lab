"""
Intelligent Auto-Scaling System for Protein Diffusion Design Lab

This module provides predictive auto-scaling, intelligent resource provisioning,
and adaptive capacity management for dynamic workload scaling.
"""

import time
import asyncio
import logging
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import math

# Mock imports for environments without full dependencies
try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    stats = None
    SCIPY_AVAILABLE = False
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Horizontal scaling
    SCALE_IN = "scale_in"    # Reduce horizontal resources
    NO_CHANGE = "no_change"


class ResourceTier(Enum):
    """Resource tier levels for scaling."""
    MINIMAL = "minimal"
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"           # React to current load
    PREDICTIVE = "predictive"       # Predict future load
    SCHEDULED = "scheduled"         # Time-based scaling
    ADAPTIVE = "adaptive"           # Learn from patterns
    COST_OPTIMIZED = "cost_optimized"  # Optimize for cost
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Optimize for performance


class WorkloadPattern(Enum):
    """Types of workload patterns."""
    STEADY_STATE = "steady_state"
    PERIODIC = "periodic"
    BURSTY = "bursty"
    GROWING = "growing"
    DECLINING = "declining"
    IRREGULAR = "irregular"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float = field(default_factory=time.time)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    network_throughput: float = 0.0
    request_rate: float = 0.0
    response_time: float = 0.0
    queue_length: int = 0
    active_connections: int = 0
    error_rate: float = 0.0
    cost_per_hour: float = 0.0
    workload_score: float = 0.0  # Combined workload indicator
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    event_id: str
    timestamp: float = field(default_factory=time.time)
    direction: ScalingDirection = ScalingDirection.NO_CHANGE
    strategy: ScalingStrategy = ScalingStrategy.REACTIVE
    resource_type: str = "compute"
    old_capacity: int = 0
    new_capacity: int = 0
    trigger_metrics: Optional[ScalingMetrics] = None
    predicted_metrics: Optional[ScalingMetrics] = None
    duration_seconds: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    cost_impact: float = 0.0
    performance_impact: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingRule:
    """Defines a scaling rule."""
    rule_id: str
    name: str
    metric_name: str
    threshold_up: float
    threshold_down: float
    comparison_operator: str = "greater_than"  # greater_than, less_than, equals
    evaluation_periods: int = 2
    cooldown_seconds: float = 300.0
    scaling_adjustment: int = 1
    enabled: bool = True
    priority: int = 1
    conditions: List[str] = field(default_factory=list)


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling system."""
    enable_scaling: bool = True
    scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE
    min_capacity: int = 1
    max_capacity: int = 100
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 75.0
    target_response_time_ms: float = 200.0
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    evaluation_interval: float = 60.0  # 1 minute
    prediction_window_minutes: int = 30
    history_retention_hours: int = 48
    enable_predictive_scaling: bool = True
    enable_cost_optimization: bool = True
    enable_performance_optimization: bool = True
    aggressive_scaling_threshold: float = 90.0
    conservative_scaling_threshold: float = 40.0
    emergency_scaling_threshold: float = 95.0
    cost_weight: float = 0.3
    performance_weight: float = 0.7


class WorkloadPredictor:
    """Predicts future workload patterns."""
    
    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self.metrics_history: deque = deque(maxlen=2000)
        self.pattern_models = {}
        self.seasonal_patterns = {}
        
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history for pattern learning."""
        self.metrics_history.append(metrics)
        self._update_pattern_models()
        
    def predict_workload(self, horizon_minutes: int = 30) -> List[ScalingMetrics]:
        """Predict workload for the specified time horizon."""
        if len(self.metrics_history) < 10:
            # Not enough history for prediction
            return self._generate_baseline_predictions(horizon_minutes)
            
        # Detect current pattern
        current_pattern = self._detect_workload_pattern()
        
        # Generate predictions based on pattern
        if current_pattern == WorkloadPattern.PERIODIC:
            return self._predict_periodic_workload(horizon_minutes)
        elif current_pattern == WorkloadPattern.BURSTY:
            return self._predict_bursty_workload(horizon_minutes)
        elif current_pattern == WorkloadPattern.GROWING:
            return self._predict_growing_workload(horizon_minutes)
        elif current_pattern == WorkloadPattern.DECLINING:
            return self._predict_declining_workload(horizon_minutes)
        else:
            return self._predict_steady_state_workload(horizon_minutes)
            
    def _detect_workload_pattern(self) -> WorkloadPattern:
        """Detect the current workload pattern."""
        if len(self.metrics_history) < 20:
            return WorkloadPattern.STEADY_STATE
            
        recent_metrics = list(self.metrics_history)[-20:]
        workload_scores = [m.workload_score for m in recent_metrics]
        
        # Calculate pattern indicators
        mean_workload = statistics.mean(workload_scores)
        std_workload = statistics.stdev(workload_scores) if len(workload_scores) > 1 else 0
        
        # Trend analysis
        if SCIPY_AVAILABLE and len(workload_scores) >= 10:
            x = list(range(len(workload_scores)))
            slope, _, _, _, _ = stats.linregress(x, workload_scores)
            
            # Growing trend
            if slope > 0.05:
                return WorkloadPattern.GROWING
            # Declining trend
            elif slope < -0.05:
                return WorkloadPattern.DECLINING
                
        # Variability analysis
        coefficient_of_variation = std_workload / mean_workload if mean_workload > 0 else 0
        
        if coefficient_of_variation > 0.5:
            return WorkloadPattern.BURSTY
        elif coefficient_of_variation < 0.1:
            return WorkloadPattern.STEADY_STATE
        else:
            # Check for periodicity
            if self._detect_periodicity(workload_scores):
                return WorkloadPattern.PERIODIC
            else:
                return WorkloadPattern.IRREGULAR
                
    def _detect_periodicity(self, values: List[float]) -> bool:
        """Detect if values show periodic behavior."""
        if not NUMPY_AVAILABLE or len(values) < 12:
            return False
            
        # Simple periodicity detection using autocorrelation
        try:
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Look for peaks in autocorrelation (indicating periodicity)
            if len(autocorr) > 3:
                peak_threshold = np.max(autocorr) * 0.7
                peaks = [i for i, val in enumerate(autocorr[1:], 1) if val > peak_threshold]
                return len(peaks) >= 2
                
        except Exception as e:
            logger.debug(f"Periodicity detection error: {e}")
            
        return False
        
    def _predict_periodic_workload(self, horizon_minutes: int) -> List[ScalingMetrics]:
        """Predict workload assuming periodic pattern."""
        predictions = []
        current_time = time.time()
        
        # Use historical pattern to predict
        history_values = [m.workload_score for m in list(self.metrics_history)[-60:]]  # Last hour
        
        for i in range(horizon_minutes):
            future_time = current_time + (i * 60)
            
            # Simple periodic prediction: repeat historical pattern
            history_index = i % len(history_values)
            predicted_score = history_values[history_index]
            
            # Add some trend adjustment
            trend_adjustment = 1.0 + (i * 0.001)  # Slight upward trend
            predicted_score *= trend_adjustment
            
            prediction = ScalingMetrics(
                timestamp=future_time,
                workload_score=predicted_score,
                cpu_utilization=min(95, predicted_score * 0.8),
                memory_utilization=min(90, predicted_score * 0.7),
                request_rate=predicted_score * 10,
                response_time=max(50, 100 + (predicted_score - 50) * 2)
            )
            predictions.append(prediction)
            
        return predictions
        
    def _predict_bursty_workload(self, horizon_minutes: int) -> List[ScalingMetrics]:
        """Predict workload assuming bursty pattern."""
        predictions = []
        current_time = time.time()
        
        # Analyze burst characteristics
        recent_scores = [m.workload_score for m in list(self.metrics_history)[-30:]]
        baseline = statistics.mean(recent_scores)
        
        for i in range(horizon_minutes):
            future_time = current_time + (i * 60)
            
            # Model bursts as random spikes with exponential decay
            burst_probability = 0.1  # 10% chance of burst per minute
            
            if i == 0:
                # Current state continues briefly
                predicted_score = recent_scores[-1] if recent_scores else baseline
            else:
                # Decay towards baseline with occasional bursts
                decay_factor = 0.9 ** i
                burst_factor = 1.0
                
                # Random burst simulation (simplified)
                if (i * 7 + 13) % 17 == 0:  # Pseudo-random burst
                    burst_factor = 2.0
                    
                predicted_score = baseline + (recent_scores[-1] - baseline) * decay_factor * burst_factor
                
            prediction = ScalingMetrics(
                timestamp=future_time,
                workload_score=max(0, predicted_score),
                cpu_utilization=min(95, predicted_score * 0.8),
                memory_utilization=min(90, predicted_score * 0.7),
                request_rate=predicted_score * 10,
                response_time=max(50, 100 + max(0, predicted_score - 50) * 2)
            )
            predictions.append(prediction)
            
        return predictions
        
    def _predict_growing_workload(self, horizon_minutes: int) -> List[ScalingMetrics]:
        """Predict workload assuming growing trend."""
        predictions = []
        current_time = time.time()
        
        # Calculate growth rate
        recent_scores = [m.workload_score for m in list(self.metrics_history)[-20:]]
        if len(recent_scores) >= 2:
            growth_rate = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        else:
            growth_rate = 0.1  # Default growth
            
        base_score = recent_scores[-1] if recent_scores else 50
        
        for i in range(horizon_minutes):
            future_time = current_time + (i * 60)
            
            # Linear growth with some randomness
            predicted_score = base_score + (growth_rate * i)
            
            # Add some noise and cap at reasonable values
            predicted_score = max(0, min(100, predicted_score))
            
            prediction = ScalingMetrics(
                timestamp=future_time,
                workload_score=predicted_score,
                cpu_utilization=min(95, predicted_score * 0.8),
                memory_utilization=min(90, predicted_score * 0.75),
                request_rate=predicted_score * 12,
                response_time=max(50, 80 + predicted_score * 1.5)
            )
            predictions.append(prediction)
            
        return predictions
        
    def _predict_declining_workload(self, horizon_minutes: int) -> List[ScalingMetrics]:
        """Predict workload assuming declining trend."""
        predictions = []
        current_time = time.time()
        
        # Calculate decline rate
        recent_scores = [m.workload_score for m in list(self.metrics_history)[-20:]]
        if len(recent_scores) >= 2:
            decline_rate = (recent_scores[0] - recent_scores[-1]) / len(recent_scores)
        else:
            decline_rate = 0.1  # Default decline
            
        base_score = recent_scores[-1] if recent_scores else 50
        
        for i in range(horizon_minutes):
            future_time = current_time + (i * 60)
            
            # Exponential decay
            decay_factor = 0.98 ** i
            predicted_score = base_score * decay_factor
            
            # Floor at minimum level
            predicted_score = max(10, predicted_score)
            
            prediction = ScalingMetrics(
                timestamp=future_time,
                workload_score=predicted_score,
                cpu_utilization=min(95, predicted_score * 0.7),
                memory_utilization=min(90, predicted_score * 0.6),
                request_rate=predicted_score * 8,
                response_time=max(30, 60 + predicted_score * 1.2)
            )
            predictions.append(prediction)
            
        return predictions
        
    def _predict_steady_state_workload(self, horizon_minutes: int) -> List[ScalingMetrics]:
        """Predict workload assuming steady state."""
        predictions = []
        current_time = time.time()
        
        # Use recent average as baseline
        recent_scores = [m.workload_score for m in list(self.metrics_history)[-10:]]
        baseline_score = statistics.mean(recent_scores) if recent_scores else 50
        
        for i in range(horizon_minutes):
            future_time = current_time + (i * 60)
            
            # Steady state with minor fluctuations
            fluctuation = 0.95 + (0.1 * ((i * 3 + 7) % 10) / 10)  # ±5% variation
            predicted_score = baseline_score * fluctuation
            
            prediction = ScalingMetrics(
                timestamp=future_time,
                workload_score=predicted_score,
                cpu_utilization=min(95, predicted_score * 0.75),
                memory_utilization=min(90, predicted_score * 0.7),
                request_rate=predicted_score * 10,
                response_time=max(40, 70 + predicted_score * 1.3)
            )
            predictions.append(prediction)
            
        return predictions
        
    def _generate_baseline_predictions(self, horizon_minutes: int) -> List[ScalingMetrics]:
        """Generate baseline predictions when insufficient history."""
        predictions = []
        current_time = time.time()
        
        # Use current metrics as baseline
        current_metrics = self.metrics_history[-1] if self.metrics_history else ScalingMetrics()
        
        for i in range(horizon_minutes):
            future_time = current_time + (i * 60)
            
            # Assume steady state with slight variations
            variation = 0.95 + (0.1 * (i % 10) / 10)
            
            prediction = ScalingMetrics(
                timestamp=future_time,
                workload_score=current_metrics.workload_score * variation,
                cpu_utilization=current_metrics.cpu_utilization * variation,
                memory_utilization=current_metrics.memory_utilization * variation,
                request_rate=current_metrics.request_rate * variation,
                response_time=current_metrics.response_time * (2.0 - variation)  # Inverse relationship
            )
            predictions.append(prediction)
            
        return predictions
        
    def _update_pattern_models(self):
        """Update pattern recognition models with new data."""
        if len(self.metrics_history) < 50:
            return
            
        # Simple pattern recognition - in production would use more sophisticated ML
        recent_data = list(self.metrics_history)[-50:]
        
        # Update seasonal patterns
        self._update_seasonal_patterns(recent_data)
        
    def _update_seasonal_patterns(self, data: List[ScalingMetrics]):
        """Update seasonal pattern recognition."""
        # Group by hour of day to detect daily patterns
        hourly_patterns = defaultdict(list)
        
        for metrics in data:
            hour = int((metrics.timestamp % 86400) // 3600)  # Hour of day
            hourly_patterns[hour].append(metrics.workload_score)
            
        # Calculate average workload by hour
        for hour, scores in hourly_patterns.items():
            if len(scores) >= 3:  # Need minimum data points
                self.seasonal_patterns[f"hour_{hour}"] = {
                    'mean': statistics.mean(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'samples': len(scores)
                }


class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on metrics and predictions."""
    
    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self.scaling_rules: List[ScalingRule] = []
        self.cooldown_state = {
            'last_scale_up': 0,
            'last_scale_down': 0
        }
        self.current_capacity = config.min_capacity
        
        # Initialize default scaling rules
        self._create_default_rules()
        
    def _create_default_rules(self):
        """Create default scaling rules."""
        # CPU-based scaling
        self.scaling_rules.append(ScalingRule(
            rule_id="cpu_scale_up",
            name="Scale up on high CPU",
            metric_name="cpu_utilization",
            threshold_up=self.config.target_cpu_utilization + 10,
            threshold_down=self.config.target_cpu_utilization - 10,
            comparison_operator="greater_than",
            evaluation_periods=2,
            cooldown_seconds=self.config.scale_up_cooldown,
            scaling_adjustment=1
        ))
        
        # Memory-based scaling
        self.scaling_rules.append(ScalingRule(
            rule_id="memory_scale_up",
            name="Scale up on high memory",
            metric_name="memory_utilization",
            threshold_up=self.config.target_memory_utilization + 10,
            threshold_down=self.config.target_memory_utilization - 10,
            comparison_operator="greater_than",
            evaluation_periods=2,
            cooldown_seconds=self.config.scale_up_cooldown,
            scaling_adjustment=1
        ))
        
        # Response time-based scaling
        self.scaling_rules.append(ScalingRule(
            rule_id="latency_scale_up",
            name="Scale up on high latency",
            metric_name="response_time",
            threshold_up=self.config.target_response_time_ms + 100,
            threshold_down=self.config.target_response_time_ms - 50,
            comparison_operator="greater_than",
            evaluation_periods=3,
            cooldown_seconds=self.config.scale_up_cooldown,
            scaling_adjustment=2  # More aggressive for latency
        ))
        
        # Emergency scaling
        self.scaling_rules.append(ScalingRule(
            rule_id="emergency_scale_up",
            name="Emergency scale up",
            metric_name="cpu_utilization",
            threshold_up=self.config.emergency_scaling_threshold,
            threshold_down=0,  # Never scale down on this rule
            comparison_operator="greater_than",
            evaluation_periods=1,  # Immediate action
            cooldown_seconds=60,  # Short cooldown for emergencies
            scaling_adjustment=3,  # Aggressive scaling
            priority=10  # High priority
        ))
        
    def evaluate_scaling_decision(
        self,
        current_metrics: ScalingMetrics,
        predicted_metrics: List[ScalingMetrics] = None,
        historical_metrics: List[ScalingMetrics] = None
    ) -> ScalingEvent:
        """Evaluate whether scaling is needed and determine the action."""
        
        event = ScalingEvent(
            event_id=f"eval_{int(time.time() * 1000)}",
            trigger_metrics=current_metrics,
            predicted_metrics=predicted_metrics[0] if predicted_metrics else None,
            old_capacity=self.current_capacity
        )
        
        # Check cooldown periods
        current_time = time.time()
        if self._is_in_cooldown(current_time):
            event.direction = ScalingDirection.NO_CHANGE
            event.error_message = "Scaling action is in cooldown period"
            return event
            
        # Evaluate scaling rules
        scale_up_votes = 0
        scale_down_votes = 0
        max_adjustment = 0
        triggered_rules = []
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
                
            decision = self._evaluate_rule(rule, current_metrics, historical_metrics)
            if decision['triggered']:
                triggered_rules.append(rule.name)
                if decision['direction'] == 'up':
                    scale_up_votes += rule.priority
                    max_adjustment = max(max_adjustment, rule.scaling_adjustment)
                elif decision['direction'] == 'down':
                    scale_down_votes += rule.priority
                    
        # Predictive scaling adjustments
        if predicted_metrics and self.config.enable_predictive_scaling:
            predictive_adjustment = self._evaluate_predictive_scaling(predicted_metrics)
            if predictive_adjustment > 0:
                scale_up_votes += predictive_adjustment
            elif predictive_adjustment < 0:
                scale_down_votes += abs(predictive_adjustment)
                
        # Make final decision
        if scale_up_votes > scale_down_votes:
            event.direction = ScalingDirection.SCALE_UP
            event.strategy = ScalingStrategy.REACTIVE if not predicted_metrics else ScalingStrategy.PREDICTIVE
            proposed_capacity = self.current_capacity + max(1, max_adjustment)
            event.new_capacity = min(self.config.max_capacity, proposed_capacity)
        elif scale_down_votes > scale_up_votes:
            event.direction = ScalingDirection.SCALE_DOWN
            event.strategy = ScalingStrategy.REACTIVE
            event.new_capacity = max(self.config.min_capacity, self.current_capacity - 1)
        else:
            event.direction = ScalingDirection.NO_CHANGE
            event.new_capacity = self.current_capacity
            
        # Cost optimization adjustments
        if self.config.enable_cost_optimization:
            event = self._apply_cost_optimization(event, current_metrics)
            
        # Performance optimization adjustments
        if self.config.enable_performance_optimization:
            event = self._apply_performance_optimization(event, current_metrics)
            
        event.metadata['triggered_rules'] = triggered_rules
        event.metadata['scale_up_votes'] = scale_up_votes
        event.metadata['scale_down_votes'] = scale_down_votes
        
        return event
        
    def _evaluate_rule(
        self,
        rule: ScalingRule,
        current_metrics: ScalingMetrics,
        historical_metrics: List[ScalingMetrics] = None
    ) -> Dict[str, Any]:
        """Evaluate a single scaling rule."""
        
        # Get metric value
        metric_value = getattr(current_metrics, rule.metric_name, 0)
        
        # Evaluate conditions
        triggered = False
        direction = None
        
        if rule.comparison_operator == "greater_than":
            if metric_value > rule.threshold_up:
                triggered = True
                direction = 'up'
            elif metric_value < rule.threshold_down:
                triggered = True
                direction = 'down'
        elif rule.comparison_operator == "less_than":
            if metric_value < rule.threshold_down:
                triggered = True
                direction = 'up'
            elif metric_value > rule.threshold_up:
                triggered = True
                direction = 'down'
                
        # Check evaluation periods with historical data
        if triggered and historical_metrics and rule.evaluation_periods > 1:
            recent_metrics = historical_metrics[-(rule.evaluation_periods-1):] + [current_metrics]
            
            # All recent metrics must trigger the rule
            for metrics in recent_metrics:
                metric_val = getattr(metrics, rule.metric_name, 0)
                if rule.comparison_operator == "greater_than":
                    if direction == 'up' and metric_val <= rule.threshold_up:
                        triggered = False
                        break
                    elif direction == 'down' and metric_val >= rule.threshold_down:
                        triggered = False
                        break
                        
        return {
            'triggered': triggered,
            'direction': direction,
            'metric_value': metric_value,
            'threshold_up': rule.threshold_up,
            'threshold_down': rule.threshold_down
        }
        
    def _evaluate_predictive_scaling(self, predicted_metrics: List[ScalingMetrics]) -> int:
        """Evaluate need for predictive scaling based on predictions."""
        if not predicted_metrics:
            return 0
            
        # Look at predictions for next 10-15 minutes
        near_future = predicted_metrics[:15]
        
        # Count how many predictions exceed thresholds
        cpu_violations = sum(1 for m in near_future if m.cpu_utilization > self.config.target_cpu_utilization + 15)
        memory_violations = sum(1 for m in near_future if m.memory_utilization > self.config.target_memory_utilization + 15)
        latency_violations = sum(1 for m in near_future if m.response_time > self.config.target_response_time_ms + 100)
        
        total_violations = cpu_violations + memory_violations + latency_violations
        
        if total_violations >= 5:  # Significant predicted load
            return 2  # Strong recommendation to scale up
        elif total_violations >= 2:
            return 1  # Mild recommendation to scale up
        else:
            # Check for scale down opportunity
            cpu_underutilized = sum(1 for m in near_future if m.cpu_utilization < self.config.conservative_scaling_threshold)
            if cpu_underutilized >= 10:  # Most of the period underutilized
                return -1  # Mild recommendation to scale down
                
        return 0
        
    def _apply_cost_optimization(self, event: ScalingEvent, metrics: ScalingMetrics) -> ScalingEvent:
        """Apply cost optimization to scaling decision."""
        if event.direction == ScalingDirection.NO_CHANGE:
            return event
            
        # Estimate cost impact
        cost_per_unit = 0.10  # Mock cost per unit per hour
        
        if event.direction == ScalingDirection.SCALE_UP:
            cost_increase = (event.new_capacity - event.old_capacity) * cost_per_unit
            event.cost_impact = cost_increase
            
            # If cost increase is significant and utilization is not critical, be more conservative
            if cost_increase > 1.0 and metrics.cpu_utilization < self.config.aggressive_scaling_threshold:
                event.new_capacity = event.old_capacity + 1  # More conservative scaling
                event.cost_impact = cost_per_unit
                
        elif event.direction == ScalingDirection.SCALE_DOWN:
            cost_decrease = (event.old_capacity - event.new_capacity) * cost_per_unit
            event.cost_impact = -cost_decrease  # Negative because it's a saving
            
        return event
        
    def _apply_performance_optimization(self, event: ScalingEvent, metrics: ScalingMetrics) -> ScalingEvent:
        """Apply performance optimization to scaling decision."""
        if event.direction == ScalingDirection.NO_CHANGE:
            return event
            
        # Calculate performance impact
        if event.direction == ScalingDirection.SCALE_UP:
            # Estimate performance improvement
            capacity_increase_ratio = event.new_capacity / max(event.old_capacity, 1)
            estimated_cpu_reduction = metrics.cpu_utilization / capacity_increase_ratio
            estimated_latency_improvement = max(0.1, metrics.response_time * 0.8)
            
            event.performance_impact = (metrics.cpu_utilization - estimated_cpu_reduction) / 100
            
            # If performance is critical, be more aggressive
            if (metrics.cpu_utilization > self.config.aggressive_scaling_threshold or
                metrics.response_time > self.config.target_response_time_ms * 2):
                event.new_capacity += 1  # Extra aggressive scaling
                
        return event
        
    def _is_in_cooldown(self, current_time: float) -> bool:
        """Check if scaling is in cooldown period."""
        scale_up_cooldown = current_time - self.cooldown_state['last_scale_up'] < self.config.scale_up_cooldown
        scale_down_cooldown = current_time - self.cooldown_state['last_scale_down'] < self.config.scale_down_cooldown
        return scale_up_cooldown or scale_down_cooldown
        
    def execute_scaling_decision(self, event: ScalingEvent) -> ScalingEvent:
        """Execute the scaling decision."""
        start_time = time.time()
        
        try:
            if event.direction == ScalingDirection.NO_CHANGE:
                event.success = True
                event.duration_seconds = time.time() - start_time
                return event
                
            # Simulate scaling operation
            old_capacity = self.current_capacity
            new_capacity = event.new_capacity
            
            # Update cooldown state
            current_time = time.time()
            if event.direction in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
                self.cooldown_state['last_scale_up'] = current_time
            else:
                self.cooldown_state['last_scale_down'] = current_time
                
            # Simulate scaling delay
            scaling_delay = min(30, max(5, abs(new_capacity - old_capacity) * 2))
            time.sleep(scaling_delay * 0.01)  # Simulate quick scaling for demo
            
            # Update current capacity
            self.current_capacity = new_capacity
            
            event.success = True
            event.duration_seconds = time.time() - start_time
            
            logger.info(f"Scaling executed: {old_capacity} -> {new_capacity} ({event.direction.value})")
            
        except Exception as e:
            event.success = False
            event.error_message = str(e)
            event.duration_seconds = time.time() - start_time
            logger.error(f"Scaling execution failed: {e}")
            
        return event


class IntelligentAutoScalingSystem:
    """
    Intelligent Auto-Scaling System
    
    Provides predictive auto-scaling with intelligent resource provisioning,
    cost optimization, and adaptive capacity management.
    """
    
    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self.workload_predictor = WorkloadPredictor(config)
        self.decision_engine = ScalingDecisionEngine(config)
        
        # State management
        self.is_running = False
        self.scaling_thread = None
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_events: List[ScalingEvent] = []
        self.lock = threading.Lock()
        
        # Statistics
        self.scaling_stats = {
            'total_scaling_events': 0,
            'successful_scale_ups': 0,
            'successful_scale_downs': 0,
            'failed_scaling_events': 0,
            'cost_saved': 0.0,
            'performance_improved': 0.0
        }
        
        logger.info("Intelligent Auto-Scaling System initialized")
        
    def start_auto_scaling(self):
        """Start the auto-scaling system."""
        if self.is_running:
            return
            
        self.is_running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Auto-scaling system started")
        
    def stop_auto_scaling(self):
        """Stop the auto-scaling system."""
        self.is_running = False
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)
            
        logger.info("Auto-scaling system stopped")
        
    def add_metrics(self, metrics: ScalingMetrics):
        """Add current metrics for scaling evaluation."""
        with self.lock:
            self.metrics_history.append(metrics)
            self.workload_predictor.add_metrics(metrics)
            
    def _scaling_loop(self):
        """Main auto-scaling evaluation loop."""
        while self.is_running:
            try:
                if self.config.enable_scaling:
                    self._evaluate_and_scale()
                    
                time.sleep(self.config.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)  # Back off on errors
                
    def _evaluate_and_scale(self):
        """Evaluate current state and execute scaling if needed."""
        with self.lock:
            if not self.metrics_history:
                return
                
            current_metrics = self.metrics_history[-1]
            historical_metrics = list(self.metrics_history)
            
        # Generate predictions
        predictions = None
        if self.config.enable_predictive_scaling:
            predictions = self.workload_predictor.predict_workload(
                self.config.prediction_window_minutes
            )
            
        # Make scaling decision
        scaling_event = self.decision_engine.evaluate_scaling_decision(
            current_metrics, predictions, historical_metrics
        )
        
        # Execute scaling if needed
        if scaling_event.direction != ScalingDirection.NO_CHANGE:
            scaling_event = self.decision_engine.execute_scaling_decision(scaling_event)
            
            # Update statistics
            self._update_scaling_stats(scaling_event)
            
        # Store event
        with self.lock:
            self.scaling_events.append(scaling_event)
            if len(self.scaling_events) > 1000:  # Keep only recent events
                self.scaling_events.pop(0)
                
    def _update_scaling_stats(self, event: ScalingEvent):
        """Update scaling statistics."""
        self.scaling_stats['total_scaling_events'] += 1
        
        if event.success:
            if event.direction in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
                self.scaling_stats['successful_scale_ups'] += 1
            elif event.direction in [ScalingDirection.SCALE_DOWN, ScalingDirection.SCALE_IN]:
                self.scaling_stats['successful_scale_downs'] += 1
                
            # Update cost and performance stats
            if event.cost_impact < 0:  # Cost saving
                self.scaling_stats['cost_saved'] += abs(event.cost_impact)
            if event.performance_impact > 0:  # Performance improvement
                self.scaling_stats['performance_improved'] += event.performance_impact
        else:
            self.scaling_stats['failed_scaling_events'] += 1
            
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status."""
        with self.lock:
            current_capacity = self.decision_engine.current_capacity
            recent_events = self.scaling_events[-10:] if self.scaling_events else []
            recent_metrics = list(self.metrics_history)[-5:] if self.metrics_history else []
            
        # Generate predictions for status
        predictions = []
        if self.config.enable_predictive_scaling and self.metrics_history:
            predictions = self.workload_predictor.predict_workload(15)[:5]  # Next 5 minutes
            
        return {
            'system_status': {
                'is_running': self.is_running,
                'current_capacity': current_capacity,
                'min_capacity': self.config.min_capacity,
                'max_capacity': self.config.max_capacity,
                'scaling_strategy': self.config.scaling_strategy.value
            },
            'current_metrics': recent_metrics[-1].__dict__ if recent_metrics else {},
            'predicted_metrics': [p.__dict__ for p in predictions],
            'recent_events': [
                {
                    'timestamp': event.timestamp,
                    'direction': event.direction.value,
                    'old_capacity': event.old_capacity,
                    'new_capacity': event.new_capacity,
                    'success': event.success,
                    'cost_impact': event.cost_impact,
                    'performance_impact': event.performance_impact
                }
                for event in recent_events
            ],
            'scaling_statistics': self.scaling_stats.copy(),
            'configuration': {
                'target_cpu_utilization': self.config.target_cpu_utilization,
                'target_memory_utilization': self.config.target_memory_utilization,
                'target_response_time_ms': self.config.target_response_time_ms,
                'scale_up_cooldown': self.config.scale_up_cooldown,
                'scale_down_cooldown': self.config.scale_down_cooldown
            }
        }
        
    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get scaling optimization recommendations."""
        recommendations = []
        
        with self.lock:
            if not self.metrics_history:
                return recommendations
                
            recent_metrics = list(self.metrics_history)[-20:]
            
        # Analyze recent performance
        avg_cpu = statistics.mean([m.cpu_utilization for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_utilization for m in recent_metrics])
        avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
        
        # Generate recommendations
        if avg_cpu < self.config.conservative_scaling_threshold:
            recommendations.append({
                'type': 'cost_optimization',
                'priority': 'medium',
                'recommendation': 'CPU utilization is consistently low. Consider scaling down to reduce costs.',
                'potential_savings': f"${self.scaling_stats.get('cost_saved', 0):.2f}/hour"
            })
            
        if avg_response_time > self.config.target_response_time_ms * 1.5:
            recommendations.append({
                'type': 'performance_optimization',
                'priority': 'high',
                'recommendation': 'Response time is consistently high. Consider more aggressive scaling policies.',
                'current_value': f"{avg_response_time:.1f}ms",
                'target_value': f"{self.config.target_response_time_ms}ms"
            })
            
        if avg_memory > 85:
            recommendations.append({
                'type': 'resource_optimization',
                'priority': 'high',
                'recommendation': 'Memory utilization is very high. Consider scaling up or optimizing memory usage.',
                'current_value': f"{avg_memory:.1f}%"
            })
            
        # Analyze scaling frequency
        recent_scale_events = len([e for e in self.scaling_events[-50:] if e.success])
        if recent_scale_events > 20:
            recommendations.append({
                'type': 'configuration_optimization',
                'priority': 'medium',
                'recommendation': 'High scaling frequency detected. Consider adjusting thresholds or cooldown periods.',
                'scaling_events_per_hour': recent_scale_events
            })
            
        return recommendations


# Demo and testing
def demo_autoscaling_system():
    """Demonstrate the intelligent auto-scaling system."""
    config = AutoScalingConfig(
        enable_scaling=True,
        min_capacity=2,
        max_capacity=20,
        target_cpu_utilization=70.0,
        enable_predictive_scaling=True,
        evaluation_interval=10.0  # Fast evaluation for demo
    )
    
    autoscaler = IntelligentAutoScalingSystem(config)
    
    print("=== Intelligent Auto-Scaling System Demo ===")
    
    # Start auto-scaling
    autoscaler.start_auto_scaling()
    
    print("Simulating workload changes...")
    
    # Simulate varying workload
    for i in range(20):
        # Create varying metrics
        base_load = 30 + 20 * math.sin(i * 0.3)  # Oscillating load
        spike = 40 if i in [8, 15] else 0  # Occasional spikes
        
        metrics = ScalingMetrics(
            cpu_utilization=base_load + spike + (i % 3) * 5,
            memory_utilization=base_load * 0.8 + spike * 0.5,
            response_time=50 + base_load + spike,
            request_rate=base_load * 2,
            workload_score=base_load + spike,
            queue_length=max(0, int((base_load + spike - 50) / 10))
        )
        
        autoscaler.add_metrics(metrics)
        
        if i % 5 == 0:  # Print status every 5 iterations
            status = autoscaler.get_scaling_status()
            print(f"Step {i}: Capacity={status['system_status']['current_capacity']}, "
                  f"CPU={metrics.cpu_utilization:.1f}%, "
                  f"Response={metrics.response_time:.0f}ms")
                  
        time.sleep(1)  # Simulate time passing
        
    # Get final status and recommendations
    final_status = autoscaler.get_scaling_status()
    recommendations = autoscaler.get_scaling_recommendations()
    
    print(f"\n=== Final Status ===")
    print(f"Final Capacity: {final_status['system_status']['current_capacity']}")
    print(f"Total Scaling Events: {final_status['scaling_statistics']['total_scaling_events']}")
    print(f"Successful Scale-ups: {final_status['scaling_statistics']['successful_scale_ups']}")
    print(f"Successful Scale-downs: {final_status['scaling_statistics']['successful_scale_downs']}")
    
    print(f"\n=== Recommendations ===")
    for rec in recommendations:
        print(f"- {rec['recommendation']}")
        
    # Stop auto-scaling
    autoscaler.stop_auto_scaling()
    
    print("\nAuto-scaling demo completed.")


if __name__ == "__main__":
    demo_autoscaling_system()