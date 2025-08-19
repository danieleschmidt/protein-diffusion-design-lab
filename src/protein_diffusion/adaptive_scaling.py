"""
Adaptive Scaling System - Dynamic resource scaling and load balancing.

This module provides intelligent auto-scaling capabilities that dynamically adjust
resources based on workload patterns, performance metrics, and system constraints.
"""

import logging
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operation."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    WORKERS = "workers"
    MEMORY = "memory"
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"


class ScalingTrigger(Enum):
    """Triggers for scaling operations."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_SIZE = "queue_size"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingRule:
    """Definition of a scaling rule."""
    name: str
    trigger: ScalingTrigger
    resource_type: ResourceType
    direction: ScalingDirection
    threshold: float
    comparison: str = "greater_than"  # greater_than, less_than, equals
    cooldown_seconds: float = 300.0  # 5 minutes
    scale_factor: float = 1.5  # Factor to scale by
    min_instances: int = 1
    max_instances: int = 10
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    event_id: str
    timestamp: float = field(default_factory=time.time)
    rule_name: str = ""
    resource_type: ResourceType = ResourceType.WORKERS
    direction: ScalingDirection = ScalingDirection.MAINTAIN
    trigger_value: float = 0.0
    threshold: float = 0.0
    old_capacity: int = 0
    new_capacity: int = 0
    success: bool = False
    duration: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    timestamp: float = field(default_factory=time.time)
    
    # System metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_io_rate: float = 0.0
    
    # Application metrics
    active_workers: int = 0
    queue_size: int = 0
    processing_rate: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Capacity information
    current_capacity: Dict[str, int] = field(default_factory=dict)
    max_capacity: Dict[str, int] = field(default_factory=dict)
    
    # Load balancing
    load_distribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingConfig:
    """Configuration for adaptive scaling system."""
    # Scaling behavior
    enable_auto_scaling: bool = True
    scaling_check_interval: float = 60.0  # 1 minute
    metric_collection_interval: float = 10.0  # 10 seconds
    
    # Default thresholds
    cpu_scale_up_threshold: float = 80.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 85.0
    memory_scale_down_threshold: float = 40.0
    queue_scale_up_threshold: float = 100.0
    queue_scale_down_threshold: float = 10.0
    
    # Scaling limits
    min_workers: int = 2
    max_workers: int = 20
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    
    # Cooldown periods
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    
    # Prediction and intelligence
    enable_predictive_scaling: bool = True
    prediction_window_minutes: float = 30.0
    enable_learning: bool = True
    
    # Constraints
    max_cost_per_hour: Optional[float] = None
    preferred_availability_zones: List[str] = field(default_factory=list)
    
    # Monitoring and alerting
    enable_scaling_alerts: bool = True
    alert_webhook_urls: List[str] = field(default_factory=list)
    
    # Storage and logging
    save_scaling_history: bool = True
    history_retention_days: int = 30
    log_directory: str = "./scaling_logs"


class MetricsCollector:
    """Collects and aggregates system and application metrics."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.metrics_history: deque = deque(maxlen=1000)
        self.custom_metric_providers: Dict[str, Callable[[], float]] = {}
        self.lock = threading.RLock()
        
        # Integration points
        self.distributed_manager = None
        self.monitoring_system = None
        
        logger.info("Metrics collector initialized")
    
    def register_distributed_manager(self, manager):
        """Register distributed processing manager for metrics."""
        self.distributed_manager = manager
        logger.info("Registered distributed processing manager")
    
    def register_monitoring_system(self, monitor):
        """Register system monitor for metrics."""
        self.monitoring_system = monitor
        logger.info("Registered monitoring system")
    
    def register_custom_metric(self, name: str, provider: Callable[[], float]):
        """Register custom metric provider."""
        self.custom_metric_providers[name] = provider
        logger.info(f"Registered custom metric: {name}")
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        metrics = ResourceMetrics()
        
        try:
            # System metrics
            if PSUTIL_AVAILABLE:
                metrics.cpu_utilization = psutil.cpu_percent(interval=0.1)
                
                memory = psutil.virtual_memory()
                metrics.memory_utilization = memory.percent
                
                disk = psutil.disk_usage('/')
                metrics.disk_utilization = disk.used / disk.total * 100
                
                net_io = psutil.net_io_counters()
                metrics.network_io_rate = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB/s
            
            # Application metrics from distributed manager
            if self.distributed_manager:
                status = self.distributed_manager.get_system_status()
                metrics.active_workers = status.get("total_workers", 0)
                
                queue_stats = status.get("queue_stats", {})
                metrics.queue_size = queue_stats.get("total_queued", 0)
                
                perf_metrics = status.get("performance_metrics", {})
                metrics.avg_response_time = perf_metrics.get("avg_response_time", 0.0)
                metrics.processing_rate = perf_metrics.get("avg_completion_rate", 0.0)
                
                # Set capacity info
                metrics.current_capacity["workers"] = metrics.active_workers
                metrics.max_capacity["workers"] = self.config.max_workers
            
            # Metrics from monitoring system
            if self.monitoring_system:
                monitor_metrics = self.monitoring_system.get_current_metrics()
                app_metrics = monitor_metrics.get("application_metrics", {})
                
                metrics.error_rate = app_metrics.get("error_rate", 0.0)
                metrics.throughput = app_metrics.get("requests_per_second", 0.0)
            
            # Custom metrics
            for name, provider in self.custom_metric_providers.items():
                try:
                    metrics.custom_metrics[name] = provider()
                except Exception as e:
                    logger.warning(f"Failed to collect custom metric {name}: {e}")
            
            # Store in history
            with self.lock:
                self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return metrics
    
    def get_metrics_trend(self, metric_name: str, window_minutes: float = 15.0) -> Tuple[List[float], float]:
        """Get trend for a specific metric."""
        with self.lock:
            if not self.metrics_history:
                return [], 0.0
            
            current_time = time.time()
            window_seconds = window_minutes * 60
            
            # Filter metrics within time window
            recent_metrics = [
                m for m in self.metrics_history
                if current_time - m.timestamp <= window_seconds
            ]
            
            if len(recent_metrics) < 2:
                return [], 0.0
            
            # Extract metric values
            values = []
            for m in recent_metrics:
                if hasattr(m, metric_name):
                    values.append(getattr(m, metric_name))
                elif metric_name in m.custom_metrics:
                    values.append(m.custom_metrics[metric_name])
            
            if not values:
                return [], 0.0
            
            # Calculate trend (simple linear regression)
            n = len(values)
            x_values = list(range(n))
            
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n
            
            numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
            
            trend = numerator / denominator if denominator != 0 else 0.0
            
            return values, trend
    
    def get_metrics_summary(self, window_minutes: float = 60.0) -> Dict[str, Any]:
        """Get summary of metrics over time window."""
        with self.lock:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            current_time = time.time()
            window_seconds = window_minutes * 60
            
            # Filter recent metrics
            recent_metrics = [
                m for m in self.metrics_history
                if current_time - m.timestamp <= window_seconds
            ]
            
            if not recent_metrics:
                return {"error": "No recent metrics available"}
            
            # Calculate summaries
            summary = {
                "window_minutes": window_minutes,
                "sample_count": len(recent_metrics),
                "timestamp": current_time,
            }
            
            # System metrics summaries
            metrics_fields = [
                "cpu_utilization", "memory_utilization", "disk_utilization",
                "active_workers", "queue_size", "processing_rate",
                "avg_response_time", "error_rate", "throughput"
            ]
            
            for field in metrics_fields:
                values = [getattr(m, field) for m in recent_metrics]
                if values:
                    summary[field] = {
                        "current": values[-1],
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "trend": self.get_metrics_trend(field, window_minutes)[1]
                    }
            
            return summary


class PredictiveScaler:
    """Predictive scaling based on historical patterns and trends."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.pattern_history: deque = deque(maxlen=10000)  # Store pattern data
        self.predictions: Dict[str, float] = {}
        self.learning_enabled = config.enable_learning
        
        logger.info("Predictive scaler initialized")
    
    def learn_from_metrics(self, metrics: ResourceMetrics):
        """Learn patterns from historical metrics."""
        if not self.learning_enabled:
            return
        
        try:
            # Extract pattern features
            pattern = {
                "timestamp": metrics.timestamp,
                "hour_of_day": time.localtime(metrics.timestamp).tm_hour,
                "day_of_week": time.localtime(metrics.timestamp).tm_wday,
                "cpu_utilization": metrics.cpu_utilization,
                "memory_utilization": metrics.memory_utilization,
                "queue_size": metrics.queue_size,
                "throughput": metrics.throughput,
                "active_workers": metrics.active_workers
            }
            
            self.pattern_history.append(pattern)
            
        except Exception as e:
            logger.error(f"Failed to learn from metrics: {e}")
    
    def predict_resource_need(
        self,
        resource_type: ResourceType,
        prediction_window_minutes: float
    ) -> Tuple[float, float]:
        """
        Predict future resource need.
        
        Returns:
            Tuple of (predicted_value, confidence_score)
        """
        if len(self.pattern_history) < 10:
            return 0.0, 0.0
        
        try:
            current_time = time.time()
            current_hour = time.localtime(current_time).tm_hour
            current_day = time.localtime(current_time).tm_wday
            
            # Find similar historical patterns
            similar_patterns = []
            for pattern in self.pattern_history:
                pattern_hour = time.localtime(pattern["timestamp"]).tm_hour
                pattern_day = time.localtime(pattern["timestamp"]).tm_wday
                
                # Check if pattern is from similar time
                if abs(pattern_hour - current_hour) <= 1 and pattern_day == current_day:
                    similar_patterns.append(pattern)
            
            if len(similar_patterns) < 3:
                return 0.0, 0.0
            
            # Predict based on resource type
            if resource_type == ResourceType.WORKERS:
                # Predict worker need based on queue size and throughput patterns
                avg_queue_size = sum(p["queue_size"] for p in similar_patterns) / len(similar_patterns)
                avg_throughput = sum(p["throughput"] for p in similar_patterns) / len(similar_patterns)
                
                # Simple prediction: workers needed = queue_size / throughput_per_worker
                throughput_per_worker = avg_throughput / max(1, sum(p["active_workers"] for p in similar_patterns) / len(similar_patterns))
                predicted_workers = avg_queue_size / max(1, throughput_per_worker)
                
                confidence = min(1.0, len(similar_patterns) / 20.0)  # Max confidence at 20 samples
                
                return predicted_workers, confidence
            
            elif resource_type == ResourceType.MEMORY:
                # Predict memory need based on worker count and utilization patterns
                avg_memory_util = sum(p["memory_utilization"] for p in similar_patterns) / len(similar_patterns)
                confidence = min(1.0, len(similar_patterns) / 15.0)
                
                return avg_memory_util, confidence
            
            elif resource_type == ResourceType.COMPUTE:
                # Predict compute need based on CPU utilization patterns
                avg_cpu_util = sum(p["cpu_utilization"] for p in similar_patterns) / len(similar_patterns)
                confidence = min(1.0, len(similar_patterns) / 15.0)
                
                return avg_cpu_util, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed for {resource_type}: {e}")
        
        return 0.0, 0.0
    
    def get_scaling_recommendation(
        self,
        current_metrics: ResourceMetrics,
        prediction_window_minutes: float = 30.0
    ) -> Dict[ResourceType, Tuple[ScalingDirection, float, float]]:
        """
        Get scaling recommendations based on predictions.
        
        Returns:
            Dict mapping resource type to (direction, recommended_capacity, confidence)
        """
        recommendations = {}
        
        try:
            for resource_type in [ResourceType.WORKERS, ResourceType.MEMORY, ResourceType.COMPUTE]:
                predicted_need, confidence = self.predict_resource_need(
                    resource_type, prediction_window_minutes
                )
                
                if confidence < 0.3:  # Low confidence, skip recommendation
                    continue
                
                current_capacity = current_metrics.current_capacity.get(resource_type.value, 1)
                
                # Determine scaling direction
                if predicted_need > current_capacity * 1.2:  # Need 20% more
                    direction = ScalingDirection.SCALE_UP
                    recommended_capacity = min(
                        current_metrics.max_capacity.get(resource_type.value, 10),
                        predicted_need * 1.1  # Add 10% buffer
                    )
                elif predicted_need < current_capacity * 0.7:  # Need 30% less
                    direction = ScalingDirection.SCALE_DOWN
                    recommended_capacity = max(
                        1,  # Minimum capacity
                        predicted_need * 1.2  # Add 20% buffer
                    )
                else:
                    direction = ScalingDirection.MAINTAIN
                    recommended_capacity = current_capacity
                
                recommendations[resource_type] = (direction, recommended_capacity, confidence)
            
        except Exception as e:
            logger.error(f"Failed to generate scaling recommendations: {e}")
        
        return recommendations


class AdaptiveScalingManager:
    """
    Main adaptive scaling manager that coordinates all scaling operations.
    
    This class provides:
    - Rule-based scaling triggers
    - Predictive scaling capabilities
    - Integration with distributed processing
    - Performance-aware scaling decisions
    - Cost-conscious resource management
    
    Example:
        >>> scaler = AdaptiveScalingManager()
        >>> scaler.register_distributed_manager(dist_manager)
        >>> scaler.start_scaling()
        >>> 
        >>> # Add custom scaling rule
        >>> rule = ScalingRule(
        ...     name="high_queue_scale_up",
        ...     trigger=ScalingTrigger.QUEUE_SIZE,
        ...     resource_type=ResourceType.WORKERS,
        ...     direction=ScalingDirection.SCALE_UP,
        ...     threshold=50
        ... )
        >>> scaler.add_scaling_rule(rule)
    """
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        
        # Initialize components
        self.metrics_collector = MetricsCollector(self.config)
        self.predictive_scaler = PredictiveScaler(self.config)
        
        # Scaling rules and state
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: deque = deque(maxlen=1000)
        self.last_scaling_actions: Dict[str, float] = defaultdict(float)
        
        # Integration points
        self.distributed_manager = None
        self.resource_providers: Dict[ResourceType, Callable] = {}
        
        # Control state
        self.scaling_active = False
        self.scaling_thread = None
        self.metrics_thread = None
        
        # Setup default rules
        self._setup_default_scaling_rules()
        
        # Create log directory
        if self.config.save_scaling_history:
            Path(self.config.log_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Adaptive scaling manager initialized")
    
    def _setup_default_scaling_rules(self):
        """Setup default scaling rules."""
        # CPU-based worker scaling
        self.scaling_rules["cpu_scale_up"] = ScalingRule(
            name="cpu_scale_up",
            trigger=ScalingTrigger.CPU_UTILIZATION,
            resource_type=ResourceType.WORKERS,
            direction=ScalingDirection.SCALE_UP,
            threshold=self.config.cpu_scale_up_threshold,
            cooldown_seconds=self.config.scale_up_cooldown,
            scale_factor=self.config.scale_up_factor,
            max_instances=self.config.max_workers
        )
        
        self.scaling_rules["cpu_scale_down"] = ScalingRule(
            name="cpu_scale_down",
            trigger=ScalingTrigger.CPU_UTILIZATION,
            resource_type=ResourceType.WORKERS,
            direction=ScalingDirection.SCALE_DOWN,
            threshold=self.config.cpu_scale_down_threshold,
            comparison="less_than",
            cooldown_seconds=self.config.scale_down_cooldown,
            scale_factor=self.config.scale_down_factor,
            min_instances=self.config.min_workers
        )
        
        # Memory-based scaling
        self.scaling_rules["memory_scale_up"] = ScalingRule(
            name="memory_scale_up",
            trigger=ScalingTrigger.MEMORY_UTILIZATION,
            resource_type=ResourceType.WORKERS,
            direction=ScalingDirection.SCALE_UP,
            threshold=self.config.memory_scale_up_threshold,
            cooldown_seconds=self.config.scale_up_cooldown,
            scale_factor=self.config.scale_up_factor
        )
        
        # Queue-based scaling
        self.scaling_rules["queue_scale_up"] = ScalingRule(
            name="queue_scale_up",
            trigger=ScalingTrigger.QUEUE_SIZE,
            resource_type=ResourceType.WORKERS,
            direction=ScalingDirection.SCALE_UP,
            threshold=self.config.queue_scale_up_threshold,
            cooldown_seconds=self.config.scale_up_cooldown / 2,  # More responsive to queue
            priority=2
        )
        
        logger.info(f"Setup {len(self.scaling_rules)} default scaling rules")
    
    def register_distributed_manager(self, manager):
        """Register distributed processing manager."""
        self.distributed_manager = manager
        self.metrics_collector.register_distributed_manager(manager)
        
        # Register worker scaling provider
        self.resource_providers[ResourceType.WORKERS] = self._scale_workers
        
        logger.info("Registered distributed processing manager")
    
    def register_monitoring_system(self, monitor):
        """Register system monitor."""
        self.metrics_collector.register_monitoring_system(monitor)
        logger.info("Registered monitoring system")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules[rule.name] = rule
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str) -> bool:
        """Remove a scaling rule."""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            logger.info(f"Removed scaling rule: {rule_name}")
            return True
        return False
    
    def start_scaling(self):
        """Start the adaptive scaling system."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
        self.metrics_thread.start()
        
        # Start scaling decision thread
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Adaptive scaling started")
    
    def stop_scaling(self):
        """Stop the adaptive scaling system."""
        self.scaling_active = False
        
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5)
        
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5)
        
        logger.info("Adaptive scaling stopped")
    
    def _metrics_loop(self):
        """Background metrics collection loop."""
        while self.scaling_active:
            try:
                metrics = self.metrics_collector.collect_metrics()
                
                # Feed to predictive scaler
                if self.config.enable_predictive_scaling:
                    self.predictive_scaler.learn_from_metrics(metrics)
                
                time.sleep(self.config.metric_collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                time.sleep(10)
    
    def _scaling_loop(self):
        """Background scaling decision loop."""
        while self.scaling_active:
            try:
                if self.config.enable_auto_scaling:
                    self._evaluate_scaling_rules()
                
                if self.config.enable_predictive_scaling:
                    self._evaluate_predictive_scaling()
                
                time.sleep(self.config.scaling_check_interval)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(30)
    
    def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules and trigger actions."""
        try:
            current_metrics = self.metrics_collector.collect_metrics()
            
            # Sort rules by priority (higher first)
            sorted_rules = sorted(
                self.scaling_rules.values(),
                key=lambda r: r.priority,
                reverse=True
            )
            
            for rule in sorted_rules:
                if not rule.enabled:
                    continue
                
                if self._should_trigger_rule(rule, current_metrics):
                    self._execute_scaling_action(rule, current_metrics)
                    break  # Only execute one rule per cycle
            
        except Exception as e:
            logger.error(f"Scaling rule evaluation failed: {e}")
    
    def _should_trigger_rule(self, rule: ScalingRule, metrics: ResourceMetrics) -> bool:
        """Check if scaling rule should be triggered."""
        try:
            # Check cooldown
            current_time = time.time()
            last_action_time = self.last_scaling_actions.get(rule.name, 0)
            
            if current_time - last_action_time < rule.cooldown_seconds:
                return False
            
            # Get metric value
            metric_value = self._get_metric_value(rule.trigger, metrics)
            
            # Check threshold
            if rule.comparison == "greater_than":
                threshold_met = metric_value > rule.threshold
            elif rule.comparison == "less_than":
                threshold_met = metric_value < rule.threshold
            elif rule.comparison == "equals":
                threshold_met = abs(metric_value - rule.threshold) < 0.01
            else:
                threshold_met = False
            
            if not threshold_met:
                return False
            
            # Check capacity constraints
            current_capacity = metrics.current_capacity.get(rule.resource_type.value, 1)
            
            if rule.direction == ScalingDirection.SCALE_UP:
                if current_capacity >= rule.max_instances:
                    return False
            elif rule.direction == ScalingDirection.SCALE_DOWN:
                if current_capacity <= rule.min_instances:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rule trigger check failed for {rule.name}: {e}")
            return False
    
    def _get_metric_value(self, trigger: ScalingTrigger, metrics: ResourceMetrics) -> float:
        """Get metric value for trigger type."""
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            return metrics.cpu_utilization
        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            return metrics.memory_utilization
        elif trigger == ScalingTrigger.QUEUE_SIZE:
            return metrics.queue_size
        elif trigger == ScalingTrigger.RESPONSE_TIME:
            return metrics.avg_response_time
        elif trigger == ScalingTrigger.THROUGHPUT:
            return metrics.throughput
        elif trigger == ScalingTrigger.ERROR_RATE:
            return metrics.error_rate
        else:
            return 0.0
    
    def _execute_scaling_action(self, rule: ScalingRule, metrics: ResourceMetrics):
        """Execute scaling action for rule."""
        start_time = time.time()
        
        scaling_event = ScalingEvent(
            event_id=f"{rule.name}_{int(start_time)}",
            rule_name=rule.name,
            resource_type=rule.resource_type,
            direction=rule.direction,
            trigger_value=self._get_metric_value(rule.trigger, metrics),
            threshold=rule.threshold,
            old_capacity=metrics.current_capacity.get(rule.resource_type.value, 1)
        )
        
        try:
            logger.info(f"Executing scaling action: {rule.name} ({rule.direction.value})")
            
            # Calculate new capacity
            current_capacity = scaling_event.old_capacity
            
            if rule.direction == ScalingDirection.SCALE_UP:
                new_capacity = min(
                    rule.max_instances,
                    int(current_capacity * rule.scale_factor)
                )
                new_capacity = max(new_capacity, current_capacity + 1)  # At least +1
            elif rule.direction == ScalingDirection.SCALE_DOWN:
                new_capacity = max(
                    rule.min_instances,
                    int(current_capacity * rule.scale_factor)
                )
                new_capacity = min(new_capacity, current_capacity - 1)  # At least -1
            else:
                new_capacity = current_capacity
            
            scaling_event.new_capacity = new_capacity
            
            # Execute scaling action
            if rule.resource_type in self.resource_providers:
                provider = self.resource_providers[rule.resource_type]
                success = provider(rule.direction, new_capacity, current_capacity)
                scaling_event.success = success
            else:
                logger.warning(f"No provider for resource type: {rule.resource_type}")
                scaling_event.success = False
                scaling_event.error_message = "No resource provider"
            
            scaling_event.duration = time.time() - start_time
            
            # Update last action time
            self.last_scaling_actions[rule.name] = start_time
            
            # Record scaling event
            self.scaling_history.append(scaling_event)
            
            # Log the event
            if scaling_event.success:
                logger.info(
                    f"Scaling successful: {rule.resource_type.value} "
                    f"{current_capacity} -> {new_capacity}"
                )
            else:
                logger.error(
                    f"Scaling failed: {rule.resource_type.value} "
                    f"{current_capacity} -> {new_capacity}: {scaling_event.error_message}"
                )
            
            # Save to file if configured
            if self.config.save_scaling_history:
                self._save_scaling_event(scaling_event)
            
        except Exception as e:
            scaling_event.success = False
            scaling_event.error_message = str(e)
            scaling_event.duration = time.time() - start_time
            
            logger.error(f"Scaling action failed for rule {rule.name}: {e}")
            self.scaling_history.append(scaling_event)
    
    def _scale_workers(self, direction: ScalingDirection, new_capacity: int, current_capacity: int) -> bool:
        """Scale worker instances."""
        if not self.distributed_manager:
            return False
        
        try:
            if direction == ScalingDirection.SCALE_UP:
                # Add workers
                workers_to_add = new_capacity - current_capacity
                for _ in range(workers_to_add):
                    worker_id = self.distributed_manager._start_worker(
                        self.distributed_manager.WorkerType.GENERAL_WORKER
                    )
                    logger.info(f"Added worker: {worker_id}")
                
            elif direction == ScalingDirection.SCALE_DOWN:
                # Remove workers
                workers_to_remove = current_capacity - new_capacity
                for _ in range(workers_to_remove):
                    self.distributed_manager._stop_excess_worker()
                    logger.info("Removed excess worker")
            
            return True
            
        except Exception as e:
            logger.error(f"Worker scaling failed: {e}")
            return False
    
    def _evaluate_predictive_scaling(self):
        """Evaluate predictive scaling recommendations."""
        try:
            current_metrics = self.metrics_collector.collect_metrics()
            
            recommendations = self.predictive_scaler.get_scaling_recommendation(
                current_metrics,
                self.config.prediction_window_minutes
            )
            
            for resource_type, (direction, recommended_capacity, confidence) in recommendations.items():
                if confidence < 0.5:  # Low confidence, skip
                    continue
                
                # Only act on high-confidence predictions
                if confidence > 0.8 and resource_type in self.resource_providers:
                    current_capacity = current_metrics.current_capacity.get(resource_type.value, 1)
                    
                    # Only make significant changes
                    capacity_change = abs(recommended_capacity - current_capacity) / current_capacity
                    
                    if capacity_change > 0.2:  # 20% change threshold
                        logger.info(
                            f"Predictive scaling recommendation: {resource_type.value} "
                            f"{current_capacity} -> {recommended_capacity:.0f} (confidence: {confidence:.2f})"
                        )
                        
                        # Create synthetic scaling event for logging
                        event = ScalingEvent(
                            event_id=f"predictive_{resource_type.value}_{int(time.time())}",
                            rule_name="predictive_scaling",
                            resource_type=resource_type,
                            direction=direction,
                            old_capacity=current_capacity,
                            new_capacity=int(recommended_capacity),
                            metadata={"confidence": confidence, "type": "predictive"}
                        )
                        
                        provider = self.resource_providers[resource_type]
                        event.success = provider(direction, int(recommended_capacity), current_capacity)
                        
                        self.scaling_history.append(event)
                        
        except Exception as e:
            logger.error(f"Predictive scaling evaluation failed: {e}")
    
    def _save_scaling_event(self, event: ScalingEvent):
        """Save scaling event to file."""
        try:
            log_file = Path(self.config.log_directory) / f"scaling_events_{time.strftime('%Y%m%d')}.jsonl"
            
            with open(log_file, 'a') as f:
                event_data = {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "rule_name": event.rule_name,
                    "resource_type": event.resource_type.value,
                    "direction": event.direction.value,
                    "trigger_value": event.trigger_value,
                    "threshold": event.threshold,
                    "old_capacity": event.old_capacity,
                    "new_capacity": event.new_capacity,
                    "success": event.success,
                    "duration": event.duration,
                    "error_message": event.error_message,
                    "metadata": event.metadata
                }
                f.write(json.dumps(event_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save scaling event: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        try:
            current_metrics = self.metrics_collector.collect_metrics()
            metrics_summary = self.metrics_collector.get_metrics_summary(60.0)
            
            # Recent scaling events
            recent_events = [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "rule_name": event.rule_name,
                    "resource_type": event.resource_type.value,
                    "direction": event.direction.value,
                    "old_capacity": event.old_capacity,
                    "new_capacity": event.new_capacity,
                    "success": event.success
                }
                for event in list(self.scaling_history)[-10:]  # Last 10 events
            ]
            
            # Rule status
            rule_status = {
                name: {
                    "enabled": rule.enabled,
                    "trigger": rule.trigger.value,
                    "threshold": rule.threshold,
                    "last_action": self.last_scaling_actions.get(name, 0),
                    "cooldown_remaining": max(0, rule.cooldown_seconds - (time.time() - self.last_scaling_actions.get(name, 0)))
                }
                for name, rule in self.scaling_rules.items()
            }
            
            return {
                "timestamp": time.time(),
                "scaling_active": self.scaling_active,
                "current_metrics": {
                    "cpu_utilization": current_metrics.cpu_utilization,
                    "memory_utilization": current_metrics.memory_utilization,
                    "active_workers": current_metrics.active_workers,
                    "queue_size": current_metrics.queue_size,
                    "throughput": current_metrics.throughput
                },
                "metrics_summary": metrics_summary,
                "scaling_rules": rule_status,
                "recent_scaling_events": recent_events,
                "total_scaling_events": len(self.scaling_history),
                "predictive_scaling_enabled": self.config.enable_predictive_scaling
            }
            
        except Exception as e:
            logger.error(f"Failed to get scaling status: {e}")
            return {"error": str(e)}
    
    def force_scaling_action(
        self,
        resource_type: ResourceType,
        direction: ScalingDirection,
        target_capacity: int
    ) -> bool:
        """Force a scaling action (bypass rules and cooldowns)."""
        try:
            current_metrics = self.metrics_collector.collect_metrics()
            current_capacity = current_metrics.current_capacity.get(resource_type.value, 1)
            
            logger.info(f"Force scaling: {resource_type.value} {current_capacity} -> {target_capacity}")
            
            if resource_type in self.resource_providers:
                provider = self.resource_providers[resource_type]
                success = provider(direction, target_capacity, current_capacity)
                
                # Record forced scaling event
                event = ScalingEvent(
                    event_id=f"forced_{resource_type.value}_{int(time.time())}",
                    rule_name="manual_override",
                    resource_type=resource_type,
                    direction=direction,
                    old_capacity=current_capacity,
                    new_capacity=target_capacity,
                    success=success,
                    metadata={"type": "forced"}
                )
                
                self.scaling_history.append(event)
                
                return success
            
        except Exception as e:
            logger.error(f"Forced scaling action failed: {e}")
        
        return False


# Convenience functions
def create_adaptive_scaler(config: Optional[ScalingConfig] = None) -> AdaptiveScalingManager:
    """Create and configure adaptive scaling manager."""
    return AdaptiveScalingManager(config)


def setup_auto_scaling(
    distributed_manager,
    monitoring_system=None,
    min_workers: int = 2,
    max_workers: int = 10
) -> AdaptiveScalingManager:
    """
    Setup auto-scaling for distributed processing system.
    
    Args:
        distributed_manager: Distributed processing manager
        monitoring_system: Optional monitoring system
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers
        
    Returns:
        Configured adaptive scaling manager
    """
    config = ScalingConfig(
        min_workers=min_workers,
        max_workers=max_workers,
        enable_auto_scaling=True,
        enable_predictive_scaling=True
    )
    
    scaler = AdaptiveScalingManager(config)
    scaler.register_distributed_manager(distributed_manager)
    
    if monitoring_system:
        scaler.register_monitoring_system(monitoring_system)
    
    scaler.start_scaling()
    
    logger.info(f"Auto-scaling setup complete: {min_workers}-{max_workers} workers")
    return scaler
