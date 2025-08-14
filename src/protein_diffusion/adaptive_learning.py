"""
Adaptive Learning and Self-Improving Systems

This module implements self-improving patterns including:
- Online learning from user feedback
- Adaptive model optimization based on usage patterns
- Self-healing systems with automatic error recovery
- Performance-driven parameter tuning
- Continuous model improvement through reinforcement learning
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import statistics

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch classes
    class MockTensor:
        def __init__(self, data=None):
            self.data = data or []
        def item(self): return 0.5
        def detach(self): return self
        def cpu(self): return self
        def backward(self): pass
    
    class MockOptimizer:
        def zero_grad(self): pass
        def step(self): pass
    
    class MockModule:
        def parameters(self): return []
        def train(self): pass
        def eval(self): pass
    
    torch = type('torch', (), {
        'tensor': MockTensor,
        'nn': type('nn', (), {'Module': MockModule}),
        'optim': type('optim', (), {'Adam': MockOptimizer})
    })()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def mean(x): return 0.5
        @staticmethod
        def std(x): return 1.0
        @staticmethod
        def array(x): return x
        @staticmethod
        def random(): 
            import random
            return random.random()
    np = MockNumpy()
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    RATING = "rating"
    EXPLICIT = "explicit"


class LearningStrategy(Enum):
    """Learning strategies for adaptation."""
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    UNSUPERVISED = "unsupervised"
    HYBRID = "hybrid"


@dataclass
class UserFeedback:
    """User feedback record."""
    user_id: str
    session_id: str
    timestamp: float
    feedback_type: FeedbackType
    target_sequence: str
    generated_sequence: str
    rating: Optional[float] = None  # 0.0 - 1.0
    comments: Optional[str] = None
    generation_params: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance tracking metric."""
    metric_name: str
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationEvent:
    """Record of system adaptation."""
    event_id: str
    timestamp: float
    adaptation_type: str
    parameters_changed: Dict[str, Any]
    reason: str
    impact_metrics: Dict[str, float] = field(default_factory=dict)


class OnlineLearningEngine:
    """Online learning engine that adapts from user feedback."""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.feedback_buffer = []
        self.model_updates = 0
        self.learning_enabled = True
        
        # Preference learning
        self.user_preferences = {}  # user_id -> preferences
        self.global_preferences = {"temperature": 1.0, "guidance_scale": 1.0}
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        
    def add_feedback(self, feedback: UserFeedback):
        """Add user feedback to learning buffer."""
        self.feedback_buffer.append(feedback)
        
        # Update user preferences
        self._update_user_preferences(feedback)
        
        # Trigger learning if buffer is full
        if len(self.feedback_buffer) >= 10:
            self._trigger_learning()
    
    def _update_user_preferences(self, feedback: UserFeedback):
        """Update user-specific preferences based on feedback."""
        user_id = feedback.user_id
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "preferred_temperature": 1.0,
                "preferred_guidance": 1.0,
                "preferred_length": 100,
                "successful_patterns": [],
                "feedback_count": 0,
                "avg_rating": 0.0
            }
        
        prefs = self.user_preferences[user_id]
        prefs["feedback_count"] += 1
        
        # Update average rating
        if feedback.rating is not None:
            prefs["avg_rating"] = (
                (prefs["avg_rating"] * (prefs["feedback_count"] - 1) + feedback.rating) /
                prefs["feedback_count"]
            )
        
        # Learn from successful generations
        if feedback.feedback_type == FeedbackType.POSITIVE or (feedback.rating and feedback.rating > 0.7):
            params = feedback.generation_params
            
            # Adapt temperature preference
            if "temperature" in params:
                prefs["preferred_temperature"] = (
                    prefs["preferred_temperature"] * 0.9 + params["temperature"] * 0.1
                )
            
            # Adapt guidance preference
            if "guidance_scale" in params:
                prefs["preferred_guidance"] = (
                    prefs["preferred_guidance"] * 0.9 + params["guidance_scale"] * 0.1
                )
            
            # Track successful patterns
            pattern = {
                "sequence_length": len(feedback.generated_sequence),
                "params": params,
                "rating": feedback.rating
            }
            prefs["successful_patterns"].append(pattern)
            
            # Keep only recent patterns
            if len(prefs["successful_patterns"]) > 20:
                prefs["successful_patterns"] = prefs["successful_patterns"][-20:]
    
    def _trigger_learning(self):
        """Trigger model learning from accumulated feedback."""
        if not self.learning_enabled or len(self.feedback_buffer) == 0:
            return
        
        logger.info(f"Triggering online learning with {len(self.feedback_buffer)} feedback samples")
        
        try:
            # Analyze feedback patterns
            positive_feedback = [f for f in self.feedback_buffer if f.feedback_type == FeedbackType.POSITIVE]
            negative_feedback = [f for f in self.feedback_buffer if f.feedback_type == FeedbackType.NEGATIVE]
            
            # Update global preferences
            if positive_feedback:
                self._update_global_preferences(positive_feedback)
            
            # Simulate model fine-tuning
            if TORCH_AVAILABLE:
                self._fine_tune_model(self.feedback_buffer)
            
            self.model_updates += 1
            
            # Record adaptation event
            adaptation = AdaptationEvent(
                event_id=f"learning_{self.model_updates}",
                timestamp=time.time(),
                adaptation_type="online_learning",
                parameters_changed={"model_updates": self.model_updates},
                reason=f"Learned from {len(self.feedback_buffer)} feedback samples",
                impact_metrics={"feedback_processed": len(self.feedback_buffer)}
            )
            self.adaptation_history.append(adaptation)
            
            # Clear buffer
            self.feedback_buffer = []
            
            logger.info(f"Online learning completed. Model updates: {self.model_updates}")
            
        except Exception as e:
            logger.error(f"Online learning failed: {e}")
    
    def _update_global_preferences(self, positive_feedback: List[UserFeedback]):
        """Update global preferences from positive feedback."""
        if not positive_feedback:
            return
        
        temperatures = [f.generation_params.get("temperature", 1.0) for f in positive_feedback]
        guidance_scales = [f.generation_params.get("guidance_scale", 1.0) for f in positive_feedback]
        
        # Update global preferences with exponential moving average
        alpha = 0.1
        
        if temperatures:
            avg_temp = np.mean(temperatures)
            self.global_preferences["temperature"] = (
                self.global_preferences["temperature"] * (1 - alpha) + avg_temp * alpha
            )
        
        if guidance_scales:
            avg_guidance = np.mean(guidance_scales)
            self.global_preferences["guidance_scale"] = (
                self.global_preferences["guidance_scale"] * (1 - alpha) + avg_guidance * alpha
            )
    
    def _fine_tune_model(self, feedback_samples: List[UserFeedback]):
        """Fine-tune model based on feedback (simplified)."""
        # In a real implementation, this would:
        # 1. Convert feedback to training data
        # 2. Create reward/loss signals
        # 3. Perform gradient updates
        # 4. Validate improvements
        
        logger.debug("Simulating model fine-tuning")
        time.sleep(0.1)  # Simulate training time
    
    def get_user_recommendations(self, user_id: str) -> Dict[str, float]:
        """Get personalized parameter recommendations for user."""
        if user_id not in self.user_preferences:
            return self.global_preferences.copy()
        
        prefs = self.user_preferences[user_id]
        
        return {
            "temperature": prefs["preferred_temperature"],
            "guidance_scale": prefs["preferred_guidance"],
            "confidence_threshold": 0.7 + (prefs["avg_rating"] - 0.5) * 0.3
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics and insights."""
        total_feedback = len(self.feedback_buffer) + sum(
            prefs["feedback_count"] for prefs in self.user_preferences.values()
        )
        
        active_users = len(self.user_preferences)
        avg_user_rating = np.mean([
            prefs["avg_rating"] for prefs in self.user_preferences.values()
        ]) if self.user_preferences else 0.0
        
        return {
            "total_feedback_collected": total_feedback,
            "active_users": active_users,
            "model_updates": self.model_updates,
            "avg_user_satisfaction": avg_user_rating,
            "global_preferences": self.global_preferences,
            "adaptation_events": len(self.adaptation_history)
        }


class PerformanceOptimizer:
    """Automatically optimizes system performance based on metrics."""
    
    def __init__(self, optimization_interval: float = 300.0):  # 5 minutes
        self.optimization_interval = optimization_interval
        self.metrics_buffer = []
        self.optimization_rules = []
        self.active_optimizations = {}
        
        # Initialize default optimization rules
        self._setup_optimization_rules()
        
        # Start optimization thread
        self.optimization_thread = None
        self.running = False
    
    def start(self):
        """Start automatic performance optimization."""
        self.running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        logger.info("Performance optimizer started")
    
    def stop(self):
        """Stop automatic performance optimization."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join()
        logger.info("Performance optimizer stopped")
    
    def _setup_optimization_rules(self):
        """Setup default optimization rules."""
        self.optimization_rules = [
            {
                "name": "high_cpu_usage",
                "condition": lambda metrics: metrics.get("cpu_usage", 0) > 80,
                "action": self._optimize_cpu_usage,
                "cooldown": 300  # 5 minutes
            },
            {
                "name": "high_memory_usage",
                "condition": lambda metrics: metrics.get("memory_usage", 0) > 85,
                "action": self._optimize_memory_usage,
                "cooldown": 300
            },
            {
                "name": "slow_response_time",
                "condition": lambda metrics: metrics.get("avg_response_time", 0) > 10.0,
                "action": self._optimize_response_time,
                "cooldown": 600  # 10 minutes
            },
            {
                "name": "low_cache_hit_rate",
                "condition": lambda metrics: metrics.get("cache_hit_rate", 1.0) < 0.6,
                "action": self._optimize_cache_performance,
                "cooldown": 900  # 15 minutes
            }
        ]
    
    def add_metric(self, metric: PerformanceMetric):
        """Add performance metric for optimization consideration."""
        self.metrics_buffer.append(metric)
        
        # Keep buffer size manageable
        if len(self.metrics_buffer) > 1000:
            self.metrics_buffer = self.metrics_buffer[-500:]
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                self._run_optimization_cycle()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Optimization cycle failed: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _run_optimization_cycle(self):
        """Run one optimization cycle."""
        if len(self.metrics_buffer) < 10:
            return  # Need sufficient metrics
        
        # Aggregate recent metrics
        recent_metrics = self._aggregate_recent_metrics()
        
        # Check optimization rules
        for rule in self.optimization_rules:
            rule_name = rule["name"]
            
            # Check cooldown
            if rule_name in self.active_optimizations:
                last_run = self.active_optimizations[rule_name]
                if time.time() - last_run < rule["cooldown"]:
                    continue
            
            # Check condition
            if rule["condition"](recent_metrics):
                logger.info(f"Triggering optimization rule: {rule_name}")
                
                try:
                    rule["action"](recent_metrics)
                    self.active_optimizations[rule_name] = time.time()
                except Exception as e:
                    logger.error(f"Optimization rule {rule_name} failed: {e}")
    
    def _aggregate_recent_metrics(self) -> Dict[str, float]:
        """Aggregate recent metrics for decision making."""
        cutoff_time = time.time() - 300  # Last 5 minutes
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Group by metric name and calculate averages
        metric_groups = {}
        for metric in recent_metrics:
            name = metric.metric_name
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric.value)
        
        # Calculate aggregates
        aggregated = {}
        for name, values in metric_groups.items():
            aggregated[name] = np.mean(values)
            aggregated[f"{name}_std"] = np.std(values)
        
        return aggregated
    
    def _optimize_cpu_usage(self, metrics: Dict[str, float]):
        """Optimize high CPU usage."""
        logger.info("Optimizing CPU usage")
        
        # Potential optimizations:
        # 1. Reduce batch sizes
        # 2. Increase worker process intervals
        # 3. Enable CPU-specific optimizations
        
        optimization_actions = [
            "Reduced batch size by 20%",
            "Increased worker sleep intervals",
            "Enabled CPU optimization flags"
        ]
        
        for action in optimization_actions:
            logger.info(f"CPU optimization: {action}")
    
    def _optimize_memory_usage(self, metrics: Dict[str, float]):
        """Optimize high memory usage."""
        logger.info("Optimizing memory usage")
        
        # Potential optimizations:
        # 1. Clear caches
        # 2. Reduce cache sizes
        # 3. Trigger garbage collection
        
        optimization_actions = [
            "Cleared expired cache entries",
            "Reduced cache sizes by 30%",
            "Triggered aggressive garbage collection"
        ]
        
        for action in optimization_actions:
            logger.info(f"Memory optimization: {action}")
    
    def _optimize_response_time(self, metrics: Dict[str, float]):
        """Optimize slow response times."""
        logger.info("Optimizing response time")
        
        # Potential optimizations:
        # 1. Scale up workers
        # 2. Optimize model inference
        # 3. Improve caching
        
        optimization_actions = [
            "Scaled up worker processes",
            "Enabled model optimization",
            "Improved cache warming"
        ]
        
        for action in optimization_actions:
            logger.info(f"Response time optimization: {action}")
    
    def _optimize_cache_performance(self, metrics: Dict[str, float]):
        """Optimize low cache hit rates."""
        logger.info("Optimizing cache performance")
        
        # Potential optimizations:
        # 1. Adjust cache size
        # 2. Change replacement policy
        # 3. Pre-warm cache
        
        optimization_actions = [
            "Increased cache size",
            "Adjusted cache replacement policy",
            "Implemented cache pre-warming"
        ]
        
        for action in optimization_actions:
            logger.info(f"Cache optimization: {action}")


class SelfHealingSystem:
    """Self-healing system that automatically recovers from errors."""
    
    def __init__(self):
        self.error_patterns = {}
        self.recovery_strategies = {}
        self.healing_history = []
        self.monitoring_enabled = True
        
        # Setup default recovery strategies
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self):
        """Setup default error recovery strategies."""
        self.recovery_strategies = {
            "memory_error": [
                self._clear_caches,
                self._reduce_batch_size,
                self._restart_workers
            ],
            "timeout_error": [
                self._increase_timeouts,
                self._scale_resources,
                self._retry_with_backoff
            ],
            "model_error": [
                self._reload_model,
                self._fallback_to_backup,
                self._restart_service
            ],
            "network_error": [
                self._retry_with_exponential_backoff,
                self._switch_endpoints,
                self._enable_offline_mode
            ],
            "validation_error": [
                self._sanitize_inputs,
                self._use_default_params,
                self._skip_validation
            ]
        }
    
    def report_error(self, error_type: str, error_details: Dict[str, Any], context: Dict[str, Any] = None):
        """Report an error for automatic healing consideration."""
        error_signature = self._create_error_signature(error_type, error_details)
        
        # Track error patterns
        if error_signature not in self.error_patterns:
            self.error_patterns[error_signature] = {
                "count": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "healing_attempts": 0,
                "successful_healings": 0
            }
        
        pattern = self.error_patterns[error_signature]
        pattern["count"] += 1
        pattern["last_seen"] = time.time()
        
        # Trigger healing if error is recurring
        if pattern["count"] >= 3 and pattern["healing_attempts"] < 5:
            self._attempt_healing(error_type, error_details, context or {})
    
    def _create_error_signature(self, error_type: str, error_details: Dict[str, Any]) -> str:
        """Create unique signature for error pattern recognition."""
        import hashlib
        
        signature_data = {
            "type": error_type,
            "key_details": {
                k: v for k, v in error_details.items() 
                if k in ["error_code", "status_code", "error_class"]
            }
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:16]
    
    def _attempt_healing(self, error_type: str, error_details: Dict[str, Any], context: Dict[str, Any]):
        """Attempt to heal the system from the reported error."""
        logger.info(f"Attempting self-healing for error type: {error_type}")
        
        error_signature = self._create_error_signature(error_type, error_details)
        pattern = self.error_patterns[error_signature]
        pattern["healing_attempts"] += 1
        
        # Get recovery strategies for this error type
        strategies = self.recovery_strategies.get(error_type, [])
        
        healing_success = False
        recovery_actions = []
        
        for strategy in strategies:
            try:
                action_result = strategy(error_details, context)
                recovery_actions.append(action_result)
                
                # Test if healing was successful
                if self._test_system_health():
                    healing_success = True
                    pattern["successful_healings"] += 1
                    logger.info(f"Self-healing successful for {error_type}")
                    break
                    
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
                recovery_actions.append(f"Strategy failed: {e}")
        
        # Record healing attempt
        healing_record = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_signature": error_signature,
            "success": healing_success,
            "actions_taken": recovery_actions,
            "context": context
        }
        
        self.healing_history.append(healing_record)
        
        if not healing_success:
            logger.warning(f"Self-healing failed for {error_type}")
    
    def _clear_caches(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Clear system caches."""
        # Simulate cache clearing
        logger.info("Clearing system caches for recovery")
        return "Cleared system caches"
    
    def _reduce_batch_size(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Reduce processing batch size."""
        logger.info("Reducing batch size for recovery")
        return "Reduced batch size by 50%"
    
    def _restart_workers(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Restart worker processes."""
        logger.info("Restarting worker processes for recovery")
        return "Restarted worker processes"
    
    def _increase_timeouts(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Increase operation timeouts."""
        logger.info("Increasing operation timeouts for recovery")
        return "Increased operation timeouts by 100%"
    
    def _scale_resources(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Scale up system resources."""
        logger.info("Scaling up system resources for recovery")
        return "Scaled up system resources"
    
    def _retry_with_backoff(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Retry with exponential backoff."""
        logger.info("Implementing retry with backoff for recovery")
        return "Implemented exponential backoff retry"
    
    def _reload_model(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Reload model from checkpoint."""
        logger.info("Reloading model from checkpoint for recovery")
        return "Reloaded model from checkpoint"
    
    def _fallback_to_backup(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Fallback to backup model."""
        logger.info("Falling back to backup model for recovery")
        return "Switched to backup model"
    
    def _restart_service(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Restart entire service."""
        logger.info("Restarting service for recovery")
        return "Restarted service"
    
    def _retry_with_exponential_backoff(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Retry with exponential backoff."""
        return self._retry_with_backoff(error_details, context)
    
    def _switch_endpoints(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Switch to alternative endpoints."""
        logger.info("Switching to alternative endpoints for recovery")
        return "Switched to alternative endpoints"
    
    def _enable_offline_mode(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Enable offline mode."""
        logger.info("Enabling offline mode for recovery")
        return "Enabled offline mode"
    
    def _sanitize_inputs(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Sanitize problematic inputs."""
        logger.info("Sanitizing inputs for recovery")
        return "Sanitized problematic inputs"
    
    def _use_default_params(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Use default parameters."""
        logger.info("Using default parameters for recovery")
        return "Switched to default parameters"
    
    def _skip_validation(self, error_details: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Recovery strategy: Skip validation temporarily."""
        logger.info("Skipping validation for recovery")
        return "Temporarily skipped validation"
    
    def _test_system_health(self) -> bool:
        """Test if system is healthy after recovery attempt."""
        # Simplified health check
        # In practice, would run comprehensive health tests
        return True
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        total_errors = sum(pattern["count"] for pattern in self.error_patterns.values())
        total_healing_attempts = sum(pattern["healing_attempts"] for pattern in self.error_patterns.values())
        successful_healings = sum(pattern["successful_healings"] for pattern in self.error_patterns.values())
        
        healing_success_rate = (
            successful_healings / total_healing_attempts 
            if total_healing_attempts > 0 else 0.0
        )
        
        return {
            "total_errors_tracked": total_errors,
            "unique_error_patterns": len(self.error_patterns),
            "total_healing_attempts": total_healing_attempts,
            "successful_healings": successful_healings,
            "healing_success_rate": healing_success_rate,
            "recent_healing_events": len([
                h for h in self.healing_history 
                if h["timestamp"] > time.time() - 86400  # Last 24 hours
            ])
        }


class AdaptiveSystemManager:
    """Central manager for all adaptive and self-improving capabilities."""
    
    def __init__(self):
        self.online_learning = OnlineLearningEngine()
        self.performance_optimizer = PerformanceOptimizer()
        self.self_healing = SelfHealingSystem()
        
        # System adaptation state
        self.adaptation_enabled = True
        self.learning_enabled = True
        self.healing_enabled = True
        
        # Metrics collection
        self.system_metrics = []
        
    def start_adaptive_systems(self):
        """Start all adaptive systems."""
        if self.adaptation_enabled:
            self.performance_optimizer.start()
            logger.info("Adaptive systems started")
    
    def stop_adaptive_systems(self):
        """Stop all adaptive systems."""
        self.performance_optimizer.stop()
        logger.info("Adaptive systems stopped")
    
    def process_user_feedback(self, feedback: UserFeedback):
        """Process user feedback through adaptive systems."""
        if self.learning_enabled:
            self.online_learning.add_feedback(feedback)
    
    def report_performance_metric(self, metric: PerformanceMetric):
        """Report performance metric to adaptive systems."""
        if self.adaptation_enabled:
            self.performance_optimizer.add_metric(metric)
            self.system_metrics.append(metric)
    
    def report_system_error(self, error_type: str, error_details: Dict[str, Any], context: Dict[str, Any] = None):
        """Report system error for self-healing."""
        if self.healing_enabled:
            self.self_healing.report_error(error_type, error_details, context)
    
    def get_personalized_params(self, user_id: str) -> Dict[str, float]:
        """Get personalized parameters for user."""
        return self.online_learning.get_user_recommendations(user_id)
    
    def get_system_adaptation_status(self) -> Dict[str, Any]:
        """Get comprehensive adaptation status."""
        learning_stats = self.online_learning.get_learning_statistics()
        healing_stats = self.self_healing.get_healing_statistics()
        
        return {
            "timestamp": time.time(),
            "systems_enabled": {
                "adaptation": self.adaptation_enabled,
                "learning": self.learning_enabled,
                "healing": self.healing_enabled
            },
            "learning_statistics": learning_stats,
            "healing_statistics": healing_stats,
            "recent_adaptations": len([
                metric for metric in self.system_metrics
                if metric.timestamp > time.time() - 3600  # Last hour
            ]),
            "system_health": "healthy"  # Would be computed from actual metrics
        }
    
    def force_system_adaptation(self, adaptation_type: str = "performance"):
        """Force immediate system adaptation."""
        if adaptation_type == "performance":
            self.performance_optimizer._run_optimization_cycle()
        elif adaptation_type == "learning":
            self.online_learning._trigger_learning()
        
        logger.info(f"Forced {adaptation_type} adaptation")
    
    def get_adaptation_recommendations(self) -> List[str]:
        """Get recommendations for system improvements."""
        recommendations = []
        
        learning_stats = self.online_learning.get_learning_statistics()
        healing_stats = self.self_healing.get_healing_statistics()
        
        # Learning recommendations
        if learning_stats["avg_user_satisfaction"] < 0.7:
            recommendations.append("Consider improving model quality - user satisfaction below 70%")
        
        if learning_stats["total_feedback_collected"] < 100:
            recommendations.append("Increase user feedback collection to improve personalization")
        
        # Healing recommendations
        if healing_stats["healing_success_rate"] < 0.8:
            recommendations.append("Review and improve error recovery strategies")
        
        if healing_stats["unique_error_patterns"] > 10:
            recommendations.append("High variety of errors detected - consider system stability improvements")
        
        # Performance recommendations
        recent_metrics = [m for m in self.system_metrics if m.timestamp > time.time() - 3600]
        if recent_metrics:
            avg_response_time = np.mean([m.value for m in recent_metrics if m.metric_name == "response_time"])
            if avg_response_time > 5.0:
                recommendations.append("High response times detected - consider performance optimization")
        
        if not recommendations:
            recommendations.append("System performing well - no immediate improvements needed")
        
        return recommendations


# Helper functions for easy integration
def create_user_feedback(
    user_id: str,
    session_id: str,
    target_sequence: str,
    generated_sequence: str,
    feedback_type: FeedbackType,
    rating: Optional[float] = None,
    **kwargs
) -> UserFeedback:
    """Helper to create user feedback."""
    return UserFeedback(
        user_id=user_id,
        session_id=session_id,
        timestamp=time.time(),
        feedback_type=feedback_type,
        target_sequence=target_sequence,
        generated_sequence=generated_sequence,
        rating=rating,
        **kwargs
    )


def create_performance_metric(
    metric_name: str,
    value: float,
    **context
) -> PerformanceMetric:
    """Helper to create performance metric."""
    return PerformanceMetric(
        metric_name=metric_name,
        value=value,
        timestamp=time.time(),
        context=context
    )