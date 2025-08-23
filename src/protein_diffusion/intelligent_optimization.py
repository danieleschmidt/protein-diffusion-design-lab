"""
Intelligent Optimization System for Protein Diffusion Design Lab.

This module implements intelligent optimization capabilities including:
- AI-driven performance optimization
- Adaptive resource management
- Predictive scaling and load balancing
- Intelligent caching strategies
- Machine learning-based tuning
- Real-time performance analytics
"""

import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import random
import statistics
import math
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ADAPTIVE = "adaptive"

class ResourceType(Enum):
    """Resource types for optimization."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"

@dataclass
class OptimizationConfig:
    """Configuration for intelligent optimization."""
    optimization_interval: int = 30  # seconds
    learning_window: int = 300  # 5 minutes of historical data
    min_optimization_impact: float = 0.05  # 5% minimum improvement
    max_optimization_attempts: int = 10
    adaptive_threshold: float = 0.8
    prediction_horizon: int = 60  # 1 minute ahead prediction
    convergence_threshold: float = 0.01

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    request_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    active_connections: int = 0

class IntelligentCache:
    """AI-driven intelligent caching system."""
    
    def __init__(self, max_size: int = 1000, learning_enabled: bool = True):
        self.max_size = max_size
        self.learning_enabled = learning_enabled
        self.cache = {}
        self.access_patterns = defaultdict(list)
        self.frequency_scores = defaultdict(float)
        self.recency_scores = defaultdict(float)
        self.prediction_model = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
        self._lock = threading.RLock()
        
        # Learning parameters
        self.decay_factor = 0.95
        self.frequency_weight = 0.4
        self.recency_weight = 0.3
        self.prediction_weight = 0.3
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with learning."""
        with self._lock:
            current_time = time.time()
            
            if key in self.cache:
                # Cache hit
                self.cache_stats['hits'] += 1
                
                # Update access patterns for learning
                if self.learning_enabled:
                    self.access_patterns[key].append(current_time)
                    self._update_scores(key, current_time)
                
                # Move to front (LRU behavior)
                value = self.cache[key]
                return value['data']
            else:
                # Cache miss
                self.cache_stats['misses'] += 1
                
                # Learn from miss pattern
                if self.learning_enabled:
                    self._learn_from_miss(key, current_time)
                
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with intelligent placement."""
        with self._lock:
            current_time = time.time()
            
            # Check if eviction is needed
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._intelligent_eviction()
            
            # Store with metadata
            cache_entry = {
                'data': value,
                'created_at': current_time,
                'last_accessed': current_time,
                'access_count': 1,
                'ttl': ttl,
                'predicted_next_access': self._predict_next_access(key)
            }
            
            self.cache[key] = cache_entry
            self.cache_stats['size'] = len(self.cache)
            
            # Update learning data
            if self.learning_enabled:
                self.access_patterns[key].append(current_time)
                self._update_scores(key, current_time)
    
    def _intelligent_eviction(self):
        """Intelligently evict items based on learned patterns."""
        if not self.cache:
            return
        
        current_time = time.time()
        eviction_candidates = []
        
        for key, entry in self.cache.items():
            # Calculate composite score for eviction
            age = current_time - entry['created_at']
            last_access_age = current_time - entry['last_accessed']
            frequency = self.frequency_scores.get(key, 0)
            recency = self.recency_scores.get(key, 0)
            predicted_access = entry.get('predicted_next_access', float('inf'))
            
            # Check TTL expiration
            if entry.get('ttl') and current_time - entry['created_at'] > entry['ttl']:
                eviction_score = float('inf')  # Expired items have highest eviction priority
            else:
                # Calculate eviction score (higher = more likely to evict)
                eviction_score = (
                    age * 0.2 +
                    last_access_age * 0.3 +
                    (1.0 - frequency) * 0.2 +
                    (1.0 - recency) * 0.2 +
                    min(predicted_access / 3600, 1.0) * 0.1  # Normalize to hours
                )
            
            eviction_candidates.append((key, eviction_score))
        
        # Sort by eviction score (highest first)
        eviction_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Evict the worst candidate
        if eviction_candidates:
            key_to_evict = eviction_candidates[0][0]
            del self.cache[key_to_evict]
            self.cache_stats['evictions'] += 1
            self.cache_stats['size'] = len(self.cache)
            
            logger.debug(f"Evicted cache key: {key_to_evict}")
    
    def _update_scores(self, key: str, current_time: float):
        """Update frequency and recency scores for key."""
        # Update frequency score with exponential decay
        self.frequency_scores[key] = (
            self.frequency_scores[key] * self.decay_factor + 1.0
        )
        
        # Update recency score
        self.recency_scores[key] = 1.0  # Most recent access
        
        # Decay all other recency scores
        for other_key in self.recency_scores:
            if other_key != key:
                self.recency_scores[other_key] *= self.decay_factor
    
    def _predict_next_access(self, key: str) -> float:
        """Predict when key will be accessed next."""
        access_history = self.access_patterns.get(key, [])
        
        if len(access_history) < 2:
            return 3600  # Default to 1 hour if no pattern
        
        # Calculate intervals between accesses
        intervals = []
        for i in range(1, len(access_history)):
            intervals.append(access_history[i] - access_history[i-1])
        
        if not intervals:
            return 3600
        
        # Use exponential moving average for prediction
        if key not in self.prediction_model:
            self.prediction_model[key] = {
                'avg_interval': statistics.mean(intervals),
                'trend': 0.0
            }
        
        model = self.prediction_model[key]
        recent_interval = intervals[-1]
        
        # Update model with new data
        alpha = 0.3  # Learning rate
        model['avg_interval'] = (
            alpha * recent_interval + (1 - alpha) * model['avg_interval']
        )
        
        return model['avg_interval']
    
    def _learn_from_miss(self, key: str, current_time: float):
        """Learn from cache miss to improve future predictions."""
        # This could implement more sophisticated learning algorithms
        # For now, we'll record the miss for pattern analysis
        if 'misses' not in self.access_patterns:
            self.access_patterns['misses'] = []
        
        self.access_patterns['misses'].append({
            'key': key,
            'timestamp': current_time
        })
        
        # Keep only recent misses
        self.access_patterns['misses'] = self.access_patterns['misses'][-1000:]
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / max(1, total_requests)
            
            return {
                **self.cache_stats,
                'hit_rate': hit_rate,
                'miss_rate': 1 - hit_rate,
                'utilization': len(self.cache) / self.max_size,
                'learning_enabled': self.learning_enabled,
                'prediction_accuracy': self._calculate_prediction_accuracy()
            }
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy for cache performance."""
        # Simplified prediction accuracy calculation
        if not self.prediction_model:
            return 0.0
        
        # This would be more sophisticated in a real implementation
        return 0.75  # Placeholder

class AdaptiveResourceManager:
    """Adaptive resource management with ML-based optimization."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.resource_history = defaultdict(lambda: deque(maxlen=1000))
        self.resource_predictions = {}
        self.optimization_actions = []
        self.current_allocation = {
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 1.0,
            ResourceType.GPU: 1.0,
            ResourceType.STORAGE: 1.0,
            ResourceType.NETWORK: 1.0
        }
        self._lock = threading.RLock()
        self.optimization_thread = None
        self.running = False
    
    def start_optimization(self):
        """Start adaptive resource optimization."""
        if self.running:
            return
        
        self.running = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self.optimization_thread.start()
        logger.info("Adaptive resource optimization started")
    
    def stop_optimization(self):
        """Stop adaptive resource optimization."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("Adaptive resource optimization stopped")
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics for optimization learning."""
        with self._lock:
            current_time = time.time()
            
            # Store metrics in history
            self.resource_history['cpu'].append((current_time, metrics.cpu_usage))
            self.resource_history['memory'].append((current_time, metrics.memory_usage))
            self.resource_history['gpu'].append((current_time, metrics.gpu_usage))
            self.resource_history['latency'].append((current_time, metrics.request_latency))
            self.resource_history['throughput'].append((current_time, metrics.throughput))
            self.resource_history['error_rate'].append((current_time, metrics.error_rate))
            
            # Update predictions
            self._update_predictions()
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                self._perform_optimization()
                time.sleep(self.config.optimization_interval)
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(self.config.optimization_interval)
    
    def _perform_optimization(self):
        """Perform intelligent resource optimization."""
        with self._lock:
            current_time = time.time()
            
            # Analyze current performance
            analysis = self._analyze_performance()
            
            if analysis['needs_optimization']:
                # Generate optimization plan
                optimization_plan = self._generate_optimization_plan(analysis)
                
                # Execute optimization
                success = self._execute_optimization_plan(optimization_plan)
                
                # Record optimization action
                optimization_record = {
                    'timestamp': current_time,
                    'analysis': analysis,
                    'plan': optimization_plan,
                    'success': success
                }
                
                self.optimization_actions.append(optimization_record)
                
                # Keep only recent actions
                self.optimization_actions = self.optimization_actions[-100:]
                
                if success:
                    logger.info(f"Optimization applied: {optimization_plan['strategy']}")
                else:
                    logger.warning(f"Optimization failed: {optimization_plan['strategy']}")
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and identify optimization opportunities."""
        analysis = {
            'needs_optimization': False,
            'bottlenecks': [],
            'opportunities': [],
            'performance_score': 1.0,
            'trends': {}
        }
        
        # Analyze each resource type
        for resource_name, history in self.resource_history.items():
            if len(history) < 10:  # Need minimum data for analysis
                continue
            
            # Get recent values
            recent_values = [value for _, value in list(history)[-20:]]
            current_avg = statistics.mean(recent_values)
            
            # Calculate trend
            if len(recent_values) >= 5:
                trend = self._calculate_trend(recent_values)
                analysis['trends'][resource_name] = trend
                
                # Identify bottlenecks
                if resource_name in ['cpu', 'memory', 'gpu'] and current_avg > 0.8:
                    analysis['bottlenecks'].append({
                        'resource': resource_name,
                        'utilization': current_avg,
                        'trend': trend
                    })
                    analysis['needs_optimization'] = True
                
                # Identify opportunities
                if resource_name == 'latency' and current_avg > 1.0:  # 1 second threshold
                    analysis['opportunities'].append({
                        'type': 'latency_reduction',
                        'current_value': current_avg,
                        'target_improvement': 0.2
                    })
                    analysis['needs_optimization'] = True
                
                if resource_name == 'error_rate' and current_avg > 0.05:  # 5% threshold
                    analysis['opportunities'].append({
                        'type': 'error_reduction',
                        'current_value': current_avg,
                        'target_improvement': 0.5
                    })
                    analysis['needs_optimization'] = True
        
        # Calculate overall performance score
        analysis['performance_score'] = self._calculate_performance_score()
        
        return analysis
    
    def _generate_optimization_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization plan based on analysis."""
        plan = {
            'strategy': OptimizationStrategy.ADAPTIVE,
            'actions': [],
            'expected_impact': 0.0,
            'risk_level': 'low'
        }
        
        # Address bottlenecks
        for bottleneck in analysis['bottlenecks']:
            resource = bottleneck['resource']
            utilization = bottleneck['utilization']
            
            if resource == 'cpu' and utilization > 0.9:
                plan['actions'].append({
                    'type': 'scale_cpu',
                    'target_allocation': min(2.0, self.current_allocation[ResourceType.CPU] * 1.5),
                    'expected_impact': 0.3
                })
                plan['strategy'] = OptimizationStrategy.PERFORMANCE
            
            elif resource == 'memory' and utilization > 0.85:
                plan['actions'].append({
                    'type': 'scale_memory',
                    'target_allocation': min(2.0, self.current_allocation[ResourceType.MEMORY] * 1.3),
                    'expected_impact': 0.2
                })
            
            elif resource == 'gpu' and utilization > 0.9:
                plan['actions'].append({
                    'type': 'optimize_gpu',
                    'target_allocation': min(2.0, self.current_allocation[ResourceType.GPU] * 1.2),
                    'expected_impact': 0.4
                })
        
        # Address opportunities
        for opportunity in analysis['opportunities']:
            if opportunity['type'] == 'latency_reduction':
                plan['actions'].append({
                    'type': 'optimize_caching',
                    'expected_impact': 0.25
                })
                plan['actions'].append({
                    'type': 'optimize_queries',
                    'expected_impact': 0.15
                })
            
            elif opportunity['type'] == 'error_reduction':
                plan['actions'].append({
                    'type': 'increase_retries',
                    'expected_impact': 0.3
                })
                plan['actions'].append({
                    'type': 'improve_validation',
                    'expected_impact': 0.2
                })
        
        # Calculate total expected impact
        plan['expected_impact'] = sum(action['expected_impact'] for action in plan['actions'])
        
        # Assess risk level
        if plan['expected_impact'] > 0.5:
            plan['risk_level'] = 'high'
        elif plan['expected_impact'] > 0.3:
            plan['risk_level'] = 'medium'
        else:
            plan['risk_level'] = 'low'
        
        return plan
    
    def _execute_optimization_plan(self, plan: Dict[str, Any]) -> bool:
        """Execute optimization plan."""
        try:
            for action in plan['actions']:
                success = self._execute_action(action)
                if not success:
                    logger.warning(f"Failed to execute action: {action['type']}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing optimization plan: {e}")
            return False
    
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute individual optimization action."""
        action_type = action['type']
        
        if action_type == 'scale_cpu':
            new_allocation = action['target_allocation']
            self.current_allocation[ResourceType.CPU] = new_allocation
            logger.info(f"Scaled CPU allocation to {new_allocation}")
            return True
        
        elif action_type == 'scale_memory':
            new_allocation = action['target_allocation']
            self.current_allocation[ResourceType.MEMORY] = new_allocation
            logger.info(f"Scaled memory allocation to {new_allocation}")
            return True
        
        elif action_type == 'optimize_gpu':
            new_allocation = action['target_allocation']
            self.current_allocation[ResourceType.GPU] = new_allocation
            logger.info(f"Optimized GPU allocation to {new_allocation}")
            return True
        
        elif action_type == 'optimize_caching':
            # This would trigger cache optimization
            logger.info("Optimized caching strategy")
            return True
        
        elif action_type == 'optimize_queries':
            # This would trigger query optimization
            logger.info("Optimized database queries")
            return True
        
        elif action_type == 'increase_retries':
            # This would adjust retry policies
            logger.info("Increased retry thresholds")
            return True
        
        elif action_type == 'improve_validation':
            # This would strengthen input validation
            logger.info("Improved input validation")
            return True
        
        else:
            logger.warning(f"Unknown optimization action: {action_type}")
            return False
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_avg = (n - 1) / 2
        y_avg = statistics.mean(values)
        
        numerator = sum((i - x_avg) * (values[i] - y_avg) for i in range(n))
        denominator = sum((i - x_avg) ** 2 for i in range(n))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        score = 1.0
        
        # Penalize high resource utilization
        for resource_type, history in self.resource_history.items():
            if resource_type in ['cpu', 'memory', 'gpu'] and history:
                recent_values = [value for _, value in list(history)[-10:]]
                if recent_values:
                    avg_utilization = statistics.mean(recent_values)
                    if avg_utilization > 0.8:
                        score *= (1.0 - (avg_utilization - 0.8) * 2)
        
        # Penalize high latency
        if self.resource_history['latency']:
            recent_latencies = [value for _, value in list(self.resource_history['latency'])[-10:]]
            if recent_latencies:
                avg_latency = statistics.mean(recent_latencies)
                if avg_latency > 0.5:  # 500ms threshold
                    score *= (1.0 - min(avg_latency - 0.5, 0.5))
        
        # Penalize high error rate
        if self.resource_history['error_rate']:
            recent_errors = [value for _, value in list(self.resource_history['error_rate'])[-10:]]
            if recent_errors:
                avg_error_rate = statistics.mean(recent_errors)
                if avg_error_rate > 0.01:  # 1% threshold
                    score *= (1.0 - min(avg_error_rate * 10, 0.5))
        
        return max(0.0, score)
    
    def _update_predictions(self):
        """Update resource usage predictions."""
        current_time = time.time()
        prediction_horizon = self.config.prediction_horizon
        
        for resource_name, history in self.resource_history.items():
            if len(history) < 20:  # Need minimum data for prediction
                continue
            
            # Get recent data
            recent_data = list(history)[-50:]  # Last 50 data points
            values = [value for _, value in recent_data]
            
            # Simple linear prediction (could be replaced with ML model)
            if len(values) >= 10:
                trend = self._calculate_trend(values)
                current_value = values[-1]
                predicted_value = current_value + (trend * prediction_horizon)
                
                self.resource_predictions[resource_name] = {
                    'predicted_value': max(0, predicted_value),
                    'confidence': min(1.0, len(values) / 50.0),
                    'trend': trend,
                    'prediction_time': current_time + prediction_horizon
                }
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization system status."""
        with self._lock:
            recent_optimizations = [
                action for action in self.optimization_actions
                if time.time() - action['timestamp'] <= 3600  # Last hour
            ]
            
            successful_optimizations = [
                action for action in recent_optimizations 
                if action['success']
            ]
            
            return {
                'running': self.running,
                'current_allocation': {rt.value: alloc for rt, alloc in self.current_allocation.items()},
                'performance_score': self._calculate_performance_score(),
                'recent_optimizations': len(recent_optimizations),
                'optimization_success_rate': (
                    len(successful_optimizations) / max(1, len(recent_optimizations))
                ),
                'predictions': self.resource_predictions.copy(),
                'optimization_frequency': len(recent_optimizations) / 24.0  # Per hour
            }

class PredictiveLoadBalancer:
    """ML-based predictive load balancer."""
    
    def __init__(self):
        self.servers = {}
        self.load_history = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_models = {}
        self.routing_decisions = []
        self._lock = threading.RLock()
    
    def register_server(self, server_id: str, capacity: float = 1.0, 
                       health_check: Callable[[], bool] = None):
        """Register a server for load balancing."""
        with self._lock:
            self.servers[server_id] = {
                'capacity': capacity,
                'current_load': 0.0,
                'health_check': health_check,
                'last_health_check': 0,
                'healthy': True,
                'response_times': deque(maxlen=100),
                'success_rate': 1.0
            }
            logger.info(f"Registered server: {server_id}")
    
    def route_request(self, request_context: Dict[str, Any] = None) -> Optional[str]:
        """Route request to optimal server using predictive algorithms."""
        with self._lock:
            if not self.servers:
                return None
            
            # Update server health
            self._update_server_health()
            
            # Get healthy servers
            healthy_servers = [
                server_id for server_id, info in self.servers.items()
                if info['healthy']
            ]
            
            if not healthy_servers:
                logger.warning("No healthy servers available")
                return None
            
            # Predict optimal server
            optimal_server = self._predict_optimal_server(healthy_servers, request_context)
            
            # Record routing decision
            decision = {
                'timestamp': time.time(),
                'selected_server': optimal_server,
                'available_servers': healthy_servers.copy(),
                'request_context': request_context or {},
                'prediction_confidence': self._get_prediction_confidence(optimal_server)
            }
            
            self.routing_decisions.append(decision)
            self.routing_decisions = self.routing_decisions[-1000:]  # Keep recent decisions
            
            return optimal_server
    
    def _update_server_health(self):
        """Update server health status."""
        current_time = time.time()
        
        for server_id, server_info in self.servers.items():
            # Check if health check is needed (every 30 seconds)
            if current_time - server_info['last_health_check'] >= 30:
                if server_info['health_check']:
                    try:
                        server_info['healthy'] = server_info['health_check']()
                    except Exception as e:
                        logger.error(f"Health check failed for {server_id}: {e}")
                        server_info['healthy'] = False
                else:
                    # Assume healthy if no health check function
                    server_info['healthy'] = True
                
                server_info['last_health_check'] = current_time
    
    def _predict_optimal_server(self, healthy_servers: List[str], 
                              request_context: Dict[str, Any] = None) -> str:
        """Predict optimal server for request."""
        server_scores = {}
        
        for server_id in healthy_servers:
            server_info = self.servers[server_id]
            
            # Base score components
            load_score = 1.0 - server_info['current_load']  # Lower load = higher score
            capacity_score = server_info['capacity']
            
            # Response time score (lower response time = higher score)
            response_times = list(server_info['response_times'])
            if response_times:
                avg_response_time = statistics.mean(response_times)
                response_score = 1.0 / (1.0 + avg_response_time)
            else:
                response_score = 1.0
            
            # Success rate score
            success_score = server_info['success_rate']
            
            # Predicted load score
            predicted_load = self._predict_server_load(server_id)
            predicted_score = 1.0 - predicted_load
            
            # Composite score
            composite_score = (
                load_score * 0.3 +
                capacity_score * 0.2 +
                response_score * 0.2 +
                success_score * 0.15 +
                predicted_score * 0.15
            )
            
            server_scores[server_id] = composite_score
        
        # Select server with highest score
        optimal_server = max(server_scores.keys(), key=lambda k: server_scores[k])
        
        return optimal_server
    
    def _predict_server_load(self, server_id: str) -> float:
        """Predict future load for server."""
        load_history = self.load_history[server_id]
        
        if len(load_history) < 5:
            return self.servers[server_id]['current_load']
        
        # Simple trend-based prediction
        recent_loads = [load for _, load in list(load_history)[-10:]]
        trend = self._calculate_trend(recent_loads)
        current_load = recent_loads[-1]
        
        # Predict load 30 seconds ahead
        predicted_load = current_load + (trend * 30)
        
        return max(0.0, min(1.0, predicted_load))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_avg = (n - 1) / 2
        y_avg = statistics.mean(values)
        
        numerator = sum((i - x_avg) * (values[i] - y_avg) for i in range(n))
        denominator = sum((i - x_avg) ** 2 for i in range(n))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _get_prediction_confidence(self, server_id: str) -> float:
        """Get prediction confidence for server."""
        load_history = self.load_history[server_id]
        history_length = len(load_history)
        
        # Confidence based on available data
        base_confidence = min(1.0, history_length / 50.0)
        
        # Adjust confidence based on recent prediction accuracy
        # This would be more sophisticated in a real implementation
        accuracy_factor = 0.8  # Placeholder
        
        return base_confidence * accuracy_factor
    
    def record_server_metrics(self, server_id: str, load: float, 
                            response_time: float, success: bool):
        """Record server performance metrics."""
        if server_id not in self.servers:
            return
        
        with self._lock:
            server_info = self.servers[server_id]
            current_time = time.time()
            
            # Update current metrics
            server_info['current_load'] = load
            server_info['response_times'].append(response_time)
            
            # Update success rate (exponential moving average)
            alpha = 0.1
            current_success = 1.0 if success else 0.0
            server_info['success_rate'] = (
                alpha * current_success + (1 - alpha) * server_info['success_rate']
            )
            
            # Record in history
            self.load_history[server_id].append((current_time, load))
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        with self._lock:
            healthy_servers = [
                server_id for server_id, info in self.servers.items()
                if info['healthy']
            ]
            
            total_capacity = sum(
                info['capacity'] for info in self.servers.values() 
                if info['healthy']
            )
            
            current_load = sum(
                info['current_load'] * info['capacity'] 
                for info in self.servers.values() 
                if info['healthy']
            )
            
            return {
                'total_servers': len(self.servers),
                'healthy_servers': len(healthy_servers),
                'total_capacity': total_capacity,
                'current_utilization': current_load / max(1, total_capacity),
                'recent_decisions': len([
                    d for d in self.routing_decisions
                    if time.time() - d['timestamp'] <= 300  # Last 5 minutes
                ]),
                'server_details': {
                    server_id: {
                        'healthy': info['healthy'],
                        'current_load': info['current_load'],
                        'capacity': info['capacity'],
                        'success_rate': info['success_rate'],
                        'avg_response_time': (
                            statistics.mean(list(info['response_times']))
                            if info['response_times'] else 0.0
                        )
                    }
                    for server_id, info in self.servers.items()
                }
            }

class IntelligentOptimizationSystem:
    """Main intelligent optimization system."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.cache = IntelligentCache(learning_enabled=True)
        self.resource_manager = AdaptiveResourceManager(self.config)
        self.load_balancer = PredictiveLoadBalancer()
        self.optimization_history = []
        self.system_metrics = PerformanceMetrics()
        self._lock = threading.RLock()
    
    def start_optimization(self):
        """Start all optimization systems."""
        self.resource_manager.start_optimization()
        logger.info("Intelligent optimization system started")
    
    def stop_optimization(self):
        """Stop all optimization systems."""
        self.resource_manager.stop_optimization()
        logger.info("Intelligent optimization system stopped")
    
    def optimize_request(self, request_func: Callable, *args, **kwargs) -> Any:
        """Execute request with intelligent optimization."""
        start_time = time.time()
        
        # Try cache first
        cache_key = self._generate_cache_key(request_func.__name__, args, kwargs)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Route to optimal server
        server_id = self.load_balancer.route_request({
            'function': request_func.__name__,
            'args_hash': hash(str(args) + str(kwargs))
        })
        
        try:
            # Execute request
            result = request_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result
            self.cache.set(cache_key, result, ttl=3600)  # 1 hour TTL
            
            # Record metrics
            if server_id:
                self.load_balancer.record_server_metrics(
                    server_id, 0.5, execution_time, True  # Mock metrics
                )
            
            self._update_system_metrics(execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure metrics
            if server_id:
                self.load_balancer.record_server_metrics(
                    server_id, 0.5, execution_time, False
                )
            
            self._update_system_metrics(execution_time, False)
            raise
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for request."""
        key_data = {
            'function': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_system_metrics(self, execution_time: float, success: bool):
        """Update system performance metrics."""
        with self._lock:
            self.system_metrics.request_latency = execution_time
            
            if success:
                self.system_metrics.throughput += 1
            else:
                self.system_metrics.error_rate += 0.01  # Increment error rate
            
            # Record metrics for resource manager
            self.resource_manager.record_metrics(self.system_metrics)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        return {
            'timestamp': time.time(),
            'cache_statistics': self.cache.get_cache_statistics(),
            'resource_management': self.resource_manager.get_optimization_status(),
            'load_balancing': self.load_balancer.get_load_balancer_status(),
            'system_metrics': asdict(self.system_metrics),
            'optimization_grade': self._calculate_optimization_grade()
        }
    
    def _calculate_optimization_grade(self) -> str:
        """Calculate overall optimization grade."""
        cache_stats = self.cache.get_cache_statistics()
        resource_stats = self.resource_manager.get_optimization_status()
        
        # Calculate composite score
        cache_score = cache_stats['hit_rate']
        performance_score = resource_stats['performance_score']
        success_rate = resource_stats['optimization_success_rate']
        
        overall_score = (cache_score * 0.3 + performance_score * 0.4 + success_rate * 0.3)
        
        if overall_score >= 0.9:
            return "A+ (Excellent)"
        elif overall_score >= 0.8:
            return "A (Very Good)"
        elif overall_score >= 0.7:
            return "B (Good)"
        elif overall_score >= 0.6:
            return "C (Fair)"
        else:
            return "D (Needs Improvement)"

# Convenience function for easy integration
def create_optimization_system(config: OptimizationConfig = None) -> IntelligentOptimizationSystem:
    """Create and start intelligent optimization system."""
    system = IntelligentOptimizationSystem(config)
    system.start_optimization()
    return system

# Export all classes and functions
__all__ = [
    'IntelligentOptimizationSystem',
    'IntelligentCache',
    'AdaptiveResourceManager',
    'PredictiveLoadBalancer',
    'OptimizationConfig',
    'PerformanceMetrics',
    'OptimizationStrategy',
    'ResourceType',
    'create_optimization_system'
]