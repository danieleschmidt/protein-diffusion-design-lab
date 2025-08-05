"""
Auto-scaling and load balancing for protein diffusion services.

This module provides dynamic scaling, load balancing, and resource
management for production deployments.
"""

import time
import threading
import queue
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import math

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service instance status."""
    STARTING = "starting"
    HEALTHY = "healthy"
    OVERLOADED = "overloaded"
    FAILING = "failing"
    STOPPED = "stopped"

@dataclass
class ServiceInstance:
    """Represents a service instance."""
    instance_id: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.STARTING
    current_load: float = 0.0
    max_capacity: int = 100
    health_score: float = 1.0
    last_health_check: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_request_time(self, response_time: float, success: bool = True):
        """Record a request."""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        self.response_times.append(response_time)
        self.current_load = len(self.response_times)
    
    def get_average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_failure_rate(self) -> float:
        """Get failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        return (self.status in [ServiceStatus.HEALTHY, ServiceStatus.OVERLOADED] and
                self.health_score > 0.5 and
                self.get_failure_rate() < 0.1)
    
    def get_capacity_utilization(self) -> float:
        """Get current capacity utilization (0-1)."""
        return min(1.0, self.current_load / self.max_capacity)

@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: float
    total_requests: int
    average_response_time: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    queue_length: int = 0
    active_instances: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_requests": self.total_requests,
            "average_response_time": self.average_response_time,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "queue_length": self.queue_length,
            "active_instances": self.active_instances,
        }

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    # Scaling thresholds
    scale_up_cpu_threshold: float = 0.8
    scale_up_memory_threshold: float = 0.8
    scale_up_response_time_threshold: float = 5.0
    scale_up_queue_threshold: int = 50
    
    scale_down_cpu_threshold: float = 0.3
    scale_down_memory_threshold: float = 0.3
    scale_down_response_time_threshold: float = 1.0
    scale_down_idle_time: int = 300  # seconds
    
    # Instance limits
    min_instances: int = 1
    max_instances: int = 10
    
    # Scaling behavior
    scale_up_cooldown: int = 60  # seconds
    scale_down_cooldown: int = 300  # seconds
    health_check_interval: int = 30  # seconds
    
    # Load balancing
    load_balance_algorithm: str = "least_connections"  # "round_robin", "least_connections", "weighted"

class LoadBalancer:
    """Load balancer for distributing requests across instances."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.instances: Dict[str, ServiceInstance] = {}
        self.round_robin_index = 0
        self._lock = threading.RLock()
    
    def add_instance(self, instance: ServiceInstance):
        """Add a service instance."""
        with self._lock:
            self.instances[instance.instance_id] = instance
            logger.info(f"Added instance {instance.instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str):
        """Remove a service instance."""
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                logger.info(f"Removed instance {instance_id} from load balancer")
    
    def get_instance(self) -> Optional[ServiceInstance]:
        """Get next instance based on load balancing algorithm."""
        with self._lock:
            healthy_instances = [
                instance for instance in self.instances.values()
                if instance.is_healthy()
            ]
            
            if not healthy_instances:
                logger.warning("No healthy instances available")
                return None
            
            if self.config.load_balance_algorithm == "round_robin":
                return self._round_robin_selection(healthy_instances)
            elif self.config.load_balance_algorithm == "least_connections":
                return self._least_connections_selection(healthy_instances)
            elif self.config.load_balance_algorithm == "weighted":
                return self._weighted_selection(healthy_instances)
            else:
                return healthy_instances[0]  # Fallback
    
    def _round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection."""
        instance = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return instance
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections."""
        return min(instances, key=lambda x: x.current_load)
    
    def _weighted_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted selection based on capacity and health."""
        # Calculate weights based on available capacity and health
        weights = []
        for instance in instances:
            available_capacity = 1.0 - instance.get_capacity_utilization()
            weight = available_capacity * instance.health_score
            weights.append(weight)
        
        if sum(weights) == 0:
            return instances[0]  # Fallback
        
        # Weighted random selection
        import random
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return instances[i]
        
        return instances[-1]  # Fallback
    
    def update_instance_metrics(self, instance_id: str, response_time: float, success: bool = True):
        """Update instance metrics after request."""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id].add_request_time(response_time, success)
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            stats = {
                "total_instances": len(self.instances),
                "healthy_instances": sum(1 for i in self.instances.values() if i.is_healthy()),
                "total_requests": sum(i.total_requests for i in self.instances.values()),
                "instances": {}
            }
            
            for instance_id, instance in self.instances.items():
                stats["instances"][instance_id] = {
                    "status": instance.status.value,
                    "current_load": instance.current_load,
                    "health_score": instance.health_score,
                    "total_requests": instance.total_requests,
                    "failure_rate": instance.get_failure_rate(),
                    "avg_response_time": instance.get_average_response_time(),
                    "capacity_utilization": instance.get_capacity_utilization(),
                }
            
            return stats

class AutoScaler:
    """Auto-scaling controller for managing service instances."""
    
    def __init__(self, config: ScalingConfig, 
                 instance_creator: Optional[Callable] = None,
                 instance_destroyer: Optional[Callable] = None):
        self.config = config
        self.load_balancer = LoadBalancer(config)
        self.instance_creator = instance_creator
        self.instance_destroyer = instance_destroyer
        
        self.metrics_history: deque = deque(maxlen=1000)
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        
        self._monitoring = False
        self._monitor_thread = None
        self._request_queue = queue.Queue()
        
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started auto-scaling monitoring")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped auto-scaling monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Make scaling decisions
                self._evaluate_scaling(metrics)
                
                # Health checks
                self._perform_health_checks()
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        current_time = time.time()
        
        # Get system metrics
        cpu_usage = 0.0
        memory_usage = 0.0
        gpu_usage = 0.0
        
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent() / 100.0
            memory_usage = psutil.virtual_memory().percent / 100.0
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # Average GPU usage across all devices
                gpu_memory_used = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
                gpu_memory_total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
                gpu_usage = gpu_memory_used / gpu_memory_total if gpu_memory_total > 0 else 0.0
            except Exception:
                gpu_usage = 0.0
        
        # Get load balancer stats
        load_stats = self.load_balancer.get_load_stats()
        
        # Calculate average response time
        avg_response_time = 0.0
        if load_stats["healthy_instances"] > 0:
            response_times = []
            for instance_stats in load_stats["instances"].values():
                if instance_stats["avg_response_time"] > 0:
                    response_times.append(instance_stats["avg_response_time"])
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
        
        return ScalingMetrics(
            timestamp=current_time,
            total_requests=load_stats["total_requests"],
            average_response_time=avg_response_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            queue_length=self._request_queue.qsize(),
            active_instances=load_stats["healthy_instances"]
        )
    
    def _evaluate_scaling(self, metrics: ScalingMetrics):
        """Evaluate whether to scale up or down."""
        current_time = time.time()
        
        # Check scale-up conditions
        should_scale_up = (
            (metrics.cpu_usage > self.config.scale_up_cpu_threshold or
             metrics.memory_usage > self.config.scale_up_memory_threshold or
             metrics.average_response_time > self.config.scale_up_response_time_threshold or
             metrics.queue_length > self.config.scale_up_queue_threshold) and
            metrics.active_instances < self.config.max_instances and
            current_time - self.last_scale_up_time > self.config.scale_up_cooldown
        )
        
        if should_scale_up:
            self._scale_up()
            self.last_scale_up_time = current_time
            return
        
        # Check scale-down conditions
        should_scale_down = (
            metrics.cpu_usage < self.config.scale_down_cpu_threshold and
            metrics.memory_usage < self.config.scale_down_memory_threshold and
            metrics.average_response_time < self.config.scale_down_response_time_threshold and
            metrics.queue_length == 0 and
            metrics.active_instances > self.config.min_instances and
            current_time - self.last_scale_down_time > self.config.scale_down_cooldown
        )
        
        if should_scale_down:
            self._scale_down()
            self.last_scale_down_time = current_time
    
    def _scale_up(self):
        """Scale up by adding a new instance."""
        if self.instance_creator:
            try:
                new_instance = self.instance_creator()
                self.load_balancer.add_instance(new_instance)
                logger.info(f"Scaled up: added instance {new_instance.instance_id}")
            except Exception as e:
                logger.error(f"Failed to scale up: {e}")
        else:
            logger.warning("No instance creator configured for scaling up")
    
    def _scale_down(self):
        """Scale down by removing an instance."""
        # Find instance with lowest load
        with self.load_balancer._lock:
            if not self.load_balancer.instances:
                return
            
            instance_to_remove = min(
                self.load_balancer.instances.values(),
                key=lambda x: x.current_load
            )
            
            if instance_to_remove.current_load == 0:  # Only remove idle instances
                try:
                    if self.instance_destroyer:
                        self.instance_destroyer(instance_to_remove)
                    
                    self.load_balancer.remove_instance(instance_to_remove.instance_id)
                    logger.info(f"Scaled down: removed instance {instance_to_remove.instance_id}")
                
                except Exception as e:
                    logger.error(f"Failed to scale down: {e}")
    
    def _perform_health_checks(self):
        """Perform health checks on all instances."""
        current_time = time.time()
        
        with self.load_balancer._lock:
            for instance in self.load_balancer.instances.values():
                # Simple health check based on metrics
                if current_time - instance.last_health_check > self.config.health_check_interval:
                    self._update_instance_health(instance)
                    instance.last_health_check = current_time
    
    def _update_instance_health(self, instance: ServiceInstance):
        """Update instance health score."""
        # Calculate health score based on various factors
        factors = []
        
        # Response time factor
        avg_response_time = instance.get_average_response_time()
        if avg_response_time > 0:
            response_time_factor = max(0.0, 1.0 - (avg_response_time / 10.0))  # Penalize high response times
            factors.append(response_time_factor)
        
        # Failure rate factor
        failure_rate = instance.get_failure_rate()
        failure_rate_factor = max(0.0, 1.0 - failure_rate * 2.0)  # Penalize failures
        factors.append(failure_rate_factor)
        
        # Load factor
        capacity_utilization = instance.get_capacity_utilization()
        load_factor = max(0.0, 1.0 - capacity_utilization)  # Penalize high load
        factors.append(load_factor)
        
        # Calculate overall health score
        if factors:
            instance.health_score = sum(factors) / len(factors)
        else:
            instance.health_score = 1.0
        
        # Update status based on health score and metrics
        if instance.health_score < 0.3 or instance.get_failure_rate() > 0.2:
            instance.status = ServiceStatus.FAILING
        elif capacity_utilization > 0.9:
            instance.status = ServiceStatus.OVERLOADED
        elif instance.health_score > 0.7:
            instance.status = ServiceStatus.HEALTHY
    
    def add_instance(self, instance: ServiceInstance):
        """Add an instance to the load balancer."""
        self.load_balancer.add_instance(instance)
    
    def get_instance_for_request(self) -> Optional[ServiceInstance]:
        """Get an instance for handling a request."""
        return self.load_balancer.get_instance()
    
    def record_request_metrics(self, instance_id: str, response_time: float, success: bool = True):
        """Record request metrics for an instance."""
        self.load_balancer.update_instance_metrics(instance_id, response_time, success)
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
        
        stats = {
            "config": {
                "min_instances": self.config.min_instances,
                "max_instances": self.config.max_instances,
                "scale_up_cpu_threshold": self.config.scale_up_cpu_threshold,
                "scale_down_cpu_threshold": self.config.scale_down_cpu_threshold,
            },
            "current_metrics": recent_metrics[-1].to_dict() if recent_metrics else {},
            "load_balancer": self.load_balancer.get_load_stats(),
            "scaling_history": {
                "last_scale_up": self.last_scale_up_time,
                "last_scale_down": self.last_scale_down_time,
                "total_metrics_collected": len(self.metrics_history),
            }
        }
        
        return stats

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
                
                return result
            
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
                
                raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }

class ServiceRegistry:
    """Service registry for managing service instances."""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, ServiceInstance]] = {}
        self._lock = threading.RLock()
    
    def register_service(self, service_name: str, instance: ServiceInstance):
        """Register a service instance."""
        with self._lock:
            if service_name not in self.services:
                self.services[service_name] = {}
            
            self.services[service_name][instance.instance_id] = instance
            logger.info(f"Registered {service_name} instance {instance.instance_id}")
    
    def unregister_service(self, service_name: str, instance_id: str):
        """Unregister a service instance."""
        with self._lock:
            if service_name in self.services and instance_id in self.services[service_name]:
                del self.services[service_name][instance_id]
                logger.info(f"Unregistered {service_name} instance {instance_id}")
                
                if not self.services[service_name]:
                    del self.services[service_name]
    
    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances for a service."""
        with self._lock:
            return list(self.services.get(service_name, {}).values())
    
    def get_healthy_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get healthy instances for a service."""
        instances = self.get_service_instances(service_name)
        return [instance for instance in instances if instance.is_healthy()]
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get service registry statistics."""
        with self._lock:
            stats = {}
            for service_name, instances in self.services.items():
                healthy_count = sum(1 for i in instances.values() if i.is_healthy())
                stats[service_name] = {
                    "total_instances": len(instances),
                    "healthy_instances": healthy_count,
                    "instances": {
                        instance_id: {
                            "status": instance.status.value,
                            "health_score": instance.health_score,
                            "current_load": instance.current_load,
                        }
                        for instance_id, instance in instances.items()
                    }
                }
            
            return stats

# Global instances
_auto_scaler = None
_service_registry = ServiceRegistry()

def get_auto_scaler(config: Optional[ScalingConfig] = None, **kwargs) -> AutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        config = config or ScalingConfig()
        _auto_scaler = AutoScaler(config, **kwargs)
    return _auto_scaler

def get_service_registry() -> ServiceRegistry:
    """Get global service registry instance."""
    return _service_registry