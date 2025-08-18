"""
Advanced Performance Optimization for Protein Diffusion Design Lab

This module implements sophisticated performance optimization techniques:
- Dynamic batch sizing and request grouping
- Memory pool management and GPU optimization
- Adaptive model serving with resource-aware scheduling
- Intelligent caching strategies with predictive prefetching
- Performance monitoring and auto-tuning
"""

import time
import threading
import queue
import logging
import statistics
import hashlib
import json
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import gc
import weakref

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch functionality
    class MockTensor:
        def __init__(self, data):
            self.data = data
        def to(self, device): return self
        def cuda(self): return self
        def cpu(self): return self
        def size(self): return (1,)
        def dim(self): return 1
        def numpy(self): return [0.5]
        def item(self): return 0.5
        def detach(self): return self
        def clone(self): return self
        
    class MockTorch:
        @staticmethod
        def tensor(data): return MockTensor(data)
        @staticmethod
        def cat(tensors, dim=0): return MockTensor([])
        @staticmethod
        def stack(tensors, dim=0): return MockTensor([])
        @staticmethod
        def cuda():
            return type('MockCuda', (), {
                'is_available': lambda: False,
                'device_count': lambda: 0,
                'current_device': lambda: 0,
                'get_device_name': lambda i: 'MockGPU',
                'memory_allocated': lambda i: 0,
                'memory_reserved': lambda i: 0,
                'max_memory_allocated': lambda i: 0,
                'empty_cache': lambda: None
            })()
        class no_grad:
            def __enter__(self): return self
            def __exit__(self, *args): pass
    torch = MockTorch()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    throughput: float = 0.0  # requests per second
    latency_p95: float = 0.0  # 95th percentile latency
    memory_usage: float = 0.0  # memory usage percentage
    gpu_utilization: float = 0.0  # GPU utilization percentage
    cache_hit_rate: float = 0.0  # cache hit rate
    error_rate: float = 0.0  # error rate
    queue_depth: int = 0  # current queue depth
    active_connections: int = 0  # active connections


@dataclass
class BatchRequest:
    """Represents a batched request for processing."""
    id: str
    timestamp: float
    requests: List[Dict[str, Any]]
    priority: int = 0
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.timestamp}_{len(self.requests)}".encode()
            ).hexdigest()[:8]


class AdaptiveBatcher:
    """Dynamically adjusts batch sizes for optimal performance."""
    
    def __init__(self, min_batch_size: int = 1, max_batch_size: int = 32, 
                 target_latency: float = 100.0):  # 100ms target
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = min_batch_size
        self.target_latency = target_latency
        
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        self.batch_size_history = deque(maxlen=100)
        
        self.lock = threading.RLock()
        self.adjustment_threshold = 0.1  # 10% change threshold
    
    def record_batch_performance(self, batch_size: int, latency: float, 
                                throughput: float):
        """Record performance metrics for a completed batch."""
        with self.lock:
            self.latency_history.append(latency)
            self.throughput_history.append(throughput)
            self.batch_size_history.append(batch_size)
            
            # Adjust batch size based on performance
            self._adjust_batch_size()
    
    def _adjust_batch_size(self):
        """Adjust batch size based on recent performance."""
        if len(self.latency_history) < 5:
            return
        
        recent_latency = statistics.mean(list(self.latency_history)[-5:])
        recent_throughput = statistics.mean(list(self.throughput_history)[-5:])
        
        # If latency is too high, reduce batch size
        if recent_latency > self.target_latency * 1.2:
            new_size = max(self.min_batch_size, 
                          int(self.current_batch_size * 0.8))
            if new_size != self.current_batch_size:
                logger.info(f"Reducing batch size: {self.current_batch_size} → {new_size} "
                           f"(latency: {recent_latency:.1f}ms)")
                self.current_batch_size = new_size
        
        # If latency is good and we're not at max, try increasing
        elif (recent_latency < self.target_latency * 0.8 and 
              self.current_batch_size < self.max_batch_size):
            new_size = min(self.max_batch_size,
                          int(self.current_batch_size * 1.2))
            if new_size != self.current_batch_size:
                logger.info(f"Increasing batch size: {self.current_batch_size} → {new_size} "
                           f"(latency: {recent_latency:.1f}ms)")
                self.current_batch_size = new_size
    
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size."""
        with self.lock:
            return self.current_batch_size
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self.lock:
            if not self.latency_history:
                return {"status": "no_data"}
            
            return {
                "current_batch_size": self.current_batch_size,
                "avg_latency": statistics.mean(self.latency_history),
                "avg_throughput": statistics.mean(self.throughput_history),
                "latency_p95": self._percentile(list(self.latency_history), 95),
                "samples": len(self.latency_history),
                "target_latency": self.target_latency
            }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        lower = int(index)
        upper = lower + 1
        if upper >= len(sorted_values):
            return sorted_values[-1]
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


class MemoryPool:
    """Manages memory allocation and reuse for optimal performance."""
    
    def __init__(self, initial_size: int = 10):
        self.pool = queue.Queue()
        self.allocated = set()
        self.total_created = 0
        self.total_reused = 0
        self.lock = threading.RLock()
        
        # Pre-allocate initial tensors
        for _ in range(initial_size):
            self._create_tensor()
    
    def _create_tensor(self) -> Any:
        """Create a new tensor for the pool."""
        if TORCH_AVAILABLE:
            # Create tensors on appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tensor = torch.empty(0, device=device)  # Empty tensor that can be resized
        else:
            tensor = torch.tensor([])  # Mock tensor
        
        self.total_created += 1
        return tensor
    
    def get_tensor(self, shape: Tuple[int, ...] = None) -> Any:
        """Get a tensor from the pool or create a new one."""
        with self.lock:
            try:
                tensor = self.pool.get_nowait()
                self.total_reused += 1
                logger.debug("Reused tensor from pool")
            except queue.Empty:
                tensor = self._create_tensor()
                logger.debug("Created new tensor")
            
            # Resize if needed
            if TORCH_AVAILABLE and shape:
                tensor = tensor.new_empty(shape)
            
            self.allocated.add(id(tensor))
            return tensor
    
    def return_tensor(self, tensor: Any):
        """Return a tensor to the pool."""
        with self.lock:
            tensor_id = id(tensor)
            if tensor_id in self.allocated:
                self.allocated.remove(tensor_id)
                
                # Clear tensor data and return to pool
                if TORCH_AVAILABLE and hasattr(tensor, 'zero_'):
                    tensor.zero_()
                
                self.pool.put(tensor)
                logger.debug("Returned tensor to pool")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            return {
                "pool_size": self.pool.qsize(),
                "allocated": len(self.allocated),
                "total_created": self.total_created,
                "total_reused": self.total_reused,
                "reuse_rate": self.total_reused / max(self.total_created, 1),
                "efficiency": f"{self.total_reused / max(self.total_created + self.total_reused, 1):.2%}"
            }
    
    def cleanup(self):
        """Clean up memory pool."""
        with self.lock:
            # Clear the pool
            while not self.pool.empty():
                try:
                    self.pool.get_nowait()
                except queue.Empty:
                    break
            
            self.allocated.clear()
            
            # Force garbage collection
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()


class IntelligentCache:
    """Advanced caching with predictive prefetching and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        
        self.cache = {}  # key -> (value, timestamp, access_count, last_access)
        self.access_pattern = defaultdict(int)  # key -> access_frequency
        self.prefetch_predictions = defaultdict(list)  # key -> [predicted_next_keys]
        
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'prefetches': 0,
            'prefetch_hits': 0
        }
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get value from cache. Returns (value, hit)."""
        with self.lock:
            if key in self.cache:
                value, timestamp, access_count, last_access = self.cache[key]
                
                # Check if expired
                if time.time() - timestamp > self.ttl:
                    del self.cache[key]
                    self.stats['misses'] += 1
                    return None, False
                
                # Update access info
                self.cache[key] = (value, timestamp, access_count + 1, time.time())
                self.access_pattern[key] += 1
                self.stats['hits'] += 1
                
                # Update prefetch predictions
                self._update_predictions(key)
                
                return value, True
            else:
                self.stats['misses'] += 1
                return None, False
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self.lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = (value, current_time, 1, current_time)
            self.access_pattern[key] += 1
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find least recently used item
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k][3])  # last_access time
        
        del self.cache[lru_key]
        self.stats['evictions'] += 1
        logger.debug(f"Evicted cache key: {lru_key}")
    
    def _update_predictions(self, accessed_key: str):
        """Update predictions for prefetching based on access patterns."""
        # Simple co-occurrence based prediction
        # In a real implementation, this could use more sophisticated ML models
        
        current_time = time.time()
        time_window = 300  # 5 minutes
        
        # Find recently accessed keys
        recent_keys = [
            k for k, (_, timestamp, _, _) in self.cache.items()
            if current_time - timestamp < time_window and k != accessed_key
        ]
        
        # Update co-occurrence predictions
        for recent_key in recent_keys[:5]:  # Limit to top 5
            if len(self.prefetch_predictions[accessed_key]) < 10:
                if recent_key not in self.prefetch_predictions[accessed_key]:
                    self.prefetch_predictions[accessed_key].append(recent_key)
    
    def prefetch(self, key: str, prefetch_fn: Callable[[str], Any]):
        """Prefetch related items based on predictions."""
        with self.lock:
            if key in self.prefetch_predictions:
                for predicted_key in self.prefetch_predictions[key][:3]:  # Top 3
                    if predicted_key not in self.cache:
                        try:
                            value = prefetch_fn(predicted_key)
                            if value is not None:
                                self.set(predicted_key, value)
                                self.stats['prefetches'] += 1
                                logger.debug(f"Prefetched: {predicted_key}")
                        except Exception as e:
                            logger.debug(f"Prefetch failed for {predicted_key}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(total_requests, 1)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                **self.stats,
                'top_accessed': sorted(
                    self.access_pattern.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
            }
    
    def cleanup_expired(self):
        """Clean up expired cache entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp, _, _) in self.cache.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
                
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


class ResourceMonitor:
    """Monitors system resources for optimization decisions."""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=100)
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.RLock()
    
    def start_monitoring(self, interval: float = 10.0):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        metrics = PerformanceMetrics()
        current_time = time.time()
        
        # CPU and memory metrics
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                metrics.memory_usage = memory.percent / 100.0
            except:
                pass
        
        # GPU metrics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_count = torch.cuda.device_count()
                total_memory = 0
                used_memory = 0
                
                for i in range(device_count):
                    total_memory += torch.cuda.get_device_properties(i).total_memory
                    used_memory += torch.cuda.memory_allocated(i)
                
                if total_memory > 0:
                    metrics.gpu_utilization = used_memory / total_memory
            except:
                pass
        
        return metrics
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics."""
        with self.lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 readings
            
            return {
                "samples": len(recent_metrics),
                "avg_memory_usage": statistics.mean(m.memory_usage for m in recent_metrics),
                "avg_gpu_utilization": statistics.mean(m.gpu_utilization for m in recent_metrics),
                "max_memory_usage": max(m.memory_usage for m in recent_metrics),
                "max_gpu_utilization": max(m.gpu_utilization for m in recent_metrics),
                "current": recent_metrics[-1] if recent_metrics else None
            }


class PerformanceOptimizer:
    """Main performance optimization controller."""
    
    def __init__(self):
        self.batcher = AdaptiveBatcher()
        self.memory_pool = MemoryPool()
        self.cache = IntelligentCache()
        self.resource_monitor = ResourceMonitor()
        
        self.optimization_rules = []
        self.auto_tuning_enabled = True
        self.last_optimization = 0
        
        # Register default optimization rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default optimization rules."""
        
        def memory_pressure_rule(metrics: PerformanceMetrics) -> Dict[str, Any]:
            """React to high memory pressure."""
            if metrics.memory_usage > 0.9:
                return {
                    'action': 'reduce_cache_size',
                    'reason': 'high_memory_pressure',
                    'severity': 'high',
                    'params': {'new_size': int(self.cache.max_size * 0.7)}
                }
            return {}
        
        def gpu_utilization_rule(metrics: PerformanceMetrics) -> Dict[str, Any]:
            """React to GPU utilization."""
            if metrics.gpu_utilization > 0.95:
                return {
                    'action': 'reduce_batch_size',
                    'reason': 'gpu_memory_pressure',
                    'severity': 'medium',
                    'params': {'factor': 0.8}
                }
            elif metrics.gpu_utilization < 0.3:
                return {
                    'action': 'increase_batch_size',
                    'reason': 'low_gpu_utilization',
                    'severity': 'low',
                    'params': {'factor': 1.2}
                }
            return {}
        
        def cache_efficiency_rule(metrics: PerformanceMetrics) -> Dict[str, Any]:
            """React to cache performance."""
            cache_stats = self.cache.get_stats()
            if cache_stats['hit_rate'] < 0.3 and cache_stats['total_requests'] > 100:
                return {
                    'action': 'increase_cache_size',
                    'reason': 'low_cache_hit_rate',
                    'severity': 'medium',
                    'params': {'factor': 1.5}
                }
            return {}
        
        self.optimization_rules = [
            memory_pressure_rule,
            gpu_utilization_rule,
            cache_efficiency_rule
        ]
    
    def start_optimization(self):
        """Start performance optimization."""
        self.resource_monitor.start_monitoring()
        
        # Start auto-tuning thread
        def auto_tune_loop():
            while True:
                try:
                    if self.auto_tuning_enabled:
                        self.auto_tune()
                    time.sleep(30)  # Auto-tune every 30 seconds
                except Exception as e:
                    logger.error(f"Auto-tuning error: {e}")
                    time.sleep(30)
        
        threading.Thread(target=auto_tune_loop, daemon=True).start()
        logger.info("Performance optimization started")
    
    def stop_optimization(self):
        """Stop performance optimization."""
        self.resource_monitor.stop_monitoring()
        self.auto_tuning_enabled = False
        logger.info("Performance optimization stopped")
    
    def auto_tune(self):
        """Perform automatic performance tuning."""
        current_time = time.time()
        
        # Don't optimize too frequently
        if current_time - self.last_optimization < 30:
            return
        
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return
        
        # Apply optimization rules
        for rule in self.optimization_rules:
            try:
                action = rule(current_metrics)
                if action:
                    self._apply_optimization(action)
            except Exception as e:
                logger.error(f"Optimization rule failed: {e}")
        
        # Cleanup expired cache entries
        self.cache.cleanup_expired()
        
        self.last_optimization = current_time
    
    def _apply_optimization(self, action: Dict[str, Any]):
        """Apply an optimization action."""
        action_type = action.get('action')
        params = action.get('params', {})
        reason = action.get('reason', 'unknown')
        severity = action.get('severity', 'low')
        
        logger.info(f"Applying optimization: {action_type} (reason: {reason}, severity: {severity})")
        
        try:
            if action_type == 'reduce_cache_size':
                new_size = params.get('new_size', int(self.cache.max_size * 0.8))
                self.cache.max_size = new_size
                logger.info(f"Reduced cache size to {new_size}")
            
            elif action_type == 'increase_cache_size':
                factor = params.get('factor', 1.2)
                new_size = int(self.cache.max_size * factor)
                self.cache.max_size = min(new_size, 5000)  # Cap at 5000
                logger.info(f"Increased cache size to {self.cache.max_size}")
            
            elif action_type == 'reduce_batch_size':
                factor = params.get('factor', 0.8)
                new_size = max(1, int(self.batcher.current_batch_size * factor))
                self.batcher.current_batch_size = new_size
                logger.info(f"Reduced batch size to {new_size}")
            
            elif action_type == 'increase_batch_size':
                factor = params.get('factor', 1.2)
                new_size = min(self.batcher.max_batch_size, 
                              int(self.batcher.current_batch_size * factor))
                self.batcher.current_batch_size = new_size
                logger.info(f"Increased batch size to {new_size}")
            
            else:
                logger.warning(f"Unknown optimization action: {action_type}")
                
        except Exception as e:
            logger.error(f"Failed to apply optimization {action_type}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'timestamp': time.time(),
            'batcher': self.batcher.get_performance_summary(),
            'memory_pool': self.memory_pool.get_stats(),
            'cache': self.cache.get_stats(),
            'resources': self.resource_monitor.get_metrics_summary(),
            'auto_tuning_enabled': self.auto_tuning_enabled,
            'last_optimization': self.last_optimization
        }
    
    def force_optimization(self):
        """Force immediate optimization."""
        logger.info("Forcing performance optimization")
        self.last_optimization = 0  # Reset timer
        self.auto_tune()
    
    def cleanup(self):
        """Clean up optimization resources."""
        self.stop_optimization()
        self.memory_pool.cleanup()
        logger.info("Performance optimizer cleanup completed")


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


def optimize_performance(func: Callable) -> Callable:
    """Decorator to apply performance optimizations to functions."""
    def wrapper(*args, **kwargs):
        # Use memory pool for tensor operations
        if TORCH_AVAILABLE and any('tensor' in str(type(arg)).lower() for arg in args):
            with torch.no_grad():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    return wrapper


def start_performance_optimization():
    """Start global performance optimization."""
    performance_optimizer.start_optimization()


def stop_performance_optimization():
    """Stop global performance optimization."""
    performance_optimizer.stop_optimization()


def get_optimization_status() -> Dict[str, Any]:
    """Get current optimization status."""
    return performance_optimizer.get_performance_summary()