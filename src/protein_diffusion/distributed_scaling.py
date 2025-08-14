"""
Distributed Scaling Infrastructure for Protein Diffusion

This module implements enterprise-grade scaling capabilities:
- Distributed model serving with load balancing
- Auto-scaling based on demand and resource utilization
- Intelligent caching with hierarchical storage
- Parallel processing with work queue management
- Resource optimization and GPU utilization
"""

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock distributed functionality
    class MockDist:
        @staticmethod
        def init_process_group(*args, **kwargs): pass
        @staticmethod
        def get_rank(): return 0
        @staticmethod
        def get_world_size(): return 1
        @staticmethod
        def barrier(): pass
        @staticmethod
        def all_gather(tensor_list, tensor): pass
    dist = MockDist()
    mp = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def mean(x): return 0.5
        @staticmethod
        def array(x): return x
        @staticmethod
        def concatenate(arrays): return sum(arrays, [])
    np = MockNumpy()
    NUMPY_AVAILABLE = False

import time
import queue
import threading
import logging
import psutil
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import json

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"  # Scale based on current load
    PREDICTIVE = "predictive"  # Scale based on predicted load
    HYBRID = "hybrid"  # Combine reactive and predictive
    MANUAL = "manual"  # Manual scaling only


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"  # In-memory cache
    L2_SSD = "l2_ssd"  # SSD cache
    L3_NETWORK = "l3_network"  # Network cache
    L4_COLD = "l4_cold"  # Cold storage


@dataclass
class WorkerConfig:
    """Configuration for worker processes."""
    worker_id: int
    gpu_id: Optional[int] = None
    cpu_cores: int = 1
    memory_limit_gb: float = 4.0
    batch_size: int = 8
    model_path: str = ""
    cache_size_mb: int = 512


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    timestamp: float
    active_workers: int
    queue_length: int
    avg_response_time: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: Optional[float]
    requests_per_second: float
    error_rate: float
    cost_per_hour: float


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    avg_access_time: float
    size_bytes: int
    max_size_bytes: int
    levels: Dict[str, Dict[str, Any]]


class DistributedCache:
    """Hierarchical distributed cache system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.levels = {}
        self.stats = {level.value: {'hits': 0, 'misses': 0, 'size': 0} for level in CacheLevel}
        
        # Initialize cache levels
        self._init_cache_levels()
        
        # Cache replacement policies
        self.replacement_policies = {
            CacheLevel.L1_MEMORY: LRUCache(config.get('l1_size', 1024)),
            CacheLevel.L2_SSD: LFUCache(config.get('l2_size', 10240)),
            CacheLevel.L3_NETWORK: TimeBasedCache(config.get('l3_size', 102400)),
            CacheLevel.L4_COLD: NoEvictionCache()
        }
    
    def _init_cache_levels(self):
        """Initialize different cache levels."""
        # L1: In-memory cache (fastest)
        self.levels[CacheLevel.L1_MEMORY] = {
            'storage': {},
            'access_times': {},
            'max_size': self.config.get('l1_size', 1024) * 1024 * 1024,  # GB to bytes
            'current_size': 0
        }
        
        # L2: SSD cache (fast)
        self.levels[CacheLevel.L2_SSD] = {
            'storage': {},
            'access_times': {},
            'max_size': self.config.get('l2_size', 10) * 1024 * 1024 * 1024,
            'current_size': 0
        }
        
        # L3: Network cache (medium)
        self.levels[CacheLevel.L3_NETWORK] = {
            'storage': {},
            'access_times': {},
            'max_size': self.config.get('l3_size', 100) * 1024 * 1024 * 1024,
            'current_size': 0
        }
        
        # L4: Cold storage (slow but unlimited)
        self.levels[CacheLevel.L4_COLD] = {
            'storage': {},
            'access_times': {},
            'max_size': float('inf'),
            'current_size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache hierarchy."""
        start_time = time.time()
        
        # Check each level in order
        for level in CacheLevel:
            if key in self.levels[level]['storage']:
                # Cache hit
                value = self.levels[level]['storage'][key]
                self.levels[level]['access_times'][key] = time.time()
                self.stats[level.value]['hits'] += 1
                
                # Promote to higher level if not already there
                self._promote_to_higher_level(key, value, level)
                
                access_time = time.time() - start_time
                logger.debug(f"Cache hit at {level.value} for key {key[:16]}... (access_time: {access_time:.4f}s)")
                return value
        
        # Cache miss at all levels
        for level in CacheLevel:
            self.stats[level.value]['misses'] += 1
        
        logger.debug(f"Cache miss for key {key[:16]}...")
        return None
    
    def set(self, key: str, value: Any, level: CacheLevel = CacheLevel.L1_MEMORY):
        """Store item in cache at specified level."""
        value_size = self._estimate_size(value)
        
        # Check if we need to evict items
        if self._needs_eviction(level, value_size):
            self._evict_items(level, value_size)
        
        # Store the item
        self.levels[level]['storage'][key] = value
        self.levels[level]['access_times'][key] = time.time()
        self.levels[level]['current_size'] += value_size
        
        logger.debug(f"Cached item at {level.value}: {key[:16]}... (size: {value_size} bytes)")
    
    def _promote_to_higher_level(self, key: str, value: Any, current_level: CacheLevel):
        """Promote frequently accessed items to higher cache levels."""
        if current_level == CacheLevel.L1_MEMORY:
            return  # Already at highest level
        
        # Determine target level
        target_levels = {
            CacheLevel.L2_SSD: CacheLevel.L1_MEMORY,
            CacheLevel.L3_NETWORK: CacheLevel.L2_SSD,
            CacheLevel.L4_COLD: CacheLevel.L3_NETWORK
        }
        
        target_level = target_levels.get(current_level)
        if target_level:
            try:
                self.set(key, value, target_level)
            except Exception as e:
                logger.warning(f"Failed to promote cache item: {e}")
    
    def _needs_eviction(self, level: CacheLevel, new_item_size: int) -> bool:
        """Check if eviction is needed."""
        level_info = self.levels[level]
        return level_info['current_size'] + new_item_size > level_info['max_size']
    
    def _evict_items(self, level: CacheLevel, required_space: int):
        """Evict items using replacement policy."""
        policy = self.replacement_policies[level]
        level_info = self.levels[level]
        
        evicted_size = 0
        while evicted_size < required_space and level_info['storage']:
            key_to_evict = policy.select_eviction_candidate(
                level_info['storage'], 
                level_info['access_times']
            )
            
            if key_to_evict:
                value = level_info['storage'].pop(key_to_evict)
                level_info['access_times'].pop(key_to_evict, None)
                
                evicted_size += self._estimate_size(value)
                level_info['current_size'] -= self._estimate_size(value)
                
                # Try to store in lower level
                lower_level = self._get_lower_level(level)
                if lower_level:
                    try:
                        self.set(key_to_evict, value, lower_level)
                    except Exception as e:
                        logger.warning(f"Failed to store evicted item in lower level: {e}")
            else:
                break
    
    def _get_lower_level(self, current_level: CacheLevel) -> Optional[CacheLevel]:
        """Get the next lower cache level."""
        level_hierarchy = [
            CacheLevel.L1_MEMORY,
            CacheLevel.L2_SSD,
            CacheLevel.L3_NETWORK,
            CacheLevel.L4_COLD
        ]
        
        try:
            current_index = level_hierarchy.index(current_level)
            if current_index < len(level_hierarchy) - 1:
                return level_hierarchy[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value."""
        try:
            if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            else:
                # Rough estimate for other types
                return len(str(value)) * 4
        except Exception:
            return 1024  # Default estimate
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        total_hits = sum(stats['hits'] for stats in self.stats.values())
        total_misses = sum(stats['misses'] for stats in self.stats.values())
        total_requests = total_hits + total_misses
        
        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        miss_rate = total_misses / total_requests if total_requests > 0 else 0.0
        
        total_size = sum(level['current_size'] for level in self.levels.values())
        max_total_size = sum(level['max_size'] for level in self.levels.values() if level['max_size'] != float('inf'))
        
        level_stats = {}
        for level in CacheLevel:
            level_info = self.levels[level]
            level_stats[level.value] = {
                'hits': self.stats[level.value]['hits'],
                'misses': self.stats[level.value]['misses'],
                'size_bytes': level_info['current_size'],
                'max_size_bytes': level_info['max_size'],
                'utilization': level_info['current_size'] / level_info['max_size'] if level_info['max_size'] != float('inf') else 0.0
            }
        
        return CacheStats(
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            eviction_rate=0.0,  # Would calculate from eviction events
            avg_access_time=0.001,  # Would measure actual access times
            size_bytes=int(total_size),
            max_size_bytes=int(max_total_size) if max_total_size != float('inf') else -1,
            levels=level_stats
        )


class CacheReplacementPolicy(ABC):
    """Abstract base class for cache replacement policies."""
    
    @abstractmethod
    def select_eviction_candidate(self, storage: Dict, access_times: Dict) -> Optional[str]:
        """Select key to evict."""
        pass


class LRUCache(CacheReplacementPolicy):
    """Least Recently Used cache replacement."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
    
    def select_eviction_candidate(self, storage: Dict, access_times: Dict) -> Optional[str]:
        if not access_times:
            return None
        
        # Find least recently used item
        lru_key = min(access_times.keys(), key=lambda k: access_times[k])
        return lru_key


class LFUCache(CacheReplacementPolicy):
    """Least Frequently Used cache replacement."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.frequency_counts = {}
    
    def select_eviction_candidate(self, storage: Dict, access_times: Dict) -> Optional[str]:
        if not storage:
            return None
        
        # Update frequency counts
        for key in storage:
            if key not in self.frequency_counts:
                self.frequency_counts[key] = 0
            self.frequency_counts[key] += 1
        
        # Find least frequently used item
        lfu_key = min(self.frequency_counts.keys(), key=lambda k: self.frequency_counts[k])
        return lfu_key


class TimeBasedCache(CacheReplacementPolicy):
    """Time-based cache replacement (TTL)."""
    
    def __init__(self, capacity: int, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
    
    def select_eviction_candidate(self, storage: Dict, access_times: Dict) -> Optional[str]:
        current_time = time.time()
        
        # Find expired items first
        for key, access_time in access_times.items():
            if current_time - access_time > self.ttl_seconds:
                return key
        
        # If no expired items, use LRU
        if access_times:
            return min(access_times.keys(), key=lambda k: access_times[k])
        
        return None


class NoEvictionCache(CacheReplacementPolicy):
    """No eviction policy for cold storage."""
    
    def select_eviction_candidate(self, storage: Dict, access_times: Dict) -> Optional[str]:
        return None  # Never evict


class WorkerProcess:
    """Individual worker process for distributed processing."""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.model = None
        self.cache = None
        self.stats = {
            'requests_processed': 0,
            'total_processing_time': 0.0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.is_healthy = True
        self.last_heartbeat = time.time()
    
    def initialize(self):
        """Initialize worker with model and resources."""
        try:
            # Set GPU if available
            if TORCH_AVAILABLE and self.config.gpu_id is not None:
                torch.cuda.set_device(self.config.gpu_id)
            
            # Load model (placeholder)
            logger.info(f"Worker {self.config.worker_id} initializing...")
            
            # Initialize local cache
            cache_config = {'l1_size': self.config.cache_size_mb / 1024}  # Convert MB to GB
            self.cache = DistributedCache(cache_config)
            
            self.is_healthy = True
            logger.info(f"Worker {self.config.worker_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Worker {self.config.worker_id} initialization failed: {e}")
            self.is_healthy = False
            raise
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process generation request."""
        start_time = time.time()
        
        try:
            # Update heartbeat
            self.last_heartbeat = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.debug(f"Worker {self.config.worker_id}: Cache hit for request")
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # Process request (placeholder)
            result = self._generate_response(request)
            
            # Cache result
            self.cache.set(cache_key, result)
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['requests_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            
            logger.debug(f"Worker {self.config.worker_id}: Processed request in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Worker {self.config.worker_id} processing error: {e}")
            raise
    
    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        # Create deterministic key from request parameters
        key_data = {
            'motif': request.get('motif', ''),
            'num_samples': request.get('num_samples', 1),
            'temperature': request.get('temperature', 1.0),
            'max_length': request.get('max_length', 100)
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _generate_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response (placeholder implementation)."""
        # Simulate processing time
        time.sleep(0.1)
        
        num_samples = request.get('num_samples', 1)
        max_length = request.get('max_length', 100)
        
        # Generate mock sequences
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        sequences = []
        
        for i in range(num_samples):
            sequence = ''.join(np.random.choice(list(amino_acids), size=max_length))
            sequences.append({
                'sequence': sequence,
                'confidence': 0.8 + 0.2 * np.random.random(),
                'worker_id': self.config.worker_id
            })
        
        return {
            'sequences': sequences,
            'processing_time': 0.1,
            'worker_id': self.config.worker_id,
            'timestamp': time.time()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get worker health status."""
        avg_processing_time = (
            self.stats['total_processing_time'] / self.stats['requests_processed']
            if self.stats['requests_processed'] > 0 else 0.0
        )
        
        cache_stats = self.cache.get_stats() if self.cache else None
        
        return {
            'worker_id': self.config.worker_id,
            'is_healthy': self.is_healthy,
            'last_heartbeat': self.last_heartbeat,
            'requests_processed': self.stats['requests_processed'],
            'avg_processing_time': avg_processing_time,
            'error_rate': self.stats['errors'] / max(1, self.stats['requests_processed']),
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'cache_stats': cache_stats.__dict__ if cache_stats else None
        }


class LoadBalancer:
    """Intelligent load balancer for distributing requests."""
    
    def __init__(self, workers: List[WorkerProcess]):
        self.workers = workers
        self.worker_loads = {worker.config.worker_id: 0 for worker in workers}
        self.worker_health = {worker.config.worker_id: True for worker in workers}
        self.request_history = []
        
    def select_worker(self, request: Dict[str, Any]) -> Optional[WorkerProcess]:
        """Select best worker for request using intelligent routing."""
        healthy_workers = [w for w in self.workers if self.worker_health[w.config.worker_id]]
        
        if not healthy_workers:
            logger.error("No healthy workers available")
            return None
        
        # Strategy 1: Least loaded worker
        least_loaded_worker = min(healthy_workers, key=lambda w: self.worker_loads[w.config.worker_id])
        
        # Strategy 2: Consider GPU affinity for large requests
        if request.get('num_samples', 1) > 50:
            gpu_workers = [w for w in healthy_workers if w.config.gpu_id is not None]
            if gpu_workers:
                least_loaded_worker = min(gpu_workers, key=lambda w: self.worker_loads[w.config.worker_id])
        
        # Strategy 3: Cache affinity (route similar requests to same worker)
        cache_key = self._generate_routing_key(request)
        for worker in healthy_workers:
            if hasattr(worker, 'cache') and worker.cache.get(cache_key):
                return worker
        
        return least_loaded_worker
    
    def _generate_routing_key(self, request: Dict[str, Any]) -> str:
        """Generate routing key for cache affinity."""
        key_data = {
            'motif': request.get('motif', ''),
            'temperature': round(request.get('temperature', 1.0), 1)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def update_worker_load(self, worker_id: int, load_delta: int):
        """Update worker load."""
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] += load_delta
            self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id])
    
    def update_worker_health(self, worker_id: int, is_healthy: bool):
        """Update worker health status."""
        self.worker_health[worker_id] = is_healthy
        if not is_healthy:
            logger.warning(f"Worker {worker_id} marked as unhealthy")
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution."""
        total_load = sum(self.worker_loads.values())
        healthy_workers = sum(1 for h in self.worker_health.values() if h)
        
        return {
            'total_load': total_load,
            'worker_loads': self.worker_loads.copy(),
            'worker_health': self.worker_health.copy(),
            'healthy_workers': healthy_workers,
            'avg_load_per_worker': total_load / max(1, healthy_workers)
        }


class AutoScaler:
    """Automatic scaling based on metrics and predictions."""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.metrics_history = []
        self.scaling_decisions = []
        self.min_workers = 1
        self.max_workers = 20
        self.scale_up_threshold = 0.8  # CPU/memory utilization
        self.scale_down_threshold = 0.3
        self.cooldown_period = 300  # 5 minutes
        self.last_scaling_action = 0
    
    def should_scale(self, current_metrics: ScalingMetrics) -> Tuple[bool, int]:
        """Determine if scaling is needed and by how many workers."""
        self.metrics_history.append(current_metrics)
        
        # Keep only recent history
        cutoff_time = time.time() - 3600  # 1 hour
        self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        # Check cooldown period
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return False, 0
        
        # Reactive scaling
        scale_decision = 0
        if self.strategy in [ScalingStrategy.REACTIVE, ScalingStrategy.HYBRID]:
            scale_decision = self._reactive_scaling(current_metrics)
        
        # Predictive scaling
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            predictive_decision = self._predictive_scaling(current_metrics)
            if abs(predictive_decision) > abs(scale_decision):
                scale_decision = predictive_decision
        
        # Apply constraints
        new_worker_count = current_metrics.active_workers + scale_decision
        new_worker_count = max(self.min_workers, min(self.max_workers, new_worker_count))
        final_scale_decision = new_worker_count - current_metrics.active_workers
        
        if final_scale_decision != 0:
            self.last_scaling_action = time.time()
            self.scaling_decisions.append({
                'timestamp': time.time(),
                'decision': final_scale_decision,
                'reason': self._get_scaling_reason(current_metrics, final_scale_decision),
                'metrics': current_metrics
            })
        
        return final_scale_decision != 0, final_scale_decision
    
    def _reactive_scaling(self, metrics: ScalingMetrics) -> int:
        """Reactive scaling based on current metrics."""
        # Scale up conditions
        if (metrics.cpu_utilization > self.scale_up_threshold or
            metrics.memory_utilization > self.scale_up_threshold or
            metrics.queue_length > metrics.active_workers * 2 or
            metrics.avg_response_time > 10.0):
            return min(3, int(metrics.active_workers * 0.5))  # Scale up by 50%, max 3
        
        # Scale down conditions
        if (metrics.cpu_utilization < self.scale_down_threshold and
            metrics.memory_utilization < self.scale_down_threshold and
            metrics.queue_length < metrics.active_workers * 0.5 and
            metrics.avg_response_time < 2.0):
            return -min(2, int(metrics.active_workers * 0.3))  # Scale down by 30%, max 2
        
        return 0
    
    def _predictive_scaling(self, metrics: ScalingMetrics) -> int:
        """Predictive scaling based on trends and patterns."""
        if len(self.metrics_history) < 10:
            return 0  # Need more data for prediction
        
        # Analyze trends
        recent_rps = [m.requests_per_second for m in self.metrics_history[-10:]]
        recent_response_times = [m.avg_response_time for m in self.metrics_history[-10:]]
        
        # Simple trend analysis
        rps_trend = (recent_rps[-1] - recent_rps[0]) / len(recent_rps)
        response_time_trend = (recent_response_times[-1] - recent_response_times[0]) / len(recent_response_times)
        
        # Predict future load
        if rps_trend > 0.5 and response_time_trend > 0.1:
            # Load is increasing, scale up proactively
            return min(2, int(metrics.active_workers * 0.3))
        elif rps_trend < -0.5 and response_time_trend < -0.1:
            # Load is decreasing, scale down proactively
            return -min(1, int(metrics.active_workers * 0.2))
        
        return 0
    
    def _get_scaling_reason(self, metrics: ScalingMetrics, decision: int) -> str:
        """Get human-readable scaling reason."""
        if decision > 0:
            reasons = []
            if metrics.cpu_utilization > self.scale_up_threshold:
                reasons.append(f"high CPU ({metrics.cpu_utilization:.1%})")
            if metrics.memory_utilization > self.scale_up_threshold:
                reasons.append(f"high memory ({metrics.memory_utilization:.1%})")
            if metrics.queue_length > metrics.active_workers * 2:
                reasons.append(f"high queue length ({metrics.queue_length})")
            if metrics.avg_response_time > 10.0:
                reasons.append(f"high response time ({metrics.avg_response_time:.1f}s)")
            
            return f"Scale up: {', '.join(reasons)}"
        
        elif decision < 0:
            return f"Scale down: low utilization (CPU: {metrics.cpu_utilization:.1%}, Memory: {metrics.memory_utilization:.1%})"
        
        return "No scaling needed"
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling decision history."""
        cutoff_time = time.time() - (hours * 3600)
        return [d for d in self.scaling_decisions if d['timestamp'] >= cutoff_time]


class DistributedProcessingManager:
    """Manages distributed processing across multiple workers."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers = []
        self.load_balancer = None
        self.auto_scaler = AutoScaler()
        self.global_cache = DistributedCache({'l1_size': 2, 'l2_size': 20, 'l3_size': 200})
        self.request_queue = queue.Queue()
        self.result_futures = {}
        self.metrics_collector = MetricsCollector()
        
        # Thread pool for managing workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers * 2)
        
    def initialize(self):
        """Initialize the distributed processing system."""
        logger.info(f"Initializing distributed processing with {self.num_workers} workers")
        
        # Create and initialize workers
        for i in range(self.num_workers):
            config = WorkerConfig(
                worker_id=i,
                gpu_id=i if TORCH_AVAILABLE and torch.cuda.is_available() else None,
                cpu_cores=2,
                memory_limit_gb=4.0,
                batch_size=8
            )
            
            worker = WorkerProcess(config)
            try:
                worker.initialize()
                self.workers.append(worker)
            except Exception as e:
                logger.error(f"Failed to initialize worker {i}: {e}")
        
        # Initialize load balancer
        if self.workers:
            self.load_balancer = LoadBalancer(self.workers)
            logger.info(f"Load balancer initialized with {len(self.workers)} workers")
        else:
            raise RuntimeError("No workers successfully initialized")
        
        # Start metrics collection
        self.metrics_collector.start()
    
    def process_request_distributed(self, request: Dict[str, Any]) -> Future:
        """Process request using distributed workers."""
        # Check global cache first
        cache_key = self._generate_request_key(request)
        cached_result = self.global_cache.get(cache_key)
        
        if cached_result:
            # Return completed future with cached result
            future = Future()
            future.set_result(cached_result)
            return future
        
        # Submit to worker
        future = self.executor.submit(self._process_with_worker, request)
        return future
    
    def _process_with_worker(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request with selected worker."""
        # Select worker
        worker = self.load_balancer.select_worker(request)
        if not worker:
            raise RuntimeError("No healthy workers available")
        
        # Update load
        self.load_balancer.update_worker_load(worker.config.worker_id, 1)
        
        try:
            # Process request
            result = worker.process_request(request)
            
            # Cache result globally
            cache_key = self._generate_request_key(request)
            self.global_cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            # Mark worker as unhealthy if it fails
            self.load_balancer.update_worker_health(worker.config.worker_id, False)
            raise
        finally:
            # Update load
            self.load_balancer.update_worker_load(worker.config.worker_id, -1)
    
    def _generate_request_key(self, request: Dict[str, Any]) -> str:
        """Generate unique key for request caching."""
        key_data = {
            'motif': request.get('motif', ''),
            'num_samples': request.get('num_samples', 1),
            'temperature': request.get('temperature', 1.0),
            'max_length': request.get('max_length', 100),
            'sampling_method': request.get('sampling_method', 'ddpm')
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        current_metrics = self.metrics_collector.get_current_metrics()
        current_metrics.active_workers = len([w for w in self.workers if w.is_healthy])
        
        should_scale, scale_amount = self.auto_scaler.should_scale(current_metrics)
        
        if should_scale:
            if scale_amount > 0:
                self._scale_up(scale_amount)
            else:
                self._scale_down(abs(scale_amount))
    
    def _scale_up(self, num_workers: int):
        """Scale up by adding workers."""
        logger.info(f"Scaling up by {num_workers} workers")
        
        for i in range(num_workers):
            new_worker_id = len(self.workers)
            config = WorkerConfig(
                worker_id=new_worker_id,
                gpu_id=new_worker_id % torch.cuda.device_count() if TORCH_AVAILABLE and torch.cuda.is_available() else None,
                cpu_cores=2,
                memory_limit_gb=4.0
            )
            
            try:
                worker = WorkerProcess(config)
                worker.initialize()
                self.workers.append(worker)
                self.load_balancer.workers.append(worker)
                self.load_balancer.worker_loads[worker.config.worker_id] = 0
                self.load_balancer.worker_health[worker.config.worker_id] = True
                
                logger.info(f"Added worker {new_worker_id}")
            except Exception as e:
                logger.error(f"Failed to add worker {new_worker_id}: {e}")
    
    def _scale_down(self, num_workers: int):
        """Scale down by removing workers."""
        logger.info(f"Scaling down by {num_workers} workers")
        
        # Remove least loaded workers
        workers_to_remove = sorted(
            self.workers,
            key=lambda w: self.load_balancer.worker_loads[w.config.worker_id]
        )[:num_workers]
        
        for worker in workers_to_remove:
            self.workers.remove(worker)
            self.load_balancer.workers.remove(worker)
            del self.load_balancer.worker_loads[worker.config.worker_id]
            del self.load_balancer.worker_health[worker.config.worker_id]
            
            logger.info(f"Removed worker {worker.config.worker_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        worker_statuses = [worker.get_health_status() for worker in self.workers]
        healthy_workers = sum(1 for status in worker_statuses if status['is_healthy'])
        
        load_distribution = self.load_balancer.get_load_distribution()
        cache_stats = self.global_cache.get_stats()
        current_metrics = self.metrics_collector.get_current_metrics()
        
        return {
            'timestamp': time.time(),
            'total_workers': len(self.workers),
            'healthy_workers': healthy_workers,
            'worker_statuses': worker_statuses,
            'load_distribution': load_distribution,
            'cache_stats': cache_stats.__dict__,
            'current_metrics': current_metrics.__dict__,
            'auto_scaling': {
                'enabled': True,
                'strategy': self.auto_scaler.strategy.value,
                'recent_decisions': self.auto_scaler.get_scaling_history(hours=1)
            }
        }


class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self):
        self.metrics_history = []
        self.collection_interval = 30  # seconds
        self.is_running = False
        self.collection_thread = None
    
    def start(self):
        """Start metrics collection."""
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while self.is_running:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent history (24 hours)
                cutoff_time = time.time() - 86400
                self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
                
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics (if available)
        gpu_utilization = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_utilization = torch.cuda.utilization()
            except Exception:
                gpu_utilization = None
        
        # Application metrics (placeholders)
        active_workers = 4  # Would get from actual worker manager
        queue_length = 0   # Would get from actual queue
        avg_response_time = 2.5  # Would calculate from request history
        requests_per_second = 10.0  # Would calculate from request rate
        error_rate = 0.01  # Would calculate from error logs
        cost_per_hour = 5.0  # Would calculate from resource usage
        
        return ScalingMetrics(
            timestamp=time.time(),
            active_workers=active_workers,
            queue_length=queue_length,
            avg_response_time=avg_response_time,
            cpu_utilization=cpu_percent / 100.0,
            memory_utilization=memory.percent / 100.0,
            gpu_utilization=gpu_utilization / 100.0 if gpu_utilization else None,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            cost_per_hour=cost_per_hour
        )
    
    def get_current_metrics(self) -> ScalingMetrics:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        else:
            return self._collect_current_metrics()
    
    def get_metrics_history(self, hours: int = 1) -> List[ScalingMetrics]:
        """Get metrics history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]