"""
Performance optimization utilities for protein diffusion models.

This module provides caching, model optimization, batch processing,
and other performance enhancements for production deployments.
"""

import time
import pickle
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
from functools import wraps, lru_cache
from collections import OrderedDict
import logging
import json

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self._lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size(),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate()
        }

class RedisCache:
    """Redis-based cache for distributed caching."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 expire_time: int = 3600, prefix: str = "protein_diffusion"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.expire_time = expire_time
        self.prefix = prefix
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = self.client.get(redis_key)
            
            if data is not None:
                self.hits += 1
                return pickle.loads(data)
            else:
                self.misses += 1
                return None
        
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put value in Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = pickle.dumps(value)
            return self.client.setex(redis_key, self.expire_time, data)
        
        except Exception as e:
            logger.warning(f"Redis put error: {e}")
            return False
    
    def clear(self) -> None:
        """Clear cache entries with our prefix."""
        try:
            pattern = f"{self.prefix}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            self.hits = 0
            self.misses = 0
        
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "redis_info": self._get_redis_info()
        }
    
    def _get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        try:
            info = self.client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "connected_clients": info.get("connected_clients"),
            }
        except Exception:
            return {}

class ModelCache:
    """Specialized cache for model outputs."""
    
    def __init__(self, cache_backend: Union[LRUCache, RedisCache], 
                 enable_sequence_cache: bool = True,
                 enable_structure_cache: bool = True):
        self.cache = cache_backend
        self.enable_sequence_cache = enable_sequence_cache
        self.enable_structure_cache = enable_structure_cache
    
    def _hash_inputs(self, *args, **kwargs) -> str:
        """Create hash key from inputs."""
        # Create a deterministic hash from inputs
        content = json.dumps([args, sorted(kwargs.items())], sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()
    
    def cache_sequence_generation(self, func: Callable) -> Callable:
        """Decorator to cache sequence generation results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_sequence_cache:
                return func(*args, **kwargs)
            
            cache_key = f"seq_gen:{self._hash_inputs(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for sequence generation: {cache_key[:16]}...")
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            self.cache.put(cache_key, result)
            logger.debug(f"Cached sequence generation result: {cache_key[:16]}...")
            
            return result
        
        return wrapper
    
    def cache_structure_prediction(self, func: Callable) -> Callable:
        """Decorator to cache structure prediction results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_structure_cache:
                return func(*args, **kwargs)
            
            cache_key = f"struct_pred:{self._hash_inputs(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for structure prediction: {cache_key[:16]}...")
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            self.cache.put(cache_key, result)
            logger.debug(f"Cached structure prediction result: {cache_key[:16]}...")
            
            return result
        
        return wrapper
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()

class BatchProcessor:
    """Optimize batch processing for better throughput."""
    
    def __init__(self, max_batch_size: int = 32, optimal_batch_size: int = 16):
        self.max_batch_size = max_batch_size
        self.optimal_batch_size = optimal_batch_size
    
    def optimize_batches(self, items: List[Any], batch_size: Optional[int] = None) -> List[List[Any]]:
        """
        Split items into optimally-sized batches.
        
        Args:
            items: Items to batch
            batch_size: Override batch size
            
        Returns:
            List of batches
        """
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        batch_size = min(batch_size, self.max_batch_size)
        
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def dynamic_batch_size(self, item_sizes: List[int], memory_limit: float = 8.0) -> int:
        """
        Determine optimal batch size based on item sizes and memory limit.
        
        Args:
            item_sizes: Sizes of items (e.g., sequence lengths)
            memory_limit: Memory limit in GB
            
        Returns:
            Optimal batch size
        """
        if not item_sizes:
            return self.optimal_batch_size
        
        avg_size = sum(item_sizes) / len(item_sizes)
        max_size = max(item_sizes)
        
        # Estimate memory usage per item (rough approximation)
        memory_per_item_gb = (avg_size * 4 * 1024) / 1e9  # 4 bytes per token, 1024 hidden dim
        
        # Calculate max batch size that fits in memory
        max_batch_from_memory = int(memory_limit / memory_per_item_gb)
        
        # Use conservative estimate
        optimal = min(
            max_batch_from_memory // 2,  # Use half of available memory
            self.max_batch_size,
            max(1, 32 if max_size < 128 else 16 if max_size < 256 else 8)
        )
        
        return max(1, optimal)

class ModelOptimizer:
    """Optimize PyTorch models for inference."""
    
    def __init__(self):
        self.optimizations_applied = []
    
    def optimize_for_inference(self, model: nn.Module, 
                             use_jit: bool = True,
                             use_half_precision: bool = False,
                             optimize_for_mobile: bool = False) -> nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: PyTorch model
            use_jit: Apply TorchScript JIT compilation
            use_half_precision: Convert to half precision
            optimize_for_mobile: Optimize for mobile deployment
            
        Returns:
            Optimized model
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning original model")
            return model
        
        optimized_model = model
        
        # Set to evaluation mode
        optimized_model.eval()
        self.optimizations_applied.append("eval_mode")
        
        # Half precision
        if use_half_precision and torch.cuda.is_available():
            optimized_model = optimized_model.half()
            self.optimizations_applied.append("half_precision")
            logger.info("Applied half precision optimization")
        
        # TorchScript JIT compilation
        if use_jit:
            try:
                # Create example inputs for tracing
                example_input = self._create_example_input(optimized_model)
                if example_input is not None:
                    optimized_model = torch.jit.trace(optimized_model, example_input)
                    self.optimizations_applied.append("torchscript_trace")
                    logger.info("Applied TorchScript tracing")
                else:
                    # Fallback to scripting
                    optimized_model = torch.jit.script(optimized_model)
                    self.optimizations_applied.append("torchscript_script")
                    logger.info("Applied TorchScript scripting")
            
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
        
        # Mobile optimization
        if optimize_for_mobile and hasattr(torch.utils, 'mobile_optimizer'):
            try:
                optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(optimized_model)
                self.optimizations_applied.append("mobile_optimization")
                logger.info("Applied mobile optimization")
            except Exception as e:
                logger.warning(f"Mobile optimization failed: {e}")
        
        logger.info(f"Model optimization complete. Applied: {self.optimizations_applied}")
        return optimized_model
    
    def _create_example_input(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Create example input for model tracing."""
        try:
            # Try to infer input shape from model
            if hasattr(model, 'config'):
                config = model.config
                batch_size = 1
                seq_len = getattr(config, 'max_seq_len', 256)
                
                if hasattr(model, 'forward'):
                    # Inspect forward method signature
                    import inspect
                    sig = inspect.signature(model.forward)
                    params = list(sig.parameters.keys())
                    
                    if 'input_ids' in params:
                        return torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
            
            return None
        
        except Exception as e:
            logger.debug(f"Could not create example input: {e}")
            return None
    
    def benchmark_model(self, model: nn.Module, input_tensor: torch.Tensor, 
                       num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            input_tensor: Example input
            num_runs: Number of runs for benchmarking
            
        Returns:
            Performance metrics
        """
        if not TORCH_AVAILABLE:
            return {}
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = 1.0 / avg_time
        
        return {
            "total_time": total_time,
            "average_time": avg_time,
            "throughput_samples_per_second": throughput,
            "num_runs": num_runs
        }

class MemoryOptimizer:
    """Optimize memory usage."""
    
    @staticmethod
    def enable_memory_efficient_attention():
        """Enable memory efficient attention if available."""
        if TORCH_AVAILABLE:
            try:
                # Enable memory efficient attention in PyTorch 2.0+
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Enabled Flash SDP (Scaled Dot Product) attention")
                return True
            except Exception as e:
                logger.debug(f"Could not enable Flash SDP: {e}")
        
        return False
    
    @staticmethod
    def optimize_cuda_settings():
        """Optimize CUDA settings for memory and performance."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Enable memory pool
            try:
                torch.cuda.empty_cache()
                
                # Set memory fraction if needed
                # torch.cuda.set_per_process_memory_fraction(0.8)
                
                # Enable cudnn benchmark for consistent input sizes
                torch.backends.cudnn.benchmark = True
                
                logger.info("Optimized CUDA settings")
                return True
            
            except Exception as e:
                logger.warning(f"CUDA optimization failed: {e}")
        
        return False
    
    @staticmethod
    def get_memory_stats() -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_stats = {
                    "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated(i) / 1e9,
                    "max_reserved_gb": torch.cuda.max_memory_reserved(i) / 1e9,
                }
                stats[f"gpu_{i}"] = device_stats
        
        return stats

class AsyncProcessor:
    """Asynchronous processing for better throughput."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = None
    
    def __enter__(self):
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
    
    def submit_batch(self, func: Callable, batches: List[Any]) -> List[Any]:
        """Submit batches for parallel processing."""
        if not self._executor:
            raise RuntimeError("AsyncProcessor not properly initialized")
        
        futures = []
        for batch in batches:
            future = self._executor.submit(func, batch)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                results.append(None)
        
        return results

class PerformanceManager:
    """Manage all performance optimizations."""
    
    def __init__(self, 
                 cache_backend: str = "lru",
                 cache_size: int = 1000,
                 redis_config: Optional[Dict[str, Any]] = None):
        
        # Initialize cache
        if cache_backend == "redis" and REDIS_AVAILABLE:
            redis_config = redis_config or {}
            self.cache = RedisCache(**redis_config)
        else:
            self.cache = LRUCache(max_size=cache_size)
        
        self.model_cache = ModelCache(self.cache)
        self.batch_processor = BatchProcessor()
        self.model_optimizer = ModelOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        
        # Apply system optimizations
        self.memory_optimizer.enable_memory_efficient_attention()
        self.memory_optimizer.optimize_cuda_settings()
    
    def optimize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Optimize a model for inference."""
        return self.model_optimizer.optimize_for_inference(model, **kwargs)
    
    def get_cache_decorator(self, cache_type: str = "sequence") -> Callable:
        """Get cache decorator for specific operation type."""
        if cache_type == "sequence":
            return self.model_cache.cache_sequence_generation
        elif cache_type == "structure":
            return self.model_cache.cache_structure_prediction
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "cache_stats": self.model_cache.get_cache_stats(),
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "optimizations_applied": self.model_optimizer.optimizations_applied,
            "system_info": {
                "torch_available": TORCH_AVAILABLE,
                "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
                "redis_available": REDIS_AVAILABLE,
            }
        }
    
    def clear_caches(self):
        """Clear all caches."""
        self.cache.clear()
        logger.info("All caches cleared")

# Global performance manager
_performance_manager = None

def get_performance_manager(**kwargs) -> PerformanceManager:
    """Get global performance manager instance."""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = PerformanceManager(**kwargs)
    return _performance_manager

# Convenience decorators using global manager
def cache_sequence_generation(func: Callable) -> Callable:
    """Convenience decorator for caching sequence generation."""
    manager = get_performance_manager()
    return manager.get_cache_decorator("sequence")(func)

def cache_structure_prediction(func: Callable) -> Callable:
    """Convenience decorator for caching structure prediction."""
    manager = get_performance_manager()
    return manager.get_cache_decorator("structure")(func)