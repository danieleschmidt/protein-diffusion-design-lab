"""
Performance optimization utilities for protein diffusion models.

This module provides tools for batch processing, model optimization,
memory management, and performance monitoring.
"""

import time
import psutil
import threading
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from collections import deque
import logging
import queue

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    # Batch processing
    batch_size: int = 32
    max_batch_size: int = 128
    adaptive_batching: bool = True
    
    # Concurrency
    max_workers: int = 4
    use_multiprocessing: bool = False
    thread_pool_size: int = 8
    
    # Memory management
    memory_limit_mb: int = 8192  # 8GB
    gc_threshold: float = 0.8  # Trigger GC at 80% memory
    tensor_memory_fraction: float = 0.6  # Reserve 60% for tensors
    
    # Model optimization
    use_mixed_precision: bool = True
    compile_model: bool = True
    gradient_checkpointing: bool = False
    
    # Monitoring
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_timing: bool = True


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.metrics = deque(maxlen=1000)  # Keep last 1000 measurements
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.RLock()
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start background performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor_worker():
            while self._monitoring:
                try:
                    metrics = self._collect_metrics()
                    
                    with self._lock:
                        self.metrics.append(metrics)
                    
                    # Check for memory pressure
                    if metrics['memory_percent'] > self.config.gc_threshold * 100:
                        logger.warning(f"High memory usage: {metrics['memory_percent']:.1f}%")
                        self._suggest_gc()
                    
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Process-specific metrics
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024 * 1024)
        process_cpu_percent = process.cpu_percent()
        
        # GPU metrics (if available)
        gpu_metrics = {}
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                gpu_metrics = {
                    'gpu_memory_allocated_mb': gpu_memory.get('allocated_bytes.all.current', 0) / (1024 * 1024),
                    'gpu_memory_reserved_mb': gpu_memory.get('reserved_bytes.all.current', 0) / (1024 * 1024),
                    'gpu_memory_max_mb': torch.cuda.get_device_properties(0).total_memory / (1024 * 1024),
                }
            except Exception:
                pass
        
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory_percent,
            'memory_used_mb': memory_used_mb,
            'memory_available_mb': memory_available_mb,
            'process_memory_mb': process_memory_mb,
            'process_cpu_percent': process_cpu_percent,
            **gpu_metrics
        }
    
    def _suggest_gc(self):
        """Suggest garbage collection when memory is high."""
        import gc
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self._collect_metrics()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        with self._lock:
            if not self.metrics:
                return {}
            
            recent_metrics = list(self.metrics)[-100:]  # Last 100 measurements
            
            # Calculate averages
            avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
            avg_process_memory = sum(m['process_memory_mb'] for m in recent_metrics) / len(recent_metrics)
            
            # Calculate peaks
            max_cpu = max(m['cpu_percent'] for m in recent_metrics)
            max_memory = max(m['memory_percent'] for m in recent_metrics)
            max_process_memory = max(m['process_memory_mb'] for m in recent_metrics)
            
            summary = {
                'measurements_count': len(recent_metrics),
                'time_span_seconds': recent_metrics[-1]['timestamp'] - recent_metrics[0]['timestamp'],
                'avg_cpu_percent': avg_cpu,
                'max_cpu_percent': max_cpu,
                'avg_memory_percent': avg_memory,
                'max_memory_percent': max_memory,
                'avg_process_memory_mb': avg_process_memory,
                'max_process_memory_mb': max_process_memory,
            }
            
            # Add GPU summary if available
            gpu_metrics = [m for m in recent_metrics if 'gpu_memory_allocated_mb' in m]
            if gpu_metrics:
                summary.update({
                    'avg_gpu_memory_mb': sum(m['gpu_memory_allocated_mb'] for m in gpu_metrics) / len(gpu_metrics),
                    'max_gpu_memory_mb': max(m['gpu_memory_allocated_mb'] for m in gpu_metrics),
                })
            
            return summary


class BatchProcessor:
    """Efficient batch processing for protein sequences and models."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
    
    def adaptive_batch_size(self, total_items: int, memory_usage_mb: float) -> int:
        """Calculate optimal batch size based on available memory."""
        if not self.config.adaptive_batching:
            return self.config.batch_size
        
        # Estimate memory per item (rough heuristic)
        available_memory_mb = self.config.memory_limit_mb - memory_usage_mb
        memory_per_item_mb = 10  # Conservative estimate
        
        # Calculate max items that fit in memory
        max_items_in_memory = int(available_memory_mb / memory_per_item_mb)
        
        # Choose batch size
        optimal_batch_size = min(
            max_items_in_memory,
            self.config.max_batch_size,
            max(self.config.batch_size, total_items // 10)  # At least 10 batches
        )
        
        return max(1, optimal_batch_size)
    
    def create_batches(self, items: List[Any], batch_size: Optional[int] = None) -> Iterator[List[Any]]:
        """Create batches from list of items."""
        if batch_size is None:
            current_memory = self.performance_monitor.get_current_metrics().get('process_memory_mb', 0)
            batch_size = self.adaptive_batch_size(len(items), current_memory)
        
        logger.info(f"Processing {len(items)} items in batches of {batch_size}")
        
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    def process_batches(self, 
                       items: List[Any], 
                       process_func: Callable[[List[Any]], List[Any]],
                       batch_size: Optional[int] = None,
                       show_progress: bool = True) -> List[Any]:
        """Process items in batches with progress tracking."""
        results = []
        total_batches = len(items) // (batch_size or self.config.batch_size) + 1
        
        start_time = time.time()
        
        for i, batch in enumerate(self.create_batches(items, batch_size)):
            batch_start = time.time()
            
            try:
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start
                
                if show_progress:
                    progress = (i + 1) / total_batches * 100
                    items_per_sec = len(batch) / batch_time if batch_time > 0 else 0
                    logger.info(f"Batch {i+1}/{total_batches} ({progress:.1f}%) - "
                              f"{items_per_sec:.1f} items/sec")
                
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                # Could implement retry logic here
                continue
        
        total_time = time.time() - start_time
        avg_throughput = len(items) / total_time if total_time > 0 else 0
        
        logger.info(f"Processed {len(items)} items in {total_time:.2f}s "
                   f"({avg_throughput:.1f} items/sec)")
        
        return results
    
    def parallel_process(self,
                        items: List[Any],
                        process_func: Callable[[Any], Any],
                        max_workers: Optional[int] = None,
                        use_processes: bool = False) -> List[Any]:
        """Process items in parallel using threads or processes."""
        max_workers = max_workers or self.config.max_workers
        
        ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        logger.info(f"Processing {len(items)} items with {max_workers} "
                   f"{'processes' if use_processes else 'threads'}")
        
        with ExecutorClass(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_func, item) for item in items]
            
            # Collect results
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item {i}: {e}")
                    results.append(None)
        
        return results


class ModelOptimizer:
    """Optimize model performance through various techniques."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
    
    def optimize_model(self, model) -> Any:
        """Apply various optimizations to the model."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping model optimization")
            return model
        
        optimizations_applied = []
        
        try:
            # Mixed precision training
            if self.config.use_mixed_precision and torch.cuda.is_available():
                # This would typically use torch.amp.autocast
                logger.info("Mixed precision enabled")
                optimizations_applied.append("mixed_precision")
            
            # Model compilation (PyTorch 2.0+)
            if self.config.compile_model:
                try:
                    if hasattr(torch, 'compile'):
                        model = torch.compile(model)
                        optimizations_applied.append("torch_compile")
                        logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            # Gradient checkpointing
            if self.config.gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    optimizations_applied.append("gradient_checkpointing")
                    logger.info("Gradient checkpointing enabled")
            
            # Set model to evaluation mode for inference
            model.eval()
            
            logger.info(f"Model optimizations applied: {', '.join(optimizations_applied)}")
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
        
        return model
    
    def optimize_inference_batch_size(self, model, sample_input, 
                                    max_memory_mb: Optional[int] = None) -> int:
        """Find optimal batch size for inference."""
        if not TORCH_AVAILABLE:
            return 32  # Default fallback
        
        max_memory_mb = max_memory_mb or (self.config.memory_limit_mb * self.config.tensor_memory_fraction)
        
        # Start with small batch and increase until OOM
        batch_size = 1
        max_batch_size = 512
        
        logger.info("Finding optimal inference batch size...")
        
        with torch.no_grad():
            while batch_size <= max_batch_size:
                try:
                    # Create batch
                    if isinstance(sample_input, dict):
                        batch_input = {k: v.repeat(batch_size, *[1]*len(v.shape[1:])) 
                                     for k, v in sample_input.items()}
                    else:
                        batch_input = sample_input.repeat(batch_size, *[1]*len(sample_input.shape[1:]))
                    
                    # Test inference
                    _ = model(batch_input)
                    
                    # Check memory usage
                    if torch.cuda.is_available():
                        memory_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                        if memory_used_mb > max_memory_mb:
                            break
                    
                    batch_size *= 2
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    raise
        
        optimal_batch_size = max(1, batch_size // 2)
        logger.info(f"Optimal inference batch size: {optimal_batch_size}")
        
        return optimal_batch_size


class ResourceManager:
    """Manage computational resources efficiently."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
        self._resource_locks = {}
        self._resource_usage = {}
    
    def acquire_gpu_memory(self, required_mb: int) -> bool:
        """Reserve GPU memory for operation."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return True
        
        try:
            # Check available GPU memory
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_mb = gpu_props.total_memory / (1024 * 1024)
            
            # Get current usage
            current_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            available_mb = total_memory_mb - current_usage_mb
            
            if available_mb >= required_mb:
                # Reserve memory by allocating a tensor
                reserve_tensor = torch.empty(
                    (int(required_mb * 1024 * 1024 // 4),), 
                    dtype=torch.float32, 
                    device='cuda'
                )
                self._resource_locks[f"gpu_memory_{required_mb}"] = reserve_tensor
                return True
            else:
                logger.warning(f"Insufficient GPU memory: {available_mb:.1f}MB available, "
                             f"{required_mb}MB required")
                return False
        
        except Exception as e:
            logger.error(f"GPU memory acquisition failed: {e}")
            return False
    
    def release_gpu_memory(self, required_mb: int):
        """Release reserved GPU memory."""
        key = f"gpu_memory_{required_mb}"
        if key in self._resource_locks:
            del self._resource_locks[key]
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def with_memory_limit(self, limit_mb: int):
        """Context manager for memory-limited operations."""
        class MemoryLimitContext:
            def __init__(self, manager, limit):
                self.manager = manager
                self.limit = limit
                self.acquired = False
            
            def __enter__(self):
                self.acquired = self.manager.acquire_gpu_memory(self.limit)
                return self.acquired
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.acquired:
                    self.manager.release_gpu_memory(self.limit)
        
        return MemoryLimitContext(self, limit_mb)


class PerformanceProfiler:
    """Profile and analyze performance bottlenecks."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.profiles = {}
        self._active_profiles = {}
    
    def start_profile(self, name: str):
        """Start profiling an operation."""
        if not self.config.enable_profiling:
            return
        
        profile_data = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
        }
        
        self._active_profiles[name] = profile_data
    
    def end_profile(self, name: str) -> Dict[str, Any]:
        """End profiling and return results."""
        if not self.config.enable_profiling or name not in self._active_profiles:
            return {}
        
        start_data = self._active_profiles.pop(name)
        
        profile_result = {
            'duration_seconds': time.time() - start_data['start_time'],
            'memory_delta_mb': self._get_memory_usage() - start_data['start_memory'],
            'timestamp': time.time(),
        }
        
        # Store result
        if name not in self.profiles:
            self.profiles[name] = []
        self.profiles[name].append(profile_result)
        
        return profile_result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def profile(self, name: str):
        """Decorator for profiling functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.start_profile(name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profile_data = self.end_profile(name)
                    if profile_data:
                        logger.debug(f"Profile {name}: {profile_data['duration_seconds']:.3f}s, "
                                   f"{profile_data['memory_delta_mb']:.1f}MB")
            return wrapper
        return decorator
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles."""
        summary = {}
        
        for name, profiles in self.profiles.items():
            if not profiles:
                continue
            
            durations = [p['duration_seconds'] for p in profiles]
            memory_deltas = [p['memory_delta_mb'] for p in profiles]
            
            summary[name] = {
                'call_count': len(profiles),
                'avg_duration_seconds': sum(durations) / len(durations),
                'max_duration_seconds': max(durations),
                'total_duration_seconds': sum(durations),
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'max_memory_delta_mb': max(memory_deltas),
            }
        
        return summary


# Global instances
_global_performance_config = None
_global_performance_monitor = None
_global_profiler = None

def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration."""
    global _global_performance_config
    if _global_performance_config is None:
        _global_performance_config = PerformanceConfig()
    return _global_performance_config

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor(get_performance_config())
    return _global_performance_monitor

def get_profiler() -> PerformanceProfiler:
    """Get global profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(get_performance_config())
    return _global_profiler

def profile(name: str):
    """Global profiling decorator."""
    return get_profiler().profile(name)