"""
Advanced Performance Optimization Framework for Protein Diffusion Design Lab.

This module provides comprehensive performance optimization including batch processing,
parallel execution, GPU acceleration, and intelligent resource management.
"""

import logging
import time
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import queue
import concurrent.futures
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    # Mock torch module for when PyTorch is not available
    class MockTorch:
        class device:
            def __init__(self, device_name):
                self.type = device_name
            
            def __str__(self):
                return self.type
        
        class cuda:
            @staticmethod
            def is_available():
                return False
            
            @staticmethod
            def device_count():
                return 0
            
            @staticmethod
            def memory_allocated(device=None):
                return 0
            
            @staticmethod
            def memory_reserved(device=None):
                return 0
            
            @staticmethod
            def empty_cache():
                pass
            
            @staticmethod
            def set_per_process_memory_fraction(fraction, device=None):
                pass
            
            @staticmethod
            def get_device_properties(device):
                class MockProps:
                    total_memory = 0
                    multi_processor_count = 0
                return MockProps()
        
        class nn:
            class Module:
                def to(self, device):
                    return self
                def eval(self):
                    return self
        
        @staticmethod
        def compile(model):
            return model
    
    torch = MockTorch()
    nn = MockTorch.nn
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREADS = "parallel_threads"
    PARALLEL_PROCESSES = "parallel_processes"
    ASYNC_CONCURRENT = "async_concurrent"
    GPU_ACCELERATED = "gpu_accelerated"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Resource types for allocation."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    # Parallel processing
    max_workers: int = multiprocessing.cpu_count()
    thread_pool_size: int = 16
    process_pool_size: int = 4
    
    # GPU settings
    enable_gpu: bool = True
    gpu_device_ids: List[int] = None
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    
    # Batch processing
    default_batch_size: int = 32
    max_batch_size: int = 128
    adaptive_batching: bool = True
    batch_timeout: float = 1.0
    
    # Memory management
    memory_limit_gb: float = 16.0
    enable_memory_mapping: bool = True
    garbage_collection_threshold: int = 1000
    
    # Async settings
    max_concurrent_tasks: int = 100
    async_timeout: float = 300.0
    
    # Optimization strategies
    preferred_strategy: OptimizationStrategy = OptimizationStrategy.HYBRID
    fallback_strategy: OptimizationStrategy = OptimizationStrategy.PARALLEL_THREADS
    
    # Resource management
    cpu_affinity: List[int] = None
    priority_level: int = 0  # -20 to 19 (Linux)
    
    def __post_init__(self):
        if self.gpu_device_ids is None:
            self.gpu_device_ids = [0]


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float = 0.0
    throughput: float = 0.0  # items/second
    cpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    
    # Batch processing metrics
    batch_count: int = 0
    avg_batch_size: float = 0.0
    batch_efficiency: float = 0.0
    
    # Parallel processing metrics
    worker_count: int = 0
    parallel_efficiency: float = 0.0
    overhead_time: float = 0.0


class ResourceMonitor:
    """Monitor and manage system resources."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.resource_usage: Dict[ResourceType, deque] = {
            resource: deque(maxlen=100) for resource in ResourceType
        }
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._collect_resource_metrics()
                time.sleep(1.0)  # Monitor every second
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(1.0)
    
    def _collect_resource_metrics(self):
        """Collect current resource metrics."""
        try:
            import psutil
            
            with self.lock:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.resource_usage[ResourceType.CPU].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.resource_usage[ResourceType.MEMORY].append(memory_mb)
                
                # GPU usage (if available)
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.resource_usage[ResourceType.GPU].append(gpu_memory)
                
        except ImportError:
            # Fallback when psutil not available
            with self.lock:
                self.resource_usage[ResourceType.CPU].append(50.0)  # Mock value
                self.resource_usage[ResourceType.MEMORY].append(2048.0)  # Mock value
    
    def get_current_usage(self, resource_type: ResourceType) -> float:
        """Get current usage for a resource type."""
        with self.lock:
            usage_history = self.resource_usage[resource_type]
            return usage_history[-1] if usage_history else 0.0
    
    def get_average_usage(self, resource_type: ResourceType, window: int = 10) -> float:
        """Get average usage over a time window."""
        with self.lock:
            usage_history = self.resource_usage[resource_type]
            recent_usage = list(usage_history)[-window:]
            return sum(recent_usage) / len(recent_usage) if recent_usage else 0.0
    
    def is_resource_available(self, resource_type: ResourceType, threshold: float = 0.8) -> bool:
        """Check if resource is available (below threshold)."""
        current_usage = self.get_current_usage(resource_type)
        
        if resource_type == ResourceType.CPU:
            return current_usage < (threshold * 100)
        elif resource_type == ResourceType.MEMORY:
            memory_limit = self.config.memory_limit_gb * 1024  # MB
            return current_usage < (threshold * memory_limit)
        elif resource_type == ResourceType.GPU:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory_limit = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                return current_usage < (threshold * gpu_memory_limit)
        
        return True


class BatchProcessor:
    """Intelligent batch processing with adaptive sizing."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.batch_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing = False
        self.processor_thread = None
        
        # Adaptive batching state
        self.current_batch_size = config.default_batch_size
        self.recent_processing_times = deque(maxlen=10)
        self.recent_throughputs = deque(maxlen=10)
    
    def start_processing(self, processor_func: Callable):
        """Start batch processing."""
        if self.processing:
            return
        
        self.processing = True
        self.processor_func = processor_func
        self.processor_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processor_thread.start()
        
        logger.info("Batch processing started")
    
    def stop_processing(self):
        """Stop batch processing."""
        self.processing = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        
        logger.info("Batch processing stopped")
    
    def submit(self, item: Any) -> None:
        """Submit item for batch processing."""
        self.batch_queue.put(item)
    
    def get_results(self, timeout: float = None) -> List[Any]:
        """Get processed results."""
        results = []
        
        try:
            while True:
                result = self.result_queue.get(timeout=timeout)
                if result is None:  # Sentinel value
                    break
                results.append(result)
        except queue.Empty:
            pass
        
        return results
    
    def _process_batches(self):
        """Main batch processing loop."""
        batch = []
        last_batch_time = time.time()
        
        while self.processing:
            try:
                # Collect items for batch
                while len(batch) < self.current_batch_size:
                    try:
                        item = self.batch_queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        break
                
                # Process batch if ready
                current_time = time.time()
                batch_timeout_reached = (current_time - last_batch_time) >= self.config.batch_timeout
                
                if batch and (len(batch) >= self.current_batch_size or batch_timeout_reached):
                    self._process_single_batch(batch)
                    batch = []
                    last_batch_time = current_time
                
                # Small sleep to prevent busy waiting
                if not batch:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                time.sleep(0.1)
        
        # Process remaining batch
        if batch:
            self._process_single_batch(batch)
    
    def _process_single_batch(self, batch: List[Any]):
        """Process a single batch."""
        start_time = time.time()
        
        try:
            # Process batch
            results = self.processor_func(batch)
            
            # Record timing
            processing_time = time.time() - start_time
            throughput = len(batch) / processing_time
            
            self.recent_processing_times.append(processing_time)
            self.recent_throughputs.append(throughput)
            
            # Adapt batch size if enabled
            if self.config.adaptive_batching:
                self._adapt_batch_size()
            
            # Queue results
            for result in results:
                self.result_queue.put(result)
            
            logger.debug(f"Processed batch of {len(batch)} items in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Queue error results
            for _ in batch:
                self.result_queue.put({"error": str(e)})
    
    def _adapt_batch_size(self):
        """Adapt batch size based on performance metrics."""
        if len(self.recent_throughputs) < 3:
            return
        
        # Calculate recent performance trend
        recent_throughput = sum(self.recent_throughputs) / len(self.recent_throughputs)
        
        # Adjust batch size based on throughput
        if len(self.recent_throughputs) >= 2:
            prev_throughput = self.recent_throughputs[-2]
            current_throughput = self.recent_throughputs[-1]
            
            if current_throughput > prev_throughput * 1.1:
                # Performance improving, try larger batch
                self.current_batch_size = min(
                    self.current_batch_size + 4,
                    self.config.max_batch_size
                )
            elif current_throughput < prev_throughput * 0.9:
                # Performance degrading, try smaller batch
                self.current_batch_size = max(
                    self.current_batch_size - 4,
                    4  # Minimum batch size
                )
        
        logger.debug(f"Adapted batch size to {self.current_batch_size}")


class GPUManager:
    """Manage GPU resources and optimization."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.device_count = 0
        self.devices = []
        
        if TORCH_AVAILABLE and torch.cuda.is_available() and config.enable_gpu:
            self.device_count = torch.cuda.device_count()
            self.devices = [torch.device(f"cuda:{i}") for i in range(self.device_count)]
            
            # Set memory fraction
            for i in range(self.device_count):
                torch.cuda.set_per_process_memory_fraction(
                    config.gpu_memory_fraction, device=i
                )
            
            logger.info(f"GPU Manager initialized with {self.device_count} GPUs")
        else:
            logger.info("GPU Manager initialized in CPU-only mode")
    
    def get_optimal_device(self) -> torch.device:
        """Get the optimal GPU device based on current usage."""
        if not self.devices:
            return torch.device("cpu")
        
        if len(self.devices) == 1:
            return self.devices[0]
        
        # Find device with lowest memory usage
        min_memory_usage = float('inf')
        optimal_device = self.devices[0]
        
        for device in self.devices:
            memory_usage = torch.cuda.memory_allocated(device)
            if memory_usage < min_memory_usage:
                min_memory_usage = memory_usage
                optimal_device = device
        
        return optimal_device
    
    def get_device_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all GPU devices."""
        stats = {}
        
        for i, device in enumerate(self.devices):
            props = torch.cuda.get_device_properties(device)
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
            memory_cached = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
            memory_total = props.total_memory / (1024 ** 3)  # GB
            
            stats[f"cuda:{i}"] = {
                "memory_allocated_gb": memory_allocated,
                "memory_cached_gb": memory_cached,
                "memory_total_gb": memory_total,
                "memory_utilization": memory_allocated / memory_total,
                "multiprocessor_count": props.multi_processor_count
            }
        
        return stats
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference."""
        if not TORCH_AVAILABLE:
            return model
        
        # Move to optimal device
        device = self.get_optimal_device()
        model = model.to(device)
        
        # Set to evaluation mode
        model.eval()
        
        # Enable mixed precision if available
        if self.config.mixed_precision and hasattr(torch.cuda, 'amp'):
            # Model will use autocast during forward pass
            logger.info("Mixed precision enabled for inference")
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Model compiled for optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            
            logger.info("GPU memory cache cleared")


class ParallelExecutor:
    """Execute tasks in parallel with different strategies."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.thread_pool = None
        self.process_pool = None
        
        # Initialize thread pool
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.thread_pool_size
        )
        
        # Initialize process pool
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=config.process_pool_size
        )
        
        logger.info(f"Parallel Executor initialized (threads: {config.thread_pool_size}, processes: {config.process_pool_size})")
    
    def execute_parallel(
        self,
        tasks: List[Callable],
        strategy: OptimizationStrategy = None,
        timeout: float = None
    ) -> List[Any]:
        """Execute tasks in parallel."""
        if strategy is None:
            strategy = self.config.preferred_strategy
        
        if not tasks:
            return []
        
        start_time = time.time()
        
        try:
            if strategy == OptimizationStrategy.SEQUENTIAL:
                results = self._execute_sequential(tasks)
            elif strategy == OptimizationStrategy.PARALLEL_THREADS:
                results = self._execute_threaded(tasks, timeout)
            elif strategy == OptimizationStrategy.PARALLEL_PROCESSES:
                results = self._execute_multiprocess(tasks, timeout)
            elif strategy == OptimizationStrategy.ASYNC_CONCURRENT:
                results = self._execute_async(tasks, timeout)
            elif strategy == OptimizationStrategy.HYBRID:
                results = self._execute_hybrid(tasks, timeout)
            else:
                results = self._execute_threaded(tasks, timeout)
            
            execution_time = time.time() - start_time
            throughput = len(tasks) / execution_time if execution_time > 0 else 0
            
            logger.info(f"Executed {len(tasks)} tasks in {execution_time:.3f}s (throughput: {throughput:.2f} tasks/s)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
            # Fallback to sequential execution
            return self._execute_sequential(tasks)
    
    def _execute_sequential(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks sequentially."""
        results = []
        for task in tasks:
            try:
                result = task()
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results
    
    def _execute_threaded(self, tasks: List[Callable], timeout: float = None) -> List[Any]:
        """Execute tasks using thread pool."""
        futures = [self.thread_pool.submit(task) for task in tasks]
        results = []
        
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    def _execute_multiprocess(self, tasks: List[Callable], timeout: float = None) -> List[Any]:
        """Execute tasks using process pool."""
        try:
            futures = [self.process_pool.submit(task) for task in tasks]
            results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
            
            return results
        except Exception as e:
            logger.warning(f"Multiprocessing failed, falling back to threading: {e}")
            return self._execute_threaded(tasks, timeout)
    
    def _execute_async(self, tasks: List[Callable], timeout: float = None) -> List[Any]:
        """Execute tasks using async/await."""
        async def run_async_tasks():
            async_tasks = []
            for task in tasks:
                async_task = asyncio.get_event_loop().run_in_executor(None, task)
                async_tasks.append(async_task)
            
            return await asyncio.gather(*async_tasks, return_exceptions=True)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                asyncio.wait_for(run_async_tasks(), timeout=timeout)
            )
            loop.close()
            
            # Convert exceptions to error dictionaries
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({"error": str(result)})
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.warning(f"Async execution failed, falling back to threading: {e}")
            return self._execute_threaded(tasks, timeout)
    
    def _execute_hybrid(self, tasks: List[Callable], timeout: float = None) -> List[Any]:
        """Execute tasks using hybrid strategy."""
        # Use different strategies based on task characteristics
        if len(tasks) <= 4:
            return self._execute_sequential(tasks)
        elif len(tasks) <= 20:
            return self._execute_threaded(tasks, timeout)
        else:
            return self._execute_multiprocess(tasks, timeout)
    
    def shutdown(self):
        """Shutdown executor pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Parallel executor shutdown")


class PerformanceOptimizer:
    """Main performance optimization manager."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(self.config)
        self.batch_processor = BatchProcessor(self.config)
        self.gpu_manager = GPUManager(self.config)
        self.parallel_executor = ParallelExecutor(self.config)
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_enabled = True
        
        logger.info("Performance Optimizer initialized")
    
    def start_optimization(self):
        """Start performance optimization."""
        self.resource_monitor.start_monitoring()
        self.optimization_enabled = True
        
        logger.info("Performance optimization started")
    
    def stop_optimization(self):
        """Stop performance optimization."""
        self.resource_monitor.stop_monitoring()
        self.batch_processor.stop_processing()
        self.parallel_executor.shutdown()
        self.optimization_enabled = False
        
        logger.info("Performance optimization stopped")
    
    def optimize_function(
        self,
        func: Callable,
        strategy: OptimizationStrategy = None,
        enable_caching: bool = True
    ) -> Callable:
        """Optimize a function with various strategies."""
        @wraps(func)
        def optimized_wrapper(*args, **kwargs):
            if not self.optimization_enabled:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            try:
                # Choose optimization strategy
                if strategy is None:
                    chosen_strategy = self._choose_optimal_strategy()
                else:
                    chosen_strategy = strategy
                
                # Execute with chosen strategy
                if chosen_strategy == OptimizationStrategy.GPU_ACCELERATED:
                    result = self._gpu_accelerated_execution(func, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record metrics
                execution_time = time.time() - start_time
                self._record_performance_metrics(execution_time, chosen_strategy)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in optimized function execution: {e}")
                # Fallback to original function
                return func(*args, **kwargs)
        
        return optimized_wrapper
    
    def process_batch_optimized(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_size: int = None
    ) -> List[Any]:
        """Process items in optimized batches."""
        if not items:
            return []
        
        if batch_size is None:
            batch_size = self.config.default_batch_size
        
        # Adjust batch size based on resource availability
        if self.resource_monitor.is_resource_available(ResourceType.MEMORY, 0.7):
            batch_size = min(batch_size * 2, self.config.max_batch_size)
        elif not self.resource_monitor.is_resource_available(ResourceType.MEMORY, 0.9):
            batch_size = max(batch_size // 2, 4)
        
        # Process in batches
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                batch_results = processor_func(batch)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Process items individually as fallback
                for item in batch:
                    try:
                        result = processor_func([item])
                        results.extend(result)
                    except Exception as item_error:
                        results.append({"error": str(item_error)})
        
        return results
    
    def _choose_optimal_strategy(self) -> OptimizationStrategy:
        """Choose optimal strategy based on current conditions."""
        # Check resource availability
        cpu_available = self.resource_monitor.is_resource_available(ResourceType.CPU, 0.8)
        memory_available = self.resource_monitor.is_resource_available(ResourceType.MEMORY, 0.8)
        gpu_available = self.resource_monitor.is_resource_available(ResourceType.GPU, 0.8)
        
        # Choose strategy based on resource availability
        if gpu_available and self.config.enable_gpu:
            return OptimizationStrategy.GPU_ACCELERATED
        elif cpu_available and memory_available:
            return OptimizationStrategy.PARALLEL_THREADS
        elif cpu_available:
            return OptimizationStrategy.ASYNC_CONCURRENT
        else:
            return OptimizationStrategy.SEQUENTIAL
    
    def _gpu_accelerated_execution(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with GPU acceleration."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return func(*args, **kwargs)
        
        # Move tensor arguments to GPU if possible
        gpu_args = []
        gpu_kwargs = {}
        
        device = self.gpu_manager.get_optimal_device()
        
        for arg in args:
            if TORCH_AVAILABLE and isinstance(arg, torch.Tensor):
                gpu_args.append(arg.to(device))
            else:
                gpu_args.append(arg)
        
        for key, value in kwargs.items():
            if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                gpu_kwargs[key] = value.to(device)
            else:
                gpu_kwargs[key] = value
        
        # Execute function
        with torch.cuda.device(device):
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    result = func(*gpu_args, **gpu_kwargs)
            else:
                result = func(*gpu_args, **gpu_kwargs)
        
        # Move result back to CPU if it's a tensor
        if TORCH_AVAILABLE and isinstance(result, torch.Tensor):
            result = result.cpu()
        
        return result
    
    def _record_performance_metrics(self, execution_time: float, strategy: OptimizationStrategy):
        """Record performance metrics."""
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            throughput=1.0 / execution_time if execution_time > 0 else 0,
            cpu_utilization=self.resource_monitor.get_current_usage(ResourceType.CPU),
            memory_usage_mb=self.resource_monitor.get_current_usage(ResourceType.MEMORY),
            gpu_utilization=self.resource_monitor.get_current_usage(ResourceType.GPU)
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance optimization summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 operations
        
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_cpu_utilization = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        
        return {
            "status": "active" if self.optimization_enabled else "inactive",
            "recent_performance": {
                "avg_execution_time": avg_execution_time,
                "avg_throughput": avg_throughput,
                "avg_cpu_utilization": avg_cpu_utilization,
                "avg_memory_usage_mb": avg_memory_usage,
            },
            "resource_status": {
                "cpu_available": self.resource_monitor.is_resource_available(ResourceType.CPU),
                "memory_available": self.resource_monitor.is_resource_available(ResourceType.MEMORY),
                "gpu_available": self.resource_monitor.is_resource_available(ResourceType.GPU),
            },
            "gpu_stats": self.gpu_manager.get_device_stats() if self.gpu_manager.devices else None,
            "config": asdict(self.config)
        }


# Utility decorators and functions

def optimize_performance(
    strategy: OptimizationStrategy = None,
    enable_gpu: bool = True,
    batch_size: int = None
):
    """Decorator to optimize function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_global_performance_optimizer()
            
            if batch_size and hasattr(args[0], '__len__') and len(args[0]) > batch_size:
                # Handle batch processing
                items = args[0]
                other_args = args[1:]
                
                def batch_processor(batch):
                    return func(batch, *other_args, **kwargs)
                
                return optimizer.process_batch_optimized(items, batch_processor, batch_size)
            else:
                # Handle single execution
                optimized_func = optimizer.optimize_function(func, strategy)
                return optimized_func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global performance optimizer instance
_global_performance_optimizer = None

def get_global_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_performance_optimizer
    if _global_performance_optimizer is None:
        _global_performance_optimizer = PerformanceOptimizer()
    return _global_performance_optimizer