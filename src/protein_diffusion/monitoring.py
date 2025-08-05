"""
Monitoring and logging utilities for protein diffusion models.

This module provides comprehensive monitoring, metrics collection,
and logging capabilities for production deployments.
"""

import time
import logging
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import contextlib
import functools

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

@dataclass
class MetricPoint:
    """A single metric measurement."""
    timestamp: float
    value: Any
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class PerformanceMetrics:
    """Performance metrics for model operations."""
    operation: str
    duration: float
    memory_usage: float
    gpu_memory_usage: float = 0.0
    batch_size: int = 1
    sequence_length: int = 0
    throughput: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.duration > 0 and self.batch_size > 0:
            self.throughput = self.batch_size / self.duration

@dataclass
class ModelMetrics:
    """Metrics specific to model performance."""
    model_name: str
    inference_time: float
    confidence_scores: List[float]
    sequence_lengths: List[int]
    memory_peak: float
    gpu_utilization: float = 0.0
    batch_size: int = 1
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class MetricsCollector:
    """Collect and store metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.metrics = defaultdict(deque)
        self.max_history = max_history
        self._lock = threading.Lock()
    
    def record(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """Record a metric value."""
        with self._lock:
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            
            self.metrics[metric_name].append(metric_point)
            
            # Keep only recent history
            if len(self.metrics[metric_name]) > self.max_history:
                self.metrics[metric_name].popleft()
    
    def get_recent(self, metric_name: str, seconds: int = 60) -> List[MetricPoint]:
        """Get recent metric points within the specified time window."""
        cutoff_time = time.time() - seconds
        with self._lock:
            return [
                point for point in self.metrics[metric_name]
                if point.timestamp >= cutoff_time
            ]
    
    def get_summary(self, metric_name: str, seconds: int = 300) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        points = self.get_recent(metric_name, seconds)
        if not points:
            return {}
        
        values = [p.value for p in points if isinstance(p.value, (int, float))]
        if not values:
            return {"count": len(points)}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "time_range": f"{len(points)} points over {seconds}s"
        }
    
    def clear(self, metric_name: Optional[str] = None):
        """Clear metrics."""
        with self._lock:
            if metric_name:
                self.metrics[metric_name].clear()
            else:
                self.metrics.clear()

class SystemMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.is_monitoring = False
        self._monitor_thread = None
        self.metrics_collector = MetricsCollector()
    
    def start_monitoring(self, interval: float = 5.0):
        """Start system monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Stopped system monitoring")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        timestamp = time.time()
        
        if PSUTIL_AVAILABLE:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.metrics_collector.record("system.cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record("system.memory_percent", memory.percent)
            self.metrics_collector.record("system.memory_available_gb", memory.available / 1e9)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.metrics_collector.record("system.disk_read_mb", disk_io.read_bytes / 1e6)
                self.metrics_collector.record("system.disk_write_mb", disk_io.write_bytes / 1e6)
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # GPU metrics
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(i) / 1e9
                    
                    self.metrics_collector.record(
                        f"gpu.{i}.memory_allocated_gb", 
                        memory_allocated,
                        {"device": str(i)}
                    )
                    self.metrics_collector.record(
                        f"gpu.{i}.memory_reserved_gb", 
                        memory_reserved,
                        {"device": str(i)}
                    )
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        stats = {}
        
        if PSUTIL_AVAILABLE:
            stats.update({
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / 1e9,
            })
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_stats = {}
            for i in range(torch.cuda.device_count()):
                gpu_stats[f"gpu_{i}"] = {
                    "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                    "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                }
            stats["gpu"] = gpu_stats
        
        return stats

class PerformanceProfiler:
    """Profile performance of operations."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.active_profiles = {}
    
    @contextlib.contextmanager
    def profile(self, operation_name: str, **kwargs):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            memory_usage = self._get_memory_usage() - start_memory
            gpu_memory_usage = self._get_gpu_memory_usage() - start_gpu_memory
            
            metrics = PerformanceMetrics(
                operation=operation_name,
                duration=duration,
                memory_usage=memory_usage,
                gpu_memory_usage=gpu_memory_usage,
                **kwargs
            )
            
            self._record_performance_metrics(metrics)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if PSUTIL_AVAILABLE:
            return psutil.Process().memory_info().rss / 1e9
        return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def _record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_collector.record(
            f"performance.{metrics.operation}.duration",
            metrics.duration,
            {"operation": metrics.operation}
        )
        
        self.metrics_collector.record(
            f"performance.{metrics.operation}.throughput",
            metrics.throughput,
            {"operation": metrics.operation}
        )
        
        self.metrics_collector.record(
            f"performance.{metrics.operation}.memory_usage",
            metrics.memory_usage,
            {"operation": metrics.operation}
        )
        
        if metrics.gpu_memory_usage > 0:
            self.metrics_collector.record(
                f"performance.{metrics.operation}.gpu_memory_usage",
                metrics.gpu_memory_usage,
                {"operation": metrics.operation}
            )
        
        logger.info(f"Performance [{metrics.operation}]: "
                   f"{metrics.duration:.3f}s, "
                   f"{metrics.throughput:.1f} samples/s, "
                   f"{metrics.memory_usage:.2f}GB mem")

def profile_performance(operation_name: str, **kwargs):
    """Decorator for profiling function performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            profiler = PerformanceProfiler()
            with profiler.profile(operation_name, **kwargs):
                return func(*args, **func_kwargs)
        return wrapper
    return decorator

class ModelMonitor:
    """Monitor model-specific metrics during inference."""
    
    def __init__(self, model_name: str = "protein_diffusion"):
        self.model_name = model_name
        self.metrics_collector = MetricsCollector()
        self.inference_count = 0
        self.total_sequences = 0
    
    def record_inference(
        self,
        inference_time: float,
        confidence_scores: List[float],
        sequence_lengths: List[int],
        batch_size: int = 1
    ):
        """Record metrics from a model inference."""
        self.inference_count += 1
        self.total_sequences += batch_size
        
        # Record basic metrics
        self.metrics_collector.record("model.inference_time", inference_time)
        self.metrics_collector.record("model.batch_size", batch_size)
        self.metrics_collector.record("model.sequences_per_second", batch_size / inference_time)
        
        # Record confidence metrics
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            min_confidence = min(confidence_scores)
            max_confidence = max(confidence_scores)
            
            self.metrics_collector.record("model.confidence.mean", avg_confidence)
            self.metrics_collector.record("model.confidence.min", min_confidence)
            self.metrics_collector.record("model.confidence.max", max_confidence)
        
        # Record sequence length metrics
        if sequence_lengths:
            avg_length = sum(sequence_lengths) / len(sequence_lengths)
            min_length = min(sequence_lengths)
            max_length = max(sequence_lengths)
            
            self.metrics_collector.record("model.sequence_length.mean", avg_length)
            self.metrics_collector.record("model.sequence_length.min", min_length)
            self.metrics_collector.record("model.sequence_length.max", max_length)
        
        # Log periodic summaries
        if self.inference_count % 100 == 0:
            self._log_summary()
    
    def _log_summary(self):
        """Log summary of recent model performance."""
        recent_times = self.metrics_collector.get_recent("model.inference_time", 300)  # 5 minutes
        recent_confidences = self.metrics_collector.get_recent("model.confidence.mean", 300)
        
        if recent_times:
            avg_time = sum(p.value for p in recent_times) / len(recent_times)
            logger.info(f"Model performance summary: "
                       f"{len(recent_times)} inferences, "
                       f"avg time {avg_time:.3f}s, "
                       f"total sequences: {self.total_sequences}")
        
        if recent_confidences:
            avg_conf = sum(p.value for p in recent_confidences) / len(recent_confidences)
            logger.info(f"Model quality summary: avg confidence {avg_conf:.3f}")

class LoggingSetup:
    """Setup comprehensive logging configuration."""
    
    @staticmethod
    def setup_logging(
        level: str = "INFO",
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        structured: bool = True
    ):
        """Setup logging configuration."""
        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create formatters
        if structured:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        if log_file or log_dir:
            if log_dir:
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"protein_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
        
        # Set library log levels
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        logger.info(f"Logging setup complete: level={level}")

class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class HealthChecker:
    """Health checking for the protein diffusion system."""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool], interval: int = 60):
        """Register a health check function."""
        self.checks[name] = {
            "func": check_func,
            "interval": interval,
            "last_result": None,
            "last_error": None
        }
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {"status": "error", "message": f"Unknown check: {name}"}
        
        check = self.checks[name]
        current_time = time.time()
        
        # Check if we need to run the check
        last_check = self.last_check_time.get(name, 0)
        if current_time - last_check < check["interval"]:
            return check["last_result"] or {"status": "pending"}
        
        try:
            result = check["func"]()
            check["last_result"] = {
                "status": "healthy" if result else "unhealthy",
                "timestamp": current_time,
                "check": name
            }
            check["last_error"] = None
        except Exception as e:
            check["last_result"] = {
                "status": "error",
                "timestamp": current_time,
                "check": name,
                "error": str(e)
            }
            check["last_error"] = str(e)
        
        self.last_check_time[name] = current_time
        return check["last_result"]
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = "healthy"
        
        for name in self.checks:
            result = self.run_check(name)
            results[name] = result
            
            if result["status"] != "healthy":
                overall_status = "unhealthy"
        
        return {
            "overall_status": overall_status,
            "timestamp": time.time(),
            "checks": results
        }

# Global instances
_system_monitor = SystemMonitor()
_performance_profiler = PerformanceProfiler()

def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    return _system_monitor

def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    return _performance_profiler