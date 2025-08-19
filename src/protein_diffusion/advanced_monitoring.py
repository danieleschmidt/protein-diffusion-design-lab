"""
Advanced Monitoring System - Comprehensive observability and metrics collection.

This module provides real-time monitoring, alerting, performance tracking,
and observability for protein diffusion workflows with integration for
prometheus, grafana, and custom dashboard capabilities.
"""

import logging
import time
import threading
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import json
from enum import Enum

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    component: str = "system"
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """System performance metrics snapshot."""
    timestamp: float = field(default_factory=time.time)
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: List[float] = field(default_factory=list)
    
    # Memory metrics
    memory_total: int = 0
    memory_available: int = 0
    memory_used: int = 0
    memory_percent: float = 0.0
    
    # GPU metrics
    gpu_count: int = 0
    gpu_memory_used: List[int] = field(default_factory=list)
    gpu_memory_total: List[int] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)
    
    # Disk metrics
    disk_usage: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)
    
    # Network metrics
    network_io: Dict[str, int] = field(default_factory=dict)
    
    # Process metrics
    process_count: int = 0
    thread_count: int = 0


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: float = field(default_factory=time.time)
    
    # Generation metrics
    sequences_generated: int = 0
    generation_rate: float = 0.0
    avg_generation_time: float = 0.0
    
    # Ranking metrics
    sequences_ranked: int = 0
    ranking_rate: float = 0.0
    avg_ranking_time: float = 0.0
    
    # API metrics
    requests_total: int = 0
    requests_per_second: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    
    # Queue metrics
    queue_size: int = 0
    queue_processing_time: float = 0.0
    
    # Model metrics
    model_inference_time: float = 0.0
    model_memory_usage: int = 0
    
    # Error metrics
    total_errors: int = 0
    error_rate_per_minute: float = 0.0
    circuit_breakers_open: int = 0


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    # Collection settings
    collection_interval: float = 5.0  # seconds
    metric_retention_time: float = 3600.0  # 1 hour
    enable_system_metrics: bool = True
    enable_application_metrics: bool = True
    enable_gpu_monitoring: bool = True
    
    # Alerting settings
    enable_alerting: bool = True
    alert_retention_time: float = 86400.0  # 24 hours
    
    # Thresholds for alerts
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    error_rate_threshold: float = 0.05
    response_time_threshold: float = 5.0
    
    # Storage settings
    save_metrics: bool = True
    metrics_directory: str = "./monitoring_data"
    
    # Dashboard settings
    enable_dashboard: bool = False
    dashboard_port: int = 8080
    dashboard_host: str = "localhost"
    
    # External integration
    prometheus_enabled: bool = False
    prometheus_port: int = 8000
    webhook_urls: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collects and manages various system and application metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def record_metric(self, metric: MetricData):
        """Record a metric data point."""
        with self.lock:
            self.metrics[metric.name].append(metric)
            
            if metric.metric_type == MetricType.COUNTER:
                self.counters[metric.name] += metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self.gauges[metric.name] = metric.value
            elif metric.metric_type == MetricType.TIMER:
                self.timers[metric.name].append(metric.value)
                # Keep only last 100 timer values
                if len(self.timers[metric.name]) > 100:
                    self.timers[metric.name] = self.timers[metric.name][-100:]
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        metric = MetricData(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        metric = MetricData(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        metric = MetricData(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return {"error": f"Metric {name} not found"}
            
            values = [m.value for m in self.metrics[name]]
            
            if not values:
                return {"error": f"No data for metric {name}"}
            
            return {
                "name": name,
                "count": len(values),
                "latest": values[-1],
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "total": sum(values) if name in self.counters else None
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers": {name: {
                    "count": len(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                } for name, values in self.timers.items()}
            }


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.alerts: List[Alert] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_metrics = None
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=100)
        self.application_history: deque = deque(maxlen=100)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Create storage directory
        if self.config.save_metrics:
            Path(self.config.metrics_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("System monitor initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                if self.config.enable_system_metrics:
                    system_metrics = self._collect_system_metrics()
                    self.performance_history.append(system_metrics)
                    self._check_system_alerts(system_metrics)
                
                # Collect application metrics
                if self.config.enable_application_metrics:
                    app_metrics = self._collect_application_metrics()
                    self.application_history.append(app_metrics)
                    self._check_application_alerts(app_metrics)
                
                # Save metrics if configured
                if self.config.save_metrics:
                    self._save_metrics_snapshot()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.config.collection_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect system performance metrics."""
        metrics = PerformanceMetrics()
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - system metrics limited")
            return metrics
        
        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.cpu_count = psutil.cpu_count()
            
            try:
                metrics.load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                metrics.load_average = [0.0, 0.0, 0.0]
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_total = memory.total
            metrics.memory_available = memory.available
            metrics.memory_used = memory.used
            metrics.memory_percent = memory.percent
            
            # Disk metrics
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    metrics.disk_usage[partition.mountpoint] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.used / usage.total * 100
                    }
                except (PermissionError, OSError):
                    continue
            
            # Network metrics
            try:
                net_io = psutil.net_io_counters()
                metrics.network_io = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            except (AttributeError, OSError):
                pass
            
            # Process metrics
            metrics.process_count = len(psutil.pids())
            try:
                current_process = psutil.Process()
                metrics.thread_count = current_process.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        # GPU metrics
        if self.config.enable_gpu_monitoring and GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                metrics.gpu_count = len(gpus)
                
                for gpu in gpus:
                    metrics.gpu_memory_used.append(int(gpu.memoryUsed))
                    metrics.gpu_memory_total.append(int(gpu.memoryTotal))
                    metrics.gpu_utilization.append(gpu.load * 100)
                    metrics.gpu_temperature.append(gpu.temperature)
                    
            except Exception as e:
                logger.warning(f"Error collecting GPU metrics: {e}")
        
        # Update metrics collector
        self.metrics_collector.set_gauge("cpu_percent", metrics.cpu_percent)
        self.metrics_collector.set_gauge("memory_percent", metrics.memory_percent)
        self.metrics_collector.set_gauge("disk_usage_max", max(
            [usage["percent"] for usage in metrics.disk_usage.values()]
        ) if metrics.disk_usage else 0)
        
        return metrics
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        metrics = ApplicationMetrics()
        
        # Get metrics from collector
        collector_metrics = self.metrics_collector.get_all_metrics()
        
        # Map collector metrics to application metrics
        counters = collector_metrics.get("counters", {})
        gauges = collector_metrics.get("gauges", {})
        timers = collector_metrics.get("timers", {})
        
        metrics.sequences_generated = counters.get("sequences_generated", 0)
        metrics.sequences_ranked = counters.get("sequences_ranked", 0)
        metrics.requests_total = counters.get("requests_total", 0)
        metrics.total_errors = counters.get("total_errors", 0)
        
        metrics.generation_rate = gauges.get("generation_rate", 0.0)
        metrics.ranking_rate = gauges.get("ranking_rate", 0.0)
        metrics.requests_per_second = gauges.get("requests_per_second", 0.0)
        metrics.error_rate = gauges.get("error_rate", 0.0)
        metrics.cache_hit_rate = gauges.get("cache_hit_rate", 0.0)
        
        # Calculate averages from timers
        if "generation_time" in timers:
            metrics.avg_generation_time = timers["generation_time"].get("avg", 0.0)
        if "ranking_time" in timers:
            metrics.avg_ranking_time = timers["ranking_time"].get("avg", 0.0)
        if "response_time" in timers:
            metrics.avg_response_time = timers["response_time"].get("avg", 0.0)
        
        return metrics
    
    def _check_system_alerts(self, metrics: PerformanceMetrics):
        """Check for system-level alerts."""
        current_time = time.time()
        
        # CPU alert
        if metrics.cpu_percent > self.config.cpu_threshold:
            self._create_alert(
                f"high_cpu_{int(current_time)}",
                "High CPU Usage",
                AlertSeverity.WARNING,
                f"CPU usage at {metrics.cpu_percent:.1f}%",
                "system",
                {"cpu_percent": metrics.cpu_percent, "threshold": self.config.cpu_threshold}
            )
        
        # Memory alert
        if metrics.memory_percent > self.config.memory_threshold:
            self._create_alert(
                f"high_memory_{int(current_time)}",
                "High Memory Usage",
                AlertSeverity.WARNING,
                f"Memory usage at {metrics.memory_percent:.1f}%",
                "system",
                {"memory_percent": metrics.memory_percent, "threshold": self.config.memory_threshold}
            )
        
        # Disk alerts
        for mount, usage in metrics.disk_usage.items():
            if usage["percent"] > self.config.disk_threshold:
                self._create_alert(
                    f"high_disk_{mount.replace('/', '_')}_{int(current_time)}",
                    "High Disk Usage",
                    AlertSeverity.ERROR,
                    f"Disk usage on {mount} at {usage['percent']:.1f}%",
                    "system",
                    {"mount": mount, "percent": usage["percent"], "threshold": self.config.disk_threshold}
                )
        
        # GPU alerts
        for i, temp in enumerate(metrics.gpu_temperature):
            if temp > 80:  # 80C threshold
                self._create_alert(
                    f"high_gpu_temp_{i}_{int(current_time)}",
                    "High GPU Temperature",
                    AlertSeverity.WARNING,
                    f"GPU {i} temperature at {temp:.1f}°C",
                    "gpu",
                    {"gpu_id": i, "temperature": temp}
                )
    
    def _check_application_alerts(self, metrics: ApplicationMetrics):
        """Check for application-level alerts."""
        current_time = time.time()
        
        # Error rate alert
        if metrics.error_rate > self.config.error_rate_threshold:
            self._create_alert(
                f"high_error_rate_{int(current_time)}",
                "High Error Rate",
                AlertSeverity.ERROR,
                f"Error rate at {metrics.error_rate:.3f}",
                "application",
                {"error_rate": metrics.error_rate, "threshold": self.config.error_rate_threshold}
            )
        
        # Response time alert
        if metrics.avg_response_time > self.config.response_time_threshold:
            self._create_alert(
                f"high_response_time_{int(current_time)}",
                "High Response Time",
                AlertSeverity.WARNING,
                f"Average response time at {metrics.avg_response_time:.2f}s",
                "application",
                {"response_time": metrics.avg_response_time, "threshold": self.config.response_time_threshold}
            )
    
    def _create_alert(self, alert_id: str, name: str, severity: AlertSeverity, message: str, component: str, context: Dict[str, Any]):
        """Create and process a new alert."""
        # Check if alert already exists and is not resolved
        for existing_alert in self.alerts:
            if existing_alert.alert_id == alert_id and not existing_alert.resolved:
                return  # Don't create duplicate alerts
        
        alert = Alert(
            alert_id=alert_id,
            name=name,
            severity=severity,
            message=message,
            component=component,
            context=context
        )
        
        self.alerts.append(alert)
        logger.log(
            logging.ERROR if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] else logging.WARNING,
            f"Alert: {name} - {message}"
        )
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _cleanup_old_alerts(self):
        """Remove old resolved alerts."""
        current_time = time.time()
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or (current_time - alert.resolved_timestamp) < self.config.alert_retention_time
        ]
    
    def _save_metrics_snapshot(self):
        """Save current metrics to disk."""
        try:
            timestamp = int(time.time())
            metrics_file = Path(self.config.metrics_directory) / f"metrics_{timestamp}.json"
            
            snapshot = {
                "timestamp": timestamp,
                "system_metrics": self.performance_history[-1].__dict__ if self.performance_history else {},
                "application_metrics": self.application_history[-1].__dict__ if self.application_history else {},
                "collector_metrics": self.metrics_collector.get_all_metrics(),
                "active_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "component": alert.component,
                        "timestamp": alert.timestamp,
                        "resolved": alert.resolved
                    }
                    for alert in self.alerts if not alert.resolved
                ]
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to save metrics snapshot: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and application metrics."""
        return {
            "timestamp": time.time(),
            "system_metrics": self.performance_history[-1].__dict__ if self.performance_history else {},
            "application_metrics": self.application_history[-1].__dict__ if self.application_history else {},
            "collector_metrics": self.metrics_collector.get_all_metrics(),
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "monitoring_active": self.monitoring_active
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        current_metrics = self.get_current_metrics()
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        # Determine health status
        if any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            health_status = "critical"
        elif any(a.severity == AlertSeverity.ERROR for a in active_alerts):
            health_status = "unhealthy"
        elif any(a.severity == AlertSeverity.WARNING for a in active_alerts):
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "active_alerts": len(active_alerts),
            "monitoring_active": self.monitoring_active,
            "metrics_available": bool(self.performance_history),
            "uptime": time.time() - self.performance_history[0].timestamp if self.performance_history else 0
        }
    
    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_timestamp = time.time()
                logger.info(f"Alert {alert_id} resolved manually")
                return True
        return False
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def record_custom_metric(self, name: str, value: Union[int, float], metric_type: MetricType = MetricType.GAUGE, tags: Dict[str, str] = None):
        """Record a custom application metric."""
        if metric_type == MetricType.COUNTER:
            self.metrics_collector.increment_counter(name, int(value), tags)
        elif metric_type == MetricType.GAUGE:
            self.metrics_collector.set_gauge(name, float(value), tags)
        elif metric_type == MetricType.TIMER:
            self.metrics_collector.record_timer(name, float(value), tags)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.performance_history or not self.application_history:
            return {"error": "No metrics data available"}
        
        # Calculate averages and trends
        recent_system = list(self.performance_history)[-10:]  # Last 10 samples
        recent_app = list(self.application_history)[-10:]
        
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
        avg_generation_time = sum(m.avg_generation_time for m in recent_app) / len(recent_app)
        avg_response_time = sum(m.avg_response_time for m in recent_app) / len(recent_app)
        
        return {
            "summary_period": "last_10_samples",
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_generation_time": avg_generation_time,
            "avg_response_time": avg_response_time,
            "total_sequences_generated": sum(m.sequences_generated for m in recent_app),
            "total_sequences_ranked": sum(m.sequences_ranked for m in recent_app),
            "total_requests": sum(m.requests_total for m in recent_app),
            "current_error_rate": recent_app[-1].error_rate if recent_app else 0,
            "active_alerts": len([a for a in self.alerts if not a.resolved])
        }


# Decorator for automatic timing
def monitor_execution_time(metric_name: str, monitor: SystemMonitor):
    """Decorator to automatically monitor execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                monitor.record_custom_metric(metric_name, execution_time, MetricType.TIMER)
        return wrapper
    return decorator


# Context manager for monitoring operations
class MonitoredOperation:
    """Context manager for monitoring operations with automatic metrics."""
    
    def __init__(self, monitor: SystemMonitor, operation_name: str, component: str = "application"):
        self.monitor = monitor
        self.operation_name = operation_name
        self.component = component
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.monitor.metrics_collector.increment_counter(f"{self.operation_name}_started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_custom_metric(f"{self.operation_name}_duration", duration, MetricType.TIMER)
            
            if exc_type is None:
                self.monitor.metrics_collector.increment_counter(f"{self.operation_name}_success")
            else:
                self.monitor.metrics_collector.increment_counter(f"{self.operation_name}_error")
