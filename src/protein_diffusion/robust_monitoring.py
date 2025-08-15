"""
Comprehensive Monitoring and Observability Framework for Protein Diffusion Design Lab.

This module provides advanced monitoring, metrics collection, alerting,
and observability capabilities for production environments.
"""

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import queue
import statistics

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # e.g., "> 100", "< 0.5"
    threshold: float
    severity: AlertSeverity
    duration: float = 60.0  # Seconds condition must be true
    description: str = ""
    enabled: bool = True


@dataclass
class Alert:
    """Alert instance."""
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    timestamp: float
    description: str
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    process_count: int
    timestamp: float


class MetricsCollector:
    """Collects and stores metrics with time series support."""
    
    def __init__(self, max_data_points: int = 10000):
        self.max_data_points = max_data_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_data_points))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        
        logger.info("Metrics collector initialized")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        with self._lock:
            metric_key = self._build_metric_key(name, labels)
            self.counters[metric_key] += value
            
            metric_value = MetricValue(
                name=name,
                value=self.counters[metric_key],
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
                labels=labels or {}
            )
            
            self.metrics[metric_key].append(metric_value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        with self._lock:
            metric_key = self._build_metric_key(name, labels)
            self.gauges[metric_key] = value
            
            metric_value = MetricValue(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                labels=labels or {}
            )
            
            self.metrics[metric_key].append(metric_value)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric value."""
        with self._lock:
            metric_key = self._build_metric_key(name, labels)
            
            metric_value = MetricValue(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=time.time(),
                labels=labels or {}
            )
            
            self.metrics[metric_key].append(metric_value)
    
    def _build_metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Build unique metric key including labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metric_values(self, name: str, labels: Dict[str, str] = None) -> List[MetricValue]:
        """Get metric values for a specific metric."""
        metric_key = self._build_metric_key(name, labels)
        with self._lock:
            return list(self.metrics.get(metric_key, []))
    
    def get_latest_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get the latest value for a metric."""
        values = self.get_metric_values(name, labels)
        return values[-1].value if values else None
    
    def get_metric_summary(self, name: str, labels: Dict[str, str] = None, 
                          duration: float = 300.0) -> Dict[str, float]:
        """Get summary statistics for a metric over a time period."""
        values = self.get_metric_values(name, labels)
        
        # Filter by time duration
        cutoff_time = time.time() - duration
        recent_values = [v.value for v in values if v.timestamp >= cutoff_time]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "sum": sum(recent_values),
            "avg": statistics.mean(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "median": statistics.median(recent_values),
            "p95": self._percentile(recent_values, 0.95),
            "p99": self._percentile(recent_values, 0.99)
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, List[MetricValue]]:
        """Get all metrics."""
        with self._lock:
            return {name: list(values) for name, values in self.metrics.items()}


class SystemMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 10.0  # seconds
        
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self._record_system_metrics(metrics)
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics."""
        if not PSUTIL_AVAILABLE:
            # Return mock metrics if psutil not available
            return SystemMetrics(
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_used_mb=2048.0,
                memory_available_mb=2048.0,
                disk_usage_percent=70.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000000,
                network_bytes_recv=1000000,
                load_average=[1.0, 1.0, 1.0],
                process_count=200,
                timestamp=time.time()
            )
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Network metrics
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # Load average
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            load_average = [0.0, 0.0, 0.0]  # Windows doesn't have getloadavg
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            load_average=load_average,
            process_count=process_count,
            timestamp=time.time()
        )
    
    def _record_system_metrics(self, metrics: SystemMetrics):
        """Record system metrics."""
        self.metrics_collector.set_gauge("system_cpu_percent", metrics.cpu_percent)
        self.metrics_collector.set_gauge("system_memory_percent", metrics.memory_percent)
        self.metrics_collector.set_gauge("system_memory_used_mb", metrics.memory_used_mb)
        self.metrics_collector.set_gauge("system_memory_available_mb", metrics.memory_available_mb)
        self.metrics_collector.set_gauge("system_disk_usage_percent", metrics.disk_usage_percent)
        self.metrics_collector.set_gauge("system_disk_free_gb", metrics.disk_free_gb)
        self.metrics_collector.set_gauge("system_network_bytes_sent", metrics.network_bytes_sent)
        self.metrics_collector.set_gauge("system_network_bytes_recv", metrics.network_bytes_recv)
        self.metrics_collector.set_gauge("system_process_count", metrics.process_count)
        
        for i, load in enumerate(metrics.load_average):
            self.metrics_collector.set_gauge(f"system_load_avg_{i + 1}min", load)


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Alert processing
        self.alert_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.processing_thread.start()
        
        # Alert evaluation
        self.evaluating = False
        self.evaluation_thread = None
        self.evaluation_interval = 30.0  # seconds
        
        logger.info("Alert manager initialized")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler."""
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")
    
    def start_evaluation(self):
        """Start alert evaluation."""
        if self.evaluating:
            return
        
        self.evaluating = True
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        
        logger.info("Alert evaluation started")
    
    def stop_evaluation(self):
        """Stop alert evaluation."""
        self.evaluating = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5.0)
        
        logger.info("Alert evaluation stopped")
    
    def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self.evaluating:
            try:
                self._evaluate_alert_rules()
                time.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                time.sleep(self.evaluation_interval)
    
    def _evaluate_alert_rules(self):
        """Evaluate all alert rules."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        # Get latest metric value
        latest_value = self.metrics_collector.get_latest_value(rule.metric_name)
        
        if latest_value is None:
            return
        
        # Evaluate condition
        condition_met = self._evaluate_condition(latest_value, rule.condition, rule.threshold)
        
        if condition_met:
            # Check if alert should be triggered
            if rule.name not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    current_value=latest_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    timestamp=time.time(),
                    description=rule.description or f"{rule.metric_name} {rule.condition} {rule.threshold}"
                )
                
                self.active_alerts[rule.name] = alert
                self.alert_queue.put(alert)
                
                logger.warning(f"Alert triggered: {rule.name}")
        else:
            # Check if alert should be resolved
            if rule.name in self.active_alerts:
                alert = self.active_alerts[rule.name]
                alert.resolved = True
                alert.resolved_timestamp = time.time()
                
                del self.active_alerts[rule.name]
                self.alert_history.append(alert)
                
                logger.info(f"Alert resolved: {rule.name}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        condition = condition.strip()
        
        if condition.startswith('>'):
            return value > threshold
        elif condition.startswith('<'):
            return value < threshold
        elif condition.startswith('>='):
            return value >= threshold
        elif condition.startswith('<='):
            return value <= threshold
        elif condition.startswith('=='):
            return value == threshold
        elif condition.startswith('!='):
            return value != threshold
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    def _process_alerts(self):
        """Process alerts and send notifications."""
        while True:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                
                # Send notifications
                for handler in self.notification_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in notification handler: {e}")
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:] if self.alert_history else []


class ApplicationMonitor:
    """Monitor application-specific metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
    def record_generation_metrics(self, generation_result: Dict[str, Any]):
        """Record protein generation metrics."""
        # Generation performance
        if 'generation_stats' in generation_result:
            stats = generation_result['generation_stats']
            
            self.metrics_collector.record_histogram(
                "generation_time_seconds",
                stats.get('generation_time', 0)
            )
            
            self.metrics_collector.set_gauge(
                "generation_success_rate",
                stats.get('success_rate', 0)
            )
            
            self.metrics_collector.increment_counter(
                "sequences_generated_total",
                stats.get('successful_generations', 0)
            )
            
            self.metrics_collector.record_histogram(
                "sequence_length",
                stats.get('avg_sequence_length', 0)
            )
            
            self.metrics_collector.record_histogram(
                "sequence_confidence",
                stats.get('avg_confidence', 0)
            )
    
    def record_ranking_metrics(self, ranking_result: Dict[str, Any]):
        """Record protein ranking metrics."""
        if 'ranking_stats' in ranking_result:
            stats = ranking_result['ranking_stats']
            
            self.metrics_collector.record_histogram(
                "ranking_time_seconds",
                stats.get('ranking_time', 0)
            )
            
            self.metrics_collector.increment_counter(
                "sequences_ranked_total",
                stats.get('total_ranked', 0)
            )
            
            self.metrics_collector.record_histogram(
                "binding_affinity",
                stats.get('best_binding_affinity', 0)
            )
            
            self.metrics_collector.set_gauge(
                "diversity_score",
                stats.get('diversity_score', 0)
            )
    
    def record_error_metrics(self, error_type: str, operation: str):
        """Record error metrics."""
        self.metrics_collector.increment_counter(
            "errors_total",
            labels={"error_type": error_type, "operation": operation}
        )
    
    def record_request_metrics(self, endpoint: str, method: str, status_code: int, 
                             duration: float):
        """Record HTTP request metrics."""
        self.metrics_collector.increment_counter(
            "http_requests_total",
            labels={"endpoint": endpoint, "method": method, "status": str(status_code)}
        )
        
        self.metrics_collector.record_histogram(
            "http_request_duration_seconds",
            duration,
            labels={"endpoint": endpoint, "method": method}
        )


class MonitoringManager:
    """Main monitoring manager orchestrating all monitoring components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.app_monitor = ApplicationMonitor(self.metrics_collector)
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup default notification handlers
        self._setup_default_notification_handlers()
        
        logger.info("Monitoring manager initialized")
    
    def start_monitoring(self):
        """Start all monitoring components."""
        self.system_monitor.start_monitoring()
        self.alert_manager.start_evaluation()
        
        logger.info("All monitoring components started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        self.alert_manager.stop_evaluation()
        
        logger.info("All monitoring components stopped")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        # System resource alerts
        self.alert_manager.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            metric_name="system_cpu_percent",
            condition="> 80",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            description="High CPU usage detected"
        ))
        
        self.alert_manager.add_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="system_memory_percent",
            condition="> 85",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            description="High memory usage detected"
        ))
        
        self.alert_manager.add_alert_rule(AlertRule(
            name="low_disk_space",
            metric_name="system_disk_free_gb",
            condition="< 5",
            threshold=5.0,
            severity=AlertSeverity.ERROR,
            description="Low disk space remaining"
        ))
        
        # Application performance alerts
        self.alert_manager.add_alert_rule(AlertRule(
            name="slow_generation",
            metric_name="generation_time_seconds",
            condition="> 300",
            threshold=300.0,
            severity=AlertSeverity.WARNING,
            description="Slow protein generation detected"
        ))
        
        self.alert_manager.add_alert_rule(AlertRule(
            name="low_success_rate",
            metric_name="generation_success_rate",
            condition="< 0.8",
            threshold=0.8,
            severity=AlertSeverity.ERROR,
            description="Low generation success rate"
        ))
    
    def _setup_default_notification_handlers(self):
        """Setup default notification handlers."""
        # Log notifications
        def log_notification(alert: Alert):
            level = logging.WARNING if alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR] else logging.CRITICAL
            logger.log(level, f"ALERT: {alert.rule_name} - {alert.description} (value: {alert.current_value})")
        
        self.alert_manager.add_notification_handler(log_notification)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        # System metrics
        system_metrics = {}
        for metric_name in ["system_cpu_percent", "system_memory_percent", "system_disk_usage_percent"]:
            latest_value = self.metrics_collector.get_latest_value(metric_name)
            if latest_value is not None:
                system_metrics[metric_name] = latest_value
        
        # Application metrics
        app_metrics = {}
        for metric_name in ["generation_success_rate", "diversity_score"]:
            latest_value = self.metrics_collector.get_latest_value(metric_name)
            if latest_value is not None:
                app_metrics[metric_name] = latest_value
        
        # Active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Recent errors
        error_count = self.metrics_collector.get_latest_value("errors_total") or 0
        
        return {
            "timestamp": time.time(),
            "system_metrics": system_metrics,
            "application_metrics": app_metrics,
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "error_count": error_count,
            "monitoring_status": "active"
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in various formats."""
        if format == "json":
            return self._export_json_metrics()
        elif format == "prometheus":
            return self._export_prometheus_metrics()
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def _export_json_metrics(self) -> str:
        """Export metrics in JSON format."""
        all_metrics = self.metrics_collector.get_all_metrics()
        
        export_data = {
            "timestamp": time.time(),
            "metrics": {}
        }
        
        for metric_name, values in all_metrics.items():
            if values:
                latest_value = values[-1]
                export_data["metrics"][metric_name] = {
                    "value": latest_value.value,
                    "type": latest_value.metric_type.value,
                    "timestamp": latest_value.timestamp,
                    "labels": latest_value.labels
                }
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        all_metrics = self.metrics_collector.get_all_metrics()
        lines = []
        
        for metric_name, values in all_metrics.items():
            if values:
                latest_value = values[-1]
                
                # Clean metric name for Prometheus
                clean_name = metric_name.replace(".", "_").replace("-", "_")
                
                # Add labels if present
                if latest_value.labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in latest_value.labels.items())
                    metric_line = f'{clean_name}{{{label_str}}} {latest_value.value} {int(latest_value.timestamp * 1000)}'
                else:
                    metric_line = f'{clean_name} {latest_value.value} {int(latest_value.timestamp * 1000)}'
                
                lines.append(metric_line)
        
        return "\n".join(lines)


# Convenience functions and decorators

def monitor_execution_time(metric_name: str, monitoring_manager: Optional[MonitoringManager] = None):
    """Decorator to monitor function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if monitoring_manager:
                    monitoring_manager.metrics_collector.record_histogram(metric_name, duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                if monitoring_manager:
                    monitoring_manager.metrics_collector.record_histogram(metric_name, duration)
                    monitoring_manager.app_monitor.record_error_metrics(
                        type(e).__name__, func.__name__
                    )
                
                raise
        
        return wrapper
    return decorator


# Global monitoring manager instance
_global_monitoring_manager = None

def get_global_monitoring_manager() -> MonitoringManager:
    """Get global monitoring manager instance."""
    global _global_monitoring_manager
    if _global_monitoring_manager is None:
        _global_monitoring_manager = MonitoringManager()
    return _global_monitoring_manager