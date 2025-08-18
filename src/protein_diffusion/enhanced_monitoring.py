"""
Enhanced Monitoring and Observability System for Protein Diffusion Design Lab

This module provides comprehensive monitoring, metrics collection, alerting,
and observability features for the protein generation and analysis pipeline.
"""

import time
import threading
import json
import logging
import statistics
import psutil
import os
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import queue
import uuid
from datetime import datetime, timedelta

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricEvent:
    """Represents a metric event."""
    name: str
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, timer


@dataclass
class AlertRule:
    """Defines an alert rule."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: Union[int, float]
    duration_seconds: int = 60
    severity: str = "warning"  # info, warning, error, critical
    callback: Optional[Callable] = None


class MetricsCollector:
    """Collects and stores metrics."""
    
    def __init__(self, max_metrics: int = 100000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.RLock()
        self.max_metrics = max_metrics
        self.start_time = time.time()
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric (current value)."""
        with self.lock:
            self.gauges[name] = value
            event = MetricEvent(name, value, tags=tags or {}, metric_type="gauge")
            self.metrics[name].append(event)
    
    def counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
            event = MetricEvent(name, self.counters[name], tags=tags or {}, metric_type="counter")
            self.metrics[name].append(event)
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        with self.lock:
            self.histograms[name].append(value)
            # Keep only last 1000 values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            
            event = MetricEvent(name, value, tags=tags or {}, metric_type="histogram")
            self.metrics[name].append(event)
    
    def timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer value."""
        with self.lock:
            self.timers[name].append(duration)
            event = MetricEvent(name, duration, tags=tags or {}, metric_type="timer")
            self.metrics[name].append(event)
    
    @contextmanager
    def time_block(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing code blocks."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timer(name, duration, tags)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return {"error": "metric_not_found"}
            
            events = list(self.metrics[name])
            if not events:
                return {"error": "no_data"}
            
            values = [event.value for event in events]
            recent_values = [event.value for event in events if time.time() - event.timestamp < 300]
            
            summary = {
                "name": name,
                "total_events": len(events),
                "recent_events": len(recent_values),
                "latest_value": values[-1] if values else None,
                "latest_timestamp": events[-1].timestamp if events else None
            }
            
            if values:
                summary.update({
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values)
                })
                
                if len(values) > 1:
                    summary["std"] = statistics.stdev(values)
                
                if recent_values:
                    summary.update({
                        "recent_min": min(recent_values),
                        "recent_max": max(recent_values),
                        "recent_mean": statistics.mean(recent_values)
                    })
            
            # Add percentiles for histograms
            if name in self.histograms and self.histograms[name]:
                hist_values = sorted(self.histograms[name])
                summary["percentiles"] = {
                    "p50": self._percentile(hist_values, 50),
                    "p90": self._percentile(hist_values, 90),
                    "p95": self._percentile(hist_values, 95),
                    "p99": self._percentile(hist_values, 99)
                }
            
            return summary
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0
        
        index = (percentile / 100) * (len(values) - 1)
        lower = int(index)
        upper = lower + 1
        
        if upper >= len(values):
            return values[-1]
        
        weight = index - lower
        return values[lower] * (1 - weight) + values[upper] * weight
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get summaries of all metrics."""
        with self.lock:
            return {
                metric_name: self.get_metric_summary(metric_name)
                for metric_name in self.metrics.keys()
            }
    
    def cleanup_old_metrics(self, max_age_seconds: int = 3600):
        """Remove metrics older than specified age."""
        cutoff_time = time.time() - max_age_seconds
        
        with self.lock:
            for metric_name, events in self.metrics.items():
                # Filter out old events
                recent_events = deque(
                    [event for event in events if event.timestamp >= cutoff_time],
                    maxlen=events.maxlen
                )
                self.metrics[metric_name] = recent_events


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds
    
    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._collect_gpu_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.gauge("system.cpu.usage_percent", cpu_percent)
            
            cpu_count = psutil.cpu_count()
            self.metrics.gauge("system.cpu.count", cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.gauge("system.memory.total_gb", memory.total / (1024**3))
            self.metrics.gauge("system.memory.available_gb", memory.available / (1024**3))
            self.metrics.gauge("system.memory.used_percent", memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')\n            self.metrics.gauge("system.disk.total_gb", disk.total / (1024**3))
            self.metrics.gauge("system.disk.free_gb", disk.free / (1024**3))
            self.metrics.gauge("system.disk.used_percent", (disk.used / disk.total) * 100)
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self.metrics.gauge("process.memory.rss_mb", process_memory.rss / (1024**2))
            self.metrics.gauge("process.memory.vms_mb", process_memory.vms / (1024**2))
            self.metrics.gauge("process.cpu.percent", process.cpu_percent())
            
            # Load average (Unix only)
            if hasattr(os, 'getloadavg'):
                load1, load5, load15 = os.getloadavg()
                self.metrics.gauge("system.load.1min", load1)
                self.metrics.gauge("system.load.5min", load5)
                self.metrics.gauge("system.load.15min", load15)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            import torch
            if not torch.cuda.is_available():
                return
            
            for i in range(torch.cuda.device_count()):
                device = f"cuda:{i}"
                
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)   # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                self.metrics.gauge(f"gpu.{i}.memory.allocated_gb", memory_allocated)
                self.metrics.gauge(f"gpu.{i}.memory.reserved_gb", memory_reserved)
                self.metrics.gauge(f"gpu.{i}.memory.total_gb", memory_total)
                self.metrics.gauge(f"gpu.{i}.memory.utilization_percent", 
                                 (memory_allocated + memory_reserved) / memory_total * 100)
                
                # Device properties
                props = torch.cuda.get_device_properties(i)
                self.metrics.gauge(f"gpu.{i}.compute_capability", 
                                 props.major + props.minor / 10)
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.checking_active = False
        self.checking_thread = None
        self.check_interval = 30  # seconds
        self.lock = threading.RLock()
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self.lock:
            self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        with self.lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                if rule_name in self.active_alerts:
                    del self.active_alerts[rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    def start_alert_checking(self):
        """Start alert checking in background thread."""
        if self.checking_active:
            return
        
        self.checking_active = True
        self.checking_thread = threading.Thread(target=self._alert_checking_loop, daemon=True)
        self.checking_thread.start()
        logger.info("Alert checking started")
    
    def stop_alert_checking(self):
        """Stop alert checking."""
        self.checking_active = False
        if self.checking_thread:
            self.checking_thread.join(timeout=5)
        logger.info("Alert checking stopped")
    
    def _alert_checking_loop(self):
        """Main alert checking loop."""
        while self.checking_active:
            try:
                self._check_all_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in alert checking loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_all_alerts(self):
        """Check all alert rules."""
        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                try:
                    self._check_alert_rule(rule)
                except Exception as e:
                    logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _check_alert_rule(self, rule: AlertRule):
        """Check a single alert rule."""
        # Get current metric value
        metric_summary = self.metrics.get_metric_summary(rule.metric_name)
        if "error" in metric_summary:
            return
        
        current_value = metric_summary.get("latest_value")
        if current_value is None:
            return
        
        # Check condition
        condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)
        
        current_time = time.time()
        
        if condition_met:
            # Check if alert is already active
            if rule.name in self.active_alerts:
                alert = self.active_alerts[rule.name]
                # Check if duration threshold met
                if current_time - alert["start_time"] >= rule.duration_seconds:
                    # Update existing alert
                    alert["last_trigger_time"] = current_time
                    alert["trigger_count"] += 1
                    alert["current_value"] = current_value
            else:
                # Create new alert
                alert = {
                    "rule_name": rule.name,
                    "metric_name": rule.metric_name,
                    "severity": rule.severity,
                    "condition": f"{rule.metric_name} {rule.condition} {rule.threshold}",
                    "current_value": current_value,
                    "threshold": rule.threshold,
                    "start_time": current_time,
                    "last_trigger_time": current_time,
                    "trigger_count": 1,
                    "status": "pending"
                }
                self.active_alerts[rule.name] = alert
                
                # Fire alert after duration
                if rule.duration_seconds == 0:
                    self._fire_alert(alert, rule)
        else:
            # Condition not met, resolve alert if active
            if rule.name in self.active_alerts:
                alert = self.active_alerts[rule.name]
                alert["status"] = "resolved"
                alert["resolved_time"] = current_time
                
                # Move to history and remove from active
                self.alert_history.append(alert.copy())
                del self.active_alerts[rule.name]
                
                logger.info(f"Alert resolved: {rule.name}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 1e-10
        elif condition == "ne":
            return abs(value - threshold) >= 1e-10
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    def _fire_alert(self, alert: Dict, rule: AlertRule):
        """Fire an alert."""
        alert["status"] = "firing"
        alert["fired_time"] = time.time()
        
        # Log alert
        logger.error(f"ALERT FIRED: {alert['rule_name']} - {alert['condition']} "
                    f"(current: {alert['current_value']}, threshold: {alert['threshold']})")
        
        # Call custom callback if provided
        if rule.callback:
            try:
                rule.callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Add to history
        self.alert_history.append(alert.copy())
    
    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history."""
        with self.lock:
            return list(self.alert_history)[-limit:]


class PerformanceProfiler:
    """Profiles performance of functions and operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.profiles: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    @contextmanager
    def profile(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory if start_memory and end_memory else 0
            
            # Record metrics
            self.metrics.timer(f"operation.{operation_name}.duration", duration, tags)
            if memory_delta != 0:
                self.metrics.histogram(f"operation.{operation_name}.memory_delta_mb", memory_delta, tags)
            
            # Update profile statistics
            with self.lock:
                if operation_name not in self.profiles:
                    self.profiles[operation_name] = {
                        'call_count': 0,
                        'total_duration': 0,
                        'min_duration': float('inf'),
                        'max_duration': 0,
                        'durations': deque(maxlen=1000)
                    }
                
                profile = self.profiles[operation_name]
                profile['call_count'] += 1
                profile['total_duration'] += duration
                profile['min_duration'] = min(profile['min_duration'], duration)
                profile['max_duration'] = max(profile['max_duration'], duration)
                profile['durations'].append(duration)
                profile['avg_duration'] = profile['total_duration'] / profile['call_count']
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return None
    
    def get_profile_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get profile summary for an operation."""
        with self.lock:
            if operation_name not in self.profiles:
                return {"error": "profile_not_found"}
            
            profile = self.profiles[operation_name]
            durations = list(profile['durations'])
            
            summary = {
                'operation': operation_name,
                'call_count': profile['call_count'],
                'total_duration': profile['total_duration'],
                'avg_duration': profile['avg_duration'],
                'min_duration': profile['min_duration'],
                'max_duration': profile['max_duration']
            }
            
            if durations:
                summary.update({
                    'median_duration': statistics.median(durations),
                    'p95_duration': self.metrics._percentile(sorted(durations), 95),
                    'p99_duration': self.metrics._percentile(sorted(durations), 99)
                })
                
                if len(durations) > 1:
                    summary['std_duration'] = statistics.stdev(durations)
            
            return summary
    
    def get_all_profiles(self) -> Dict[str, Any]:
        """Get all profile summaries."""
        with self.lock:
            return {
                operation: self.get_profile_summary(operation)
                for operation in self.profiles.keys()
            }


class MonitoringDashboard:
    """Provides dashboard-style monitoring data."""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 alert_manager: AlertManager, 
                 profiler: PerformanceProfiler):
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.profiler = profiler
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_time = time.time()
        
        # System overview
        system_metrics = {}
        for metric_name in ["system.cpu.usage_percent", "system.memory.used_percent", 
                          "system.disk.used_percent"]:
            summary = self.metrics.get_metric_summary(metric_name)
            if "error" not in summary:
                system_metrics[metric_name] = summary.get("latest_value", 0)
        
        # GPU overview
        gpu_metrics = {}
        gpu_count = 0
        if TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    for i in range(gpu_count):
                        metric_name = f"gpu.{i}.memory.utilization_percent"
                        summary = self.metrics.get_metric_summary(metric_name)
                        if "error" not in summary:
                            gpu_metrics[f"gpu_{i}_utilization"] = summary.get("latest_value", 0)
            except:
                pass
        
        # Performance overview
        key_operations = ["generation", "ranking", "validation", "structure_prediction"]
        performance_data = {}
        for op in key_operations:
            profile = self.profiler.get_profile_summary(op)
            if "error" not in profile:
                performance_data[op] = {
                    "avg_duration": profile.get("avg_duration", 0),
                    "call_count": profile.get("call_count", 0),
                    "p95_duration": profile.get("p95_duration", 0)
                }
        
        # Alert overview
        active_alerts = self.alerts.get_active_alerts()
        alert_summary = {
            "active_count": len(active_alerts),
            "critical_count": len([a for a in active_alerts if a["severity"] == "critical"]),
            "warning_count": len([a for a in active_alerts if a["severity"] == "warning"])
        }
        
        # Application metrics
        app_metrics = {}
        for metric_name in ["protein.generation.count", "protein.generation.success_rate",
                          "protein.ranking.count", "protein.ranking.average_score"]:
            summary = self.metrics.get_metric_summary(metric_name)
            if "error" not in summary:
                app_metrics[metric_name] = summary.get("latest_value", 0)
        
        return {
            "timestamp": current_time,
            "uptime_seconds": current_time - self.metrics.start_time,
            "system_metrics": system_metrics,
            "gpu_metrics": gpu_metrics,
            "gpu_count": gpu_count,
            "performance_data": performance_data,
            "alert_summary": alert_summary,
            "application_metrics": app_metrics,
            "health_score": self._calculate_health_score(
                system_metrics, active_alerts, performance_data
            )
        }
    
    def _calculate_health_score(self, system_metrics: Dict, active_alerts: List, 
                              performance_data: Dict) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # Penalize high resource usage
        cpu_usage = system_metrics.get("system.cpu.usage_percent", 0)
        memory_usage = system_metrics.get("system.memory.used_percent", 0)
        
        if cpu_usage > 80:
            score -= 20
        elif cpu_usage > 60:
            score -= 10
        
        if memory_usage > 90:
            score -= 30
        elif memory_usage > 80:
            score -= 15
        
        # Penalize active alerts
        if len(active_alerts) > 0:
            score -= len(active_alerts) * 10
            critical_alerts = len([a for a in active_alerts if a.get("severity") == "critical"])
            score -= critical_alerts * 20
        
        # Consider performance degradation
        for op, data in performance_data.items():
            avg_duration = data.get("avg_duration", 0)
            if avg_duration > 10:  # More than 10 seconds
                score -= 15
            elif avg_duration > 5:  # More than 5 seconds
                score -= 5
        
        return max(0, min(100, score))


# Global instances
metrics_collector = MetricsCollector()
system_monitor = SystemMonitor(metrics_collector)
alert_manager = AlertManager(metrics_collector)
performance_profiler = PerformanceProfiler(metrics_collector)
monitoring_dashboard = MonitoringDashboard(metrics_collector, alert_manager, performance_profiler)


# Default alert rules
def setup_default_alerts():
    """Setup default alert rules."""
    
    # CPU usage alert
    cpu_alert = AlertRule(
        name="high_cpu_usage",
        metric_name="system.cpu.usage_percent",
        condition="gt",
        threshold=85.0,
        duration_seconds=300,  # 5 minutes
        severity="warning"
    )
    alert_manager.add_alert_rule(cpu_alert)
    
    # Memory usage alert
    memory_alert = AlertRule(
        name="high_memory_usage",
        metric_name="system.memory.used_percent",
        condition="gt",
        threshold=90.0,
        duration_seconds=60,  # 1 minute
        severity="critical"
    )
    alert_manager.add_alert_rule(memory_alert)
    
    # Disk usage alert
    disk_alert = AlertRule(
        name="high_disk_usage",
        metric_name="system.disk.used_percent",
        condition="gt",
        threshold=85.0,
        duration_seconds=300,  # 5 minutes
        severity="warning"
    )
    alert_manager.add_alert_rule(disk_alert)
    
    # GPU memory alert (if available)
    if TORCH_AVAILABLE:
        gpu_alert = AlertRule(
            name="high_gpu_memory",
            metric_name="gpu.0.memory.utilization_percent",
            condition="gt",
            threshold=90.0,
            duration_seconds=60,
            severity="warning"
        )
        alert_manager.add_alert_rule(gpu_alert)


def start_monitoring():
    """Start all monitoring components."""
    setup_default_alerts()
    system_monitor.start_monitoring()
    alert_manager.start_alert_checking()
    logger.info("Enhanced monitoring system started")


def stop_monitoring():
    """Stop all monitoring components."""
    system_monitor.stop_monitoring()
    alert_manager.stop_alert_checking()
    logger.info("Enhanced monitoring system stopped")


def get_monitoring_status() -> Dict[str, Any]:
    """Get comprehensive monitoring status."""
    return {
        "dashboard_data": monitoring_dashboard.get_dashboard_data(),
        "metrics_summary": metrics_collector.get_all_metrics(),
        "active_alerts": alert_manager.get_active_alerts(),
        "performance_profiles": performance_profiler.get_all_profiles()
    }