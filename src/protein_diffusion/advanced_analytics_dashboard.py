"""
Advanced Analytics Dashboard for Protein Diffusion Design Lab

This module provides a comprehensive analytics dashboard with real-time metrics,
interactive visualizations, and advanced insights for protein design workflows.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
import statistics
from collections import defaultdict, deque
from enum import Enum
import logging

# Mock imports for environments without full dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked by the dashboard."""
    GENERATION_RATE = "generation_rate"
    PREDICTION_ACCURACY = "prediction_accuracy"
    BINDING_AFFINITY = "binding_affinity"
    SYSTEM_PERFORMANCE = "system_performance"
    USER_ACTIVITY = "user_activity"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    WORKFLOW_THROUGHPUT = "workflow_throughput"


class TimeWindow(Enum):
    """Time windows for analytics aggregation."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class MetricDataPoint:
    """Represents a single metric data point."""
    timestamp: float
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AnalyticsSummary:
    """Summary statistics for a metric over a time period."""
    metric_type: MetricType
    time_window: TimeWindow
    start_time: float
    end_time: float
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    trend_direction: str  # "up", "down", "stable"
    trend_strength: float  # 0.0 to 1.0


@dataclass
class DashboardConfig:
    """Configuration for the analytics dashboard."""
    retention_days: int = 30
    max_data_points: int = 100000
    real_time_interval_seconds: float = 5.0
    enable_predictions: bool = True
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 2.0  # Standard deviations
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores metrics for analytics."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.data_points: deque = deque(maxlen=config.max_data_points)
        self.aggregated_data: Dict[str, Dict[str, AnalyticsSummary]] = defaultdict(dict)
        self._lock = None  # Would use asyncio.Lock in async context
        
    def add_metric(self, data_point: MetricDataPoint):
        """Add a metric data point."""
        self.data_points.append(data_point)
        self._update_aggregations(data_point)
        
    def add_protein_generation_metric(
        self,
        proteins_generated: int,
        generation_time: float,
        avg_confidence: float,
        temperature: float = 0.8
    ):
        """Add protein generation metrics."""
        timestamp = time.time()
        
        # Generation rate metric
        rate_metric = MetricDataPoint(
            timestamp=timestamp,
            metric_type=MetricType.GENERATION_RATE,
            value=proteins_generated / generation_time,
            metadata={
                'proteins_generated': proteins_generated,
                'generation_time': generation_time,
                'temperature': temperature
            },
            labels={'method': 'diffusion'}
        )
        self.add_metric(rate_metric)
        
        # Average confidence metric
        confidence_metric = MetricDataPoint(
            timestamp=timestamp,
            metric_type=MetricType.PREDICTION_ACCURACY,
            value=avg_confidence,
            metadata={'metric_subtype': 'generation_confidence'},
            labels={'method': 'diffusion'}
        )
        self.add_metric(confidence_metric)
        
    def add_structure_prediction_metric(
        self,
        sequences_predicted: int,
        prediction_time: float,
        avg_confidence: float,
        avg_tm_score: float
    ):
        """Add structure prediction metrics."""
        timestamp = time.time()
        
        # Prediction rate metric
        rate_metric = MetricDataPoint(
            timestamp=timestamp,
            metric_type=MetricType.GENERATION_RATE,
            value=sequences_predicted / prediction_time,
            metadata={
                'sequences_predicted': sequences_predicted,
                'prediction_time': prediction_time
            },
            labels={'method': 'structure_prediction'}
        )
        self.add_metric(rate_metric)
        
        # Prediction accuracy metrics
        confidence_metric = MetricDataPoint(
            timestamp=timestamp,
            metric_type=MetricType.PREDICTION_ACCURACY,
            value=avg_confidence,
            metadata={'metric_subtype': 'prediction_confidence'},
            labels={'method': 'structure_prediction'}
        )
        self.add_metric(confidence_metric)
        
        tm_score_metric = MetricDataPoint(
            timestamp=timestamp,
            metric_type=MetricType.PREDICTION_ACCURACY,
            value=avg_tm_score,
            metadata={'metric_subtype': 'tm_score'},
            labels={'method': 'structure_prediction'}
        )
        self.add_metric(tm_score_metric)
        
    def add_binding_affinity_metric(
        self,
        pairs_calculated: int,
        calculation_time: float,
        avg_affinity: float,
        avg_confidence: float
    ):
        """Add binding affinity calculation metrics."""
        timestamp = time.time()
        
        # Calculation rate metric
        rate_metric = MetricDataPoint(
            timestamp=timestamp,
            metric_type=MetricType.GENERATION_RATE,
            value=pairs_calculated / calculation_time,
            metadata={
                'pairs_calculated': pairs_calculated,
                'calculation_time': calculation_time
            },
            labels={'method': 'binding_affinity'}
        )
        self.add_metric(rate_metric)
        
        # Binding affinity metric
        affinity_metric = MetricDataPoint(
            timestamp=timestamp,
            metric_type=MetricType.BINDING_AFFINITY,
            value=avg_affinity,
            metadata={'avg_confidence': avg_confidence},
            labels={'method': 'autodock'}
        )
        self.add_metric(affinity_metric)
        
    def add_system_performance_metric(
        self,
        cpu_usage: float,
        memory_usage: float,
        gpu_usage: float = 0.0,
        network_throughput: float = 0.0
    ):
        """Add system performance metrics."""
        timestamp = time.time()
        
        for metric_name, value in [
            ('cpu_usage', cpu_usage),
            ('memory_usage', memory_usage),
            ('gpu_usage', gpu_usage),
            ('network_throughput', network_throughput)
        ]:
            if value > 0:  # Only add non-zero metrics
                metric = MetricDataPoint(
                    timestamp=timestamp,
                    metric_type=MetricType.SYSTEM_PERFORMANCE,
                    value=value,
                    metadata={'metric_subtype': metric_name},
                    labels={'resource': metric_name}
                )
                self.add_metric(metric)
                
    def add_error_metric(self, error_count: int, total_requests: int, error_type: str = "general"):
        """Add error rate metrics."""
        timestamp = time.time()
        error_rate = error_count / max(total_requests, 1)
        
        error_metric = MetricDataPoint(
            timestamp=timestamp,
            metric_type=MetricType.ERROR_RATE,
            value=error_rate,
            metadata={
                'error_count': error_count,
                'total_requests': total_requests,
                'error_type': error_type
            },
            labels={'error_type': error_type}
        )
        self.add_metric(error_metric)
        
    def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricDataPoint]:
        """Retrieve metrics with filtering."""
        filtered_metrics = []
        
        for data_point in self.data_points:
            # Filter by metric type
            if metric_type and data_point.metric_type != metric_type:
                continue
                
            # Filter by time range
            if start_time and data_point.timestamp < start_time:
                continue
            if end_time and data_point.timestamp > end_time:
                continue
                
            # Filter by labels
            if labels:
                if not all(data_point.labels.get(k) == v for k, v in labels.items()):
                    continue
                    
            filtered_metrics.append(data_point)
            
        return filtered_metrics
        
    def _update_aggregations(self, data_point: MetricDataPoint):
        """Update aggregated statistics for different time windows."""
        for time_window in TimeWindow:
            window_key = f"{data_point.metric_type.value}_{time_window.value}"
            
            # Get time window boundaries
            window_start, window_end = self._get_time_window_bounds(
                data_point.timestamp, time_window
            )
            
            # Get existing summary or create new one
            summary = self.aggregated_data[window_key].get(window_start)
            if not summary:
                summary = self._create_summary(
                    data_point.metric_type,
                    time_window,
                    window_start,
                    window_end
                )
                self.aggregated_data[window_key][window_start] = summary
                
            # Update summary would go here
            # In a full implementation, we'd update the running statistics
            
    def _get_time_window_bounds(self, timestamp: float, window: TimeWindow) -> Tuple[float, float]:
        """Get start and end bounds for a time window."""
        dt = datetime.fromtimestamp(timestamp)
        
        if window == TimeWindow.MINUTE:
            start = dt.replace(second=0, microsecond=0)
            end = start + timedelta(minutes=1)
        elif window == TimeWindow.HOUR:
            start = dt.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif window == TimeWindow.DAY:
            start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif window == TimeWindow.WEEK:
            start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            start = start - timedelta(days=start.weekday())
            end = start + timedelta(weeks=1)
        elif window == TimeWindow.MONTH:
            start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        else:
            # Default to hour
            start = dt.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
            
        return start.timestamp(), end.timestamp()
        
    def _create_summary(
        self,
        metric_type: MetricType,
        time_window: TimeWindow,
        start_time: float,
        end_time: float
    ) -> AnalyticsSummary:
        """Create a new analytics summary."""
        return AnalyticsSummary(
            metric_type=metric_type,
            time_window=time_window,
            start_time=start_time,
            end_time=end_time,
            count=0,
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            min_value=float('inf'),
            max_value=float('-inf'),
            percentile_95=0.0,
            percentile_99=0.0,
            trend_direction="stable",
            trend_strength=0.0
        )


class AnomalyDetector:
    """Detects anomalies in metric data."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.baseline_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def update_baseline(self, metrics: List[MetricDataPoint]):
        """Update baseline statistics for anomaly detection."""
        metric_groups = defaultdict(list)
        
        # Group metrics by type and labels
        for metric in metrics:
            key = f"{metric.metric_type.value}_{hash(frozenset(metric.labels.items()))}"
            metric_groups[key].append(metric.value)
            
        # Calculate baseline statistics
        for key, values in metric_groups.items():
            if len(values) >= 10:  # Need minimum samples for baseline
                self.baseline_stats[key] = {
                    'mean': statistics.mean(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'count': len(values)
                }
                
    def detect_anomalies(self, metrics: List[MetricDataPoint]) -> List[Dict[str, Any]]:
        """Detect anomalies in recent metrics."""
        anomalies = []
        
        for metric in metrics:
            key = f"{metric.metric_type.value}_{hash(frozenset(metric.labels.items()))}"
            baseline = self.baseline_stats.get(key)
            
            if not baseline:
                continue
                
            # Calculate z-score
            if baseline['std_dev'] > 0:
                z_score = abs(metric.value - baseline['mean']) / baseline['std_dev']
                
                if z_score > self.config.anomaly_threshold:
                    anomalies.append({
                        'timestamp': metric.timestamp,
                        'metric_type': metric.metric_type.value,
                        'value': metric.value,
                        'expected_value': baseline['mean'],
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3.0 else 'medium',
                        'labels': metric.labels,
                        'metadata': metric.metadata
                    })
                    
        return anomalies


class VisualizationEngine:
    """Generates interactive visualizations for the dashboard."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        
    def create_time_series_plot(
        self,
        metrics: List[MetricDataPoint],
        title: str = "Time Series",
        height: int = 400
    ) -> Optional[Any]:
        """Create a time series plot."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for visualization")
            return None
            
        if not metrics:
            return None
            
        # Group metrics by labels for separate traces
        metric_groups = defaultdict(list)
        for metric in metrics:
            labels_key = tuple(sorted(metric.labels.items()))
            metric_groups[labels_key].append(metric)
            
        fig = go.Figure()
        
        # Add trace for each group
        for labels_key, group_metrics in metric_groups.items():
            timestamps = [datetime.fromtimestamp(m.timestamp) for m in group_metrics]
            values = [m.value for m in group_metrics]
            
            trace_name = ", ".join([f"{k}={v}" for k, v in labels_key]) if labels_key else "Default"
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=trace_name,
                line=dict(width=2),
                marker=dict(size=4)
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            height=height,
            hovermode='x unified',
            showlegend=True if len(metric_groups) > 1 else False
        )
        
        return fig
        
    def create_histogram(
        self,
        metrics: List[MetricDataPoint],
        title: str = "Distribution",
        bins: int = 30
    ) -> Optional[Any]:
        """Create a histogram of metric values."""
        if not PLOTLY_AVAILABLE or not metrics:
            return None
            
        values = [m.value for m in metrics]
        
        fig = go.Figure(data=[go.Histogram(
            x=values,
            nbinsx=bins,
            opacity=0.7
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            bargap=0.1
        )
        
        return fig
        
    def create_performance_dashboard(
        self,
        metrics_collector: MetricsCollector,
        time_window_hours: int = 1
    ) -> Optional[Any]:
        """Create a comprehensive performance dashboard."""
        if not PLOTLY_AVAILABLE:
            return None
            
        # Get recent metrics
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Generation Rate', 'Prediction Accuracy',
                'System Performance', 'Binding Affinity',
                'Error Rate', 'Resource Utilization'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Generation rate
        gen_metrics = metrics_collector.get_metrics(
            MetricType.GENERATION_RATE, start_time, end_time
        )
        if gen_metrics:
            timestamps = [datetime.fromtimestamp(m.timestamp) for m in gen_metrics]
            values = [m.value for m in gen_metrics]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name="Gen Rate", mode='lines'),
                row=1, col=1
            )
            
        # Prediction accuracy
        acc_metrics = metrics_collector.get_metrics(
            MetricType.PREDICTION_ACCURACY, start_time, end_time
        )
        if acc_metrics:
            timestamps = [datetime.fromtimestamp(m.timestamp) for m in acc_metrics]
            values = [m.value for m in acc_metrics]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name="Accuracy", mode='lines'),
                row=1, col=2
            )
            
        # System performance (could add multiple traces for CPU, memory, etc.)
        sys_metrics = metrics_collector.get_metrics(
            MetricType.SYSTEM_PERFORMANCE, start_time, end_time
        )
        if sys_metrics:
            # Group by resource type
            resource_groups = defaultdict(list)
            for metric in sys_metrics:
                resource_type = metric.labels.get('resource', 'unknown')
                resource_groups[resource_type].append(metric)
                
            for resource_type, group_metrics in resource_groups.items():
                timestamps = [datetime.fromtimestamp(m.timestamp) for m in group_metrics]
                values = [m.value for m in group_metrics]
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name=resource_type, mode='lines'),
                    row=2, col=1
                )
                
        # Binding affinity
        binding_metrics = metrics_collector.get_metrics(
            MetricType.BINDING_AFFINITY, start_time, end_time
        )
        if binding_metrics:
            timestamps = [datetime.fromtimestamp(m.timestamp) for m in binding_metrics]
            values = [m.value for m in binding_metrics]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name="Binding", mode='lines'),
                row=2, col=2
            )
            
        # Error rate
        error_metrics = metrics_collector.get_metrics(
            MetricType.ERROR_RATE, start_time, end_time
        )
        if error_metrics:
            timestamps = [datetime.fromtimestamp(m.timestamp) for m in error_metrics]
            values = [m.value for m in error_metrics]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name="Errors", mode='lines'),
                row=3, col=1
            )
            
        # Resource utilization (similar to system performance)
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Resources", mode='lines'),
            row=3, col=2
        )
        
        fig.update_layout(
            title="Protein Diffusion Analytics Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
        
    def create_real_time_gauge(
        self,
        current_value: float,
        min_value: float = 0.0,
        max_value: float = 100.0,
        title: str = "Metric",
        unit: str = ""
    ) -> Optional[Any]:
        """Create a real-time gauge chart."""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': (max_value + min_value) / 2},
            gauge={
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_value, max_value * 0.7], 'color': "lightgray"},
                    {'range': [max_value * 0.7, max_value * 0.9], 'color': "yellow"},
                    {'range': [max_value * 0.9, max_value], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        
        return fig


class AdvancedAnalyticsDashboard:
    """
    Advanced Analytics Dashboard
    
    Provides comprehensive analytics, visualizations, and insights for
    protein diffusion design workflows.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.visualization_engine = VisualizationEngine(config)
        
        # Dashboard state
        self.is_running = False
        self.last_update = time.time()
        
        # Real-time data
        self.current_metrics = {
            'generation_rate': 0.0,
            'prediction_accuracy': 0.0,
            'system_cpu': 0.0,
            'system_memory': 0.0,
            'error_rate': 0.0
        }
        
        logger.info("Advanced Analytics Dashboard initialized")
        
    def start_real_time_monitoring(self):
        """Start real-time monitoring and data collection."""
        self.is_running = True
        logger.info("Started real-time monitoring")
        
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        self.is_running = False
        logger.info("Stopped real-time monitoring")
        
    def record_protein_generation(
        self,
        proteins_generated: int,
        generation_time: float,
        avg_confidence: float,
        temperature: float = 0.8
    ):
        """Record protein generation metrics."""
        self.metrics_collector.add_protein_generation_metric(
            proteins_generated, generation_time, avg_confidence, temperature
        )
        
        # Update current metrics
        self.current_metrics['generation_rate'] = proteins_generated / generation_time
        self.current_metrics['prediction_accuracy'] = avg_confidence
        
        logger.debug(f"Recorded protein generation: {proteins_generated} proteins in {generation_time:.2f}s")
        
    def record_structure_prediction(
        self,
        sequences_predicted: int,
        prediction_time: float,
        avg_confidence: float,
        avg_tm_score: float
    ):
        """Record structure prediction metrics."""
        self.metrics_collector.add_structure_prediction_metric(
            sequences_predicted, prediction_time, avg_confidence, avg_tm_score
        )
        
        logger.debug(f"Recorded structure prediction: {sequences_predicted} sequences in {prediction_time:.2f}s")
        
    def record_binding_affinity(
        self,
        pairs_calculated: int,
        calculation_time: float,
        avg_affinity: float,
        avg_confidence: float
    ):
        """Record binding affinity calculation metrics."""
        self.metrics_collector.add_binding_affinity_metric(
            pairs_calculated, calculation_time, avg_affinity, avg_confidence
        )
        
        logger.debug(f"Recorded binding affinity: {pairs_calculated} pairs in {calculation_time:.2f}s")
        
    def record_system_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        gpu_usage: float = 0.0,
        network_throughput: float = 0.0
    ):
        """Record system performance metrics."""
        self.metrics_collector.add_system_performance_metric(
            cpu_usage, memory_usage, gpu_usage, network_throughput
        )
        
        # Update current system metrics
        self.current_metrics['system_cpu'] = cpu_usage
        self.current_metrics['system_memory'] = memory_usage
        
    def record_error(self, error_count: int, total_requests: int, error_type: str = "general"):
        """Record error metrics."""
        self.metrics_collector.add_error_metric(error_count, total_requests, error_type)
        
        # Update current error rate
        self.current_metrics['error_rate'] = error_count / max(total_requests, 1)
        
    def get_dashboard_data(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        # Get all metrics for the time window
        all_metrics = []
        for metric_type in MetricType:
            metrics = self.metrics_collector.get_metrics(
                metric_type, start_time, end_time
            )
            all_metrics.extend(metrics)
            
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(all_metrics)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(all_metrics)
        
        return {
            'timestamp': end_time,
            'time_window_hours': time_window_hours,
            'total_metrics': len(all_metrics),
            'current_metrics': self.current_metrics.copy(),
            'summary_stats': summary_stats,
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'dashboard_status': {
                'is_running': self.is_running,
                'last_update': self.last_update,
                'uptime_hours': (end_time - self.last_update) / 3600 if self.is_running else 0
            }
        }
        
    def create_visualizations(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Create dashboard visualizations."""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not available for visualizations'}
            
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        visualizations = {}
        
        # Generation rate time series
        gen_metrics = self.metrics_collector.get_metrics(
            MetricType.GENERATION_RATE, start_time, end_time
        )
        if gen_metrics:
            visualizations['generation_rate'] = self.visualization_engine.create_time_series_plot(
                gen_metrics, "Protein Generation Rate"
            )
            
        # Prediction accuracy time series
        acc_metrics = self.metrics_collector.get_metrics(
            MetricType.PREDICTION_ACCURACY, start_time, end_time
        )
        if acc_metrics:
            visualizations['prediction_accuracy'] = self.visualization_engine.create_time_series_plot(
                acc_metrics, "Prediction Accuracy"
            )
            
        # System performance dashboard
        visualizations['performance_dashboard'] = self.visualization_engine.create_performance_dashboard(
            self.metrics_collector, time_window_hours
        )
        
        # Real-time gauges
        visualizations['cpu_gauge'] = self.visualization_engine.create_real_time_gauge(
            self.current_metrics['system_cpu'], 0, 100, "CPU Usage", "%"
        )
        
        visualizations['memory_gauge'] = self.visualization_engine.create_real_time_gauge(
            self.current_metrics['system_memory'], 0, 100, "Memory Usage", "%"
        )
        
        visualizations['error_rate_gauge'] = self.visualization_engine.create_real_time_gauge(
            self.current_metrics['error_rate'] * 100, 0, 10, "Error Rate", "%"
        )
        
        return visualizations
        
    def get_insights(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get analytical insights from the data."""
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        insights = {
            'performance_insights': [],
            'anomaly_insights': [],
            'trend_insights': [],
            'recommendations': []
        }
        
        # Performance insights
        gen_metrics = self.metrics_collector.get_metrics(
            MetricType.GENERATION_RATE, start_time, end_time
        )
        if gen_metrics:
            avg_rate = statistics.mean([m.value for m in gen_metrics])
            insights['performance_insights'].append({
                'metric': 'generation_rate',
                'insight': f'Average generation rate: {avg_rate:.2f} proteins/second',
                'value': avg_rate
            })
            
        # Anomaly insights
        all_metrics = []
        for metric_type in MetricType:
            metrics = self.metrics_collector.get_metrics(
                metric_type, start_time, end_time
            )
            all_metrics.extend(metrics)
            
        anomalies = self.anomaly_detector.detect_anomalies(all_metrics)
        if anomalies:
            high_severity_count = len([a for a in anomalies if a['severity'] == 'high'])
            insights['anomaly_insights'].append({
                'total_anomalies': len(anomalies),
                'high_severity_anomalies': high_severity_count,
                'insight': f'Detected {len(anomalies)} anomalies, {high_severity_count} high severity'
            })
            
        # Trend insights
        if gen_metrics and len(gen_metrics) >= 10:
            recent_values = [m.value for m in gen_metrics[-10:]]
            earlier_values = [m.value for m in gen_metrics[:10]]
            
            if statistics.mean(recent_values) > statistics.mean(earlier_values):
                insights['trend_insights'].append({
                    'metric': 'generation_rate',
                    'trend': 'improving',
                    'insight': 'Generation rate is trending upward'
                })
                
        # Recommendations
        if self.current_metrics['error_rate'] > 0.05:  # 5% error rate
            insights['recommendations'].append({
                'priority': 'high',
                'recommendation': 'Error rate is high. Check system logs and consider scaling resources.',
                'metric': 'error_rate',
                'current_value': self.current_metrics['error_rate']
            })
            
        if self.current_metrics['system_cpu'] > 80:  # 80% CPU usage
            insights['recommendations'].append({
                'priority': 'medium',
                'recommendation': 'CPU usage is high. Consider scaling up computational resources.',
                'metric': 'cpu_usage',
                'current_value': self.current_metrics['system_cpu']
            })
            
        return insights
        
    def _calculate_summary_stats(self, metrics: List[MetricDataPoint]) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics by metric type."""
        stats = {}
        
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.metric_type.value].append(metric.value)
            
        for metric_type, values in metric_groups.items():
            if values:
                stats[metric_type] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values)
                }
                
                # Add percentiles if we have enough data
                if len(values) >= 20:
                    sorted_values = sorted(values)
                    stats[metric_type]['percentile_95'] = sorted_values[int(0.95 * len(values))]
                    stats[metric_type]['percentile_99'] = sorted_values[int(0.99 * len(values))]
                    
        return stats
        
    def export_data(
        self,
        format: str = "json",
        time_window_hours: int = 24
    ) -> Union[str, bytes]:
        """Export dashboard data in various formats."""
        dashboard_data = self.get_dashboard_data(time_window_hours)
        
        if format.lower() == "json":
            return json.dumps(dashboard_data, indent=2, default=str)
        elif format.lower() == "csv" and PANDAS_AVAILABLE:
            # Convert metrics to DataFrame and export as CSV
            end_time = time.time()
            start_time = end_time - (time_window_hours * 3600)
            
            all_metrics = []
            for metric_type in MetricType:
                metrics = self.metrics_collector.get_metrics(
                    metric_type, start_time, end_time
                )
                for metric in metrics:
                    all_metrics.append({
                        'timestamp': metric.timestamp,
                        'metric_type': metric.metric_type.value,
                        'value': metric.value,
                        'labels': str(metric.labels),
                        'metadata': str(metric.metadata)
                    })
                    
            df = pd.DataFrame(all_metrics)
            return df.to_csv(index=False)
        else:
            return json.dumps({'error': f'Unsupported format: {format}'})


# Demo and testing functions
def simulate_protein_generation_workflow():
    """Simulate a protein generation workflow with metrics."""
    config = DashboardConfig(
        retention_days=1,
        enable_anomaly_detection=True,
        enable_predictions=True
    )
    
    dashboard = AdvancedAnalyticsDashboard(config)
    dashboard.start_real_time_monitoring()
    
    # Simulate protein generation events
    import random
    
    for i in range(20):
        # Simulate protein generation
        num_proteins = random.randint(50, 200)
        generation_time = random.uniform(5.0, 30.0)
        avg_confidence = random.uniform(0.8, 0.95)
        temperature = random.choice([0.6, 0.8, 1.0])
        
        dashboard.record_protein_generation(
            num_proteins, generation_time, avg_confidence, temperature
        )
        
        # Simulate system metrics
        cpu_usage = random.uniform(20, 80)
        memory_usage = random.uniform(40, 90)
        gpu_usage = random.uniform(10, 95)
        
        dashboard.record_system_metrics(cpu_usage, memory_usage, gpu_usage)
        
        # Simulate occasional errors
        if random.random() < 0.1:  # 10% chance of errors
            dashboard.record_error(1, 10, "generation_error")
        else:
            dashboard.record_error(0, 10)
            
        time.sleep(0.1)  # Simulate time between events
        
    # Get dashboard data and insights
    dashboard_data = dashboard.get_dashboard_data(1)  # 1 hour window
    insights = dashboard.get_insights(1)
    
    print("Dashboard Data:")
    print(json.dumps(dashboard_data, indent=2, default=str))
    
    print("\nInsights:")
    print(json.dumps(insights, indent=2, default=str))
    
    dashboard.stop_real_time_monitoring()
    
    return dashboard


if __name__ == "__main__":
    # Run simulation
    dashboard = simulate_protein_generation_workflow()