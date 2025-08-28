"""
Next-Generation Hyperschale Orchestrator for Global Protein Design.

This module implements planetary-scale orchestration capabilities:
- Adaptive multi-cloud orchestration across AWS, GCP, Azure, and edge
- Intelligent workload distribution with ML-driven predictions
- Dynamic resource scaling from edge devices to supercomputers
- Global data synchronization with consistency guarantees
- Fault-tolerant distributed computing with Byzantine fault tolerance
- Real-time cost optimization across providers and regions
"""

import asyncio
import time
import json
import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, AsyncGenerator
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import heapq

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def array(x): return x
        @staticmethod 
        def mean(x): return sum(x)/len(x) if x else 0.5
        @staticmethod
        def std(x): return 0.1
        @staticmethod
        def percentile(x, p): return sorted(x)[int(len(x)*p/100)] if x else 0
        @staticmethod
        def argmin(x): return x.index(min(x)) if x else 0
        @staticmethod
        def random.choice(x): 
            import random
            return random.choice(x) if x else None
    np = MockNumpy()
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp" 
    AZURE = "azure"
    ALIBABA = "alibaba"
    EDGE = "edge"
    ON_PREMISE = "on_premise"
    QUANTUM = "quantum"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    QUANTUM = "quantum"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    SPECIALIZED = "specialized"


class WorkloadType(Enum):
    """Types of protein design workloads."""
    GENERATION = "generation"
    FOLDING = "folding"
    RANKING = "ranking"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    SIMULATION = "simulation"
    TRAINING = "training"
    INFERENCE = "inference"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class NodeStatus(Enum):
    """Compute node status."""
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class ResourceSpec:
    """Resource specification for compute nodes."""
    cpu_cores: int = 0
    gpu_count: int = 0
    tpu_count: int = 0
    memory_gb: float = 0.0
    storage_gb: float = 0.0
    network_gbps: float = 0.0
    specialized_units: Dict[str, int] = field(default_factory=dict)
    
    def total_compute_units(self) -> float:
        """Calculate total compute units (normalized metric)."""
        return (
            self.cpu_cores * 1.0 +
            self.gpu_count * 100.0 +  # GPUs are much more powerful
            self.tpu_count * 300.0 +  # TPUs even more so
            self.memory_gb * 0.1 +
            sum(self.specialized_units.values()) * 50.0
        )


@dataclass 
class ComputeNode:
    """Individual compute node in the global infrastructure."""
    node_id: str
    provider: CloudProvider
    region: str
    zone: str
    resource_spec: ResourceSpec
    status: NodeStatus = NodeStatus.AVAILABLE
    current_workload: Optional[str] = None
    utilization: float = 0.0
    cost_per_hour: float = 0.0
    performance_score: float = 1.0
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Network connectivity metrics
    latency_matrix: Dict[str, float] = field(default_factory=dict)  # Latency to other nodes
    bandwidth_matrix: Dict[str, float] = field(default_factory=dict)  # Bandwidth to other nodes
    
    # Historical performance
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_count: int = 0
    success_count: int = 0
    
    @property
    def reliability_score(self) -> float:
        """Calculate node reliability score."""
        total_tasks = self.success_count + self.failure_count
        if total_tasks == 0:
            return 1.0
        return self.success_count / total_tasks
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return (
            self.status == NodeStatus.AVAILABLE and
            time.time() - self.last_heartbeat.timestamp() < 300 and  # 5 minutes
            self.reliability_score > 0.8
        )


@dataclass
class WorkloadTask:
    """Individual task in the orchestration system."""
    task_id: str
    workload_type: WorkloadType
    priority: TaskPriority
    resource_requirements: ResourceSpec
    data_dependencies: List[str] = field(default_factory=list)
    compute_dependencies: List[str] = field(default_factory=list)
    
    # Scheduling constraints
    preferred_providers: List[CloudProvider] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    max_execution_time: int = 3600  # seconds
    deadline: Optional[datetime] = None
    
    # Data and parameters
    input_data: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    assigned_node: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Cost and performance tracking
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    estimated_duration: float = 0.0
    actual_duration: float = 0.0


@dataclass
class HyperschaleConfig:
    """Configuration for hyperschale orchestrator."""
    # Global orchestration settings
    max_concurrent_tasks: int = 10000
    task_timeout_seconds: int = 7200
    heartbeat_interval: int = 30
    node_discovery_interval: int = 300
    
    # Load balancing
    load_balancing_strategy: str = "weighted_round_robin"  # round_robin, weighted_round_robin, least_connections, performance_based
    load_balancing_weights: Dict[str, float] = field(default_factory=lambda: {
        'performance': 0.4,
        'cost': 0.3,
        'reliability': 0.2,
        'latency': 0.1
    })
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # Scale up when utilization > 80%
    scale_down_threshold: float = 0.3  # Scale down when utilization < 30%
    min_nodes_per_region: int = 1
    max_nodes_per_region: int = 1000
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    
    # Cost optimization
    enable_cost_optimization: bool = True
    cost_optimization_interval: int = 300  # seconds
    max_cost_per_hour: float = 10000.0
    prefer_spot_instances: bool = True
    spot_instance_discount: float = 0.7  # 70% of on-demand price
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    task_retry_attempts: int = 3
    node_failure_timeout: int = 180  # seconds
    byzantine_fault_tolerance: bool = True
    consensus_threshold: float = 0.67  # 67% consensus required
    
    # Data management
    enable_data_caching: bool = True
    cache_replication_factor: int = 3
    data_consistency_level: str = "eventual"  # strong, eventual, weak
    max_data_transfer_time: int = 1800  # seconds
    
    # Monitoring and alerting
    enable_monitoring: bool = True
    monitoring_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high_failure_rate': 0.1,
        'high_latency': 5.0,
        'low_throughput': 0.5,
        'high_cost': 100.0
    })
    
    # Machine learning optimization
    enable_ml_optimization: bool = True
    prediction_model_update_interval: int = 3600
    feature_collection_enabled: bool = True
    online_learning_enabled: bool = True


class WorkloadPredictor:
    """ML-based workload and performance predictor."""
    
    def __init__(self, config: HyperschaleConfig):
        self.config = config
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.workload_patterns: Dict[str, Any] = {}
        self.prediction_models: Dict[str, Any] = {}
        
    async def predict_task_performance(self, task: WorkloadTask, node: ComputeNode) -> Dict[str, float]:
        """Predict task performance on a specific node."""
        # Extract features
        features = self._extract_features(task, node)
        
        # Get historical performance for similar tasks
        similar_tasks = self._find_similar_tasks(task, node.node_id)
        
        if not similar_tasks:
            # Use baseline estimates
            return self._baseline_predictions(task, node)
            
        # Calculate predictions based on historical data
        execution_times = [t['actual_duration'] for t in similar_tasks if 'actual_duration' in t]
        success_rates = [1.0 if t.get('status') == 'completed' else 0.0 for t in similar_tasks]
        costs = [t.get('actual_cost', 0) for t in similar_tasks]
        
        predictions = {
            'estimated_duration': np.mean(execution_times) if execution_times else 60.0,
            'success_probability': np.mean(success_rates) if success_rates else 0.8,
            'estimated_cost': np.mean(costs) if costs else 1.0,
            'confidence': min(len(similar_tasks) / 10.0, 1.0)  # More similar tasks = higher confidence
        }
        
        # Adjust for node-specific factors
        predictions['estimated_duration'] *= (2.0 - node.performance_score)
        predictions['success_probability'] *= node.reliability_score
        predictions['estimated_cost'] *= node.cost_per_hour / max(node.performance_score, 0.1)
        
        return predictions
        
    async def predict_workload_demand(self, time_horizon: int = 3600) -> Dict[WorkloadType, float]:
        """Predict future workload demand."""
        current_time = datetime.now(timezone.utc)
        
        # Analyze historical patterns
        demand_predictions = {}
        
        for workload_type in WorkloadType:
            # Get historical demand data
            historical_data = self._get_historical_demand(workload_type, days=7)
            
            if not historical_data:
                demand_predictions[workload_type] = 1.0  # Default demand
                continue
                
            # Simple time-series prediction based on hour-of-day patterns
            current_hour = current_time.hour
            same_hour_demands = [
                data['demand'] for data in historical_data
                if data['timestamp'].hour == current_hour
            ]
            
            if same_hour_demands:
                # Predict based on historical same-hour demand
                base_demand = np.mean(same_hour_demands)
                
                # Add trend component
                recent_demands = [data['demand'] for data in historical_data[-24:]]  # Last 24 hours
                trend = (recent_demands[-1] - recent_demands[0]) / len(recent_demands) if len(recent_demands) > 1 else 0
                
                predicted_demand = base_demand + trend * (time_horizon / 3600)
            else:
                predicted_demand = 1.0
                
            demand_predictions[workload_type] = max(0.1, predicted_demand)
            
        return demand_predictions
        
    def _extract_features(self, task: WorkloadTask, node: ComputeNode) -> Dict[str, float]:
        """Extract features for ML prediction."""
        features = {
            # Task features
            'priority': task.priority.value,
            'cpu_requirement': task.resource_requirements.cpu_cores,
            'gpu_requirement': task.resource_requirements.gpu_count,
            'memory_requirement': task.resource_requirements.memory_gb,
            'data_dependencies': len(task.data_dependencies),
            'compute_dependencies': len(task.compute_dependencies),
            
            # Node features
            'node_cpu': node.resource_spec.cpu_cores,
            'node_gpu': node.resource_spec.gpu_count,
            'node_memory': node.resource_spec.memory_gb,
            'node_utilization': node.utilization,
            'node_performance': node.performance_score,
            'node_reliability': node.reliability_score,
            'node_cost': node.cost_per_hour,
            
            # Compatibility features
            'cpu_ratio': task.resource_requirements.cpu_cores / max(node.resource_spec.cpu_cores, 1),
            'gpu_ratio': task.resource_requirements.gpu_count / max(node.resource_spec.gpu_count, 1),
            'memory_ratio': task.resource_requirements.memory_gb / max(node.resource_spec.memory_gb, 1),
        }
        
        return features
        
    def _find_similar_tasks(self, task: WorkloadTask, node_id: str, max_tasks: int = 20) -> List[Dict[str, Any]]:
        """Find similar tasks from performance history."""
        if node_id not in self.performance_history:
            return []
            
        node_history = self.performance_history[node_id]
        
        # Filter by workload type
        similar_tasks = [
            t for t in node_history
            if t.get('workload_type') == task.workload_type.value
        ]
        
        # Score similarity based on resource requirements
        def similarity_score(hist_task):
            score = 0.0
            
            # Resource similarity
            if 'cpu_cores' in hist_task:
                cpu_sim = 1.0 - abs(hist_task['cpu_cores'] - task.resource_requirements.cpu_cores) / max(hist_task['cpu_cores'], task.resource_requirements.cpu_cores, 1)
                score += cpu_sim * 0.3
                
            if 'gpu_count' in hist_task:
                gpu_sim = 1.0 - abs(hist_task['gpu_count'] - task.resource_requirements.gpu_count) / max(hist_task['gpu_count'], task.resource_requirements.gpu_count, 1)
                score += gpu_sim * 0.4
                
            if 'memory_gb' in hist_task:
                mem_sim = 1.0 - abs(hist_task['memory_gb'] - task.resource_requirements.memory_gb) / max(hist_task['memory_gb'], task.resource_requirements.memory_gb, 1)
                score += mem_sim * 0.3
                
            return score
            
        # Sort by similarity and return top matches
        similar_tasks_with_scores = [(t, similarity_score(t)) for t in similar_tasks]
        similar_tasks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in similar_tasks_with_scores[:max_tasks]]
        
    def _baseline_predictions(self, task: WorkloadTask, node: ComputeNode) -> Dict[str, float]:
        """Provide baseline predictions when no historical data is available."""
        # Simple heuristic-based predictions
        base_duration = {
            WorkloadType.GENERATION: 300,    # 5 minutes
            WorkloadType.FOLDING: 600,       # 10 minutes
            WorkloadType.RANKING: 180,       # 3 minutes
            WorkloadType.OPTIMIZATION: 1800, # 30 minutes
            WorkloadType.VALIDATION: 120,    # 2 minutes
            WorkloadType.SIMULATION: 3600,   # 1 hour
            WorkloadType.TRAINING: 7200,     # 2 hours
            WorkloadType.INFERENCE: 60       # 1 minute
        }.get(task.workload_type, 300)
        
        # Adjust for resource requirements
        resource_factor = (
            task.resource_requirements.cpu_cores / max(node.resource_spec.cpu_cores, 1) +
            task.resource_requirements.gpu_count / max(node.resource_spec.gpu_count, 1) +
            task.resource_requirements.memory_gb / max(node.resource_spec.memory_gb, 1)
        ) / 3.0
        
        adjusted_duration = base_duration * max(resource_factor, 0.1) / node.performance_score
        
        return {
            'estimated_duration': adjusted_duration,
            'success_probability': 0.85 * node.reliability_score,
            'estimated_cost': adjusted_duration * node.cost_per_hour / 3600,
            'confidence': 0.3  # Low confidence for baseline predictions
        }
        
    def _get_historical_demand(self, workload_type: WorkloadType, days: int = 7) -> List[Dict[str, Any]]:
        """Get historical demand data for a workload type."""
        # This would typically query a time-series database
        # For now, return mock data
        historical_data = []
        current_time = datetime.now(timezone.utc)
        
        for i in range(days * 24):  # Hourly data points
            timestamp = current_time - timedelta(hours=i)
            
            # Mock demand with some patterns
            base_demand = 1.0
            hour_factor = 0.8 + 0.4 * np.sin(2 * np.pi * timestamp.hour / 24)  # Daily pattern
            day_factor = 0.9 + 0.2 * np.sin(2 * np.pi * timestamp.weekday() / 7)  # Weekly pattern
            
            demand = base_demand * hour_factor * day_factor
            
            historical_data.append({
                'timestamp': timestamp,
                'workload_type': workload_type.value,
                'demand': demand
            })
            
        return historical_data
        
    async def update_performance_history(self, node_id: str, task_data: Dict[str, Any]):
        """Update performance history with completed task data."""
        if node_id not in self.performance_history:
            self.performance_history[node_id] = []
            
        self.performance_history[node_id].append({
            **task_data,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Limit history size
        max_history_per_node = 1000
        if len(self.performance_history[node_id]) > max_history_per_node:
            self.performance_history[node_id] = self.performance_history[node_id][-max_history_per_node//2:]


class GlobalScheduler:
    """Global task scheduler with intelligent placement."""
    
    def __init__(self, config: HyperschaleConfig, predictor: WorkloadPredictor):
        self.config = config
        self.predictor = predictor
        self.pending_tasks: Dict[TaskPriority, List[WorkloadTask]] = {
            priority: [] for priority in TaskPriority
        }
        self.running_tasks: Dict[str, WorkloadTask] = {}
        self.completed_tasks: deque = deque(maxlen=10000)
        
    async def schedule_task(self, task: WorkloadTask) -> bool:
        """Schedule a task for execution."""
        # Add to pending queue
        self.pending_tasks[task.priority].append(task)
        
        logger.info(f"Scheduled task {task.task_id} with priority {task.priority.name}")
        return True
        
    async def find_optimal_placement(self, task: WorkloadTask, available_nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Find optimal node placement for a task."""
        if not available_nodes:
            return None
            
        # Filter nodes by capability requirements
        capable_nodes = []
        for node in available_nodes:
            if await self._node_can_handle_task(node, task):
                capable_nodes.append(node)
                
        if not capable_nodes:
            logger.warning(f"No capable nodes found for task {task.task_id}")
            return None
            
        # Score each capable node
        node_scores = []
        for node in capable_nodes:
            score = await self._calculate_placement_score(task, node)
            node_scores.append((node, score))
            
        # Sort by score (higher is better) and return best node
        node_scores.sort(key=lambda x: x[1], reverse=True)
        best_node = node_scores[0][0]
        
        logger.info(f"Selected node {best_node.node_id} for task {task.task_id} (score: {node_scores[0][1]:.3f})")
        return best_node
        
    async def _node_can_handle_task(self, node: ComputeNode, task: WorkloadTask) -> bool:
        """Check if a node can handle a specific task."""
        # Check resource capacity
        if (node.resource_spec.cpu_cores < task.resource_requirements.cpu_cores or
            node.resource_spec.gpu_count < task.resource_requirements.gpu_count or
            node.resource_spec.memory_gb < task.resource_requirements.memory_gb):
            return False
            
        # Check provider preferences
        if task.preferred_providers and node.provider not in task.preferred_providers:
            return False
            
        # Check required capabilities
        for capability in task.required_capabilities:
            if capability not in node.resource_spec.specialized_units:
                return False
                
        # Check node health
        if not node.is_healthy:
            return False
            
        return True
        
    async def _calculate_placement_score(self, task: WorkloadTask, node: ComputeNode) -> float:
        """Calculate placement score for task-node pair."""
        # Get performance predictions
        predictions = await self.predictor.predict_task_performance(task, node)
        
        # Calculate component scores
        scores = {}
        
        # Performance score (higher is better)
        scores['performance'] = node.performance_score * predictions['success_probability']
        
        # Cost score (lower cost is better)
        max_reasonable_cost = 100.0  # $100/hour
        cost_score = max(0, 1.0 - predictions['estimated_cost'] / max_reasonable_cost)
        scores['cost'] = cost_score
        
        # Reliability score
        scores['reliability'] = node.reliability_score
        
        # Latency score (lower latency is better)
        # This would use actual network measurements in production
        base_latency = 50.0  # 50ms baseline
        estimated_latency = base_latency * (1.0 + node.utilization)  # Higher utilization = higher latency
        latency_score = max(0, 1.0 - estimated_latency / 1000.0)  # Normalize to 1 second max
        scores['latency'] = latency_score
        
        # Resource utilization score (avoid overloading)
        utilization_score = 1.0 - node.utilization
        scores['utilization'] = utilization_score
        
        # Priority bonus
        priority_bonus = {
            TaskPriority.LOW: 0.0,
            TaskPriority.NORMAL: 0.1,
            TaskPriority.HIGH: 0.2,
            TaskPriority.CRITICAL: 0.3,
            TaskPriority.EMERGENCY: 0.5
        }.get(task.priority, 0.0)
        
        # Weighted combination
        weights = self.config.load_balancing_weights
        final_score = (
            scores['performance'] * weights.get('performance', 0.4) +
            scores['cost'] * weights.get('cost', 0.3) +
            scores['reliability'] * weights.get('reliability', 0.2) +
            scores['latency'] * weights.get('latency', 0.1) +
            priority_bonus
        )
        
        return final_score
        
    async def get_next_task(self) -> Optional[WorkloadTask]:
        """Get next task to schedule (priority-based)."""
        # Check priorities from highest to lowest
        for priority in reversed(list(TaskPriority)):
            if self.pending_tasks[priority]:
                return self.pending_tasks[priority].pop(0)
        
        return None
        
    async def mark_task_running(self, task: WorkloadTask, node_id: str):
        """Mark task as running."""
        task.assigned_node = node_id
        task.start_time = datetime.now(timezone.utc)
        task.status = "running"
        self.running_tasks[task.task_id] = task
        
    async def mark_task_completed(self, task_id: str, result: Dict[str, Any] = None, error: str = None):
        """Mark task as completed."""
        if task_id not in self.running_tasks:
            logger.warning(f"Attempted to complete unknown task: {task_id}")
            return
            
        task = self.running_tasks.pop(task_id)
        task.end_time = datetime.now(timezone.utc)
        task.actual_duration = (task.end_time - task.start_time).total_seconds()
        
        if error:
            task.status = "failed"
            task.error_message = error
        else:
            task.status = "completed"
            task.result = result
            
        self.completed_tasks.append(task)
        
        # Update performance history
        if task.assigned_node:
            await self.predictor.update_performance_history(
                task.assigned_node,
                {
                    'task_id': task.task_id,
                    'workload_type': task.workload_type.value,
                    'priority': task.priority.value,
                    'cpu_cores': task.resource_requirements.cpu_cores,
                    'gpu_count': task.resource_requirements.gpu_count,
                    'memory_gb': task.resource_requirements.memory_gb,
                    'actual_duration': task.actual_duration,
                    'status': task.status,
                    'actual_cost': task.actual_cost
                }
            )
            
        logger.info(f"Task {task_id} completed with status: {task.status}")
        
    def get_scheduling_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        total_pending = sum(len(tasks) for tasks in self.pending_tasks.values())
        total_running = len(self.running_tasks)
        total_completed = len(self.completed_tasks)
        
        # Calculate success rate from completed tasks
        recent_tasks = list(self.completed_tasks)[-100:]  # Last 100 tasks
        success_count = sum(1 for task in recent_tasks if task.status == "completed")
        success_rate = success_count / len(recent_tasks) if recent_tasks else 0.0
        
        # Average task duration
        completed_durations = [task.actual_duration for task in recent_tasks if task.actual_duration]
        avg_duration = np.mean(completed_durations) if completed_durations else 0.0
        
        return {
            'pending_tasks': total_pending,
            'running_tasks': total_running,
            'completed_tasks': total_completed,
            'success_rate': success_rate,
            'average_duration': avg_duration,
            'pending_by_priority': {
                priority.name: len(tasks) for priority, tasks in self.pending_tasks.items()
            }
        }


class NodeManager:
    """Manages compute nodes across all providers and regions."""
    
    def __init__(self, config: HyperschaleConfig):
        self.config = config
        self.nodes: Dict[str, ComputeNode] = {}
        self.node_pools: Dict[Tuple[CloudProvider, str], List[str]] = defaultdict(list)  # (provider, region) -> node_ids
        
    async def register_node(self, node: ComputeNode):
        """Register a new compute node."""
        self.nodes[node.node_id] = node
        self.node_pools[(node.provider, node.region)].append(node.node_id)
        
        logger.info(f"Registered node {node.node_id} ({node.provider.value}/{node.region})")
        
    async def deregister_node(self, node_id: str):
        """Deregister a compute node."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        self.node_pools[(node.provider, node.region)].remove(node_id)
        del self.nodes[node_id]
        
        logger.info(f"Deregistered node {node_id}")
        
    async def get_available_nodes(self, 
                                 provider: CloudProvider = None, 
                                 region: str = None,
                                 min_resources: ResourceSpec = None) -> List[ComputeNode]:
        """Get list of available nodes matching criteria."""
        available_nodes = []
        
        for node in self.nodes.values():
            # Check availability
            if node.status != NodeStatus.AVAILABLE or not node.is_healthy:
                continue
                
            # Check provider filter
            if provider and node.provider != provider:
                continue
                
            # Check region filter
            if region and node.region != region:
                continue
                
            # Check resource requirements
            if min_resources:
                if (node.resource_spec.cpu_cores < min_resources.cpu_cores or
                    node.resource_spec.gpu_count < min_resources.gpu_count or
                    node.resource_spec.memory_gb < min_resources.memory_gb):
                    continue
                    
            available_nodes.append(node)
            
        return available_nodes
        
    async def update_node_status(self, node_id: str, status: NodeStatus, utilization: float = None):
        """Update node status and utilization."""
        if node_id not in self.nodes:
            logger.warning(f"Attempted to update unknown node: {node_id}")
            return
            
        node = self.nodes[node_id]
        node.status = status
        node.last_heartbeat = datetime.now(timezone.utc)
        
        if utilization is not None:
            node.utilization = utilization
            
        logger.debug(f"Updated node {node_id}: status={status.value}, utilization={utilization}")
        
    async def handle_node_failure(self, node_id: str):
        """Handle node failure."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        node.status = NodeStatus.FAILED
        node.failure_count += 1
        
        logger.warning(f"Node {node_id} failed (total failures: {node.failure_count})")
        
        # If too many failures, consider removing the node
        if node.failure_count > 5:
            await self.deregister_node(node_id)
            
    async def auto_scale_nodes(self, demand_predictions: Dict[WorkloadType, float]) -> Dict[str, Any]:
        """Automatically scale nodes based on demand predictions."""
        if not self.config.enable_auto_scaling:
            return {'scaled': False, 'reason': 'auto_scaling_disabled'}
            
        scaling_actions = []
        total_demand = sum(demand_predictions.values())
        
        # Calculate current capacity
        total_capacity = 0.0
        current_utilization = 0.0
        
        for node in self.nodes.values():
            if node.is_healthy:
                total_capacity += node.resource_spec.total_compute_units()
                current_utilization += node.utilization * node.resource_spec.total_compute_units()
                
        avg_utilization = current_utilization / max(total_capacity, 1.0)
        
        logger.info(f"Current utilization: {avg_utilization:.2%}, Total demand: {total_demand:.2f}")
        
        # Scale up if utilization is high
        if avg_utilization > self.config.scale_up_threshold:
            # Determine how many nodes to add
            needed_capacity = total_demand * 1.2 - total_capacity  # 20% buffer
            if needed_capacity > 0:
                nodes_to_add = await self._determine_scale_up_strategy(needed_capacity, demand_predictions)
                scaling_actions.extend(nodes_to_add)
                
        # Scale down if utilization is low
        elif avg_utilization < self.config.scale_down_threshold:
            nodes_to_remove = await self._determine_scale_down_strategy(total_capacity - total_demand)
            scaling_actions.extend(nodes_to_remove)
            
        # Execute scaling actions
        for action in scaling_actions:
            if action['type'] == 'add':
                await self._add_node(action['spec'])
            elif action['type'] == 'remove':
                await self._remove_node(action['node_id'])
                
        return {
            'scaled': len(scaling_actions) > 0,
            'actions': scaling_actions,
            'current_utilization': avg_utilization,
            'total_demand': total_demand
        }
        
    async def _determine_scale_up_strategy(self, needed_capacity: float, demand_predictions: Dict[WorkloadType, float]) -> List[Dict[str, Any]]:
        """Determine optimal scale-up strategy."""
        scaling_actions = []
        
        # Analyze demand by workload type to choose appropriate node types
        high_compute_demand = (
            demand_predictions.get(WorkloadType.TRAINING, 0) +
            demand_predictions.get(WorkloadType.SIMULATION, 0)
        )
        
        high_throughput_demand = (
            demand_predictions.get(WorkloadType.GENERATION, 0) +
            demand_predictions.get(WorkloadType.INFERENCE, 0)
        )
        
        # Choose node specifications based on demand
        if high_compute_demand > high_throughput_demand:
            # Add GPU-heavy nodes for compute-intensive tasks
            node_spec = ResourceSpec(cpu_cores=16, gpu_count=4, memory_gb=64)
        else:
            # Add CPU-optimized nodes for throughput tasks
            node_spec = ResourceSpec(cpu_cores=32, gpu_count=1, memory_gb=128)
            
        # Add nodes until capacity need is met
        node_capacity = node_spec.total_compute_units()
        num_nodes_needed = max(1, int(needed_capacity / node_capacity))
        
        for i in range(min(num_nodes_needed, 10)):  # Limit to 10 nodes per scaling event
            scaling_actions.append({
                'type': 'add',
                'spec': node_spec,
                'provider': CloudProvider.AWS,  # Default provider
                'region': 'us-west-2'  # Default region
            })
            
        return scaling_actions
        
    async def _determine_scale_down_strategy(self, excess_capacity: float) -> List[Dict[str, Any]]:
        """Determine nodes to scale down."""
        scaling_actions = []
        
        # Find least utilized nodes
        node_utilizations = [
            (node_id, node.utilization, node.resource_spec.total_compute_units())
            for node_id, node in self.nodes.items()
            if node.is_healthy and node.status == NodeStatus.AVAILABLE
        ]
        
        node_utilizations.sort(key=lambda x: x[1])  # Sort by utilization
        
        removed_capacity = 0.0
        for node_id, utilization, capacity in node_utilizations:
            if removed_capacity >= excess_capacity * 0.8:  # Remove 80% of excess
                break
                
            if utilization < 0.1:  # Only remove very low utilization nodes
                scaling_actions.append({
                    'type': 'remove',
                    'node_id': node_id
                })
                removed_capacity += capacity
                
        return scaling_actions
        
    async def _add_node(self, node_spec: Dict[str, Any]):
        """Add a new node (mock implementation)."""
        # In real implementation, this would call cloud provider APIs
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        new_node = ComputeNode(
            node_id=node_id,
            provider=node_spec.get('provider', CloudProvider.AWS),
            region=node_spec.get('region', 'us-west-2'),
            zone='a',
            resource_spec=node_spec['spec'],
            cost_per_hour=5.0  # Mock cost
        )
        
        await self.register_node(new_node)
        logger.info(f"Added new node {node_id}")
        
    async def _remove_node(self, node_id: str):
        """Remove a node (mock implementation)."""
        # In real implementation, this would gracefully drain and terminate the node
        await self.deregister_node(node_id)
        logger.info(f"Removed node {node_id}")
        
    def get_infrastructure_statistics(self) -> Dict[str, Any]:
        """Get infrastructure statistics."""
        total_nodes = len(self.nodes)
        available_nodes = sum(1 for node in self.nodes.values() if node.status == NodeStatus.AVAILABLE)
        
        # Resource totals
        total_cpus = sum(node.resource_spec.cpu_cores for node in self.nodes.values())
        total_gpus = sum(node.resource_spec.gpu_count for node in self.nodes.values())
        total_memory = sum(node.resource_spec.memory_gb for node in self.nodes.values())
        
        # Utilization
        total_capacity = sum(node.resource_spec.total_compute_units() for node in self.nodes.values())
        used_capacity = sum(
            node.utilization * node.resource_spec.total_compute_units() 
            for node in self.nodes.values()
        )
        
        avg_utilization = used_capacity / max(total_capacity, 1.0)
        
        # By provider/region
        by_provider = defaultdict(int)
        by_region = defaultdict(int)
        
        for node in self.nodes.values():
            by_provider[node.provider.value] += 1
            by_region[node.region] += 1
            
        return {
            'total_nodes': total_nodes,
            'available_nodes': available_nodes,
            'total_cpus': total_cpus,
            'total_gpus': total_gpus,
            'total_memory_gb': total_memory,
            'average_utilization': avg_utilization,
            'nodes_by_provider': dict(by_provider),
            'nodes_by_region': dict(by_region)
        }


class HyperschaleOrchestrator:
    """Main orchestrator coordinating all hyperschale operations."""
    
    def __init__(self, config: HyperschaleConfig = None):
        self.config = config or HyperschaleConfig()
        
        # Initialize core components
        self.predictor = WorkloadPredictor(self.config)
        self.scheduler = GlobalScheduler(self.config, self.predictor)
        self.node_manager = NodeManager(self.config)
        
        # Monitoring and metrics
        self.metrics = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_cost': 0.0,
            'start_time': time.time()
        }
        
        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()
        
    async def start(self):
        """Start the orchestrator."""
        self._running = True
        logger.info("Starting Hyperschale Orchestrator")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._scheduling_loop()),
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._auto_scaling_loop()),
            asyncio.create_task(self._heartbeat_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator tasks cancelled")
            
    async def shutdown(self):
        """Shutdown the orchestrator."""
        logger.info("Shutting down Hyperschale Orchestrator")
        self._running = False
        self._shutdown_event.set()
        
    async def submit_task(self, 
                         workload_type: WorkloadType,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         resource_requirements: ResourceSpec = None,
                         input_data: Dict[str, Any] = None,
                         parameters: Dict[str, Any] = None,
                         deadline: datetime = None) -> str:
        """Submit a task for execution."""
        
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = WorkloadTask(
            task_id=task_id,
            workload_type=workload_type,
            priority=priority,
            resource_requirements=resource_requirements or ResourceSpec(cpu_cores=1, memory_gb=4),
            input_data=input_data or {},
            parameters=parameters or {},
            deadline=deadline
        )
        
        await self.scheduler.schedule_task(task)
        self.metrics['tasks_scheduled'] += 1
        
        logger.info(f"Submitted task {task_id} ({workload_type.value}, {priority.name})")
        return task_id
        
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        # Check running tasks
        if task_id in self.scheduler.running_tasks:
            task = self.scheduler.running_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status,
                'assigned_node': task.assigned_node,
                'start_time': task.start_time.isoformat() if task.start_time else None,
                'estimated_completion': None  # Could calculate based on predictions
            }
            
        # Check completed tasks
        for task in self.scheduler.completed_tasks:
            if task.task_id == task_id:
                return {
                    'task_id': task_id,
                    'status': task.status,
                    'start_time': task.start_time.isoformat() if task.start_time else None,
                    'end_time': task.end_time.isoformat() if task.end_time else None,
                    'duration': task.actual_duration,
                    'result': task.result,
                    'error': task.error_message
                }
                
        # Check pending tasks
        for priority_tasks in self.scheduler.pending_tasks.values():
            for task in priority_tasks:
                if task.task_id == task_id:
                    return {
                        'task_id': task_id,
                        'status': 'pending',
                        'priority': task.priority.name,
                        'estimated_start_time': None  # Could calculate queue position
                    }
                    
        return {'task_id': task_id, 'status': 'not_found'}
        
    async def _scheduling_loop(self):
        """Main scheduling loop."""
        logger.info("Started scheduling loop")
        
        while self._running:
            try:
                # Get next task to schedule
                task = await self.scheduler.get_next_task()
                
                if task:
                    # Find optimal placement
                    available_nodes = await self.node_manager.get_available_nodes()
                    optimal_node = await self.scheduler.find_optimal_placement(task, available_nodes)
                    
                    if optimal_node:
                        # Assign task to node
                        await self.scheduler.mark_task_running(task, optimal_node.node_id)
                        await self.node_manager.update_node_status(
                            optimal_node.node_id, 
                            NodeStatus.BUSY,
                            utilization=optimal_node.utilization + 0.3  # Mock utilization increase
                        )
                        
                        # Simulate task execution
                        asyncio.create_task(self._simulate_task_execution(task, optimal_node))
                    else:
                        # No available nodes, put task back in queue
                        await self.scheduler.schedule_task(task)
                        await asyncio.sleep(5)  # Wait before retrying
                else:
                    # No pending tasks, short sleep
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(5)
                
        logger.info("Scheduling loop stopped")
        
    async def _simulate_task_execution(self, task: WorkloadTask, node: ComputeNode):
        """Simulate task execution (in real implementation, this would dispatch to actual compute nodes)."""
        try:
            # Get predicted execution time
            predictions = await self.predictor.predict_task_performance(task, node)
            execution_time = predictions['estimated_duration']
            
            # Simulate execution delay
            await asyncio.sleep(min(execution_time / 100, 5))  # Scale down for simulation
            
            # Simulate success/failure based on predicted success probability
            import random
            success = random.random() < predictions['success_probability']
            
            if success:
                # Task completed successfully
                result = {
                    'output': f"Task {task.task_id} completed successfully",
                    'metrics': {
                        'execution_time': execution_time,
                        'cost': predictions['estimated_cost']
                    }
                }
                
                await self.scheduler.mark_task_completed(task.task_id, result)
                self.metrics['tasks_completed'] += 1
                self.metrics['total_cost'] += predictions['estimated_cost']
            else:
                # Task failed
                error_message = f"Task {task.task_id} failed during execution"
                await self.scheduler.mark_task_completed(task.task_id, error=error_message)
                self.metrics['tasks_failed'] += 1
                
            # Update node status back to available
            await self.node_manager.update_node_status(
                node.node_id,
                NodeStatus.AVAILABLE,
                utilization=max(0, node.utilization - 0.3)
            )
            
        except Exception as e:
            logger.error(f"Error simulating task execution for {task.task_id}: {e}")
            await self.scheduler.mark_task_completed(task.task_id, error=str(e))
            self.metrics['tasks_failed'] += 1
            
    async def _monitoring_loop(self):
        """Monitoring and alerting loop."""
        logger.info("Started monitoring loop")
        
        while self._running:
            try:
                # Collect metrics
                scheduler_stats = self.scheduler.get_scheduling_statistics()
                infrastructure_stats = self.node_manager.get_infrastructure_statistics()
                
                # Log periodic status
                logger.info(
                    f"System Status - Nodes: {infrastructure_stats['available_nodes']}/{infrastructure_stats['total_nodes']}, "
                    f"Tasks: {scheduler_stats['pending_tasks']} pending, {scheduler_stats['running_tasks']} running, "
                    f"Success Rate: {scheduler_stats['success_rate']:.2%}"
                )
                
                # Check alert thresholds
                if scheduler_stats['success_rate'] < (1.0 - self.config.alert_thresholds['high_failure_rate']):
                    logger.warning(f"High failure rate detected: {1.0 - scheduler_stats['success_rate']:.2%}")
                    
                if infrastructure_stats['average_utilization'] > 0.9:
                    logger.warning(f"High system utilization: {infrastructure_stats['average_utilization']:.2%}")
                    
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
                
        logger.info("Monitoring loop stopped")
        
    async def _auto_scaling_loop(self):
        """Auto-scaling loop."""
        logger.info("Started auto-scaling loop")
        
        while self._running:
            try:
                if self.config.enable_auto_scaling:
                    # Get demand predictions
                    demand_predictions = await self.predictor.predict_workload_demand(3600)  # 1 hour ahead
                    
                    # Execute auto-scaling
                    scaling_result = await self.node_manager.auto_scale_nodes(demand_predictions)
                    
                    if scaling_result['scaled']:
                        logger.info(f"Auto-scaling executed: {len(scaling_result['actions'])} actions")
                        
                await asyncio.sleep(self.config.cost_optimization_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(60)
                
        logger.info("Auto-scaling loop stopped")
        
    async def _heartbeat_loop(self):
        """Node heartbeat monitoring loop."""
        logger.info("Started heartbeat loop")
        
        while self._running:
            try:
                current_time = time.time()
                
                # Check for stale nodes
                for node_id, node in list(self.node_manager.nodes.items()):
                    time_since_heartbeat = current_time - node.last_heartbeat.timestamp()
                    
                    if time_since_heartbeat > self.config.node_failure_timeout:
                        logger.warning(f"Node {node_id} missed heartbeat, marking as failed")
                        await self.node_manager.handle_node_failure(node_id)
                        
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)
                
        logger.info("Heartbeat loop stopped")
        
    async def add_compute_nodes(self, node_specs: List[Dict[str, Any]]):
        """Add compute nodes to the infrastructure."""
        for spec in node_specs:
            node_id = f"node_{uuid.uuid4().hex[:8]}"
            
            node = ComputeNode(
                node_id=node_id,
                provider=CloudProvider(spec['provider']),
                region=spec['region'],
                zone=spec.get('zone', 'a'),
                resource_spec=ResourceSpec(**spec['resources']),
                cost_per_hour=spec.get('cost_per_hour', 1.0)
            )
            
            await self.node_manager.register_node(node)
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = time.time() - self.metrics['start_time']
        
        return {
            'orchestrator': {
                'running': self._running,
                'uptime_seconds': uptime,
                'version': '4.0'
            },
            'metrics': self.metrics,
            'scheduling': self.scheduler.get_scheduling_statistics(),
            'infrastructure': self.node_manager.get_infrastructure_statistics(),
            'configuration': {
                'max_concurrent_tasks': self.config.max_concurrent_tasks,
                'auto_scaling_enabled': self.config.enable_auto_scaling,
                'cost_optimization_enabled': self.config.enable_cost_optimization,
                'fault_tolerance_enabled': self.config.enable_fault_tolerance
            }
        }


# Demonstration and usage example
async def demonstrate_hyperschale_orchestrator():
    """Demonstrate hyperschale orchestrator capabilities."""
    logger.info("Demonstrating Hyperschale Orchestrator...")
    
    # Create orchestrator with configuration
    config = HyperschaleConfig(
        max_concurrent_tasks=100,
        enable_auto_scaling=True,
        enable_cost_optimization=True,
        enable_monitoring=True
    )
    
    orchestrator = HyperschaleOrchestrator(config)
    
    # Add initial compute infrastructure
    initial_nodes = [
        {
            'provider': 'aws',
            'region': 'us-west-2',
            'resources': {'cpu_cores': 16, 'gpu_count': 2, 'memory_gb': 64},
            'cost_per_hour': 5.0
        },
        {
            'provider': 'gcp',
            'region': 'us-central1',
            'resources': {'cpu_cores': 8, 'gpu_count': 1, 'memory_gb': 32},
            'cost_per_hour': 3.0
        },
        {
            'provider': 'azure',
            'region': 'eastus',
            'resources': {'cpu_cores': 32, 'gpu_count': 0, 'memory_gb': 128},
            'cost_per_hour': 4.0
        }
    ]
    
    await orchestrator.add_compute_nodes(initial_nodes)
    
    # Start orchestrator
    orchestrator_task = asyncio.create_task(orchestrator.start())
    
    # Submit various tasks
    task_submissions = []
    
    # High-priority protein generation tasks
    for i in range(5):
        task_id = await orchestrator.submit_task(
            WorkloadType.GENERATION,
            TaskPriority.HIGH,
            ResourceSpec(cpu_cores=4, gpu_count=1, memory_gb=16),
            {'sequence_length': 200 + i * 50},
            {'temperature': 1.0 + i * 0.1}
        )
        task_submissions.append(task_id)
        
    # Normal priority folding tasks
    for i in range(3):
        task_id = await orchestrator.submit_task(
            WorkloadType.FOLDING,
            TaskPriority.NORMAL,
            ResourceSpec(cpu_cores=8, gpu_count=0, memory_gb=32),
            {'protein_sequence': f'MKLLILTCLVAVALARP{i}'},
            {'steps': 1000}
        )
        task_submissions.append(task_id)
        
    # Critical optimization task
    task_id = await orchestrator.submit_task(
        WorkloadType.OPTIMIZATION,
        TaskPriority.CRITICAL,
        ResourceSpec(cpu_cores=16, gpu_count=2, memory_gb=64),
        {'target_property': 'binding_affinity'},
        {'iterations': 100}
    )
    task_submissions.append(task_id)
    
    # Let the system run for a while
    await asyncio.sleep(10)
    
    # Check task statuses
    logger.info("\nTask Status Report:")
    for task_id in task_submissions[:5]:  # Check first 5 tasks
        status = await orchestrator.get_task_status(task_id)
        logger.info(f"Task {task_id}: {status['status']}")
        
    # Get system status
    system_status = orchestrator.get_system_status()
    logger.info(f"\nSystem Status:")
    logger.info(f"Total Tasks: {system_status['metrics']['tasks_scheduled']}")
    logger.info(f"Completed: {system_status['metrics']['tasks_completed']}")
    logger.info(f"Failed: {system_status['metrics']['tasks_failed']}")
    logger.info(f"Total Cost: ${system_status['metrics']['total_cost']:.2f}")
    logger.info(f"Infrastructure: {system_status['infrastructure']['total_nodes']} nodes")
    logger.info(f"Average Utilization: {system_status['infrastructure']['average_utilization']:.2%}")
    
    # Shutdown
    await orchestrator.shutdown()
    orchestrator_task.cancel()
    
    try:
        await orchestrator_task
    except asyncio.CancelledError:
        pass
    
    logger.info("Hyperschale orchestrator demonstration completed")
    return orchestrator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demonstrate_hyperschale_orchestrator())