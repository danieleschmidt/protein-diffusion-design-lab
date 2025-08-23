"""
Global Scale Orchestrator for Protein Diffusion Design Lab.

This module implements global-scale orchestration including:
- Distributed system coordination
- Auto-scaling across multiple regions
- Global load balancing and failover
- Multi-cloud deployment management
- Edge computing integration
- Real-time global monitoring
"""

import asyncio
import time
import logging
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import random
import statistics
import hashlib
from enum import Enum
import uuid
import math

logger = logging.getLogger(__name__)

class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"

class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    EDGE = "edge"

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"

class ServiceTier(Enum):
    """Service tier definitions."""
    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"

@dataclass
class RegionConfig:
    """Configuration for deployment region."""
    region: DeploymentRegion
    cloud_provider: CloudProvider
    capacity: int = 100
    latency_target: float = 0.1  # 100ms
    availability_target: float = 0.999  # 99.9%
    cost_per_hour: float = 10.0
    regulatory_compliance: List[str] = field(default_factory=list)
    edge_locations: List[str] = field(default_factory=list)

@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int = 1
    max_instances: int = 100
    target_cpu_utilization: float = 0.7
    target_memory_utilization: float = 0.8
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 600  # 10 minutes
    strategy: ScalingStrategy = ScalingStrategy.HYBRID

@dataclass
class GlobalMetrics:
    """Global system metrics."""
    timestamp: float = field(default_factory=time.time)
    total_requests: int = 0
    requests_per_second: float = 0.0
    global_latency_p50: float = 0.0
    global_latency_p95: float = 0.0
    global_latency_p99: float = 0.0
    error_rate: float = 0.0
    availability: float = 1.0
    active_regions: int = 0
    total_instances: int = 0
    total_cost: float = 0.0

class RegionManager:
    """Manages resources and services in a specific region."""
    
    def __init__(self, config: RegionConfig):
        self.config = config
        self.instances = {}
        self.instance_metrics = defaultdict(lambda: deque(maxlen=100))
        self.region_metrics = GlobalMetrics()
        self.scaling_policy = ScalingPolicy()
        self.failover_targets = []
        self.is_healthy = True
        self.last_health_check = time.time()
        self._lock = threading.RLock()
        
        logger.info(f"Initialized region manager for {config.region.value}")
    
    def deploy_instance(self, instance_spec: Dict[str, Any]) -> str:
        """Deploy new instance in this region."""
        instance_id = f"{self.config.region.value}-{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            instance = {
                'id': instance_id,
                'spec': instance_spec,
                'status': 'deploying',
                'created_at': time.time(),
                'last_health_check': time.time(),
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'requests_handled': 0,
                'errors': 0,
                'healthy': True
            }
            
            self.instances[instance_id] = instance
            
            # Simulate deployment time
            deployment_time = random.uniform(30, 120)  # 30-120 seconds
            threading.Timer(deployment_time, self._complete_deployment, args=[instance_id]).start()
            
            logger.info(f"Deploying instance {instance_id} in {self.config.region.value}")
            return instance_id
    
    def _complete_deployment(self, instance_id: str):
        """Complete instance deployment."""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id]['status'] = 'running'
                logger.info(f"Instance {instance_id} deployed successfully")
    
    def terminate_instance(self, instance_id: str):
        """Terminate instance."""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id]['status'] = 'terminating'
                
                # Graceful shutdown delay
                def complete_termination():
                    with self._lock:
                        if instance_id in self.instances:
                            del self.instances[instance_id]
                            logger.info(f"Instance {instance_id} terminated")
                
                threading.Timer(30, complete_termination).start()
    
    def scale_instances(self, target_count: int) -> List[str]:
        """Scale instances to target count."""
        with self._lock:
            current_running = len([i for i in self.instances.values() if i['status'] == 'running'])
            
            if target_count > current_running:
                # Scale up
                instances_to_add = target_count - current_running
                new_instances = []
                
                for _ in range(instances_to_add):
                    instance_spec = {
                        'cpu': 2,
                        'memory': 4096,
                        'service_tier': ServiceTier.STANDARD.value
                    }
                    instance_id = self.deploy_instance(instance_spec)
                    new_instances.append(instance_id)
                
                logger.info(f"Scaling up {instances_to_add} instances in {self.config.region.value}")
                return new_instances
                
            elif target_count < current_running:
                # Scale down
                instances_to_remove = current_running - target_count
                running_instances = [
                    instance_id for instance_id, instance in self.instances.items()
                    if instance['status'] == 'running'
                ]
                
                # Select instances with lowest utilization for termination
                instances_by_utilization = sorted(
                    running_instances,
                    key=lambda iid: (
                        self.instances[iid]['cpu_usage'] + 
                        self.instances[iid]['memory_usage']
                    ) / 2
                )
                
                instances_to_terminate = instances_by_utilization[:instances_to_remove]
                
                for instance_id in instances_to_terminate:
                    self.terminate_instance(instance_id)
                
                logger.info(f"Scaling down {instances_to_remove} instances in {self.config.region.value}")
                return instances_to_terminate
            
            return []
    
    def route_request(self, request_id: str, request_context: Dict[str, Any] = None) -> Optional[str]:
        """Route request to best available instance."""
        with self._lock:
            available_instances = [
                instance_id for instance_id, instance in self.instances.items()
                if instance['status'] == 'running' and instance['healthy']
            ]
            
            if not available_instances:
                return None
            
            # Select instance with lowest load
            best_instance = min(
                available_instances,
                key=lambda iid: (
                    self.instances[iid]['cpu_usage'] + 
                    self.instances[iid]['memory_usage']
                ) / 2
            )
            
            # Update instance metrics
            self.instances[best_instance]['requests_handled'] += 1
            
            return best_instance
    
    def update_instance_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """Update metrics for specific instance."""
        if instance_id not in self.instances:
            return
        
        with self._lock:
            instance = self.instances[instance_id]
            current_time = time.time()
            
            # Update instance metrics
            instance['cpu_usage'] = metrics.get('cpu_usage', 0.0)
            instance['memory_usage'] = metrics.get('memory_usage', 0.0)
            instance['last_health_check'] = current_time
            
            # Record in history
            self.instance_metrics[instance_id].append({
                'timestamp': current_time,
                'cpu_usage': instance['cpu_usage'],
                'memory_usage': instance['memory_usage'],
                'latency': metrics.get('latency', 0.0),
                'error_rate': metrics.get('error_rate', 0.0)
            })
    
    def check_scaling_needs(self) -> Dict[str, Any]:
        """Check if scaling is needed based on current metrics."""
        with self._lock:
            running_instances = [i for i in self.instances.values() if i['status'] == 'running']
            
            if not running_instances:
                return {'action': 'scale_up', 'target_count': self.scaling_policy.min_instances}
            
            # Calculate average utilization
            total_cpu = sum(i['cpu_usage'] for i in running_instances)
            total_memory = sum(i['memory_usage'] for i in running_instances)
            avg_cpu = total_cpu / len(running_instances)
            avg_memory = total_memory / len(running_instances)
            avg_utilization = (avg_cpu + avg_memory) / 2
            
            current_count = len(running_instances)
            
            # Check scaling conditions
            if avg_utilization > self.scaling_policy.scale_up_threshold:
                target_count = min(
                    current_count * 2,  # Double capacity
                    self.scaling_policy.max_instances
                )
                return {
                    'action': 'scale_up',
                    'target_count': target_count,
                    'reason': f'High utilization: {avg_utilization:.2f}'
                }
            
            elif avg_utilization < self.scaling_policy.scale_down_threshold:
                target_count = max(
                    current_count // 2,  # Half capacity
                    self.scaling_policy.min_instances
                )
                return {
                    'action': 'scale_down',
                    'target_count': target_count,
                    'reason': f'Low utilization: {avg_utilization:.2f}'
                }
            
            return {'action': 'no_change', 'target_count': current_count}
    
    def get_region_status(self) -> Dict[str, Any]:
        """Get comprehensive region status."""
        with self._lock:
            running_instances = [i for i in self.instances.values() if i['status'] == 'running']
            healthy_instances = [i for i in running_instances if i['healthy']]
            
            # Calculate metrics
            total_cpu = sum(i['cpu_usage'] for i in running_instances) if running_instances else 0
            total_memory = sum(i['memory_usage'] for i in running_instances) if running_instances else 0
            avg_cpu = total_cpu / len(running_instances) if running_instances else 0
            avg_memory = total_memory / len(running_instances) if running_instances else 0
            
            # Calculate recent request rate
            recent_requests = sum(i['requests_handled'] for i in running_instances)
            
            return {
                'region': self.config.region.value,
                'cloud_provider': self.config.cloud_provider.value,
                'healthy': self.is_healthy,
                'instances': {
                    'total': len(self.instances),
                    'running': len(running_instances),
                    'healthy': len(healthy_instances),
                    'deploying': len([i for i in self.instances.values() if i['status'] == 'deploying']),
                    'terminating': len([i for i in self.instances.values() if i['status'] == 'terminating'])
                },
                'utilization': {
                    'cpu': avg_cpu,
                    'memory': avg_memory,
                    'average': (avg_cpu + avg_memory) / 2
                },
                'capacity': self.config.capacity,
                'recent_requests': recent_requests,
                'cost_estimate': len(running_instances) * self.config.cost_per_hour,
                'scaling_policy': asdict(self.scaling_policy)
            }

class GlobalLoadBalancer:
    """Global load balancer with intelligent routing."""
    
    def __init__(self):
        self.regions = {}
        self.routing_table = {}
        self.health_checks = {}
        self.routing_history = deque(maxlen=10000)
        self.failover_rules = []
        self._lock = threading.RLock()
    
    def register_region(self, region_manager: RegionManager):
        """Register region for global load balancing."""
        region_id = region_manager.config.region.value
        
        with self._lock:
            self.regions[region_id] = region_manager
            self.routing_table[region_id] = {
                'weight': 100,
                'healthy': True,
                'latency': 0.0,
                'error_rate': 0.0,
                'capacity_utilization': 0.0
            }
            
            logger.info(f"Registered region {region_id} for global load balancing")
    
    def route_request(self, client_location: str = None, 
                     request_context: Dict[str, Any] = None) -> Optional[Tuple[str, str]]:
        """Route request to optimal region and instance."""
        with self._lock:
            healthy_regions = [
                region_id for region_id, info in self.routing_table.items()
                if info['healthy'] and region_id in self.regions
            ]
            
            if not healthy_regions:
                logger.error("No healthy regions available for routing")
                return None
            
            # Select optimal region
            optimal_region = self._select_optimal_region(healthy_regions, client_location)
            
            if not optimal_region:
                return None
            
            # Route to instance in selected region
            region_manager = self.regions[optimal_region]
            instance_id = region_manager.route_request(
                f"req_{uuid.uuid4().hex[:8]}", request_context
            )
            
            if instance_id:
                # Record routing decision
                routing_record = {
                    'timestamp': time.time(),
                    'client_location': client_location,
                    'selected_region': optimal_region,
                    'selected_instance': instance_id,
                    'available_regions': healthy_regions.copy(),
                    'routing_factors': self._get_routing_factors(optimal_region)
                }
                
                self.routing_history.append(routing_record)
                
                return optimal_region, instance_id
            
            return None
    
    def _select_optimal_region(self, healthy_regions: List[str], 
                             client_location: str = None) -> Optional[str]:
        """Select optimal region for request routing."""
        region_scores = {}
        
        for region_id in healthy_regions:
            region_info = self.routing_table[region_id]
            region_manager = self.regions[region_id]
            
            # Base score components
            latency_score = 1.0 - min(region_info['latency'] / 1.0, 1.0)  # Normalize to 1s max
            error_score = 1.0 - region_info['error_rate']
            capacity_score = 1.0 - region_info['capacity_utilization']
            
            # Geographic proximity score
            proximity_score = self._calculate_proximity_score(region_id, client_location)
            
            # Cost efficiency score
            cost_score = self._calculate_cost_efficiency(region_id)
            
            # Composite score
            composite_score = (
                latency_score * 0.25 +
                error_score * 0.2 +
                capacity_score * 0.25 +
                proximity_score * 0.2 +
                cost_score * 0.1
            )
            
            region_scores[region_id] = composite_score
        
        # Select region with highest score
        if region_scores:
            optimal_region = max(region_scores.keys(), key=lambda k: region_scores[k])
            return optimal_region
        
        return None
    
    def _calculate_proximity_score(self, region_id: str, client_location: str = None) -> float:
        """Calculate geographic proximity score."""
        if not client_location:
            return 0.5  # Neutral score if no client location
        
        # Simplified proximity calculation based on region mappings
        region_proximity = {
            ('us-east-1', 'north_america'): 1.0,
            ('us-west-2', 'north_america'): 0.9,
            ('eu-west-1', 'europe'): 1.0,
            ('eu-central-1', 'europe'): 0.9,
            ('ap-southeast-1', 'asia'): 1.0,
            ('ap-northeast-1', 'asia'): 0.9,
            ('ca-central-1', 'north_america'): 0.8,
            ('ap-southeast-2', 'oceania'): 1.0,
        }
        
        return region_proximity.get((region_id, client_location), 0.3)
    
    def _calculate_cost_efficiency(self, region_id: str) -> float:
        """Calculate cost efficiency score for region."""
        region_manager = self.regions[region_id]
        region_status = region_manager.get_region_status()
        
        # Simple cost efficiency based on utilization and cost
        utilization = region_status['utilization']['average']
        cost_per_hour = region_manager.config.cost_per_hour
        
        # Higher utilization and lower cost = higher efficiency
        efficiency = (utilization * 0.7) + ((20.0 - cost_per_hour) / 20.0 * 0.3)
        
        return max(0.0, min(1.0, efficiency))
    
    def _get_routing_factors(self, region_id: str) -> Dict[str, float]:
        """Get routing factors used for region selection."""
        region_info = self.routing_table[region_id]
        
        return {
            'latency': region_info['latency'],
            'error_rate': region_info['error_rate'],
            'capacity_utilization': region_info['capacity_utilization'],
            'health_status': 1.0 if region_info['healthy'] else 0.0
        }
    
    def update_region_metrics(self, region_id: str, metrics: Dict[str, Any]):
        """Update metrics for region routing decisions."""
        if region_id not in self.routing_table:
            return
        
        with self._lock:
            region_info = self.routing_table[region_id]
            
            # Update routing metrics with exponential moving average
            alpha = 0.3  # Learning rate
            
            if 'latency' in metrics:
                region_info['latency'] = (
                    alpha * metrics['latency'] + (1 - alpha) * region_info['latency']
                )
            
            if 'error_rate' in metrics:
                region_info['error_rate'] = (
                    alpha * metrics['error_rate'] + (1 - alpha) * region_info['error_rate']
                )
            
            if 'capacity_utilization' in metrics:
                region_info['capacity_utilization'] = (
                    alpha * metrics['capacity_utilization'] + 
                    (1 - alpha) * region_info['capacity_utilization']
                )
    
    def add_failover_rule(self, primary_region: str, failover_regions: List[str],
                         conditions: Dict[str, Any]):
        """Add failover rule for region disaster recovery."""
        rule = {
            'id': str(uuid.uuid4()),
            'primary_region': primary_region,
            'failover_regions': failover_regions,
            'conditions': conditions,
            'active': True,
            'created_at': time.time()
        }
        
        self.failover_rules.append(rule)
        logger.info(f"Added failover rule: {primary_region} -> {failover_regions}")
    
    def check_failover_conditions(self) -> List[Dict[str, Any]]:
        """Check if any failover conditions are met."""
        triggered_failovers = []
        
        for rule in self.failover_rules:
            if not rule['active']:
                continue
            
            primary_region = rule['primary_region']
            conditions = rule['conditions']
            
            if primary_region not in self.routing_table:
                continue
            
            region_info = self.routing_table[primary_region]
            
            # Check failover conditions
            failover_needed = False
            reasons = []
            
            if 'max_error_rate' in conditions:
                if region_info['error_rate'] > conditions['max_error_rate']:
                    failover_needed = True
                    reasons.append(f"High error rate: {region_info['error_rate']:.3f}")
            
            if 'max_latency' in conditions:
                if region_info['latency'] > conditions['max_latency']:
                    failover_needed = True
                    reasons.append(f"High latency: {region_info['latency']:.3f}s")
            
            if 'min_health' in conditions:
                health_score = 1.0 if region_info['healthy'] else 0.0
                if health_score < conditions['min_health']:
                    failover_needed = True
                    reasons.append(f"Poor health: {health_score}")
            
            if failover_needed:
                triggered_failovers.append({
                    'rule_id': rule['id'],
                    'primary_region': primary_region,
                    'failover_regions': rule['failover_regions'],
                    'reasons': reasons,
                    'timestamp': time.time()
                })
        
        return triggered_failovers
    
    def get_global_routing_status(self) -> Dict[str, Any]:
        """Get global routing status."""
        with self._lock:
            healthy_regions = [
                region_id for region_id, info in self.routing_table.items()
                if info['healthy']
            ]
            
            # Calculate routing statistics
            recent_routes = [
                r for r in self.routing_history
                if time.time() - r['timestamp'] <= 300  # Last 5 minutes
            ]
            
            region_distribution = defaultdict(int)
            for route in recent_routes:
                region_distribution[route['selected_region']] += 1
            
            return {
                'total_regions': len(self.regions),
                'healthy_regions': len(healthy_regions),
                'routing_table': dict(self.routing_table),
                'recent_requests': len(recent_routes),
                'region_distribution': dict(region_distribution),
                'active_failover_rules': len([r for r in self.failover_rules if r['active']]),
                'average_routing_latency': self._calculate_average_routing_latency()
            }
    
    def _calculate_average_routing_latency(self) -> float:
        """Calculate average routing decision latency."""
        # This would measure the time taken to make routing decisions
        return 0.001  # 1ms average (placeholder)

class GlobalAutoScaler:
    """Global auto-scaling orchestrator."""
    
    def __init__(self):
        self.region_managers = {}
        self.scaling_decisions = []
        self.scaling_policies = {}
        self.predictive_models = {}
        self.scaling_active = False
        self._lock = threading.RLock()
        self.scaling_thread = None
    
    def register_region(self, region_manager: RegionManager, policy: ScalingPolicy = None):
        """Register region for auto-scaling."""
        region_id = region_manager.config.region.value
        
        with self._lock:
            self.region_managers[region_id] = region_manager
            self.scaling_policies[region_id] = policy or ScalingPolicy()
            
            # Initialize predictive model
            self.predictive_models[region_id] = {
                'demand_history': deque(maxlen=288),  # 24 hours of 5-minute intervals
                'predictions': {},
                'accuracy': 0.0
            }
            
            logger.info(f"Registered region {region_id} for auto-scaling")
    
    def start_scaling(self):
        """Start auto-scaling orchestrator."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Global auto-scaling started")
    
    def stop_scaling(self):
        """Stop auto-scaling orchestrator."""
        self.scaling_active = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        
        logger.info("Global auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main scaling orchestration loop."""
        while self.scaling_active:
            try:
                self._perform_scaling_analysis()
                time.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)
    
    def _perform_scaling_analysis(self):
        """Perform global scaling analysis and decisions."""
        with self._lock:
            current_time = time.time()
            scaling_decisions_batch = []
            
            for region_id, region_manager in self.region_managers.items():
                policy = self.scaling_policies[region_id]
                
                # Get current region status
                region_status = region_manager.get_region_status()
                
                # Update demand history
                self._update_demand_history(region_id, region_status)
                
                # Check if scaling is needed
                scaling_need = region_manager.check_scaling_needs()
                
                if scaling_need['action'] != 'no_change':
                    # Consider predictive scaling
                    predicted_demand = self._predict_demand(region_id)
                    
                    # Make scaling decision
                    decision = self._make_scaling_decision(
                        region_id, scaling_need, predicted_demand, policy
                    )
                    
                    if decision:
                        scaling_decisions_batch.append(decision)
            
            # Execute scaling decisions
            if scaling_decisions_batch:
                self._execute_scaling_decisions(scaling_decisions_batch)
    
    def _update_demand_history(self, region_id: str, region_status: Dict[str, Any]):
        """Update demand history for predictive scaling."""
        model = self.predictive_models[region_id]
        
        demand_point = {
            'timestamp': time.time(),
            'requests': region_status['recent_requests'],
            'utilization': region_status['utilization']['average'],
            'instances': region_status['instances']['running']
        }
        
        model['demand_history'].append(demand_point)
    
    def _predict_demand(self, region_id: str) -> Dict[str, Any]:
        """Predict future demand for region."""
        model = self.predictive_models[region_id]
        history = model['demand_history']
        
        if len(history) < 10:
            return {'predicted_utilization': 0.5, 'confidence': 0.0}
        
        # Simple trend-based prediction
        recent_utilizations = [point['utilization'] for point in list(history)[-20:]]
        current_utilization = recent_utilizations[-1]
        
        # Calculate trend
        trend = self._calculate_trend(recent_utilizations)
        
        # Predict utilization 10 minutes ahead
        predicted_utilization = current_utilization + (trend * 600)  # 10 minutes in seconds
        predicted_utilization = max(0.0, min(1.0, predicted_utilization))
        
        # Simple confidence based on trend consistency
        recent_trends = []
        for i in range(5, len(recent_utilizations)):
            segment = recent_utilizations[i-5:i]
            segment_trend = self._calculate_trend(segment)
            recent_trends.append(segment_trend)
        
        if recent_trends:
            trend_variance = statistics.variance(recent_trends) if len(recent_trends) > 1 else 0
            confidence = max(0.0, 1.0 - (trend_variance * 10))  # Normalize variance
        else:
            confidence = 0.5
        
        return {
            'predicted_utilization': predicted_utilization,
            'trend': trend,
            'confidence': confidence
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_avg = (n - 1) / 2
        y_avg = statistics.mean(values)
        
        numerator = sum((i - x_avg) * (values[i] - y_avg) for i in range(n))
        denominator = sum((i - x_avg) ** 2 for i in range(n))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _make_scaling_decision(self, region_id: str, scaling_need: Dict[str, Any],
                             predicted_demand: Dict[str, Any], policy: ScalingPolicy) -> Optional[Dict[str, Any]]:
        """Make intelligent scaling decision."""
        current_time = time.time()
        
        # Check cooldown periods
        recent_decisions = [
            d for d in self.scaling_decisions
            if (d['region_id'] == region_id and 
                current_time - d['timestamp'] < policy.scale_up_cooldown)
        ]
        
        if recent_decisions:
            logger.debug(f"Scaling cooldown active for {region_id}")
            return None
        
        action = scaling_need['action']
        current_target = scaling_need['target_count']
        
        # Adjust target based on predictions
        if policy.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            predicted_util = predicted_demand['predicted_utilization']
            confidence = predicted_demand['confidence']
            
            if confidence > 0.7:  # High confidence in prediction
                if predicted_util > policy.scale_up_threshold:
                    # Preemptive scale up
                    current_target = min(current_target * 1.5, policy.max_instances)
                    action = 'scale_up'
                elif predicted_util < policy.scale_down_threshold:
                    # Preemptive scale down
                    current_target = max(current_target * 0.7, policy.min_instances)
                    action = 'scale_down'
        
        if action == 'no_change':
            return None
        
        decision = {
            'region_id': region_id,
            'action': action,
            'current_instances': scaling_need.get('current_count', 0),
            'target_instances': int(current_target),
            'reason': scaling_need.get('reason', 'Predictive scaling'),
            'predicted_demand': predicted_demand,
            'policy_used': policy.strategy.value,
            'timestamp': current_time,
            'executed': False
        }
        
        return decision
    
    def _execute_scaling_decisions(self, decisions: List[Dict[str, Any]]):
        """Execute scaling decisions across regions."""
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            for decision in decisions:
                future = executor.submit(self._execute_single_scaling_decision, decision)
                futures.append((future, decision))
            
            for future, decision in futures:
                try:
                    success = future.result(timeout=30)
                    decision['executed'] = success
                    decision['execution_time'] = time.time()
                    
                    self.scaling_decisions.append(decision)
                    
                    if success:
                        logger.info(f"Scaling executed: {decision['region_id']} "
                                  f"{decision['action']} to {decision['target_instances']}")
                    else:
                        logger.warning(f"Scaling failed: {decision['region_id']} {decision['action']}")
                        
                except Exception as e:
                    logger.error(f"Scaling execution error: {e}")
                    decision['executed'] = False
                    decision['error'] = str(e)
                    self.scaling_decisions.append(decision)
        
        # Keep only recent decisions
        self.scaling_decisions = self.scaling_decisions[-1000:]
    
    def _execute_single_scaling_decision(self, decision: Dict[str, Any]) -> bool:
        """Execute single scaling decision."""
        region_id = decision['region_id']
        target_instances = decision['target_instances']
        
        if region_id not in self.region_managers:
            return False
        
        region_manager = self.region_managers[region_id]
        
        try:
            region_manager.scale_instances(target_instances)
            return True
        except Exception as e:
            logger.error(f"Failed to scale {region_id}: {e}")
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get global scaling status."""
        with self._lock:
            recent_decisions = [
                d for d in self.scaling_decisions
                if time.time() - d['timestamp'] <= 3600  # Last hour
            ]
            
            successful_decisions = [d for d in recent_decisions if d['executed']]
            
            # Calculate scaling statistics
            scale_up_count = len([d for d in recent_decisions if d['action'] == 'scale_up'])
            scale_down_count = len([d for d in recent_decisions if d['action'] == 'scale_down'])
            
            return {
                'scaling_active': self.scaling_active,
                'registered_regions': len(self.region_managers),
                'recent_scaling_decisions': len(recent_decisions),
                'scaling_success_rate': (
                    len(successful_decisions) / max(1, len(recent_decisions))
                ),
                'scale_up_decisions': scale_up_count,
                'scale_down_decisions': scale_down_count,
                'predictive_accuracy': self._calculate_predictive_accuracy(),
                'region_policies': {
                    region_id: policy.strategy.value 
                    for region_id, policy in self.scaling_policies.items()
                }
            }
    
    def _calculate_predictive_accuracy(self) -> float:
        """Calculate accuracy of predictive scaling models."""
        total_accuracy = 0.0
        model_count = 0
        
        for region_id, model in self.predictive_models.items():
            if model['accuracy'] > 0:
                total_accuracy += model['accuracy']
                model_count += 1
        
        return total_accuracy / max(1, model_count)

class GlobalScaleOrchestrator:
    """Main global scale orchestrator."""
    
    def __init__(self):
        self.region_managers = {}
        self.load_balancer = GlobalLoadBalancer()
        self.auto_scaler = GlobalAutoScaler()
        self.global_metrics = GlobalMetrics()
        self.monitoring_active = False
        self.monitoring_thread = None
        self._lock = threading.RLock()
        
        logger.info("Global scale orchestrator initialized")
    
    def add_region(self, config: RegionConfig, scaling_policy: ScalingPolicy = None) -> RegionManager:
        """Add new deployment region."""
        region_manager = RegionManager(config)
        region_id = config.region.value
        
        with self._lock:
            self.region_managers[region_id] = region_manager
            self.load_balancer.register_region(region_manager)
            self.auto_scaler.register_region(region_manager, scaling_policy)
            
            logger.info(f"Added region: {region_id}")
            return region_manager
    
    def start_orchestration(self):
        """Start global orchestration."""
        self.auto_scaler.start_scaling()
        
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
        
        logger.info("Global orchestration started")
    
    def stop_orchestration(self):
        """Stop global orchestration."""
        self.auto_scaler.stop_scaling()
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Global orchestration stopped")
    
    def _monitoring_loop(self):
        """Global monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_global_metrics()
                self._check_global_health()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _update_global_metrics(self):
        """Update global system metrics."""
        with self._lock:
            current_time = time.time()
            
            # Aggregate metrics from all regions
            total_requests = 0
            total_instances = 0
            total_cost = 0.0
            active_regions = 0
            latencies = []
            error_rates = []
            
            for region_id, region_manager in self.region_managers.items():
                region_status = region_manager.get_region_status()
                
                if region_status['healthy']:
                    active_regions += 1
                    total_requests += region_status['recent_requests']
                    total_instances += region_status['instances']['running']
                    total_cost += region_status['cost_estimate']
                    
                    # Collect latency and error rate data (simulated)
                    latencies.extend([0.1, 0.15, 0.12])  # Placeholder latencies
                    error_rates.append(0.001)  # Placeholder error rate
            
            # Calculate global metrics
            self.global_metrics.timestamp = current_time
            self.global_metrics.total_requests = total_requests
            self.global_metrics.requests_per_second = total_requests / 300.0  # 5-minute window
            self.global_metrics.active_regions = active_regions
            self.global_metrics.total_instances = total_instances
            self.global_metrics.total_cost = total_cost
            
            if latencies:
                self.global_metrics.global_latency_p50 = statistics.median(latencies)
                self.global_metrics.global_latency_p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
                self.global_metrics.global_latency_p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            
            if error_rates:
                self.global_metrics.error_rate = statistics.mean(error_rates)
            
            # Calculate availability
            if active_regions > 0:
                self.global_metrics.availability = min(1.0, active_regions / len(self.region_managers))
            else:
                self.global_metrics.availability = 0.0
    
    def _check_global_health(self):
        """Check global system health and trigger actions."""
        # Check for failover conditions
        failovers = self.load_balancer.check_failover_conditions()
        
        for failover in failovers:
            logger.warning(f"Failover triggered: {failover['primary_region']} -> "
                         f"{failover['failover_regions']} ({failover['reasons']})")
            
            # In a real system, this would trigger actual failover procedures
    
    def route_global_request(self, client_location: str = None, 
                           request_context: Dict[str, Any] = None) -> Optional[Tuple[str, str]]:
        """Route request globally."""
        return self.load_balancer.route_request(client_location, request_context)
    
    def deploy_global_service(self, service_spec: Dict[str, Any], 
                            regions: List[DeploymentRegion] = None) -> Dict[str, List[str]]:
        """Deploy service globally across regions."""
        if regions is None:
            regions = list(self.region_managers.keys())
        
        deployment_results = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for region_id in regions:
                if region_id in self.region_managers:
                    future = executor.submit(
                        self.region_managers[region_id].deploy_instance, 
                        service_spec
                    )
                    futures.append((future, region_id))
            
            for future, region_id in futures:
                try:
                    instance_id = future.result(timeout=300)  # 5 minute timeout
                    if region_id not in deployment_results:
                        deployment_results[region_id] = []
                    deployment_results[region_id].append(instance_id)
                    
                except Exception as e:
                    logger.error(f"Deployment failed in {region_id}: {e}")
                    deployment_results[region_id] = []
        
        logger.info(f"Global deployment completed: {len(deployment_results)} regions")
        return deployment_results
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global status."""
        with self._lock:
            region_statuses = {}
            for region_id, region_manager in self.region_managers.items():
                region_statuses[region_id] = region_manager.get_region_status()
            
            load_balancer_status = self.load_balancer.get_global_routing_status()
            scaling_status = self.auto_scaler.get_scaling_status()
            
            return {
                'timestamp': time.time(),
                'global_metrics': asdict(self.global_metrics),
                'regions': region_statuses,
                'load_balancing': load_balancer_status,
                'auto_scaling': scaling_status,
                'orchestration_active': self.monitoring_active,
                'global_health_grade': self._calculate_global_health_grade(),
                'cost_efficiency': self._calculate_cost_efficiency()
            }
    
    def _calculate_global_health_grade(self) -> str:
        """Calculate global system health grade."""
        availability = self.global_metrics.availability
        error_rate = self.global_metrics.error_rate
        latency_p95 = self.global_metrics.global_latency_p95
        
        # Calculate composite health score
        availability_score = availability
        error_score = 1.0 - min(error_rate * 100, 1.0)  # Assume error rate is small
        latency_score = 1.0 - min(latency_p95 / 1.0, 1.0)  # Normalize to 1s
        
        health_score = (availability_score * 0.4 + error_score * 0.3 + latency_score * 0.3)
        
        if health_score >= 0.95:
            return "A+ (Excellent)"
        elif health_score >= 0.9:
            return "A (Very Good)"
        elif health_score >= 0.8:
            return "B (Good)"
        elif health_score >= 0.7:
            return "C (Fair)"
        else:
            return "D (Poor)"
    
    def _calculate_cost_efficiency(self) -> Dict[str, Any]:
        """Calculate global cost efficiency metrics."""
        total_cost = self.global_metrics.total_cost
        total_requests = self.global_metrics.total_requests
        
        cost_per_request = total_cost / max(1, total_requests) if total_requests > 0 else 0
        cost_per_hour = total_cost
        
        # Calculate efficiency based on utilization and cost
        total_capacity = sum(
            region_status['capacity'] for region_status in [
                rm.get_region_status() for rm in self.region_managers.values()
            ]
        )
        
        utilization = self.global_metrics.total_instances / max(1, total_capacity)
        
        return {
            'cost_per_request': cost_per_request,
            'cost_per_hour': cost_per_hour,
            'utilization_efficiency': utilization,
            'cost_optimization_score': min(1.0, utilization * 1.5)  # Higher utilization = better efficiency
        }

# Convenience function for creating global orchestrator
def create_global_orchestrator(regions: List[Dict[str, Any]] = None) -> GlobalScaleOrchestrator:
    """Create global scale orchestrator with default regions."""
    orchestrator = GlobalScaleOrchestrator()
    
    if regions is None:
        # Default regions
        default_regions = [
            {
                'region': DeploymentRegion.US_EAST,
                'cloud_provider': CloudProvider.AWS,
                'capacity': 100,
                'cost_per_hour': 10.0
            },
            {
                'region': DeploymentRegion.EU_WEST,
                'cloud_provider': CloudProvider.AWS,
                'capacity': 80,
                'cost_per_hour': 12.0
            },
            {
                'region': DeploymentRegion.ASIA_PACIFIC,
                'cloud_provider': CloudProvider.AWS,
                'capacity': 60,
                'cost_per_hour': 15.0
            }
        ]
        regions = default_regions
    
    # Add regions to orchestrator
    for region_config in regions:
        config = RegionConfig(**region_config)
        scaling_policy = ScalingPolicy(
            min_instances=2,
            max_instances=50,
            strategy=ScalingStrategy.HYBRID
        )
        orchestrator.add_region(config, scaling_policy)
    
    return orchestrator

# Export all classes and functions
__all__ = [
    'GlobalScaleOrchestrator',
    'RegionManager',
    'GlobalLoadBalancer',
    'GlobalAutoScaler',
    'RegionConfig',
    'ScalingPolicy',
    'GlobalMetrics',
    'DeploymentRegion',
    'CloudProvider',
    'ScalingStrategy',
    'ServiceTier',
    'create_global_orchestrator'
]