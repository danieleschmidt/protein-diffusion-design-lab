"""
Microservices Architecture for Protein Diffusion Platform

Advanced microservices architecture with service discovery,
load balancing, and distributed processing capabilities.
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

try:
    import grpc
    import grpc.aio
    from grpc import StatusCode
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import etcd3
    ETCD_AVAILABLE = True
except ImportError:
    ETCD_AVAILABLE = False


logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Types of microservices."""
    GENERATION_SERVICE = "generation"
    VALIDATION_SERVICE = "validation" 
    FOLDING_SERVICE = "folding"
    RANKING_SERVICE = "ranking"
    CACHING_SERVICE = "caching"
    NOTIFICATION_SERVICE = "notification"
    ORCHESTRATION_SERVICE = "orchestration"
    MONITORING_SERVICE = "monitoring"


class ServiceStatus(Enum):
    """Service status types."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


@dataclass
class ServiceInfo:
    """Service registration information."""
    service_id: str
    service_type: ServiceType
    name: str
    version: str
    host: str
    port: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_url: str = ""
    status: ServiceStatus = ServiceStatus.STARTING
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_type": self.service_type.value,
            "name": self.name,
            "version": self.version,
            "host": self.host,
            "port": self.port,
            "metadata": self.metadata,
            "health_check_url": self.health_check_url,
            "status": self.status.value,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat()
        }


@dataclass
class ServiceRequest:
    """Service request message."""
    request_id: str
    service_type: ServiceType
    method: str
    payload: Dict[str, Any]
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service_type": self.service_type.value,
            "method": self.method,
            "payload": self.payload,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ServiceResponse:
    """Service response message."""
    request_id: str
    service_id: str
    success: bool
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service_id": self.service_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class ServiceRegistry:
    """Service discovery and registry."""
    
    def __init__(self, use_etcd: bool = False, etcd_host: str = "localhost", etcd_port: int = 2379):
        self.services: Dict[str, ServiceInfo] = {}
        self.use_etcd = use_etcd and ETCD_AVAILABLE
        self.etcd_client = None
        
        if self.use_etcd:
            try:
                self.etcd_client = etcd3.client(host=etcd_host, port=etcd_port)
            except Exception as e:
                logger.warning(f"Failed to connect to etcd: {e}")
                self.use_etcd = False
        
        self._lock = threading.RLock()
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        def cleanup():
            while True:
                time.sleep(60)  # Cleanup every minute
                self.cleanup_stale_services()
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a service."""
        try:
            with self._lock:
                self.services[service_info.service_id] = service_info
                
                if self.use_etcd and self.etcd_client:
                    key = f"/services/{service_info.service_type.value}/{service_info.service_id}"
                    value = json.dumps(service_info.to_dict())
                    self.etcd_client.put(key, value)
            
            logger.info(f"Registered service {service_info.service_id} ({service_info.service_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service_info.service_id}: {e}")
            return False
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service."""
        try:
            with self._lock:
                if service_id in self.services:
                    service_info = self.services[service_id]
                    del self.services[service_id]
                    
                    if self.use_etcd and self.etcd_client:
                        key = f"/services/{service_info.service_type.value}/{service_id}"
                        self.etcd_client.delete(key)
            
            logger.info(f"Unregistered service {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister service {service_id}: {e}")
            return False
    
    def get_services(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Get all healthy services of a given type."""
        with self._lock:
            return [
                service for service in self.services.values()
                if service.service_type == service_type and service.status == ServiceStatus.HEALTHY
            ]
    
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """Get specific service info."""
        return self.services.get(service_id)
    
    def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """Update service status."""
        try:
            with self._lock:
                if service_id in self.services:
                    self.services[service_id].status = status
                    self.services[service_id].last_heartbeat = datetime.now(timezone.utc)
                    
                    if self.use_etcd and self.etcd_client:
                        service_info = self.services[service_id]
                        key = f"/services/{service_info.service_type.value}/{service_id}"
                        value = json.dumps(service_info.to_dict())
                        self.etcd_client.put(key, value)
                    
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update service status {service_id}: {e}")
            return False
    
    def cleanup_stale_services(self, max_age_minutes: int = 5):
        """Clean up services that haven't sent heartbeats."""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_minutes * 60)
        
        stale_services = []
        with self._lock:
            for service_id, service_info in self.services.items():
                if service_info.last_heartbeat.timestamp() < cutoff:
                    stale_services.append(service_id)
        
        for service_id in stale_services:
            logger.warning(f"Removing stale service {service_id}")
            self.unregister_service(service_id)


class LoadBalancer:
    """Load balancer for service requests."""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.round_robin_counters: Dict[ServiceType, int] = {}
        self.service_weights: Dict[str, float] = {}
        self.service_response_times: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def select_service(self, service_type: ServiceType, strategy: str = "round_robin") -> Optional[ServiceInfo]:
        """Select best service instance using load balancing strategy."""
        services = self.registry.get_services(service_type)
        if not services:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_select(services, service_type)
        elif strategy == "weighted_round_robin":
            return self._weighted_round_robin_select(services)
        elif strategy == "least_response_time":
            return self._least_response_time_select(services)
        elif strategy == "random":
            import random
            return random.choice(services)
        else:
            return services[0]  # Fallback to first available
    
    def _round_robin_select(self, services: List[ServiceInfo], service_type: ServiceType) -> ServiceInfo:
        """Round-robin service selection."""
        with self._lock:
            if service_type not in self.round_robin_counters:
                self.round_robin_counters[service_type] = 0
            
            index = self.round_robin_counters[service_type] % len(services)
            self.round_robin_counters[service_type] += 1
            return services[index]
    
    def _weighted_round_robin_select(self, services: List[ServiceInfo]) -> ServiceInfo:
        """Weighted round-robin based on service performance."""
        if not services:
            return None
        
        # Calculate weights based on average response time (lower = better)
        weighted_services = []
        for service in services:
            service_id = service.service_id
            if service_id in self.service_response_times:
                avg_time = sum(self.service_response_times[service_id]) / len(self.service_response_times[service_id])
                weight = 1.0 / (avg_time + 0.001)  # Avoid division by zero
            else:
                weight = 1.0
            weighted_services.append((service, weight))
        
        # Select based on weights
        total_weight = sum(weight for _, weight in weighted_services)
        import random
        r = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for service, weight in weighted_services:
            cumulative_weight += weight
            if r <= cumulative_weight:
                return service
        
        return services[0]  # Fallback
    
    def _least_response_time_select(self, services: List[ServiceInfo]) -> ServiceInfo:
        """Select service with lowest average response time."""
        if not services:
            return None
        
        best_service = services[0]
        best_avg_time = float('inf')
        
        for service in services:
            service_id = service.service_id
            if service_id in self.service_response_times and self.service_response_times[service_id]:
                avg_time = sum(self.service_response_times[service_id]) / len(self.service_response_times[service_id])
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_service = service
        
        return best_service
    
    def record_response_time(self, service_id: str, response_time: float):
        """Record service response time for performance tracking."""
        with self._lock:
            if service_id not in self.service_response_times:
                self.service_response_times[service_id] = []
            
            # Keep only last 100 response times
            self.service_response_times[service_id].append(response_time)
            if len(self.service_response_times[service_id]) > 100:
                self.service_response_times[service_id] = self.service_response_times[service_id][-100:]


class BaseService(ABC):
    """Base class for microservices."""
    
    def __init__(self, name: str, version: str = "1.0.0", host: str = "localhost", port: int = 0):
        self.service_id = str(uuid.uuid4())
        self.name = name
        self.version = version
        self.host = host
        self.port = port if port != 0 else self._find_free_port()
        self.status = ServiceStatus.STARTING
        self.registry: Optional[ServiceRegistry] = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.shutdown_event = threading.Event()
        
    def _find_free_port(self) -> int:
        """Find an available port."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    @abstractmethod
    def get_service_type(self) -> ServiceType:
        """Get the service type."""
        pass
    
    @abstractmethod
    async def handle_request(self, request: ServiceRequest) -> ServiceResponse:
        """Handle incoming service request."""
        pass
    
    def register_with_registry(self, registry: ServiceRegistry) -> bool:
        """Register this service with the registry."""
        self.registry = registry
        
        service_info = ServiceInfo(
            service_id=self.service_id,
            service_type=self.get_service_type(),
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
            status=self.status,
            metadata=self.get_metadata()
        )
        
        return registry.register_service(service_info)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get service metadata."""
        return {
            "capabilities": self.get_capabilities(),
            "max_concurrent_requests": 10,
            "average_response_time": 0.5
        }
    
    def get_capabilities(self) -> List[str]:
        """Get service capabilities."""
        return []
    
    def start(self):
        """Start the service."""
        self.status = ServiceStatus.HEALTHY
        if self.registry:
            self.registry.update_service_status(self.service_id, self.status)
        
        logger.info(f"Started service {self.name} ({self.service_id}) on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the service."""
        self.status = ServiceStatus.SHUTTING_DOWN
        if self.registry:
            self.registry.update_service_status(self.service_id, self.status)
            self.registry.unregister_service(self.service_id)
        
        self.executor.shutdown(wait=True)
        self.shutdown_event.set()
        
        logger.info(f"Stopped service {self.name} ({self.service_id})")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": self.status.value,
            "service_id": self.service_id,
            "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            "active_requests": 0,  # Would track actual active requests
            "memory_usage": 0,     # Would get actual memory usage
        }


class ProteinGenerationService(BaseService):
    """Protein generation microservice."""
    
    def __init__(self):
        super().__init__("protein-generation-service")
        
    def get_service_type(self) -> ServiceType:
        return ServiceType.GENERATION_SERVICE
    
    def get_capabilities(self) -> List[str]:
        return ["diffusion_generation", "conditional_generation", "batch_generation"]
    
    async def handle_request(self, request: ServiceRequest) -> ServiceResponse:
        """Handle protein generation requests."""
        start_time = time.time()
        
        try:
            if request.method == "generate":
                result = await self._generate_proteins(request.payload)
            elif request.method == "batch_generate":
                result = await self._batch_generate_proteins(request.payload)
            else:
                raise ValueError(f"Unknown method: {request.method}")
            
            execution_time = time.time() - start_time
            
            return ServiceResponse(
                request_id=request.request_id,
                service_id=self.service_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ServiceResponse(
                request_id=request.request_id,
                service_id=self.service_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _generate_proteins(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate proteins based on request."""
        motif = payload.get("motif", "")
        num_samples = payload.get("num_samples", 1)
        temperature = payload.get("temperature", 0.8)
        
        # Simulate generation
        await asyncio.sleep(0.5)  # Simulate processing time
        
        sequences = []
        for i in range(num_samples):
            sequence = f"MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG_{i}"
            sequences.append({
                "sequence": sequence,
                "confidence": 0.85 + (i * 0.02),
                "motif_match": 0.92,
                "novelty_score": 0.78
            })
        
        return {
            "sequences": sequences,
            "generation_params": {
                "motif": motif,
                "num_samples": num_samples,
                "temperature": temperature
            }
        }
    
    async def _batch_generate_proteins(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multiple protein batches."""
        batch_requests = payload.get("batch_requests", [])
        results = []
        
        for batch_request in batch_requests:
            batch_result = await self._generate_proteins(batch_request)
            results.append(batch_result)
        
        return {"batch_results": results}


class ProteinValidationService(BaseService):
    """Protein validation microservice."""
    
    def __init__(self):
        super().__init__("protein-validation-service")
    
    def get_service_type(self) -> ServiceType:
        return ServiceType.VALIDATION_SERVICE
    
    def get_capabilities(self) -> List[str]:
        return ["sequence_validation", "structure_validation", "binding_validation"]
    
    async def handle_request(self, request: ServiceRequest) -> ServiceResponse:
        """Handle protein validation requests."""
        start_time = time.time()
        
        try:
            if request.method == "validate_sequence":
                result = await self._validate_sequence(request.payload)
            elif request.method == "validate_structure":
                result = await self._validate_structure(request.payload)
            else:
                raise ValueError(f"Unknown method: {request.method}")
            
            execution_time = time.time() - start_time
            
            return ServiceResponse(
                request_id=request.request_id,
                service_id=self.service_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ServiceResponse(
                request_id=request.request_id,
                service_id=self.service_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _validate_sequence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate protein sequence."""
        sequence = payload.get("sequence", "")
        
        # Simulate validation
        await asyncio.sleep(0.2)
        
        return {
            "valid": True,
            "length": len(sequence),
            "composition_score": 0.89,
            "stability_score": 0.76,
            "folding_probability": 0.92,
            "issues": []
        }
    
    async def _validate_structure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate protein structure."""
        structure_data = payload.get("structure_data", {})
        
        # Simulate structure validation
        await asyncio.sleep(0.3)
        
        return {
            "valid": True,
            "ramachandran_score": 0.94,
            "clash_score": 0.88,
            "geometry_score": 0.91,
            "overall_quality": 0.91
        }


class ServiceOrchestrator:
    """Orchestrates microservice interactions."""
    
    def __init__(self, registry: ServiceRegistry, load_balancer: LoadBalancer):
        self.registry = registry
        self.load_balancer = load_balancer
        self.request_cache: Dict[str, ServiceResponse] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    async def call_service(
        self,
        service_type: ServiceType,
        method: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
        retry_attempts: int = 3
    ) -> ServiceResponse:
        """Call a microservice with retry logic."""
        
        request_id = str(uuid.uuid4())
        request = ServiceRequest(
            request_id=request_id,
            service_type=service_type,
            method=method,
            payload=payload,
            timeout=timeout,
            max_retries=retry_attempts
        )
        
        for attempt in range(retry_attempts + 1):
            try:
                service = self.load_balancer.select_service(service_type)
                if not service:
                    raise RuntimeError(f"No healthy services available for {service_type.value}")
                
                request.retry_count = attempt
                response = await self._execute_request(service, request)
                
                if response.success:
                    self.load_balancer.record_response_time(service.service_id, response.execution_time)
                    return response
                else:
                    logger.warning(f"Service call failed (attempt {attempt + 1}): {response.error}")
                    
            except Exception as e:
                logger.error(f"Service call error (attempt {attempt + 1}): {e}")
                
                if attempt == retry_attempts:
                    return ServiceResponse(
                        request_id=request_id,
                        service_id="orchestrator",
                        success=False,
                        error=f"All retry attempts failed: {str(e)}"
                    )
                
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        return ServiceResponse(
            request_id=request_id,
            service_id="orchestrator",
            success=False,
            error="Max retries exceeded"
        )
    
    async def _execute_request(self, service: ServiceInfo, request: ServiceRequest) -> ServiceResponse:
        """Execute request on specific service."""
        # This is a simplified simulation - in practice would use actual HTTP/gRPC calls
        
        if service.service_type == ServiceType.GENERATION_SERVICE:
            generation_service = ProteinGenerationService()
            return await generation_service.handle_request(request)
        elif service.service_type == ServiceType.VALIDATION_SERVICE:
            validation_service = ProteinValidationService()
            return await validation_service.handle_request(request)
        else:
            raise NotImplementedError(f"Service type {service.service_type.value} not implemented")
    
    async def parallel_call_services(
        self,
        service_calls: List[Dict[str, Any]]
    ) -> List[ServiceResponse]:
        """Execute multiple service calls in parallel."""
        
        tasks = []
        for call in service_calls:
            task = self.call_service(
                service_type=ServiceType(call["service_type"]),
                method=call["method"],
                payload=call.get("payload", {}),
                timeout=call.get("timeout", 30.0),
                retry_attempts=call.get("retry_attempts", 3)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def workflow_execution(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complex workflow with multiple service calls."""
        
        workflow_id = str(uuid.uuid4())
        steps = workflow_definition.get("steps", [])
        context = workflow_definition.get("context", {})
        results = {}
        
        try:
            for step in steps:
                step_name = step["name"]
                service_type = ServiceType(step["service_type"])
                method = step["method"]
                payload = step["payload"]
                
                # Replace context variables in payload
                payload = self._substitute_context_variables(payload, context)
                
                logger.info(f"Executing workflow {workflow_id} step: {step_name}")
                
                response = await self.call_service(service_type, method, payload)
                
                if response.success:
                    results[step_name] = response.result
                    # Update context with results
                    context.update(response.result)
                else:
                    logger.error(f"Workflow {workflow_id} failed at step {step_name}: {response.error}")
                    return {
                        "workflow_id": workflow_id,
                        "success": False,
                        "error": f"Failed at step {step_name}: {response.error}",
                        "completed_steps": list(results.keys())
                    }
            
            return {
                "workflow_id": workflow_id,
                "success": True,
                "results": results,
                "completed_steps": list(results.keys())
            }
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} execution error: {e}")
            return {
                "workflow_id": workflow_id,
                "success": False,
                "error": str(e),
                "completed_steps": list(results.keys())
            }
    
    def _substitute_context_variables(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Replace context variables in payload."""
        import re
        
        def replace_vars(obj):
            if isinstance(obj, str):
                # Replace ${variable} patterns
                def replacer(match):
                    var_name = match.group(1)
                    return str(context.get(var_name, match.group(0)))
                
                return re.sub(r'\$\{([^}]+)\}', replacer, obj)
            elif isinstance(obj, dict):
                return {k: replace_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_vars(item) for item in obj]
            else:
                return obj
        
        return replace_vars(payload)


# Global instances
service_registry = ServiceRegistry()
load_balancer = LoadBalancer(service_registry)
service_orchestrator = ServiceOrchestrator(service_registry, load_balancer)


# Example usage
async def example_microservices_usage():
    """Example of how to use the microservices architecture."""
    
    # Register services
    gen_service = ProteinGenerationService()
    val_service = ProteinValidationService()
    
    gen_service.register_with_registry(service_registry)
    val_service.register_with_registry(service_registry)
    
    gen_service.start()
    val_service.start()
    
    try:
        # Single service call
        response = await service_orchestrator.call_service(
            ServiceType.GENERATION_SERVICE,
            "generate",
            {"motif": "HELIX_SHEET_HELIX", "num_samples": 3}
        )
        
        print(f"Generation response: {response.success}")
        
        # Parallel service calls
        parallel_calls = [
            {
                "service_type": "generation",
                "method": "generate",
                "payload": {"motif": "BETA_SHEET", "num_samples": 2}
            },
            {
                "service_type": "validation",
                "method": "validate_sequence",
                "payload": {"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"}
            }
        ]
        
        parallel_responses = await service_orchestrator.parallel_call_services(parallel_calls)
        print(f"Parallel calls completed: {len(parallel_responses)}")
        
        # Workflow execution
        workflow = {
            "steps": [
                {
                    "name": "generate_protein",
                    "service_type": "generation",
                    "method": "generate",
                    "payload": {"motif": "ALPHA_HELIX", "num_samples": 1}
                },
                {
                    "name": "validate_protein",
                    "service_type": "validation", 
                    "method": "validate_sequence",
                    "payload": {"sequence": "${sequences[0].sequence}"}
                }
            ],
            "context": {}
        }
        
        workflow_result = await service_orchestrator.workflow_execution(workflow)
        print(f"Workflow completed: {workflow_result['success']}")
        
    finally:
        gen_service.stop()
        val_service.stop()


if __name__ == "__main__":
    asyncio.run(example_microservices_usage())