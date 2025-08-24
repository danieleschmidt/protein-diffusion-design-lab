"""
Next-Generation Orchestration Engine for Protein Diffusion Design Lab

This module implements a comprehensive orchestration system that coordinates
advanced AI workflows, manages resource allocation, and provides intelligent
automation for protein design tasks.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Mock imports for environments without full dependencies
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    np = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of orchestrated tasks."""
    PROTEIN_GENERATION = "protein_generation"
    STRUCTURE_PREDICTION = "structure_prediction"  
    BINDING_AFFINITY = "binding_affinity"
    OPTIMIZATION = "optimization"
    RESEARCH_PIPELINE = "research_pipeline"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_STREAMING = "real_time_streaming"


class TaskPriority(Enum):
    """Task execution priorities."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ResourceRequirements:
    """Resource requirements for tasks."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 0.1
    network_bandwidth_mbps: float = 10.0
    estimated_duration_seconds: float = 60.0
    max_duration_seconds: float = 3600.0
    requires_gpu: bool = False
    requires_tpu: bool = False
    

@dataclass
class TaskDefinition:
    """Defines an orchestrated task."""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    parameters: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout_seconds: Optional[int] = None
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration engine."""
    max_concurrent_tasks: int = 10
    max_gpu_tasks: int = 2
    max_tpu_tasks: int = 1
    resource_check_interval: float = 5.0
    task_cleanup_interval: float = 300.0
    enable_auto_scaling: bool = True
    enable_load_balancing: bool = True
    enable_fault_tolerance: bool = True
    enable_telemetry: bool = True
    telemetry_interval: float = 30.0
    log_level: str = "INFO"


class ResourceManager:
    """Manages computational resources and allocation."""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.allocated_resources = {
            'cpu_cores': 0,
            'memory_gb': 0.0,
            'gpu_memory_gb': 0.0,
            'storage_gb': 0.0,
            'network_bandwidth_mbps': 0.0
        }
        self.available_resources = self._detect_available_resources()
        
    def _detect_available_resources(self) -> Dict[str, Any]:
        """Detect available system resources."""
        try:
            import psutil
            
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            storage_gb = psutil.disk_usage('/').free / (1024**3)
            
            # GPU detection
            gpu_memory_gb = 0.0
            has_gpu = False
            if TORCH_AVAILABLE and torch.cuda.is_available():
                has_gpu = True
                for i in range(torch.cuda.device_count()):
                    gpu_memory_gb += torch.cuda.get_device_properties(i).total_memory / (1024**3)
            
            # TPU detection (simplified)
            has_tpu = False
            try:
                # In actual implementation, would check for TPU availability
                has_tpu = False
            except Exception:
                has_tpu = False
                
            return {
                'cpu_cores': cpu_count,
                'memory_gb': memory_gb,
                'gpu_memory_gb': gpu_memory_gb,
                'storage_gb': storage_gb,
                'network_bandwidth_mbps': 1000.0,  # Assume gigabit
                'has_gpu': has_gpu,
                'has_tpu': has_tpu,
                'gpu_devices': torch.cuda.device_count() if has_gpu else 0
            }
        except ImportError:
            # Fallback for environments without psutil
            return {
                'cpu_cores': 4,
                'memory_gb': 16.0,
                'gpu_memory_gb': 8.0 if TORCH_AVAILABLE else 0.0,
                'storage_gb': 100.0,
                'network_bandwidth_mbps': 100.0,
                'has_gpu': TORCH_AVAILABLE and torch.cuda.is_available() if torch else False,
                'has_tpu': False,
                'gpu_devices': 1 if TORCH_AVAILABLE else 0
            }
    
    def can_allocate(self, requirements: ResourceRequirements) -> bool:
        """Check if resources can be allocated for a task."""
        available = self.available_resources
        allocated = self.allocated_resources
        
        # Check each resource requirement
        if (allocated['cpu_cores'] + requirements.cpu_cores) > available['cpu_cores']:
            return False
        if (allocated['memory_gb'] + requirements.memory_gb) > available['memory_gb']:
            return False
        if (allocated['gpu_memory_gb'] + requirements.gpu_memory_gb) > available['gpu_memory_gb']:
            return False
        if (allocated['storage_gb'] + requirements.storage_gb) > available['storage_gb']:
            return False
        if (allocated['network_bandwidth_mbps'] + requirements.network_bandwidth_mbps) > available['network_bandwidth_mbps']:
            return False
            
        # Check special requirements
        if requirements.requires_gpu and not available['has_gpu']:
            return False
        if requirements.requires_tpu and not available['has_tpu']:
            return False
            
        return True
    
    def allocate(self, requirements: ResourceRequirements) -> bool:
        """Allocate resources for a task."""
        if not self.can_allocate(requirements):
            return False
            
        self.allocated_resources['cpu_cores'] += requirements.cpu_cores
        self.allocated_resources['memory_gb'] += requirements.memory_gb
        self.allocated_resources['gpu_memory_gb'] += requirements.gpu_memory_gb
        self.allocated_resources['storage_gb'] += requirements.storage_gb
        self.allocated_resources['network_bandwidth_mbps'] += requirements.network_bandwidth_mbps
        
        logger.info(f"Resources allocated: CPU={requirements.cpu_cores}, "
                   f"Memory={requirements.memory_gb}GB, GPU={requirements.gpu_memory_gb}GB")
        return True
    
    def deallocate(self, requirements: ResourceRequirements):
        """Deallocate resources after task completion."""
        self.allocated_resources['cpu_cores'] = max(0, self.allocated_resources['cpu_cores'] - requirements.cpu_cores)
        self.allocated_resources['memory_gb'] = max(0.0, self.allocated_resources['memory_gb'] - requirements.memory_gb)
        self.allocated_resources['gpu_memory_gb'] = max(0.0, self.allocated_resources['gpu_memory_gb'] - requirements.gpu_memory_gb)
        self.allocated_resources['storage_gb'] = max(0.0, self.allocated_resources['storage_gb'] - requirements.storage_gb)
        self.allocated_resources['network_bandwidth_mbps'] = max(0.0, self.allocated_resources['network_bandwidth_mbps'] - requirements.network_bandwidth_mbps)
        
        logger.info(f"Resources deallocated: CPU={requirements.cpu_cores}, "
                   f"Memory={requirements.memory_gb}GB, GPU={requirements.gpu_memory_gb}GB")


class TaskExecutor:
    """Executes individual tasks with monitoring and error handling."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        
    async def execute_task(self, task: TaskDefinition) -> TaskDefinition:
        """Execute a single task with full monitoring."""
        task.started_at = time.time()
        task.status = TaskStatus.RUNNING
        
        # Allocate resources
        if not self.resource_manager.allocate(task.resource_requirements):
            task.status = TaskStatus.FAILED
            task.error = "Failed to allocate required resources"
            return task
            
        try:
            # Execute based on task type
            if task.task_type == TaskType.PROTEIN_GENERATION:
                result = await self._execute_protein_generation(task)
            elif task.task_type == TaskType.STRUCTURE_PREDICTION:
                result = await self._execute_structure_prediction(task)
            elif task.task_type == TaskType.BINDING_AFFINITY:
                result = await self._execute_binding_affinity(task)
            elif task.task_type == TaskType.OPTIMIZATION:
                result = await self._execute_optimization(task)
            elif task.task_type == TaskType.RESEARCH_PIPELINE:
                result = await self._execute_research_pipeline(task)
            elif task.task_type == TaskType.BATCH_PROCESSING:
                result = await self._execute_batch_processing(task)
            elif task.task_type == TaskType.REAL_TIME_STREAMING:
                result = await self._execute_real_time_streaming(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.progress = 1.0
            
            # Update metrics
            task.metrics['execution_time'] = task.completed_at - task.started_at
            task.metrics['resource_efficiency'] = self._calculate_resource_efficiency(task)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            logger.error(f"Task {task.task_id} failed: {e}")
            logger.error(traceback.format_exc())
            
        finally:
            # Always deallocate resources
            self.resource_manager.deallocate(task.resource_requirements)
            
        return task
    
    async def _execute_protein_generation(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute protein generation task."""
        params = task.parameters
        
        # Simulate protein generation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        num_proteins = params.get('num_proteins', 10)
        temperature = params.get('temperature', 0.8)
        motif = params.get('motif', 'HELIX_SHEET_HELIX')
        
        # Mock protein generation results
        proteins = []
        for i in range(num_proteins):
            protein = {
                'id': f'protein_{i}',
                'sequence': f'MKLLVLLVLL{motif}GGGHHHHHHH{i:02d}',
                'length': 50 + i,
                'temperature': temperature,
                'confidence': 0.85 + (i % 3) * 0.05,
                'generation_method': 'NextGen Diffusion'
            }
            proteins.append(protein)
            
            # Update progress
            task.progress = (i + 1) / num_proteins
            
        return {
            'proteins': proteins,
            'generation_params': params,
            'total_generated': len(proteins),
            'avg_confidence': sum(p['confidence'] for p in proteins) / len(proteins)
        }
    
    async def _execute_structure_prediction(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute structure prediction task."""
        params = task.parameters
        sequences = params.get('sequences', [])
        
        await asyncio.sleep(0.05 * len(sequences))
        
        structures = []
        for i, seq in enumerate(sequences):
            structure = {
                'sequence': seq,
                'pdb_content': f'MOCK_PDB_CONTENT_FOR_{seq[:10]}',
                'confidence': 0.9 + (i % 2) * 0.05,
                'tm_score': 0.85 + (i % 3) * 0.05,
                'method': 'ESMFold-NextGen'
            }
            structures.append(structure)
            task.progress = (i + 1) / len(sequences)
            
        return {
            'structures': structures,
            'total_predicted': len(structures),
            'avg_confidence': sum(s['confidence'] for s in structures) / len(structures) if structures else 0
        }
    
    async def _execute_binding_affinity(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute binding affinity calculation task."""
        params = task.parameters
        protein_pairs = params.get('protein_pairs', [])
        
        await asyncio.sleep(0.02 * len(protein_pairs))
        
        affinities = []
        for i, (protein1, protein2) in enumerate(protein_pairs):
            affinity = {
                'protein1': protein1,
                'protein2': protein2,
                'binding_affinity_kcal_mol': -8.5 - (i % 4) * 1.2,
                'confidence': 0.88 + (i % 3) * 0.04,
                'method': 'AutoDock-NextGen'
            }
            affinities.append(affinity)
            task.progress = (i + 1) / len(protein_pairs)
            
        return {
            'affinities': affinities,
            'total_calculated': len(affinities),
            'avg_affinity': sum(a['binding_affinity_kcal_mol'] for a in affinities) / len(affinities) if affinities else 0
        }
    
    async def _execute_optimization(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute optimization task."""
        params = task.parameters
        optimization_type = params.get('type', 'sequence_optimization')
        iterations = params.get('iterations', 100)
        
        best_score = 0.0
        optimization_history = []
        
        for i in range(iterations):
            # Simulate optimization step
            await asyncio.sleep(0.001)
            
            score = 0.5 + 0.4 * (1 - (i / iterations)) + 0.1 * (i % 10) / 10
            if score > best_score:
                best_score = score
                
            optimization_history.append({
                'iteration': i,
                'score': score,
                'best_score': best_score
            })
            
            task.progress = (i + 1) / iterations
            
        return {
            'optimization_type': optimization_type,
            'final_score': best_score,
            'iterations_completed': iterations,
            'optimization_history': optimization_history,
            'convergence_achieved': best_score > 0.9
        }
    
    async def _execute_research_pipeline(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute research pipeline task."""
        params = task.parameters
        pipeline_steps = params.get('steps', ['data_prep', 'training', 'evaluation'])
        
        results = {}
        for i, step in enumerate(pipeline_steps):
            await asyncio.sleep(0.1)  # Simulate step execution
            
            results[step] = {
                'status': 'completed',
                'execution_time': 0.1,
                'metrics': {
                    'accuracy': 0.92 + (i % 3) * 0.02,
                    'loss': 0.05 - (i % 2) * 0.01
                }
            }
            task.progress = (i + 1) / len(pipeline_steps)
            
        return {
            'pipeline_results': results,
            'total_steps': len(pipeline_steps),
            'overall_success': all(r['status'] == 'completed' for r in results.values())
        }
    
    async def _execute_batch_processing(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute batch processing task."""
        params = task.parameters
        batch_items = params.get('items', [])
        batch_size = params.get('batch_size', 10)
        
        processed_items = []
        for i, item in enumerate(batch_items):
            await asyncio.sleep(0.01)  # Simulate item processing
            
            processed_item = {
                'input': item,
                'output': f'processed_{item}',
                'processing_time': 0.01,
                'success': True
            }
            processed_items.append(processed_item)
            task.progress = (i + 1) / len(batch_items)
            
        return {
            'processed_items': processed_items,
            'total_processed': len(processed_items),
            'success_rate': 1.0,
            'batch_size_used': batch_size
        }
    
    async def _execute_real_time_streaming(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute real-time streaming task."""
        params = task.parameters
        stream_duration = params.get('duration_seconds', 10)
        update_interval = params.get('update_interval', 1.0)
        
        stream_data = []
        start_time = time.time()
        
        while (time.time() - start_time) < stream_duration:
            await asyncio.sleep(update_interval)
            
            data_point = {
                'timestamp': time.time(),
                'value': 0.5 + 0.3 * np.sin((time.time() - start_time) * 2) if np else 0.5,
                'metadata': {'stream_id': task.task_id}
            }
            stream_data.append(data_point)
            
            task.progress = (time.time() - start_time) / stream_duration
            
        return {
            'stream_data': stream_data,
            'total_points': len(stream_data),
            'stream_duration': time.time() - start_time,
            'avg_update_rate': len(stream_data) / (time.time() - start_time)
        }
    
    def _calculate_resource_efficiency(self, task: TaskDefinition) -> float:
        """Calculate resource utilization efficiency."""
        if not task.started_at or not task.completed_at:
            return 0.0
            
        execution_time = task.completed_at - task.started_at
        estimated_time = task.resource_requirements.estimated_duration_seconds
        
        # Simple efficiency calculation
        if estimated_time > 0:
            time_efficiency = min(1.0, estimated_time / execution_time)
        else:
            time_efficiency = 1.0
            
        return time_efficiency


class NextGenOrchestrationEngine:
    """
    Next-Generation Orchestration Engine
    
    Provides intelligent task scheduling, resource management, and workflow automation
    for advanced protein design and research workloads.
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.resource_manager = ResourceManager(config)
        self.task_executor = TaskExecutor(self.resource_manager)
        
        # Task management
        self.tasks: Dict[str, TaskDefinition] = {}
        self.task_queue: List[TaskDefinition] = []
        self.running_tasks: Dict[str, TaskDefinition] = {}
        self.completed_tasks: Dict[str, TaskDefinition] = {}
        
        # Orchestration state
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        
        # Telemetry and monitoring
        self.telemetry_data = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'resource_utilization': {}
        }
        
        logger.info(f"NextGen Orchestration Engine initialized with {config.max_concurrent_tasks} max concurrent tasks")
        
    def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task for orchestrated execution."""
        self.tasks[task.task_id] = task
        self.task_queue.append(task)
        self.telemetry_data['tasks_submitted'] += 1
        
        logger.info(f"Task {task.task_id} submitted for {task.task_type.value} execution")
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Optional[TaskDefinition]:
        """Get the current status of a task."""
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED
                task.completed_at = time.time()
                
                # Remove from queue if pending
                if task in self.task_queue:
                    self.task_queue.remove(task)
                    
                logger.info(f"Task {task_id} cancelled")
                return True
        return False
    
    async def start_orchestration(self):
        """Start the orchestration engine."""
        self.is_running = True
        logger.info("NextGen Orchestration Engine started")
        
        # Start main orchestration loop
        orchestration_task = asyncio.create_task(self._orchestration_loop())
        telemetry_task = asyncio.create_task(self._telemetry_loop())
        
        try:
            await asyncio.gather(orchestration_task, telemetry_task)
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
        finally:
            self.is_running = False
            
    async def stop_orchestration(self):
        """Stop the orchestration engine gracefully."""
        self.is_running = False
        
        # Wait for running tasks to complete
        while self.running_tasks:
            await asyncio.sleep(1)
            logger.info(f"Waiting for {len(self.running_tasks)} tasks to complete...")
            
        self.executor.shutdown(wait=True)
        logger.info("NextGen Orchestration Engine stopped")
        
    async def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.is_running:
            try:
                # Process task queue
                await self._process_task_queue()
                
                # Check running tasks
                await self._check_running_tasks()
                
                # Clean up completed tasks
                self._cleanup_old_tasks()
                
                # Resource optimization
                if self.config.enable_auto_scaling:
                    await self._optimize_resources()
                    
                await asyncio.sleep(self.config.resource_check_interval)
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(5)  # Back off on errors
                
    async def _process_task_queue(self):
        """Process pending tasks in the queue."""
        # Sort by priority
        self.task_queue.sort(key=lambda t: (t.priority.value, t.created_at))
        
        tasks_to_start = []
        for task in self.task_queue[:]:
            # Check dependencies
            if self._check_dependencies(task):
                # Check resource availability
                if self.resource_manager.can_allocate(task.resource_requirements):
                    # Check concurrency limits
                    if len(self.running_tasks) < self.config.max_concurrent_tasks:
                        tasks_to_start.append(task)
                        self.task_queue.remove(task)
                        
        # Start tasks
        for task in tasks_to_start:
            await self._start_task(task)
            
    async def _start_task(self, task: TaskDefinition):
        """Start executing a task."""
        self.running_tasks[task.task_id] = task
        
        # Execute task asynchronously
        task_coroutine = self.task_executor.execute_task(task)
        
        # Use asyncio to manage the task
        asyncio_task = asyncio.create_task(task_coroutine)
        asyncio_task.add_done_callback(lambda t: self._task_completed(task.task_id, t))
        
        logger.info(f"Started task {task.task_id} ({task.task_type.value})")
        
    def _task_completed(self, task_id: str, asyncio_task):
        """Handle task completion."""
        if task_id in self.running_tasks:
            task = self.running_tasks.pop(task_id)
            
            try:
                completed_task = asyncio_task.result()
                self.completed_tasks[task_id] = completed_task
                
                if completed_task.status == TaskStatus.COMPLETED:
                    self.telemetry_data['tasks_completed'] += 1
                    logger.info(f"Task {task_id} completed successfully")
                else:
                    self.telemetry_data['tasks_failed'] += 1
                    logger.warning(f"Task {task_id} failed: {completed_task.error}")
                    
                # Update telemetry
                if completed_task.started_at and completed_task.completed_at:
                    execution_time = completed_task.completed_at - completed_task.started_at
                    self.telemetry_data['total_execution_time'] += execution_time
                    
                # Call callback if provided
                if completed_task.callback:
                    try:
                        completed_task.callback(completed_task)
                    except Exception as e:
                        logger.error(f"Task callback error for {task_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Error handling task completion for {task_id}: {e}")
                
    def _check_dependencies(self, task: TaskDefinition) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            dep_task = self.completed_tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
        
    async def _check_running_tasks(self):
        """Check status of running tasks."""
        current_time = time.time()
        tasks_to_timeout = []
        
        for task_id, task in self.running_tasks.items():
            # Check for timeout
            if task.timeout_seconds and task.started_at:
                if (current_time - task.started_at) > task.timeout_seconds:
                    tasks_to_timeout.append(task_id)
                    
        # Handle timeouts
        for task_id in tasks_to_timeout:
            self.cancel_task(task_id)
            logger.warning(f"Task {task_id} timed out")
            
    def _cleanup_old_tasks(self):
        """Clean up old completed tasks."""
        current_time = time.time()
        cleanup_age = self.config.task_cleanup_interval
        
        tasks_to_remove = []
        for task_id, task in self.completed_tasks.items():
            if task.completed_at and (current_time - task.completed_at) > cleanup_age:
                tasks_to_remove.append(task_id)
                
        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]
            del self.tasks[task_id]
            
    async def _optimize_resources(self):
        """Optimize resource allocation and scaling."""
        # Simple auto-scaling logic
        queue_length = len(self.task_queue)
        running_count = len(self.running_tasks)
        
        # Scale up if queue is backing up
        if queue_length > self.config.max_concurrent_tasks and running_count < self.config.max_concurrent_tasks:
            logger.info("Auto-scaling: Queue backup detected, considering scale-up")
            
        # Resource utilization monitoring
        total_resources = self.resource_manager.available_resources
        allocated_resources = self.resource_manager.allocated_resources
        
        for resource_type, allocated in allocated_resources.items():
            if resource_type in total_resources:
                utilization = allocated / max(total_resources[resource_type], 1.0)
                self.telemetry_data['resource_utilization'][resource_type] = utilization
                
    async def _telemetry_loop(self):
        """Collect and report telemetry data."""
        while self.is_running:
            try:
                if self.config.enable_telemetry:
                    # Update telemetry metrics
                    self.telemetry_data['active_tasks'] = len(self.running_tasks)
                    self.telemetry_data['queued_tasks'] = len(self.task_queue)
                    self.telemetry_data['completed_tasks'] = len(self.completed_tasks)
                    
                    # Calculate success rate
                    total_finished = self.telemetry_data['tasks_completed'] + self.telemetry_data['tasks_failed']
                    if total_finished > 0:
                        self.telemetry_data['success_rate'] = self.telemetry_data['tasks_completed'] / total_finished
                    else:
                        self.telemetry_data['success_rate'] = 1.0
                        
                    logger.debug(f"Telemetry: {json.dumps(self.telemetry_data, indent=2)}")
                    
                await asyncio.sleep(self.config.telemetry_interval)
                
            except Exception as e:
                logger.error(f"Telemetry error: {e}")
                await asyncio.sleep(10)
                
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data."""
        return self.telemetry_data.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'orchestration_status': 'running' if self.is_running else 'stopped',
            'resource_manager': {
                'available_resources': self.resource_manager.available_resources,
                'allocated_resources': self.resource_manager.allocated_resources,
                'utilization_rates': {
                    resource: allocated / max(self.resource_manager.available_resources.get(resource, 1), 1)
                    for resource, allocated in self.resource_manager.allocated_resources.items()
                }
            },
            'task_statistics': {
                'total_tasks': len(self.tasks),
                'queued_tasks': len(self.task_queue),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks)
            },
            'telemetry': self.telemetry_data,
            'configuration': {
                'max_concurrent_tasks': self.config.max_concurrent_tasks,
                'auto_scaling_enabled': self.config.enable_auto_scaling,
                'fault_tolerance_enabled': self.config.enable_fault_tolerance
            }
        }


# Convenience functions for quick task creation
def create_protein_generation_task(
    task_id: str,
    num_proteins: int = 100,
    temperature: float = 0.8,
    motif: str = "HELIX_SHEET_HELIX",
    priority: TaskPriority = TaskPriority.MEDIUM
) -> TaskDefinition:
    """Create a protein generation task."""
    return TaskDefinition(
        task_id=task_id,
        task_type=TaskType.PROTEIN_GENERATION,
        priority=priority,
        parameters={
            'num_proteins': num_proteins,
            'temperature': temperature,
            'motif': motif
        },
        resource_requirements=ResourceRequirements(
            cpu_cores=2,
            memory_gb=4.0,
            gpu_memory_gb=2.0,
            estimated_duration_seconds=30.0,
            requires_gpu=True
        )
    )


def create_structure_prediction_task(
    task_id: str,
    sequences: List[str],
    priority: TaskPriority = TaskPriority.HIGH
) -> TaskDefinition:
    """Create a structure prediction task."""
    return TaskDefinition(
        task_id=task_id,
        task_type=TaskType.STRUCTURE_PREDICTION,
        priority=priority,
        parameters={'sequences': sequences},
        resource_requirements=ResourceRequirements(
            cpu_cores=4,
            memory_gb=8.0,
            gpu_memory_gb=4.0,
            estimated_duration_seconds=60.0 * len(sequences),
            requires_gpu=True
        )
    )


def create_research_pipeline_task(
    task_id: str,
    pipeline_steps: List[str],
    priority: TaskPriority = TaskPriority.LOW
) -> TaskDefinition:
    """Create a research pipeline task."""
    return TaskDefinition(
        task_id=task_id,
        task_type=TaskType.RESEARCH_PIPELINE,
        priority=priority,
        parameters={'steps': pipeline_steps},
        resource_requirements=ResourceRequirements(
            cpu_cores=8,
            memory_gb=16.0,
            gpu_memory_gb=8.0,
            estimated_duration_seconds=300.0,
            requires_gpu=True
        )
    )


# Main orchestration interface
async def demo_orchestration():
    """Demonstrate the orchestration engine capabilities."""
    config = OrchestrationConfig(
        max_concurrent_tasks=5,
        enable_telemetry=True,
        telemetry_interval=10.0
    )
    
    engine = NextGenOrchestrationEngine(config)
    
    # Create sample tasks
    tasks = [
        create_protein_generation_task("gen_1", num_proteins=50),
        create_structure_prediction_task("struct_1", ["MKLLVLLLVL", "ASDGHKLQWE"]),
        create_research_pipeline_task("research_1", ["data_prep", "training", "evaluation"])
    ]
    
    # Submit tasks
    for task in tasks:
        engine.submit_task(task)
        
    # Start orchestration for a short demo
    print("Starting NextGen Orchestration Engine...")
    
    # Run for 30 seconds
    try:
        await asyncio.wait_for(engine.start_orchestration(), timeout=30.0)
    except asyncio.TimeoutError:
        await engine.stop_orchestration()
        
    # Show results
    print("\nOrchestration Results:")
    print(json.dumps(engine.get_system_status(), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(demo_orchestration())