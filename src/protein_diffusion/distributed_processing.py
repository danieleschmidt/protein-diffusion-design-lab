"""
Distributed Processing System - High-performance distributed computing for protein diffusion.

This module provides distributed processing capabilities including worker management,
load balancing, message queuing, and horizontal scaling for protein design workflows.
"""

import asyncio
import logging
import time
import uuid
import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
from enum import Enum
from queue import Queue, Empty
import multiprocessing as mp
from collections import defaultdict, deque

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import celery
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    CANCELLED = "cancelled"


class WorkerType(Enum):
    """Types of worker processes."""
    GENERATION_WORKER = "generation"
    RANKING_WORKER = "ranking"
    EVALUATION_WORKER = "evaluation"
    GENERAL_WORKER = "general"


@dataclass
class TaskDefinition:
    """Definition of a distributed task."""
    task_id: str
    task_type: str
    function_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    worker_type: WorkerType = WorkerType.GENERAL_WORKER
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a distributed task."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerInfo:
    """Information about a worker process."""
    worker_id: str
    worker_type: WorkerType
    pid: int
    started_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task: Optional[str] = None
    status: str = "idle"  # idle, busy, error, shutdown
    cpu_percent: float = 0.0
    memory_usage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedConfig:
    """Configuration for distributed processing system."""
    # Worker management
    min_workers: int = 2
    max_workers: int = 10
    worker_timeout: float = 300.0  # 5 minutes
    heartbeat_interval: float = 30.0  # 30 seconds
    
    # Task queue settings
    max_queue_size: int = 1000
    task_timeout: float = 600.0  # 10 minutes
    result_ttl: float = 3600.0  # 1 hour
    
    # Redis/Message broker settings
    use_redis: bool = REDIS_AVAILABLE
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Celery settings
    use_celery: bool = CELERY_AVAILABLE
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # Load balancing
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, random
    auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # Queue utilization to scale up
    scale_down_threshold: float = 0.2  # Queue utilization to scale down
    
    # Performance optimization
    batch_processing: bool = True
    batch_size: int = 10
    prefetch_multiplier: int = 4
    
    # Monitoring and logging
    enable_monitoring: bool = True
    metrics_interval: float = 60.0
    log_level: str = "INFO"


class TaskQueue:
    """High-performance task queue with priority support."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        
        if config.use_redis and REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                decode_responses=True
            )
            self.use_redis = True
            logger.info("Using Redis for task queue")
        else:
            self.use_redis = False
            self.local_queue = Queue(maxsize=config.max_queue_size)
            self.priority_queues = {
                1: Queue(),  # Low priority
                2: Queue(),  # Medium priority  
                3: Queue(),  # High priority
            }
            logger.info("Using local queues")
        
        self.pending_tasks: Dict[str, TaskDefinition] = {}
        self.results: Dict[str, TaskResult] = {}
        self.lock = threading.RLock()
    
    def enqueue(self, task: TaskDefinition) -> bool:
        """Enqueue a task for processing."""
        try:
            with self.lock:
                if self.use_redis:
                    task_data = {
                        'task_id': task.task_id,
                        'task_type': task.task_type,
                        'function_name': task.function_name,
                        'args': task.args,
                        'kwargs': task.kwargs,
                        'priority': task.priority,
                        'max_retries': task.max_retries,
                        'timeout': task.timeout,
                        'worker_type': task.worker_type.value,
                        'created_at': task.created_at,
                        'metadata': task.metadata
                    }
                    
                    # Use Redis sorted sets for priority queues
                    queue_key = f"tasks:{task.worker_type.value}"
                    self.redis_client.zadd(queue_key, {json.dumps(task_data): task.priority})
                    
                    # Store task details
                    self.redis_client.hset(f"task:{task.task_id}", mapping=task_data)
                    self.redis_client.expire(f"task:{task.task_id}", int(self.config.result_ttl))
                    
                else:
                    # Local queue implementation
                    self.pending_tasks[task.task_id] = task
                    
                    if task.priority in self.priority_queues:
                        self.priority_queues[task.priority].put(task)
                    else:
                        self.local_queue.put(task)
                
                logger.debug(f"Enqueued task {task.task_id} with priority {task.priority}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    def dequeue(self, worker_type: WorkerType, timeout: float = 1.0) -> Optional[TaskDefinition]:
        """Dequeue the highest priority task for worker type."""
        try:
            if self.use_redis:
                # Get highest priority task from Redis
                queue_key = f"tasks:{worker_type.value}"
                result = self.redis_client.zpopmax(queue_key)
                
                if result:
                    task_data_json, priority = result[0]
                    task_data = json.loads(task_data_json)
                    
                    # Reconstruct task definition
                    task = TaskDefinition(
                        task_id=task_data['task_id'],
                        task_type=task_data['task_type'],
                        function_name=task_data['function_name'],
                        args=task_data['args'],
                        kwargs=task_data['kwargs'],
                        priority=task_data['priority'],
                        max_retries=task_data['max_retries'],
                        timeout=task_data.get('timeout'),
                        worker_type=WorkerType(task_data['worker_type']),
                        created_at=task_data['created_at'],
                        metadata=task_data.get('metadata', {})
                    )
                    
                    return task
                
            else:
                # Local queue implementation - check priority queues first
                for priority in sorted(self.priority_queues.keys(), reverse=True):
                    try:
                        task = self.priority_queues[priority].get(timeout=timeout/3)
                        with self.lock:
                            if task.task_id in self.pending_tasks:
                                del self.pending_tasks[task.task_id]
                        return task
                    except Empty:
                        continue
                
                # Check general queue
                try:
                    task = self.local_queue.get(timeout=timeout)
                    with self.lock:
                        if task.task_id in self.pending_tasks:
                            del self.pending_tasks[task.task_id]
                    return task
                except Empty:
                    pass
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
    
    def store_result(self, result: TaskResult):
        """Store task result."""
        try:
            with self.lock:
                if self.use_redis:
                    result_data = {
                        'task_id': result.task_id,
                        'status': result.status.value,
                        'result': json.dumps(result.result, default=str) if result.result else None,
                        'error': result.error,
                        'worker_id': result.worker_id,
                        'execution_time': result.execution_time,
                        'retry_count': result.retry_count,
                        'started_at': result.started_at,
                        'completed_at': result.completed_at,
                        'metadata': json.dumps(result.metadata)
                    }
                    
                    self.redis_client.hset(f"result:{result.task_id}", mapping=result_data)
                    self.redis_client.expire(f"result:{result.task_id}", int(self.config.result_ttl))
                    
                else:
                    self.results[result.task_id] = result
                
                logger.debug(f"Stored result for task {result.task_id}")
                
        except Exception as e:
            logger.error(f"Failed to store result for task {result.task_id}: {e}")
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result by ID."""
        try:
            if self.use_redis:
                result_data = self.redis_client.hgetall(f"result:{task_id}")
                if result_data:
                    return TaskResult(
                        task_id=result_data['task_id'],
                        status=TaskStatus(result_data['status']),
                        result=json.loads(result_data['result']) if result_data.get('result') else None,
                        error=result_data.get('error'),
                        worker_id=result_data.get('worker_id'),
                        execution_time=float(result_data.get('execution_time', 0)),
                        retry_count=int(result_data.get('retry_count', 0)),
                        started_at=float(result_data['started_at']) if result_data.get('started_at') else None,
                        completed_at=float(result_data['completed_at']) if result_data.get('completed_at') else None,
                        metadata=json.loads(result_data.get('metadata', '{}'))
                    )
            else:
                return self.results.get(task_id)
            
        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {e}")
        
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {
            "timestamp": time.time(),
            "backend": "redis" if self.use_redis else "local"
        }
        
        try:
            if self.use_redis:
                # Get stats from Redis
                queue_lengths = {}
                for worker_type in WorkerType:
                    queue_key = f"tasks:{worker_type.value}"
                    queue_lengths[worker_type.value] = self.redis_client.zcard(queue_key)
                
                stats.update({
                    "queue_lengths": queue_lengths,
                    "total_queued": sum(queue_lengths.values()),
                    "pending_results": len(self.redis_client.keys("result:*"))
                })
            else:
                # Local queue stats
                priority_lengths = {str(p): q.qsize() for p, q in self.priority_queues.items()}
                stats.update({
                    "queue_lengths": {
                        "general": self.local_queue.qsize(),
                        **priority_lengths
                    },
                    "total_queued": self.local_queue.qsize() + sum(q.qsize() for q in self.priority_queues.values()),
                    "pending_tasks": len(self.pending_tasks),
                    "stored_results": len(self.results)
                })
                
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            stats["error"] = str(e)
        
        return stats


class Worker:
    """Distributed worker process for executing tasks."""
    
    def __init__(self, worker_id: str, worker_type: WorkerType, config: DistributedConfig, task_queue: TaskQueue):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.config = config
        self.task_queue = task_queue
        
        # Worker state
        self.running = False
        self.current_task = None
        self.tasks_completed = 0
        self.tasks_failed = 0
        
        # Function registry
        self.function_registry: Dict[str, Callable] = {}
        
        # Register default functions
        self._register_default_functions()
        
        logger.info(f"Worker {worker_id} ({worker_type.value}) initialized")
    
    def _register_default_functions(self):
        """Register default task functions."""
        # Import and register protein diffusion functions
        try:
            from ..diffuser import ProteinDiffuser
            from ..ranker import AffinityRanker
            
            self.function_registry.update({
                'generate_proteins': self._wrap_generation_function,
                'rank_proteins': self._wrap_ranking_function,
                'evaluate_sequences': self._wrap_evaluation_function,
            })
            
        except ImportError as e:
            logger.warning(f"Could not import protein diffusion modules: {e}")
    
    def _wrap_generation_function(self, *args, **kwargs):
        """Wrapper for protein generation."""
        try:
            # This would be implemented with actual diffuser instance
            # For demo, returning mock result
            time.sleep(0.1)  # Simulate processing time
            return {
                "sequences": ["MKLLILTCLVAVALARPKHPIP"] * kwargs.get('num_samples', 10),
                "confidence": [0.85] * kwargs.get('num_samples', 10)
            }
        except Exception as e:
            logger.error(f"Generation function failed: {e}")
            raise
    
    def _wrap_ranking_function(self, *args, **kwargs):
        """Wrapper for protein ranking."""
        try:
            sequences = kwargs.get('sequences', [])
            # Mock ranking result
            ranked_results = []
            for i, seq in enumerate(sequences):
                ranked_results.append({
                    "sequence": seq,
                    "rank": i + 1,
                    "score": 0.9 - (i * 0.1),
                    "binding_affinity": -8.5 - (i * 0.5)
                })
            return ranked_results
        except Exception as e:
            logger.error(f"Ranking function failed: {e}")
            raise
    
    def _wrap_evaluation_function(self, *args, **kwargs):
        """Wrapper for sequence evaluation."""
        try:
            sequences = kwargs.get('sequences', [])
            # Mock evaluation result
            return {
                "evaluations": [
                    {"sequence": seq, "quality": 0.8, "stability": 0.75}
                    for seq in sequences
                ]
            }
        except Exception as e:
            logger.error(f"Evaluation function failed: {e}")
            raise
    
    def register_function(self, name: str, func: Callable):
        """Register a task function."""
        self.function_registry[name] = func
        logger.info(f"Registered function '{name}' in worker {self.worker_id}")
    
    def start(self):
        """Start the worker process."""
        self.running = True
        logger.info(f"Worker {self.worker_id} started")
        
        while self.running:
            try:
                # Get next task
                task = self.task_queue.dequeue(self.worker_type, timeout=1.0)
                
                if task:
                    self._execute_task(task)
                else:
                    # No task available, brief sleep
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                time.sleep(1.0)
    
    def _execute_task(self, task: TaskDefinition):
        """Execute a single task."""
        start_time = time.time()
        self.current_task = task.task_id
        
        logger.info(f"Worker {self.worker_id} executing task {task.task_id}")
        
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            worker_id=self.worker_id,
            started_at=start_time
        )
        
        try:
            # Check if function exists
            if task.function_name not in self.function_registry:
                raise ValueError(f"Function '{task.function_name}' not registered")
            
            func = self.function_registry[task.function_name]
            
            # Execute with timeout if specified
            if task.timeout:
                # For simplicity, not implementing actual timeout here
                # In production, would use threading.Timer or similar
                pass
            
            # Execute function
            task_result = func(*task.args, **task.kwargs)
            
            # Task completed successfully
            result.status = TaskStatus.SUCCESS
            result.result = task_result
            result.execution_time = time.time() - start_time
            result.completed_at = time.time()
            
            self.tasks_completed += 1
            
            logger.info(f"Task {task.task_id} completed successfully in {result.execution_time:.2f}s")
            
        except Exception as e:
            # Task failed
            result.status = TaskStatus.FAILURE
            result.error = str(e)
            result.execution_time = time.time() - start_time
            result.completed_at = time.time()
            
            self.tasks_failed += 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Store result
            self.task_queue.store_result(result)
            self.current_task = None
    
    def stop(self):
        """Stop the worker."""
        self.running = False
        logger.info(f"Worker {self.worker_id} stopped")
    
    def get_info(self) -> WorkerInfo:
        """Get worker information."""
        try:
            import psutil
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_usage = process.memory_info().rss
        except ImportError:
            cpu_percent = 0.0
            memory_usage = 0
        
        return WorkerInfo(
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            pid=mp.current_process().pid,
            last_heartbeat=time.time(),
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            current_task=self.current_task,
            status="busy" if self.current_task else "idle",
            cpu_percent=cpu_percent,
            memory_usage=memory_usage
        )


class DistributedProcessingManager:
    """
    Distributed processing manager for protein diffusion workflows.
    
    This class provides:
    - Worker pool management with auto-scaling
    - Task distribution and load balancing
    - Result aggregation and monitoring
    - Fault tolerance and recovery
    
    Example:
        >>> manager = DistributedProcessingManager()
        >>> manager.start()
        >>> 
        >>> # Submit tasks
        >>> task_ids = manager.submit_batch_generation(
        ...     motifs=["HELIX_SHEET", "SHEET_LOOP"],
        ...     candidates_per_motif=100
        ... )
        >>> 
        >>> # Get results
        >>> results = manager.get_batch_results(task_ids)
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        
        # Initialize task queue
        self.task_queue = TaskQueue(self.config)
        
        # Worker management
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_processes: Dict[str, mp.Process] = {}
        
        # Load balancing
        self.current_worker_index = 0
        
        # Monitoring
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.performance_history: deque = deque(maxlen=100)
        
        # Auto-scaling
        self.last_scale_check = time.time()
        
        logger.info("Distributed processing manager initialized")
    
    def start(self, num_workers: Optional[int] = None):
        """Start the distributed processing system."""
        num_workers = num_workers or self.config.min_workers
        
        logger.info(f"Starting distributed processing with {num_workers} workers")
        
        # Start workers
        for i in range(num_workers):
            self._start_worker(WorkerType.GENERAL_WORKER)
        
        # Start monitoring thread if enabled
        if self.config.enable_monitoring:
            monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitoring_thread.start()
        
        logger.info("Distributed processing system started")
    
    def _start_worker(self, worker_type: WorkerType) -> str:
        """Start a new worker process."""
        worker_id = f"{worker_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Create worker process
        process = mp.Process(
            target=self._worker_main,
            args=(worker_id, worker_type, self.config, self.task_queue)
        )
        process.start()
        
        # Track worker
        self.worker_processes[worker_id] = process
        self.workers[worker_id] = WorkerInfo(
            worker_id=worker_id,
            worker_type=worker_type,
            pid=process.pid
        )
        
        logger.info(f"Started worker {worker_id} (PID: {process.pid})")
        return worker_id
    
    def _worker_main(self, worker_id: str, worker_type: WorkerType, config: DistributedConfig, task_queue: TaskQueue):
        """Main function for worker process."""
        try:
            worker = Worker(worker_id, worker_type, config, task_queue)
            worker.start()
        except Exception as e:
            logger.error(f"Worker {worker_id} crashed: {e}")
    
    def submit_task(
        self,
        function_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: int = 1,
        worker_type: WorkerType = WorkerType.GENERAL_WORKER,
        timeout: Optional[float] = None
    ) -> str:
        """Submit a single task for processing."""
        task_id = f"{function_name}_{uuid.uuid4().hex[:8]}"
        
        task = TaskDefinition(
            task_id=task_id,
            task_type="single_task",
            function_name=function_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            worker_type=worker_type,
            timeout=timeout
        )
        
        if self.task_queue.enqueue(task):
            self.metrics['tasks_submitted'] += 1
            logger.info(f"Submitted task {task_id}")
            return task_id
        else:
            raise RuntimeError(f"Failed to enqueue task {task_id}")
    
    def submit_batch_generation(
        self,
        motifs: List[str],
        candidates_per_motif: int = 50,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Submit batch protein generation tasks."""
        task_ids = []
        base_params = generation_params or {}
        
        for motif in motifs:
            task_params = {
                "motif": motif,
                "num_samples": candidates_per_motif,
                **base_params
            }
            
            task_id = self.submit_task(
                function_name="generate_proteins",
                kwargs=task_params,
                priority=2,
                worker_type=WorkerType.GENERATION_WORKER
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch generation for {len(motifs)} motifs")
        return task_ids
    
    def submit_batch_ranking(
        self,
        sequence_batches: List[List[str]],
        target_pdb: Optional[str] = None,
        ranking_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Submit batch protein ranking tasks."""
        task_ids = []
        base_params = ranking_params or {}
        
        for sequences in sequence_batches:
            task_params = {
                "sequences": sequences,
                "target_pdb": target_pdb,
                **base_params
            }
            
            task_id = self.submit_task(
                function_name="rank_proteins",
                kwargs=task_params,
                priority=2,
                worker_type=WorkerType.RANKING_WORKER
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch ranking for {len(sequence_batches)} batches")
        return task_ids
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result for a specific task."""
        start_time = time.time()
        
        while True:
            result = self.task_queue.get_result(task_id)
            
            if result and result.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE]:
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for result of task {task_id}")
                return None
            
            time.sleep(0.1)
    
    def get_batch_results(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None
    ) -> Dict[str, TaskResult]:
        """Get results for multiple tasks."""
        results = {}
        remaining_tasks = set(task_ids)
        start_time = time.time()
        
        while remaining_tasks:
            for task_id in list(remaining_tasks):
                result = self.task_queue.get_result(task_id)
                
                if result and result.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE]:
                    results[task_id] = result
                    remaining_tasks.remove(task_id)
            
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for {len(remaining_tasks)} task results")
                break
            
            if remaining_tasks:
                time.sleep(0.1)
        
        return results
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Collect performance metrics
                self._collect_performance_metrics()
                
                # Auto-scaling check
                if self.config.auto_scaling:
                    self._check_auto_scaling()
                
                # Clean up dead workers
                self._cleanup_dead_workers()
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _collect_performance_metrics(self):
        """Collect system performance metrics."""
        try:
            queue_stats = self.task_queue.get_queue_stats()
            
            metrics = {
                "timestamp": time.time(),
                "active_workers": len([w for w in self.workers.values() if w.status in ["idle", "busy"]]),
                "total_workers": len(self.workers),
                "queue_stats": queue_stats,
                "tasks_submitted": self.metrics['tasks_submitted'],
                "tasks_completed": sum(w.tasks_completed for w in self.workers.values()),
                "tasks_failed": sum(w.tasks_failed for w in self.workers.values())
            }
            
            self.performance_history.append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        current_time = time.time()
        
        # Only check every minute
        if current_time - self.last_scale_check < 60:
            return
        
        self.last_scale_check = current_time
        
        try:
            queue_stats = self.task_queue.get_queue_stats()
            total_queued = queue_stats.get("total_queued", 0)
            active_workers = len([w for w in self.workers.values() if w.status in ["idle", "busy"]])
            
            # Calculate queue utilization
            max_capacity = active_workers * 10  # Assume each worker can handle 10 tasks
            utilization = total_queued / max_capacity if max_capacity > 0 else 0
            
            # Scale up if needed
            if (
                utilization > self.config.scale_up_threshold and
                active_workers < self.config.max_workers
            ):
                self._start_worker(WorkerType.GENERAL_WORKER)
                logger.info(f"Scaled up: added worker (utilization: {utilization:.2f})")
            
            # Scale down if needed
            elif (
                utilization < self.config.scale_down_threshold and
                active_workers > self.config.min_workers
            ):
                self._stop_excess_worker()
                logger.info(f"Scaled down: removed worker (utilization: {utilization:.2f})")
                
        except Exception as e:
            logger.error(f"Auto-scaling check failed: {e}")
    
    def _cleanup_dead_workers(self):
        """Clean up dead or unresponsive workers."""
        try:
            dead_workers = []
            
            for worker_id, process in self.worker_processes.items():
                if not process.is_alive():
                    dead_workers.append(worker_id)
            
            for worker_id in dead_workers:
                logger.warning(f"Cleaning up dead worker {worker_id}")
                process = self.worker_processes[worker_id]
                process.join(timeout=5)
                
                del self.worker_processes[worker_id]
                if worker_id in self.workers:
                    del self.workers[worker_id]
                
                # Start replacement worker
                self._start_worker(WorkerType.GENERAL_WORKER)
                
        except Exception as e:
            logger.error(f"Worker cleanup failed: {e}")
    
    def _stop_excess_worker(self):
        """Stop one excess worker."""
        # Find idle worker to stop
        for worker_id, worker_info in self.workers.items():
            if worker_info.status == "idle" and worker_id in self.worker_processes:
                process = self.worker_processes[worker_id]
                process.terminate()
                process.join(timeout=5)
                
                del self.worker_processes[worker_id]
                del self.workers[worker_id]
                
                logger.info(f"Stopped excess worker {worker_id}")
                break
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            queue_stats = self.task_queue.get_queue_stats()
            
            # Worker status summary
            worker_status = defaultdict(int)
            for worker in self.workers.values():
                worker_status[worker.status] += 1
            
            # Performance summary
            if self.performance_history:
                latest_metrics = self.performance_history[-1]
                avg_completion_rate = sum(
                    m["tasks_completed"] for m in list(self.performance_history)[-10:]
                ) / min(10, len(self.performance_history))
            else:
                latest_metrics = {}
                avg_completion_rate = 0
            
            return {
                "timestamp": time.time(),
                "total_workers": len(self.workers),
                "worker_status": dict(worker_status),
                "queue_stats": queue_stats,
                "performance_metrics": latest_metrics,
                "avg_completion_rate": avg_completion_rate,
                "system_healthy": len([w for w in self.workers.values() if w.status != "error"]) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def shutdown(self):
        """Gracefully shutdown the distributed processing system."""
        logger.info("Shutting down distributed processing system")
        
        # Stop all workers
        for worker_id, process in self.worker_processes.items():
            logger.info(f"Stopping worker {worker_id}")
            process.terminate()
        
        # Wait for workers to stop
        for process in self.worker_processes.values():
            process.join(timeout=10)
        
        self.worker_processes.clear()
        self.workers.clear()
        
        logger.info("Distributed processing system shutdown complete")


# Convenience functions for high-level operations
def create_distributed_manager(config: Optional[DistributedConfig] = None) -> DistributedProcessingManager:
    """Create and configure distributed processing manager."""
    return DistributedProcessingManager(config)


def run_distributed_generation(
    motifs: List[str],
    candidates_per_motif: int = 50,
    num_workers: int = 4,
    timeout: float = 300.0
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run distributed protein generation for multiple motifs.
    
    Args:
        motifs: List of protein motifs to generate
        candidates_per_motif: Number of candidates per motif
        num_workers: Number of worker processes
        timeout: Timeout for completion
        
    Returns:
        Dictionary mapping motifs to generation results
    """
    manager = create_distributed_manager()
    manager.start(num_workers=num_workers)
    
    try:
        # Submit batch generation
        task_ids = manager.submit_batch_generation(
            motifs=motifs,
            candidates_per_motif=candidates_per_motif
        )
        
        # Get results
        results = manager.get_batch_results(task_ids, timeout=timeout)
        
        # Organize results by motif
        motif_results = {}
        for i, motif in enumerate(motifs):
            task_id = task_ids[i]
            if task_id in results and results[task_id].status == TaskStatus.SUCCESS:
                motif_results[motif] = results[task_id].result
            else:
                motif_results[motif] = {"error": "Generation failed"}
        
        return motif_results
        
    finally:
        manager.shutdown()
