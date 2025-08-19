"""
Workflow Automation - Automated protein design pipelines and batch processing.

This module provides automated workflows for complex protein design tasks,
including batch processing, pipeline orchestration, and result optimization.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Iterator
import json
import csv

try:
    from .integration_manager import IntegrationManager, IntegrationConfig, WorkflowResult
    from .diffuser import ProteinDiffuserConfig
    from .ranker import AffinityRankerConfig
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """Individual task in a batch workflow."""
    task_id: str
    task_type: str  # "generation", "evaluation", "design_and_rank"
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[WorkflowResult] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def execution_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


@dataclass
class AutomationConfig:
    """Configuration for workflow automation."""
    max_concurrent_tasks: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout_seconds: int = 3600  # 1 hour
    save_intermediate_results: bool = True
    output_directory: str = "./automation_output"
    enable_optimization: bool = True
    optimization_iterations: int = 3
    
    # Integration config
    integration_config: Optional[IntegrationConfig] = None
    
    def __post_init__(self):
        if self.integration_config is None and INTEGRATION_AVAILABLE:
            self.integration_config = IntegrationConfig()


@dataclass
class PipelineResult:
    """Result from an automated pipeline execution."""
    pipeline_id: str
    success: bool
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    execution_time: float
    results: List[WorkflowResult] = field(default_factory=list)
    task_results: Dict[str, BatchTask] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class WorkflowAutomation:
    """
    Automated workflow system for protein design pipelines.
    
    This class provides high-level automation capabilities including:
    - Batch processing of protein design tasks
    - Pipeline orchestration with dependency management
    - Adaptive parameter optimization
    - Result aggregation and analysis
    
    Example:
        >>> automation = WorkflowAutomation()
        >>> pipeline = automation.create_design_pipeline(
        ...     motifs=["HELIX_SHEET", "SHEET_LOOP_SHEET"],
        ...     candidates_per_motif=100,
        ...     target_pdb="spike_protein.pdb"
        ... )
        >>> result = automation.execute_pipeline(pipeline)
    """
    
    def __init__(self, config: Optional[AutomationConfig] = None):
        if not INTEGRATION_AVAILABLE:
            raise ImportError("Integration components not available for automation")
        
        self.config = config or AutomationConfig()
        self.integration_manager = IntegrationManager(self.config.integration_config)
        
        # Task execution state
        self.active_pipelines: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        
        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Workflow Automation system initialized")
    
    def create_batch_generation_pipeline(
        self,
        motifs: List[str],
        candidates_per_motif: int = 50,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> List[BatchTask]:
        """
        Create batch generation pipeline for multiple motifs.
        
        Args:
            motifs: List of protein motifs to generate
            candidates_per_motif: Number of candidates per motif
            generation_params: Additional generation parameters
            
        Returns:
            List of batch tasks for execution
        """
        tasks = []
        base_params = generation_params or {}
        
        for i, motif in enumerate(motifs):
            task_id = f"gen_{i}_{motif}_{uuid.uuid4().hex[:8]}"
            task_params = {
                "motif": motif,
                "num_samples": candidates_per_motif,
                "client_id": f"batch_gen_{i}",
                **base_params
            }
            
            task = BatchTask(
                task_id=task_id,
                task_type="generation",
                parameters=task_params,
                priority=1
            )
            tasks.append(task)
        
        logger.info(f"Created batch generation pipeline with {len(tasks)} tasks")
        return tasks
    
    def create_design_and_rank_pipeline(
        self,
        motifs: List[str],
        candidates_per_motif: int = 50,
        target_pdb: Optional[str] = None,
        max_ranked_per_motif: int = 20,
        design_params: Optional[Dict[str, Any]] = None
    ) -> List[BatchTask]:
        """
        Create complete design and ranking pipeline.
        
        Args:
            motifs: List of protein motifs
            candidates_per_motif: Candidates to generate per motif
            target_pdb: Target PDB file path
            max_ranked_per_motif: Maximum ranked results per motif
            design_params: Additional design parameters
            
        Returns:
            List of batch tasks for execution
        """
        tasks = []
        base_params = design_params or {}
        
        for i, motif in enumerate(motifs):
            task_id = f"design_rank_{i}_{motif}_{uuid.uuid4().hex[:8]}"
            task_params = {
                "motif": motif,
                "num_candidates": candidates_per_motif,
                "target_pdb": target_pdb,
                "max_ranked": max_ranked_per_motif,
                "client_id": f"pipeline_{i}",
                **base_params
            }
            
            task = BatchTask(
                task_id=task_id,
                task_type="design_and_rank",
                parameters=task_params,
                priority=1
            )
            tasks.append(task)
        
        logger.info(f"Created design and rank pipeline with {len(tasks)} tasks")
        return tasks
    
    def create_optimization_pipeline(
        self,
        base_motif: str,
        parameter_grid: Dict[str, List[Any]],
        target_pdb: Optional[str] = None,
        candidates_per_config: int = 25
    ) -> List[BatchTask]:
        """
        Create parameter optimization pipeline.
        
        Args:
            base_motif: Base protein motif
            parameter_grid: Grid of parameters to optimize
            target_pdb: Target PDB file path
            candidates_per_config: Candidates per parameter configuration
            
        Returns:
            List of optimization tasks
        """
        tasks = []
        
        # Generate all parameter combinations
        from itertools import product
        
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        for i, param_combo in enumerate(product(*param_values)):
            param_dict = dict(zip(param_names, param_combo))
            
            task_id = f"opt_{i}_{uuid.uuid4().hex[:8]}"
            task_params = {
                "motif": base_motif,
                "num_candidates": candidates_per_config,
                "target_pdb": target_pdb,
                "max_ranked": 10,
                "client_id": f"optimization_{i}",
                **param_dict
            }
            
            task = BatchTask(
                task_id=task_id,
                task_type="design_and_rank",
                parameters=task_params,
                priority=2  # Higher priority for optimization
            )
            tasks.append(task)
        
        logger.info(f"Created optimization pipeline with {len(tasks)} parameter combinations")
        return tasks
    
    def create_comparative_study_pipeline(
        self,
        motif_groups: Dict[str, List[str]],
        candidates_per_motif: int = 50,
        target_pdbs: Optional[List[str]] = None
    ) -> List[BatchTask]:
        """
        Create comparative study pipeline across motif groups.
        
        Args:
            motif_groups: Dictionary of group_name -> motif_list
            candidates_per_motif: Candidates per motif
            target_pdbs: List of target PDB files
            
        Returns:
            List of comparative study tasks
        """
        tasks = []
        
        for group_name, motifs in motif_groups.items():
            for target_pdb in (target_pdbs or [None]):
                for i, motif in enumerate(motifs):
                    task_id = f"comp_{group_name}_{i}_{uuid.uuid4().hex[:8]}"
                    task_params = {
                        "motif": motif,
                        "num_candidates": candidates_per_motif,
                        "target_pdb": target_pdb,
                        "max_ranked": 15,
                        "client_id": f"comparative_{group_name}_{i}",
                        "study_group": group_name,
                        "target_name": Path(target_pdb).stem if target_pdb else "none"
                    }
                    
                    task = BatchTask(
                        task_id=task_id,
                        task_type="design_and_rank",
                        parameters=task_params,
                        priority=1
                    )
                    tasks.append(task)
        
        logger.info(f"Created comparative study with {len(tasks)} tasks across {len(motif_groups)} groups")
        return tasks
    
    def execute_pipeline(
        self,
        tasks: List[BatchTask],
        pipeline_name: str = "automated_pipeline"
    ) -> PipelineResult:
        """
        Execute a pipeline of batch tasks.
        
        Args:
            tasks: List of tasks to execute
            pipeline_name: Name for the pipeline
            
        Returns:
            Pipeline execution results
        """
        pipeline_id = f"{pipeline_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        logger.info(f"Starting pipeline execution: {pipeline_id} with {len(tasks)} tasks")
        
        # Initialize result tracking
        result = PipelineResult(
            pipeline_id=pipeline_id,
            success=False,
            total_tasks=len(tasks),
            completed_tasks=0,
            failed_tasks=0,
            execution_time=0.0
        )
        
        # Track pipeline
        self.active_pipelines[pipeline_id] = {
            "start_time": start_time,
            "status": "running",
            "total_tasks": len(tasks),
            "completed_tasks": 0
        }
        
        try:
            # Sort tasks by priority
            sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
            
            # Execute tasks with dependency resolution
            completed_tasks = self._execute_tasks_with_dependencies(sorted_tasks, result)
            
            # Calculate final metrics
            result.execution_time = time.time() - start_time
            result.completed_tasks = len([t for t in completed_tasks if t.status == "completed"])
            result.failed_tasks = len([t for t in completed_tasks if t.status == "failed"])
            result.success = result.failed_tasks == 0
            
            # Store task results
            result.task_results = {task.task_id: task for task in completed_tasks}
            
            # Calculate performance metrics
            result.performance_metrics = self._calculate_pipeline_metrics(completed_tasks)
            
            # Save results if configured
            if self.config.save_intermediate_results:
                self._save_pipeline_results(result)
            
            # Perform optimization analysis if enabled
            if self.config.enable_optimization:
                optimization_results = self._analyze_optimization_results(completed_tasks)
                result.performance_metrics["optimization"] = optimization_results
            
            logger.info(f"Pipeline {pipeline_id} completed: {result.completed_tasks}/{result.total_tasks} successful")
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            result.errors.append(str(e))
            result.execution_time = time.time() - start_time
        
        finally:
            # Update pipeline tracking
            if pipeline_id in self.active_pipelines:
                self.active_pipelines[pipeline_id]["status"] = "completed" if result.success else "failed"
                self.active_pipelines[pipeline_id]["end_time"] = time.time()
        
        return result
    
    def _execute_tasks_with_dependencies(self, tasks: List[BatchTask], result: PipelineResult) -> List[BatchTask]:
        """
        Execute tasks respecting dependency order.
        
        Args:
            tasks: List of tasks to execute
            result: Pipeline result to update
            
        Returns:
            List of completed tasks
        """
        completed_tasks = []
        remaining_tasks = tasks.copy()
        task_map = {task.task_id: task for task in tasks}
        
        # Future objects for concurrent execution
        futures = {}
        
        while remaining_tasks or futures:
            # Find tasks ready to execute (no pending dependencies)
            ready_tasks = []
            for task in remaining_tasks.copy():
                if all(dep_id in [t.task_id for t in completed_tasks if t.status == "completed"]
                       for dep_id in task.dependencies):
                    ready_tasks.append(task)
                    remaining_tasks.remove(task)
            
            # Submit ready tasks for execution
            for task in ready_tasks:
                if len(futures) < self.config.max_concurrent_tasks:
                    future = self.executor.submit(self._execute_single_task, task)
                    futures[future] = task
            
            # Check for completed tasks
            if futures:
                for future in as_completed(futures.keys(), timeout=1):
                    task = futures.pop(future)
                    try:
                        completed_task = future.result()
                        completed_tasks.append(completed_task)
                        
                        if completed_task.status == "completed":
                            result.completed_tasks += 1
                        else:
                            result.failed_tasks += 1
                            result.errors.append(f"Task {task.task_id} failed: {task.error}")
                        
                        # Update pipeline tracking
                        if result.pipeline_id in self.active_pipelines:
                            self.active_pipelines[result.pipeline_id]["completed_tasks"] = len(completed_tasks)
                        
                    except Exception as e:
                        logger.error(f"Task execution error: {e}")
                        task.status = "failed"
                        task.error = str(e)
                        completed_tasks.append(task)
                        result.failed_tasks += 1
                        result.errors.append(str(e))
            
            # Check for timeout
            if time.time() - result.timestamp > self.config.timeout_seconds:
                logger.warning(f"Pipeline timeout reached, stopping execution")
                result.warnings.append("Pipeline execution timed out")
                break
        
        return completed_tasks
    
    def _execute_single_task(self, task: BatchTask) -> BatchTask:
        """
        Execute a single batch task with retry logic.
        
        Args:
            task: Task to execute
            
        Returns:
            Updated task with results
        """
        task.start_time = time.time()
        task.status = "running"
        
        logger.info(f"Executing task {task.task_id} of type {task.task_type}")
        
        for attempt in range(self.config.retry_attempts):
            try:
                if task.task_type == "generation":
                    result = self.integration_manager.diffuser.generate(**task.parameters)
                    task.result = WorkflowResult(
                        success=True,
                        workflow_id=task.task_id,
                        generation_results=result
                    )
                
                elif task.task_type == "evaluation":
                    result = self.integration_manager.evaluate_sequences(**task.parameters)
                    task.result = result
                
                elif task.task_type == "design_and_rank":
                    result = self.integration_manager.design_and_rank_proteins(**task.parameters)
                    task.result = result
                
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                task.status = "completed"
                task.end_time = time.time()
                
                logger.info(f"Task {task.task_id} completed successfully in {task.execution_time:.2f}s")
                break
                
            except Exception as e:
                logger.warning(f"Task {task.task_id} attempt {attempt + 1} failed: {e}")
                
                if attempt == self.config.retry_attempts - 1:
                    task.status = "failed"
                    task.error = str(e)
                    task.end_time = time.time()
                else:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        return task
    
    def _calculate_pipeline_metrics(self, tasks: List[BatchTask]) -> Dict[str, Any]:
        """
        Calculate performance metrics for completed tasks.
        
        Args:
            tasks: List of completed tasks
            
        Returns:
            Performance metrics dictionary
        """
        completed_tasks = [t for t in tasks if t.status == "completed"]
        failed_tasks = [t for t in tasks if t.status == "failed"]
        
        if not completed_tasks:
            return {"error": "No completed tasks to analyze"}
        
        execution_times = [t.execution_time for t in completed_tasks]
        
        # Basic metrics
        metrics = {
            "total_tasks": len(tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(completed_tasks) / len(tasks),
            "average_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "total_execution_time": sum(execution_times)
        }
        
        # Task type breakdown
        task_types = {}
        for task in completed_tasks:
            task_type = task.task_type
            if task_type not in task_types:
                task_types[task_type] = {"count": 0, "avg_time": 0, "total_time": 0}
            
            task_types[task_type]["count"] += 1
            task_types[task_type]["total_time"] += task.execution_time
        
        for task_type in task_types:
            task_types[task_type]["avg_time"] = task_types[task_type]["total_time"] / task_types[task_type]["count"]
        
        metrics["task_type_breakdown"] = task_types
        
        # Results analysis
        total_sequences_generated = 0
        total_sequences_ranked = 0
        
        for task in completed_tasks:
            if task.result and hasattr(task.result, 'generation_results'):
                total_sequences_generated += len(task.result.generation_results or [])
            if task.result and hasattr(task.result, 'ranking_results'):
                total_sequences_ranked += len(task.result.ranking_results or [])
        
        metrics["output_metrics"] = {
            "total_sequences_generated": total_sequences_generated,
            "total_sequences_ranked": total_sequences_ranked,
            "sequences_per_task": total_sequences_generated / len(completed_tasks) if completed_tasks else 0
        }
        
        return metrics
    
    def _analyze_optimization_results(self, tasks: List[BatchTask]) -> Dict[str, Any]:
        """
        Analyze optimization results to find best parameters.
        
        Args:
            tasks: List of completed optimization tasks
            
        Returns:
            Optimization analysis results
        """
        optimization_tasks = [t for t in tasks if t.task_type == "design_and_rank" and t.status == "completed"]
        
        if not optimization_tasks:
            return {"error": "No optimization tasks to analyze"}
        
        # Extract performance data
        performance_data = []
        for task in optimization_tasks:
            if task.result and task.result.ranking_results:
                # Get best composite score from ranking results
                best_score = max(r.get("composite_score", 0) for r in task.result.ranking_results)
                avg_score = sum(r.get("composite_score", 0) for r in task.result.ranking_results) / len(task.result.ranking_results)
                
                performance_data.append({
                    "task_id": task.task_id,
                    "parameters": task.parameters,
                    "best_score": best_score,
                    "average_score": avg_score,
                    "num_results": len(task.result.ranking_results),
                    "execution_time": task.execution_time
                })
        
        if not performance_data:
            return {"error": "No performance data available"}
        
        # Find best configuration
        best_config = max(performance_data, key=lambda x: x["best_score"])
        
        # Calculate parameter correlations (simplified)
        param_performance = {}
        for data in performance_data:
            for param, value in data["parameters"].items():
                if param not in ["motif", "client_id", "num_candidates", "max_ranked"]:
                    if param not in param_performance:
                        param_performance[param] = []
                    param_performance[param].append((value, data["best_score"]))
        
        return {
            "total_configurations": len(performance_data),
            "best_configuration": best_config,
            "average_best_score": sum(d["best_score"] for d in performance_data) / len(performance_data),
            "parameter_analysis": param_performance,
            "optimization_efficiency": best_config["best_score"] / best_config["execution_time"]
        }
    
    def _save_pipeline_results(self, result: PipelineResult):
        """
        Save pipeline results to disk.
        
        Args:
            result: Pipeline result to save
        """
        try:
            output_dir = Path(self.config.output_directory) / result.pipeline_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main result summary
            summary_file = output_dir / "pipeline_summary.json"
            summary_data = {
                "pipeline_id": result.pipeline_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "total_tasks": result.total_tasks,
                "completed_tasks": result.completed_tasks,
                "failed_tasks": result.failed_tasks,
                "performance_metrics": result.performance_metrics,
                "errors": result.errors,
                "warnings": result.warnings,
                "timestamp": result.timestamp
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            # Save detailed task results
            for task_id, task in result.task_results.items():
                task_file = output_dir / f"task_{task_id}.json"
                task_data = {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status,
                    "parameters": task.parameters,
                    "execution_time": task.execution_time,
                    "error": task.error
                }
                
                # Add result data if successful
                if task.result:
                    task_data["result_summary"] = {
                        "success": task.result.success,
                        "generation_count": len(task.result.generation_results),
                        "ranking_count": len(task.result.ranking_results),
                        "execution_time": task.result.execution_time
                    }
                    
                    # Save detailed results separately
                    if task.result.generation_results:
                        gen_file = output_dir / f"task_{task_id}_generation.json"
                        with open(gen_file, 'w') as f:
                            json.dump(task.result.generation_results, f, indent=2, default=str)
                    
                    if task.result.ranking_results:
                        rank_file = output_dir / f"task_{task_id}_ranking.json"
                        with open(rank_file, 'w') as f:
                            json.dump(task.result.ranking_results, f, indent=2, default=str)
                
                with open(task_file, 'w') as f:
                    json.dump(task_data, f, indent=2, default=str)
            
            # Save CSV summary for easy analysis
            csv_file = output_dir / "task_summary.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['task_id', 'task_type', 'status', 'execution_time', 'error'])
                
                for task in result.task_results.values():
                    writer.writerow([
                        task.task_id,
                        task.task_type,
                        task.status,
                        task.execution_time,
                        task.error or ''
                    ])
            
            logger.info(f"Pipeline results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running pipeline."""
        return self.active_pipelines.get(pipeline_id)
    
    def list_active_pipelines(self) -> List[Dict[str, Any]]:
        """List all active pipelines."""
        return list(self.active_pipelines.values())
    
    def shutdown(self):
        """Shutdown the automation system."""
        logger.info("Shutting down workflow automation system")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Shutdown integration manager
        self.integration_manager.shutdown()
        
        logger.info("Workflow automation shutdown complete")


# Convenience functions for creating common pipelines
def create_motif_screening_pipeline(
    motifs: List[str],
    target_pdb: str,
    candidates_per_motif: int = 100,
    config: Optional[AutomationConfig] = None
) -> WorkflowAutomation:
    """
    Create workflow automation for motif screening.
    
    Args:
        motifs: List of motifs to screen
        target_pdb: Target PDB file
        candidates_per_motif: Candidates per motif
        config: Automation configuration
        
    Returns:
        Configured WorkflowAutomation instance
    """
    automation = WorkflowAutomation(config)
    return automation


def create_parameter_optimization(
    base_motif: str,
    parameter_ranges: Dict[str, List[Any]],
    target_pdb: str,
    config: Optional[AutomationConfig] = None
) -> WorkflowAutomation:
    """
    Create workflow automation for parameter optimization.
    
    Args:
        base_motif: Base motif for optimization
        parameter_ranges: Dictionary of parameters and their ranges
        target_pdb: Target PDB file
        config: Automation configuration
        
    Returns:
        Configured WorkflowAutomation instance
    """
    automation = WorkflowAutomation(config)
    return automation
