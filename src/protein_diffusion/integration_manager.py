"""
Integration Manager - Unified interface for all protein diffusion components.

This module provides a centralized integration layer that coordinates
between diffusion, ranking, structure prediction, and other subsystems
to provide seamless end-to-end protein design workflows.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from .diffuser import ProteinDiffuser, ProteinDiffuserConfig
    from .ranker import AffinityRanker, AffinityRankerConfig
    from .monitoring import SystemMonitor
    # Note: Using simplified configs since advanced components may not be available
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some components not available: {e}")
    COMPONENTS_AVAILABLE = False
    ProteinDiffuser = None
    ProteinDiffuserConfig = None
    AffinityRanker = None 
    AffinityRankerConfig = None
    SystemMonitor = None

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for the integration manager."""
    # Component configurations  
    diffuser_config: Optional[Dict[str, Any]] = None
    ranker_config: Optional[Dict[str, Any]] = None
    monitoring_config: Optional[Dict[str, Any]] = None
    cache_config: Optional[Dict[str, Any]] = None
    security_config: Optional[Dict[str, Any]] = None
    
    # Integration settings
    enable_validation: bool = True
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_security: bool = True
    
    # Workflow settings
    auto_rank_generated: bool = True
    auto_validate_inputs: bool = True
    save_intermediate_results: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # 5 minutes
    
    # Output settings
    output_dir: str = "./integration_output"
    log_level: str = "INFO"
    
    def __post_init__(self):
        if COMPONENTS_AVAILABLE:
            if self.diffuser_config is None:
                self.diffuser_config = {}
            if self.ranker_config is None:
                self.ranker_config = {}
            if self.monitoring_config is None:
                self.monitoring_config = {}
            if self.cache_config is None:
                self.cache_config = {}
            if self.security_config is None:
                self.security_config = {}


@dataclass
class WorkflowResult:
    """Result from an integrated workflow."""
    success: bool
    workflow_id: str
    generation_results: List[Dict[str, Any]] = field(default_factory=list)
    ranking_results: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class IntegrationManager:
    """
    Unified integration manager for protein diffusion workflows.
    
    This class coordinates between all subsystems to provide seamless
    end-to-end protein design workflows with monitoring, caching,
    validation, and security.
    
    Example:
        >>> manager = IntegrationManager()
        >>> result = manager.design_and_rank_proteins(
        ...     motif="HELIX_SHEET_HELIX",
        ...     num_candidates=100,
        ...     target_pdb="spike_protein.pdb"
        ... )
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        if config is None:
            config = IntegrationConfig()
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Track active workflows
        self.active_workflows: Dict[str, Dict] = {}
        
        logger.info("Integration Manager initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components."""
        if not COMPONENTS_AVAILABLE:
            logger.warning("Components not available - running in limited mode")
            self.diffuser = None
            self.ranker = None
            self.monitor = None
            self.cache_manager = None
            self.security_manager = None
            self.validation_manager = None
            return
        
        try:
            # Initialize diffuser if available
            if ProteinDiffuser and ProteinDiffuserConfig:
                try:
                    diffuser_config = ProteinDiffuserConfig() if ProteinDiffuserConfig else None
                    self.diffuser = ProteinDiffuser(diffuser_config)
                    logger.info("Protein diffuser initialized")
                except Exception as e:
                    logger.warning(f"Protein diffuser initialization failed: {e}")
                    self.diffuser = None
            else:
                self.diffuser = None
                logger.warning("Protein diffuser not available")
            
            # Initialize ranker if available
            if AffinityRanker and AffinityRankerConfig:
                try:
                    ranker_config = AffinityRankerConfig() if AffinityRankerConfig else None
                    self.ranker = AffinityRanker(ranker_config)
                    logger.info("Affinity ranker initialized")
                except Exception as e:
                    logger.warning(f"Affinity ranker initialization failed: {e}")
                    self.ranker = None
            else:
                self.ranker = None
                logger.warning("Affinity ranker not available")
            
            # Initialize monitoring if available
            if self.config.enable_monitoring and SystemMonitor:
                self.monitor = SystemMonitor()
                self.monitor.start_monitoring()
                logger.info("System monitoring enabled")
            else:
                self.monitor = None
                if self.config.enable_monitoring:
                    logger.warning("System monitoring requested but not available")
            
            # Initialize optional components (not available yet)
            self.cache_manager = None
            self.security_manager = None 
            self.validation_manager = None
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise RuntimeError(f"Integration manager initialization failed: {e}")
    
    def design_and_rank_proteins(
        self,
        motif: Optional[str] = None,
        num_candidates: int = 50,
        target_pdb: Optional[str] = None,
        max_ranked: int = 20,
        client_id: str = "default",
        **generation_kwargs
    ) -> WorkflowResult:
        """
        Complete protein design and ranking workflow.
        
        Args:
            motif: Target protein motif
            num_candidates: Number of candidates to generate
            target_pdb: Target PDB for binding evaluation
            max_ranked: Maximum number of ranked results
            client_id: Client identifier for security/monitoring
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Complete workflow results
        """
        workflow_id = f"design_rank_{int(time.time())}_{client_id}"
        start_time = time.time()
        
        logger.info(f"Starting design and rank workflow {workflow_id}")
        
        result = WorkflowResult(
            success=False,
            workflow_id=workflow_id
        )
        
        try:
            # Security check
            if self.security_manager and not self._security_check(client_id):
                result.errors.append("Security check failed")
                return result
            
            # Input validation
            if self.validation_manager and self.config.auto_validate_inputs:
                validation_result = self._validate_workflow_inputs(
                    motif, num_candidates, target_pdb, client_id
                )
                result.validation_results = validation_result
                
                if not validation_result.get("valid", True):
                    result.errors.extend(validation_result.get("errors", []))
                    return result
            
            # Track workflow
            self.active_workflows[workflow_id] = {
                "start_time": start_time,
                "client_id": client_id,
                "status": "running"
            }
            
            # Step 1: Generate protein candidates
            if self.diffuser:
                logger.info(f"Generating {num_candidates} protein candidates")
                
                # Check cache first
                cache_key = None
                if self.cache_manager:
                    cache_key = self._generate_cache_key("generation", {
                        "motif": motif,
                        "num_candidates": num_candidates,
                        **generation_kwargs
                    })
                    cached_results = self.cache_manager.get(cache_key)
                    if cached_results:
                        logger.info("Using cached generation results")
                        result.generation_results = cached_results
                    
                if not result.generation_results:
                    generation_results = self.diffuser.generate(
                        motif=motif,
                        num_samples=num_candidates,
                        client_id=client_id,
                        **generation_kwargs
                    )
                    result.generation_results = generation_results
                    
                    # Cache results
                    if self.cache_manager and cache_key:
                        self.cache_manager.set(cache_key, generation_results)
                
                logger.info(f"Generated {len(result.generation_results)} candidates")
            else:
                result.errors.append("Diffuser not available")
                return result
            
            # Step 2: Rank candidates (if requested and successful generation)
            if self.config.auto_rank_generated and result.generation_results and self.ranker:
                logger.info("Ranking generated candidates")
                
                # Extract sequences for ranking
                sequences = [r.get("sequence", "") for r in result.generation_results]
                sequences = [s for s in sequences if s]  # Remove empty sequences
                
                if sequences:
                    ranking_results = self.ranker.rank(
                        sequences=sequences,
                        target_pdb=target_pdb,
                        return_detailed=True
                    )
                    
                    # Limit results
                    if len(ranking_results) > max_ranked:
                        ranking_results = ranking_results[:max_ranked]
                    
                    result.ranking_results = ranking_results
                    logger.info(f"Ranked top {len(ranking_results)} candidates")
                else:
                    result.warnings.append("No valid sequences to rank")
            
            # Step 3: Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_workflow_results(workflow_id, result)
            
            # Calculate performance metrics
            end_time = time.time()
            result.execution_time = end_time - start_time
            result.performance_metrics = {
                "execution_time": result.execution_time,
                "candidates_generated": len(result.generation_results),
                "candidates_ranked": len(result.ranking_results),
                "generation_rate": len(result.generation_results) / result.execution_time,
            }
            
            # Add monitoring data
            if self.monitor:
                result.performance_metrics.update(self.monitor.get_current_metrics())
            
            result.success = True
            logger.info(f"Workflow {workflow_id} completed successfully in {result.execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            result.errors.append(str(e))
            result.execution_time = time.time() - start_time
        
        finally:
            # Cleanup workflow tracking
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = "completed" if result.success else "failed"
                self.active_workflows[workflow_id]["end_time"] = time.time()
        
        return result
    
    def evaluate_sequences(
        self,
        sequences: List[str],
        target_pdb: Optional[str] = None,
        include_structure: bool = True,
        include_binding: bool = True,
        client_id: str = "default"
    ) -> WorkflowResult:
        """
        Evaluate existing protein sequences.
        
        Args:
            sequences: List of protein sequences to evaluate
            target_pdb: Target PDB for binding evaluation
            include_structure: Include structure prediction
            include_binding: Include binding affinity prediction
            client_id: Client identifier
            
        Returns:
            Evaluation workflow results
        """
        workflow_id = f"evaluate_{int(time.time())}_{client_id}"
        start_time = time.time()
        
        logger.info(f"Starting evaluation workflow {workflow_id} for {len(sequences)} sequences")
        
        result = WorkflowResult(
            success=False,
            workflow_id=workflow_id
        )
        
        try:
            # Security check
            if self.security_manager and not self._security_check(client_id):
                result.errors.append("Security check failed")
                return result
            
            # Input validation
            if self.validation_manager:
                for i, seq in enumerate(sequences):
                    if not seq or not isinstance(seq, str):
                        result.errors.append(f"Invalid sequence at index {i}")
                        return result
            
            # Rank/evaluate sequences
            if self.ranker:
                ranking_results = self.ranker.rank(
                    sequences=sequences,
                    target_pdb=target_pdb,
                    return_detailed=True
                )
                result.ranking_results = ranking_results
            
            # Additional structure evaluation if diffuser available
            if self.diffuser and include_structure:
                structure_results = self.diffuser.evaluate_sequences(
                    sequences=sequences,
                    target_pdb=target_pdb,
                    compute_structure=include_structure,
                    compute_binding=include_binding
                )
                result.generation_results = structure_results
            
            result.execution_time = time.time() - start_time
            result.success = True
            
            logger.info(f"Evaluation workflow {workflow_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation workflow {workflow_id} failed: {e}")
            result.errors.append(str(e))
            result.execution_time = time.time() - start_time
        
        return result
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            System health information
        """
        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {},
            "active_workflows": len(self.active_workflows),
            "performance_metrics": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check diffuser health
            if self.diffuser:
                diffuser_health = self.diffuser.health_check()
                health_status["components"]["diffuser"] = diffuser_health
                if diffuser_health.get("overall_status") != "healthy":
                    health_status["warnings"].extend(diffuser_health.get("warnings", []))
                    health_status["errors"].extend(diffuser_health.get("errors", []))
            
            # Check ranker health (basic check)
            if self.ranker:
                health_status["components"]["ranker"] = {"status": "healthy", "loaded": True}
            
            # Check monitor health
            if self.monitor:
                monitor_health = self.monitor.get_health_status()
                health_status["components"]["monitor"] = monitor_health
                health_status["performance_metrics"] = self.monitor.get_current_metrics()
            
            # Check cache health
            if self.cache_manager:
                cache_stats = self.cache_manager.get_stats()
                health_status["components"]["cache"] = {"status": "healthy", "stats": cache_stats}
            
            # Determine overall status
            component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
            if any(status in ["error", "unhealthy"] for status in component_statuses):
                health_status["overall_status"] = "unhealthy"
            elif any(status in ["warning", "degraded"] for status in component_statuses) or health_status["warnings"]:
                health_status["overall_status"] = "degraded"
            
        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["errors"].append(f"Health check failed: {e}")
            logger.error(f"System health check failed: {e}")
        
        return health_status
    
    def _security_check(self, client_id: str) -> bool:
        """Perform security validation."""
        if not self.security_manager:
            return True
        
        try:
            return self.security_manager.validate_client(client_id)
        except Exception as e:
            logger.warning(f"Security check failed for {client_id}: {e}")
            return False
    
    def _validate_workflow_inputs(self, motif, num_candidates, target_pdb, client_id) -> Dict[str, Any]:
        """Validate workflow inputs."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate parameters
            if num_candidates <= 0 or num_candidates > 1000:
                validation_result["errors"].append(f"Invalid num_candidates: {num_candidates}")
                validation_result["valid"] = False
            
            if motif and not isinstance(motif, str):
                validation_result["errors"].append(f"Invalid motif type: {type(motif)}")
                validation_result["valid"] = False
            
            if target_pdb and not Path(target_pdb).exists():
                validation_result["warnings"].append(f"Target PDB file not found: {target_pdb}")
            
            if not isinstance(client_id, str):
                validation_result["errors"].append(f"Invalid client_id type: {type(client_id)}")
                validation_result["valid"] = False
                
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
            validation_result["valid"] = False
        
        return validation_result
    
    def _generate_cache_key(self, operation: str, params: Dict) -> str:
        """Generate cache key for operation and parameters."""
        import hashlib
        
        key_data = {"operation": operation, **params}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _save_workflow_results(self, workflow_id: str, result: WorkflowResult):
        """Save workflow results to disk."""
        try:
            output_file = Path(self.config.output_dir) / f"workflow_{workflow_id}.json"
            
            # Prepare serializable data
            serializable_data = {
                "workflow_id": result.workflow_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp,
                "generation_count": len(result.generation_results),
                "ranking_count": len(result.ranking_results),
                "performance_metrics": result.performance_metrics,
                "errors": result.errors,
                "warnings": result.warnings,
            }
            
            # Save detailed results separately for large datasets
            if result.generation_results:
                gen_file = Path(self.config.output_dir) / f"generation_{workflow_id}.json"
                with open(gen_file, 'w') as f:
                    json.dump(result.generation_results, f, indent=2, default=str)
                serializable_data["generation_file"] = str(gen_file)
            
            if result.ranking_results:
                rank_file = Path(self.config.output_dir) / f"ranking_{workflow_id}.json"
                with open(rank_file, 'w') as f:
                    json.dump(result.ranking_results, f, indent=2, default=str)
                serializable_data["ranking_file"] = str(rank_file)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Workflow results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save workflow results: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active workflow."""
        return self.active_workflows.get(workflow_id)
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows."""
        return list(self.active_workflows.values())
    
    def shutdown(self):
        """Gracefully shutdown the integration manager."""
        logger.info("Shutting down integration manager")
        
        try:
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
            
            # Clear active workflows
            self.active_workflows.clear()
            
            # Cleanup cache
            if self.cache_manager:
                self.cache_manager.cleanup()
            
            logger.info("Integration manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
