"""
Comprehensive Workflow Manager for Protein Diffusion Design Lab.

This module provides high-level orchestration of the complete protein design
workflow from generation through validation and ranking.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

try:
    from .diffuser import ProteinDiffuser, ProteinDiffuserConfig
    from .ranker import AffinityRanker, AffinityRankerConfig
    from .validation import ValidationManager
    from .security_framework import SecurityManager
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for the complete protein design workflow."""
    # Generation settings
    generation_config: Optional[ProteinDiffuserConfig] = None
    
    # Ranking settings
    ranking_config: Optional[AffinityRankerConfig] = None
    
    # Workflow control
    max_concurrent_tasks: int = 4
    enable_validation: bool = True
    enable_security: bool = True
    save_intermediate_results: bool = True
    
    # Output settings
    output_dir: str = "./workflow_output"
    experiment_name: str = "protein_design_experiment"
    
    # Quality gates
    min_generation_success_rate: float = 0.8
    min_ranking_success_rate: float = 0.9
    
    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = ProteinDiffuserConfig()
        if self.ranking_config is None:
            self.ranking_config = AffinityRankerConfig()


@dataclass
class WorkflowResult:
    """Results from a complete workflow execution."""
    experiment_id: str
    timestamp: float
    config: WorkflowConfig
    
    # Generation results
    generated_sequences: List[Dict[str, Any]]
    generation_stats: Dict[str, Any]
    
    # Ranking results
    ranked_sequences: List[Dict[str, Any]]
    ranking_stats: Dict[str, Any]
    
    # Overall metrics
    success: bool
    total_runtime: float
    error_messages: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ProteinDesignWorkflow:
    """
    Complete protein design workflow orchestrator.
    
    This class manages the entire pipeline from sequence generation through
    ranking and validation, providing a single interface for complex
    protein design tasks.
    
    Example:
        >>> workflow = ProteinDesignWorkflow()
        >>> result = workflow.run_complete_pipeline(
        ...     motif="HELIX_SHEET_HELIX",
        ...     num_sequences=100,
        ...     target_pdb="target.pdb"
        ... )
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.diffuser = None
        self.ranker = None
        self.validation_manager = None
        self.security_manager = None
        
        # Workflow state
        self.current_experiment_id = None
        self.workflow_start_time = None
        
        logger.info(f"Workflow manager initialized with output dir: {self.output_dir}")
    
    def initialize_components(self):
        """Initialize all workflow components."""
        if not MODULES_AVAILABLE:
            raise RuntimeError("Required modules not available")
        
        try:
            # Initialize diffuser
            self.diffuser = ProteinDiffuser(self.config.generation_config)
            logger.info("Protein diffuser initialized")
            
            # Initialize ranker
            self.ranker = AffinityRanker(self.config.ranking_config)
            logger.info("Affinity ranker initialized")
            
            # Initialize validation if enabled
            if self.config.enable_validation:
                self.validation_manager = ValidationManager()
                logger.info("Validation manager initialized")
            
            # Initialize security if enabled
            if self.config.enable_security:
                self.security_manager = SecurityManager()
                logger.info("Security manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_complete_pipeline(
        self,
        motif: Optional[str] = None,
        num_sequences: int = 100,
        target_pdb: Optional[str] = None,
        experiment_name: Optional[str] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Run the complete protein design pipeline.
        
        Args:
            motif: Target protein motif
            num_sequences: Number of sequences to generate
            target_pdb: Target PDB for binding evaluation
            experiment_name: Name for this experiment
            **kwargs: Additional parameters
            
        Returns:
            Complete workflow results
        """
        # Setup experiment
        if experiment_name:
            self.config.experiment_name = experiment_name
        
        experiment_id = self._generate_experiment_id()
        self.current_experiment_id = experiment_id
        self.workflow_start_time = time.time()
        
        logger.info(f"Starting protein design pipeline: {experiment_id}")
        
        error_messages = []
        
        try:
            # Initialize components if not already done
            if self.diffuser is None:
                self.initialize_components()
            
            # Phase 1: Security and Validation
            if self.config.enable_security and self.security_manager:
                security_result = self._run_security_checks({
                    'motif': motif,
                    'num_sequences': num_sequences,
                    'target_pdb': target_pdb,
                    **kwargs
                })
                if not security_result['passed']:
                    raise RuntimeError(f"Security checks failed: {security_result['errors']}")
            
            # Phase 2: Sequence Generation
            logger.info("Phase 2: Generating protein sequences")
            generation_result = self._run_generation_phase(motif, num_sequences, **kwargs)
            
            if not generation_result['success']:
                error_messages.extend(generation_result['errors'])
                if len(generation_result['sequences']) < num_sequences * self.config.min_generation_success_rate:
                    raise RuntimeError("Generation success rate below threshold")
            
            # Phase 3: Sequence Ranking and Evaluation
            logger.info("Phase 3: Ranking and evaluating sequences")
            ranking_result = self._run_ranking_phase(
                generation_result['sequences'], 
                target_pdb, 
                **kwargs
            )
            
            if not ranking_result['success']:
                error_messages.extend(ranking_result['errors'])
                if len(ranking_result['ranked_sequences']) < len(generation_result['sequences']) * self.config.min_ranking_success_rate:
                    raise RuntimeError("Ranking success rate below threshold")
            
            # Phase 4: Results Analysis and Export
            logger.info("Phase 4: Analyzing results and exporting")
            analysis_result = self._run_analysis_phase(
                generation_result, 
                ranking_result
            )
            
            # Create final result
            total_runtime = time.time() - self.workflow_start_time
            
            workflow_result = WorkflowResult(
                experiment_id=experiment_id,
                timestamp=self.workflow_start_time,
                config=self.config,
                generated_sequences=generation_result['sequences'],
                generation_stats=generation_result['stats'],
                ranked_sequences=ranking_result['ranked_sequences'],
                ranking_stats=ranking_result['stats'],
                success=True,
                total_runtime=total_runtime,
                error_messages=error_messages
            )
            
            # Save results
            if self.config.save_intermediate_results:
                self._save_workflow_result(workflow_result)
            
            logger.info(f"Workflow completed successfully in {total_runtime:.2f}s")
            return workflow_result
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            error_messages.append(str(e))
            
            # Create error result
            total_runtime = time.time() - self.workflow_start_time
            
            return WorkflowResult(
                experiment_id=experiment_id,
                timestamp=self.workflow_start_time,
                config=self.config,
                generated_sequences=[],
                generation_stats={},
                ranked_sequences=[],
                ranking_stats={},
                success=False,
                total_runtime=total_runtime,
                error_messages=error_messages
            )
    
    def _run_security_checks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run security validation on input parameters."""
        try:
            if self.security_manager:
                result = self.security_manager.validate_workflow_parameters(params)
                return {'passed': True, 'result': result, 'errors': []}
            else:
                return {'passed': True, 'result': {}, 'errors': []}
        except Exception as e:
            return {'passed': False, 'result': {}, 'errors': [str(e)]}
    
    def _run_generation_phase(
        self, 
        motif: Optional[str], 
        num_sequences: int, 
        **kwargs
    ) -> Dict[str, Any]:
        """Run the sequence generation phase."""
        start_time = time.time()
        
        try:
            # Generate sequences
            sequences = self.diffuser.generate(
                motif=motif,
                num_samples=num_sequences,
                **kwargs
            )
            
            # Filter successful generations
            successful_sequences = [s for s in sequences if not s.get('error')]
            
            # Calculate statistics
            stats = {
                'total_requested': num_sequences,
                'total_generated': len(sequences),
                'successful_generations': len(successful_sequences),
                'success_rate': len(successful_sequences) / num_sequences if num_sequences > 0 else 0,
                'generation_time': time.time() - start_time,
                'avg_sequence_length': sum(len(s.get('sequence', '')) for s in successful_sequences) / len(successful_sequences) if successful_sequences else 0,
                'avg_confidence': sum(s.get('confidence', 0) for s in successful_sequences) / len(successful_sequences) if successful_sequences else 0,
            }
            
            logger.info(f"Generation phase completed: {stats['successful_generations']}/{stats['total_requested']} successful")
            
            return {
                'success': stats['success_rate'] >= self.config.min_generation_success_rate,
                'sequences': successful_sequences,
                'stats': stats,
                'errors': [s.get('error', '') for s in sequences if s.get('error')]
            }
            
        except Exception as e:
            logger.error(f"Generation phase failed: {e}")
            return {
                'success': False,
                'sequences': [],
                'stats': {'generation_time': time.time() - start_time},
                'errors': [str(e)]
            }
    
    def _run_ranking_phase(
        self, 
        sequences: List[Dict[str, Any]], 
        target_pdb: Optional[str], 
        **kwargs
    ) -> Dict[str, Any]:
        """Run the sequence ranking and evaluation phase."""
        start_time = time.time()
        
        try:
            # Extract sequence strings
            sequence_strings = [s['sequence'] for s in sequences]
            
            # Rank sequences
            ranked_results = self.ranker.rank(
                sequences=sequence_strings,
                target_pdb=target_pdb,
                return_detailed=True
            )
            
            # Calculate ranking statistics
            ranking_stats = self.ranker.get_ranking_statistics(ranked_results)
            ranking_stats['ranking_time'] = time.time() - start_time
            ranking_stats['total_ranked'] = len(ranked_results)
            
            logger.info(f"Ranking phase completed: {len(ranked_results)} sequences ranked")
            
            return {
                'success': True,
                'ranked_sequences': ranked_results,
                'stats': ranking_stats,
                'errors': []
            }
            
        except Exception as e:
            logger.error(f"Ranking phase failed: {e}")
            return {
                'success': False,
                'ranked_sequences': [],
                'stats': {'ranking_time': time.time() - start_time},
                'errors': [str(e)]
            }
    
    def _run_analysis_phase(
        self, 
        generation_result: Dict[str, Any], 
        ranking_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run results analysis and generate insights."""
        try:
            analysis = {
                'top_candidates': ranking_result['ranked_sequences'][:10],
                'diversity_analysis': self._analyze_diversity(ranking_result['ranked_sequences']),
                'quality_distribution': self._analyze_quality_distribution(ranking_result['ranked_sequences']),
                'performance_metrics': self._calculate_performance_metrics(generation_result, ranking_result)
            }
            
            logger.info("Analysis phase completed")
            return {'success': True, 'analysis': analysis}
            
        except Exception as e:
            logger.error(f"Analysis phase failed: {e}")
            return {'success': False, 'analysis': {}, 'errors': [str(e)]}
    
    def _analyze_diversity(self, ranked_sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sequence diversity metrics."""
        if not ranked_sequences:
            return {}
        
        sequences = [r['sequence'] for r in ranked_sequences]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                # Simple similarity calculation
                seq1, seq2 = sequences[i], sequences[j]
                min_len = min(len(seq1), len(seq2))
                matches = sum(1 for k in range(min_len) if seq1[k] == seq2[k])
                similarity = matches / max(len(seq1), len(seq2)) if max(len(seq1), len(seq2)) > 0 else 0
                similarities.append(similarity)
        
        return {
            'avg_similarity': sum(similarities) / len(similarities) if similarities else 0,
            'diversity_score': 1 - (sum(similarities) / len(similarities)) if similarities else 1,
            'total_comparisons': len(similarities)
        }
    
    def _analyze_quality_distribution(self, ranked_sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality score distributions."""
        if not ranked_sequences:
            return {}
        
        binding_affinities = [r.get('binding_affinity', 0) for r in ranked_sequences]
        structure_qualities = [r.get('structure_quality', 0) for r in ranked_sequences]
        composite_scores = [r.get('composite_score', 0) for r in ranked_sequences]
        
        return {
            'binding_affinity': {
                'mean': sum(binding_affinities) / len(binding_affinities),
                'min': min(binding_affinities),
                'max': max(binding_affinities),
                'high_quality_count': sum(1 for x in binding_affinities if x < -10)  # Better binding
            },
            'structure_quality': {
                'mean': sum(structure_qualities) / len(structure_qualities),
                'min': min(structure_qualities),
                'max': max(structure_qualities),
                'high_quality_count': sum(1 for x in structure_qualities if x > 0.8)
            },
            'composite_score': {
                'mean': sum(composite_scores) / len(composite_scores),
                'min': min(composite_scores),
                'max': max(composite_scores)
            }
        }
    
    def _calculate_performance_metrics(
        self, 
        generation_result: Dict[str, Any], 
        ranking_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall workflow performance metrics."""
        return {
            'total_sequences_generated': generation_result['stats'].get('successful_generations', 0),
            'total_sequences_ranked': ranking_result['stats'].get('total_ranked', 0),
            'generation_success_rate': generation_result['stats'].get('success_rate', 0),
            'ranking_success_rate': 1.0 if ranking_result['success'] else 0.0,
            'total_processing_time': generation_result['stats'].get('generation_time', 0) + ranking_result['stats'].get('ranking_time', 0),
            'throughput_sequences_per_second': generation_result['stats'].get('successful_generations', 0) / max(generation_result['stats'].get('generation_time', 1), 1)
        }
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        import hashlib
        timestamp = str(time.time())
        experiment_name = self.config.experiment_name
        unique_string = f"{experiment_name}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _save_workflow_result(self, result: WorkflowResult):
        """Save workflow results to disk."""
        output_file = self.output_dir / f"{result.experiment_id}_workflow_result.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Workflow results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save workflow results: {e}")
    
    def run_batch_experiments(
        self,
        experiment_configs: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None
    ) -> List[WorkflowResult]:
        """
        Run multiple experiments in parallel.
        
        Args:
            experiment_configs: List of experiment configurations
            max_concurrent: Maximum concurrent experiments
            
        Returns:
            List of workflow results
        """
        if max_concurrent is None:
            max_concurrent = self.config.max_concurrent_tasks
        
        logger.info(f"Running {len(experiment_configs)} batch experiments with {max_concurrent} concurrent tasks")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self.run_complete_pipeline, **config): config
                for config in experiment_configs
            }
            
            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Experiment completed: {result.experiment_id}")
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    # Create error result for failed experiment
                    error_result = WorkflowResult(
                        experiment_id="failed_experiment",
                        timestamp=time.time(),
                        config=self.config,
                        generated_sequences=[],
                        generation_stats={},
                        ranked_sequences=[],
                        ranking_stats={},
                        success=False,
                        total_runtime=0,
                        error_messages=[str(e)]
                    )
                    results.append(error_result)
        
        logger.info(f"Batch experiments completed: {len(results)} results")
        return results
    
    def get_workflow_health(self) -> Dict[str, Any]:
        """Get health status of all workflow components."""
        health_status = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'components': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check diffuser
            if self.diffuser:
                diffuser_health = self.diffuser.health_check()
                health_status['components']['diffuser'] = diffuser_health
                if diffuser_health['overall_status'] != 'healthy':
                    health_status['warnings'].append('Diffuser not healthy')
            else:
                health_status['components']['diffuser'] = {'status': 'not_initialized'}
                health_status['warnings'].append('Diffuser not initialized')
            
            # Check ranker
            if self.ranker:
                health_status['components']['ranker'] = {'status': 'healthy', 'initialized': True}
            else:
                health_status['components']['ranker'] = {'status': 'not_initialized'}
                health_status['warnings'].append('Ranker not initialized')
            
            # Check validation
            if self.validation_manager:
                health_status['components']['validation'] = {'status': 'healthy', 'enabled': True}
            else:
                health_status['components']['validation'] = {'status': 'disabled'}
            
            # Check security
            if self.security_manager:
                health_status['components']['security'] = {'status': 'healthy', 'enabled': True}
            else:
                health_status['components']['security'] = {'status': 'disabled'}
            
            # Determine overall status
            if health_status['errors']:
                health_status['overall_status'] = 'error'
            elif health_status['warnings']:
                health_status['overall_status'] = 'warning'
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['errors'].append(f"Health check failed: {e}")
        
        return health_status


# Convenience functions for common workflows

def quick_protein_design(
    motif: str,
    num_sequences: int = 50,
    target_pdb: Optional[str] = None,
    output_dir: str = "./quick_design_output"
) -> WorkflowResult:
    """
    Quick protein design with sensible defaults.
    
    Args:
        motif: Target protein motif
        num_sequences: Number of sequences to generate
        target_pdb: Optional target PDB for binding evaluation
        output_dir: Output directory
        
    Returns:
        Workflow results
    """
    config = WorkflowConfig(
        output_dir=output_dir,
        experiment_name="quick_design"
    )
    
    workflow = ProteinDesignWorkflow(config)
    return workflow.run_complete_pipeline(
        motif=motif,
        num_sequences=num_sequences,
        target_pdb=target_pdb
    )


def batch_motif_exploration(
    motifs: List[str],
    num_sequences_per_motif: int = 25,
    output_dir: str = "./batch_exploration"
) -> List[WorkflowResult]:
    """
    Explore multiple motifs in parallel.
    
    Args:
        motifs: List of motifs to explore
        num_sequences_per_motif: Sequences to generate per motif
        output_dir: Output directory
        
    Returns:
        List of workflow results
    """
    config = WorkflowConfig(
        output_dir=output_dir,
        experiment_name="motif_exploration"
    )
    
    workflow = ProteinDesignWorkflow(config)
    
    # Create experiment configs
    experiment_configs = [
        {
            'motif': motif,
            'num_sequences': num_sequences_per_motif,
            'experiment_name': f"motif_{motif.replace('_', '-')}"
        }
        for motif in motifs
    ]
    
    return workflow.run_batch_experiments(experiment_configs)