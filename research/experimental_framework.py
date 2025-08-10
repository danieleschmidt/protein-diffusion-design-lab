#!/usr/bin/env python3
"""
Experimental Framework for Protein Diffusion Research

This module provides a comprehensive experimental framework for conducting
rigorous research studies with proper statistical analysis, reproducibility,
and publication-ready results.

Features:
- Controlled experiments with multiple random seeds
- Statistical significance testing with multiple hypothesis correction
- Effect size calculations and confidence intervals  
- Publication-ready figures and tables
- Reproducible experimental protocols
"""

import sys
import os
import time
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from collections import defaultdict
import warnings

# Statistical analysis
try:
    from scipy import stats
    from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available - using simplified statistics")

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available - no plots will be generated")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from research.advanced_methods import (
        MultiObjectiveOptimizer, PhysicsInformedDiffusion, 
        AdversarialValidator, AdaptiveSampler, ResearchConfig
    )
    ADVANCED_METHODS_AVAILABLE = True
except ImportError:
    ADVANCED_METHODS_AVAILABLE = False
    warnings.warn("Advanced methods not available")

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experimental studies."""
    # Experiment identification
    experiment_name: str = "protein_diffusion_study"
    experiment_version: str = "1.0"
    description: str = "Comprehensive protein diffusion research study"
    
    # Reproducibility
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 101112])
    num_repeats_per_seed: int = 3
    
    # Statistical analysis
    alpha_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"  # "bonferroni", "fdr", "holm"
    effect_size_threshold: float = 0.2  # Minimum meaningful effect size
    
    # Data collection
    collect_intermediate_results: bool = True
    save_raw_data: bool = True
    compute_confidence_intervals: bool = True
    
    # Output
    output_dir: str = "./experimental_results"
    generate_figures: bool = True
    generate_report: bool = True
    
    # Computational resources
    max_parallel_jobs: int = 4
    timeout_per_experiment: int = 3600  # seconds
    
    # Research-specific parameters
    baseline_methods: List[str] = field(default_factory=lambda: ["random", "greedy", "simple_diffusion"])
    experimental_methods: List[str] = field(default_factory=lambda: ["multi_objective", "physics_informed", "adversarial"])
    evaluation_metrics: List[str] = field(default_factory=lambda: ["diversity", "quality", "novelty", "synthesizability"])


class ExperimentRunner:
    """Main experiment runner with statistical rigor."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracking
        self.experiment_id = self._generate_experiment_id()
        self.results = defaultdict(list)
        self.metadata = {
            'experiment_id': self.experiment_id,
            'start_time': datetime.now().isoformat(),
            'config': config,
            'system_info': self._collect_system_info()
        }
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized experiment runner: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        return f"{self.config.experiment_name}_{timestamp}_{config_hash}"
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for reproducibility."""
        import platform
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'scipy_available': SCIPY_AVAILABLE,
            'matplotlib_available': PLOTTING_AVAILABLE,
            'cpu_count': os.cpu_count(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _setup_logging(self):
        """Setup detailed logging for experiment."""
        log_file = self.output_dir / f"{self.experiment_id}.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def run_controlled_experiment(self, 
                                baseline_data: List[str],
                                experimental_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run controlled experiment with multiple seeds and statistical analysis.
        """
        logger.info(f"Starting controlled experiment: {self.experiment_id}")
        
        experiment_results = {
            'metadata': self.metadata,
            'conditions': experimental_conditions,
            'baseline_results': {},
            'experimental_results': {},
            'statistical_analysis': {},
            'raw_data': {} if self.config.save_raw_data else None
        }
        
        # Run baseline methods
        logger.info("Running baseline methods...")
        for method in self.config.baseline_methods:
            logger.info(f"Running baseline method: {method}")
            method_results = self._run_method_with_seeds(
                method, baseline_data, experimental_conditions, is_baseline=True
            )
            experiment_results['baseline_results'][method] = method_results
        
        # Run experimental methods
        logger.info("Running experimental methods...")
        for method in self.config.experimental_methods:
            logger.info(f"Running experimental method: {method}")
            method_results = self._run_method_with_seeds(
                method, baseline_data, experimental_conditions, is_baseline=False
            )
            experiment_results['experimental_results'][method] = method_results
        
        # Perform statistical analysis
        logger.info("Performing statistical analysis...")
        statistical_results = self._perform_statistical_analysis(experiment_results)
        experiment_results['statistical_analysis'] = statistical_results
        
        # Generate visualizations
        if self.config.generate_figures and PLOTTING_AVAILABLE:
            logger.info("Generating visualizations...")
            self._generate_visualizations(experiment_results)
        
        # Generate report
        if self.config.generate_report:
            logger.info("Generating experimental report...")
            self._generate_report(experiment_results)
        
        # Save results
        self._save_experiment_results(experiment_results)
        
        logger.info(f"Experiment completed: {self.experiment_id}")
        
        return experiment_results
    
    def _run_method_with_seeds(self, 
                              method_name: str,
                              data: List[str], 
                              conditions: Dict[str, Any],
                              is_baseline: bool = False) -> Dict[str, Any]:
        """Run a method with multiple random seeds for statistical validity."""
        
        method_results = {
            'method_name': method_name,
            'is_baseline': is_baseline,
            'runs': [],
            'aggregated_metrics': {},
            'execution_info': {}
        }
        
        total_runs = len(self.config.random_seeds) * self.config.num_repeats_per_seed
        run_id = 0
        
        for seed in self.config.random_seeds:
            for repeat in range(self.config.num_repeats_per_seed):
                run_id += 1
                logger.info(f"Running {method_name} - seed {seed}, repeat {repeat+1}/{self.config.num_repeats_per_seed} ({run_id}/{total_runs})")
                
                start_time = time.time()
                
                try:
                    # Set random seed for reproducibility
                    np.random.seed(seed + repeat * 1000)
                    
                    # Run the method
                    run_result = self._execute_method(method_name, data, conditions, seed)
                    
                    execution_time = time.time() - start_time
                    
                    # Calculate metrics for this run
                    metrics = self._calculate_metrics(run_result, data)
                    
                    run_info = {
                        'seed': seed,
                        'repeat': repeat,
                        'run_id': run_id,
                        'execution_time': execution_time,
                        'metrics': metrics,
                        'result': run_result if self.config.save_raw_data else None,
                        'success': True,
                        'error': None
                    }
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"Run failed: {method_name} seed={seed} repeat={repeat}: {e}")
                    
                    run_info = {
                        'seed': seed,
                        'repeat': repeat,
                        'run_id': run_id,
                        'execution_time': execution_time,
                        'metrics': {},
                        'result': None,
                        'success': False,
                        'error': str(e)
                    }
                
                method_results['runs'].append(run_info)
        
        # Aggregate results across all runs
        successful_runs = [run for run in method_results['runs'] if run['success']]
        
        if successful_runs:
            method_results['aggregated_metrics'] = self._aggregate_metrics(successful_runs)
            method_results['execution_info'] = {
                'total_runs': total_runs,
                'successful_runs': len(successful_runs),
                'success_rate': len(successful_runs) / total_runs,
                'avg_execution_time': np.mean([run['execution_time'] for run in successful_runs]),
                'std_execution_time': np.std([run['execution_time'] for run in successful_runs])
            }
        else:
            logger.error(f"All runs failed for method: {method_name}")
            method_results['aggregated_metrics'] = {}
            method_results['execution_info'] = {
                'total_runs': total_runs,
                'successful_runs': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'std_execution_time': 0.0
            }
        
        return method_results
    
    def _execute_method(self, method_name: str, data: List[str], conditions: Dict[str, Any], seed: int) -> Any:
        """Execute a specific method."""
        
        if method_name == "random":
            return self._run_random_baseline(data, conditions, seed)
        elif method_name == "greedy":
            return self._run_greedy_baseline(data, conditions, seed)
        elif method_name == "simple_diffusion":
            return self._run_simple_diffusion(data, conditions, seed)
        elif method_name == "multi_objective" and ADVANCED_METHODS_AVAILABLE:
            return self._run_multi_objective(data, conditions, seed)
        elif method_name == "physics_informed" and ADVANCED_METHODS_AVAILABLE:
            return self._run_physics_informed(data, conditions, seed)
        elif method_name == "adversarial" and ADVANCED_METHODS_AVAILABLE:
            return self._run_adversarial(data, conditions, seed)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def _run_random_baseline(self, data: List[str], conditions: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Random baseline - generate random protein sequences."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        num_sequences = conditions.get('num_samples', 50)
        target_length = conditions.get('target_length', 100)
        
        generated_sequences = []
        for _ in range(num_sequences):
            length = target_length + np.random.randint(-20, 20)
            length = max(20, length)
            sequence = ''.join(np.random.choice(list(amino_acids)) for _ in range(length))
            generated_sequences.append(sequence)
        
        return {
            'method': 'random',
            'sequences': generated_sequences,
            'metadata': {
                'num_generated': len(generated_sequences),
                'avg_length': np.mean([len(s) for s in generated_sequences]),
                'generation_strategy': 'uniform_random'
            }
        }
    
    def _run_greedy_baseline(self, data: List[str], conditions: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Greedy baseline - select best sequences from training data."""
        num_sequences = conditions.get('num_samples', 50)
        
        # Simple greedy selection based on sequence diversity
        selected_sequences = []
        remaining_data = data.copy()
        
        # Select first sequence randomly
        if remaining_data:
            selected_sequences.append(remaining_data.pop(np.random.randint(len(remaining_data))))
        
        # Greedily select diverse sequences
        while len(selected_sequences) < num_sequences and remaining_data:
            best_seq = None
            best_diversity = -1
            
            for candidate in remaining_data[:100]:  # Limit search for efficiency
                diversity = self._calculate_sequence_diversity(candidate, selected_sequences)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_seq = candidate
            
            if best_seq:
                selected_sequences.append(best_seq)
                remaining_data.remove(best_seq)
            else:
                break
        
        return {
            'method': 'greedy',
            'sequences': selected_sequences,
            'metadata': {
                'num_generated': len(selected_sequences),
                'avg_length': np.mean([len(s) for s in selected_sequences]),
                'selection_strategy': 'diversity_greedy'
            }
        }
    
    def _run_simple_diffusion(self, data: List[str], conditions: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Simple diffusion baseline - basic sequence generation."""
        # Mock simple diffusion for baseline
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        num_sequences = conditions.get('num_samples', 50)
        
        # Use Markov chain based on training data
        transitions = self._build_markov_model(data, order=2)
        
        generated_sequences = []
        for _ in range(num_sequences):
            sequence = self._generate_markov_sequence(transitions, length=100)
            generated_sequences.append(sequence)
        
        return {
            'method': 'simple_diffusion',
            'sequences': generated_sequences,
            'metadata': {
                'num_generated': len(generated_sequences),
                'avg_length': np.mean([len(s) for s in generated_sequences]),
                'generation_strategy': 'markov_chain_order_2'
            }
        }
    
    def _run_multi_objective(self, data: List[str], conditions: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Run multi-objective optimization method."""
        if not ADVANCED_METHODS_AVAILABLE:
            raise ImportError("Advanced methods not available")
        
        research_config = ResearchConfig(
            population_size=min(50, len(data)),
            num_generations=20,
            random_seeds=[seed]
        )
        
        optimizer = MultiObjectiveOptimizer(research_config)
        results = optimizer.optimize(data[:50])
        
        pareto_sequences = [ind['sequence'] for ind in results['final_pareto_front']]
        
        return {
            'method': 'multi_objective',
            'sequences': pareto_sequences,
            'metadata': {
                'num_generated': len(pareto_sequences),
                'avg_length': np.mean([len(s) for s in pareto_sequences]) if pareto_sequences else 0,
                'generation_strategy': 'pareto_optimization',
                'pareto_front_size': len(results['final_pareto_front']),
                'total_generations': results['total_generations']
            }
        }
    
    def _run_physics_informed(self, data: List[str], conditions: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Run physics-informed diffusion method."""
        if not ADVANCED_METHODS_AVAILABLE:
            raise ImportError("Advanced methods not available")
        
        research_config = ResearchConfig(random_seeds=[seed])
        physics_diffusion = PhysicsInformedDiffusion(research_config)
        
        results = physics_diffusion.physics_guided_sampling(
            motif=conditions.get('motif', 'HELIX_SHEET_HELIX'),
            num_samples=conditions.get('num_samples', 50)
        )
        
        sequences = [result['sequence'] for result in results]
        avg_physics_score = np.mean([result.get('physics_score', 0) for result in results])
        
        return {
            'method': 'physics_informed',
            'sequences': sequences,
            'metadata': {
                'num_generated': len(sequences),
                'avg_length': np.mean([len(s) for s in sequences]) if sequences else 0,
                'generation_strategy': 'physics_guided_diffusion',
                'avg_physics_score': avg_physics_score
            }
        }
    
    def _run_adversarial(self, data: List[str], conditions: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Run adversarial validation method."""
        if not ADVANCED_METHODS_AVAILABLE:
            raise ImportError("Advanced methods not available")
        
        research_config = ResearchConfig(random_seeds=[seed])
        validator = AdversarialValidator(research_config)
        
        # Split data for training/testing
        split_idx = len(data) // 2
        real_sequences = data[:split_idx]
        test_sequences = data[split_idx:]
        
        # Train discriminator
        training_results = validator.train_discriminator(
            real_sequences[:100], test_sequences[:100], num_epochs=50
        )
        
        # Validate test sequences
        validation_results = validator.validate_sequences(test_sequences[:50])
        high_quality_sequences = [
            result['sequence'] for result in validation_results 
            if result['adversarial_quality'] > 0.7
        ]
        
        return {
            'method': 'adversarial',
            'sequences': high_quality_sequences,
            'metadata': {
                'num_generated': len(high_quality_sequences),
                'avg_length': np.mean([len(s) for s in high_quality_sequences]) if high_quality_sequences else 0,
                'generation_strategy': 'adversarial_filtering',
                'discriminator_accuracy': training_results.get('final_accuracy', 0),
                'filtering_rate': len(high_quality_sequences) / len(test_sequences) if test_sequences else 0
            }
        }
    
    def _calculate_sequence_diversity(self, sequence: str, reference_sequences: List[str]) -> float:
        """Calculate diversity of a sequence relative to reference set."""
        if not reference_sequences:
            return 1.0
        
        similarities = []
        for ref_seq in reference_sequences:
            # Simple sequence identity
            min_len = min(len(sequence), len(ref_seq))
            if min_len == 0:
                continue
            
            matches = sum(1 for i in range(min_len) if sequence[i] == ref_seq[i])
            similarity = matches / max(len(sequence), len(ref_seq))
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1.0 - avg_similarity
    
    def _build_markov_model(self, sequences: List[str], order: int = 2) -> Dict[str, Dict[str, float]]:
        """Build Markov model from training sequences."""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for sequence in sequences:
            for i in range(len(sequence) - order):
                context = sequence[i:i+order]
                next_char = sequence[i+order]
                transitions[context][next_char] += 1
        
        # Normalize to probabilities
        normalized_transitions = {}
        for context, next_chars in transitions.items():
            total = sum(next_chars.values())
            normalized_transitions[context] = {
                char: count / total for char, count in next_chars.items()
            }
        
        return normalized_transitions
    
    def _generate_markov_sequence(self, transitions: Dict[str, Dict[str, float]], length: int) -> str:
        """Generate sequence using Markov model."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Start with random context
        order = len(list(transitions.keys())[0]) if transitions else 2
        sequence = ''.join(np.random.choice(list(amino_acids)) for _ in range(order))
        
        while len(sequence) < length:
            context = sequence[-order:]
            
            if context in transitions:
                # Sample next character based on learned probabilities
                chars = list(transitions[context].keys())
                probs = list(transitions[context].values())
                next_char = np.random.choice(chars, p=probs)
            else:
                # Fallback to random
                next_char = np.random.choice(list(amino_acids))
            
            sequence += next_char
        
        return sequence
    
    def _calculate_metrics(self, result: Dict[str, Any], reference_data: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics for a method result."""
        sequences = result.get('sequences', [])
        if not sequences:
            return {}
        
        metrics = {}
        
        # Diversity metric
        metrics['diversity'] = self._calculate_dataset_diversity(sequences)
        
        # Quality metrics (simplified)
        metrics['avg_length'] = np.mean([len(seq) for seq in sequences])
        metrics['length_std'] = np.std([len(seq) for seq in sequences])
        
        # Amino acid composition similarity to reference
        metrics['composition_similarity'] = self._calculate_composition_similarity(sequences, reference_data)
        
        # Uniqueness
        metrics['uniqueness'] = len(set(sequences)) / len(sequences) if sequences else 0
        
        # Synthesizability (simplified)
        metrics['synthesizability'] = self._calculate_synthesizability_score(sequences)
        
        return metrics
    
    def _calculate_dataset_diversity(self, sequences: List[str]) -> float:
        """Calculate overall diversity of sequence dataset."""
        if len(sequences) < 2:
            return 0.0
        
        similarities = []
        for i in range(min(100, len(sequences))):  # Sample for efficiency
            for j in range(i + 1, min(100, len(sequences))):
                seq1, seq2 = sequences[i], sequences[j]
                min_len = min(len(seq1), len(seq2))
                if min_len > 0:
                    matches = sum(1 for k in range(min_len) if seq1[k] == seq2[k])
                    similarity = matches / max(len(seq1), len(seq2))
                    similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1.0 - avg_similarity
    
    def _calculate_composition_similarity(self, sequences: List[str], reference_data: List[str]) -> float:
        """Calculate amino acid composition similarity to reference dataset."""
        if not sequences or not reference_data:
            return 0.0
        
        # Calculate AA composition for generated sequences
        all_generated = ''.join(sequences)
        gen_composition = {}
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            gen_composition[aa] = all_generated.count(aa) / len(all_generated) if all_generated else 0
        
        # Calculate AA composition for reference
        all_reference = ''.join(reference_data)
        ref_composition = {}
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            ref_composition[aa] = all_reference.count(aa) / len(all_reference) if all_reference else 0
        
        # Calculate similarity (using cosine similarity)
        gen_vector = [gen_composition[aa] for aa in "ACDEFGHIKLMNPQRSTVWY"]
        ref_vector = [ref_composition[aa] for aa in "ACDEFGHIKLMNPQRSTVWY"]
        
        dot_product = sum(g * r for g, r in zip(gen_vector, ref_vector))
        gen_norm = sum(g * g for g in gen_vector) ** 0.5
        ref_norm = sum(r * r for r in ref_vector) ** 0.5
        
        if gen_norm > 0 and ref_norm > 0:
            return dot_product / (gen_norm * ref_norm)
        else:
            return 0.0
    
    def _calculate_synthesizability_score(self, sequences: List[str]) -> float:
        """Calculate synthesizability score based on sequence properties."""
        if not sequences:
            return 0.0
        
        scores = []
        for seq in sequences:
            # Factors affecting synthesizability
            rare_codons = "MWC"  # Simplified
            rare_content = sum(1 for aa in seq if aa in rare_codons) / len(seq)
            
            # Length penalty
            length_penalty = max(0, (len(seq) - 300) / 1000.0)
            
            # Repetitive patterns
            repetitiveness = self._calculate_repetitiveness_simple(seq)
            
            synthesizability = 1.0 - rare_content - length_penalty - repetitiveness
            scores.append(max(0.0, synthesizability))
        
        return np.mean(scores)
    
    def _calculate_repetitiveness_simple(self, sequence: str, window_size: int = 4) -> float:
        """Calculate sequence repetitiveness."""
        if len(sequence) < window_size:
            return 0.0
        
        kmers = {}
        for i in range(len(sequence) - window_size + 1):
            kmer = sequence[i:i + window_size]
            kmers[kmer] = kmers.get(kmer, 0) + 1
        
        # Calculate fraction of repeated k-mers
        repeated_kmers = sum(1 for count in kmers.values() if count > 1)
        total_kmers = len(kmers)
        
        return repeated_kmers / total_kmers if total_kmers > 0 else 0.0
    
    def _aggregate_metrics(self, successful_runs: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across multiple runs."""
        all_metrics = [run['metrics'] for run in successful_runs if run['metrics']]
        if not all_metrics:
            return {}
        
        # Get all metric names
        metric_names = set()
        for metrics in all_metrics:
            metric_names.update(metrics.keys())
        
        aggregated = {}
        for metric in metric_names:
            values = [metrics.get(metric, 0) for metrics in all_metrics]
            
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
            
            # Calculate confidence interval if scipy is available
            if SCIPY_AVAILABLE and len(values) > 1:
                ci = stats.t.interval(
                    1 - self.config.alpha_level,
                    len(values) - 1,
                    loc=np.mean(values),
                    scale=stats.sem(values)
                )
                aggregated[metric]['ci_lower'] = ci[0]
                aggregated[metric]['ci_upper'] = ci[1]
        
        return aggregated
    
    def _perform_statistical_analysis(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        logger.info("Performing statistical significance testing...")
        
        statistical_results = {
            'pairwise_comparisons': {},
            'effect_sizes': {},
            'significant_findings': [],
            'method_rankings': {},
            'multiple_testing_correction': self.config.multiple_testing_correction,
            'alpha_level': self.config.alpha_level
        }
        
        # Collect data for comparison
        method_data = {}
        
        # Baseline methods
        for method, results in experiment_results.get('baseline_results', {}).items():
            successful_runs = [run for run in results['runs'] if run['success']]
            if successful_runs:
                method_data[f"baseline_{method}"] = successful_runs
        
        # Experimental methods
        for method, results in experiment_results.get('experimental_results', {}).items():
            successful_runs = [run for run in results['runs'] if run['success']]
            if successful_runs:
                method_data[f"experimental_{method}"] = successful_runs
        
        # Pairwise statistical comparisons
        method_names = list(method_data.keys())
        p_values = []
        comparisons = []
        
        for metric in self.config.evaluation_metrics:
            statistical_results['pairwise_comparisons'][metric] = {}
            
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names):
                    if i < j:  # Avoid duplicate comparisons
                        # Extract metric values
                        values1 = [run['metrics'].get(metric, 0) for run in method_data[method1]]
                        values2 = [run['metrics'].get(metric, 0) for run in method_data[method2]]
                        
                        if len(values1) > 1 and len(values2) > 1:
                            # Perform statistical test
                            if SCIPY_AVAILABLE:
                                # Use Mann-Whitney U test (non-parametric)
                                statistic, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
                                
                                # Calculate effect size (Cohen's d)
                                pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1) + 
                                                    (len(values2) - 1) * np.var(values2)) / 
                                                   (len(values1) + len(values2) - 2))
                                effect_size = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
                            else:
                                # Simplified test without scipy
                                mean1, mean2 = np.mean(values1), np.mean(values2)
                                std1, std2 = np.std(values1), np.std(values2)
                                
                                # Simple t-test approximation
                                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                                t_stat = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                                p_value = 2 * (1 - abs(t_stat) / 10)  # Very rough approximation
                                p_value = max(0.001, min(0.999, p_value))
                                
                                effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                            
                            comparison_key = f"{method1}_vs_{method2}"
                            statistical_results['pairwise_comparisons'][metric][comparison_key] = {
                                'p_value': p_value,
                                'effect_size': effect_size,
                                'mean_difference': np.mean(values1) - np.mean(values2),
                                'method1_mean': np.mean(values1),
                                'method2_mean': np.mean(values2),
                                'method1_std': np.std(values1),
                                'method2_std': np.std(values2)
                            }
                            
                            p_values.append(p_value)
                            comparisons.append((metric, comparison_key))
        
        # Multiple testing correction
        if p_values and SCIPY_AVAILABLE:
            from statsmodels.stats.multitest import multipletests
            
            if self.config.multiple_testing_correction == "bonferroni":
                corrected_pvals = [p * len(p_values) for p in p_values]
                corrected_pvals = [min(1.0, p) for p in corrected_pvals]
            elif self.config.multiple_testing_correction == "fdr":
                _, corrected_pvals, _, _ = multipletests(p_values, method='fdr_bh')
            else:
                corrected_pvals = p_values  # No correction
            
            # Update results with corrected p-values
            for (metric, comparison_key), corrected_p in zip(comparisons, corrected_pvals):
                statistical_results['pairwise_comparisons'][metric][comparison_key]['corrected_p_value'] = corrected_p
                
                # Check for significance
                is_significant = corrected_p < self.config.alpha_level
                effect_size = statistical_results['pairwise_comparisons'][metric][comparison_key]['effect_size']
                is_meaningful = abs(effect_size) > self.config.effect_size_threshold
                
                if is_significant and is_meaningful:
                    finding = f"{metric}: {comparison_key} (p={corrected_p:.4f}, d={effect_size:.3f})"
                    statistical_results['significant_findings'].append(finding)
        
        # Method rankings based on performance
        for metric in self.config.evaluation_metrics:
            metric_scores = {}
            for method_name, runs in method_data.items():
                values = [run['metrics'].get(metric, 0) for run in runs]
                if values:
                    metric_scores[method_name] = np.mean(values)
            
            # Rank methods by performance
            ranked_methods = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
            statistical_results['method_rankings'][metric] = ranked_methods
        
        return statistical_results
    
    def _generate_visualizations(self, experiment_results: Dict[str, Any]) -> None:
        """Generate publication-quality visualizations."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available - skipping visualizations")
            return
        
        logger.info("Generating visualizations...")
        
        # Set style for publication quality
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 1. Method performance comparison
        self._plot_method_comparison(experiment_results, figures_dir)
        
        # 2. Metric distributions
        self._plot_metric_distributions(experiment_results, figures_dir)
        
        # 3. Statistical significance heatmap
        self._plot_significance_heatmap(experiment_results, figures_dir)
        
        # 4. Effect sizes visualization
        self._plot_effect_sizes(experiment_results, figures_dir)
        
        logger.info(f"Visualizations saved to {figures_dir}")
    
    def _plot_method_comparison(self, experiment_results: Dict[str, Any], figures_dir: Path) -> None:
        """Plot method performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.config.evaluation_metrics[:4]):
            ax = axes[i]
            
            # Collect data for plotting
            method_names = []
            metric_values = []
            method_types = []
            
            # Baseline methods
            for method, results in experiment_results.get('baseline_results', {}).items():
                if metric in results.get('aggregated_metrics', {}):
                    method_names.append(f"Baseline\n{method}")
                    values = [run['metrics'].get(metric, 0) for run in results['runs'] if run['success']]
                    metric_values.extend(values)
                    method_types.extend(['Baseline'] * len(values))
            
            # Experimental methods  
            for method, results in experiment_results.get('experimental_results', {}).items():
                if metric in results.get('aggregated_metrics', {}):
                    method_names.append(f"Experimental\n{method}")
                    values = [run['metrics'].get(metric, 0) for run in results['runs'] if run['success']]
                    metric_values.extend(values)
                    method_types.extend(['Experimental'] * len(values))
            
            if method_names and metric_values:
                # Create DataFrame for seaborn
                plot_data = {
                    'Method': [],
                    'Value': [],
                    'Type': []
                }
                
                # Baseline methods
                for method, results in experiment_results.get('baseline_results', {}).items():
                    if metric in results.get('aggregated_metrics', {}):
                        values = [run['metrics'].get(metric, 0) for run in results['runs'] if run['success']]
                        plot_data['Method'].extend([method] * len(values))
                        plot_data['Value'].extend(values)
                        plot_data['Type'].extend(['Baseline'] * len(values))
                
                # Experimental methods
                for method, results in experiment_results.get('experimental_results', {}).items():
                    if metric in results.get('aggregated_metrics', {}):
                        values = [run['metrics'].get(metric, 0) for run in results['runs'] if run['success']]
                        plot_data['Method'].extend([method] * len(values))
                        plot_data['Value'].extend(values)
                        plot_data['Type'].extend(['Experimental'] * len(values))
                
                if plot_data['Method']:
                    df = pd.DataFrame(plot_data)
                    sns.boxplot(data=df, x='Method', y='Value', hue='Type', ax=ax)
                    ax.set_title(f'{metric.title()} Comparison')
                    ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_distributions(self, experiment_results: Dict[str, Any], figures_dir: Path) -> None:
        """Plot metric distributions for each method."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.config.evaluation_metrics[:4]):
            ax = axes[i]
            
            # Collect all values for this metric
            all_data = []
            labels = []
            
            # Baseline methods
            for method, results in experiment_results.get('baseline_results', {}).items():
                values = [run['metrics'].get(metric, 0) for run in results['runs'] if run['success']]
                if values:
                    all_data.append(values)
                    labels.append(f"B-{method}")
            
            # Experimental methods
            for method, results in experiment_results.get('experimental_results', {}).items():
                values = [run['metrics'].get(metric, 0) for run in results['runs'] if run['success']]
                if values:
                    all_data.append(values)
                    labels.append(f"E-{method}")
            
            if all_data:
                ax.hist(all_data, alpha=0.7, label=labels, bins=20)
                ax.set_title(f'{metric.title()} Distribution')
                ax.set_xlabel(metric.title())
                ax.set_ylabel('Frequency')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_significance_heatmap(self, experiment_results: Dict[str, Any], figures_dir: Path) -> None:
        """Plot statistical significance heatmap."""
        statistical_analysis = experiment_results.get('statistical_analysis', {})
        pairwise_comparisons = statistical_analysis.get('pairwise_comparisons', {})
        
        if not pairwise_comparisons:
            return
        
        # Create significance matrix
        methods = set()
        for metric_comparisons in pairwise_comparisons.values():
            for comparison in metric_comparisons.keys():
                method1, method2 = comparison.split('_vs_')
                methods.add(method1)
                methods.add(method2)
        
        methods = sorted(list(methods))
        n_methods = len(methods)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.config.evaluation_metrics[:4]):
            if metric not in pairwise_comparisons:
                continue
                
            ax = axes[i]
            
            # Create significance matrix for this metric
            sig_matrix = np.ones((n_methods, n_methods))
            
            for comparison, results in pairwise_comparisons[metric].items():
                method1, method2 = comparison.split('_vs_')
                if method1 in methods and method2 in methods:
                    idx1, idx2 = methods.index(method1), methods.index(method2)
                    p_value = results.get('corrected_p_value', results.get('p_value', 1.0))
                    sig_matrix[idx1, idx2] = p_value
                    sig_matrix[idx2, idx1] = p_value
            
            # Plot heatmap
            im = ax.imshow(sig_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
            ax.set_xticks(range(n_methods))
            ax.set_yticks(range(n_methods))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_yticklabels(methods)
            ax.set_title(f'{metric.title()} - P-values')
            
            # Add p-value text
            for j in range(n_methods):
                for k in range(n_methods):
                    text = ax.text(k, j, f'{sig_matrix[j, k]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax, label='P-value')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'significance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_effect_sizes(self, experiment_results: Dict[str, Any], figures_dir: Path) -> None:
        """Plot effect sizes for significant comparisons."""
        statistical_analysis = experiment_results.get('statistical_analysis', {})
        pairwise_comparisons = statistical_analysis.get('pairwise_comparisons', {})
        
        if not pairwise_comparisons:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Collect significant effect sizes
        effect_data = []
        
        for metric, comparisons in pairwise_comparisons.items():
            for comparison, results in comparisons.items():
                p_value = results.get('corrected_p_value', results.get('p_value', 1.0))
                effect_size = results.get('effect_size', 0)
                
                if p_value < self.config.alpha_level and abs(effect_size) > self.config.effect_size_threshold:
                    effect_data.append({
                        'Comparison': f"{metric}\n{comparison.replace('_vs_', ' vs ')}",
                        'Effect Size': effect_size,
                        'P-value': p_value,
                        'Significant': True
                    })
        
        if effect_data:
            df = pd.DataFrame(effect_data)
            
            # Create effect size plot
            bars = ax.barh(df['Comparison'], df['Effect Size'], 
                          color=['green' if x > 0 else 'red' for x in df['Effect Size']])
            
            ax.set_xlabel('Effect Size (Cohen\'s d)')
            ax.set_title('Significant Effect Sizes')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=self.config.effect_size_threshold, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=-self.config.effect_size_threshold, color='gray', linestyle='--', alpha=0.5)
            
            # Add p-value annotations
            for i, (bar, p_val) in enumerate(zip(bars, df['P-value'])):
                ax.text(bar.get_width() + 0.01 if bar.get_width() > 0 else bar.get_width() - 0.01,
                       bar.get_y() + bar.get_height()/2,
                       f'p={p_val:.3f}',
                       ha='left' if bar.get_width() > 0 else 'right',
                       va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_report(self, experiment_results: Dict[str, Any]) -> None:
        """Generate comprehensive experimental report."""
        report_path = self.output_dir / f"{self.experiment_id}_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Experimental Report: {self.config.experiment_name}\n\n")
            f.write(f"**Experiment ID:** {self.experiment_id}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Description:** {self.config.description}\n\n")
            
            # Experiment configuration
            f.write("## Experiment Configuration\n\n")
            f.write(f"- **Random Seeds:** {self.config.random_seeds}\n")
            f.write(f"- **Repeats per Seed:** {self.config.num_repeats_per_seed}\n")
            f.write(f"- **Alpha Level:** {self.config.alpha_level}\n")
            f.write(f"- **Multiple Testing Correction:** {self.config.multiple_testing_correction}\n")
            f.write(f"- **Effect Size Threshold:** {self.config.effect_size_threshold}\n\n")
            
            # Methods tested
            f.write("## Methods Tested\n\n")
            f.write("### Baseline Methods\n")
            for method in self.config.baseline_methods:
                f.write(f"- {method}\n")
            
            f.write("\n### Experimental Methods\n")
            for method in self.config.experimental_methods:
                f.write(f"- {method}\n")
            
            # Results summary
            f.write("\n## Results Summary\n\n")
            
            # Baseline results
            f.write("### Baseline Methods Performance\n\n")
            for method, results in experiment_results.get('baseline_results', {}).items():
                execution_info = results.get('execution_info', {})
                f.write(f"#### {method.title()}\n")
                f.write(f"- Success Rate: {execution_info.get('success_rate', 0):.1%}\n")
                f.write(f"- Average Execution Time: {execution_info.get('avg_execution_time', 0):.2f}s\n")
                
                # Metrics
                aggregated = results.get('aggregated_metrics', {})
                for metric_name, stats in aggregated.items():
                    f.write(f"- {metric_name.title()}: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                f.write("\n")
            
            # Experimental results
            f.write("### Experimental Methods Performance\n\n")
            for method, results in experiment_results.get('experimental_results', {}).items():
                execution_info = results.get('execution_info', {})
                f.write(f"#### {method.title()}\n")
                f.write(f"- Success Rate: {execution_info.get('success_rate', 0):.1%}\n")
                f.write(f"- Average Execution Time: {execution_info.get('avg_execution_time', 0):.2f}s\n")
                
                # Metrics
                aggregated = results.get('aggregated_metrics', {})
                for metric_name, stats in aggregated.items():
                    f.write(f"- {metric_name.title()}: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                f.write("\n")
            
            # Statistical analysis
            statistical_analysis = experiment_results.get('statistical_analysis', {})
            
            f.write("## Statistical Analysis\n\n")
            
            # Significant findings
            significant_findings = statistical_analysis.get('significant_findings', [])
            if significant_findings:
                f.write("### Significant Findings\n\n")
                for finding in significant_findings:
                    f.write(f"- {finding}\n")
                f.write("\n")
            else:
                f.write("### No statistically significant differences found.\n\n")
            
            # Method rankings
            rankings = statistical_analysis.get('method_rankings', {})
            if rankings:
                f.write("### Method Rankings by Metric\n\n")
                for metric, ranked_methods in rankings.items():
                    f.write(f"#### {metric.title()}\n")
                    for i, (method, score) in enumerate(ranked_methods, 1):
                        f.write(f"{i}. {method}: {score:.3f}\n")
                    f.write("\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            f.write("Based on the experimental results:\n\n")
            
            # Automatically generate conclusions based on results
            if significant_findings:
                f.write(f"- Found {len(significant_findings)} statistically significant differences between methods\n")
                
                # Find best performing experimental method
                best_experimental = None
                best_score = -float('inf')
                
                for method, results in experiment_results.get('experimental_results', {}).items():
                    aggregated = results.get('aggregated_metrics', {})
                    if 'diversity' in aggregated:
                        score = aggregated['diversity']['mean']
                        if score > best_score:
                            best_score = score
                            best_experimental = method
                
                if best_experimental:
                    f.write(f"- {best_experimental} showed the best performance among experimental methods\n")
            else:
                f.write("- No significant differences found between baseline and experimental methods\n")
                f.write("- Further investigation may be needed with larger sample sizes\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("- Consider increasing sample sizes for future experiments\n")
            f.write("- Investigate additional evaluation metrics\n")
            f.write("- Validate results on different datasets\n")
        
        logger.info(f"Experimental report generated: {report_path}")
    
    def _save_experiment_results(self, experiment_results: Dict[str, Any]) -> None:
        """Save experiment results in multiple formats."""
        # JSON format
        json_path = self.output_dir / f"{self.experiment_id}_results.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_json_serializable(experiment_results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Pickle format for full Python objects
        pickle_path = self.output_dir / f"{self.experiment_id}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(experiment_results, f)
        
        logger.info(f"Results saved: {json_path}, {pickle_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-JSON types to serializable formats."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj


def run_publication_study() -> Dict[str, Any]:
    """
    Run a complete publication-ready experimental study.
    """
    logger.info("Starting publication-ready experimental study")
    
    # Create experimental configuration
    experiment_config = ExperimentConfig(
        experiment_name="protein_diffusion_comprehensive_study",
        experiment_version="2.0",
        description="Comprehensive evaluation of novel protein diffusion methods",
        random_seeds=[42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627],
        num_repeats_per_seed=5,
        alpha_level=0.01,  # Stricter alpha for publication
        multiple_testing_correction="fdr",
        effect_size_threshold=0.3,
        generate_figures=True,
        generate_report=True
    )
    
    # Create experiment runner
    runner = ExperimentRunner(experiment_config)
    
    # Generate mock baseline data (in real study, this would be real protein sequences)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    baseline_sequences = []
    
    for i in range(1000):
        length = np.random.randint(50, 300)
        sequence = ''.join(np.random.choice(list(amino_acids)) for _ in range(length))
        baseline_sequences.append(sequence)
    
    # Experimental conditions
    conditions = {
        'num_samples': 100,
        'target_length': 150,
        'motif': 'HELIX_SHEET_HELIX',
        'temperature': 1.0
    }
    
    # Run the experiment
    results = runner.run_controlled_experiment(baseline_sequences, conditions)
    
    logger.info("Publication study completed")
    
    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run publication study
    study_results = run_publication_study()
    
    print("\n" + "="*60)
    print("EXPERIMENTAL STUDY COMPLETED")
    print("="*60)
    
    # Print summary
    significant_findings = study_results.get('statistical_analysis', {}).get('significant_findings', [])
    print(f"\nSignificant findings: {len(significant_findings)}")
    
    for finding in significant_findings[:5]:  # Show first 5
        print(f"  • {finding}")
    
    if len(significant_findings) > 5:
        print(f"  ... and {len(significant_findings) - 5} more")
    
    print(f"\nResults saved to: {study_results['metadata']['experiment_id']}")
    print("="*60)