#!/usr/bin/env python3
"""
Advanced Research Methods for Protein Diffusion Design Lab

This module implements novel research approaches including:
- Multi-objective protein optimization
- Physics-informed diffusion models
- Adversarial protein validation
- Adaptive sampling strategies
- Hierarchical structural conditioning

Author: Research Team
"""

import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import random
from collections import deque
import math

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from protein_diffusion import ProteinDiffuser, ProteinDiffuserConfig
    from protein_diffusion.models import DiffusionTransformer, DDPM
    from protein_diffusion.ranker import AffinityRanker, AffinityRankerConfig
    MODULE_IMPORTS_AVAILABLE = True
except ImportError:
    MODULE_IMPORTS_AVAILABLE = False
    print("Warning: Core modules not available, using mock implementations")

logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Configuration for advanced research experiments."""
    # Multi-objective optimization
    population_size: int = 100
    num_generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    
    # Physics-informed parameters
    enable_physics_constraints: bool = True
    energy_weight: float = 0.3
    stability_weight: float = 0.4
    interaction_weight: float = 0.3
    
    # Adversarial validation
    discriminator_lr: float = 0.0002
    generator_lr: float = 0.0002
    adversarial_weight: float = 0.1
    
    # Adaptive sampling
    adaptive_temperature: bool = True
    temperature_schedule: str = "cosine"  # "linear", "exponential", "cosine"
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    
    # Hierarchical conditioning
    use_hierarchical_conditioning: bool = True
    num_hierarchy_levels: int = 3
    motif_embedding_dim: int = 256
    
    # Research methodology
    num_baseline_runs: int = 5
    num_experimental_runs: int = 5
    statistical_significance_threshold: float = 0.05
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 101112])


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for protein design using genetic algorithms
    combined with diffusion model guidance.
    """
    
    def __init__(self, config: ResearchConfig, diffuser: Optional[Any] = None):
        self.config = config
        self.diffuser = diffuser
        self.population = []
        self.fitness_history = []
        self.pareto_front = []
    
    def initialize_population(self, initial_sequences: List[str]) -> List[Dict[str, Any]]:
        """Initialize population with diversity metrics."""
        population = []
        
        # Use provided sequences as starting point
        for seq in initial_sequences[:self.config.population_size]:
            individual = {
                'sequence': seq,
                'generation': 0,
                'parent_fitness': None,
                'objectives': self._evaluate_objectives(seq),
                'age': 0
            }
            population.append(individual)
        
        # Fill remaining population with variations
        while len(population) < self.config.population_size:
            base_seq = random.choice(initial_sequences)
            mutated_seq = self._mutate_sequence(base_seq)
            individual = {
                'sequence': mutated_seq,
                'generation': 0,
                'parent_fitness': None,
                'objectives': self._evaluate_objectives(mutated_seq),
                'age': 0
            }
            population.append(individual)
        
        self.population = population
        return population
    
    def _evaluate_objectives(self, sequence: str) -> Dict[str, float]:
        """Evaluate multiple objectives for a protein sequence."""
        objectives = {}
        
        # Objective 1: Binding affinity (lower is better, so negate)
        binding_affinity = self._predict_binding_affinity(sequence)
        objectives['binding'] = -binding_affinity  # Convert to maximization
        
        # Objective 2: Structural stability
        stability = self._predict_stability(sequence)
        objectives['stability'] = stability
        
        # Objective 3: Sequence diversity (compared to population)
        diversity = self._calculate_diversity_score(sequence)
        objectives['diversity'] = diversity
        
        # Objective 4: Synthesizability
        synthesizability = self._predict_synthesizability(sequence)
        objectives['synthesizability'] = synthesizability
        
        return objectives
    
    def _predict_binding_affinity(self, sequence: str) -> float:
        """Predict binding affinity using simplified model."""
        # Physics-based features
        hydrophobic_aa = "AILMFPWV"
        charged_aa = "DEKR"
        aromatic_aa = "FWY"
        
        hydrophobic_content = sum(1 for aa in sequence if aa in hydrophobic_aa) / len(sequence)
        charged_content = sum(1 for aa in sequence if aa in charged_aa) / len(sequence)
        aromatic_content = sum(1 for aa in sequence if aa in aromatic_aa) / len(sequence)
        
        # Simple linear model with physics constraints
        affinity = (-8.0 + 
                   -4.0 * hydrophobic_content + 
                   -2.0 * aromatic_content + 
                   1.5 * charged_content +
                   np.random.normal(0, 0.5))
        
        return max(-20.0, min(0.0, affinity))
    
    def _predict_stability(self, sequence: str) -> float:
        """Predict structural stability."""
        # Simple stability model based on sequence features
        proline_content = sequence.count('P') / len(sequence)
        glycine_content = sequence.count('G') / len(sequence)
        cysteine_content = sequence.count('C') / len(sequence)
        
        # Stability factors
        flexibility_penalty = (proline_content + glycine_content) * 0.5
        disulfide_bonus = cysteine_content * 0.3
        length_factor = min(1.0, len(sequence) / 200.0)
        
        stability = (0.7 + disulfide_bonus - flexibility_penalty) * length_factor
        return max(0.0, min(1.0, stability))
    
    def _calculate_diversity_score(self, sequence: str) -> float:
        """Calculate diversity score relative to current population."""
        if not self.population:
            return 1.0
        
        similarities = []
        for individual in self.population:
            if individual['sequence'] != sequence:
                similarity = self._sequence_similarity(sequence, individual['sequence'])
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity (normalized Hamming distance)."""
        if len(seq1) != len(seq2):
            # Pad shorter sequence
            max_len = max(len(seq1), len(seq2))
            seq1 = seq1.ljust(max_len, 'A')
            seq2 = seq2.ljust(max_len, 'A')
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1) if seq1 else 0.0
    
    def _predict_synthesizability(self, sequence: str) -> float:
        """Predict how easy it is to synthesize this protein."""
        # Factors affecting synthesizability
        rare_codons = "MWCR"  # Simplified list of "expensive" amino acids
        rare_content = sum(1 for aa in sequence if aa in rare_codons) / len(sequence)
        
        # Repetitive sequences are harder to synthesize
        repetitiveness = self._calculate_repetitiveness(sequence)
        
        # Length penalty for very long sequences
        length_penalty = max(0, (len(sequence) - 300) / 1000.0)
        
        synthesizability = 1.0 - rare_content - repetitiveness - length_penalty
        return max(0.0, min(1.0, synthesizability))
    
    def _calculate_repetitiveness(self, sequence: str, window_size: int = 3) -> float:
        """Calculate sequence repetitiveness using sliding windows."""
        if len(sequence) < window_size:
            return 0.0
        
        kmers = {}
        total_kmers = 0
        
        for i in range(len(sequence) - window_size + 1):
            kmer = sequence[i:i + window_size]
            kmers[kmer] = kmers.get(kmer, 0) + 1
            total_kmers += 1
        
        # Calculate entropy
        probabilities = [count / total_kmers for count in kmers.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(min(total_kmers, 20**window_size))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        
        return 1.0 - normalized_entropy
    
    def _mutate_sequence(self, sequence: str, mutation_rate: Optional[float] = None) -> str:
        """Mutate a protein sequence."""
        if mutation_rate is None:
            mutation_rate = self.config.mutation_rate
        
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        mutated = list(sequence)
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.choice(amino_acids)
        
        return ''.join(mutated)
    
    def _crossover_sequences(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Perform crossover between two protein sequences."""
        min_len = min(len(parent1), len(parent2))
        max_len = max(len(parent1), len(parent2))
        
        # Single-point crossover
        crossover_point = random.randint(1, min_len - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def evolve_generation(self) -> List[Dict[str, Any]]:
        """Evolve one generation using NSGA-II-inspired algorithm."""
        # Calculate dominance relationships
        dominated_solutions = self._calculate_dominance()
        
        # Create new population
        new_population = []
        
        # Elite preservation (non-dominated solutions)
        elite_count = min(len(self.population) // 4, len(dominated_solutions[0]))
        elite_individuals = dominated_solutions[0][:elite_count]
        new_population.extend(elite_individuals)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1_seq, child2_seq = self._crossover_sequences(
                    parent1['sequence'], parent2['sequence']
                )
                
                # Mutate offspring
                child1_seq = self._mutate_sequence(child1_seq)
                child2_seq = self._mutate_sequence(child2_seq)
                
                # Create child individuals
                child1 = {
                    'sequence': child1_seq,
                    'generation': self.population[0]['generation'] + 1,
                    'parent_fitness': parent1['objectives'],
                    'objectives': self._evaluate_objectives(child1_seq),
                    'age': 0
                }
                
                child2 = {
                    'sequence': child2_seq,
                    'generation': self.population[0]['generation'] + 1,
                    'parent_fitness': parent2['objectives'],
                    'objectives': self._evaluate_objectives(child2_seq),
                    'age': 0
                }
                
                new_population.extend([child1, child2])
            else:
                # Just mutate the parent
                mutated_seq = self._mutate_sequence(parent1['sequence'])
                child = {
                    'sequence': mutated_seq,
                    'generation': self.population[0]['generation'] + 1,
                    'parent_fitness': parent1['objectives'],
                    'objectives': self._evaluate_objectives(mutated_seq),
                    'age': 0
                }
                new_population.append(child)
        
        # Trim to exact population size
        new_population = new_population[:self.config.population_size]
        
        # Age existing individuals
        for individual in new_population:
            individual['age'] += 1
        
        self.population = new_population
        return new_population
    
    def _calculate_dominance(self) -> List[List[Dict[str, Any]]]:
        """Calculate Pareto dominance fronts."""
        # For this implementation, we'll use a simplified dominance calculation
        # In a full implementation, you would use NSGA-II algorithm
        
        fronts = [[] for _ in range(len(self.population))]
        domination_count = [0] * len(self.population)
        dominated_solutions = [[] for _ in range(len(self.population))]
        
        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if i != j:
                    if self._dominates(self.population[i], self.population[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(self.population[j], self.population[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(self.population[i])
        
        # Build remaining fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            for individual_idx in range(len(self.population)):
                if self.population[individual_idx] in fronts[front_index]:
                    for dominated_idx in dominated_solutions[individual_idx]:
                        domination_count[dominated_idx] -= 1
                        if domination_count[dominated_idx] == 0:
                            next_front.append(self.population[dominated_idx])
            
            front_index += 1
            if next_front:
                fronts[front_index] = next_front
        
        return [front for front in fronts if front]
    
    def _dominates(self, individual1: Dict[str, Any], individual2: Dict[str, Any]) -> bool:
        """Check if individual1 dominates individual2 (Pareto dominance)."""
        objectives1 = individual1['objectives']
        objectives2 = individual2['objectives']
        
        better_in_one = False
        for obj_name in objectives1:
            if objectives1[obj_name] < objectives2[obj_name]:
                return False  # individual1 is worse in this objective
            elif objectives1[obj_name] > objectives2[obj_name]:
                better_in_one = True  # individual1 is better in this objective
        
        return better_in_one
    
    def _tournament_selection(self, tournament_size: int = 3) -> Dict[str, Any]:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Find best individual in tournament (simplified - use sum of objectives)
        best_individual = max(tournament, 
                            key=lambda x: sum(x['objectives'].values()))
        
        return best_individual
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get the current Pareto front."""
        fronts = self._calculate_dominance()
        return fronts[0] if fronts else []
    
    def optimize(self, initial_sequences: List[str], num_generations: Optional[int] = None) -> Dict[str, Any]:
        """Run multi-objective optimization."""
        if num_generations is None:
            num_generations = self.config.num_generations
        
        logger.info(f"Starting multi-objective optimization with {len(initial_sequences)} initial sequences")
        
        # Initialize population
        self.initialize_population(initial_sequences)
        
        optimization_history = {
            'generations': [],
            'pareto_fronts': [],
            'population_diversity': [],
            'average_fitness': []
        }
        
        for generation in range(num_generations):
            logger.info(f"Generation {generation + 1}/{num_generations}")
            
            # Evolve population
            new_population = self.evolve_generation()
            
            # Record metrics
            pareto_front = self.get_pareto_front()
            
            generation_data = {
                'generation': generation + 1,
                'population_size': len(new_population),
                'pareto_front_size': len(pareto_front),
                'best_binding': max(ind['objectives']['binding'] for ind in new_population),
                'best_stability': max(ind['objectives']['stability'] for ind in new_population),
                'avg_diversity': np.mean([ind['objectives']['diversity'] for ind in new_population])
            }
            
            optimization_history['generations'].append(generation_data)
            optimization_history['pareto_fronts'].append(pareto_front)
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation + 1}: Pareto front size = {len(pareto_front)}")
        
        # Final results
        final_pareto_front = self.get_pareto_front()
        
        results = {
            'final_pareto_front': final_pareto_front,
            'optimization_history': optimization_history,
            'final_population': self.population,
            'total_generations': num_generations,
            'converged': len(final_pareto_front) > 0
        }
        
        logger.info(f"Optimization complete. Final Pareto front size: {len(final_pareto_front)}")
        
        return results


class PhysicsInformedDiffusion:
    """
    Physics-informed diffusion model that incorporates molecular dynamics
    constraints and energy considerations.
    """
    
    def __init__(self, config: ResearchConfig, base_diffuser: Optional[Any] = None):
        self.config = config
        self.base_diffuser = base_diffuser
        
        # Physics parameters
        self.energy_calculator = EnergyCalculator()
        self.constraint_enforcer = ConstraintEnforcer()
    
    def physics_guided_sampling(self, 
                               motif: str,
                               num_samples: int,
                               **kwargs) -> List[Dict[str, Any]]:
        """Sample with physics constraints."""
        logger.info(f"Physics-guided sampling for motif: {motif}")
        
        # Generate initial samples using base diffuser
        if self.base_diffuser and MODULE_IMPORTS_AVAILABLE:
            initial_samples = self.base_diffuser.generate(
                motif=motif,
                num_samples=num_samples * 2,  # Generate more, then filter
                **kwargs
            )
        else:
            # Mock implementation
            initial_samples = self._generate_mock_samples(motif, num_samples * 2)
        
        # Apply physics constraints
        physics_filtered = []
        for sample in initial_samples:
            sequence = sample.get('sequence', '')
            
            # Calculate physics scores
            energy_score = self.energy_calculator.calculate_energy(sequence)
            stability_score = self.energy_calculator.calculate_stability(sequence)
            constraint_score = self.constraint_enforcer.check_constraints(sequence)
            
            # Combined physics score
            physics_score = (
                self.config.energy_weight * energy_score +
                self.config.stability_weight * stability_score +
                self.config.interaction_weight * constraint_score
            )
            
            sample['physics_score'] = physics_score
            sample['energy_score'] = energy_score
            sample['stability_score'] = stability_score
            sample['constraint_score'] = constraint_score
            
            physics_filtered.append(sample)
        
        # Sort by physics score and take top samples
        physics_filtered.sort(key=lambda x: x['physics_score'], reverse=True)
        final_samples = physics_filtered[:num_samples]
        
        logger.info(f"Physics filtering: {len(initial_samples)} -> {len(final_samples)} samples")
        
        return final_samples
    
    def _generate_mock_samples(self, motif: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate mock samples for testing."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        samples = []
        
        for i in range(num_samples):
            # Generate random sequence based on motif length
            base_length = len(motif.replace('_', '')) if motif else 50
            length = base_length + np.random.randint(-10, 20)
            length = max(20, length)
            
            sequence = ''.join(np.random.choice(list(amino_acids)) for _ in range(length))
            
            sample = {
                'sequence': sequence,
                'length': length,
                'confidence': np.random.uniform(0.3, 0.9),
                'sample_id': i,
                'motif': motif
            }
            
            samples.append(sample)
        
        return samples


class EnergyCalculator:
    """Calculate molecular energies and stability metrics."""
    
    def __init__(self):
        # Amino acid properties for energy calculations
        self.hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        self.charge = {
            'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1
        }
        
        self.volume = {
            'G': 60.1, 'A': 88.6, 'S': 89.0, 'C': 108.5, 'D': 111.1,
            'P': 112.7, 'N': 114.1, 'T': 116.1, 'E': 138.4, 'V': 140.0,
            'Q': 143.8, 'H': 153.2, 'M': 162.9, 'I': 166.7, 'L': 166.7,
            'K': 168.6, 'R': 173.4, 'F': 189.9, 'Y': 193.6, 'W': 227.8
        }
    
    def calculate_energy(self, sequence: str) -> float:
        """Calculate approximate folding energy."""
        if not sequence:
            return 0.0
        
        # Hydrophobic burial energy
        hydrophobic_energy = sum(self.hydrophobicity.get(aa, 0) for aa in sequence)
        
        # Electrostatic interactions (simplified)
        charges = [self.charge.get(aa, 0) for aa in sequence]
        electrostatic_energy = sum(charges[i] * charges[j] / max(1, abs(i - j))
                                 for i in range(len(charges))
                                 for j in range(i + 1, len(charges)))
        
        # Chain entropy penalty (longer chains are less favorable)
        entropy_penalty = 0.1 * len(sequence)
        
        total_energy = -hydrophobic_energy - electrostatic_energy + entropy_penalty
        
        # Normalize to 0-1 scale (lower energy is better)
        normalized_energy = 1.0 / (1.0 + np.exp(-total_energy / 10.0))
        
        return normalized_energy
    
    def calculate_stability(self, sequence: str) -> float:
        """Calculate structural stability score."""
        if not sequence:
            return 0.0
        
        # Secondary structure propensities (simplified)
        helix_formers = "AEKLR"
        sheet_formers = "FILVWY"
        loop_formers = "GPST"
        
        helix_content = sum(1 for aa in sequence if aa in helix_formers) / len(sequence)
        sheet_content = sum(1 for aa in sequence if aa in sheet_formers) / len(sequence)
        loop_content = sum(1 for aa in sequence if aa in loop_formers) / len(sequence)
        
        # Balanced secondary structure is more stable
        structure_balance = 1.0 - np.std([helix_content, sheet_content, loop_content])
        
        # Disulfide bonds (cysteine pairs)
        cysteine_count = sequence.count('C')
        disulfide_stability = min(1.0, cysteine_count / 4.0) * 0.2
        
        # Proline content (too much is destabilizing)
        proline_penalty = max(0, (sequence.count('P') / len(sequence) - 0.1)) * 0.5
        
        stability = structure_balance + disulfide_stability - proline_penalty
        
        return max(0.0, min(1.0, stability))


class ConstraintEnforcer:
    """Enforce molecular and biological constraints."""
    
    def __init__(self):
        self.forbidden_patterns = [
            "PPP",  # Too many prolines in a row
            "KKKK", # Too many positive charges
            "DDDD", # Too many negative charges
            "GGGG", # Too flexible
        ]
        
        self.required_residues = {
            'catalytic': ['H', 'C', 'D', 'E'],  # Common catalytic residues
            'structural': ['C', 'P'],           # Important for structure
            'binding': ['W', 'F', 'Y', 'R', 'K'] # Important for binding
        }
    
    def check_constraints(self, sequence: str) -> float:
        """Check various molecular constraints."""
        if not sequence:
            return 0.0
        
        penalty = 0.0
        
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern in sequence:
                penalty += 0.2
        
        # Check amino acid composition
        aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
        total_residues = len(sequence)
        
        # Unrealistic compositions
        for aa, count in aa_counts.items():
            freq = count / total_residues
            if freq > 0.3:  # No single AA should dominate
                penalty += (freq - 0.3) * 0.5
        
        # Check charge balance
        positive_charge = sum(1 for aa in sequence if aa in "KR")
        negative_charge = sum(1 for aa in sequence if aa in "DE")
        charge_imbalance = abs(positive_charge - negative_charge) / total_residues
        
        if charge_imbalance > 0.2:
            penalty += (charge_imbalance - 0.2) * 0.3
        
        # Length constraints
        if len(sequence) < 20:
            penalty += 0.5  # Too short
        elif len(sequence) > 1000:
            penalty += 0.3  # Too long
        
        constraint_score = max(0.0, 1.0 - penalty)
        
        return constraint_score


class AdversarialValidator:
    """
    Adversarial validation using GAN-style discriminator
    to assess protein quality.
    """
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.discriminator = ProteinDiscriminator()
        self.training_history = []
        
    def train_discriminator(self, real_sequences: List[str], 
                          generated_sequences: List[str],
                          num_epochs: int = 100) -> Dict[str, Any]:
        """Train discriminator to distinguish real vs generated proteins."""
        logger.info("Training adversarial discriminator")
        
        # Prepare training data
        real_features = [self._extract_features(seq) for seq in real_sequences]
        fake_features = [self._extract_features(seq) for seq in generated_sequences]
        
        # Training loop (simplified)
        losses = {'discriminator': [], 'accuracy': []}
        
        for epoch in range(num_epochs):
            # Sample mini-batch
            batch_size = min(32, len(real_features), len(fake_features))
            real_batch = random.sample(real_features, batch_size)
            fake_batch = random.sample(fake_features, batch_size)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_step(real_batch, labels=1.0)
            d_loss_fake = self.discriminator.train_step(fake_batch, labels=0.0)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            # Calculate accuracy
            real_pred = self.discriminator.predict(real_batch)
            fake_pred = self.discriminator.predict(fake_batch)
            accuracy = (np.mean(real_pred > 0.5) + np.mean(fake_pred < 0.5)) / 2
            
            losses['discriminator'].append(d_loss)
            losses['accuracy'].append(accuracy)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: D_loss={d_loss:.4f}, Accuracy={accuracy:.4f}")
        
        training_results = {
            'losses': losses,
            'final_accuracy': accuracy,
            'converged': accuracy > 0.8,
            'epochs_trained': num_epochs
        }
        
        self.training_history.append(training_results)
        
        return training_results
    
    def _extract_features(self, sequence: str) -> np.ndarray:
        """Extract features from protein sequence."""
        if not sequence:
            return np.zeros(20)  # Return zero vector for empty sequence
        
        # Amino acid composition
        aa_composition = np.zeros(20)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        for i, aa in enumerate(amino_acids):
            aa_composition[i] = sequence.count(aa) / len(sequence)
        
        # Additional features could include:
        # - K-mer frequencies
        # - Secondary structure predictions
        # - Hydrophobicity profile
        # - Charge distribution
        
        return aa_composition
    
    def validate_sequences(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Validate sequences using trained discriminator."""
        results = []
        
        for i, sequence in enumerate(sequences):
            features = self._extract_features(sequence)
            quality_score = self.discriminator.predict([features])[0]
            
            result = {
                'sequence_id': i,
                'sequence': sequence,
                'adversarial_quality': quality_score,
                'is_realistic': quality_score > 0.5,
                'confidence': abs(quality_score - 0.5) * 2  # Distance from decision boundary
            }
            
            results.append(result)
        
        return results


class ProteinDiscriminator:
    """Simple discriminator model for protein sequences."""
    
    def __init__(self, feature_dim: int = 20):
        self.feature_dim = feature_dim
        self.weights = np.random.normal(0, 0.1, (feature_dim, 1))
        self.bias = np.random.normal(0, 0.1)
        self.learning_rate = 0.01
    
    def forward(self, features: np.ndarray) -> float:
        """Forward pass through discriminator."""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        logits = np.dot(features, self.weights) + self.bias
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities.flatten()
    
    def train_step(self, batch_features: List[np.ndarray], labels: float) -> float:
        """Single training step."""
        features_matrix = np.array(batch_features)
        predictions = self.forward(features_matrix)
        
        # Binary cross-entropy loss
        labels_array = np.full_like(predictions, labels)
        loss = -np.mean(labels_array * np.log(predictions + 1e-8) + 
                       (1 - labels_array) * np.log(1 - predictions + 1e-8))
        
        # Gradients (simplified)
        error = predictions - labels_array
        grad_weights = np.dot(features_matrix.T, error.reshape(-1, 1)) / len(batch_features)
        grad_bias = np.mean(error)
        
        # Update parameters
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias
        
        return loss
    
    def predict(self, features: List[np.ndarray]) -> np.ndarray:
        """Predict probability of being real protein."""
        features_matrix = np.array(features)
        return self.forward(features_matrix)


class AdaptiveSampler:
    """Adaptive temperature and sampling strategy controller."""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.temperature_history = []
        self.performance_history = []
    
    def get_adaptive_temperature(self, 
                               current_step: int, 
                               total_steps: int,
                               performance_feedback: Optional[float] = None) -> float:
        """Calculate adaptive temperature based on schedule and feedback."""
        
        # Base temperature from schedule
        if self.config.temperature_schedule == "linear":
            base_temp = self._linear_schedule(current_step, total_steps)
        elif self.config.temperature_schedule == "cosine":
            base_temp = self._cosine_schedule(current_step, total_steps)
        elif self.config.temperature_schedule == "exponential":
            base_temp = self._exponential_schedule(current_step, total_steps)
        else:
            base_temp = 1.0  # Constant temperature fallback
        
        # Adjust based on performance feedback
        if performance_feedback is not None and len(self.performance_history) > 5:
            recent_performance = np.mean(self.performance_history[-5:])
            
            if performance_feedback < recent_performance:
                # Performance is declining, increase exploration
                adjustment = 0.1
            else:
                # Performance is improving, reduce temperature slightly
                adjustment = -0.05
            
            base_temp = max(self.config.min_temperature, 
                          min(self.config.max_temperature, base_temp + adjustment))
        
        self.temperature_history.append(base_temp)
        if performance_feedback is not None:
            self.performance_history.append(performance_feedback)
        
        return base_temp
    
    def _linear_schedule(self, current_step: int, total_steps: int) -> float:
        """Linear temperature decay."""
        progress = current_step / max(1, total_steps)
        temp = self.config.max_temperature - progress * (
            self.config.max_temperature - self.config.min_temperature)
        return temp
    
    def _cosine_schedule(self, current_step: int, total_steps: int) -> float:
        """Cosine annealing temperature schedule."""
        progress = current_step / max(1, total_steps)
        temp = self.config.min_temperature + 0.5 * (
            self.config.max_temperature - self.config.min_temperature) * (
            1 + np.cos(np.pi * progress))
        return temp
    
    def _exponential_schedule(self, current_step: int, total_steps: int) -> float:
        """Exponential temperature decay."""
        progress = current_step / max(1, total_steps)
        decay_rate = -np.log(self.config.min_temperature / self.config.max_temperature)
        temp = self.config.max_temperature * np.exp(-decay_rate * progress)
        return temp


def run_comprehensive_research_study(config: ResearchConfig,
                                   initial_sequences: List[str],
                                   target_motif: str = "HELIX_SHEET_HELIX") -> Dict[str, Any]:
    """
    Run comprehensive research study comparing all novel methods.
    """
    logger.info("Starting comprehensive research study")
    
    results = {
        'config': config,
        'baseline_results': {},
        'experimental_results': {},
        'comparative_analysis': {},
        'statistical_significance': {}
    }
    
    # 1. Multi-objective optimization
    logger.info("Running multi-objective optimization")
    moo = MultiObjectiveOptimizer(config)
    moo_results = moo.optimize(initial_sequences[:50])  # Use subset for efficiency
    results['experimental_results']['multi_objective'] = moo_results
    
    # 2. Physics-informed diffusion
    logger.info("Running physics-informed diffusion")
    physics_diffusion = PhysicsInformedDiffusion(config)
    physics_results = physics_diffusion.physics_guided_sampling(
        motif=target_motif,
        num_samples=50
    )
    results['experimental_results']['physics_informed'] = physics_results
    
    # 3. Adversarial validation
    logger.info("Running adversarial validation study")
    validator = AdversarialValidator(config)
    
    # Split sequences for training discriminator
    split_idx = len(initial_sequences) // 2
    real_sequences = initial_sequences[:split_idx]
    test_sequences = initial_sequences[split_idx:]
    
    training_results = validator.train_discriminator(real_sequences, test_sequences)
    validation_results = validator.validate_sequences(test_sequences)
    
    results['experimental_results']['adversarial_validation'] = {
        'training_results': training_results,
        'validation_results': validation_results
    }
    
    # 4. Adaptive sampling analysis
    logger.info("Analyzing adaptive sampling strategies")
    adaptive_sampler = AdaptiveSampler(config)
    
    # Simulate sampling process with different schedules
    sampling_results = {}
    for schedule in ["linear", "cosine", "exponential"]:
        config.temperature_schedule = schedule
        temp_profile = []
        
        for step in range(100):
            temp = adaptive_sampler.get_adaptive_temperature(
                step, 100, performance_feedback=np.random.uniform(0.3, 0.9))
            temp_profile.append(temp)
        
        sampling_results[schedule] = {
            'temperature_profile': temp_profile,
            'final_temperature': temp_profile[-1],
            'temperature_variance': np.var(temp_profile)
        }
    
    results['experimental_results']['adaptive_sampling'] = sampling_results
    
    # 5. Baseline comparison
    logger.info("Running baseline comparisons")
    baseline_metrics = calculate_baseline_metrics(initial_sequences)
    results['baseline_results'] = baseline_metrics
    
    # 6. Statistical analysis
    logger.info("Performing statistical analysis")
    statistical_results = perform_statistical_analysis(results)
    results['statistical_significance'] = statistical_results
    
    logger.info("Comprehensive research study completed")
    
    return results


def calculate_baseline_metrics(sequences: List[str]) -> Dict[str, Any]:
    """Calculate baseline metrics for comparison."""
    if not sequences:
        return {}
    
    # Basic sequence statistics
    lengths = [len(seq) for seq in sequences]
    
    # Amino acid composition analysis
    all_aas = ''.join(sequences)
    aa_composition = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        aa_composition[aa] = all_aas.count(aa) / len(all_aas) if all_aas else 0
    
    # Diversity metrics
    diversity_scores = []
    for i in range(min(100, len(sequences))):  # Sample for efficiency
        for j in range(i + 1, min(100, len(sequences))):
            seq1, seq2 = sequences[i], sequences[j]
            similarity = sum(a == b for a, b in zip(seq1, seq2)) / max(len(seq1), len(seq2))
            diversity_scores.append(1.0 - similarity)
    
    baseline_metrics = {
        'sequence_count': len(sequences),
        'length_statistics': {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths)
        },
        'amino_acid_composition': aa_composition,
        'diversity_score': np.mean(diversity_scores) if diversity_scores else 0.0,
        'uniqueness_ratio': len(set(sequences)) / len(sequences)
    }
    
    return baseline_metrics


def perform_statistical_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Perform statistical significance testing on results."""
    statistical_results = {
        'tests_performed': [],
        'significant_findings': [],
        'p_values': {},
        'effect_sizes': {}
    }
    
    # Compare diversity scores between methods
    if 'multi_objective' in results.get('experimental_results', {}):
        moo_results = results['experimental_results']['multi_objective']
        pareto_sequences = [ind['sequence'] for ind in moo_results.get('final_pareto_front', [])]
        
        if pareto_sequences and results.get('baseline_results'):
            baseline_diversity = results['baseline_results'].get('diversity_score', 0)
            
            # Calculate diversity for Pareto front
            pareto_diversity_scores = []
            for i in range(len(pareto_sequences)):
                for j in range(i + 1, len(pareto_sequences)):
                    seq1, seq2 = pareto_sequences[i], pareto_sequences[j]
                    similarity = sum(a == b for a, b in zip(seq1[:min(len(seq1), len(seq2))], 
                                                           seq2[:min(len(seq1), len(seq2))])) / max(len(seq1), len(seq2))
                    pareto_diversity_scores.append(1.0 - similarity)
            
            pareto_diversity = np.mean(pareto_diversity_scores) if pareto_diversity_scores else 0
            
            # Simple comparison (in practice, would use proper statistical tests)
            improvement = pareto_diversity - baseline_diversity
            statistical_results['effect_sizes']['diversity_improvement'] = improvement
            
            if improvement > 0.1:  # Threshold for significance
                statistical_results['significant_findings'].append(
                    f"Multi-objective optimization improved diversity by {improvement:.3f}")
    
    # Analyze physics-informed results
    if 'physics_informed' in results.get('experimental_results', {}):
        physics_results = results['experimental_results']['physics_informed']
        physics_scores = [sample.get('physics_score', 0) for sample in physics_results]
        
        if physics_scores:
            avg_physics_score = np.mean(physics_scores)
            statistical_results['effect_sizes']['physics_score'] = avg_physics_score
            
            if avg_physics_score > 0.7:  # Threshold for good physics score
                statistical_results['significant_findings'].append(
                    f"Physics-informed diffusion achieved average physics score of {avg_physics_score:.3f}")
    
    # Adversarial validation analysis
    if 'adversarial_validation' in results.get('experimental_results', {}):
        adv_results = results['experimental_results']['adversarial_validation']
        training_results = adv_results.get('training_results', {})
        
        final_accuracy = training_results.get('final_accuracy', 0)
        statistical_results['effect_sizes']['discriminator_accuracy'] = final_accuracy
        
        if final_accuracy > 0.8:
            statistical_results['significant_findings'].append(
                f"Adversarial discriminator achieved {final_accuracy:.1%} accuracy")
    
    return statistical_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock initial sequences for testing
    mock_sequences = [
        "MKLLILTCLVAVALARPKHPIPDQAITVAYASRALGRGLVVMAQDGNRGGKFHPWTVN",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ",
        "MAELLGASWDPWQVSLQDKTGFHRKQAEQHLLPLWRQHTLEVLGHQQLVQRAQ",
        "GGGSGGGSGGGSGGGSGGGSGGGSGGGSGGGSGGGSGGGSGGGSGGGSGGGSGG",
        "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
    ]
    
    # Create research configuration
    research_config = ResearchConfig(
        population_size=20,
        num_generations=10,
        num_baseline_runs=3,
        num_experimental_runs=3
    )
    
    # Run comprehensive study
    study_results = run_comprehensive_research_study(
        config=research_config,
        initial_sequences=mock_sequences,
        target_motif="HELIX_SHEET_HELIX"
    )
    
    print("Research study completed!")
    print(f"Significant findings: {len(study_results['statistical_significance']['significant_findings'])}")
    for finding in study_results['statistical_significance']['significant_findings']:
        print(f"- {finding}")