"""
Neural Evolution V2 - Advanced Evolutionary Algorithms for Protein Design

Generation 4 enhancement featuring:
- Neuroevolutionary protein optimization
- Population-based training with neural networks
- Genetic programming for protein design
- Co-evolutionary dynamics
- Multi-objective evolutionary optimization
"""

import asyncio
import json
import time
import uuid
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import itertools
from collections import defaultdict
import random

try:
    from .mock_torch import nn, MockTensor as torch_tensor, tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """Types of evolutionary strategies."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    EVOLUTION_STRATEGY = "evolution_strategy"
    GENETIC_PROGRAMMING = "genetic_programming"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"
    NEUROEVOLUTION = "neuroevolution"
    COEVOLUTION = "coevolution"
    MULTI_OBJECTIVE = "multi_objective"


class SelectionMethod(Enum):
    """Selection methods for evolution."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    TRUNCATION = "truncation"
    ELITISM = "elitism"
    PARETO = "pareto"


@dataclass
class Individual:
    """Individual in evolutionary population."""
    individual_id: str
    genotype: List[Any]  # Genetic representation
    phenotype: str  # Protein sequence
    fitness: Dict[str, float]
    neural_network: Optional[Any] = None  # Neural network for neuroevolution
    age: int = 0
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "individual_id": self.individual_id,
            "genotype": self.genotype,
            "phenotype": self.phenotype,
            "fitness": self.fitness,
            "age": self.age,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithms."""
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 5
    elitism_ratio: float = 0.1
    diversity_threshold: float = 0.1
    fitness_objectives: List[str] = field(default_factory=lambda: ["stability", "novelty"])
    neural_architecture: Dict[str, Any] = field(default_factory=dict)
    max_sequence_length: int = 200
    amino_acid_alphabet: str = "ACDEFGHIKLMNPQRSTVWY"
    parallel_evaluation: bool = True
    adaptive_parameters: bool = True
    coevolution_enabled: bool = False


class FitnessEvaluator:
    """Evaluates fitness of protein sequences."""
    
    def __init__(self):
        self.evaluation_cache = {}
        self.evaluation_count = 0
        
    async def evaluate_fitness(
        self,
        sequence: str,
        objectives: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate fitness of protein sequence."""
        
        cache_key = f"{sequence}_{'-'.join(objectives or [])}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        self.evaluation_count += 1
        
        # Simulate computational time for fitness evaluation
        await asyncio.sleep(0.01)
        
        objectives = objectives or ["stability", "novelty", "solubility", "bindability"]
        fitness = {}
        
        # Mock fitness calculations with realistic protein properties
        for objective in objectives:
            if objective == "stability":
                # Stability based on hydrophobic core and secondary structure
                hydrophobic_aa = "AILMFPWYV"
                hydrophobic_ratio = sum(1 for aa in sequence if aa in hydrophobic_aa) / len(sequence)
                fitness["stability"] = min(1.0, hydrophobic_ratio * 2.0 + np.random.normal(0, 0.1))
                
            elif objective == "novelty":
                # Novelty based on sequence uniqueness
                common_patterns = ["AAA", "GGG", "PPP", "CCC"]
                pattern_penalty = sum(sequence.count(p) for p in common_patterns)
                fitness["novelty"] = max(0.0, 1.0 - pattern_penalty * 0.1 + np.random.normal(0, 0.05))
                
            elif objective == "solubility":
                # Solubility based on charged and polar residues
                charged_aa = "DEKR"
                polar_aa = "STNQH"
                soluble_ratio = sum(1 for aa in sequence if aa in charged_aa + polar_aa) / len(sequence)
                fitness["solubility"] = min(1.0, soluble_ratio * 1.5 + np.random.normal(0, 0.08))
                
            elif objective == "bindability":
                # Binding potential based on aromatic and charged residues
                binding_aa = "FYWHR"
                binding_ratio = sum(1 for aa in sequence if aa in binding_aa) / len(sequence)
                fitness["bindability"] = min(1.0, binding_ratio * 2.5 + np.random.normal(0, 0.12))
                
            else:
                # Generic objective
                fitness[objective] = np.random.uniform(0.3, 0.9)
        
        # Add composite fitness score
        fitness["composite"] = np.mean(list(fitness.values()))
        
        self.evaluation_cache[cache_key] = fitness
        return fitness
    
    def get_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """Get Pareto front from population for multi-objective optimization."""
        pareto_front = []
        
        for individual in population:
            is_dominated = False
            
            for other in population:
                if individual.individual_id == other.individual_id:
                    continue
                
                # Check if other dominates individual
                dominates = True
                for objective in individual.fitness:
                    if objective == "composite":
                        continue
                    if other.fitness[objective] <= individual.fitness[objective]:
                        dominates = False
                        break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(individual)
        
        return pareto_front


class NeuroevolutionNetwork:
    """Neural network for neuroevolution."""
    
    def __init__(self, input_size: int = 20, hidden_sizes: List[int] = None, output_size: int = 20):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or [64, 32]
        self.output_size = output_size
        self.weights = self._initialize_weights()
        self.fitness_history = []
        
    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize neural network weights."""
        weights = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.normal(0, 0.1, (layer_sizes[i], layer_sizes[i + 1]))
            bias_vector = np.random.normal(0, 0.01, (1, layer_sizes[i + 1]))
            weights.append(weight_matrix)
            weights.append(bias_vector)
        
        return weights
    
    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        current_input = input_vector.reshape(1, -1)
        
        for i in range(0, len(self.weights), 2):
            weight_matrix = self.weights[i]
            bias_vector = self.weights[i + 1]
            
            current_input = np.dot(current_input, weight_matrix) + bias_vector
            
            # Apply activation function (except for output layer)
            if i < len(self.weights) - 2:
                current_input = np.tanh(current_input)
            else:
                current_input = np.sigmoid(current_input)  # Output layer
        
        return current_input.flatten()
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Mutate network weights."""
        for weight_matrix in self.weights:
            mask = np.random.random(weight_matrix.shape) < mutation_rate
            weight_matrix[mask] += np.random.normal(0, mutation_strength, weight_matrix[mask].shape)
    
    def crossover(self, other: 'NeuroevolutionNetwork') -> 'NeuroevolutionNetwork':
        """Create offspring through crossover."""
        offspring = NeuroevolutionNetwork(self.input_size, self.hidden_sizes, self.output_size)
        
        for i in range(len(self.weights)):
            # Random crossover mask
            mask = np.random.random(self.weights[i].shape) < 0.5
            offspring.weights[i] = np.where(mask, self.weights[i], other.weights[i])
        
        return offspring


class NeuralEvolutionEngine:
    """Neural evolution engine for protein design."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.fitness_evaluator = FitnessEvaluator()
        self.population: List[Individual] = []
        self.evolution_history: List[Dict[str, Any]] = []
        self.generation_count = 0
        self.best_individuals: List[Individual] = []
        
    async def initialize_population(self) -> List[Individual]:
        """Initialize evolutionary population."""
        logger.info(f"Initializing population of {self.config.population_size} individuals")
        
        population = []
        
        for i in range(self.config.population_size):
            # Generate random protein sequence
            sequence_length = np.random.randint(50, self.config.max_sequence_length)
            sequence = ''.join(np.random.choice(list(self.config.amino_acid_alphabet), sequence_length))
            
            # Create genotype (one-hot encoding of sequence)
            genotype = self._sequence_to_genotype(sequence)
            
            # Create neural network for neuroevolution
            neural_network = None
            if "neuroevolution" in str(self.config.neural_architecture):
                neural_network = NeuroevolutionNetwork(
                    input_size=len(self.config.amino_acid_alphabet),
                    hidden_sizes=self.config.neural_architecture.get("hidden_sizes", [64, 32]),
                    output_size=len(self.config.amino_acid_alphabet)
                )
            
            # Evaluate fitness
            fitness = await self.fitness_evaluator.evaluate_fitness(
                sequence, self.config.fitness_objectives
            )
            
            individual = Individual(
                individual_id=str(uuid.uuid4()),
                genotype=genotype,
                phenotype=sequence,
                fitness=fitness,
                neural_network=neural_network,
                generation=0
            )
            
            population.append(individual)
            
            if (i + 1) % 20 == 0:
                logger.debug(f"Initialized {i + 1}/{self.config.population_size} individuals")
        
        self.population = population
        logger.info(f"Population initialization complete. Average fitness: {np.mean([ind.fitness['composite'] for ind in population]):.3f}")
        
        return population
    
    def _sequence_to_genotype(self, sequence: str) -> List[List[float]]:
        """Convert sequence to genotype representation."""
        aa_to_index = {aa: i for i, aa in enumerate(self.config.amino_acid_alphabet)}
        genotype = []
        
        for aa in sequence:
            if aa in aa_to_index:
                one_hot = [0.0] * len(self.config.amino_acid_alphabet)
                one_hot[aa_to_index[aa]] = 1.0
                genotype.append(one_hot)
            else:
                # Unknown amino acid - random encoding
                genotype.append([np.random.random() for _ in range(len(self.config.amino_acid_alphabet))])
        
        return genotype
    
    def _genotype_to_sequence(self, genotype: List[List[float]]) -> str:
        """Convert genotype to sequence."""
        sequence = []
        for gene in genotype:
            max_index = np.argmax(gene)
            if max_index < len(self.config.amino_acid_alphabet):
                sequence.append(self.config.amino_acid_alphabet[max_index])
            else:
                sequence.append(np.random.choice(list(self.config.amino_acid_alphabet)))
        
        return ''.join(sequence)
    
    async def evolve_generation(self) -> Dict[str, Any]:
        """Evolve one generation."""
        self.generation_count += 1
        logger.info(f"Evolving generation {self.generation_count}")
        
        start_time = time.time()
        
        # Selection
        selected_individuals = self._selection()
        
        # Crossover and Mutation
        offspring = await self._generate_offspring(selected_individuals)
        
        # Combine population (parents + offspring)
        combined_population = self.population + offspring
        
        # Environmental selection (survival of the fittest)
        self.population = self._environmental_selection(combined_population)
        
        # Update generation number
        for individual in self.population:
            individual.generation = self.generation_count
        
        # Track best individuals
        current_best = max(self.population, key=lambda x: x.fitness["composite"])
        if not self.best_individuals or current_best.fitness["composite"] > self.best_individuals[-1].fitness["composite"]:
            self.best_individuals.append(current_best)
        
        # Adaptive parameters
        if self.config.adaptive_parameters:
            self._adapt_parameters()
        
        generation_time = time.time() - start_time
        
        # Statistics
        fitnesses = [ind.fitness["composite"] for ind in self.population]
        generation_stats = {
            "generation": self.generation_count,
            "best_fitness": max(fitnesses),
            "average_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "diversity": self._calculate_diversity(),
            "generation_time": generation_time,
            "total_evaluations": self.fitness_evaluator.evaluation_count
        }
        
        self.evolution_history.append(generation_stats)
        
        logger.info(f"Generation {self.generation_count} complete - Best: {generation_stats['best_fitness']:.3f}, Avg: {generation_stats['average_fitness']:.3f}")
        
        return generation_stats
    
    def _selection(self) -> List[Individual]:
        """Select individuals for reproduction."""
        
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection()
        elif self.config.selection_method == SelectionMethod.RANK:
            return self._rank_selection()
        elif self.config.selection_method == SelectionMethod.PARETO:
            return self._pareto_selection()
        else:
            return self._tournament_selection()  # Default
    
    def _tournament_selection(self) -> List[Individual]:
        """Tournament selection."""
        selected = []
        
        for _ in range(self.config.population_size):
            tournament = np.random.choice(self.population, self.config.tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness["composite"])
            selected.append(winner)
        
        return selected
    
    def _roulette_selection(self) -> List[Individual]:
        """Roulette wheel selection."""
        fitnesses = [max(0.001, ind.fitness["composite"]) for ind in self.population]  # Ensure positive
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        
        selected = []
        for _ in range(self.config.population_size):
            r = np.random.random()
            cumsum = 0
            for i, prob in enumerate(probabilities):
                cumsum += prob
                if r <= cumsum:
                    selected.append(self.population[i])
                    break
        
        return selected
    
    def _rank_selection(self) -> List[Individual]:
        """Rank-based selection."""
        sorted_population = sorted(self.population, key=lambda x: x.fitness["composite"])
        ranks = list(range(1, len(sorted_population) + 1))
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        
        selected = []
        for _ in range(self.config.population_size):
            r = np.random.random()
            cumsum = 0
            for i, prob in enumerate(probabilities):
                cumsum += prob
                if r <= cumsum:
                    selected.append(sorted_population[i])
                    break
        
        return selected
    
    def _pareto_selection(self) -> List[Individual]:
        """Pareto front selection for multi-objective optimization."""
        pareto_front = self.fitness_evaluator.get_pareto_front(self.population)
        
        selected = list(pareto_front)
        
        # Fill remaining slots with tournament selection
        while len(selected) < self.config.population_size:
            tournament = np.random.choice(self.population, min(3, len(self.population)), replace=False)
            winner = max(tournament, key=lambda x: x.fitness["composite"])
            selected.append(winner)
        
        return selected[:self.config.population_size]
    
    async def _generate_offspring(self, selected: List[Individual]) -> List[Individual]:
        """Generate offspring through crossover and mutation."""
        offspring = []
        
        # Elitism - keep best individuals
        elite_count = int(self.config.elitism_ratio * self.config.population_size)
        elite = sorted(self.population, key=lambda x: x.fitness["composite"], reverse=True)[:elite_count]
        
        for individual in elite:
            new_individual = Individual(
                individual_id=str(uuid.uuid4()),
                genotype=individual.genotype.copy(),
                phenotype=individual.phenotype,
                fitness=individual.fitness.copy(),
                neural_network=individual.neural_network,
                parent_ids=[individual.individual_id],
                generation=self.generation_count
            )
            offspring.append(new_individual)
        
        # Generate remaining offspring through crossover and mutation
        while len(offspring) < self.config.population_size:
            
            # Select parents
            parent1 = np.random.choice(selected)
            parent2 = np.random.choice(selected)
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child_genotype = self._crossover(parent1.genotype, parent2.genotype)
                child_neural_network = None
                
                # Neural network crossover for neuroevolution
                if parent1.neural_network and parent2.neural_network:
                    child_neural_network = parent1.neural_network.crossover(parent2.neural_network)
                
            else:
                child_genotype = parent1.genotype.copy()
                child_neural_network = parent1.neural_network
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child_genotype = self._mutate(child_genotype)
                
                if child_neural_network:
                    child_neural_network.mutate(self.config.mutation_rate)
            
            # Convert to phenotype
            child_sequence = self._genotype_to_sequence(child_genotype)
            
            # Evaluate fitness
            child_fitness = await self.fitness_evaluator.evaluate_fitness(
                child_sequence, self.config.fitness_objectives
            )
            
            child = Individual(
                individual_id=str(uuid.uuid4()),
                genotype=child_genotype,
                phenotype=child_sequence,
                fitness=child_fitness,
                neural_network=child_neural_network,
                parent_ids=[parent1.individual_id, parent2.individual_id],
                generation=self.generation_count
            )
            
            offspring.append(child)
        
        return offspring[:self.config.population_size]
    
    def _crossover(self, genotype1: List[List[float]], genotype2: List[List[float]]) -> List[List[float]]:
        """Uniform crossover between two genotypes."""
        min_length = min(len(genotype1), len(genotype2))
        child_genotype = []
        
        for i in range(min_length):
            if np.random.random() < 0.5:
                child_genotype.append(genotype1[i].copy())
            else:
                child_genotype.append(genotype2[i].copy())
        
        return child_genotype
    
    def _mutate(self, genotype: List[List[float]]) -> List[List[float]]:
        """Mutate genotype."""
        mutated_genotype = []
        
        for gene in genotype:
            mutated_gene = gene.copy()
            
            # Point mutation
            for i in range(len(mutated_gene)):
                if np.random.random() < self.config.mutation_rate:
                    mutated_gene[i] = np.random.random()
            
            # Normalize to maintain one-hot-like properties
            total = sum(mutated_gene)
            if total > 0:
                mutated_gene = [x / total for x in mutated_gene]
            
            mutated_genotype.append(mutated_gene)
        
        # Insertion/deletion mutations
        if np.random.random() < 0.05:  # 5% chance
            if np.random.random() < 0.5 and len(mutated_genotype) > 10:
                # Deletion
                del mutated_genotype[np.random.randint(0, len(mutated_genotype))]
            elif len(mutated_genotype) < self.config.max_sequence_length:
                # Insertion
                new_gene = [0.0] * len(self.config.amino_acid_alphabet)
                new_gene[np.random.randint(0, len(self.config.amino_acid_alphabet))] = 1.0
                insert_pos = np.random.randint(0, len(mutated_genotype) + 1)
                mutated_genotype.insert(insert_pos, new_gene)
        
        return mutated_genotype
    
    def _environmental_selection(self, combined_population: List[Individual]) -> List[Individual]:
        """Environmental selection to maintain population size."""
        
        if len(combined_population) <= self.config.population_size:
            return combined_population
        
        # Sort by fitness and select top individuals
        sorted_population = sorted(combined_population, key=lambda x: x.fitness["composite"], reverse=True)
        
        # Maintain diversity by removing similar individuals
        selected = []
        for individual in sorted_population:
            if len(selected) >= self.config.population_size:
                break
            
            # Check diversity
            is_diverse = True
            for selected_ind in selected:
                if self._sequence_similarity(individual.phenotype, selected_ind.phenotype) > (1 - self.config.diversity_threshold):
                    is_diverse = False
                    break
            
            if is_diverse or len(selected) < self.config.population_size // 2:
                selected.append(individual)
        
        # Fill remaining slots with top performers if needed
        while len(selected) < self.config.population_size:
            for individual in sorted_population:
                if individual not in selected:
                    selected.append(individual)
                    break
        
        return selected[:self.config.population_size]
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity."""
        if not seq1 or not seq2:
            return 0.0
        
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return matches / max(len(seq1), len(seq2))
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                sim = self._sequence_similarity(
                    self.population[i].phenotype,
                    self.population[j].phenotype
                )
                similarities.append(sim)
        
        return 1.0 - np.mean(similarities)  # Diversity is 1 - similarity
    
    def _adapt_parameters(self):
        """Adapt evolutionary parameters based on population dynamics."""
        
        if len(self.evolution_history) < 5:
            return
        
        # Check fitness improvement stagnation
        recent_fitness = [gen["best_fitness"] for gen in self.evolution_history[-5:]]
        fitness_improvement = recent_fitness[-1] - recent_fitness[0]
        
        # Adapt mutation rate
        if fitness_improvement < 0.001:  # Stagnation
            self.config.mutation_rate = min(0.5, self.config.mutation_rate * 1.1)
            logger.debug(f"Increased mutation rate to {self.config.mutation_rate:.3f}")
        else:
            self.config.mutation_rate = max(0.01, self.config.mutation_rate * 0.95)
        
        # Adapt diversity threshold
        current_diversity = self.evolution_history[-1]["diversity"]
        if current_diversity < 0.3:  # Low diversity
            self.config.diversity_threshold = max(0.05, self.config.diversity_threshold * 0.9)
            logger.debug(f"Decreased diversity threshold to {self.config.diversity_threshold:.3f}")
    
    async def run_evolution(self) -> Dict[str, Any]:
        """Run complete evolutionary process."""
        logger.info(f"Starting neural evolution with {self.config.generations} generations")
        
        start_time = time.time()
        
        # Initialize population
        await self.initialize_population()
        
        # Evolution loop
        for generation in range(self.config.generations):
            await self.evolve_generation()
            
            # Early stopping based on fitness convergence
            if len(self.evolution_history) >= 10:
                recent_improvements = [
                    self.evolution_history[i]["best_fitness"] - self.evolution_history[i-1]["best_fitness"]
                    for i in range(-5, 0)
                ]
                if all(imp < 0.001 for imp in recent_improvements):
                    logger.info(f"Early stopping at generation {generation + 1} due to fitness convergence")
                    break
        
        total_time = time.time() - start_time
        
        # Final results
        final_results = {
            "algorithm": "neural_evolution_v2",
            "configuration": asdict(self.config),
            "generations_completed": self.generation_count,
            "final_population_size": len(self.population),
            "best_individual": self.best_individuals[-1].to_dict() if self.best_individuals else None,
            "evolution_history": self.evolution_history,
            "final_diversity": self._calculate_diversity(),
            "total_evaluations": self.fitness_evaluator.evaluation_count,
            "total_runtime": total_time,
            "convergence_generation": self.generation_count,
            "pareto_front": [ind.to_dict() for ind in self.fitness_evaluator.get_pareto_front(self.population)],
            "top_10_individuals": [ind.to_dict() for ind in sorted(self.population, key=lambda x: x.fitness["composite"], reverse=True)[:10]]
        }
        
        logger.info(f"Neural evolution completed in {total_time:.2f}s")
        logger.info(f"Best fitness achieved: {final_results['best_individual']['fitness']['composite']:.3f}")
        
        return final_results


class CoevolutionaryEngine:
    """Coevolutionary system for protein-protein interactions."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.species_populations: Dict[str, List[Individual]] = {}
        self.interaction_matrix: Dict[Tuple[str, str], float] = {}
        self.coevolution_history: List[Dict[str, Any]] = []
        
    async def initialize_species(self, species_names: List[str]) -> Dict[str, List[Individual]]:
        """Initialize multiple species for coevolution."""
        logger.info(f"Initializing coevolution with species: {species_names}")
        
        for species in species_names:
            evolution_engine = NeuralEvolutionEngine(self.config)
            population = await evolution_engine.initialize_population()
            self.species_populations[species] = population
            
        return self.species_populations
    
    async def coevolve(self, generations: int = 50) -> Dict[str, Any]:
        """Run coevolutionary process."""
        logger.info(f"Starting coevolution for {generations} generations")
        
        for generation in range(generations):
            # Evaluate interactions between species
            await self._evaluate_interactions()
            
            # Evolve each species based on interactions
            for species_name in self.species_populations:
                await self._evolve_species(species_name)
            
            # Record coevolution statistics
            generation_stats = await self._record_generation_stats(generation)
            self.coevolution_history.append(generation_stats)
            
            logger.info(f"Coevolution generation {generation + 1} complete")
        
        return {
            "coevolution_history": self.coevolution_history,
            "final_species_populations": {
                species: [ind.to_dict() for ind in pop]
                for species, pop in self.species_populations.items()
            },
            "interaction_matrix": self.interaction_matrix
        }
    
    async def _evaluate_interactions(self):
        """Evaluate interactions between species."""
        species_list = list(self.species_populations.keys())
        
        for i in range(len(species_list)):
            for j in range(i + 1, len(species_list)):
                species1, species2 = species_list[i], species_list[j]
                
                # Sample individuals from each species
                sample1 = np.random.choice(self.species_populations[species1], min(10, len(self.species_populations[species1])), replace=False)
                sample2 = np.random.choice(self.species_populations[species2], min(10, len(self.species_populations[species2])), replace=False)
                
                interaction_scores = []
                for ind1 in sample1:
                    for ind2 in sample2:
                        score = await self._calculate_interaction_score(ind1, ind2)
                        interaction_scores.append(score)
                
                avg_interaction = np.mean(interaction_scores)
                self.interaction_matrix[(species1, species2)] = avg_interaction
                self.interaction_matrix[(species2, species1)] = avg_interaction
    
    async def _calculate_interaction_score(self, individual1: Individual, individual2: Individual) -> float:
        """Calculate interaction score between two individuals."""
        # Mock interaction calculation based on sequence compatibility
        seq1, seq2 = individual1.phenotype, individual2.phenotype
        
        # Complementarity score
        charge_compatibility = self._calculate_charge_compatibility(seq1, seq2)
        hydrophobic_compatibility = self._calculate_hydrophobic_compatibility(seq1, seq2)
        size_compatibility = self._calculate_size_compatibility(seq1, seq2)
        
        interaction_score = (charge_compatibility + hydrophobic_compatibility + size_compatibility) / 3
        
        return max(0.0, min(1.0, interaction_score))
    
    def _calculate_charge_compatibility(self, seq1: str, seq2: str) -> float:
        """Calculate charge compatibility between sequences."""
        positive_aa = "RHK"
        negative_aa = "DE"
        
        seq1_charge = sum(1 for aa in seq1 if aa in positive_aa) - sum(1 for aa in seq1 if aa in negative_aa)
        seq2_charge = sum(1 for aa in seq2 if aa in positive_aa) - sum(1 for aa in seq2 if aa in negative_aa)
        
        # Opposite charges attract
        compatibility = 1.0 - abs(seq1_charge + seq2_charge) / max(len(seq1), len(seq2))
        return max(0.0, compatibility)
    
    def _calculate_hydrophobic_compatibility(self, seq1: str, seq2: str) -> float:
        """Calculate hydrophobic compatibility."""
        hydrophobic_aa = "AILMFPWYV"
        
        seq1_hydrophobic = sum(1 for aa in seq1 if aa in hydrophobic_aa) / len(seq1)
        seq2_hydrophobic = sum(1 for aa in seq2 if aa in hydrophobic_aa) / len(seq2)
        
        # Similar hydrophobicity preferred for binding
        compatibility = 1.0 - abs(seq1_hydrophobic - seq2_hydrophobic)
        return max(0.0, compatibility)
    
    def _calculate_size_compatibility(self, seq1: str, seq2: str) -> float:
        """Calculate size compatibility."""
        size_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        return size_ratio
    
    async def _evolve_species(self, species_name: str):
        """Evolve a single species based on interactions."""
        population = self.species_populations[species_name]
        
        # Adjust fitness based on interactions
        for individual in population:
            interaction_bonus = 0.0
            interaction_count = 0
            
            for key, score in self.interaction_matrix.items():
                if species_name in key:
                    interaction_bonus += score
                    interaction_count += 1
            
            if interaction_count > 0:
                interaction_bonus /= interaction_count
                individual.fitness["composite"] += interaction_bonus * 0.2  # 20% contribution
        
        # Simple selection and mutation
        # Sort by fitness and keep top half
        population.sort(key=lambda x: x.fitness["composite"], reverse=True)
        survivors = population[:len(population) // 2]
        
        # Generate offspring to restore population size
        offspring = []
        while len(offspring) < len(population) - len(survivors):
            parent = np.random.choice(survivors)
            
            # Create mutated offspring
            mutated_genotype = self._mutate_genotype(parent.genotype)
            mutated_sequence = self._genotype_to_sequence(mutated_genotype)
            
            # Evaluate fitness
            fitness_evaluator = FitnessEvaluator()
            fitness = await fitness_evaluator.evaluate_fitness(mutated_sequence)
            
            offspring_individual = Individual(
                individual_id=str(uuid.uuid4()),
                genotype=mutated_genotype,
                phenotype=mutated_sequence,
                fitness=fitness,
                parent_ids=[parent.individual_id],
                generation=parent.generation + 1
            )
            
            offspring.append(offspring_individual)
        
        self.species_populations[species_name] = survivors + offspring
    
    def _mutate_genotype(self, genotype: List[List[float]]) -> List[List[float]]:
        """Simple genotype mutation."""
        mutated = []
        for gene in genotype:
            mutated_gene = gene.copy()
            if np.random.random() < 0.1:  # 10% mutation rate
                idx = np.random.randint(0, len(mutated_gene))
                mutated_gene[idx] = np.random.random()
                
                # Normalize
                total = sum(mutated_gene)
                if total > 0:
                    mutated_gene = [x / total for x in mutated_gene]
            
            mutated.append(mutated_gene)
        
        return mutated
    
    def _genotype_to_sequence(self, genotype: List[List[float]]) -> str:
        """Convert genotype to sequence."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        sequence = []
        
        for gene in genotype:
            max_idx = np.argmax(gene)
            if max_idx < len(amino_acids):
                sequence.append(amino_acids[max_idx])
            else:
                sequence.append(np.random.choice(list(amino_acids)))
        
        return ''.join(sequence)
    
    async def _record_generation_stats(self, generation: int) -> Dict[str, Any]:
        """Record statistics for current generation."""
        stats = {
            "generation": generation,
            "species_stats": {},
            "interaction_stats": {
                "average_interaction": np.mean(list(self.interaction_matrix.values())) if self.interaction_matrix else 0.0,
                "max_interaction": max(self.interaction_matrix.values()) if self.interaction_matrix else 0.0,
                "min_interaction": min(self.interaction_matrix.values()) if self.interaction_matrix else 0.0
            }
        }
        
        for species_name, population in self.species_populations.items():
            fitnesses = [ind.fitness["composite"] for ind in population]
            stats["species_stats"][species_name] = {
                "population_size": len(population),
                "best_fitness": max(fitnesses),
                "average_fitness": np.mean(fitnesses),
                "diversity": self._calculate_species_diversity(population)
            }
        
        return stats
    
    def _calculate_species_diversity(self, population: List[Individual]) -> float:
        """Calculate diversity within a species."""
        if len(population) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                sim = self._sequence_similarity(population[i].phenotype, population[j].phenotype)
                similarities.append(sim)
        
        return 1.0 - np.mean(similarities)
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity."""
        if not seq1 or not seq2:
            return 0.0
        
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return matches / max(len(seq1), len(seq2))


# Global neural evolution instance
neural_evolution_engine = None


async def run_neural_evolution_example():
    """Example of running neural evolution."""
    print("🧠 Neural Evolution V2 - Protein Design Demo")
    print("=" * 50)
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=50,
        generations=20,
        mutation_rate=0.15,
        crossover_rate=0.8,
        selection_method=SelectionMethod.TOURNAMENT,
        fitness_objectives=["stability", "novelty", "solubility"],
        neural_architecture={"hidden_sizes": [32, 16]},
        adaptive_parameters=True
    )
    
    # Run evolution
    engine = NeuralEvolutionEngine(config)
    results = await engine.run_evolution()
    
    print(f"\n✅ Evolution completed!")
    print(f"   Generations: {results['generations_completed']}")
    print(f"   Best fitness: {results['best_individual']['fitness']['composite']:.3f}")
    print(f"   Total runtime: {results['total_runtime']:.2f}s")
    
    # Show top sequences
    print(f"\n🏆 Top 3 evolved proteins:")
    for i, individual in enumerate(results['top_10_individuals'][:3]):
        print(f"   {i+1}. {individual['phenotype'][:50]}...")
        print(f"      Fitness: {individual['fitness']['composite']:.3f}")
    
    return results


async def run_coevolution_example():
    """Example of coevolutionary protein design."""
    print("\n🤝 Coevolution Example - Protein Interaction Design")
    print("=" * 50)
    
    config = EvolutionConfig(
        population_size=30,
        generations=15,
        fitness_objectives=["stability", "bindability"],
        coevolution_enabled=True
    )
    
    coevo_engine = CoevolutionaryEngine(config)
    
    # Initialize species (e.g., receptor and ligand)
    species_names = ["receptor", "ligand"]
    await coevo_engine.initialize_species(species_names)
    
    # Run coevolution
    coevo_results = await coevo_engine.coevolve(generations=15)
    
    print(f"✅ Coevolution completed!")
    print(f"   Species evolved: {len(species_names)}")
    print(f"   Average interaction score: {coevo_results['coevolution_history'][-1]['interaction_stats']['average_interaction']:.3f}")
    
    return coevo_results


if __name__ == "__main__":
    # Run examples
    evolution_results = asyncio.run(run_neural_evolution_example())
    coevolution_results = asyncio.run(run_coevolution_example())