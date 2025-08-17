"""
Distributed Swarm Intelligence Framework
Generation 4: Collaborative Protein Design

Advanced swarm intelligence with distributed agents, collective problem-solving,
and emergent behavior for collaborative protein design optimization.
"""

import asyncio
import json
import time
import uuid
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
from collections import defaultdict, deque
import math
import random
import copy
import heapq
from statistics import mean, stdev

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import scipy.spatial.distance as distance
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SwarmAlgorithm(Enum):
    """Types of swarm algorithms."""
    PARTICLE_SWARM_OPTIMIZATION = "pso"
    ANT_COLONY_OPTIMIZATION = "aco"
    ARTIFICIAL_BEE_COLONY = "abc"
    FIREFLY_ALGORITHM = "firefly"
    GREY_WOLF_OPTIMIZER = "gwo"
    WHALE_OPTIMIZATION = "woa"
    CUCKOO_SEARCH = "cuckoo"
    BAT_ALGORITHM = "bat"
    MULTI_SWARM_PSO = "multi_pso"
    HYBRID_SWARM = "hybrid"


class AgentRole(Enum):
    """Roles for swarm agents."""
    EXPLORER = "explorer"
    EXPLOITER = "exploiter"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    COMMUNICATOR = "communicator"
    EVALUATOR = "evaluator"
    MEMORY_KEEPER = "memory_keeper"
    DIVERSITY_MAINTAINER = "diversity_maintainer"


class CommunicationProtocol(Enum):
    """Communication protocols between agents."""
    BROADCAST = "broadcast"
    NEIGHBOR_ONLY = "neighbor_only"
    HIERARCHICAL = "hierarchical"
    DIRECT_MESSAGING = "direct_messaging"
    STIGMERGY = "stigmergy"  # Indirect communication through environment
    CONSENSUS = "consensus"
    GOSSIP = "gossip"


@dataclass
class SwarmAgent:
    """Individual agent in the swarm."""
    agent_id: str
    role: AgentRole
    position: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    fitness: float = -float('inf')
    personal_best: List[float] = field(default_factory=list)
    personal_best_fitness: float = -float('inf')
    memory: Dict[str, Any] = field(default_factory=dict)
    communication_radius: float = 1.0
    energy: float = 1.0
    age: int = 0
    experience: int = 0
    specialization: Optional[str] = None
    reputation: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())
        
        # Initialize personal best if not set
        if not self.personal_best and self.position:
            self.personal_best = self.position.copy()
            self.personal_best_fitness = self.fitness


@dataclass 
class SwarmCommunication:
    """Communication message between agents."""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id
        }


@dataclass
class SwarmEnvironment:
    """Environment for swarm interaction."""
    dimension: int
    bounds: List[Tuple[float, float]]
    pheromones: Dict[str, float] = field(default_factory=dict)
    landmarks: List[List[float]] = field(default_factory=list)
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    global_best: Optional[List[float]] = None
    global_best_fitness: float = -float('inf')
    temperature: float = 1.0  # For cooling schedules
    step: int = 0
    communication_network: Optional[Any] = None  # NetworkX graph if available


class SwarmOptimizer(ABC):
    """Abstract base class for swarm optimizers."""
    
    @abstractmethod
    async def optimize(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: List[Tuple[float, float]],
        max_iterations: int = 1000
    ) -> Dict[str, Any]:
        """Optimize using swarm algorithm."""
        pass
    
    @abstractmethod
    def update_agents(self, agents: List[SwarmAgent], environment: SwarmEnvironment):
        """Update agent positions and velocities."""
        pass


class ParticleSwarmOptimizer(SwarmOptimizer):
    """Particle Swarm Optimization implementation."""
    
    def __init__(
        self,
        swarm_size: int = 50,
        inertia_weight: float = 0.9,
        cognitive_coefficient: float = 2.0,
        social_coefficient: float = 2.0,
        adaptive_parameters: bool = True
    ):
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.adaptive_parameters = adaptive_parameters
        
        self.optimization_history: List[Dict[str, Any]] = []
        
    async def optimize(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: List[Tuple[float, float]],
        max_iterations: int = 1000
    ) -> Dict[str, Any]:
        """Optimize using PSO."""
        
        logger.info(f"Starting PSO optimization: {self.swarm_size} particles, {max_iterations} iterations")
        
        start_time = time.time()
        
        # Initialize environment
        environment = SwarmEnvironment(
            dimension=dimension,
            bounds=bounds
        )
        
        # Initialize swarm
        agents = self._initialize_swarm(dimension, bounds)
        
        # Evaluate initial positions
        await self._evaluate_swarm(agents, objective_function)
        self._update_global_best(agents, environment)
        
        iteration_history = []
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            # Update parameters if adaptive
            if self.adaptive_parameters:
                self._adapt_parameters(iteration, max_iterations, environment)
            
            # Update agent positions and velocities
            self.update_agents(agents, environment)
            
            # Evaluate new positions
            await self._evaluate_swarm(agents, objective_function)
            
            # Update personal and global bests
            self._update_personal_bests(agents)
            self._update_global_best(agents, environment)
            
            # Record iteration statistics
            iteration_time = time.time() - iteration_start
            stats = self._calculate_iteration_stats(agents, environment, iteration, iteration_time)
            iteration_history.append(stats)
            
            # Check convergence
            if self._check_convergence(iteration_history):
                logger.info(f"PSO converged at iteration {iteration}")
                break
                
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: Best fitness = {environment.global_best_fitness:.6f}")
        
        total_time = time.time() - start_time
        
        optimization_result = {
            "algorithm": "PSO",
            "best_solution": environment.global_best,
            "best_fitness": environment.global_best_fitness,
            "total_iterations": len(iteration_history),
            "convergence_iteration": len(iteration_history),
            "execution_time": total_time,
            "swarm_size": self.swarm_size,
            "final_parameters": {
                "inertia_weight": self.inertia_weight,
                "cognitive_coefficient": self.cognitive_coefficient,
                "social_coefficient": self.social_coefficient
            },
            "iteration_history": iteration_history,
            "swarm_diversity": self._calculate_swarm_diversity(agents),
            "convergence_rate": self._calculate_convergence_rate(iteration_history),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"PSO completed in {total_time:.2f}s, best fitness: {environment.global_best_fitness:.6f}")
        
        return optimization_result
    
    def _initialize_swarm(self, dimension: int, bounds: List[Tuple[float, float]]) -> List[SwarmAgent]:
        """Initialize swarm with random positions."""
        agents = []
        
        for i in range(self.swarm_size):
            # Random position within bounds
            position = []
            velocity = []
            
            for bound in bounds:
                pos = np.random.uniform(bound[0], bound[1])
                vel = np.random.uniform(-abs(bound[1] - bound[0]) * 0.1, abs(bound[1] - bound[0]) * 0.1)
                position.append(pos)
                velocity.append(vel)
            
            agent = SwarmAgent(
                agent_id=f"particle_{i}",
                role=AgentRole.EXPLORER,
                position=position,
                velocity=velocity
            )
            
            agents.append(agent)
        
        return agents
    
    async def _evaluate_swarm(self, agents: List[SwarmAgent], objective_function: Callable):
        """Evaluate fitness for all agents."""
        tasks = []
        
        for agent in agents:
            task = objective_function(agent.position)
            tasks.append((agent, task))
        
        # Evaluate in batches
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            if asyncio.iscoroutinefunction(objective_function):
                results = await asyncio.gather(*[task for _, task in batch])
            else:
                results = [await asyncio.get_event_loop().run_in_executor(
                    None, lambda: task) for _, task in batch]
            
            for (agent, _), fitness in zip(batch, results):
                agent.fitness = fitness
    
    def update_agents(self, agents: List[SwarmAgent], environment: SwarmEnvironment):
        """Update agent positions and velocities using PSO formula."""
        
        for agent in agents:
            # PSO velocity update
            r1, r2 = np.random.random(), np.random.random()
            
            for d in range(len(agent.position)):
                # Velocity update
                cognitive_component = (
                    self.cognitive_coefficient * r1 * 
                    (agent.personal_best[d] - agent.position[d])
                )
                
                social_component = 0
                if environment.global_best:
                    social_component = (
                        self.social_coefficient * r2 * 
                        (environment.global_best[d] - agent.position[d])
                    )
                
                agent.velocity[d] = (
                    self.inertia_weight * agent.velocity[d] +
                    cognitive_component + social_component
                )
                
                # Velocity clamping
                max_velocity = abs(environment.bounds[d][1] - environment.bounds[d][0]) * 0.1
                agent.velocity[d] = np.clip(agent.velocity[d], -max_velocity, max_velocity)
                
                # Position update
                agent.position[d] += agent.velocity[d]
                
                # Boundary handling
                if agent.position[d] < environment.bounds[d][0]:
                    agent.position[d] = environment.bounds[d][0]
                    agent.velocity[d] *= -0.5  # Reflect with damping
                elif agent.position[d] > environment.bounds[d][1]:
                    agent.position[d] = environment.bounds[d][1]
                    agent.velocity[d] *= -0.5
            
            agent.age += 1
    
    def _update_personal_bests(self, agents: List[SwarmAgent]):
        """Update personal best positions for all agents."""
        for agent in agents:
            if agent.fitness > agent.personal_best_fitness:
                agent.personal_best = agent.position.copy()
                agent.personal_best_fitness = agent.fitness
                agent.experience += 1
    
    def _update_global_best(self, agents: List[SwarmAgent], environment: SwarmEnvironment):
        """Update global best position."""
        best_agent = max(agents, key=lambda a: a.fitness)
        
        if best_agent.fitness > environment.global_best_fitness:
            environment.global_best = best_agent.position.copy()
            environment.global_best_fitness = best_agent.fitness
    
    def _adapt_parameters(self, iteration: int, max_iterations: int, environment: SwarmEnvironment):
        """Adapt PSO parameters during optimization."""
        # Linear decrease of inertia weight
        self.inertia_weight = 0.9 - (0.9 - 0.4) * iteration / max_iterations
        
        # Adaptive cognitive and social coefficients
        progress = iteration / max_iterations
        self.cognitive_coefficient = 2.5 - 1.5 * progress  # Decrease exploration
        self.social_coefficient = 0.5 + 2.0 * progress     # Increase exploitation
    
    def _calculate_iteration_stats(
        self, 
        agents: List[SwarmAgent], 
        environment: SwarmEnvironment, 
        iteration: int, 
        iteration_time: float
    ) -> Dict[str, Any]:
        """Calculate statistics for current iteration."""
        
        fitnesses = [agent.fitness for agent in agents]
        
        return {
            "iteration": iteration,
            "best_fitness": environment.global_best_fitness,
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "min_fitness": np.min(fitnesses),
            "swarm_diversity": self._calculate_swarm_diversity(agents),
            "parameters": {
                "inertia_weight": self.inertia_weight,
                "cognitive_coefficient": self.cognitive_coefficient,
                "social_coefficient": self.social_coefficient
            },
            "iteration_time": iteration_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_swarm_diversity(self, agents: List[SwarmAgent]) -> float:
        """Calculate diversity measure of swarm."""
        if len(agents) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                distance = np.linalg.norm(
                    np.array(agents[i].position) - np.array(agents[j].position)
                )
                total_distance += distance
                comparisons += 1
        
        return total_distance / max(1, comparisons)
    
    def _calculate_convergence_rate(self, iteration_history: List[Dict[str, Any]]) -> float:
        """Calculate convergence rate."""
        if len(iteration_history) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(iteration_history)):
            improvement = (
                iteration_history[i]["best_fitness"] - 
                iteration_history[i-1]["best_fitness"]
            )
            improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _check_convergence(self, iteration_history: List[Dict[str, Any]], patience: int = 50) -> bool:
        """Check if optimization has converged."""
        if len(iteration_history) < patience:
            return False
        
        recent_improvements = [
            iteration_history[i]["best_fitness"] - iteration_history[i-1]["best_fitness"]
            for i in range(-patience, 0)
        ]
        
        return max(recent_improvements) < 1e-8


class AntColonyOptimizer(SwarmOptimizer):
    """Ant Colony Optimization for continuous problems."""
    
    def __init__(
        self,
        colony_size: int = 50,
        pheromone_evaporation: float = 0.1,
        pheromone_intensity: float = 1.0,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0,   # Heuristic importance
        exploration_radius: float = 0.1
    ):
        self.colony_size = colony_size
        self.pheromone_evaporation = pheromone_evaporation
        self.pheromone_intensity = pheromone_intensity
        self.alpha = alpha
        self.beta = beta
        self.exploration_radius = exploration_radius
        
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def optimize(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: List[Tuple[float, float]],
        max_iterations: int = 1000
    ) -> Dict[str, Any]:
        """Optimize using ACO."""
        
        logger.info(f"Starting ACO optimization: {self.colony_size} ants, {max_iterations} iterations")
        
        start_time = time.time()
        
        # Initialize environment with pheromone grid
        environment = SwarmEnvironment(
            dimension=dimension,
            bounds=bounds
        )
        
        # Initialize ant colony
        agents = self._initialize_colony(dimension, bounds)
        
        iteration_history = []
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            # Move ants based on pheromones and heuristics
            await self._move_ants(agents, environment, objective_function)
            
            # Update pheromones
            self._update_pheromones(agents, environment)
            
            # Update global best
            self._update_global_best(agents, environment)
            
            # Record iteration statistics  
            iteration_time = time.time() - iteration_start
            stats = self._calculate_iteration_stats(agents, environment, iteration, iteration_time)
            iteration_history.append(stats)
            
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: Best fitness = {environment.global_best_fitness:.6f}")
        
        total_time = time.time() - start_time
        
        optimization_result = {
            "algorithm": "ACO",
            "best_solution": environment.global_best,
            "best_fitness": environment.global_best_fitness,
            "total_iterations": len(iteration_history),
            "execution_time": total_time,
            "colony_size": self.colony_size,
            "parameters": {
                "pheromone_evaporation": self.pheromone_evaporation,
                "alpha": self.alpha,
                "beta": self.beta
            },
            "iteration_history": iteration_history,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"ACO completed in {total_time:.2f}s, best fitness: {environment.global_best_fitness:.6f}")
        
        return optimization_result
    
    def _initialize_colony(self, dimension: int, bounds: List[Tuple[float, float]]) -> List[SwarmAgent]:
        """Initialize ant colony."""
        agents = []
        
        for i in range(self.colony_size):
            position = [
                np.random.uniform(bound[0], bound[1]) 
                for bound in bounds
            ]
            
            agent = SwarmAgent(
                agent_id=f"ant_{i}",
                role=AgentRole.EXPLORER,
                position=position,
                memory={"trail": [position.copy()]}
            )
            
            agents.append(agent)
        
        return agents
    
    async def _move_ants(
        self, 
        agents: List[SwarmAgent], 
        environment: SwarmEnvironment,
        objective_function: Callable
    ):
        """Move ants based on pheromone trails and heuristics."""
        
        for agent in agents:
            # Choose next position based on pheromone and heuristic information
            new_position = self._choose_next_position(agent, environment)
            agent.position = new_position
            
            # Add to trail memory
            agent.memory["trail"].append(new_position.copy())
            
            # Limit trail length
            if len(agent.memory["trail"]) > 20:
                agent.memory["trail"].pop(0)
        
        # Evaluate all new positions
        await self._evaluate_ants(agents, objective_function)
    
    def _choose_next_position(self, agent: SwarmAgent, environment: SwarmEnvironment) -> List[float]:
        """Choose next position for ant based on pheromone and heuristic info."""
        
        current_pos = np.array(agent.position)
        
        # Generate candidate positions within exploration radius
        candidates = []
        for _ in range(10):  # Generate 10 candidates
            candidate = current_pos + np.random.uniform(
                -self.exploration_radius, 
                self.exploration_radius, 
                size=len(current_pos)
            )
            
            # Ensure within bounds
            for i, bound in enumerate(environment.bounds):
                candidate[i] = np.clip(candidate[i], bound[0], bound[1])
            
            candidates.append(candidate)
        
        # Calculate probabilities based on pheromone and heuristic information
        probabilities = []
        for candidate in candidates:
            pheromone = self._get_pheromone_level(candidate, environment)
            heuristic = self._calculate_heuristic(current_pos, candidate)
            
            probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(probability)
        
        # Select position based on probabilities
        if sum(probabilities) > 0:
            probabilities = np.array(probabilities) / sum(probabilities)
            selected_idx = np.random.choice(len(candidates), p=probabilities)
            return candidates[selected_idx].tolist()
        else:
            return random.choice(candidates).tolist()
    
    def _get_pheromone_level(self, position: np.ndarray, environment: SwarmEnvironment) -> float:
        """Get pheromone level at position."""
        # Simplified pheromone lookup using grid approximation
        grid_key = tuple(np.round(position, 2))  # Round to grid
        return environment.pheromones.get(str(grid_key), 0.1)
    
    def _calculate_heuristic(self, current_pos: np.ndarray, candidate_pos: np.ndarray) -> float:
        """Calculate heuristic information for candidate position."""
        # Distance-based heuristic (prefer moderate distances)
        distance = np.linalg.norm(candidate_pos - current_pos)
        return 1.0 / (1.0 + distance)  # Inverse distance
    
    def _update_pheromones(self, agents: List[SwarmAgent], environment: SwarmEnvironment):
        """Update pheromone levels."""
        
        # Evaporate existing pheromones
        for key in environment.pheromones:
            environment.pheromones[key] *= (1 - self.pheromone_evaporation)
        
        # Add new pheromones from ants
        for agent in agents:
            for position in agent.memory.get("trail", []):
                grid_key = str(tuple(np.round(position, 2)))
                
                # Pheromone amount proportional to fitness
                pheromone_amount = max(0, agent.fitness) * self.pheromone_intensity
                
                if grid_key in environment.pheromones:
                    environment.pheromones[grid_key] += pheromone_amount
                else:
                    environment.pheromones[grid_key] = pheromone_amount
    
    async def _evaluate_ants(self, agents: List[SwarmAgent], objective_function: Callable):
        """Evaluate fitness for all ants."""
        tasks = []
        
        for agent in agents:
            task = objective_function(agent.position)
            tasks.append((agent, task))
        
        # Evaluate in batches
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            if asyncio.iscoroutinefunction(objective_function):
                results = await asyncio.gather(*[task for _, task in batch])
            else:
                results = [await asyncio.get_event_loop().run_in_executor(
                    None, lambda: task) for _, task in batch]
            
            for (agent, _), fitness in zip(batch, results):
                agent.fitness = fitness
    
    def update_agents(self, agents: List[SwarmAgent], environment: SwarmEnvironment):
        """Update method (used by base class interface)."""
        # ACO updates are handled in _move_ants method
        pass
    
    def _update_global_best(self, agents: List[SwarmAgent], environment: SwarmEnvironment):
        """Update global best solution."""
        best_agent = max(agents, key=lambda a: a.fitness)
        
        if best_agent.fitness > environment.global_best_fitness:
            environment.global_best = best_agent.position.copy()
            environment.global_best_fitness = best_agent.fitness
    
    def _calculate_iteration_stats(
        self, 
        agents: List[SwarmAgent], 
        environment: SwarmEnvironment, 
        iteration: int, 
        iteration_time: float
    ) -> Dict[str, Any]:
        """Calculate iteration statistics."""
        
        fitnesses = [agent.fitness for agent in agents]
        
        return {
            "iteration": iteration,
            "best_fitness": environment.global_best_fitness,
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "pheromone_trails": len(environment.pheromones),
            "total_pheromone": sum(environment.pheromones.values()),
            "iteration_time": iteration_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class CollaborativeSwarmSystem:
    """Multi-swarm collaborative optimization system."""
    
    def __init__(self):
        self.active_swarms: Dict[str, Dict[str, Any]] = {}
        self.communication_network: Optional[Any] = None
        self.global_memory: Dict[str, Any] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
        if NETWORKX_AVAILABLE:
            self.communication_network = nx.Graph()
    
    async def collaborative_optimization(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: List[Tuple[float, float]],
        swarm_algorithms: List[SwarmAlgorithm] = None,
        num_swarms: int = 3,
        max_iterations: int = 1000,
        communication_frequency: int = 50
    ) -> Dict[str, Any]:
        """Run collaborative optimization with multiple swarms."""
        
        if swarm_algorithms is None:
            swarm_algorithms = [
                SwarmAlgorithm.PARTICLE_SWARM_OPTIMIZATION,
                SwarmAlgorithm.ANT_COLONY_OPTIMIZATION,
                SwarmAlgorithm.ARTIFICIAL_BEE_COLONY
            ]
        
        logger.info(f"Starting collaborative optimization with {num_swarms} swarms")
        
        start_time = time.time()
        collaboration_id = str(uuid.uuid4())
        
        # Initialize swarms
        swarms = {}
        for i in range(num_swarms):
            algorithm = swarm_algorithms[i % len(swarm_algorithms)]
            
            if algorithm == SwarmAlgorithm.PARTICLE_SWARM_OPTIMIZATION:
                optimizer = ParticleSwarmOptimizer(swarm_size=30)
            elif algorithm == SwarmAlgorithm.ANT_COLONY_OPTIMIZATION:
                optimizer = AntColonyOptimizer(colony_size=30)
            else:
                # Default to PSO for other algorithms
                optimizer = ParticleSwarmOptimizer(swarm_size=30)
            
            swarm_id = f"swarm_{i}_{algorithm.value}"
            swarms[swarm_id] = {
                "optimizer": optimizer,
                "algorithm": algorithm,
                "best_solution": None,
                "best_fitness": -float('inf'),
                "communication_log": [],
                "knowledge_shared": 0,
                "knowledge_received": 0
            }
            
            self.active_swarms[swarm_id] = swarms[swarm_id]
        
        # Setup communication network
        if self.communication_network is not None:
            self._setup_communication_network(list(swarms.keys()))
        
        # Run collaborative optimization
        collaboration_tasks = []
        for swarm_id, swarm_data in swarms.items():
            task = self._run_swarm_with_collaboration(
                swarm_id,
                swarm_data["optimizer"],
                objective_function,
                dimension,
                bounds,
                max_iterations,
                communication_frequency
            )
            collaboration_tasks.append((swarm_id, task))
        
        # Wait for all swarms to complete
        results = {}
        for swarm_id, task in collaboration_tasks:
            result = await task
            results[swarm_id] = result
            swarms[swarm_id]["result"] = result
        
        # Find overall best solution
        overall_best_fitness = -float('inf')
        overall_best_solution = None
        overall_best_swarm = None
        
        for swarm_id, result in results.items():
            if result["best_fitness"] > overall_best_fitness:
                overall_best_fitness = result["best_fitness"]
                overall_best_solution = result["best_solution"]
                overall_best_swarm = swarm_id
        
        total_time = time.time() - start_time
        
        # Calculate collaboration statistics
        total_communications = sum(
            len(swarm["communication_log"]) for swarm in swarms.values()
        )
        
        knowledge_exchanges = sum(
            swarm["knowledge_shared"] + swarm["knowledge_received"] 
            for swarm in swarms.values()
        )
        
        collaboration_result = {
            "collaboration_id": collaboration_id,
            "num_swarms": num_swarms,
            "algorithms_used": [swarm["algorithm"].value for swarm in swarms.values()],
            "overall_best_solution": overall_best_solution,
            "overall_best_fitness": overall_best_fitness,
            "best_performing_swarm": overall_best_swarm,
            "individual_results": results,
            "collaboration_statistics": {
                "total_communications": total_communications,
                "knowledge_exchanges": knowledge_exchanges,
                "communication_efficiency": knowledge_exchanges / max(1, total_communications),
                "convergence_diversity": self._calculate_convergence_diversity(results),
                "collaboration_benefit": self._calculate_collaboration_benefit(results)
            },
            "execution_time": total_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.collaboration_history.append(collaboration_result)
        
        # Cleanup active swarms
        for swarm_id in swarms.keys():
            if swarm_id in self.active_swarms:
                del self.active_swarms[swarm_id]
        
        logger.info(f"Collaborative optimization completed in {total_time:.2f}s")
        logger.info(f"Best fitness: {overall_best_fitness:.6f} from {overall_best_swarm}")
        
        return collaboration_result
    
    def _setup_communication_network(self, swarm_ids: List[str]):
        """Setup communication network between swarms."""
        if self.communication_network is None:
            return
        
        # Add swarms as nodes
        for swarm_id in swarm_ids:
            self.communication_network.add_node(swarm_id)
        
        # Create communication topology (small-world network)
        for i, swarm_id1 in enumerate(swarm_ids):
            for j, swarm_id2 in enumerate(swarm_ids):
                if i != j:
                    # Connect with probability based on distance
                    if np.random.random() < 0.7:  # High connectivity
                        self.communication_network.add_edge(swarm_id1, swarm_id2)
    
    async def _run_swarm_with_collaboration(
        self,
        swarm_id: str,
        optimizer: SwarmOptimizer,
        objective_function: Callable,
        dimension: int,
        bounds: List[Tuple[float, float]],
        max_iterations: int,
        communication_frequency: int
    ) -> Dict[str, Any]:
        """Run individual swarm with collaboration capabilities."""
        
        # Create collaborative objective function that includes communication
        async def collaborative_objective(position):
            # Get base fitness
            if asyncio.iscoroutinefunction(objective_function):
                base_fitness = await objective_function(position)
            else:
                base_fitness = objective_function(position)
            
            # Add collaboration bonus based on shared knowledge
            collaboration_bonus = self._calculate_collaboration_bonus(swarm_id, position)
            
            return base_fitness + collaboration_bonus
        
        # Modify optimizer to include communication
        original_optimize = optimizer.optimize
        
        async def optimize_with_communication(*args, **kwargs):
            # Start optimization
            optimization_task = asyncio.create_task(original_optimize(*args, **kwargs))
            
            # Setup periodic communication
            communication_task = asyncio.create_task(
                self._periodic_communication(swarm_id, communication_frequency, max_iterations)
            )
            
            # Wait for optimization to complete
            result = await optimization_task
            communication_task.cancel()
            
            return result
        
        return await optimize_with_communication(
            collaborative_objective, dimension, bounds, max_iterations
        )
    
    async def _periodic_communication(
        self, 
        swarm_id: str, 
        frequency: int, 
        max_iterations: int
    ):
        """Handle periodic communication between swarms."""
        
        for iteration in range(0, max_iterations, frequency):
            await asyncio.sleep(0.1)  # Simulate communication delay
            
            # Share best solutions with neighbors
            await self._share_knowledge(swarm_id)
            
            # Receive knowledge from neighbors
            await self._receive_knowledge(swarm_id)
    
    async def _share_knowledge(self, swarm_id: str):
        """Share knowledge with neighboring swarms."""
        
        if swarm_id not in self.active_swarms:
            return
        
        swarm = self.active_swarms[swarm_id]
        
        # Create knowledge package
        knowledge = {
            "source_swarm": swarm_id,
            "best_solution": swarm.get("best_solution"),
            "best_fitness": swarm.get("best_fitness", -float('inf')),
            "algorithm": swarm["algorithm"].value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Determine neighbors to share with
        neighbors = self._get_communication_neighbors(swarm_id)
        
        for neighbor_id in neighbors:
            if neighbor_id in self.active_swarms:
                # Send knowledge
                message = SwarmCommunication(
                    sender_id=swarm_id,
                    receiver_id=neighbor_id,
                    message_type="knowledge_sharing",
                    content=knowledge
                )
                
                swarm["communication_log"].append(message.to_dict())
                swarm["knowledge_shared"] += 1
                
                # Store in global memory for receiving swarm
                memory_key = f"{neighbor_id}_received_knowledge"
                if memory_key not in self.global_memory:
                    self.global_memory[memory_key] = []
                
                self.global_memory[memory_key].append(knowledge)
    
    async def _receive_knowledge(self, swarm_id: str):
        """Receive and process knowledge from other swarms."""
        
        memory_key = f"{swarm_id}_received_knowledge"
        received_knowledge = self.global_memory.get(memory_key, [])
        
        if not received_knowledge:
            return
        
        swarm = self.active_swarms.get(swarm_id)
        if not swarm:
            return
        
        # Process received knowledge
        for knowledge in received_knowledge:
            # Update swarm's best solution if received solution is better
            if knowledge["best_fitness"] > swarm.get("best_fitness", -float('inf')):
                swarm["best_solution"] = knowledge["best_solution"]
                swarm["best_fitness"] = knowledge["best_fitness"]
            
            swarm["knowledge_received"] += 1
        
        # Clear received knowledge
        self.global_memory[memory_key] = []
    
    def _get_communication_neighbors(self, swarm_id: str) -> List[str]:
        """Get communication neighbors for a swarm."""
        
        if self.communication_network is None:
            # Return all other active swarms
            return [sid for sid in self.active_swarms.keys() if sid != swarm_id]
        
        if swarm_id in self.communication_network:
            return list(self.communication_network.neighbors(swarm_id))
        
        return []
    
    def _calculate_collaboration_bonus(self, swarm_id: str, position: List[float]) -> float:
        """Calculate collaboration bonus based on shared knowledge."""
        
        # Simple collaboration bonus based on proximity to shared solutions
        bonus = 0.0
        memory_key = f"{swarm_id}_collaboration_memory"
        shared_solutions = self.global_memory.get(memory_key, [])
        
        for solution in shared_solutions[-5:]:  # Consider last 5 shared solutions
            if solution and len(solution) == len(position):
                distance = np.linalg.norm(np.array(position) - np.array(solution))
                bonus += 0.1 * np.exp(-distance)  # Exponential decay with distance
        
        return bonus
    
    def _calculate_convergence_diversity(self, results: Dict[str, Any]) -> float:
        """Calculate diversity of convergence among swarms."""
        
        best_fitnesses = [
            result["best_fitness"] for result in results.values()
        ]
        
        if len(best_fitnesses) < 2:
            return 0.0
        
        return np.std(best_fitnesses) / max(1e-8, np.mean(best_fitnesses))
    
    def _calculate_collaboration_benefit(self, results: Dict[str, Any]) -> float:
        """Calculate benefit gained from collaboration."""
        
        # Mock calculation - in practice would compare against isolated runs
        best_fitness = max(result["best_fitness"] for result in results.values())
        mean_fitness = np.mean([result["best_fitness"] for result in results.values()])
        
        # Benefit measure: how much the best exceeds the mean
        return (best_fitness - mean_fitness) / max(1e-8, abs(mean_fitness))
    
    def get_collaboration_summary(self) -> Dict[str, Any]:
        """Get summary of collaboration history."""
        
        if not self.collaboration_history:
            return {"status": "no_collaborations_performed"}
        
        total_swarms = sum(
            collab["num_swarms"] for collab in self.collaboration_history
        )
        
        total_communications = sum(
            collab["collaboration_statistics"]["total_communications"]
            for collab in self.collaboration_history
        )
        
        return {
            "total_collaborations": len(self.collaboration_history),
            "total_swarms_deployed": total_swarms,
            "total_communications": total_communications,
            "average_collaboration_benefit": np.mean([
                collab["collaboration_statistics"]["collaboration_benefit"]
                for collab in self.collaboration_history
            ]),
            "best_collaboration_fitness": max([
                collab["overall_best_fitness"]
                for collab in self.collaboration_history
            ]),
            "collaboration_efficiency": np.mean([
                collab["collaboration_statistics"]["communication_efficiency"]
                for collab in self.collaboration_history
            ]),
            "algorithms_used": list(set([
                alg for collab in self.collaboration_history
                for alg in collab["algorithms_used"]
            ])),
            "active_swarms": len(self.active_swarms),
            "global_memory_size": len(self.global_memory),
            "last_collaboration": self.collaboration_history[-1]["timestamp"] if self.collaboration_history else None
        }


# Global collaborative swarm system
collaborative_swarm = CollaborativeSwarmSystem()


# Example usage and testing
async def example_swarm_intelligence():
    """Example collaborative swarm intelligence optimization."""
    
    print("🐝 Distributed Swarm Intelligence Demo")
    print("=" * 50)
    
    # Define test objective function (protein design fitness)
    def protein_fitness(position):
        """Mock protein design fitness function."""
        # Multi-modal function with global optimum
        x = np.array(position)
        
        # Primary objective: stability (Gaussian around origin)
        stability = np.exp(-0.5 * np.sum(x**2))
        
        # Secondary objective: binding affinity (multiple peaks)
        binding = 0.8 * np.exp(-0.5 * np.sum((x - 2)**2)) + 0.6 * np.exp(-0.5 * np.sum((x + 2)**2))
        
        # Tertiary objective: solubility (sinusoidal component)
        solubility = 0.3 * (1 + np.sin(np.sum(x)))
        
        return stability + binding + solubility
    
    # Run collaborative optimization
    result = await collaborative_swarm.collaborative_optimization(
        objective_function=protein_fitness,
        dimension=5,
        bounds=[(-5, 5)] * 5,
        swarm_algorithms=[
            SwarmAlgorithm.PARTICLE_SWARM_OPTIMIZATION,
            SwarmAlgorithm.ANT_COLONY_OPTIMIZATION
        ],
        num_swarms=3,
        max_iterations=200,
        communication_frequency=25
    )
    
    print(f"\n✅ Collaborative Optimization Results:")
    print(f"   Best Fitness: {result['overall_best_fitness']:.6f}")
    print(f"   Best Solution: {[f'{x:.3f}' for x in result['overall_best_solution']]}")
    print(f"   Best Algorithm: {result['best_performing_swarm']}")
    print(f"   Execution Time: {result['execution_time']:.2f}s")
    
    print(f"\n🤝 Collaboration Statistics:")
    stats = result["collaboration_statistics"]
    print(f"   Total Communications: {stats['total_communications']}")
    print(f"   Knowledge Exchanges: {stats['knowledge_exchanges']}")
    print(f"   Communication Efficiency: {stats['communication_efficiency']:.3f}")
    print(f"   Collaboration Benefit: {stats['collaboration_benefit']:.3f}")
    
    # Show individual swarm performance
    print(f"\n🐝 Individual Swarm Performance:")
    for swarm_id, swarm_result in result["individual_results"].items():
        print(f"   {swarm_id}: {swarm_result['best_fitness']:.6f}")
    
    # System summary
    summary = collaborative_swarm.get_collaboration_summary()
    print(f"\n📊 System Summary:")
    print(f"   Total Collaborations: {summary['total_collaborations']}")
    print(f"   Swarms Deployed: {summary['total_swarms_deployed']}")
    print(f"   Algorithms Used: {', '.join(summary['algorithms_used'])}")
    print(f"   Best Overall Fitness: {summary['best_collaboration_fitness']:.6f}")
    
    return result


if __name__ == "__main__":
    asyncio.run(example_swarm_intelligence())