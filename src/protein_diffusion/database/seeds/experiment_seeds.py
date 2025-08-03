"""
Seed data for experiments and results.

This module provides sample experiments and results for development 
and testing of the protein diffusion design lab.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

from ..models import ExperimentModel, ResultModel, ExperimentStatus
from ..repositories import ExperimentRepository, ResultRepository, ProteinRepository

logger = logging.getLogger(__name__)


# Sample experiment configurations
SAMPLE_EXPERIMENTS = [
    {
        "name": "Alpha helix design validation",
        "description": "Validation experiment for designed alpha-helical proteins",
        "parameters": {
            "motif": "HELIX",
            "num_samples": 50,
            "max_length": 100,
            "temperature": 0.8,
            "guidance_scale": 1.2,
            "target_secondary_structure": "helix",
            "min_confidence": 0.7,
        },
        "status": ExperimentStatus.COMPLETED,
    },
    {
        "name": "Beta sheet scaffold generation",
        "description": "Generate diverse beta sheet scaffolds for binding site design",
        "parameters": {
            "motif": "SHEET_LOOP_SHEET",
            "num_samples": 100,
            "max_length": 150,
            "temperature": 0.9,
            "guidance_scale": 1.0,
            "target_secondary_structure": "sheet",
            "diversity_threshold": 0.7,
        },
        "status": ExperimentStatus.COMPLETED,
    },
    {
        "name": "Mixed fold exploration",
        "description": "Explore diverse protein folds with mixed secondary structure",
        "parameters": {
            "motif": "HELIX_SHEET_HELIX",
            "num_samples": 200,
            "max_length": 200,
            "temperature": 1.0,
            "guidance_scale": 0.8,
            "target_secondary_structure": "mixed",
            "novelty_weight": 0.3,
        },
        "status": ExperimentStatus.RUNNING,
    },
    {
        "name": "Small molecule binding pocket design",
        "description": "Design proteins with specific binding pockets for small molecules",
        "parameters": {
            "motif": "HELIX_LOOP_HELIX",
            "num_samples": 75,
            "max_length": 120,
            "temperature": 0.7,
            "guidance_scale": 1.5,
            "target_function": "binding",
            "binding_target": "small_molecule",
            "pocket_volume": 500,  # Ångström³
        },
        "status": ExperimentStatus.COMPLETED,
    },
    {
        "name": "Membrane protein design",
        "description": "Design membrane-spanning proteins with transmembrane domains",
        "parameters": {
            "motif": "TRANSMEMBRANE",
            "num_samples": 30,
            "max_length": 300,
            "temperature": 0.6,
            "guidance_scale": 1.3,
            "target_environment": "membrane",
            "hydrophobicity_bias": 0.8,
        },
        "status": ExperimentStatus.PENDING,
    },
]


def seed_experiments(connection=None) -> None:
    """Seed the database with sample experiments."""
    from ..connection import get_connection
    
    if connection is None:
        connection = get_connection()
    
    experiment_repo = ExperimentRepository(connection)
    
    logger.info("Seeding sample experiments...")
    
    created_count = 0
    for exp_data in SAMPLE_EXPERIMENTS:
        # Check if experiment already exists
        existing = experiment_repo.search_by_name(exp_data["name"])
        if existing:
            logger.debug(f"Experiment already exists: {exp_data['name']}")
            continue
        
        # Create experiment
        experiment = ExperimentModel(
            name=exp_data["name"],
            description=exp_data["description"],
            parameters=exp_data["parameters"],
            status=exp_data["status"],
        )
        
        experiment_id = experiment_repo.create(experiment)
        created_count += 1
        
        logger.debug(f"Created experiment: {exp_data['name']} (ID: {experiment_id})")
        
        # Create sample results for completed experiments
        if exp_data["status"] == ExperimentStatus.COMPLETED:
            _create_sample_results(connection, experiment_id, exp_data["parameters"])
    
    logger.info(f"Created {created_count} new experiments")


def _create_sample_results(
    connection, 
    experiment_id: int, 
    exp_parameters: Dict[str, Any]
) -> None:
    """Create sample results for an experiment."""
    protein_repo = ProteinRepository(connection)
    result_repo = ResultRepository(connection)
    
    # Get some proteins from the database
    proteins = protein_repo.get_all(limit=20)
    if not proteins:
        logger.warning("No proteins available to create results")
        return
    
    num_results = min(len(proteins), exp_parameters.get("num_samples", 10))
    selected_proteins = random.sample(proteins, num_results)
    
    results = []
    for i, protein in enumerate(selected_proteins):
        # Generate realistic but fake metrics
        binding_affinity = _generate_binding_affinity(protein, exp_parameters)
        structure_quality = random.uniform(0.4, 0.95)
        diversity_score = random.uniform(0.3, 0.9)
        novelty_score = random.uniform(0.2, 0.8)
        
        # Calculate composite score
        weights = {
            "binding": exp_parameters.get("binding_weight", 0.4),
            "structure": exp_parameters.get("structure_weight", 0.3),
            "diversity": exp_parameters.get("diversity_weight", 0.2),
            "novelty": exp_parameters.get("novelty_weight", 0.1),
        }
        
        # Normalize binding affinity to 0-1 scale
        normalized_binding = max(0.0, (binding_affinity + 20.0) / 20.0) if binding_affinity else 0.5
        
        composite_score = (
            weights["binding"] * normalized_binding +
            weights["structure"] * structure_quality +
            weights["diversity"] * diversity_score +
            weights["novelty"] * novelty_score
        )
        
        # Create result
        result = ResultModel(
            experiment_id=experiment_id,
            protein_id=protein.id,
            binding_affinity=binding_affinity,
            composite_score=composite_score,
            ranking=i + 1,  # Will be updated later
            metrics={
                "structure_quality": structure_quality,
                "diversity_score": diversity_score,
                "novelty_score": novelty_score,
                "normalized_binding": normalized_binding,
                "hydrophobicity": random.uniform(-2.0, 2.0),
                "net_charge": random.randint(-5, 5),
                "molecular_weight": len(protein.sequence) * random.uniform(110, 140),
            }
        )
        
        results.append(result)
    
    # Sort by composite score and update rankings
    results.sort(key=lambda r: r.composite_score, reverse=True)
    for rank, result in enumerate(results, 1):
        result.ranking = rank
    
    # Create results in database
    result_repo.batch_create_results(results)
    
    logger.debug(f"Created {len(results)} results for experiment {experiment_id}")


def _generate_binding_affinity(protein, exp_parameters: Dict[str, Any]) -> float:
    """Generate realistic binding affinity based on protein properties."""
    # Base affinity
    base_affinity = random.uniform(-15.0, -2.0)
    
    # Adjust based on experiment parameters
    target_function = exp_parameters.get("target_function")
    if target_function == "binding":
        # Better binding expected
        base_affinity += random.uniform(-5.0, 0.0)
    
    # Adjust based on protein properties
    sequence = protein.sequence
    
    # Hydrophobic content affects binding
    hydrophobic_aa = "AILMFPWV"
    hydrophobic_content = sum(1 for aa in sequence if aa in hydrophobic_aa) / len(sequence)
    base_affinity += -3.0 * hydrophobic_content
    
    # Aromatic content affects binding
    aromatic_aa = "FWY"
    aromatic_content = sum(1 for aa in sequence if aa in aromatic_aa) / len(sequence)
    base_affinity += -2.0 * aromatic_content
    
    # Length affects binding (optimal range)
    length_factor = 1.0
    if 50 <= len(sequence) <= 150:
        length_factor = 1.2
    elif len(sequence) < 30 or len(sequence) > 300:
        length_factor = 0.7
    
    base_affinity *= length_factor
    
    # Add noise
    noise = random.gauss(0, 1.0)
    final_affinity = base_affinity + noise
    
    # Clamp to reasonable range
    return max(-25.0, min(0.0, final_affinity))


def create_benchmark_experiment(connection=None) -> int:
    """Create a benchmark experiment for testing purposes."""
    from ..connection import get_connection
    
    if connection is None:
        connection = get_connection()
    
    experiment_repo = ExperimentRepository(connection)
    
    # Create benchmark experiment
    experiment = ExperimentModel(
        name="Benchmark experiment",
        description="Benchmark experiment for performance testing and validation",
        parameters={
            "motif": "HELIX_SHEET_HELIX",
            "num_samples": 1000,
            "max_length": 200,
            "temperature": 0.8,
            "guidance_scale": 1.0,
            "benchmark": True,
            "created_for": "testing",
        },
        status=ExperimentStatus.PENDING,
    )
    
    experiment_id = experiment_repo.create(experiment)
    logger.info(f"Created benchmark experiment (ID: {experiment_id})")
    
    return experiment_id


def create_experiment_with_results(
    name: str,
    num_proteins: int = 50,
    connection=None
) -> int:
    """
    Create a complete experiment with proteins and results.
    
    Args:
        name: Experiment name
        num_proteins: Number of proteins to include
        connection: Database connection
        
    Returns:
        Experiment ID
    """
    from ..connection import get_connection
    from .protein_seeds import create_test_proteins
    
    if connection is None:
        connection = get_connection()
    
    experiment_repo = ExperimentRepository(connection)
    protein_repo = ProteinRepository(connection)
    result_repo = ResultRepository(connection)
    
    # Create experiment
    experiment = ExperimentModel(
        name=name,
        description=f"Complete experiment with {num_proteins} generated proteins",
        parameters={
            "motif": "MIXED",
            "num_samples": num_proteins,
            "max_length": 150,
            "temperature": 0.8,
            "guidance_scale": 1.0,
            "complete_experiment": True,
        },
        status=ExperimentStatus.COMPLETED,
    )
    
    experiment_id = experiment_repo.create(experiment)
    
    # Generate proteins
    test_proteins = create_test_proteins(num_proteins)
    
    # Create proteins and results
    results = []
    for i, protein in enumerate(test_proteins):
        # Create protein
        existing_protein, created = protein_repo.create_if_not_exists(protein)
        
        # Generate result
        binding_affinity = _generate_binding_affinity(existing_protein, experiment.parameters)
        composite_score = random.uniform(0.2, 0.9)
        
        result = ResultModel(
            experiment_id=experiment_id,
            protein_id=existing_protein.id,
            binding_affinity=binding_affinity,
            composite_score=composite_score,
            metrics={
                "structure_quality": random.uniform(0.5, 0.95),
                "diversity_score": random.uniform(0.3, 0.8),
                "novelty_score": random.uniform(0.4, 0.9),
            }
        )
        
        results.append(result)
    
    # Sort and rank results
    results.sort(key=lambda r: r.composite_score, reverse=True)
    for rank, result in enumerate(results, 1):
        result.ranking = rank
    
    # Save results
    result_repo.batch_create_results(results)
    
    logger.info(f"Created complete experiment '{name}' with {len(results)} results")
    
    return experiment_id


if __name__ == "__main__":
    # Run seeding if executed directly
    logging.basicConfig(level=logging.INFO)
    seed_experiments()
    create_benchmark_experiment()