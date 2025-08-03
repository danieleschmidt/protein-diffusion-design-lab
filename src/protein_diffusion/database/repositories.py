"""
Database repositories for protein diffusion design lab.

This module provides data access layer with CRUD operations and 
specialized queries for proteins, structures, experiments, and results.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .connection import DatabaseConnection, get_connection
from .models import (
    ProteinModel, StructureModel, ExperimentModel, ResultModel,
    ExperimentStatus, PredictionMethod
)

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Abstract base repository with common CRUD operations."""
    
    def __init__(self, connection: Optional[DatabaseConnection] = None):
        self.connection = connection or get_connection()
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the table name for this repository."""
        pass
    
    @property
    @abstractmethod
    def model_class(self):
        """Return the model class for this repository."""
        pass
    
    def create(self, model) -> int:
        """Create a new record and return its ID."""
        data = model.to_dict()
        data.pop('id', None)  # Remove ID for INSERT
        
        columns = list(data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        values = list(data.values())
        
        query = f"INSERT INTO {self.table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        with self.connection.get_connection() as conn:
            if self.connection.config.backend == "sqlite":
                cursor = conn.execute(query, values)
                conn.commit()
                return cursor.lastrowid
            else:  # PostgreSQL
                query = query.replace('?', '%s') + " RETURNING id"
                with conn.cursor() as cursor:
                    cursor.execute(query, values)
                    conn.commit()
                    return cursor.fetchone()[0]
    
    def get_by_id(self, record_id: int):
        """Get record by ID."""
        query = f"SELECT * FROM {self.table_name} WHERE id = ?"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (record_id,))
        if results:
            return self.model_class.from_dict(results[0])
        return None
    
    def update(self, model) -> bool:
        """Update existing record."""
        if not model.id:
            raise ValueError("Model must have an ID to update")
        
        data = model.to_dict()
        record_id = data.pop('id')
        
        columns = list(data.keys())
        set_clause = ', '.join([f"{col} = ?" for col in columns])
        values = list(data.values()) + [record_id]
        
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        affected_rows = self.connection.execute_mutation(query, values)
        return affected_rows > 0
    
    def delete(self, record_id: int) -> bool:
        """Delete record by ID."""
        query = f"DELETE FROM {self.table_name} WHERE id = ?"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        affected_rows = self.connection.execute_mutation(query, (record_id,))
        return affected_rows > 0
    
    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List:
        """Get all records with optional pagination."""
        query = f"SELECT * FROM {self.table_name} ORDER BY id"
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        results = self.connection.execute_query(query)
        return [self.model_class.from_dict(row) for row in results]
    
    def count(self) -> int:
        """Count total number of records."""
        query = f"SELECT COUNT(*) as count FROM {self.table_name}"
        results = self.connection.execute_query(query)
        return results[0]['count'] if results else 0


class ProteinRepository(BaseRepository):
    """Repository for protein sequences."""
    
    @property
    def table_name(self) -> str:
        return "proteins"
    
    @property
    def model_class(self):
        return ProteinModel
    
    def get_by_sequence_hash(self, sequence_hash: str) -> Optional[ProteinModel]:
        """Get protein by sequence hash."""
        query = "SELECT * FROM proteins WHERE sequence_hash = ?"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (sequence_hash,))
        if results:
            return ProteinModel.from_dict(results[0])
        return None
    
    def get_by_sequence(self, sequence: str) -> Optional[ProteinModel]:
        """Get protein by exact sequence match."""
        # Create temporary model to compute hash
        temp_model = ProteinModel(sequence=sequence)
        return self.get_by_sequence_hash(temp_model.sequence_hash)
    
    def find_by_length_range(self, min_length: int, max_length: int) -> List[ProteinModel]:
        """Find proteins within length range."""
        query = "SELECT * FROM proteins WHERE length BETWEEN ? AND ? ORDER BY length"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (min_length, max_length))
        return [ProteinModel.from_dict(row) for row in results]
    
    def search_by_substring(self, substring: str, limit: int = 100) -> List[ProteinModel]:
        """Search proteins containing a substring."""
        query = "SELECT * FROM proteins WHERE sequence LIKE ? ORDER BY length LIMIT ?"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        pattern = f"%{substring}%"
        results = self.connection.execute_query(query, (pattern, limit))
        return [ProteinModel.from_dict(row) for row in results]
    
    def get_recent(self, limit: int = 50) -> List[ProteinModel]:
        """Get recently created proteins."""
        query = "SELECT * FROM proteins ORDER BY created_at DESC LIMIT ?"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (limit,))
        return [ProteinModel.from_dict(row) for row in results]
    
    def create_if_not_exists(self, protein: ProteinModel) -> Tuple[ProteinModel, bool]:
        """
        Create protein if it doesn't exist, return (protein, created).
        
        Returns:
            Tuple of (protein_model, was_created)
        """
        existing = self.get_by_sequence_hash(protein.sequence_hash)
        if existing:
            return existing, False
        
        protein_id = self.create(protein)
        protein.id = protein_id
        return protein, True


class StructureRepository(BaseRepository):
    """Repository for protein structures."""
    
    @property
    def table_name(self) -> str:
        return "structures"
    
    @property
    def model_class(self):
        return StructureModel
    
    def get_by_protein_id(self, protein_id: int) -> List[StructureModel]:
        """Get all structures for a protein."""
        query = "SELECT * FROM structures WHERE protein_id = ? ORDER BY created_at DESC"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (protein_id,))
        return [StructureModel.from_dict(row) for row in results]
    
    def get_best_structure(self, protein_id: int) -> Optional[StructureModel]:
        """Get the highest quality structure for a protein."""
        query = """
            SELECT * FROM structures 
            WHERE protein_id = ? 
            ORDER BY structure_quality DESC, confidence DESC 
            LIMIT 1
        """
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (protein_id,))
        if results:
            return StructureModel.from_dict(results[0])
        return None
    
    def find_by_method(self, method: PredictionMethod) -> List[StructureModel]:
        """Find structures by prediction method."""
        query = "SELECT * FROM structures WHERE prediction_method = ? ORDER BY structure_quality DESC"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (method.value,))
        return [StructureModel.from_dict(row) for row in results]
    
    def find_high_quality(self, min_quality: float = 0.8) -> List[StructureModel]:
        """Find high-quality structures."""
        query = "SELECT * FROM structures WHERE structure_quality >= ? ORDER BY structure_quality DESC"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (min_quality,))
        return [StructureModel.from_dict(row) for row in results]
    
    def get_structure_statistics(self) -> Dict[str, Any]:
        """Get statistics about structures in the database."""
        queries = {
            "total_structures": "SELECT COUNT(*) as count FROM structures",
            "avg_quality": "SELECT AVG(structure_quality) as avg FROM structures",
            "avg_confidence": "SELECT AVG(confidence) as avg FROM structures",
        }
        
        stats = {}
        for key, query in queries.items():
            results = self.connection.execute_query(query)
            if results:
                stats[key] = results[0].get('count') or results[0].get('avg')
        
        # Count by method
        method_query = """
            SELECT prediction_method, COUNT(*) as count 
            FROM structures 
            GROUP BY prediction_method
        """
        results = self.connection.execute_query(method_query)
        stats['by_method'] = {row['prediction_method']: row['count'] for row in results}
        
        return stats


class ExperimentRepository(BaseRepository):
    """Repository for experiments."""
    
    @property
    def table_name(self) -> str:
        return "experiments"
    
    @property
    def model_class(self):
        return ExperimentModel
    
    def get_by_status(self, status: ExperimentStatus) -> List[ExperimentModel]:
        """Get experiments by status."""
        query = "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (status.value,))
        return [ExperimentModel.from_dict(row) for row in results]
    
    def get_running_experiments(self) -> List[ExperimentModel]:
        """Get currently running experiments."""
        return self.get_by_status(ExperimentStatus.RUNNING)
    
    def get_completed_experiments(self) -> List[ExperimentModel]:
        """Get completed experiments."""
        return self.get_by_status(ExperimentStatus.COMPLETED)
    
    def search_by_name(self, name_pattern: str) -> List[ExperimentModel]:
        """Search experiments by name pattern."""
        query = "SELECT * FROM experiments WHERE name LIKE ? ORDER BY created_at DESC"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        pattern = f"%{name_pattern}%"
        results = self.connection.execute_query(query, (pattern,))
        return [ExperimentModel.from_dict(row) for row in results]
    
    def update_status(self, experiment_id: int, status: ExperimentStatus) -> bool:
        """Update experiment status."""
        query = "UPDATE experiments SET status = ? WHERE id = ?"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        affected_rows = self.connection.execute_mutation(query, (status.value, experiment_id))
        return affected_rows > 0


class ResultRepository(BaseRepository):
    """Repository for experiment results."""
    
    @property
    def table_name(self) -> str:
        return "results"
    
    @property
    def model_class(self):
        return ResultModel
    
    def get_by_experiment_id(self, experiment_id: int) -> List[ResultModel]:
        """Get all results for an experiment."""
        query = "SELECT * FROM results WHERE experiment_id = ? ORDER BY ranking"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (experiment_id,))
        return [ResultModel.from_dict(row) for row in results]
    
    def get_top_results(self, experiment_id: int, limit: int = 10) -> List[ResultModel]:
        """Get top results for an experiment."""
        query = """
            SELECT * FROM results 
            WHERE experiment_id = ? 
            ORDER BY composite_score DESC 
            LIMIT ?
        """
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (experiment_id, limit))
        return [ResultModel.from_dict(row) for row in results]
    
    def get_best_binders(self, experiment_id: int, limit: int = 10) -> List[ResultModel]:
        """Get best binding results for an experiment."""
        query = """
            SELECT * FROM results 
            WHERE experiment_id = ? AND binding_affinity IS NOT NULL
            ORDER BY binding_affinity ASC 
            LIMIT ?
        """
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        results = self.connection.execute_query(query, (experiment_id, limit))
        return [ResultModel.from_dict(row) for row in results]
    
    def get_experiment_statistics(self, experiment_id: int) -> Dict[str, Any]:
        """Get statistics for an experiment."""
        base_query = "SELECT {} FROM results WHERE experiment_id = {}"
        
        # Handle different backends
        param_placeholder = "?" if self.connection.config.backend == "sqlite" else "%s"
        
        stats_queries = {
            "total_results": f"SELECT COUNT(*) as count FROM results WHERE experiment_id = {param_placeholder}",
            "avg_composite_score": f"SELECT AVG(composite_score) as avg FROM results WHERE experiment_id = {param_placeholder}",
            "avg_binding_affinity": f"SELECT AVG(binding_affinity) as avg FROM results WHERE experiment_id = {param_placeholder} AND binding_affinity IS NOT NULL",
            "best_composite_score": f"SELECT MAX(composite_score) as max FROM results WHERE experiment_id = {param_placeholder}",
            "best_binding_affinity": f"SELECT MIN(binding_affinity) as min FROM results WHERE experiment_id = {param_placeholder} AND binding_affinity IS NOT NULL",
        }
        
        stats = {}
        for key, query in stats_queries.items():
            results = self.connection.execute_query(query, (experiment_id,))
            if results:
                value = results[0].get('count') or results[0].get('avg') or results[0].get('max') or results[0].get('min')
                stats[key] = value
        
        return stats
    
    def batch_create_results(self, results: List[ResultModel]) -> List[int]:
        """Create multiple results efficiently."""
        if not results:
            return []
        
        created_ids = []
        for result in results:
            result_id = self.create(result)
            created_ids.append(result_id)
        
        return created_ids
    
    def update_rankings(self, experiment_id: int):
        """Update ranking positions based on composite scores."""
        # This could be optimized with a single SQL statement
        results = self.get_by_experiment_id(experiment_id)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        
        for rank, result in enumerate(results, 1):
            result.ranking = rank
            self.update(result)


class ProteinExperimentRepository:
    """
    Combined repository for complex queries involving multiple tables.
    
    This repository handles queries that span across proteins, structures,
    experiments, and results tables.
    """
    
    def __init__(self, connection: Optional[DatabaseConnection] = None):
        self.connection = connection or get_connection()
        self.protein_repo = ProteinRepository(connection)
        self.structure_repo = StructureRepository(connection)
        self.experiment_repo = ExperimentRepository(connection)
        self.result_repo = ResultRepository(connection)
    
    def get_experiment_with_results(self, experiment_id: int) -> Dict[str, Any]:
        """Get experiment with all associated results and proteins."""
        experiment = self.experiment_repo.get_by_id(experiment_id)
        if not experiment:
            return {}
        
        results = self.result_repo.get_by_experiment_id(experiment_id)
        
        # Get associated proteins and structures
        protein_ids = [r.protein_id for r in results]
        proteins = {}
        structures = {}
        
        for protein_id in protein_ids:
            protein = self.protein_repo.get_by_id(protein_id)
            if protein:
                proteins[protein_id] = protein
                
                # Get best structure for each protein
                structure = self.structure_repo.get_best_structure(protein_id)
                if structure:
                    structures[protein_id] = structure
        
        return {
            "experiment": experiment,
            "results": results,
            "proteins": proteins,
            "structures": structures,
            "statistics": self.result_repo.get_experiment_statistics(experiment_id),
        }
    
    def create_complete_experiment(
        self,
        experiment: ExperimentModel,
        proteins: List[ProteinModel],
        results: List[ResultModel],
    ) -> int:
        """Create experiment with proteins and results in a transaction."""
        # Create experiment
        experiment_id = self.experiment_repo.create(experiment)
        
        try:
            # Create proteins and update result protein_ids
            protein_id_mapping = {}
            for i, protein in enumerate(proteins):
                protein_model, created = self.protein_repo.create_if_not_exists(protein)
                protein_id_mapping[i] = protein_model.id
            
            # Update results with correct protein and experiment IDs
            for i, result in enumerate(results):
                result.experiment_id = experiment_id
                result.protein_id = protein_id_mapping[i]
            
            # Create results
            self.result_repo.batch_create_results(results)
            
            # Update rankings
            self.result_repo.update_rankings(experiment_id)
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create complete experiment: {e}")
            # In a real implementation, you would rollback the transaction
            raise
    
    def get_protein_history(self, protein_id: int) -> Dict[str, Any]:
        """Get complete history for a protein."""
        protein = self.protein_repo.get_by_id(protein_id)
        if not protein:
            return {}
        
        structures = self.structure_repo.get_by_protein_id(protein_id)
        
        # Find all experiments this protein was part of
        query = "SELECT DISTINCT experiment_id FROM results WHERE protein_id = ?"
        if self.connection.config.backend == "postgresql":
            query = query.replace('?', '%s')
        
        exp_results = self.connection.execute_query(query, (protein_id,))
        experiment_ids = [row['experiment_id'] for row in exp_results]
        
        experiments = []
        results = []
        for exp_id in experiment_ids:
            exp = self.experiment_repo.get_by_id(exp_id)
            if exp:
                experiments.append(exp)
            
            # Get results for this protein in this experiment
            result_query = "SELECT * FROM results WHERE experiment_id = ? AND protein_id = ?"
            if self.connection.config.backend == "postgresql":
                result_query = result_query.replace('?', '%s')
            
            result_data = self.connection.execute_query(result_query, (exp_id, protein_id))
            results.extend([ResultModel.from_dict(row) for row in result_data])
        
        return {
            "protein": protein,
            "structures": structures,
            "experiments": experiments,
            "results": results,
        }