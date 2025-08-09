"""
Affinity ranking and protein evaluation system.

This module provides comprehensive ranking and evaluation of generated
protein scaffolds based on binding affinity, structural quality, and
other biophysical properties.
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def mean(arr):
            return sum(arr)/len(arr) if arr else 0.5
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def std(arr):
            return 1.0
        @staticmethod
        def min(arr):
            return min(arr) if arr else 0
        @staticmethod
        def max(arr):
            return max(arr) if arr else 1
    np = MockNumpy()
    NUMPY_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    TORCH_AVAILABLE = False
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from .folding.structure_predictor import StructurePredictor, StructurePredictorConfig

logger = logging.getLogger(__name__)


@dataclass
class AffinityRankerConfig:
    """Configuration for affinity ranking system."""
    # Ranking criteria weights
    binding_weight: float = 0.4
    structure_weight: float = 0.3
    diversity_weight: float = 0.2
    novelty_weight: float = 0.1
    
    # Quality thresholds
    min_confidence: float = 0.7
    min_structure_quality: float = 0.6
    max_clash_score: float = 0.1
    
    # Diversity parameters
    diversity_threshold: float = 0.8
    sequence_similarity_threshold: float = 0.7
    
    # Output settings  
    max_results: int = 100
    save_rankings: bool = True
    output_dir: str = "./rankings"


class SequenceSimilarity:
    """Calculate sequence similarity and diversity metrics."""
    
    @staticmethod
    def levenshtein_distance(seq1: str, seq2: str) -> int:
        """Calculate Levenshtein distance between two sequences."""
        if len(seq1) < len(seq2):
            return SequenceSimilarity.levenshtein_distance(seq2, seq1)
        
        if len(seq2) == 0:
            return len(seq1)
        
        previous_row = list(range(len(seq2) + 1))
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def sequence_identity(seq1: str, seq2: str) -> float:
        """Calculate sequence identity percentage."""
        if not seq1 or not seq2:
            return 0.0
        
        # Align sequences to same length (simple approach)
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max(len(seq1), len(seq2))
    
    @staticmethod
    def hamming_similarity(seq1: str, seq2: str) -> float:
        """Calculate Hamming similarity (for equal length sequences)."""
        if len(seq1) != len(seq2):
            # Normalize to same length
            max_len = max(len(seq1), len(seq2))
            seq1 = seq1.ljust(max_len, 'A')
            seq2 = seq2.ljust(max_len, 'A')
        
        matches = sum(1 for c1, c2 in zip(seq1, seq2) if c1 == c2)
        return matches / len(seq1) if seq1 else 0.0
    
    @staticmethod
    def calculate_diversity_score(sequences: List[str]) -> float:
        """Calculate overall diversity score for a set of sequences."""
        if len(sequences) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                similarity = SequenceSimilarity.sequence_identity(sequences[i], sequences[j])
                similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity


class BindingAffinityPredictor:
    """Predict binding affinity using ML models or physics-based methods."""
    
    def __init__(self, config: AffinityRankerConfig):
        self.config = config
        
        # Initialize ML models for affinity prediction
        # In practice, this would load pre-trained models
        self.sequence_model = None
        self.structure_model = None
    
    def predict_from_sequence(self, sequence: str, target_info: Dict[str, Any]) -> float:
        """
        Predict binding affinity from sequence alone.
        
        Args:
            sequence: Protein sequence
            target_info: Information about binding target
            
        Returns:
            Predicted binding affinity (kcal/mol)
        """
        # Simplified affinity prediction based on sequence features
        # In practice, this would use trained ML models
        
        # Calculate basic sequence features
        hydrophobic_residues = "AILMFPWV"
        charged_residues = "DEKR"
        aromatic_residues = "FWY"
        
        hydrophobic_content = sum(1 for aa in sequence if aa in hydrophobic_residues) / len(sequence)
        charged_content = sum(1 for aa in sequence if aa in charged_residues) / len(sequence)
        aromatic_content = sum(1 for aa in sequence if aa in aromatic_residues) / len(sequence)
        
        # Simple linear model (placeholder)
        affinity = (
            -8.0 +  # Base affinity
            -5.0 * hydrophobic_content +
            -3.0 * aromatic_content +
            2.0 * charged_content +
            np.random.normal(0, 1.0)  # Add noise
        )
        
        return max(-20.0, min(0.0, affinity))
    
    def predict_from_structure(
        self,
        structure_data: Dict[str, Any],
        target_pdb: str,
    ) -> float:
        """
        Predict binding affinity from 3D structure.
        
        Args:
            structure_data: Structure prediction results
            target_pdb: Path to target PDB file
            
        Returns:
            Predicted binding affinity (kcal/mol)
        """
        # Use structure quality as proxy for binding potential
        structure_quality = structure_data.get("structure_quality", 0.0)
        confidence = structure_data.get("confidence", 0.0)
        
        # Simple structure-based affinity model
        affinity = (
            -12.0 * structure_quality +
            -8.0 * confidence +
            np.random.normal(0, 0.5)
        )
        
        return max(-20.0, min(0.0, affinity))


class NoveltyScorer:
    """Score protein novelty compared to known structures."""
    
    def __init__(self, known_sequences: Optional[List[str]] = None):
        self.known_sequences = known_sequences or []
        self.similarity_calculator = SequenceSimilarity()
    
    def calculate_novelty(self, sequence: str) -> float:
        """
        Calculate novelty score for a sequence.
        
        Args:
            sequence: Protein sequence to evaluate
            
        Returns:
            Novelty score (0-1, higher is more novel)
        """
        if not self.known_sequences:
            return 1.0  # Assume novel if no reference sequences
        
        # Find maximum similarity to known sequences
        max_similarity = 0.0
        for known_seq in self.known_sequences:
            similarity = self.similarity_calculator.sequence_identity(sequence, known_seq)
            max_similarity = max(max_similarity, similarity)
        
        # Novelty is inverse of similarity
        return 1.0 - max_similarity
    
    def add_known_sequences(self, sequences: List[str]):
        """Add sequences to the known sequence database."""
        self.known_sequences.extend(sequences)


class AffinityRanker:
    """
    Comprehensive protein ranking system based on multiple criteria.
    
    This class evaluates and ranks generated protein scaffolds based on:
    - Predicted binding affinity
    - Structural quality and stability
    - Sequence diversity
    - Novelty compared to known proteins
    
    Example:
        >>> ranker = AffinityRanker()
        >>> ranked_proteins = ranker.rank(
        ...     scaffolds,
        ...     target_pdb='targets/spike_protein.pdb'
        ... )
    """
    
    def __init__(self, config: Optional[AffinityRankerConfig] = None):
        if config is None:
            config = AffinityRankerConfig()
        self.config = config
        
        # Initialize components
        structure_config = StructurePredictorConfig()
        self.structure_predictor = StructurePredictor(structure_config)
        self.affinity_predictor = BindingAffinityPredictor(config)
        self.novelty_scorer = NoveltyScorer()
        self.similarity_calculator = SequenceSimilarity()
        
        # Create output directory
        if config.save_rankings:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def rank(
        self,
        sequences: List[str],
        target_pdb: Optional[str] = None,
        reference_sequences: Optional[List[str]] = None,
        return_detailed: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Rank protein sequences by predicted binding affinity and quality.
        
        Args:
            sequences: List of protein sequences to rank
            target_pdb: Path to target PDB for binding evaluation
            reference_sequences: Known sequences for novelty scoring
            return_detailed: Return detailed scoring breakdown
            
        Returns:
            List of ranked protein results with scores
        """
        logger.info(f"Ranking {len(sequences)} protein sequences")
        
        # Add reference sequences for novelty scoring
        if reference_sequences:
            self.novelty_scorer.add_known_sequences(reference_sequences)
        
        # Evaluate all sequences
        results = []
        for i, sequence in enumerate(sequences):
            logger.debug(f"Evaluating sequence {i+1}/{len(sequences)}")
            
            result = {
                "sequence": sequence,
                "sequence_id": i,
                "length": len(sequence),
            }
            
            # Structure prediction and quality assessment
            try:
                structure_result = self.structure_predictor.predict_structure(sequence)
                result.update(structure_result)
            except Exception as e:
                logger.warning(f"Structure prediction failed for sequence {i}: {e}")
                result["structure_error"] = str(e)
                result["structure_quality"] = 0.0
                result["confidence"] = 0.0
            
            # Binding affinity prediction
            try:
                if target_pdb and "structure_quality" in result:
                    # Structure-based prediction if available
                    binding_affinity = self.affinity_predictor.predict_from_structure(
                        result, target_pdb
                    )
                else:
                    # Sequence-based prediction as fallback
                    target_info = {"pdb_path": target_pdb} if target_pdb else {}
                    binding_affinity = self.affinity_predictor.predict_from_sequence(
                        sequence, target_info
                    )
                
                result["binding_affinity"] = binding_affinity
                
            except Exception as e:
                logger.warning(f"Binding prediction failed for sequence {i}: {e}")
                result["binding_affinity"] = 0.0
                result["binding_error"] = str(e)
            
            # Novelty scoring
            try:
                novelty_score = self.novelty_scorer.calculate_novelty(sequence)
                result["novelty_score"] = novelty_score
            except Exception as e:
                logger.warning(f"Novelty scoring failed for sequence {i}: {e}")
                result["novelty_score"] = 0.5
            
            results.append(result)
        
        # Calculate diversity scores
        self._calculate_diversity_scores(results)
        
        # Calculate composite scores and rank
        self._calculate_composite_scores(results)
        
        # Sort by composite score (descending)
        results.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)
        
        # Apply quality filters
        filtered_results = self._apply_quality_filters(results)
        
        # Limit results
        if len(filtered_results) > self.config.max_results:
            filtered_results = filtered_results[:self.config.max_results]
        
        # Save rankings if requested
        if self.config.save_rankings:
            self._save_rankings(filtered_results)
        
        logger.info(f"Ranking complete. {len(filtered_results)} sequences passed filters.")
        
        if return_detailed:
            return filtered_results
        else:
            # Return simplified results
            return [
                {
                    "sequence": r["sequence"],
                    "composite_score": r["composite_score"],
                    "binding_affinity": r.get("binding_affinity", 0.0),
                    "structure_quality": r.get("structure_quality", 0.0),
                }
                for r in filtered_results
            ]
    
    def _calculate_diversity_scores(self, results: List[Dict]):
        """Calculate diversity scores for all sequences."""
        sequences = [r["sequence"] for r in results]
        
        for i, result in enumerate(results):
            # Calculate average similarity to all other sequences
            similarities = []
            for j, other_seq in enumerate(sequences):
                if i != j:
                    similarity = self.similarity_calculator.sequence_identity(
                        result["sequence"], other_seq
                    )
                    similarities.append(similarity)
            
            # Diversity score is inverse of average similarity
            if similarities:
                avg_similarity = np.mean(similarities)
                result["diversity_score"] = 1.0 - avg_similarity
            else:
                result["diversity_score"] = 1.0
    
    def _calculate_composite_scores(self, results: List[Dict]):
        """Calculate weighted composite scores."""
        for result in results:
            # Extract individual scores with defaults
            binding_score = result.get("binding_affinity", 0.0)
            structure_score = result.get("structure_quality", 0.0)
            diversity_score = result.get("diversity_score", 0.0)
            novelty_score = result.get("novelty_score", 0.0)
            
            # Normalize binding affinity to 0-1 scale (assuming range -20 to 0 kcal/mol)
            normalized_binding = max(0.0, (binding_score + 20.0) / 20.0)
            
            # Calculate weighted composite score
            composite_score = (
                self.config.binding_weight * normalized_binding +
                self.config.structure_weight * structure_score +
                self.config.diversity_weight * diversity_score +
                self.config.novelty_weight * novelty_score
            )
            
            result["composite_score"] = composite_score
            
            # Store normalized components for analysis
            result["normalized_binding"] = normalized_binding
    
    def _apply_quality_filters(self, results: List[Dict]) -> List[Dict]:
        """Apply quality filters to remove low-quality results."""
        filtered = []
        
        for result in results:
            # Check confidence threshold
            if result.get("confidence", 0.0) < self.config.min_confidence:
                continue
            
            # Check structure quality threshold
            if result.get("structure_quality", 0.0) < self.config.min_structure_quality:
                continue
            
            # Check clash score threshold
            if result.get("clash_score", 0.0) > self.config.max_clash_score:
                continue
            
            # Skip sequences with errors
            if "structure_error" in result or "binding_error" in result:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def _save_rankings(self, results: List[Dict]):
        """Save ranking results to file."""
        import json
        from datetime import datetime
        
        output_file = Path(self.config.output_dir) / f"rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare results for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    serializable_result[key] = value
                elif isinstance(value, torch.Tensor):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = str(value)
            serializable_results.append(serializable_result)
        
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": self.config.__dict__,
                "total_sequences": len(serializable_results),
                "results": serializable_results,
            }, f, indent=2)
        
        logger.info(f"Rankings saved to {output_file}")
    
    def get_top_candidates(
        self,
        sequences: List[str],
        target_pdb: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k candidates with simplified interface.
        
        Args:
            sequences: List of sequences to evaluate
            target_pdb: Target PDB file path
            top_k: Number of top candidates to return
            
        Returns:
            List of (sequence, score) tuples
        """
        results = self.rank(sequences, target_pdb, return_detailed=False)
        top_results = results[:top_k]
        
        return [(r["sequence"], r["composite_score"]) for r in top_results]
    
    def diversify_selection(
        self,
        sequences: List[str],
        target_pdb: Optional[str] = None,
        max_selections: int = 20,
        similarity_threshold: float = 0.7,
    ) -> List[str]:
        """
        Select diverse set of high-quality sequences.
        
        Args:
            sequences: Input sequences
            target_pdb: Target for binding evaluation
            max_selections: Maximum number to select
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            List of diverse, high-quality sequences
        """
        # Rank all sequences
        results = self.rank(sequences, target_pdb)
        
        # Select diverse subset
        selected = []
        for result in results:
            if len(selected) >= max_selections:
                break
            
            sequence = result["sequence"]
            
            # Check if sufficiently different from already selected
            is_diverse = True
            for selected_seq in selected:
                similarity = self.similarity_calculator.sequence_identity(sequence, selected_seq)
                if similarity > similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(sequence)
        
        logger.info(f"Selected {len(selected)} diverse sequences from {len(sequences)} candidates")
        
        return selected
    
    def get_ranking_statistics(self, results: List[Dict]) -> Dict[str, float]:
        """Get statistics from ranking results."""
        if not results:
            return {}
        
        binding_affinities = [r.get("binding_affinity", 0.0) for r in results]
        structure_qualities = [r.get("structure_quality", 0.0) for r in results]
        composite_scores = [r.get("composite_score", 0.0) for r in results]
        
        stats = {
            "total_sequences": len(results),
            "mean_binding_affinity": np.mean(binding_affinities),
            "std_binding_affinity": np.std(binding_affinities),
            "best_binding_affinity": np.min(binding_affinities),  # More negative is better
            "mean_structure_quality": np.mean(structure_qualities),
            "std_structure_quality": np.std(structure_qualities),
            "mean_composite_score": np.mean(composite_scores),
            "diversity_score": SequenceSimilarity.calculate_diversity_score([r["sequence"] for r in results]),
        }
        
        return stats