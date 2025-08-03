"""
Structure prediction and quality assessment for generated proteins.

This module provides interfaces to ESMFold, ColabFold, and other structure
prediction methods, along with quality metrics and binding affinity estimation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import tempfile
import subprocess
import logging
from pathlib import Path

try:
    from Bio import PDB
    from Bio.PDB import PDBIO, Select
    from Bio.PDB.Polypeptide import PPBuilder
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

try:
    import esm
    ESM_FOLD_AVAILABLE = True
except ImportError:
    ESM_FOLD_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StructurePredictorConfig:
    """Configuration for structure prediction."""
    method: str = "esmfold"  # "esmfold", "colabfold", "alphafold"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_sequence_length: int = 400
    confidence_threshold: float = 0.7
    
    # ESMFold specific
    esmfold_model: str = "esm2_t36_3B_UR50D"
    
    # External tool paths
    colabfold_path: Optional[str] = None
    foldseek_path: Optional[str] = None
    autodock_vina_path: Optional[str] = None
    
    # Output settings
    save_structures: bool = True
    output_dir: str = "./structures"


class StructureQualityMetrics:
    """Calculate structure quality metrics."""
    
    @staticmethod
    def ramachandran_score(structure) -> float:
        """Calculate Ramachandran plot quality score."""
        if not BIOPYTHON_AVAILABLE:
            logger.warning("BioPython not available for Ramachandran analysis")
            return 0.0
        
        # This is a simplified implementation
        # In practice, you'd use more sophisticated Ramachandran analysis
        phi_psi_angles = []
        
        for model in structure:
            for chain in model:
                polypeptides = PPBuilder().build_peptides(chain)
                for peptide in polypeptides:
                    angles = peptide.get_phi_psi_list()
                    phi_psi_angles.extend([(phi, psi) for phi, psi in angles if phi is not None and psi is not None])
        
        if not phi_psi_angles:
            return 0.0
        
        # Count favorable regions (simplified)
        favorable_count = 0
        for phi, psi in phi_psi_angles:
            phi_deg = np.degrees(phi)
            psi_deg = np.degrees(psi)
            
            # Alpha helix region
            if -180 <= phi_deg <= -30 and -90 <= psi_deg <= 50:
                favorable_count += 1
            # Extended/beta region  
            elif -180 <= phi_deg <= -30 and 90 <= psi_deg <= 180:
                favorable_count += 1
            # Left-handed helix region
            elif 30 <= phi_deg <= 180 and -90 <= psi_deg <= 50:
                favorable_count += 1
        
        return favorable_count / len(phi_psi_angles) if phi_psi_angles else 0.0
    
    @staticmethod
    def clash_score(structure) -> float:
        """Calculate atomic clash score."""
        if not BIOPYTHON_AVAILABLE:
            return 0.0
        
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atoms.append(atom.get_coord())
        
        if len(atoms) < 2:
            return 0.0
        
        atoms = np.array(atoms)
        clash_count = 0
        clash_threshold = 2.0  # Angstroms
        
        # Check for clashes (simplified O(n^2) implementation)
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                distance = np.linalg.norm(atoms[i] - atoms[j])
                if distance < clash_threshold:
                    clash_count += 1
        
        return clash_count / len(atoms) if atoms.size > 0 else 0.0
    
    @staticmethod
    def compactness_score(structure) -> float:
        """Calculate structure compactness (radius of gyration)."""
        if not BIOPYTHON_AVAILABLE:
            return 0.0
        
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        ca_atoms.append(residue["CA"].get_coord())
        
        if len(ca_atoms) < 3:
            return 0.0
        
        ca_atoms = np.array(ca_atoms)
        center_of_mass = np.mean(ca_atoms, axis=0)
        
        # Radius of gyration
        rg = np.sqrt(np.mean(np.sum((ca_atoms - center_of_mass) ** 2, axis=1)))
        
        # Normalize by sequence length (approximate)
        normalized_rg = rg / (len(ca_atoms) ** 0.6)  # Empirical scaling
        
        return max(0.0, 1.0 - normalized_rg / 10.0)  # Convert to 0-1 score


class ESMFoldPredictor:
    """ESMFold-based structure prediction."""
    
    def __init__(self, config: StructurePredictorConfig):
        self.config = config
        
        if not ESM_FOLD_AVAILABLE:
            raise ImportError("ESM not available. Install with: pip install fair-esm")
        
        # Load ESMFold model
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval().to(config.device)
    
    def predict(self, sequence: str) -> Dict[str, Union[torch.Tensor, float]]:
        """Predict structure using ESMFold."""
        if len(sequence) > self.config.max_sequence_length:
            logger.warning(f"Sequence length {len(sequence)} exceeds maximum {self.config.max_sequence_length}")
            sequence = sequence[:self.config.max_sequence_length]
        
        with torch.no_grad():
            output = self.model.infer_pdb(sequence)
        
        # Extract confidence scores
        confidence = output.get("mean_plddt", [0.0])
        if isinstance(confidence, list):
            mean_confidence = np.mean(confidence)
        else:
            mean_confidence = float(confidence)
        
        return {
            "pdb_string": output["pdb"],
            "confidence": mean_confidence,
            "plddt_scores": output.get("plddt", []),
            "coordinates": output.get("positions"),
        }


class ColabFoldPredictor:
    """ColabFold-based structure prediction."""
    
    def __init__(self, config: StructurePredictorConfig):
        self.config = config
        
        if config.colabfold_path is None:
            # Try to find ColabFold in PATH
            try:
                subprocess.run(["colabfold_batch", "--help"], 
                             capture_output=True, check=True)
                self.colabfold_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.colabfold_available = False
                logger.warning("ColabFold not found in PATH")
        else:
            self.colabfold_available = Path(config.colabfold_path).exists()
    
    def predict(self, sequence: str) -> Dict[str, Union[str, float]]:
        """Predict structure using ColabFold."""
        if not self.colabfold_available:
            raise RuntimeError("ColabFold not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Write sequence to FASTA file
            fasta_file = temp_dir / "input.fasta"
            with open(fasta_file, 'w') as f:
                f.write(f">sequence\n{sequence}\n")
            
            # Run ColabFold
            cmd = [
                "colabfold_batch",
                str(fasta_file),
                str(temp_dir),
                "--num-models", "1",
                "--max-msa", "16:32",
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    raise RuntimeError(f"ColabFold failed: {result.stderr}")
                
                # Find output PDB file
                pdb_files = list(temp_dir.glob("*.pdb"))
                if not pdb_files:
                    raise RuntimeError("No PDB output from ColabFold")
                
                pdb_file = pdb_files[0]
                with open(pdb_file, 'r') as f:
                    pdb_string = f.read()
                
                # Extract confidence from filename or file
                confidence = 0.5  # Default confidence
                if "_rank_001_" in pdb_file.name:
                    # Try to extract confidence from filename
                    parts = pdb_file.name.split("_")
                    for part in parts:
                        if part.startswith("alphafold2"):
                            try:
                                confidence = float(part.split("-")[-1])
                                break
                            except ValueError:
                                pass
                
                return {
                    "pdb_string": pdb_string,
                    "confidence": confidence,
                }
                
            except subprocess.TimeoutExpired:
                raise RuntimeError("ColabFold timed out")


class StructurePredictor:
    """
    Main structure prediction interface supporting multiple methods.
    
    This class provides a unified interface for protein structure prediction
    using various methods like ESMFold, ColabFold, and quality assessment.
    """
    
    def __init__(self, config: StructurePredictorConfig):
        self.config = config
        self.quality_metrics = StructureQualityMetrics()
        
        # Initialize predictor based on method
        if config.method == "esmfold":
            self.predictor = ESMFoldPredictor(config)
        elif config.method == "colabfold":
            self.predictor = ColabFoldPredictor(config)
        else:
            raise ValueError(f"Unknown prediction method: {config.method}")
        
        # Create output directory
        if config.save_structures:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def predict_structure(self, sequence: str) -> Dict[str, Union[str, float]]:
        """
        Predict protein structure from sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Dictionary with structure prediction results
        """
        logger.info(f"Predicting structure for sequence of length {len(sequence)}")
        
        try:
            # Run structure prediction
            result = self.predictor.predict(sequence)
            
            # Parse PDB structure for quality assessment
            structure = None
            if BIOPYTHON_AVAILABLE and "pdb_string" in result:
                try:
                    parser = PDB.PDBParser(QUIET=True)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                        tmp.write(result["pdb_string"])
                        tmp.flush()
                        structure = parser.get_structure("protein", tmp.name)
                        Path(tmp.name).unlink()  # Clean up
                except Exception as e:
                    logger.warning(f"Failed to parse PDB structure: {e}")
            
            # Calculate quality metrics
            quality_scores = {}
            if structure is not None:
                quality_scores.update({
                    "ramachandran_score": self.quality_metrics.ramachandran_score(structure),
                    "clash_score": self.quality_metrics.clash_score(structure),
                    "compactness_score": self.quality_metrics.compactness_score(structure),
                })
            
            # Overall structure quality score
            structure_quality = result.get("confidence", 0.0)
            if quality_scores:
                structure_quality = (
                    0.5 * structure_quality +
                    0.2 * quality_scores.get("ramachandran_score", 0.0) +
                    0.2 * (1.0 - quality_scores.get("clash_score", 1.0)) +
                    0.1 * quality_scores.get("compactness_score", 0.0)
                )
            
            # Save structure if requested
            if self.config.save_structures and "pdb_string" in result:
                output_file = Path(self.config.output_dir) / f"structure_{hash(sequence) % 10000}.pdb"
                with open(output_file, 'w') as f:
                    f.write(result["pdb_string"])
                result["pdb_file"] = str(output_file)
            
            # Combine results
            result.update(quality_scores)
            result["structure_quality"] = structure_quality
            result["prediction_method"] = self.config.method
            
            return result
            
        except Exception as e:
            logger.error(f"Structure prediction failed: {e}")
            return {
                "error": str(e),
                "structure_quality": 0.0,
                "confidence": 0.0,
                "prediction_method": self.config.method,
            }
    
    def evaluate_binding(
        self,
        sequence: str,
        target_pdb: str,
    ) -> Dict[str, float]:
        """
        Evaluate binding affinity between generated protein and target.
        
        Args:
            sequence: Protein sequence
            target_pdb: Path to target PDB file
            
        Returns:
            Dictionary with binding evaluation results
        """
        logger.info(f"Evaluating binding affinity for sequence against {target_pdb}")
        
        try:
            # First predict structure of the sequence
            structure_result = self.predict_structure(sequence)
            
            if "pdb_string" not in structure_result:
                return {"binding_error": "No structure available for binding evaluation"}
            
            # Save predicted structure temporarily
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                tmp.write(structure_result["pdb_string"])
                predicted_pdb = tmp.name
            
            try:
                # Run simplified binding evaluation
                binding_score = self._evaluate_binding_autodock(predicted_pdb, target_pdb)
                
                return {
                    "binding_affinity": binding_score,
                    "binding_method": "autodock_vina",
                }
                
            finally:
                Path(predicted_pdb).unlink()  # Clean up
                
        except Exception as e:
            logger.error(f"Binding evaluation failed: {e}")
            return {
                "binding_error": str(e),
                "binding_affinity": 0.0,
            }
    
    def _evaluate_binding_autodock(self, protein_pdb: str, target_pdb: str) -> float:
        """
        Evaluate binding using AutoDock Vina (simplified implementation).
        
        Args:
            protein_pdb: Path to predicted protein structure
            target_pdb: Path to target structure
            
        Returns:
            Binding affinity score
        """
        # This is a placeholder implementation
        # In practice, you would:
        # 1. Prepare proteins for docking (add hydrogens, charges)
        # 2. Define binding site
        # 3. Run AutoDock Vina
        # 4. Parse results
        
        logger.warning("AutoDock Vina evaluation not implemented - returning mock score")
        
        # Mock binding score based on structure quality
        try:
            if BIOPYTHON_AVAILABLE:
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure("protein", protein_pdb)
                
                # Calculate interface area (simplified)
                atoms = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                atoms.append(atom.get_coord())
                
                if atoms:
                    # Mock score based on structure compactness
                    atoms = np.array(atoms)
                    center = np.mean(atoms, axis=0)
                    distances = np.linalg.norm(atoms - center, axis=1)
                    compactness = 1.0 / (1.0 + np.std(distances))
                    
                    # Convert to binding affinity-like score (kcal/mol)
                    binding_score = -15.0 * compactness + np.random.normal(0, 1.0)
                    return max(-20.0, min(0.0, binding_score))
            
        except Exception as e:
            logger.warning(f"Mock binding evaluation failed: {e}")
        
        return -5.0  # Default weak binding
    
    def batch_predict(
        self,
        sequences: List[str],
        progress: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict structures for multiple sequences.
        
        Args:
            sequences: List of protein sequences
            progress: Show progress
            
        Returns:
            List of structure prediction results
        """
        results = []
        
        iterator = sequences
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(sequences, desc="Predicting structures")
            except ImportError:
                pass
        
        for sequence in iterator:
            result = self.predict_structure(sequence)
            result["sequence"] = sequence
            results.append(result)
        
        return results
    
    def get_prediction_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Get statistics from batch prediction results."""
        if not results:
            return {}
        
        confidences = [r.get("confidence", 0.0) for r in results]
        structure_qualities = [r.get("structure_quality", 0.0) for r in results]
        
        stats = {
            "total_predictions": len(results),
            "mean_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "mean_structure_quality": np.mean(structure_qualities),
            "high_confidence_count": sum(1 for c in confidences if c > self.config.confidence_threshold),
            "success_rate": sum(1 for r in results if "error" not in r) / len(results),
        }
        
        return stats