"""
Main ProteinDiffuser class for high-level protein generation.

This module provides the primary interface for protein scaffold generation
using diffusion models, combining tokenization, model inference, and sampling.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    nn = torch.nn
    TORCH_AVAILABLE = False
from typing import List, Optional, Dict, Union, Tuple
from pathlib import Path
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Mock numpy functions we use
    class MockRandom:
        @staticmethod
        def normal(mean=0, std=1, size=None):
            import random
            if size:
                return [random.gauss(mean, std) for _ in range(size)]
            return random.gauss(mean, std)
    
    class MockNumpy:
        random = MockRandom()
        
        @staticmethod
        def mean(arr):
            if isinstance(arr, list):
                return sum(arr) / len(arr) if arr else 0
            return 0.5
        
        @staticmethod
        def array(data):
            return data
    
    np = MockNumpy()
    NUMPY_AVAILABLE = False
from dataclasses import dataclass
import logging
import warnings

from .models import DiffusionTransformer, DDPM, DiffusionTransformerConfig, DDPMConfig
from .tokenization.selfies_tokenizer import SELFIESTokenizer, TokenizerConfig
from .tokenization.protein_embeddings import ProteinEmbeddings, EmbeddingConfig
from .folding.structure_predictor import StructurePredictor, StructurePredictorConfig
try:
    from .validation import ValidationError as ProteinValidationError, ValidationManager
    from .security import SecurityManager, SecurityConfig
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    ProteinValidationError = ValueError
    ValidationManager = None
    SecurityManager = None
    SecurityConfig = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProteinDiffuserConfig:
    """Configuration for the ProteinDiffuser."""
    model_config: DiffusionTransformerConfig = None
    ddpm_config: DDPMConfig = None
    tokenizer_config: TokenizerConfig = None
    embedding_config: EmbeddingConfig = None
    structure_config: StructurePredictorConfig = None
    
    # Generation parameters
    num_samples: int = 10
    max_length: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    guidance_scale: float = 1.0
    
    # Device and precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = DiffusionTransformerConfig()
        if self.ddpm_config is None:
            self.ddpm_config = DDPMConfig()
        if self.tokenizer_config is None:
            self.tokenizer_config = TokenizerConfig()
        if self.embedding_config is None:
            self.embedding_config = EmbeddingConfig()
        if self.structure_config is None:
            self.structure_config = StructurePredictorConfig()


class ProteinDiffuser:
    """
    High-level interface for protein scaffold generation using diffusion models.
    
    This class provides a simple API for generating protein scaffolds with
    specified motifs and constraints, handling all the complexity of tokenization,
    model inference, and structure prediction.
    
    Example:
        >>> diffuser = ProteinDiffuser.from_pretrained("path/to/checkpoint")
        >>> scaffolds = diffuser.generate(
        ...     motif="HELIX_SHEET_HELIX",
        ...     num_samples=100,
        ...     temperature=0.8
        ... )
    """
    
    def __init__(
        self,
        config: ProteinDiffuserConfig,
        model: Optional[DiffusionTransformer] = None,
        tokenizer: Optional[SELFIESTokenizer] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SELFIESTokenizer(config.tokenizer_config)
        
        # Initialize embeddings
        self.embeddings = ProteinEmbeddings(config.embedding_config)
        
        # Initialize model
        if model is not None:
            self.model = model
        else:
            self.model = DiffusionTransformer(config.model_config)
        
        # Initialize DDPM
        self.ddpm = DDPM(self.model, config.ddpm_config)
        
        # Initialize structure predictor
        self.structure_predictor = StructurePredictor(config.structure_config)
        
        # Move to device
        self.model.to(self.device, dtype=config.dtype)
        self.ddpm.to(self.device)
        self.embeddings.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        self.embeddings.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[ProteinDiffuserConfig] = None,
        device: Optional[str] = None,
    ) -> 'ProteinDiffuser':
        """
        Load a pre-trained ProteinDiffuser from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Optional configuration override
            device: Device to load on
            
        Returns:
            Loaded ProteinDiffuser instance
        """
        checkpoint_path = Path(checkpoint_path)
        
        if config is None:
            config = ProteinDiffuserConfig()
        
        if device is not None:
            config.device = device
        
        # Load model
        model = DiffusionTransformer.from_pretrained(checkpoint_path / "model.pt")
        
        # Load tokenizer if available
        tokenizer = None
        if (checkpoint_path / "tokenizer").exists():
            tokenizer = SELFIESTokenizer.from_pretrained(checkpoint_path / "tokenizer")
        
        return cls(config, model=model, tokenizer=tokenizer)
    
    def save_pretrained(self, save_path: str):
        """
        Save the ProteinDiffuser to disk.
        
        Args:
            save_path: Directory to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path / "model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path / "tokenizer")
        
        # Save config
        import json
        with open(save_path / "config.json", 'w') as f:
            # Note: This is a simplified config save
            json.dump({
                "model_name": "ProteinDiffuser",
                "version": "0.1.0",
            }, f, indent=2)
    
    def generate(
        self,
        motif: Optional[str] = None,
        num_samples: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        sampling_method: str = "ddpm",
        ddim_steps: int = 50,
        progress: bool = True,
        return_intermediates: bool = False,
        client_id: str = "default",
    ) -> List[Dict[str, Union[str, torch.Tensor, float]]]:
        """
        Generate protein scaffolds using the diffusion model.
        
        Args:
            motif: Target motif specification (e.g., "HELIX_SHEET_HELIX")
            num_samples: Number of samples to generate
            max_length: Maximum sequence length
            temperature: Sampling temperature
            guidance_scale: Classifier-free guidance scale
            sampling_method: "ddpm" or "ddim"
            ddim_steps: Number of DDIM steps (if using DDIM)
            progress: Show progress bar
            return_intermediates: Return intermediate sampling states
            client_id: Client identifier for rate limiting
            
        Returns:
            List of generated protein scaffolds with metadata
        """
        try:
            # Input validation and sanitization
            if SECURITY_AVAILABLE:
                validation_manager = ValidationManager()
                generation_params = {
                    'num_samples': num_samples or self.config.num_samples,
                    'max_length': max_length or self.config.max_length,
                    'temperature': temperature or self.config.temperature,
                    'guidance_scale': guidance_scale or self.config.guidance_scale,
                }
                validation_result = validation_manager.comprehensive_validation(
                    generation_params=generation_params
                )
                
                if not validation_result.is_valid:
                    raise ProteinValidationError(f"Validation failed: {validation_result.errors}")
                
                if validation_result.warnings:
                    for warning in validation_result.warnings:
                        logger.warning(f"Generation warning: {warning}")
            
            # Use config defaults if not specified
            if num_samples is None:
                num_samples = self.config.num_samples
            if max_length is None:
                max_length = self.config.max_length
            if temperature is None:
                temperature = self.config.temperature
            if guidance_scale is None:
                guidance_scale = self.config.guidance_scale
            
            # Additional safety checks
            if num_samples <= 0 or num_samples > 1000:
                raise ValueError(f"Invalid num_samples: {num_samples}")
            if max_length <= 0 or max_length > 2048:
                raise ValueError(f"Invalid max_length: {max_length}")
            if temperature <= 0 or temperature > 5.0:
                raise ValueError(f"Invalid temperature: {temperature}")
            if sampling_method not in ["ddpm", "ddim"]:
                raise ValueError(f"Unknown sampling method: {sampling_method}")
            
            logger.info(f"Generating {num_samples} protein scaffolds with motif: {motif}")
            
            with torch.no_grad():
                try:
                    # Prepare motif conditioning
                    motif_conditioning = None
                    if motif is not None:
                        motif_conditioning = self._encode_motif(motif, num_samples, max_length)
                    
                    # Generate samples
                    if sampling_method == "ddpm":
                        samples = self.ddpm.sample(
                            shape=(num_samples, max_length),
                            device=self.device,
                            motif_conditioning=motif_conditioning,
                            progress=progress,
                        )
                    elif sampling_method == "ddim":
                        samples = self.ddpm.ddim_sample(
                            shape=(num_samples, max_length),
                            device=self.device,
                            ddim_steps=ddim_steps,
                            motif_conditioning=motif_conditioning,
                            progress=progress,
                        )
                    else:
                        raise ValueError(f"Unknown sampling method: {sampling_method}")
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        samples = samples / temperature
                    
                    # Check for NaN or inf values
                    if torch.isnan(samples).any():
                        logger.error("NaN values detected in generated samples")
                        raise RuntimeError("Generation produced NaN values")
                    if torch.isinf(samples).any():
                        logger.error("Infinite values detected in generated samples")
                        raise RuntimeError("Generation produced infinite values")
                    
                    # Convert to sequences
                    results = []
                    for i, sample in enumerate(samples):
                        try:
                            # Convert logits to tokens
                            token_ids = torch.argmax(sample, dim=-1).cpu()
                            
                            # Decode to sequence
                            sequence = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                            
                            # Basic sequence validation
                            if not sequence or len(sequence) < 5:
                                logger.warning(f"Generated very short sequence {i}: {sequence}")
                                continue
                            
                            # Calculate confidence (approximation from logit magnitudes)
                            with torch.no_grad():
                                confidence = torch.max(torch.softmax(sample, dim=-1), dim=-1)[0].mean().item()
                            
                            result = {
                                "sequence": sequence,
                                "token_ids": token_ids,
                                "logits": sample.cpu() if return_intermediates else None,
                                "confidence": confidence,
                                "length": len(sequence),
                                "motif": motif,
                                "sample_id": i,
                                "generation_params": {
                                    "temperature": temperature,
                                    "guidance_scale": guidance_scale,
                                    "sampling_method": sampling_method,
                                }
                            }
                            
                            results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Error processing sample {i}: {e}")
                            continue
                    
                    if not results:
                        raise RuntimeError("No valid sequences generated")
                    
                    logger.info(f"Generated {len(results)} sequences with average length {np.mean([r['length'] for r in results]):.1f}")
                    
                    return results
                
                except torch.cuda.OutOfMemoryError:
                    logger.error("GPU out of memory during generation")
                    raise RuntimeError("Insufficient GPU memory for generation")
                except Exception as e:
                    logger.error(f"Error during sample generation: {e}")
                    raise
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return empty list with error information instead of crashing
            return [{
                "error": str(e),
                "sequence": "",
                "confidence": 0.0,
                "length": 0,
                "sample_id": -1,
            }]
    
    def _encode_motif(self, motif: str, batch_size: int, max_length: int) -> torch.Tensor:
        """
        Encode motif specification for conditioning.
        
        Args:
            motif: Motif specification string
            batch_size: Batch size for conditioning
            max_length: Maximum sequence length
            
        Returns:
            Encoded motif conditioning tensor
        """
        # Parse motif specification
        if "_" in motif:
            # Structured motif like "HELIX_SHEET_HELIX"
            motif_parts = motif.split("_")
            motif_sequence = "".join([self._motif_to_sequence(part) for part in motif_parts])
        else:
            # Direct sequence motif
            motif_sequence = motif
        
        # Encode motif sequence
        motif_encoding = self.tokenizer.encode(
            motif_sequence,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        
        # Generate embeddings
        motif_embeddings = self.embeddings([motif_sequence])
        
        # Expand for batch
        motif_conditioning = motif_embeddings.repeat(batch_size, 1, 1)
        
        return motif_conditioning.to(self.device, dtype=self.config.dtype)
    
    def _motif_to_sequence(self, motif: str) -> str:
        """
        Convert motif specification to representative sequence.
        
        Args:
            motif: Motif name (e.g., "HELIX", "SHEET")
            
        Returns:
            Representative amino acid sequence
        """
        motif_templates = {
            "HELIX": "AEAAAKEAAAKA",  # Alpha helix template
            "SHEET": "IVIVIV",       # Beta sheet template
            "LOOP": "GGSGGS",        # Flexible loop template
            "TURN": "PGGP",          # Beta turn template
            "COIL": "GGPGAG",        # Random coil template
        }
        
        return motif_templates.get(motif.upper(), "GGGG")  # Default flexible linker
    
    def evaluate_sequences(
        self,
        sequences: List[str],
        target_pdb: Optional[str] = None,
        compute_structure: bool = True,
        compute_binding: bool = False,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Evaluate generated protein sequences for quality and properties.
        
        Args:
            sequences: List of protein sequences to evaluate
            target_pdb: Path to target PDB for binding evaluation
            compute_structure: Compute structural properties
            compute_binding: Compute binding affinity (requires target_pdb)
            
        Returns:
            List of evaluation results for each sequence
        """
        logger.info(f"Evaluating {len(sequences)} sequences")
        
        results = []
        for i, sequence in enumerate(sequences):
            result = {
                "sequence": sequence,
                "length": len(sequence),
                "sample_id": i,
            }
            
            # Basic sequence properties
            result.update(self._compute_sequence_properties(sequence))
            
            # Structural evaluation
            if compute_structure:
                try:
                    structure_result = self.structure_predictor.predict_structure(sequence)
                    result.update(structure_result)
                except Exception as e:
                    logger.warning(f"Structure prediction failed for sequence {i}: {e}")
                    result["structure_error"] = str(e)
            
            # Binding evaluation
            if compute_binding and target_pdb is not None:
                try:
                    binding_result = self.structure_predictor.evaluate_binding(sequence, target_pdb)
                    result.update(binding_result)
                except Exception as e:
                    logger.warning(f"Binding evaluation failed for sequence {i}: {e}")
                    result["binding_error"] = str(e)
            
            results.append(result)
        
        return results
    
    def _compute_sequence_properties(self, sequence: str) -> Dict[str, float]:
        """Compute basic sequence properties."""
        # Amino acid composition
        aa_counts = {aa: sequence.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        total_aa = len(sequence)
        
        # Hydrophobicity (Kyte-Doolittle scale)
        hydrophobicity_scale = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        hydrophobicity = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / total_aa
        
        # Charge at pH 7
        charge_scale = {
            'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1
        }
        net_charge = sum(charge_scale.get(aa, 0) for aa in sequence)
        
        return {
            "hydrophobicity": hydrophobicity,
            "net_charge": net_charge,
            "charge_density": net_charge / total_aa,
            "glycine_content": aa_counts.get('G', 0) / total_aa,
            "proline_content": aa_counts.get('P', 0) / total_aa,
            "aromatic_content": sum(aa_counts.get(aa, 0) for aa in "FWY") / total_aa,
        }
    
    def rank_by_fitness(
        self,
        sequences: List[str],
        criteria: List[str] = None,
        weights: Dict[str, float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank sequences by fitness criteria.
        
        Args:
            sequences: List of sequences to rank
            criteria: List of criteria to use for ranking
            weights: Weights for each criterion
            
        Returns:
            List of (sequence, fitness_score) tuples, sorted by fitness
        """
        if criteria is None:
            criteria = ["confidence", "structure_quality", "binding_affinity"]
        
        if weights is None:
            weights = {criterion: 1.0 for criterion in criteria}
        
        # Evaluate sequences
        evaluations = self.evaluate_sequences(
            sequences,
            compute_structure=("structure_quality" in criteria),
            compute_binding=("binding_affinity" in criteria),
        )
        
        # Calculate fitness scores
        scored_sequences = []
        for eval_result in evaluations:
            fitness_score = 0.0
            for criterion in criteria:
                if criterion in eval_result:
                    fitness_score += weights.get(criterion, 1.0) * eval_result[criterion]
            
            scored_sequences.append((eval_result["sequence"], fitness_score))
        
        # Sort by fitness score (descending)
        scored_sequences.sort(key=lambda x: x[1], reverse=True)
        
        return scored_sequences
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model."""
        return {
            "model_type": "DiffusionTransformer",
            "parameters": self.model.get_num_parameters(),
            "vocab_size": len(self.tokenizer),
            "max_length": self.config.max_length,
            "device": str(self.device),
            "dtype": str(self.config.dtype),
        }