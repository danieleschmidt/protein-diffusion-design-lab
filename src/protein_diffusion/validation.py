"""
Validation and error handling utilities for protein diffusion models.

This module provides comprehensive validation for protein sequences, 
model inputs, and system states to ensure robust operation.
"""

import re
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    TORCH_AVAILABLE = False
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
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Standard amino acid codes
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
EXTENDED_AA = STANDARD_AA.union({"U", "O", "X"})  # Include selenocysteine, pyrrolysine, unknown

class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class ProteinValidationError(ValueError):
    """Base exception for protein validation errors."""
    pass


class SequenceValidationError(ProteinValidationError):
    """Exception for sequence validation errors."""
    pass


class ModelValidationError(ProteinValidationError):
    """Exception for model validation errors."""
    pass


class SystemValidationError(ProteinValidationError):
    """Exception for system/environment validation errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ModelValidationError(ValidationError):
    """Exception for model-specific validation errors."""
    pass

class SequenceValidationError(ValidationError):
    """Exception for sequence validation errors."""
    pass

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_suggestion(self, suggestion: str):
        """Add a suggestion."""
        self.suggestions.append(suggestion)
    
    def merge(self, other: 'ValidationResult'):
        """Merge with another validation result."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.suggestions.extend(other.suggestions)

class SequenceValidator:
    """Validate protein sequences."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        self.level = level
    
    def validate_sequence(self, sequence: str) -> ValidationResult:
        """
        Validate a protein sequence.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Basic checks
        if not sequence:
            result.add_error("Empty sequence")
            return result
        
        if not isinstance(sequence, str):
            result.add_error(f"Sequence must be string, got {type(sequence)}")
            return result
        
        # Clean sequence
        clean_seq = sequence.upper().strip()
        
        # Length checks
        if len(clean_seq) < 10:
            if self.level == ValidationLevel.STRICT:
                result.add_error(f"Sequence too short: {len(clean_seq)} < 10")
            else:
                result.add_warning(f"Very short sequence: {len(clean_seq)} amino acids")
        
        if len(clean_seq) > 2000:
            if self.level == ValidationLevel.STRICT:
                result.add_error(f"Sequence too long: {len(clean_seq)} > 2000")
            else:
                result.add_warning(f"Very long sequence: {len(clean_seq)} amino acids")
        
        # Character validation
        invalid_chars = set(clean_seq) - EXTENDED_AA
        if invalid_chars:
            if self.level == ValidationLevel.STRICT:
                result.add_error(f"Invalid amino acid codes: {invalid_chars}")
            else:
                result.add_warning(f"Non-standard characters found: {invalid_chars}")
                result.add_suggestion("Consider removing non-standard characters")
        
        # Composition checks
        x_count = clean_seq.count('X')
        if x_count > len(clean_seq) * 0.1:  # More than 10% unknown
            result.add_warning(f"High unknown residue content: {x_count/len(clean_seq)*100:.1f}%")
        
        # Repetitive sequence detection
        self._check_repetitive_sequences(clean_seq, result)
        
        # Unusual patterns
        self._check_unusual_patterns(clean_seq, result)
        
        return result
    
    def _check_repetitive_sequences(self, sequence: str, result: ValidationResult):
        """Check for repetitive sequences."""
        seq_len = len(sequence)
        
        # Check for simple repeats
        for repeat_len in range(2, min(20, seq_len // 3)):
            for i in range(seq_len - repeat_len * 3):
                motif = sequence[i:i + repeat_len]
                if (sequence[i:i + repeat_len * 3] == motif * 3):
                    result.add_warning(f"Repetitive motif detected: {motif} (position {i})")
                    break
        
        # Check for low complexity regions
        window_size = min(20, seq_len)
        for i in range(seq_len - window_size + 1):
            window = sequence[i:i + window_size]
            unique_aa = len(set(window))
            if unique_aa < 4:  # Less than 4 unique amino acids in 20-residue window
                result.add_warning(f"Low complexity region at position {i}-{i + window_size}")
    
    def _check_unusual_patterns(self, sequence: str, result: ValidationResult):
        """Check for unusual sequence patterns."""
        # Multiple prolines in a row (rare in natural proteins)
        proline_runs = re.findall(r'P{3,}', sequence)
        if proline_runs:
            result.add_warning(f"Multiple consecutive prolines found: {proline_runs}")
        
        # High glycine content (might indicate linker regions)
        gly_content = sequence.count('G') / len(sequence)
        if gly_content > 0.15:  # More than 15% glycine
            result.add_warning(f"High glycine content: {gly_content*100:.1f}%")
        
        # Charged residue clustering
        charged_residues = 'DEKR'
        for i in range(len(sequence) - 5):
            window = sequence[i:i + 6]
            charged_count = sum(1 for aa in window if aa in charged_residues)
            if charged_count >= 5:  # 5 or more charged residues in 6-residue window
                result.add_warning(f"Highly charged region at position {i}-{i + 6}")
    
    def validate_batch(self, sequences: List[str]) -> List[ValidationResult]:
        """Validate a batch of sequences."""
        return [self.validate_sequence(seq) for seq in sequences]
    
    def filter_valid_sequences(self, sequences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Filter sequences into valid and invalid lists.
        
        Returns:
            Tuple of (valid_sequences, invalid_sequences)
        """
        valid_sequences = []
        invalid_sequences = []
        
        for seq in sequences:
            result = self.validate_sequence(seq)
            if result.is_valid:
                valid_sequences.append(seq)
            else:
                invalid_sequences.append(seq)
        
        return valid_sequences, invalid_sequences

class ModelValidator:
    """Validate model configurations and states."""
    
    @staticmethod
    def validate_model_config(config) -> ValidationResult:
        """Validate model configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check required attributes
        required_attrs = ['vocab_size', 'd_model', 'n_layers', 'n_heads']
        for attr in required_attrs:
            if not hasattr(config, attr):
                result.add_error(f"Missing required config attribute: {attr}")
        
        if not result.is_valid:
            return result
        
        # Validate dimensions
        if config.vocab_size <= 0:
            result.add_error(f"Invalid vocab_size: {config.vocab_size}")
        
        if config.d_model <= 0 or config.d_model % config.n_heads != 0:
            result.add_error(f"Invalid d_model: {config.d_model} (must be positive and divisible by n_heads)")
        
        if config.n_layers <= 0:
            result.add_error(f"Invalid n_layers: {config.n_layers}")
        
        if config.n_heads <= 0:
            result.add_error(f"Invalid n_heads: {config.n_heads}")
        
        # Check reasonable ranges
        if config.vocab_size > 100000:
            result.add_warning(f"Very large vocabulary size: {config.vocab_size}")
        
        if config.d_model > 4096:
            result.add_warning(f"Very large model dimension: {config.d_model}")
        
        if config.n_layers > 48:
            result.add_warning(f"Very deep model: {config.n_layers} layers")
        
        return result
    
    @staticmethod
    def validate_model_state(model) -> ValidationResult:
        """Validate model state and parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        try:
            # Check if model is in correct mode
            if model.training:
                result.add_warning("Model is in training mode, consider calling model.eval()")
            
            # Check for NaN parameters
            nan_params = []
            inf_params = []
            
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
                if torch.isinf(param).any():
                    inf_params.append(name)
            
            if nan_params:
                result.add_error(f"NaN parameters detected: {nan_params}")
            
            if inf_params:
                result.add_error(f"Infinite parameters detected: {inf_params}")
            
            # Check parameter magnitudes
            large_params = []
            for name, param in model.named_parameters():
                if param.abs().max() > 100:
                    large_params.append(name)
            
            if large_params:
                result.add_warning(f"Very large parameters detected: {large_params}")
        
        except Exception as e:
            result.add_error(f"Error during model validation: {str(e)}")
        
        return result

class InputValidator:
    """Validate inputs to various functions."""
    
    @staticmethod
    def validate_generation_params(
        num_samples: int,
        max_length: int,
        temperature: float,
        guidance_scale: float
    ) -> ValidationResult:
        """Validate generation parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if num_samples <= 0:
            result.add_error(f"num_samples must be positive, got {num_samples}")
        elif num_samples > 10000:
            result.add_warning(f"Very large num_samples: {num_samples}")
        
        if max_length <= 0:
            result.add_error(f"max_length must be positive, got {max_length}")
        elif max_length > 2048:
            result.add_warning(f"Very large max_length: {max_length}")
        
        if temperature <= 0:
            result.add_error(f"temperature must be positive, got {temperature}")
        elif temperature > 5.0:
            result.add_warning(f"Very high temperature: {temperature}")
        
        if guidance_scale < 1.0:
            result.add_error(f"guidance_scale must be >= 1.0, got {guidance_scale}")
        elif guidance_scale > 20.0:
            result.add_warning(f"Very high guidance_scale: {guidance_scale}")
        
        return result
    
    @staticmethod
    def validate_tensor_inputs(
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> ValidationResult:
        """Validate tensor inputs to model."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check input_ids
        if not isinstance(input_ids, torch.Tensor):
            result.add_error(f"input_ids must be tensor, got {type(input_ids)}")
        elif input_ids.dim() != 2:
            result.add_error(f"input_ids must be 2D tensor, got shape {input_ids.shape}")
        elif (input_ids < 0).any():
            result.add_error("input_ids contains negative values")
        
        # Check timesteps
        if not isinstance(timesteps, torch.Tensor):
            result.add_error(f"timesteps must be tensor, got {type(timesteps)}")
        elif timesteps.dim() != 1:
            result.add_error(f"timesteps must be 1D tensor, got shape {timesteps.shape}")
        elif (timesteps < 0).any():
            result.add_error("timesteps contains negative values")
        
        # Check dimension compatibility
        if result.is_valid and input_ids.shape[0] != timesteps.shape[0]:
            result.add_error(f"Batch size mismatch: input_ids {input_ids.shape[0]} vs timesteps {timesteps.shape[0]}")
        
        # Check condition if provided
        if condition is not None:
            if not isinstance(condition, torch.Tensor):
                result.add_error(f"condition must be tensor, got {type(condition)}")
            elif condition.dim() != 3:
                result.add_error(f"condition must be 3D tensor, got shape {condition.shape}")
            elif result.is_valid and condition.shape[0] != input_ids.shape[0]:
                result.add_error(f"Batch size mismatch: condition {condition.shape[0]} vs input_ids {input_ids.shape[0]}")
        
        return result

class SystemValidator:
    """Validate system requirements and environment."""
    
    @staticmethod
    def validate_environment() -> ValidationResult:
        """Validate the runtime environment."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check PyTorch installation
        try:
            import torch
            if not torch.__version__:
                result.add_error("PyTorch version not available")
        except ImportError:
            result.add_error("PyTorch not installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            result.add_suggestion(f"CUDA available with {gpu_count} GPU(s)")
            
            # Check GPU memory
            try:
                for i in range(gpu_count):
                    memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                    if memory_gb < 8:
                        result.add_warning(f"GPU {i} has limited memory: {memory_gb:.1f}GB")
            except Exception as e:
                result.add_warning(f"Could not check GPU memory: {e}")
        else:
            result.add_warning("CUDA not available, using CPU")
        
        # Check optional dependencies
        optional_deps = {
            'esm': 'ESM models for embeddings',
            'Bio': 'BioPython for structure analysis',
            'streamlit': 'Web interface',
            'plotly': 'Visualizations'
        }
        
        for dep, description in optional_deps.items():
            try:
                __import__(dep)
            except ImportError:
                result.add_warning(f"Optional dependency '{dep}' not available ({description})")
        
        return result
    
    @staticmethod
    def validate_paths(paths: Dict[str, str]) -> ValidationResult:
        """Validate file and directory paths."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        for name, path_str in paths.items():
            if not path_str:
                continue
                
            path = Path(path_str)
            
            if name.endswith('_dir'):
                # Directory path
                if not path.exists():
                    result.add_warning(f"Directory does not exist: {path}")
                elif not path.is_dir():
                    result.add_error(f"Path is not a directory: {path}")
            else:
                # File path
                if not path.exists():
                    result.add_warning(f"File does not exist: {path}")
                elif not path.is_file():
                    result.add_error(f"Path is not a file: {path}")
        
        return result

class ValidationManager:
    """Centralized validation management."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        self.level = level
        self.sequence_validator = SequenceValidator(level)
        self.model_validator = ModelValidator()
        self.input_validator = InputValidator()
        self.system_validator = SystemValidator()
    
    def comprehensive_validation(
        self,
        sequences: Optional[List[str]] = None,
        model_config: Optional[Any] = None,
        model: Optional[Any] = None,
        generation_params: Optional[Dict] = None
    ) -> ValidationResult:
        """Run comprehensive validation checks."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Validate sequences
        if sequences:
            logger.info(f"Validating {len(sequences)} sequences...")
            for i, seq in enumerate(sequences):
                seq_result = self.sequence_validator.validate_sequence(seq)
                if not seq_result.is_valid:
                    seq_result.errors = [f"Sequence {i}: {err}" for err in seq_result.errors]
                seq_result.warnings = [f"Sequence {i}: {warn}" for warn in seq_result.warnings]
                result.merge(seq_result)
        
        # Validate model configuration
        if model_config:
            logger.info("Validating model configuration...")
            config_result = self.model_validator.validate_model_config(model_config)
            result.merge(config_result)
        
        # Validate model state
        if model:
            logger.info("Validating model state...")
            model_result = self.model_validator.validate_model_state(model)
            result.merge(model_result)
        
        # Validate generation parameters
        if generation_params:
            logger.info("Validating generation parameters...")
            gen_result = self.input_validator.validate_generation_params(**generation_params)
            result.merge(gen_result)
        
        # Always validate environment
        logger.info("Validating system environment...")
        env_result = self.system_validator.validate_environment()
        result.merge(env_result)
        
        return result
    
    def validate_and_raise(self, *args, **kwargs) -> ValidationResult:
        """Run validation and raise exception if invalid."""
        result = self.comprehensive_validation(*args, **kwargs)
        
        if not result.is_valid:
            error_msg = "Validation failed:\n" + "\n".join(f"- {err}" for err in result.errors)
            raise ValidationError(error_msg)
        
        # Log warnings
        if result.warnings:
            warning_msg = "Validation warnings:\n" + "\n".join(f"- {warn}" for warn in result.warnings)
            logger.warning(warning_msg)
        
        return result