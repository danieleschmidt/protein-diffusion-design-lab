"""
Unit tests for validation module.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from src.protein_diffusion.validation import (
    SequenceValidator,
    ModelValidator,
    InputValidator,
    SystemValidator,
    ValidationManager,
    ValidationLevel,
    ValidationResult,
    ValidationError
)


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid == True
        assert result.errors == []
        assert result.warnings == []
        assert result.suggestions == []
    
    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        result.add_error("Test error")
        
        assert result.is_valid == False
        assert "Test error" in result.errors
    
    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        result.add_warning("Test warning")
        
        assert result.is_valid == True
        assert "Test warning" in result.warnings
    
    def test_merge(self):
        """Test merging results."""
        result1 = ValidationResult(is_valid=True, errors=[], warnings=["warn1"])
        result2 = ValidationResult(is_valid=False, errors=["error1"], warnings=["warn2"])
        
        result1.merge(result2)
        
        assert result1.is_valid == False
        assert "error1" in result1.errors
        assert "warn1" in result1.warnings
        assert "warn2" in result1.warnings


class TestSequenceValidator:
    """Test SequenceValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a sequence validator."""
        return SequenceValidator(ValidationLevel.MODERATE)
    
    def test_valid_sequence(self, validator):
        """Test validation of valid sequence."""
        result = validator.validate_sequence("MKLLFLVLVLVLVLLLQQQQQPPPPKKKGGG")
        assert result.is_valid == True
        assert len(result.errors) == 0
    
    def test_empty_sequence(self, validator):
        """Test validation of empty sequence."""
        result = validator.validate_sequence("")
        assert result.is_valid == False
        assert "Empty sequence" in result.errors[0]
    
    def test_non_string_sequence(self, validator):
        """Test validation of non-string input."""
        result = validator.validate_sequence(123)
        assert result.is_valid == False
        assert "must be string" in result.errors[0]
    
    def test_short_sequence(self, validator):
        """Test validation of short sequence."""
        result = validator.validate_sequence("MKL")
        assert len(result.warnings) > 0
        assert "short sequence" in result.warnings[0].lower()
    
    def test_invalid_amino_acids(self, validator):
        """Test validation with invalid amino acids."""
        result = validator.validate_sequence("MKLBZJ")
        assert "invalid amino acid codes" in result.warnings[0].lower() or "non-standard characters" in result.warnings[0].lower()
    
    def test_repetitive_sequence(self, validator):
        """Test detection of repetitive sequences."""
        result = validator.validate_sequence("MKLLLLLLLLLLLLAAAAAAAAAAAAKKKKKKKKKKK")
        warnings = [w.lower() for w in result.warnings]
        assert any("repetitive" in w or "low complexity" in w for w in warnings)
    
    def test_batch_validation(self, validator):
        """Test batch validation."""
        sequences = ["MKLLAVAAAA", "INVALID123", "MKLL"]
        results = validator.validate_batch(sequences)
        
        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)
    
    def test_filter_valid_sequences(self, validator):
        """Test filtering valid sequences."""
        sequences = ["MKLLLAVAAAA", "", "MKLL"]
        valid, invalid = validator.filter_valid_sequences(sequences)
        
        assert len(valid) >= 1  # At least one valid sequence
        assert len(invalid) >= 1  # At least one invalid sequence
        assert len(valid) + len(invalid) == len(sequences)


class TestModelValidator:
    """Test ModelValidator."""
    
    def test_validate_model_config(self):
        """Test model configuration validation."""
        # Mock config with required attributes
        config = Mock()
        config.vocab_size = 1000
        config.d_model = 512
        config.n_layers = 6
        config.n_heads = 8
        
        result = ModelValidator.validate_model_config(config)
        assert result.is_valid == True
    
    def test_validate_model_config_missing_attrs(self):
        """Test model config validation with missing attributes."""
        config = Mock(spec=[])  # Empty mock with no attributes
        
        result = ModelValidator.validate_model_config(config)
        assert result.is_valid == False
        assert len(result.errors) > 0
    
    def test_validate_model_config_invalid_values(self):
        """Test model config validation with invalid values."""
        config = Mock()
        config.vocab_size = -100  # Invalid
        config.d_model = 15  # Not divisible by n_heads
        config.n_layers = 0
        config.n_heads = 4
        
        result = ModelValidator.validate_model_config(config)
        assert result.is_valid == False
        assert len(result.errors) > 0
    
    def test_validate_model_state(self):
        """Test model state validation."""
        # Create a simple mock model
        model = Mock()
        model.training = False
        model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 10)),
            ("layer1.bias", torch.randn(10))
        ]
        
        result = ModelValidator.validate_model_state(model)
        assert result.is_valid == True
    
    def test_validate_model_state_with_nan(self):
        """Test model state validation with NaN parameters."""
        model = Mock()
        model.training = False
        
        # Create parameter with NaN
        param_with_nan = torch.randn(10, 10)
        param_with_nan[0, 0] = float('nan')
        
        model.named_parameters.return_value = [
            ("layer1.weight", param_with_nan)
        ]
        
        result = ModelValidator.validate_model_state(model)
        assert result.is_valid == False
        assert any("nan" in error.lower() for error in result.errors)


class TestInputValidator:
    """Test InputValidator."""
    
    def test_validate_generation_params_valid(self):
        """Test validation of valid generation parameters."""
        result = InputValidator.validate_generation_params(
            num_samples=10,
            max_length=256,
            temperature=1.0,
            guidance_scale=2.0
        )
        assert result.is_valid == True
    
    def test_validate_generation_params_invalid(self):
        """Test validation of invalid generation parameters."""
        result = InputValidator.validate_generation_params(
            num_samples=-5,  # Invalid
            max_length=0,    # Invalid
            temperature=-1.0,  # Invalid
            guidance_scale=0.5  # Invalid
        )
        assert result.is_valid == False
        assert len(result.errors) == 4  # All parameters are invalid
    
    def test_validate_tensor_inputs_valid(self):
        """Test validation of valid tensor inputs."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        timesteps = torch.randint(0, 100, (batch_size,))
        
        result = InputValidator.validate_tensor_inputs(input_ids, timesteps)
        assert result.is_valid == True
    
    def test_validate_tensor_inputs_invalid_types(self):
        """Test validation with invalid tensor types."""
        result = InputValidator.validate_tensor_inputs("not_a_tensor", "also_not_a_tensor")
        assert result.is_valid == False
        assert len(result.errors) >= 2
    
    def test_validate_tensor_inputs_wrong_dimensions(self):
        """Test validation with wrong tensor dimensions."""
        input_ids = torch.randint(0, 1000, (2, 10, 5))  # 3D instead of 2D
        timesteps = torch.randint(0, 100, (2, 3))       # 2D instead of 1D
        
        result = InputValidator.validate_tensor_inputs(input_ids, timesteps)
        assert result.is_valid == False
    
    def test_validate_tensor_inputs_batch_size_mismatch(self):
        """Test validation with batch size mismatch."""
        input_ids = torch.randint(0, 1000, (2, 10))
        timesteps = torch.randint(0, 100, (3,))  # Different batch size
        
        result = InputValidator.validate_tensor_inputs(input_ids, timesteps)
        assert result.is_valid == False
        assert any("batch size mismatch" in error.lower() for error in result.errors)


class TestSystemValidator:
    """Test SystemValidator."""
    
    def test_validate_environment(self):
        """Test environment validation."""
        result = SystemValidator.validate_environment()
        # Should at least not crash and return a result
        assert isinstance(result, ValidationResult)
    
    @patch('pathlib.Path.exists')
    def test_validate_paths_existing(self, mock_exists):
        """Test path validation with existing paths."""
        mock_exists.return_value = True
        
        with patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            
            paths = {
                "model_file": "/path/to/model.pt",
                "data_dir": "/path/to/data"
            }
            
            result = SystemValidator.validate_paths(paths)
            assert result.is_valid == True
    
    @patch('pathlib.Path.exists')
    def test_validate_paths_missing(self, mock_exists):
        """Test path validation with missing paths."""
        mock_exists.return_value = False
        
        paths = {
            "model_file": "/path/to/missing_model.pt",
            "data_dir": "/path/to/missing_data"
        }
        
        result = SystemValidator.validate_paths(paths)
        # Missing paths should generate warnings, not errors
        assert len(result.warnings) > 0


class TestValidationManager:
    """Test ValidationManager."""
    
    @pytest.fixture
    def validation_manager(self):
        """Create a validation manager."""
        return ValidationManager(ValidationLevel.MODERATE)
    
    def test_initialization(self, validation_manager):
        """Test ValidationManager initialization."""
        assert validation_manager.level == ValidationLevel.MODERATE
        assert isinstance(validation_manager.sequence_validator, SequenceValidator)
        assert isinstance(validation_manager.model_validator, ModelValidator)
        assert isinstance(validation_manager.input_validator, InputValidator)
        assert isinstance(validation_manager.system_validator, SystemValidator)
    
    def test_comprehensive_validation_sequences_only(self, validation_manager):
        """Test comprehensive validation with sequences only."""
        sequences = ["MKLLLAVAAAA", "MKLLLPPPPKKKGGG"]
        
        result = validation_manager.comprehensive_validation(sequences=sequences)
        assert isinstance(result, ValidationResult)
    
    def test_comprehensive_validation_generation_params(self, validation_manager):
        """Test comprehensive validation with generation parameters."""
        generation_params = {
            'num_samples': 10,
            'max_length': 256,
            'temperature': 1.0,
            'guidance_scale': 2.0
        }
        
        result = validation_manager.comprehensive_validation(generation_params=generation_params)
        assert isinstance(result, ValidationResult)
    
    def test_validate_and_raise_success(self, validation_manager):
        """Test validate_and_raise with valid input."""
        sequences = ["MKLLLAVAAAA"]
        
        # Should not raise an exception
        result = validation_manager.validate_and_raise(sequences=sequences)
        assert isinstance(result, ValidationResult)
    
    def test_validate_and_raise_failure(self, validation_manager):
        """Test validate_and_raise with invalid input."""
        sequences = [""]  # Empty sequence should fail
        
        with pytest.raises(ValidationError):
            validation_manager.validate_and_raise(sequences=sequences)


class TestValidationIntegration:
    """Integration tests for validation components."""
    
    def test_full_validation_pipeline(self):
        """Test the full validation pipeline."""
        manager = ValidationManager(ValidationLevel.STRICT)
        
        # Test data
        sequences = ["MKLLLAVAAAA", "MKLLLPPPPKKKGGG"]
        generation_params = {
            'num_samples': 5,
            'max_length': 128,
            'temperature': 1.0,
            'guidance_scale': 1.5
        }
        
        # Run comprehensive validation
        result = manager.comprehensive_validation(
            sequences=sequences,
            generation_params=generation_params
        )
        
        # Should complete without crashing
        assert isinstance(result, ValidationResult)
    
    def test_different_validation_levels(self):
        """Test different validation levels."""
        for level in ValidationLevel:
            manager = ValidationManager(level)
            
            # Test with a potentially problematic sequence
            sequences = ["MKL"]  # Very short sequence
            
            result = manager.comprehensive_validation(sequences=sequences)
            
            if level == ValidationLevel.STRICT:
                # Strict mode should reject very short sequences
                assert not result.is_valid
            else:
                # Other modes might allow it with warnings
                assert isinstance(result, ValidationResult)
    
    def test_error_propagation(self):
        """Test that errors propagate correctly through the system."""
        manager = ValidationManager(ValidationLevel.MODERATE)
        
        # Use clearly invalid inputs
        sequences = [None, "", 123]  # Mix of invalid types and values
        generation_params = {
            'num_samples': -1,
            'max_length': -1,
            'temperature': -1,
            'guidance_scale': 0
        }
        
        result = manager.comprehensive_validation(
            sequences=sequences,
            generation_params=generation_params
        )
        
        # Should capture multiple errors
        assert not result.is_valid
        assert len(result.errors) > 0