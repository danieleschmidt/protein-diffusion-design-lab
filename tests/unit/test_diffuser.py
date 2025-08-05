"""
Unit tests for the ProteinDiffuser class.

This module tests the core diffusion model functionality including
generation, configuration, and error handling.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.protein_diffusion.diffuser import ProteinDiffuser, ProteinDiffuserConfig
from src.protein_diffusion.models import DiffusionTransformerConfig, DDPMConfig
from src.protein_diffusion.tokenization import TokenizerConfig, EmbeddingConfig
from src.protein_diffusion.folding import StructurePredictorConfig


class TestProteinDiffuserConfig:
    """Test ProteinDiffuserConfig class."""
    
    def test_default_config(self):
        """Test default configuration initialization."""
        config = ProteinDiffuserConfig()
        
        assert config.num_samples == 10
        assert config.max_length == 256
        assert config.temperature == 1.0
        assert config.top_p == 0.9
        assert config.guidance_scale == 1.0
        assert config.dtype == torch.float16
        
        # Check that sub-configs are initialized
        assert isinstance(config.model_config, DiffusionTransformerConfig)
        assert isinstance(config.ddpm_config, DDPMConfig)
        assert isinstance(config.tokenizer_config, TokenizerConfig)
        assert isinstance(config.embedding_config, EmbeddingConfig)
        assert isinstance(config.structure_config, StructurePredictorConfig)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_model_config = DiffusionTransformerConfig(vocab_size=64, d_model=512)
        
        config = ProteinDiffuserConfig(
            model_config=custom_model_config,
            num_samples=50,
            temperature=0.8,
            device="cpu"
        )
        
        assert config.num_samples == 50
        assert config.temperature == 0.8
        assert config.device == "cpu"
        assert config.model_config.vocab_size == 64
        assert config.model_config.d_model == 512


class TestProteinDiffuser:
    """Test ProteinDiffuser class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = ProteinDiffuserConfig()
        config.device = "cpu"  # Use CPU for testing
        config.dtype = torch.float32  # Use float32 for stability
        return config
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock transformer model."""
        model = Mock()
        model.config = Mock()
        model.config.vocab_size = 32
        model.eval.return_value = model
        model.to.return_value = model
        
        # Mock forward method
        def mock_forward(input_ids, timesteps, condition=None, **kwargs):
            batch_size, seq_len = input_ids.shape
            vocab_size = 32
            return torch.randn(batch_size, seq_len, vocab_size)
        
        model.forward = mock_forward
        model.__call__ = mock_forward
        
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode.return_value = "MKLLILTCLVAVALARPKHPIP"
        tokenizer.__len__.return_value = 32
        return tokenizer
    
    @pytest.fixture
    def mock_ddpm(self):
        """Create a mock DDPM."""
        ddpm = Mock()
        ddpm.to.return_value = ddpm
        
        # Mock sample method
        def mock_sample(shape, device, **kwargs):
            batch_size, seq_len = shape
            vocab_size = 32
            return torch.randn(batch_size, seq_len, vocab_size)
        
        ddpm.sample = mock_sample
        ddpm.ddim_sample = mock_sample
        
        return ddmp
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        embeddings = Mock()
        embeddings.to.return_value = embeddings
        embeddings.eval.return_value = embeddings
        
        def mock_forward(sequences):
            batch_size = len(sequences)
            seq_len = 256
            d_model = 1024
            return torch.randn(batch_size, seq_len, d_model)
        
        embeddings.__call__ = mock_forward
        
        return embeddings
    
    @pytest.fixture
    def mock_structure_predictor(self):
        """Create mock structure predictor."""
        predictor = Mock()
        predictor.predict_structure.return_value = {
            "structure_quality": 0.8,
            "confidence": 0.9,
            "pdb_string": "HEADER    TEST\nATOM      1  N   ALA A   1      20.154  21.875  22.101  1.00 10.00           N"
        }
        predictor.evaluate_binding.return_value = {
            "binding_affinity": -8.5,
            "binding_method": "mock"
        }
        return predictor
    
    @patch('src.protein_diffusion.diffuser.DiffusionTransformer')
    @patch('src.protein_diffusion.diffuser.DDPM')
    @patch('src.protein_diffusion.diffuser.SELFIESTokenizer')
    @patch('src.protein_diffusion.diffuser.ProteinEmbeddings')
    @patch('src.protein_diffusion.diffuser.StructurePredictor')
    def test_initialization(self, mock_structure_pred, mock_embeddings, mock_tokenizer, 
                          mock_ddpm, mock_transformer, mock_config):
        """Test ProteinDiffuser initialization."""
        # Setup mocks
        mock_transformer.return_value = Mock()
        mock_ddpm.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_embeddings.return_value = Mock()
        mock_structure_pred.return_value = Mock()
        
        # Create diffuser
        diffuser = ProteinDiffuser(mock_config)
        
        # Verify initialization
        assert diffuser.config == mock_config
        assert diffuser.device.type == "cpu"
        
        # Verify components were created
        mock_transformer.assert_called_once()
        mock_ddpm.assert_called_once()
        mock_tokenizer.assert_called_once()
        mock_embeddings.assert_called_once()
        mock_structure_pred.assert_called_once()
    
    @patch('src.protein_diffusion.diffuser.DiffusionTransformer')
    @patch('src.protein_diffusion.diffuser.DDPM')
    @patch('src.protein_diffusion.diffuser.SELFIESTokenizer')
    @patch('src.protein_diffusion.diffuser.ProteinEmbeddings')
    @patch('src.protein_diffusion.diffuser.StructurePredictor')
    def test_generate_basic(self, mock_structure_pred, mock_embeddings, mock_tokenizer, 
                           mock_ddmp, mock_transformer, mock_config):
        """Test basic sequence generation."""
        # Setup mocks
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.vocab_size = 32
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_transformer.return_value = mock_model
        
        mock_ddpm_instance = Mock()
        mock_ddpm_instance.to.return_value = mock_ddpm_instance
        
        # Mock sample method to return logits
        def mock_sample(shape, device, **kwargs):
            batch_size, seq_len = shape
            vocab_size = 32
            return torch.randn(batch_size, seq_len, vocab_size)
        
        mock_ddpm_instance.sample = mock_sample
        mock_ddpm.return_value = mock_ddmp_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.decode.return_value = "MKLLILTCLVAVALARPKHPIP"
        mock_tokenizer_instance.__len__.return_value = 32
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.to.return_value = mock_embeddings_instance
        mock_embeddings_instance.eval.return_value = mock_embeddings_instance
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_structure_pred.return_value = Mock()
        
        # Create diffuser and generate
        diffuser = ProteinDiffuser(mock_config)
        results = diffuser.generate(num_samples=2, max_length=50)
        
        # Verify results
        assert len(results) == 2
        for result in results:
            assert "sequence" in result
            assert "confidence" in result
            assert "length" in result
            assert "sample_id" in result
            assert result["sequence"] == "MKLLILTCLVAVALARPKHPIP"
            assert isinstance(result["confidence"], float)
            assert isinstance(result["length"], int)
    
    def test_motif_encoding(self, mock_config):
        """Test motif encoding functionality."""
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=Mock()),
            DDPM=Mock(return_value=Mock()),
            SELFIESTokenizer=Mock(return_value=Mock()),
            ProteinEmbeddings=Mock(return_value=Mock()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            diffuser = ProteinDiffuser(mock_config)
            
            # Mock the tokenizer and embeddings for motif encoding
            diffuser.tokenizer.encode.return_value = torch.tensor([1, 2, 3, 4, 5])
            diffuser.embeddings.return_value = torch.randn(1, 5, 1024)
            
            # Test motif encoding
            motif_conditioning = diffuser._encode_motif("HELIX_SHEET_HELIX", 2, 50)
            
            assert motif_conditioning.shape[0] == 2  # batch_size
            assert motif_conditioning.shape[2] == 1024  # d_model
    
    def test_motif_to_sequence(self, mock_config):
        """Test motif to sequence conversion."""
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=Mock()),
            DDPM=Mock(return_value=Mock()),
            SELFIESTokenizer=Mock(return_value=Mock()),
            ProteinEmbeddings=Mock(return_value=Mock()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            diffuser = ProteinDiffuser(mock_config)
            
            # Test known motifs
            assert diffuser._motif_to_sequence("HELIX") == "AEAAAKEAAAKA"
            assert diffuser._motif_to_sequence("SHEET") == "IVIVIV"
            assert diffuser._motif_to_sequence("LOOP") == "GGSGGS"
            assert diffuser._motif_to_sequence("UNKNOWN") == "GGGG"
    
    def test_sequence_properties(self, mock_config):
        """Test sequence property calculation."""
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=Mock()),
            DDPM=Mock(return_value=Mock()),
            SELFIESTokenizer=Mock(return_value=Mock()),
            ProteinEmbeddings=Mock(return_value=Mock()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            diffuser = ProteinDiffuser(mock_config)
            
            # Test with known sequence
            sequence = "MKLLILTCLVAVALARPKHPIP"
            properties = diffuser._compute_sequence_properties(sequence)
            
            # Check that all expected properties are present
            expected_props = [
                "hydrophobicity", "net_charge", "charge_density",
                "glycine_content", "proline_content", "aromatic_content"
            ]
            
            for prop in expected_props:
                assert prop in properties
                assert isinstance(properties[prop], (int, float))
            
            # Check reasonable ranges
            assert -5 <= properties["hydrophobicity"] <= 5
            assert properties["glycine_content"] >= 0
            assert properties["proline_content"] >= 0
            assert properties["aromatic_content"] >= 0
    
    def test_model_info(self, mock_config):
        """Test model info retrieval."""
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=Mock()),
            DDPM=Mock(return_value=Mock()),
            SELFIESTokenizer=Mock(return_value=Mock()),
            ProteinEmbeddings=Mock(return_value=Mock()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            # Setup model mock
            mock_model = Mock()
            mock_model.get_num_parameters.return_value = 1000000
            
            diffuser = ProteinDiffuser(mock_config)
            diffuser.model = mock_model
            diffuser.tokenizer.__len__.return_value = 32
            
            info = diffuser.get_model_info()
            
            assert "model_type" in info
            assert "parameters" in info
            assert "vocab_size" in info
            assert "max_length" in info
            assert "device" in info
            assert "dtype" in info
            
            assert info["model_type"] == "DiffusionTransformer"
            assert info["parameters"] == 1000000
            assert info["vocab_size"] == 32


class TestProteinDiffuserIntegration:
    """Integration tests for ProteinDiffuser."""
    
    def test_save_and_load_pretrained(self):
        """Test saving and loading pretrained models."""
        config = ProteinDiffuserConfig()
        config.device = "cpu"
        config.dtype = torch.float32
        
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=Mock()),
            DDPM=Mock(return_value=Mock()),
            SELFIESTokenizer=Mock(return_value=Mock()),
            ProteinEmbeddings=Mock(return_value=Mock()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            # Mock the model's save method
            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            
            # Mock the tokenizer's save method
            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()
            
            diffuser = ProteinDiffuser(config)
            diffuser.model = mock_model
            diffuser.tokenizer = mock_tokenizer
            
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = temp_dir
                
                # Test saving
                diffuser.save_pretrained(save_path)
                
                # Verify save methods were called
                mock_model.save_pretrained.assert_called_once()
                mock_tokenizer.save_pretrained.assert_called_once()
                
                # Verify config file was created
                config_file = Path(save_path) / "config.json"
                assert config_file.exists()
                
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                assert saved_config["model_name"] == "ProteinDiffuser"
    
    def test_error_handling(self, mock_config):
        """Test error handling in various scenarios."""
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=Mock()),
            DDPM=Mock(return_value=Mock()),
            SELFIESTokenizer=Mock(return_value=Mock()),
            ProteinEmbeddings=Mock(return_value=Mock()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            diffuser = ProteinDiffuser(mock_config)
            
            # Test invalid sampling method
            with pytest.raises(ValueError, match="Unknown sampling method"):
                diffuser.generate(sampling_method="invalid_method")
    
    def test_generation_with_different_parameters(self, mock_config):
        """Test generation with various parameter combinations."""
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=Mock()),
            DDPM=Mock(return_value=Mock()),
            SELFIESTokenizer=Mock(return_value=Mock()),
            ProteinEmbeddings=Mock(return_value=Mock()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            diffuser = ProteinDiffuser(mock_config)
            
            # Mock DDPM methods
            def mock_sample_method(shape, device, **kwargs):
                batch_size, seq_len = shape
                vocab_size = 32
                return torch.randn(batch_size, seq_len, vocab_size)
            
            diffuser.ddpm.sample = mock_sample_method
            diffuser.ddpm.ddim_sample = mock_sample_method
            diffuser.tokenizer.decode.return_value = "TESTSEQUENCE"
            
            # Test DDPM sampling
            results_ddpm = diffuser.generate(sampling_method="ddpm", num_samples=1)
            assert len(results_ddpm) == 1
            
            # Test DDIM sampling
            results_ddim = diffuser.generate(sampling_method="ddim", num_samples=1, ddim_steps=25)
            assert len(results_ddim) == 1
            
            # Test with motif
            results_motif = diffuser.generate(motif="HELIX", num_samples=1)
            assert len(results_motif) == 1
            assert results_motif[0]["motif"] == "HELIX"


if __name__ == "__main__":
    pytest.main([__file__])