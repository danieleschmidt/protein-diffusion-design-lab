"""
Unit tests for the diffusion models.
"""

import torch
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.protein_diffusion.models import (
    DiffusionTransformer,
    DDPM,
    DiffusionTransformerConfig,
    DDPMConfig,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    TimestepEmbedding,
    TransformerBlock
)


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return DiffusionTransformerConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        max_position_embeddings=128,
        timesteps=100,
    )


@pytest.fixture
def ddpm_config():
    """Create a test DDPM configuration."""
    return DDPMConfig(
        timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )


class TestDiffusionTransformerConfig:
    """Test DiffusionTransformerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DiffusionTransformerConfig()
        assert config.vocab_size == 50000
        assert config.d_model == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.dropout == 0.1
        assert config.use_rotary_embeddings == True


class TestRotaryPositionalEmbedding:
    """Test RotaryPositionalEmbedding."""
    
    def test_initialization(self):
        """Test RoPE initialization."""
        rope = RotaryPositionalEmbedding(dim=64, max_position_embeddings=128)
        assert rope.dim == 64
        assert rope.max_position_embeddings == 128
        assert hasattr(rope, 'inv_freq')
    
    def test_forward(self):
        """Test forward pass."""
        rope = RotaryPositionalEmbedding(dim=64)
        cos, sin = rope.forward(10, torch.device('cpu'))
        
        assert cos.shape == (10, 32)  # dim // 2
        assert sin.shape == (10, 32)
        assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-6)


class TestMultiHeadAttention:
    """Test MultiHeadAttention."""
    
    def test_initialization(self, model_config):
        """Test attention initialization."""
        attention = MultiHeadAttention(model_config)
        assert attention.num_heads == model_config.num_heads
        assert attention.d_model == model_config.d_model
        assert attention.head_dim == model_config.d_model // model_config.num_heads
    
    def test_forward(self, model_config):
        """Test attention forward pass."""
        attention = MultiHeadAttention(model_config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, model_config.d_model)
        
        output, _ = attention.forward(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_mask(self, model_config):
        """Test attention with mask."""
        attention = MultiHeadAttention(model_config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, model_config.d_model)
        attention_mask = torch.ones(batch_size, 1, 1, seq_len) * -1e9
        attention_mask[:, :, :, :5] = 0  # Mask last 5 tokens
        
        output, _ = attention.forward(hidden_states, attention_mask=attention_mask)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()


class TestTimestepEmbedding:
    """Test TimestepEmbedding."""
    
    def test_initialization(self):
        """Test timestep embedding initialization."""
        embedding = TimestepEmbedding(d_model=256)
        assert hasattr(embedding, 'linear1')
        assert hasattr(embedding, 'linear2')
    
    def test_forward(self):
        """Test timestep embedding forward pass."""
        embedding = TimestepEmbedding(d_model=256)
        timesteps = torch.tensor([0, 10, 50, 99])
        
        output = embedding.forward(timesteps)
        
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestTransformerBlock:
    """Test TransformerBlock."""
    
    def test_initialization(self, model_config):
        """Test transformer block initialization."""
        block = TransformerBlock(model_config)
        assert hasattr(block, 'attention')
        assert hasattr(block, 'feed_forward')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
    
    def test_forward(self, model_config):
        """Test transformer block forward pass."""
        block = TransformerBlock(model_config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, model_config.d_model)
        
        output, _ = block.forward(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestDiffusionTransformer:
    """Test DiffusionTransformer."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = DiffusionTransformer(model_config)
        
        assert model.config == model_config
        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'timestep_embedding')
        assert hasattr(model, 'blocks')
        assert len(model.blocks) == model_config.num_layers
    
    def test_parameter_count(self, model_config):
        """Test parameter counting."""
        model = DiffusionTransformer(model_config)
        param_count = model.get_num_parameters()
        
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_forward_pass(self, model_config):
        """Test full forward pass."""
        model = DiffusionTransformer(model_config)
        model.eval()
        
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, model_config.timesteps, (batch_size,))
        
        with torch.no_grad():
            output = model.forward(input_ids, timesteps)
        
        assert 'logits' in output
        assert 'hidden_states' in output
        assert output['logits'].shape == (batch_size, seq_len, model_config.vocab_size)
        assert not torch.isnan(output['logits']).any()
    
    def test_forward_with_conditioning(self, model_config):
        """Test forward pass with conditioning."""
        model = DiffusionTransformer(model_config)
        model.eval()
        
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, model_config.timesteps, (batch_size,))
        conditioning = torch.randn(batch_size, seq_len, model_config.condition_dim)
        
        with torch.no_grad():
            output = model.forward(input_ids, timesteps, conditioning=conditioning)
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, model_config.vocab_size)
    
    def test_save_load(self, model_config, tmp_path):
        """Test model save and load."""
        model = DiffusionTransformer(model_config)
        save_path = tmp_path / "model.pt"
        
        # Save model
        model.save_pretrained(str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = DiffusionTransformer.from_pretrained(str(save_path))
        assert loaded_model.config.vocab_size == model_config.vocab_size
        assert loaded_model.config.d_model == model_config.d_model


class TestDDPM:
    """Test DDPM."""
    
    def test_initialization(self, model_config, ddpm_config):
        """Test DDPM initialization."""
        model = DiffusionTransformer(model_config)
        ddpm = DDPM(model, ddpm_config)
        
        assert ddpm.model == model
        assert ddpm.config == ddpm_config
        assert hasattr(ddpm, 'betas')
        assert hasattr(ddpm, 'alphas')
        assert hasattr(ddpm, 'alphas_cumprod')
    
    def test_noise_schedule(self, model_config, ddpm_config):
        """Test noise schedule creation."""
        model = DiffusionTransformer(model_config)
        ddpm = DDPM(model, ddpm_config)
        
        assert ddpm.betas.shape == (ddpm_config.timesteps,)
        assert torch.all(ddpm.betas > 0)
        assert torch.all(ddpm.betas < 1)
        assert torch.all(ddpm.alphas > 0)
        assert torch.all(ddpm.alphas < 1)
    
    def test_add_noise(self, model_config, ddpm_config):
        """Test noise addition."""
        model = DiffusionTransformer(model_config)
        ddpm = DDPM(model, ddpm_config)
        
        batch_size, seq_len, d_model = 2, 10, model_config.d_model
        original_samples = torch.randn(batch_size, seq_len, d_model)
        noise = torch.randn_like(original_samples)
        timesteps = torch.randint(0, ddpm_config.timesteps, (batch_size,))
        
        noisy_samples = ddpm.add_noise(original_samples, noise, timesteps)
        
        assert noisy_samples.shape == original_samples.shape
        assert not torch.isnan(noisy_samples).any()
        assert not torch.isinf(noisy_samples).any()
    
    def test_sample_timesteps(self, model_config, ddpm_config):
        """Test timestep sampling."""
        model = DiffusionTransformer(model_config)
        ddpm = DDPM(model, ddpm_config)
        
        batch_size = 4
        timesteps = ddpm.sample_timesteps(batch_size, torch.device('cpu'))
        
        assert timesteps.shape == (batch_size,)
        assert torch.all(timesteps >= 0)
        assert torch.all(timesteps < ddpm_config.timesteps)
    
    @patch('torch.randn')
    def test_forward_training(self, mock_randn, model_config, ddpm_config):
        """Test DDPM forward pass for training."""
        model = DiffusionTransformer(model_config)
        ddpm = DDPM(model, ddpm_config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, ddpm_config.timesteps, (batch_size,))
        
        # Mock noise
        mock_noise = torch.randn(batch_size, seq_len, model_config.d_model)
        mock_randn.return_value = mock_noise
        
        output = ddpm.forward(input_ids, timesteps)
        
        assert 'loss' in output
        assert 'logits' in output
        assert 'pred_embeddings' in output
        assert output['loss'].requires_grad


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_end_to_end_training_step(self, model_config, ddpm_config):
        """Test a complete training step."""
        model = DiffusionTransformer(model_config)
        ddpm = DDPM(model, ddpm_config)
        
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
        timesteps = ddpm.sample_timesteps(batch_size, torch.device('cpu'))
        
        # Forward pass
        output = ddpm.forward(input_ids, timesteps)
        
        # Check outputs
        assert 'loss' in output
        assert not torch.isnan(output['loss'])
        assert output['loss'].requires_grad
        
        # Backward pass
        output['loss'].backward()
        
        # Check gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_different_sequence_lengths(self, model_config, ddpm_config):
        """Test model with different sequence lengths."""
        model = DiffusionTransformer(model_config)
        ddpm = DDPM(model, ddpm_config)
        model.eval()
        
        for seq_len in [5, 10, 32, 64]:
            if seq_len <= model_config.max_position_embeddings:
                batch_size = 2
                input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
                timesteps = torch.randint(0, model_config.timesteps, (batch_size,))
                
                with torch.no_grad():
                    output = model.forward(input_ids, timesteps)
                
                assert output['logits'].shape == (batch_size, seq_len, model_config.vocab_size)
    
    def test_gradient_flow(self, model_config, ddpm_config):
        """Test gradient flow through the model."""
        model = DiffusionTransformer(model_config)
        ddpm = DDPM(model, ddpm_config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
        timesteps = ddpm.sample_timesteps(batch_size, torch.device('cpu'))
        
        # Forward pass
        output = ddpm.forward(input_ids, timesteps)
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        params_with_grad = 0
        total_params = 0
        
        for param in model.parameters():
            total_params += 1
            if param.requires_grad and param.grad is not None:
                params_with_grad += 1
                # Check gradient is not zero everywhere
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
        
        # Most parameters should have gradients
        assert params_with_grad / total_params > 0.9  # At least 90% of parameters