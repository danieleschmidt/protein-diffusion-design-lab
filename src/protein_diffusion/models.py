"""
Core diffusion models for protein generation.

This module implements the transformer-based diffusion model architecture
with attention mechanisms optimized for protein sequence modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class DiffusionTransformerConfig:
    """Configuration for the diffusion transformer model."""
    vocab_size: int = 50000
    d_model: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    d_ff: int = 4096
    max_position_embeddings: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    
    # Diffusion-specific parameters
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    
    # Protein-specific parameters
    use_rotary_embeddings: bool = True
    use_relative_attention: bool = True
    condition_dim: int = 512


@dataclass 
class DDPMConfig:
    """Configuration for DDPM sampling."""
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    clip_sample: bool = True
    prediction_type: str = "epsilon"


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for improved sequence modeling."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 512):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # Create rotation matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate cosine and sine embeddings."""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""
    cos = cos[:q.size(-2), :]
    sin = sin[:q.size(-2), :]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional rotary embeddings."""
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.num_heads
        
        assert self.head_dim * config.num_heads == config.d_model
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim, config.max_position_embeddings
            )
        else:
            self.rotary_emb = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(seq_len, hidden_states.device)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Handle past key values for caching
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        if use_cache:
            present_key_value = (key, value)
        else:
            present_key_value = None
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, present_key_value


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.activation == "gelu":
            self.activation = F.gelu
        elif config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization."""
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-layer norm attention
        normed_hidden_states = self.norm1(hidden_states)
        attn_output, present_key_value = self.attention(
            normed_hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Pre-layer norm feed forward
        normed_hidden_states = self.norm2(hidden_states)
        ff_output = self.feed_forward(normed_hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states, present_key_value


class TimestepEmbedding(nn.Module):
    """Embedding for diffusion timesteps."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.activation = F.silu
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Sinusoidal position encoding for timesteps
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Project through MLP
        emb = self.linear1(emb)
        emb = self.activation(emb)
        emb = self.linear2(emb)
        
        return emb


class DiffusionTransformer(nn.Module):
    """
    Transformer model for protein diffusion generation.
    
    This model implements a transformer architecture specifically designed
    for diffusion-based protein sequence generation with timestep conditioning.
    """
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional embeddings (if not using rotary)
        if not config.use_rotary_embeddings:
            self.position_embedding = nn.Embedding(
                config.max_position_embeddings, config.d_model
            )
        else:
            self.position_embedding = None
        
        # Timestep embedding
        self.timestep_embedding = TimestepEmbedding(config.d_model)
        
        # Condition projection (for motif conditioning)
        self.condition_proj = nn.Linear(config.condition_dim, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm and projection
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add positional embeddings (if not using rotary)
        if self.position_embedding is not None:
            position_ids = torch.arange(seq_len, device=device).expand(batch_size, -1)
            position_embeds = self.position_embedding(position_ids)
            hidden_states = hidden_states + position_embeds
        
        # Timestep embeddings
        timestep_embeds = self.timestep_embedding(timesteps)
        # Broadcast timestep embeddings across sequence
        timestep_embeds = timestep_embeds.unsqueeze(1).expand(-1, seq_len, -1)
        hidden_states = hidden_states + timestep_embeds
        
        # Add conditioning if provided
        if conditioning is not None:
            conditioning = self.condition_proj(conditioning)
            hidden_states = hidden_states + conditioning
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to attention bias (0 for valid tokens, -inf for padding)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Apply transformer blocks
        all_hidden_states = [] if use_cache else None
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            if use_cache:
                all_hidden_states.append(hidden_states)
            
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_key_value = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        if use_cache:
            outputs["past_key_values"] = present_key_values
        
        return outputs
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> 'DiffusionTransformer':
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', DiffusionTransformerConfig())
        
        model = cls(config)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model to checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, save_path)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for protein generation.
    
    This implements the DDPM training and sampling procedures
    with the transformer backbone for protein sequences.
    """
    
    def __init__(self, model: DiffusionTransformer, config: DDPMConfig):
        super().__init__()
        self.model = model
        self.config = config
        
        # Create noise schedule
        self.register_buffer("betas", self._create_beta_schedule())
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", 
                           F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Sampling coefficients
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", 
                           torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", 
                           torch.sqrt(1.0 / self.alphas_cumprod - 1))
        
        # DDIM sampling coefficients
        self.register_buffer("posterior_variance", 
                           self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
    
    def _create_beta_schedule(self) -> torch.Tensor:
        """Create noise schedule."""
        if self.config.beta_schedule == "linear":
            return torch.linspace(
                self.config.beta_start, 
                self.config.beta_end, 
                self.config.timesteps
            )
        elif self.config.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine noise schedule."""
        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        
        betas = []
        for i in range(self.config.timesteps):
            t1 = i / self.config.timesteps
            t2 = (i + 1) / self.config.timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        
        return torch.tensor(betas)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples according to the noise schedule."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps."""
        return torch.randint(
            0, self.config.timesteps, (batch_size,), device=device, dtype=torch.long
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        motif_conditioning: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Convert to continuous representation (embedding space)
        continuous_input = self.model.token_embedding(input_ids)
        
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(continuous_input)
        
        # Add noise according to schedule
        noisy_input = self.add_noise(continuous_input, noise, timesteps)
        
        # Predict noise
        model_output = self.model(
            input_ids=input_ids,  # Still pass discrete tokens for positional info
            timesteps=timesteps,
            attention_mask=attention_mask,
            conditioning=motif_conditioning,
        )
        
        # Convert logits back to embedding space for comparison
        pred_embeddings = torch.matmul(model_output["logits"], self.model.token_embedding.weight.T)
        
        loss = F.mse_loss(pred_embeddings, continuous_input if self.config.prediction_type == "sample" else noise)
        
        return {
            "loss": loss,
            "logits": model_output["logits"],
            "pred_embeddings": pred_embeddings,
        }
    
    def sample(
        self,
        shape: Tuple[int, int],
        device: torch.device,
        motif_conditioning: Optional[torch.Tensor] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """Sample using DDPM."""
        batch_size, seq_len = shape
        
        # Start with random noise in embedding space
        sample = torch.randn(batch_size, seq_len, self.model.config.d_model, device=device)
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(reversed(range(self.config.timesteps)), total=self.config.timesteps)
        else:
            timesteps = reversed(range(self.config.timesteps))
        
        for t in timesteps:
            timestep_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Convert continuous sample to discrete tokens for model input
            # Use argmax to get closest tokens
            logits = torch.matmul(sample, self.model.token_embedding.weight.T)
            input_ids = torch.argmax(logits, dim=-1)
            
            # Get model prediction
            with torch.no_grad():
                model_output = self.model(
                    input_ids=input_ids,
                    timesteps=timestep_batch,
                    conditioning=motif_conditioning,
                )
            
            # Convert logits to embedding space
            pred_embeddings = torch.matmul(model_output["logits"], self.model.token_embedding.weight.T)
            
            # Reverse diffusion step
            if t > 0:
                noise = torch.randn_like(sample)
            else:
                noise = torch.zeros_like(sample)
            
            # DDPM reverse step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]
            beta = self.betas[t]
            
            # Predict x_0
            pred_x0 = (sample - torch.sqrt(1 - alpha_cumprod) * pred_embeddings) / torch.sqrt(alpha_cumprod)
            
            if self.config.clip_sample:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Compute x_{t-1}
            if t > 0:
                variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                sigma = torch.sqrt(variance)
                sample = torch.sqrt(alpha_cumprod_prev) * pred_x0 + torch.sqrt(1 - alpha_cumprod_prev - variance) * pred_embeddings + sigma * noise
            else:
                sample = pred_x0
        
        # Convert final sample to logits
        final_logits = torch.matmul(sample, self.model.token_embedding.weight.T)
        
        return final_logits
    
    def ddim_sample(
        self,
        shape: Tuple[int, int],
        device: torch.device,
        ddim_steps: int = 50,
        eta: float = 0.0,
        motif_conditioning: Optional[torch.Tensor] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """Sample using DDIM (faster sampling)."""
        batch_size, seq_len = shape
        
        # Create DDIM timesteps
        step_ratio = self.config.timesteps // ddim_steps
        timesteps = (np.arange(0, ddim_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps = torch.from_numpy(timesteps).to(device)
        
        # Start with random noise
        sample = torch.randn(batch_size, seq_len, self.model.config.d_model, device=device)
        
        if progress:
            from tqdm import tqdm
            timestep_iterator = tqdm(timesteps, desc="DDIM Sampling")
        else:
            timestep_iterator = timesteps
        
        for i, timestep in enumerate(timestep_iterator):
            timestep_batch = timestep.repeat(batch_size)
            
            # Convert to discrete tokens
            logits = torch.matmul(sample, self.model.token_embedding.weight.T)
            input_ids = torch.argmax(logits, dim=-1)
            
            # Get model prediction
            with torch.no_grad():
                model_output = self.model(
                    input_ids=input_ids,
                    timesteps=timestep_batch,
                    conditioning=motif_conditioning,
                )
            
            pred_embeddings = torch.matmul(model_output["logits"], self.model.token_embedding.weight.T)
            
            # DDIM step
            alpha_cumprod = self.alphas_cumprod[timestep]
            alpha_cumprod_prev = self.alphas_cumprod_prev[timestep] if timestep > 0 else torch.tensor(1.0)
            
            # Predict x_0
            pred_x0 = (sample - torch.sqrt(1 - alpha_cumprod) * pred_embeddings) / torch.sqrt(alpha_cumprod)
            
            if self.config.clip_sample:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # DDIM step
            if i < len(timesteps) - 1:
                sigma = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * torch.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)
                noise = torch.randn_like(sample) if eta > 0 else torch.zeros_like(sample)
                sample = torch.sqrt(alpha_cumprod_prev) * pred_x0 + torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * pred_embeddings + sigma * noise
            else:
                sample = pred_x0
        
        # Convert final sample to logits
        final_logits = torch.matmul(sample, self.model.token_embedding.weight.T)
        
        return final_logits