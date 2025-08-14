"""
Novel Architecture Components for Protein Diffusion

This module implements cutting-edge architectural innovations:
- Hierarchical diffusion with multi-scale generation
- Cross-attention conditioning with structural constraints
- Memory-augmented transformers for protein knowledge
- Geometric-aware attention mechanisms
- Continuous latent diffusion for smooth protein space navigation
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    nn = torch.nn
    F = torch.F
    TORCH_AVAILABLE = False

import math
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from .models import DiffusionTransformerConfig, RotaryPositionalEmbedding


@dataclass
class HierarchicalDiffusionConfig:
    """Configuration for hierarchical multi-scale diffusion."""
    scales: List[int] = None  # [32, 64, 128, 256] - sequence lengths for each scale
    scale_weights: List[float] = None  # Weights for each scale in loss
    upsampling_method: str = "interpolation"  # "interpolation", "learned", "attention"
    downsampling_method: str = "pooling"  # "pooling", "strided_conv", "attention"
    cross_scale_attention: bool = True
    progressive_training: bool = True
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [32, 64, 128, 256]
        if self.scale_weights is None:
            self.scale_weights = [1.0] * len(self.scales)


@dataclass
class MemoryAugmentedConfig:
    """Configuration for memory-augmented transformers."""
    memory_size: int = 1000
    memory_dim: int = 512
    num_memory_heads: int = 8
    memory_update_rate: float = 0.01
    memory_gate_activation: str = "sigmoid"
    external_memory: bool = True
    episodic_memory: bool = True


@dataclass  
class GeometricAttentionConfig:
    """Configuration for geometric-aware attention."""
    use_distance_bias: bool = True
    use_angle_bias: bool = True
    max_distance: float = 20.0  # Angstroms
    distance_embedding_dim: int = 64
    angle_embedding_dim: int = 32
    geometric_dropout: float = 0.1


class HierarchicalDiffusionTransformer(nn.Module):
    """Hierarchical transformer for multi-scale protein generation."""
    
    def __init__(self, config: DiffusionTransformerConfig, hierarchical_config: HierarchicalDiffusionConfig):
        super().__init__()
        self.config = config
        self.hierarchical_config = hierarchical_config
        
        # Create transformers for each scale
        self.scale_transformers = nn.ModuleList()
        for scale in hierarchical_config.scales:
            scale_config = DiffusionTransformerConfig(
                **config.__dict__,
                max_position_embeddings=scale
            )
            transformer = DiffusionTransformerCore(scale_config)
            self.scale_transformers.append(transformer)
        
        # Cross-scale attention modules
        if hierarchical_config.cross_scale_attention:
            self.cross_scale_attention = nn.ModuleList()
            for i in range(len(hierarchical_config.scales) - 1):
                cross_attn = CrossScaleAttention(config.d_model, config.num_heads)
                self.cross_scale_attention.append(cross_attn)
        
        # Scale adapters
        self.upsampling_layers = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        
        for i in range(len(hierarchical_config.scales) - 1):
            curr_scale = hierarchical_config.scales[i]
            next_scale = hierarchical_config.scales[i + 1]
            
            # Upsampling from curr to next
            if hierarchical_config.upsampling_method == "learned":
                upsample = LearnedUpsampling(config.d_model, curr_scale, next_scale)
            elif hierarchical_config.upsampling_method == "attention":
                upsample = AttentionUpsampling(config.d_model, config.num_heads)
            else:
                upsample = InterpolationUpsampling()
            
            self.upsampling_layers.append(upsample)
            
            # Downsampling from next to curr
            if hierarchical_config.downsampling_method == "strided_conv":
                downsample = StridedDownsampling(config.d_model, next_scale, curr_scale)
            elif hierarchical_config.downsampling_method == "attention":
                downsample = AttentionDownsampling(config.d_model, config.num_heads)
            else:
                downsample = PoolingDownsampling()
            
            self.downsampling_layers.append(downsample)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        target_scale: int = -1,  # -1 for largest scale
        conditioning: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with hierarchical processing."""
        if target_scale == -1:
            target_scale = len(self.hierarchical_config.scales) - 1
        
        # Start with smallest scale
        current_ids = self._adapt_input_to_scale(input_ids, 0)
        scale_outputs = []
        
        for scale_idx in range(target_scale + 1):
            # Process at current scale
            transformer = self.scale_transformers[scale_idx]
            output = transformer(current_ids, timesteps, conditioning=conditioning)
            scale_outputs.append(output)
            
            # Upsample for next scale if not at target
            if scale_idx < target_scale:
                current_hidden = output["hidden_states"]
                upsampled = self.upsampling_layers[scale_idx](current_hidden)
                current_ids = self._hidden_to_ids(upsampled)
        
        # Combine outputs from all scales
        final_output = scale_outputs[-1]
        final_output["scale_outputs"] = scale_outputs
        
        return final_output
    
    def _adapt_input_to_scale(self, input_ids: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Adapt input to specific scale length."""
        target_length = self.hierarchical_config.scales[scale_idx]
        current_length = input_ids.size(-1)
        
        if current_length == target_length:
            return input_ids
        elif current_length > target_length:
            # Downsample
            indices = torch.linspace(0, current_length - 1, target_length, dtype=torch.long)
            return input_ids[:, indices]
        else:
            # Upsample via interpolation
            factor = target_length / current_length
            indices = (torch.arange(target_length) / factor).long()
            indices = torch.clamp(indices, 0, current_length - 1)
            return input_ids[:, indices]
    
    def _hidden_to_ids(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert hidden states back to token IDs."""
        # Project to vocab space and take argmax
        vocab_proj = nn.Linear(hidden_states.size(-1), self.config.vocab_size).to(hidden_states.device)
        logits = vocab_proj(hidden_states)
        return torch.argmax(logits, dim=-1)


class DiffusionTransformerCore(nn.Module):
    """Core transformer without external dependencies."""
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlockCore(config) for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids, timesteps, conditioning=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        hidden_states += self.position_embedding(position_ids)
        
        # Process through blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        # Final processing
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states
        }


class TransformerBlockCore(nn.Module):
    """Core transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
    
    def forward(self, x):
        # Self attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


class CrossScaleAttention(nn.Module):
    """Cross-attention between different scales."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale_adapter = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        query_states: torch.Tensor,  # From higher scale
        key_value_states: torch.Tensor,  # From lower scale
    ) -> torch.Tensor:
        batch_size, q_len, _ = query_states.shape
        kv_len = key_value_states.size(1)
        
        # Project query, key, value
        Q = self.q_proj(query_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value_states).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value_states).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        # Project output
        output = self.out_proj(attn_output)
        
        # Scale adaptation
        adapted_output = self.scale_adapter(output)
        
        return adapted_output


class LearnedUpsampling(nn.Module):
    """Learned upsampling between scales."""
    
    def __init__(self, d_model: int, source_len: int, target_len: int):
        super().__init__()
        self.source_len = source_len
        self.target_len = target_len
        
        # Learnable interpolation weights
        self.interpolation_weights = nn.Parameter(torch.randn(target_len, source_len))
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Apply learnable interpolation
        weights = F.softmax(self.interpolation_weights, dim=1)
        upsampled = torch.matmul(weights, x)  # [target_len, d_model]
        upsampled = upsampled.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Project
        upsampled = self.projection(upsampled)
        
        return upsampled


class AttentionUpsampling(nn.Module):
    """Attention-based upsampling."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        batch_size, source_len, d_model = x.shape
        
        # Create target positions
        target_positions = torch.linspace(0, 1, target_length, device=x.device)
        source_positions = torch.linspace(0, 1, source_len, device=x.device)
        
        # Position embeddings
        pos_embed = nn.Embedding(1000, d_model).to(x.device)
        target_pos_ids = (target_positions * 999).long()
        source_pos_ids = (source_positions * 999).long()
        
        target_pos_embeds = pos_embed(target_pos_ids).unsqueeze(0).expand(batch_size, -1, -1)
        source_pos_embeds = pos_embed(source_pos_ids).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add positional information
        query = target_pos_embeds
        key_value = x + source_pos_embeds
        
        # Attention
        upsampled, _ = self.attention(query, key_value, key_value)
        upsampled = self.norm(upsampled)
        
        return upsampled


class InterpolationUpsampling(nn.Module):
    """Simple interpolation upsampling."""
    
    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        # Linear interpolation
        return F.interpolate(x.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)


class PoolingDownsampling(nn.Module):
    """Pooling-based downsampling."""
    
    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = x.size(1)
        pool_size = current_length // target_length
        
        if pool_size <= 1:
            return x[:, :target_length]
        
        # Average pooling
        pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=pool_size, stride=pool_size)
        return pooled.transpose(1, 2)[:, :target_length]


class StridedDownsampling(nn.Module):
    """Strided convolution downsampling."""
    
    def __init__(self, d_model: int, source_len: int, target_len: int):
        super().__init__()
        stride = max(1, source_len // target_len)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply strided convolution
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x_conv = self.norm(x_conv)
        return x_conv


class AttentionDownsampling(nn.Module):
    """Attention-based downsampling."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        batch_size, source_len, d_model = x.shape
        
        # Learnable query tokens for downsampling
        query_tokens = nn.Parameter(torch.randn(target_length, d_model)).to(x.device)
        query = query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attention
        downsampled, _ = self.attention(query, x, x)
        downsampled = self.norm(downsampled)
        
        return downsampled


class MemoryAugmentedTransformer(nn.Module):
    """Memory-augmented transformer for protein knowledge."""
    
    def __init__(self, config: DiffusionTransformerConfig, memory_config: MemoryAugmentedConfig):
        super().__init__()
        self.config = config
        self.memory_config = memory_config
        
        # External memory
        if memory_config.external_memory:
            self.external_memory = nn.Parameter(
                torch.randn(memory_config.memory_size, memory_config.memory_dim)
            )
            self.memory_attention = nn.MultiheadAttention(
                config.d_model, memory_config.num_memory_heads, batch_first=True
            )
            self.memory_gate = nn.Linear(config.d_model * 2, config.d_model)
        
        # Episodic memory for dynamic updates
        if memory_config.episodic_memory:
            self.episodic_memory = []
            self.memory_update_gate = nn.Linear(config.d_model, 1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        update_memory: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden_states.shape
        
        # External memory attention
        if hasattr(self, 'external_memory'):
            memory = self.external_memory.unsqueeze(0).expand(batch_size, -1, -1)
            memory_output, memory_weights = self.memory_attention(
                hidden_states, memory, memory
            )
            
            # Gate memory information
            gate_input = torch.cat([hidden_states, memory_output], dim=-1)
            gate = torch.sigmoid(self.memory_gate(gate_input))
            
            hidden_states = hidden_states * gate + memory_output * (1 - gate)
        
        # Episodic memory update
        if hasattr(self, 'memory_update_gate') and update_memory:
            update_scores = torch.sigmoid(self.memory_update_gate(hidden_states))
            # Update episodic memory with high-scoring sequences
            # Implementation depends on specific memory strategy
        
        return hidden_states


class GeometricAwareAttention(nn.Module):
    """Geometric-aware attention with distance and angle biases."""
    
    def __init__(self, config: DiffusionTransformerConfig, geometric_config: GeometricAttentionConfig):
        super().__init__()
        self.config = config
        self.geometric_config = geometric_config
        
        self.attention = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
        
        if geometric_config.use_distance_bias:
            self.distance_embedding = nn.Embedding(100, geometric_config.distance_embedding_dim)
            self.distance_proj = nn.Linear(geometric_config.distance_embedding_dim, config.num_heads)
        
        if geometric_config.use_angle_bias:
            self.angle_embedding = nn.Embedding(180, geometric_config.angle_embedding_dim)
            self.angle_proj = nn.Linear(geometric_config.angle_embedding_dim, config.num_heads)
        
        self.dropout = nn.Dropout(geometric_config.geometric_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None,
        angle_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Standard attention
        attn_output, attn_weights = self.attention(
            hidden_states, hidden_states, hidden_states,
            need_weights=True, average_attn_weights=False
        )
        
        # Add geometric biases
        if distance_matrix is not None and hasattr(self, 'distance_embedding'):
            # Discretize distances
            distance_bins = torch.clamp(
                (distance_matrix / self.geometric_config.max_distance * 99).long(),
                0, 99
            )
            distance_embeds = self.distance_embedding(distance_bins)
            distance_bias = self.distance_proj(distance_embeds)
            distance_bias = distance_bias.permute(0, 3, 1, 2)  # [batch, heads, seq, seq]
            
            # Apply distance bias
            attn_weights = attn_weights + distance_bias
        
        if angle_matrix is not None and hasattr(self, 'angle_embedding'):
            # Discretize angles (0-180 degrees)
            angle_bins = torch.clamp((angle_matrix / math.pi * 179).long(), 0, 179)
            angle_embeds = self.angle_embedding(angle_bins)
            angle_bias = self.angle_proj(angle_embeds)
            angle_bias = angle_bias.permute(0, 3, 1, 2)
            
            # Apply angle bias
            attn_weights = attn_weights + angle_bias
        
        # Renormalize attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Recompute attention output with geometric biases
        value_states = hidden_states.view(batch_size, seq_len, self.config.num_heads, -1).transpose(1, 2)
        geometric_output = torch.matmul(attn_weights, value_states)
        geometric_output = geometric_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return geometric_output


class ContinuousLatentDiffusion(nn.Module):
    """Continuous latent space diffusion for smooth protein navigation."""
    
    def __init__(self, config: DiffusionTransformerConfig, latent_dim: int = 512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        
        # Encoder: sequence -> latent
        self.encoder = nn.Sequential(
            nn.Linear(config.vocab_size, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, latent_dim)
        )
        
        # Decoder: latent -> sequence
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size)
        )
        
        # Latent diffusion model
        self.latent_diffusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=8,
                dim_feedforward=latent_dim * 4,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Latent space regularization
        self.latent_norm = nn.LayerNorm(latent_dim)
        
    def encode(self, sequence_logits: torch.Tensor) -> torch.Tensor:
        """Encode sequence to continuous latent space."""
        # Convert logits to probabilities
        probs = F.softmax(sequence_logits, dim=-1)
        
        # Encode each position
        batch_size, seq_len, vocab_size = probs.shape
        probs_flat = probs.view(-1, vocab_size)
        latents_flat = self.encoder(probs_flat)
        latents = latents_flat.view(batch_size, seq_len, self.latent_dim)
        
        # Apply normalization
        latents = self.latent_norm(latents)
        
        return latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to sequence logits."""
        batch_size, seq_len, latent_dim = latents.shape
        
        # Decode each position
        latents_flat = latents.view(-1, latent_dim)
        logits_flat = self.decoder(latents_flat)
        logits = logits_flat.view(batch_size, seq_len, self.config.vocab_size)
        
        return logits
    
    def forward(
        self,
        sequence_logits: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through continuous latent diffusion."""
        # Encode to latent space
        clean_latents = self.encode(sequence_logits)
        
        # Add noise in latent space
        if noise is None:
            noise = torch.randn_like(clean_latents)
        
        # Noise scheduling in latent space
        alpha = 1.0 - timesteps.float() / 1000.0
        alpha = alpha.view(-1, 1, 1)
        
        noisy_latents = alpha * clean_latents + (1 - alpha) * noise
        
        # Diffusion in latent space
        diffused_latents = self.latent_diffusion(noisy_latents)
        
        # Decode back to sequence space
        predicted_logits = self.decode(diffused_latents)
        
        return {
            'predicted_logits': predicted_logits,
            'clean_latents': clean_latents,
            'noisy_latents': noisy_latents,
            'diffused_latents': diffused_latents
        }
    
    def interpolate(
        self,
        sequence_a: torch.Tensor,
        sequence_b: torch.Tensor,
        num_steps: int = 10
    ) -> List[torch.Tensor]:
        """Interpolate between two sequences in latent space."""
        # Encode both sequences
        latent_a = self.encode(sequence_a)
        latent_b = self.encode(sequence_b)
        
        # Interpolate in latent space
        interpolated_sequences = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            interpolated_latent = (1 - alpha) * latent_a + alpha * latent_b
            
            # Decode interpolated latent
            interpolated_logits = self.decode(interpolated_latent)
            interpolated_sequences.append(interpolated_logits)
        
        return interpolated_sequences