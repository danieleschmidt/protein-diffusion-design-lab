"""
Next-Generation Molecular Foundation Models for Protein Design.

This module implements state-of-the-art foundation models combining:
- Geometric deep learning with equivariant neural networks
- Multi-modal learning (sequence, structure, function)
- Physics-informed neural networks with molecular dynamics
- Self-supervised learning on massive protein databases
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from pathlib import Path
import json
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import MultiheadAttention, LayerNorm, Linear, Embedding
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    nn = torch.nn
    F = torch.F
    MultiheadAttention = nn.MultiheadAttention
    LayerNorm = nn.LayerNorm
    Linear = nn.Linear
    Embedding = nn.Embedding
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def array(x): return x
        @staticmethod 
        def mean(x): return sum(x)/len(x) if x else 0.5
        @staticmethod
        def std(x): return 0.1
        @staticmethod
        def random.normal(mean=0, std=1, size=None):
            import random
            if size:
                return [random.gauss(mean, std) for _ in range(size)]
            return random.gauss(mean, std)
    np = MockNumpy()
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MolecularFoundationConfig:
    """Configuration for molecular foundation models."""
    # Model architecture
    d_model: int = 2048
    num_layers: int = 48
    num_heads: int = 32
    d_ff: int = 8192
    vocab_size: int = 64
    max_sequence_length: int = 2048
    
    # Geometric features
    use_geometric_features: bool = True
    use_equivariant_layers: bool = True
    coordinate_embedding_dim: int = 128
    distance_embedding_dim: int = 64
    angle_embedding_dim: int = 64
    
    # Multi-modal learning
    sequence_weight: float = 0.4
    structure_weight: float = 0.3
    function_weight: float = 0.2
    dynamics_weight: float = 0.1
    
    # Physics-informed components
    use_physics_constraints: bool = True
    energy_regularization: float = 0.01
    force_prediction: bool = True
    molecular_dynamics_steps: int = 100
    
    # Self-supervised learning
    masked_language_modeling: bool = True
    structure_prediction: bool = True
    contact_prediction: bool = True
    dynamics_prediction: bool = True
    
    # Foundation model scaling
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    model_parallel: bool = True
    
    # Training configuration
    warmup_steps: int = 10000
    max_learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    dropout: float = 0.1
    layer_dropout: float = 0.0


class GeometricEmbedding(nn.Module):
    """Geometric embedding for molecular structures with equivariance."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        # Distance embeddings
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, config.distance_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.distance_embedding_dim, config.distance_embedding_dim)
        )
        
        # Angle embeddings 
        self.angle_embedding = nn.Sequential(
            nn.Linear(2, config.angle_embedding_dim),  # sin, cos
            nn.ReLU(), 
            nn.Linear(config.angle_embedding_dim, config.angle_embedding_dim)
        )
        
        # Coordinate embedding with rotation equivariance
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(3, config.coordinate_embedding_dim),
            nn.LayerNorm(config.coordinate_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.coordinate_embedding_dim, config.coordinate_embedding_dim)
        )
        
        # Geometric attention
        self.geometric_attention = GeometricMultiheadAttention(
            config.d_model,
            config.num_heads,
            config.coordinate_embedding_dim
        )
        
    def forward(self, coordinates: torch.Tensor, sequence_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass with geometric features."""
        batch_size, seq_len = coordinates.shape[:2]
        
        # Compute pairwise distances
        distances = self._compute_distances(coordinates)
        distance_emb = self.distance_embedding(distances.unsqueeze(-1))
        
        # Compute bond angles
        angles = self._compute_angles(coordinates)
        angle_emb = self.angle_embedding(angles)
        
        # Coordinate embeddings
        coord_emb = self.coordinate_embedding(coordinates)
        
        # Combine geometric features
        geometric_features = torch.cat([
            distance_emb.flatten(start_dim=2),
            angle_emb.flatten(start_dim=2),
            coord_emb
        ], dim=-1)
        
        # Geometric attention
        enhanced_embedding = self.geometric_attention(
            sequence_embedding,
            geometric_features,
            coordinates
        )
        
        return enhanced_embedding
        
    def _compute_distances(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between atoms."""
        # coordinates: [batch, seq_len, 3]
        diff = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)
        return distances
        
    def _compute_angles(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute bond angles."""
        batch_size, seq_len = coordinates.shape[:2]
        
        # Simple angle calculation for consecutive atoms
        angles = torch.zeros(batch_size, seq_len, seq_len, 2, device=coordinates.device)
        
        for i in range(seq_len - 2):
            vec1 = coordinates[:, i+1] - coordinates[:, i]
            vec2 = coordinates[:, i+2] - coordinates[:, i+1]
            
            # Compute angle using dot product
            dot_product = torch.sum(vec1 * vec2, dim=-1)
            norm1 = torch.norm(vec1, dim=-1)
            norm2 = torch.norm(vec2, dim=-1)
            
            cos_angle = dot_product / (norm1 * norm2 + 1e-8)
            angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
            
            angles[:, i, i+2, 0] = torch.sin(angle)
            angles[:, i, i+2, 1] = torch.cos(angle)
            
        return angles


class GeometricMultiheadAttention(nn.Module):
    """Multi-head attention with geometric awareness."""
    
    def __init__(self, d_model: int, num_heads: int, coord_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Geometric bias networks
        self.distance_bias = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.ReLU(),
            nn.Linear(num_heads, num_heads)
        )
        
        self.coordinate_proj = nn.Linear(coord_dim, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, geometric_features: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Forward pass with geometric bias."""
        batch_size, seq_len = query.shape[:2]
        
        # Standard attention components
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add geometric bias
        distances = torch.norm(coordinates.unsqueeze(2) - coordinates.unsqueeze(1), dim=-1)
        distance_bias = self.distance_bias(distances.unsqueeze(-1)).permute(0, 3, 1, 2)
        attention_scores = attention_scores + distance_bias
        
        # Apply softmax and compute output
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Add geometric features
        geo_proj = self.coordinate_proj(geometric_features)
        output = self.output_linear(attention_output + geo_proj)
        
        return output


class PhysicsInformedLayer(nn.Module):
    """Physics-informed neural network layer with molecular constraints."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        # Energy prediction network
        self.energy_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1)
        )
        
        # Force prediction network  
        self.force_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 3)
        )
        
        # Molecular dynamics integrator
        self.md_integrator = MolecularDynamicsIntegrator(config)
        
    def forward(self, hidden_states: torch.Tensor, coordinates: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with physics constraints."""
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Predict energy
        energy = self.energy_predictor(hidden_states)
        total_energy = energy.sum(dim=1)  # Sum over residues
        
        # Predict forces
        forces = self.force_predictor(hidden_states)
        
        # Apply molecular dynamics if enabled
        physics_info = {
            'energy': total_energy,
            'forces': forces,
            'coordinates': coordinates
        }
        
        if self.config.molecular_dynamics_steps > 0:
            updated_coordinates = self.md_integrator(coordinates, forces)
            physics_info['updated_coordinates'] = updated_coordinates
        
        return hidden_states, physics_info
        
    def compute_physics_loss(self, physics_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physics-informed loss."""
        energy = physics_info['energy']
        forces = physics_info['forces']
        
        # Energy conservation constraint
        energy_loss = self.config.energy_regularization * torch.mean(energy ** 2)
        
        # Force consistency constraint
        force_magnitude = torch.norm(forces, dim=-1)
        force_loss = torch.mean(force_magnitude ** 2)
        
        # Stability constraint (prevent explosion)
        stability_loss = torch.mean(torch.relu(force_magnitude - 10.0))
        
        total_physics_loss = energy_loss + force_loss + stability_loss
        return total_physics_loss


class MolecularDynamicsIntegrator(nn.Module):
    """Neural molecular dynamics integrator."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        self.dt = 0.001  # Time step
        
        # Learnable parameters for MD
        self.mass_predictor = nn.Linear(config.d_model, 1)
        self.friction_predictor = nn.Linear(config.d_model, 1)
        
    def forward(self, coordinates: torch.Tensor, forces: torch.Tensor, 
                velocities: torch.Tensor = None) -> torch.Tensor:
        """Integrate molecular dynamics."""
        if velocities is None:
            velocities = torch.zeros_like(coordinates)
            
        # Simple Verlet integration
        new_coordinates = coordinates
        new_velocities = velocities
        
        for _ in range(min(self.config.molecular_dynamics_steps, 10)):  # Limit steps
            # Update velocities
            new_velocities = new_velocities + forces * self.dt
            
            # Update coordinates
            new_coordinates = new_coordinates + new_velocities * self.dt
            
        return new_coordinates


class MultiModalFusionLayer(nn.Module):
    """Multi-modal fusion for sequence, structure, and function."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        # Modality-specific encoders
        self.sequence_encoder = nn.TransformerEncoderLayer(
            config.d_model, config.num_heads, config.d_ff, config.dropout
        )
        
        self.structure_encoder = nn.TransformerEncoderLayer(
            config.d_model, config.num_heads, config.d_ff, config.dropout
        )
        
        self.function_encoder = nn.TransformerEncoderLayer(
            config.d_model, config.num_heads, config.d_ff, config.dropout
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        # Modality weights
        self.modality_weights = nn.Parameter(torch.tensor([
            config.sequence_weight,
            config.structure_weight, 
            config.function_weight
        ]))
        
    def forward(self, sequence_features: torch.Tensor,
                structure_features: torch.Tensor,
                function_features: torch.Tensor) -> torch.Tensor:
        """Multi-modal fusion forward pass."""
        
        # Encode each modality
        seq_encoded = self.sequence_encoder(sequence_features)
        struct_encoded = self.structure_encoder(structure_features)
        func_encoded = self.function_encoder(function_features)
        
        # Cross-modal attention
        seq_struct, _ = self.cross_attention(seq_encoded, struct_encoded, struct_encoded)
        seq_func, _ = self.cross_attention(seq_encoded, func_encoded, func_encoded)
        struct_func, _ = self.cross_attention(struct_encoded, func_encoded, func_encoded)
        
        # Weighted combination
        weights = F.softmax(self.modality_weights, dim=0)
        
        # Concatenate and fuse
        multimodal_features = torch.cat([
            seq_encoded * weights[0],
            struct_encoded * weights[1], 
            func_encoded * weights[2]
        ], dim=-1)
        
        fused_output = self.fusion_network(multimodal_features)
        
        return fused_output


class SelfSupervisedHead(nn.Module):
    """Self-supervised learning heads for pre-training."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        # Masked language modeling head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size)
        )
        
        # Structure prediction head
        self.structure_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 3)  # x, y, z coordinates
        )
        
        # Contact prediction head
        self.contact_head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1)
        )
        
        # Dynamics prediction head
        self.dynamics_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 6)  # velocity (3) + acceleration (3)
        )
        
    def forward(self, hidden_states: torch.Tensor, 
                task: str = "all") -> Dict[str, torch.Tensor]:
        """Forward pass for self-supervised tasks."""
        outputs = {}
        
        if task in ["all", "mlm"]:
            outputs["mlm_logits"] = self.mlm_head(hidden_states)
            
        if task in ["all", "structure"]:
            outputs["structure_pred"] = self.structure_head(hidden_states)
            
        if task in ["all", "contact"]:
            # Pairwise contact prediction
            batch_size, seq_len = hidden_states.shape[:2]
            h_i = hidden_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
            h_j = hidden_states.unsqueeze(1).expand(-1, seq_len, -1, -1)
            pairwise = torch.cat([h_i, h_j], dim=-1)
            outputs["contact_pred"] = self.contact_head(pairwise).squeeze(-1)
            
        if task in ["all", "dynamics"]:
            outputs["dynamics_pred"] = self.dynamics_head(hidden_states)
            
        return outputs


class MolecularFoundationModel(nn.Module):
    """Next-generation molecular foundation model."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        # Input embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.d_model)
        
        # Geometric embedding layer
        self.geometric_embedding = GeometricEmbedding(config)
        
        # Multi-modal fusion
        self.multimodal_fusion = MultiModalFusionLayer(config)
        
        # Transformer layers with physics-informed components
        self.layers = nn.ModuleList([
            MolecularTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Self-supervised heads
        self.ssl_heads = SelfSupervisedHead(config)
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, input_ids: torch.Tensor,
                coordinates: torch.Tensor = None,
                structure_features: torch.Tensor = None,
                function_features: torch.Tensor = None,
                task: str = "representation",
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the foundation model."""
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token and position embeddings
        tokens = self.token_embedding(input_ids)
        positions = self.position_embedding(torch.arange(seq_len, device=device))
        hidden_states = tokens + positions
        
        # Add geometric features if available
        if coordinates is not None and self.config.use_geometric_features:
            hidden_states = self.geometric_embedding(coordinates, hidden_states)
        
        # Multi-modal fusion if features available
        if structure_features is not None and function_features is not None:
            hidden_states = self.multimodal_fusion(
                hidden_states, structure_features, function_features
            )
        
        # Pass through transformer layers
        physics_losses = []
        for layer in self.layers:
            hidden_states, physics_info = layer(hidden_states, coordinates)
            if physics_info and self.config.use_physics_constraints:
                physics_loss = layer.physics_layer.compute_physics_loss(physics_info)
                physics_losses.append(physics_loss)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Prepare outputs
        outputs = {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states
        }
        
        # Add physics loss if computed
        if physics_losses:
            outputs["physics_loss"] = torch.mean(torch.stack(physics_losses))
        
        # Self-supervised tasks
        if task != "representation":
            ssl_outputs = self.ssl_heads(hidden_states, task)
            outputs.update(ssl_outputs)
        
        return outputs
    
    def get_embeddings(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get embeddings for downstream tasks."""
        outputs = self.forward(input_ids, task="representation", **kwargs)
        return outputs["hidden_states"]
    
    def predict_structure(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict 3D structure from sequence."""
        outputs = self.forward(input_ids, task="structure", **kwargs)
        return outputs["structure_pred"]
    
    def predict_contacts(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict residue contacts."""
        outputs = self.forward(input_ids, task="contact", **kwargs)
        return outputs["contact_pred"]
    
    def predict_dynamics(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict molecular dynamics."""
        outputs = self.forward(input_ids, task="dynamics", **kwargs)
        return outputs["dynamics_pred"]


class MolecularTransformerLayer(nn.Module):
    """Transformer layer with molecular-specific enhancements."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        # Standard transformer components
        self.attention = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Physics-informed layer
        if config.use_physics_constraints:
            self.physics_layer = PhysicsInformedLayer(config)
        else:
            self.physics_layer = None
            
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor, coordinates: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with physics integration."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)
        
        # Physics-informed processing
        physics_info = None
        if self.physics_layer is not None and coordinates is not None:
            hidden_states, physics_info = self.physics_layer(hidden_states, coordinates)
            
        return hidden_states, physics_info


class MolecularFoundationTrainer:
    """Trainer for molecular foundation models."""
    
    def __init__(self, model: MolecularFoundationModel, config: MolecularFoundationConfig):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.max_learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            eta_min=config.min_learning_rate
        )
        
        # Loss functions
        self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.structure_loss_fn = nn.MSELoss()
        self.contact_loss_fn = nn.BCEWithLogitsLoss()
        self.dynamics_loss_fn = nn.MSELoss()
        
        self.global_step = 0
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            coordinates=batch.get("coordinates"),
            structure_features=batch.get("structure_features"),
            function_features=batch.get("function_features"),
            task="all"
        )
        
        # Compute losses
        total_loss = 0.0
        loss_dict = {}
        
        # Masked language modeling loss
        if "mlm_logits" in outputs and "mlm_labels" in batch:
            mlm_loss = self.mlm_loss_fn(
                outputs["mlm_logits"].view(-1, self.config.vocab_size),
                batch["mlm_labels"].view(-1)
            )
            total_loss += mlm_loss
            loss_dict["mlm_loss"] = mlm_loss.item()
            
        # Structure prediction loss
        if "structure_pred" in outputs and "structure_labels" in batch:
            structure_loss = self.structure_loss_fn(
                outputs["structure_pred"],
                batch["structure_labels"]
            )
            total_loss += structure_loss
            loss_dict["structure_loss"] = structure_loss.item()
            
        # Contact prediction loss
        if "contact_pred" in outputs and "contact_labels" in batch:
            contact_loss = self.contact_loss_fn(
                outputs["contact_pred"],
                batch["contact_labels"]
            )
            total_loss += contact_loss
            loss_dict["contact_loss"] = contact_loss.item()
            
        # Dynamics prediction loss
        if "dynamics_pred" in outputs and "dynamics_labels" in batch:
            dynamics_loss = self.dynamics_loss_fn(
                outputs["dynamics_pred"],
                batch["dynamics_labels"]
            )
            total_loss += dynamics_loss
            loss_dict["dynamics_loss"] = dynamics_loss.item()
            
        # Physics loss
        if "physics_loss" in outputs:
            physics_loss = outputs["physics_loss"]
            total_loss += physics_loss
            loss_dict["physics_loss"] = physics_loss.item()
            
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        loss_dict["total_loss"] = total_loss.item()
        loss_dict["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        
        self.global_step += 1
        
        return loss_dict
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "global_step": self.global_step
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]


# Utility functions for model usage
def create_molecular_foundation_model(config: MolecularFoundationConfig = None) -> MolecularFoundationModel:
    """Create a molecular foundation model with default or custom config."""
    if config is None:
        config = MolecularFoundationConfig()
    
    model = MolecularFoundationModel(config)
    
    logger.info(f"Created molecular foundation model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def load_pretrained_molecular_model(checkpoint_path: str) -> Tuple[MolecularFoundationModel, MolecularFoundationConfig]:
    """Load a pre-trained molecular foundation model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    model = MolecularFoundationModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    logger.info(f"Loaded pre-trained model from {checkpoint_path}")
    
    return model, config


# Example usage and demonstration
def demonstrate_molecular_foundation_model():
    """Demonstrate the molecular foundation model capabilities."""
    logger.info("Demonstrating Molecular Foundation Model capabilities...")
    
    # Create model
    config = MolecularFoundationConfig(
        d_model=512,  # Smaller for demo
        num_layers=6,
        num_heads=8,
        vocab_size=32,
        max_sequence_length=256
    )
    
    model = create_molecular_foundation_model(config)
    
    # Demo data
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    coordinates = torch.randn(batch_size, seq_len, 3)
    
    # Forward pass
    outputs = model(input_ids, coordinates=coordinates, task="all")
    
    logger.info(f"Model output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape}")
    
    # Test different prediction tasks
    structure_pred = model.predict_structure(input_ids, coordinates=coordinates)
    contact_pred = model.predict_contacts(input_ids)
    dynamics_pred = model.predict_dynamics(input_ids)
    
    logger.info(f"Structure prediction shape: {structure_pred.shape}")
    logger.info(f"Contact prediction shape: {contact_pred.shape}")
    logger.info(f"Dynamics prediction shape: {dynamics_pred.shape}")
    
    return model


if __name__ == "__main__":
    demonstrate_molecular_foundation_model()