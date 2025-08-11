"""
Protein sequence embeddings and structural encodings.

This module provides advanced embedding systems for protein sequences,
including ESM-based embeddings and geometric structure encodings.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    from .. import mock_torch as torch
    nn = torch.nn
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
        @staticmethod
        def log(x):
            import math
            return math.log(x) if isinstance(x, (int, float)) else 9.21  # log(10000.0)
        @staticmethod
        def degrees(x):
            return x * 180 / 3.14159
        @staticmethod
        def pi():
            return 3.14159
    np = MockNumpy()
    NUMPY_AVAILABLE = False
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


@dataclass
class EmbeddingConfig:
    """Configuration for protein embeddings."""
    embedding_dim: int = 1024
    max_length: int = 512
    use_esm: bool = True
    esm_model: str = "esm2_t33_650M_UR50D"
    freeze_esm: bool = True
    use_positional_encoding: bool = True
    use_structural_features: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for protein sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:x.size(0), :]


class ESMEmbedding(nn.Module):
    """ESM-based protein sequence embeddings."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        
        if not ESM_AVAILABLE:
            raise ImportError("ESM not available. Install with: pip install fair-esm")
        
        # Load ESM model
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(config.esm_model)
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        
        # Freeze ESM parameters if specified
        if config.freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # Get ESM embedding dimension
        esm_dim = self.esm_model.embed_dim
        
        # Project to desired dimension if needed
        if esm_dim != config.embedding_dim:
            self.projection = nn.Linear(esm_dim, config.embedding_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Generate ESM embeddings for protein sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Tensor of shape [batch_size, max_len, embedding_dim]
        """
        # Prepare data for ESM
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(data)
        
        batch_tokens = batch_tokens.to(next(self.esm_model.parameters()).device)
        
        # Get ESM representations
        with torch.no_grad() if self.config.freeze_esm else torch.enable_grad():
            results = self.esm_model(batch_tokens, repr_layers=[self.esm_model.num_layers])
            embeddings = results["representations"][self.esm_model.num_layers]
        
        # Remove BOS/EOS tokens
        embeddings = embeddings[:, 1:-1, :]
        
        # Project to desired dimension
        embeddings = self.projection(embeddings)
        
        return embeddings


class StructuralFeatures(nn.Module):
    """Extract structural features from protein sequences."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Amino acid properties (normalized)
        self.aa_properties = {
            'A': [0.0, 0.0, 0.0, 0.0],      # Hydrophobicity, size, charge, polarity
            'C': [0.2, 0.1, 0.0, 0.1],
            'D': [-0.8, 0.1, -1.0, 1.0],
            'E': [-0.8, 0.2, -1.0, 1.0],
            'F': [0.8, 0.7, 0.0, 0.0],
            'G': [0.0, -0.5, 0.0, 0.0],
            'H': [-0.2, 0.3, 0.5, 1.0],
            'I': [0.7, 0.3, 0.0, 0.0],
            'K': [-0.8, 0.4, 1.0, 1.0],
            'L': [0.7, 0.3, 0.0, 0.0],
            'M': [0.4, 0.3, 0.0, 0.0],
            'N': [-0.5, 0.2, 0.0, 1.0],
            'P': [-0.2, 0.1, 0.0, 0.0],
            'Q': [-0.5, 0.3, 0.0, 1.0],
            'R': [-0.8, 0.5, 1.0, 1.0],
            'S': [-0.3, 0.0, 0.0, 1.0],
            'T': [-0.2, 0.1, 0.0, 1.0],
            'V': [0.5, 0.1, 0.0, 0.0],
            'W': [1.0, 1.0, 0.0, 0.1],
            'Y': [0.3, 0.8, 0.0, 1.0],
        }
        
        # Feature projection
        self.feature_projection = nn.Linear(4, config.embedding_dim // 4)
    
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Extract structural features from sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Tensor of structural features
        """
        batch_features = []
        
        for seq in sequences:
            seq_features = []
            for aa in seq:
                if aa in self.aa_properties:
                    seq_features.append(self.aa_properties[aa])
                else:
                    seq_features.append([0.0, 0.0, 0.0, 0.0])  # Unknown AA
            
            # Pad to max length
            while len(seq_features) < self.config.max_length:
                seq_features.append([0.0, 0.0, 0.0, 0.0])
            
            # Truncate if too long
            seq_features = seq_features[:self.config.max_length]
            batch_features.append(seq_features)
        
        features = torch.tensor(batch_features, dtype=torch.float32)
        
        # Project features
        projected_features = self.feature_projection(features)
        
        return projected_features


class GeometricHashing(nn.Module):
    """Geometric hashing for 3D structure representation."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        self.hash_dim = 64
        
        # Learnable hash functions
        self.hash_layers = nn.ModuleList([
            nn.Linear(3, self.hash_dim) for _ in range(4)
        ])
        
        self.projection = nn.Linear(self.hash_dim * 4, config.embedding_dim // 4)
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Generate geometric hash codes for 3D coordinates.
        
        Args:
            coordinates: 3D coordinates [batch_size, seq_len, 3]
            
        Returns:
            Geometric hash embeddings
        """
        batch_size, seq_len, _ = coordinates.shape
        
        # Apply hash functions
        hash_codes = []
        for hash_layer in self.hash_layers:
            hash_code = torch.tanh(hash_layer(coordinates))
            hash_codes.append(hash_code)
        
        # Concatenate hash codes
        combined_hash = torch.cat(hash_codes, dim=-1)
        
        # Project to embedding dimension
        embeddings = self.projection(combined_hash)
        
        return embeddings


class ProteinEmbeddings(nn.Module):
    """
    Comprehensive protein embedding system combining multiple representations.
    
    This module combines:
    - ESM-based sequence embeddings
    - Structural property features
    - Positional encodings
    - Optional geometric hashing for 3D structures
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Initialize embedding components
        if config.use_esm and ESM_AVAILABLE:
            self.esm_embedding = ESMEmbedding(config)
            embedding_components = 1
        else:
            # Fallback to learned embeddings if ESM not available
            self.token_embedding = nn.Embedding(50000, config.embedding_dim)
            embedding_components = 1
        
        if config.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(config.embedding_dim, config.max_length)
        
        if config.use_structural_features:
            self.structural_features = StructuralFeatures(config)
            embedding_components += 1
        
        self.geometric_hashing = GeometricHashing(config)
        
        # Combine different embedding types
        if embedding_components > 1:
            self.combine_projection = nn.Linear(
                config.embedding_dim * embedding_components, 
                config.embedding_dim
            )
        else:
            self.combine_projection = nn.Identity()
        
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        sequences: List[str],
        token_ids: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate comprehensive protein embeddings.
        
        Args:
            sequences: List of protein sequences
            token_ids: Optional tokenized sequences
            coordinates: Optional 3D coordinates
            
        Returns:
            Combined protein embeddings
        """
        embeddings_list = []
        
        # Sequence embeddings
        if self.config.use_esm and hasattr(self, 'esm_embedding'):
            seq_embeddings = self.esm_embedding(sequences)
        else:
            # Use token embeddings as fallback
            if token_ids is None:
                raise ValueError("token_ids required when ESM is not available")
            seq_embeddings = self.token_embedding(token_ids)
        
        # Add positional encoding
        if self.config.use_positional_encoding:
            seq_embeddings = self.positional_encoding(seq_embeddings.transpose(0, 1)).transpose(0, 1)
        
        embeddings_list.append(seq_embeddings)
        
        # Structural features
        if self.config.use_structural_features:
            struct_features = self.structural_features(sequences)
            # Expand to match sequence embedding dimensions if needed
            if struct_features.shape[1] < seq_embeddings.shape[1]:
                padding = seq_embeddings.shape[1] - struct_features.shape[1]
                struct_features = torch.cat([
                    struct_features,
                    torch.zeros(struct_features.shape[0], padding, struct_features.shape[2])
                ], dim=1)
            embeddings_list.append(struct_features)
        
        # Geometric features (if coordinates provided)
        if coordinates is not None:
            geom_features = self.geometric_hashing(coordinates)
            embeddings_list.append(geom_features)
        
        # Combine embeddings
        if len(embeddings_list) > 1:
            combined_embeddings = torch.cat(embeddings_list, dim=-1)
            combined_embeddings = self.combine_projection(combined_embeddings)
        else:
            combined_embeddings = embeddings_list[0]
        
        # Apply layer norm and dropout
        combined_embeddings = self.layer_norm(combined_embeddings)
        combined_embeddings = self.dropout(combined_embeddings)
        
        return combined_embeddings
    
    def get_motif_embeddings(self, motif_sequences: List[str]) -> torch.Tensor:
        """
        Generate embeddings for specific protein motifs.
        
        Args:
            motif_sequences: List of motif sequences
            
        Returns:
            Motif embeddings for conditioning
        """
        return self.forward(motif_sequences)
    
    def encode_batch(
        self,
        sequences: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of sequences with proper padding and attention masks.
        
        Args:
            sequences: List of protein sequences
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with embeddings and attention masks
        """
        if max_length is None:
            max_length = self.config.max_length
        
        # Pad/truncate sequences to uniform length
        processed_sequences = []
        attention_masks = []
        
        for seq in sequences:
            if len(seq) > max_length:
                seq = seq[:max_length]
            
            attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))
            seq = seq + "A" * (max_length - len(seq))  # Pad with alanine
            
            processed_sequences.append(seq)
            attention_masks.append(attention_mask)
        
        # Generate embeddings
        embeddings = self.forward(processed_sequences)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        
        return {
            "embeddings": embeddings,
            "attention_mask": attention_masks,
        }