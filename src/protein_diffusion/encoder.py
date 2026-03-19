"""
ProteinSequenceEncoder

Encodes amino acid sequences as continuous embeddings suitable for
diffusion in embedding space. Supports the standard 20 amino acids
plus special tokens for padding and masking.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


# Standard 20 amino acids + special tokens
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>", "<MASK>", "<BOS>", "<EOS>"]
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS

VOCAB_SIZE = len(VOCAB)       # 24
PAD_IDX   = VOCAB.index("<PAD>")
MASK_IDX  = VOCAB.index("<MASK>")
BOS_IDX   = VOCAB.index("<BOS>")
EOS_IDX   = VOCAB.index("<EOS>")

AA_TO_IDX = {aa: i for i, aa in enumerate(VOCAB)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}


class ProteinSequenceEncoder(nn.Module):
    """
    Encodes protein sequences (strings of amino acids) into continuous
    embedding vectors.

    Architecture:
        - Learnable token embeddings of size `embed_dim`
        - Sinusoidal positional encodings
        - Optional lightweight transformer layer for context

    Args:
        embed_dim:   Dimension of the sequence embedding (default 64).
        max_len:     Maximum sequence length (default 128).
        use_context: If True, adds a single transformer encoder layer to
                     capture local residue context (default True).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        max_len: int = 128,
        use_context: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Token embedding table
        self.token_emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)

        # Learnable positional encodings (simple, works well in practice)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # Optional single-layer transformer for local context
        self.use_context = use_context
        if use_context:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=0.0,
                batch_first=True,
            )
            self.context = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of amino acid strings to padded token id tensors.

        Returns:
            ids:  (B, L) LongTensor of token indices
            mask: (B, L) BoolTensor — True where padding
        """
        max_seq_len = min(max(len(s) for s in sequences), self.max_len)
        B = len(sequences)
        ids = torch.full((B, max_seq_len), PAD_IDX, dtype=torch.long)
        for i, seq in enumerate(sequences):
            seq = seq.upper()[:max_seq_len]
            for j, aa in enumerate(seq):
                ids[i, j] = AA_TO_IDX.get(aa, MASK_IDX)
        pad_mask = ids == PAD_IDX
        return ids, pad_mask

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode a batch of sequences to embeddings.

        Args:
            sequences: List of amino acid strings (e.g. ["ACDE", "KFGH..."])

        Returns:
            Tensor of shape (B, L, embed_dim) — per-residue embeddings.
        """
        ids, pad_mask = self.tokenize(sequences)
        B, L = ids.shape

        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(ids) + self.pos_emb(pos)

        if self.use_context:
            x = self.context(x, src_key_padding_mask=pad_mask)

        x = self.norm(x)
        return x  # (B, L, embed_dim)

    def pool(self, sequences: List[str]) -> torch.Tensor:
        """
        Mean-pool residue embeddings (ignoring padding) to get a
        fixed-size (B, embed_dim) sequence representation.
        """
        emb = self.forward(sequences)          # (B, L, D)
        _, pad_mask = self.tokenize(sequences)
        valid = ~pad_mask                       # (B, L) — True where not padding
        # Avoid division by zero for degenerate inputs
        lengths = valid.sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled = (emb * valid.unsqueeze(-1)).sum(dim=1) / lengths
        return pooled  # (B, embed_dim)

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def decode(ids: torch.Tensor) -> List[str]:
        """
        Convert a (B, L) tensor of token ids back to amino acid strings,
        stripping special tokens.
        """
        sequences = []
        for row in ids:
            aas = []
            for idx in row.tolist():
                token = IDX_TO_AA.get(idx, "?")
                if token in SPECIAL_TOKENS:
                    continue
                aas.append(token)
            sequences.append("".join(aas))
        return sequences
