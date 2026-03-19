"""
BindingAffinityPredictor

A simple MLP that predicts binding affinity (a scalar score) from a
mean-pooled sequence embedding produced by ProteinSequenceEncoder.

Training data is synthetic: affinity is defined as a deterministic function
of the sequence composition (cysteine/tryptophan/aromatic content) plus noise,
giving the model a plausible signal to learn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .encoder import ProteinSequenceEncoder, AMINO_ACIDS


# ---------------------------------------------------------------------------
# Synthetic affinity function (used for both generating training labels
# and as an interpretable ground truth for the demo)
# ---------------------------------------------------------------------------

def synthetic_affinity(sequences: List[str]) -> torch.Tensor:
    """
    Returns a plausible (but fully synthetic) binding affinity score for each
    sequence, in the range roughly [0, 1].

    Score is based on:
    - Cysteine (C) content  — disulfide bridges promote binding
    - Tryptophan (W) content — hydrophobic anchoring
    - Charged residues (K, R, D, E) — electrostatic interactions
    - Sequence length normalisation
    """
    scores = []
    for seq in sequences:
        seq = seq.upper()
        n = max(len(seq), 1)
        cys   = seq.count("C") / n
        trp   = seq.count("W") / n
        chg   = (seq.count("K") + seq.count("R") + seq.count("D") + seq.count("E")) / n
        # Weighted combination → sigmoid to keep in (0, 1)
        raw   = 3.0 * cys + 2.0 * trp + 1.5 * chg - 0.5
        score = torch.sigmoid(torch.tensor(raw, dtype=torch.float32)).item()
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Predictor network
# ---------------------------------------------------------------------------

class BindingAffinityPredictor(nn.Module):
    """
    MLP that maps a fixed-size sequence embedding (embed_dim,) → scalar affinity.

    Args:
        encoder:     A ProteinSequenceEncoder (used to embed inputs at inference).
        hidden_dim:  Hidden layer size (default 128).
    """

    def __init__(self, encoder: ProteinSequenceEncoder, hidden_dim: int = 128):
        super().__init__()
        self.encoder    = encoder
        embed_dim       = encoder.embed_dim

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (B, embed_dim) mean-pooled sequence embeddings.

        Returns:
            affinity: (B, 1) predicted binding affinity in [0, 1].
        """
        return self.net(pooled)

    def predict(self, sequences: List[str]) -> torch.Tensor:
        """
        High-level API: encode sequences and predict affinity.

        Args:
            sequences: List of amino acid strings.

        Returns:
            affinity: (B,) predicted affinities.
        """
        with torch.no_grad():
            pooled = self.encoder.pool(sequences)
        return self.forward(pooled).squeeze(-1)

    # ------------------------------------------------------------------
    # Quick training on synthetic data
    # ------------------------------------------------------------------

    def train_on_synthetic(
        self,
        n_samples: int = 2000,
        seq_len: int = 30,
        epochs: int = 50,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> List[float]:
        """
        Generate synthetic protein sequences, compute ground-truth affinities
        from `synthetic_affinity`, and train the MLP for a few epochs.

        Returns:
            List of per-epoch training losses.
        """
        import random

        # -- Generate random sequences ------------------------------------------------
        seqs = [
            "".join(random.choices(AMINO_ACIDS, k=seq_len))
            for _ in range(n_samples)
        ]
        labels = synthetic_affinity(seqs)  # (N,)

        # -- Encode once (no gradients needed for encoder in this demo) ---------------
        with torch.no_grad():
            pooled = self.encoder.pool(seqs)  # (N, embed_dim)

        # -- Train loop ---------------------------------------------------------------
        optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        losses = []
        batch_size = 128

        for epoch in range(epochs):
            epoch_loss = 0.0
            perm = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                x   = pooled[idx]
                y   = labels[idx].unsqueeze(-1)

                pred = self.net(x)
                loss = F.mse_loss(pred, y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / (n_samples / batch_size)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}")

        return losses
