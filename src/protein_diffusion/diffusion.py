"""
DiffusionProteinDesigner

Implements a simplified DDPM (Denoising Diffusion Probabilistic Model) in
continuous sequence embedding space.

Forward process:  q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
Reverse process:  p_theta(x_{t-1} | x_t) — learned denoiser network
Noise schedule:   Linear beta schedule from beta_start to beta_end over T steps.

At inference, the reverse process starts from pure Gaussian noise and
iteratively denoises to produce a sequence embedding, which is then decoded
to an amino acid string via nearest-neighbour lookup in the encoder's token
embedding table.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .encoder import ProteinSequenceEncoder, IDX_TO_AA, SPECIAL_TOKENS, PAD_IDX


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def _linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Returns a 1-D tensor of betas for timesteps 1..T."""
    return torch.linspace(beta_start, beta_end, T)


def _precompute_schedule(betas: torch.Tensor):
    alphas           = 1.0 - betas
    alpha_bar        = torch.cumprod(alphas, dim=0)
    alpha_bar_prev   = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

    sqrt_alpha_bar         = alpha_bar.sqrt()
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
    sqrt_recip_alphas      = (1.0 / alphas).sqrt()
    posterior_variance     = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    return {
        "betas":                     betas,
        "alphas":                    alphas,
        "alpha_bar":                 alpha_bar,
        "alpha_bar_prev":            alpha_bar_prev,
        "sqrt_alpha_bar":            sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar":  sqrt_one_minus_alpha_bar,
        "sqrt_recip_alphas":         sqrt_recip_alphas,
        "posterior_variance":        posterior_variance,
    }


# ---------------------------------------------------------------------------
# Denoiser network
# ---------------------------------------------------------------------------

class _SinusoidalTimeEmb(nn.Module):
    """Sinusoidal timestep embedding (like transformer position encoding)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _DenoiserMLP(nn.Module):
    """
    A small MLP denoiser that takes (x_t, t_emb) → predicted noise.

    Input:  noisy embedding x_t of shape (B, L, D) + timestep embedding
    Output: predicted noise of the same shape (B, L, D)
    """

    def __init__(self, embed_dim: int, seq_len: int, time_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len   = seq_len
        flat_dim = embed_dim * seq_len

        self.time_mlp = nn.Sequential(
            _SinusoidalTimeEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(flat_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)  noisy embeddings
            t: (B,)        integer timesteps

        Returns:
            noise_pred: (B, L, D)
        """
        B, L, D = x.shape
        x_flat = x.reshape(B, L * D)
        t_emb  = self.time_mlp(t)
        inp    = torch.cat([x_flat, t_emb], dim=-1)
        out    = self.net(inp)
        return out.reshape(B, L, D)


# ---------------------------------------------------------------------------
# Main designer class
# ---------------------------------------------------------------------------

class DiffusionProteinDesigner(nn.Module):
    """
    DDPM-style diffusion model operating in amino acid embedding space.

    The model diffuses continuous embeddings (produced by ProteinSequenceEncoder)
    rather than discrete tokens.  After the reverse process, each position's
    embedding is decoded to an amino acid by finding the nearest token in the
    encoder's embedding table.

    Args:
        encoder:     A ProteinSequenceEncoder instance.
        seq_len:     Length of sequences to generate.
        T:           Number of diffusion timesteps (default 100).
        beta_start:  Start of linear noise schedule (default 1e-4).
        beta_end:    End of linear noise schedule (default 0.02).
        hidden_dim:  Hidden size for the denoiser MLP (default 256).
    """

    def __init__(
        self,
        encoder: ProteinSequenceEncoder,
        seq_len: int = 30,
        T: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder  = encoder
        self.seq_len  = seq_len
        self.T        = T
        self.embed_dim = encoder.embed_dim

        # Noise schedule
        betas = _linear_beta_schedule(T, beta_start, beta_end)
        schedule = _precompute_schedule(betas)
        for k, v in schedule.items():
            self.register_buffer(k, v)

        # Denoiser
        self.denoiser = _DenoiserMLP(
            embed_dim=self.embed_dim,
            seq_len=seq_len,
            hidden=hidden_dim,
        )

    # ------------------------------------------------------------------
    # Forward (training) process
    # ------------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample x_t from the forward process: q(x_t | x_0).

        Args:
            x0:    (B, L, D)  clean embeddings
            t:     (B,)        timesteps (0-indexed)
            noise: optional pre-generated noise

        Returns:
            x_t: (B, L, D)  noisy embeddings
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sqrt_ab * x0 + sqrt_1mab * noise

    def compute_loss(self, sequences: List[str]) -> torch.Tensor:
        """
        Simple DDPM training loss (MSE between predicted and actual noise).

        Args:
            sequences: List of amino acid strings.

        Returns:
            Scalar MSE loss.
        """
        with torch.no_grad():
            x0 = self.encoder(sequences)  # (B, L, D)  — may be shorter than seq_len

        B, L, D = x0.shape
        if L < self.seq_len:
            pad = torch.zeros(B, self.seq_len - L, D, device=x0.device)
            x0 = torch.cat([x0, pad], dim=1)
        elif L > self.seq_len:
            x0 = x0[:, :self.seq_len, :]

        t     = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t   = self.q_sample(x0, t, noise)

        noise_pred = self.denoiser(x_t, t)
        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # Reverse (sampling) process
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One step of the reverse diffusion process."""
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        noise_pred = self.denoiser(x_t, t_tensor)

        # Compute x_{t-1} mean
        beta_t            = self.betas[t]
        sqrt_1mab_t       = self.sqrt_one_minus_alpha_bar[t]
        sqrt_recip_a_t    = self.sqrt_recip_alphas[t]

        mean = sqrt_recip_a_t * (x_t - (beta_t / sqrt_1mab_t) * noise_pred)

        if t == 0:
            return mean
        var   = self.posterior_variance[t]
        noise = torch.randn_like(x_t)
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def sample(self, n: int = 5) -> Tuple[List[str], torch.Tensor]:
        """
        Generate n novel protein sequences via the reverse diffusion process.

        Args:
            n: Number of sequences to generate.

        Returns:
            (sequences, embeddings)
            - sequences:  List[str] of decoded amino acid strings
            - embeddings: (n, seq_len, embed_dim) final denoised embeddings
        """
        device = next(self.parameters()).device
        x = torch.randn(n, self.seq_len, self.embed_dim, device=device)

        for t in reversed(range(self.T)):
            x = self.p_sample_step(x, t)

        sequences = self._decode_embeddings(x)
        return sequences, x

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _decode_embeddings(self, embeddings: torch.Tensor) -> List[str]:
        """
        Decode continuous embeddings to amino acid strings via nearest-neighbour
        lookup in the encoder's token embedding table.

        Args:
            embeddings: (B, L, D)

        Returns:
            List of amino acid strings (length = B).
        """
        # Token embedding weight matrix: (VOCAB_SIZE, D)
        vocab_emb = self.encoder.token_emb.weight  # (V, D)

        B, L, D = embeddings.shape
        flat     = embeddings.reshape(B * L, D)     # (B*L, D)

        # Cosine similarity → nearest token
        flat_norm   = F.normalize(flat, dim=-1)
        vocab_norm  = F.normalize(vocab_emb, dim=-1)
        sims        = flat_norm @ vocab_norm.T       # (B*L, V)
        token_ids   = sims.argmax(dim=-1).reshape(B, L)  # (B, L)

        from .encoder import IDX_TO_AA, SPECIAL_TOKENS
        sequences = []
        for row in token_ids:
            aas = []
            for idx in row.tolist():
                token = IDX_TO_AA.get(idx, "X")
                if token not in SPECIAL_TOKENS:
                    aas.append(token)
            sequences.append("".join(aas))
        return sequences
