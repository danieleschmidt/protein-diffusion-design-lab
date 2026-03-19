"""
Tests for the protein diffusion design pipeline.
"""

import pytest
import torch
import random

from src.protein_diffusion import (
    ProteinSequenceEncoder,
    DiffusionProteinDesigner,
    BindingAffinityPredictor,
)
from src.protein_diffusion.encoder import AMINO_ACIDS, VOCAB_SIZE, PAD_IDX
from src.protein_diffusion.predictor import synthetic_affinity


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def encoder():
    return ProteinSequenceEncoder(embed_dim=32, max_len=50)


@pytest.fixture(scope="module")
def designer(encoder):
    return DiffusionProteinDesigner(encoder=encoder, seq_len=20, T=10)


@pytest.fixture(scope="module")
def predictor(encoder):
    return BindingAffinityPredictor(encoder=encoder, hidden_dim=64)


# ──────────────────────────────────────────────────────────────────────
# ProteinSequenceEncoder tests
# ──────────────────────────────────────────────────────────────────────

class TestProteinSequenceEncoder:

    def test_vocab_size(self):
        assert VOCAB_SIZE == 24  # 20 AA + 4 special tokens

    def test_tokenize_shape(self, encoder):
        seqs = ["ACDE", "KFGH"]
        ids, mask = encoder.tokenize(seqs)
        assert ids.shape == (2, 4)
        assert mask.shape == (2, 4)

    def test_tokenize_padding(self, encoder):
        seqs = ["AC", "ACDEFG"]
        ids, mask = encoder.tokenize(seqs)
        # First sequence should have padding in positions 2+
        assert mask[0, 2].item() is True
        assert mask[1, 2].item() is False

    def test_forward_shape(self, encoder):
        seqs = ["ACDE", "KFGH", "MNPQ"]
        out = encoder(seqs)
        assert out.shape == (3, 4, 32)

    def test_pool_shape(self, encoder):
        seqs = ["ACDE", "KFGHMNPQ"]
        pooled = encoder.pool(seqs)
        assert pooled.shape == (2, 32)

    def test_encode_decode_roundtrip(self, encoder):
        """Decoded tokens should be valid amino acid characters."""
        seqs = ["ACDEFGHIKLM"]
        ids, _ = encoder.tokenize(seqs)
        decoded = encoder.decode(ids)
        for aa in decoded[0]:
            assert aa in AMINO_ACIDS, f"Unexpected character: {aa}"

    def test_handles_lowercase(self, encoder):
        """Encoder should normalise lowercase input."""
        seqs = ["acde"]
        out = encoder(seqs)
        assert out.shape[0] == 1

    def test_unknown_aa_maps_to_mask(self, encoder):
        """Characters not in the amino acid alphabet map to MASK token."""
        seqs = ["XZZZ"]
        ids, _ = encoder.tokenize(seqs)
        from src.protein_diffusion.encoder import MASK_IDX
        assert all(ids[0, i].item() == MASK_IDX for i in range(4))


# ──────────────────────────────────────────────────────────────────────
# DiffusionProteinDesigner tests
# ──────────────────────────────────────────────────────────────────────

class TestDiffusionProteinDesigner:

    def test_noise_schedule_shape(self, designer):
        assert designer.betas.shape == (10,)
        assert designer.alpha_bar.shape == (10,)

    def test_noise_schedule_range(self, designer):
        assert (designer.betas > 0).all()
        assert (designer.alpha_bar > 0).all()
        assert (designer.alpha_bar <= 1).all()
        # alpha_bar should be monotonically decreasing
        assert (designer.alpha_bar[1:] < designer.alpha_bar[:-1]).all()

    def test_q_sample_shape(self, designer):
        x0 = torch.randn(2, 20, 32)
        t  = torch.randint(0, 10, (2,))
        xt = designer.q_sample(x0, t)
        assert xt.shape == x0.shape

    def test_q_sample_is_noisy(self, designer):
        """x_t should not equal x_0 (noise was added)."""
        torch.manual_seed(99)
        x0 = torch.randn(2, 20, 32)
        t  = torch.full((2,), 9)
        xt = designer.q_sample(x0, t)
        assert not torch.allclose(x0, xt)

    def test_compute_loss_scalar(self, designer):
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "WYSVTRQPNMLKIHGFEDCA"]
        loss = designer.compute_loss(seqs)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_sample_count(self, designer):
        seqs, embs = designer.sample(n=3)
        assert len(seqs) == 3
        assert embs.shape[0] == 3

    def test_sample_sequence_length(self, designer):
        seqs, _ = designer.sample(n=2)
        for seq in seqs:
            assert len(seq) <= 20  # ≤ seq_len (may be shorter after stripping specials)
            assert all(aa in AMINO_ACIDS for aa in seq)

    def test_sample_sequences_differ(self, designer):
        """Generated sequences should not all be identical."""
        seqs, _ = designer.sample(n=4)
        assert len(set(seqs)) > 1


# ──────────────────────────────────────────────────────────────────────
# BindingAffinityPredictor tests
# ──────────────────────────────────────────────────────────────────────

class TestBindingAffinityPredictor:

    def test_synthetic_affinity_range(self):
        seqs = ["ACDE", "WWWWWW", "CCCCCC", "GGGGGG"]
        affs = synthetic_affinity(seqs)
        assert (affs >= 0).all()
        assert (affs <= 1).all()

    def test_synthetic_affinity_cys_scores_higher(self):
        """Cysteine-rich sequence should score higher than glycine-only."""
        rich = synthetic_affinity(["CCCCCCCCCC"])
        poor = synthetic_affinity(["GGGGGGGGGG"])
        assert rich[0] > poor[0]

    def test_predict_shape(self, predictor):
        seqs = ["ACDE", "KFGH"]
        affs = predictor.predict(seqs)
        assert affs.shape == (2,)

    def test_predict_range(self, predictor):
        seqs = ["ACDEFGHIKLMNPQRSTVWY"]
        affs = predictor.predict(seqs)
        assert (affs >= 0).all()
        assert (affs <= 1).all()

    def test_train_on_synthetic_returns_losses(self, predictor):
        losses = predictor.train_on_synthetic(
            n_samples=200, seq_len=20, epochs=5, verbose=False
        )
        assert len(losses) == 5
        assert all(l >= 0 for l in losses)

    def test_train_reduces_loss(self, predictor):
        """Loss should generally decrease over training."""
        losses = predictor.train_on_synthetic(
            n_samples=500, seq_len=20, epochs=20, verbose=False
        )
        # Last epoch should be lower than first
        assert losses[-1] < losses[0]

    def test_predict_after_training(self, predictor):
        """After training, predictions should be plausible numbers in [0, 1]."""
        predictor.train_on_synthetic(n_samples=300, seq_len=20, epochs=10, verbose=False)
        seqs = ["CCCWWWKKK", "GGGGGGGGG"]
        affs = predictor.predict(seqs)
        assert (affs >= 0).all()
        assert (affs <= 1).all()
