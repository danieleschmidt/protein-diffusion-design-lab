#!/usr/bin/env python3
"""
Protein Diffusion Design Lab — Demo

Demonstrates the full pipeline:
1. Build encoder, diffusion model, and binding affinity predictor
2. Quick training loop (diffusion on random sequences)
3. Generate 5 novel protein sequences via reverse diffusion
4. Score them with the binding affinity predictor
"""

import random
import torch
from src.protein_diffusion import (
    ProteinSequenceEncoder,
    DiffusionProteinDesigner,
    BindingAffinityPredictor,
)
from src.protein_diffusion.encoder import AMINO_ACIDS

# Reproducibility
torch.manual_seed(42)
random.seed(42)

DEVICE  = "cpu"
SEQ_LEN = 30
EMBED   = 64
T       = 100

# ──────────────────────────────────────────────────────────────────────
# 1. Initialise models
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Protein Diffusion Design Lab")
print("=" * 60)

print("\n[1] Initialising models …")
encoder   = ProteinSequenceEncoder(embed_dim=EMBED, max_len=SEQ_LEN + 4).to(DEVICE)
designer  = DiffusionProteinDesigner(encoder=encoder, seq_len=SEQ_LEN, T=T).to(DEVICE)
predictor = BindingAffinityPredictor(encoder=encoder).to(DEVICE)

n_params = sum(p.numel() for p in designer.parameters() if p.requires_grad)
print(f"    Diffusion model parameters: {n_params:,}")

# ──────────────────────────────────────────────────────────────────────
# 2. Quick diffusion training (toy loop)
# ──────────────────────────────────────────────────────────────────────
print("\n[2] Training diffusion model on synthetic sequences …")
optim = torch.optim.Adam(designer.denoiser.parameters(), lr=3e-4)

TRAIN_SEQS = 200
EPOCHS     = 30
seqs_pool  = [
    "".join(random.choices(AMINO_ACIDS, k=SEQ_LEN))
    for _ in range(TRAIN_SEQS)
]

for epoch in range(EPOCHS):
    batch = random.sample(seqs_pool, 32)
    loss  = designer.compute_loss(batch)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1:2d}/{EPOCHS}  diffusion loss = {loss.item():.4f}")

# ──────────────────────────────────────────────────────────────────────
# 3. Train binding affinity predictor on synthetic data
# ──────────────────────────────────────────────────────────────────────
print("\n[3] Training binding affinity predictor on synthetic data …")
predictor.train()
predictor.train_on_synthetic(n_samples=2000, seq_len=SEQ_LEN, epochs=50, verbose=True)
predictor.eval()

# ──────────────────────────────────────────────────────────────────────
# 4. Generate novel sequences via reverse diffusion
# ──────────────────────────────────────────────────────────────────────
print("\n[4] Generating 5 novel sequences via reverse diffusion …")
designer.eval()
sequences, embeddings = designer.sample(n=5)

print("\n[5] Scoring sequences with binding affinity predictor …")
affinities = predictor.predict(sequences)

# ──────────────────────────────────────────────────────────────────────
# 5. Results
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Generated Protein Sequences")
print("=" * 60)
print(f"  {'#':<4}  {'Sequence':<35}  {'Length':>6}  {'Affinity':>8}")
print("  " + "-" * 56)

for i, (seq, aff) in enumerate(zip(sequences, affinities.tolist()), 1):
    print(f"  {i:<4}  {seq:<35}  {len(seq):>6}  {aff:>8.4f}")

best_idx = affinities.argmax().item()
print("\n  Best candidate:")
print(f"    Sequence:  {sequences[best_idx]}")
print(f"    Affinity:  {affinities[best_idx]:.4f}")
print("=" * 60)
