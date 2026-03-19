# Protein Diffusion Design Lab

A diffusion-based protein sequence design pipeline implemented in PyTorch.

Generates novel protein sequences by running a simplified DDPM (Denoising
Diffusion Probabilistic Model) in continuous amino acid embedding space, then
scores candidates with a binding affinity predictor trained on synthetic data.

---

## Architecture

```
Input sequences
      │
      ▼
ProteinSequenceEncoder          — token embeddings + positional encoding
      │                            + optional transformer context layer
      ▼
DiffusionProteinDesigner        — DDPM forward/reverse in embedding space
  • Forward process: q(x_t|x_0) = N(√ᾱ_t · x_0, (1−ᾱ_t)·I)
  • Reverse process: p_θ(x_{t-1}|x_t) via MLP denoiser
  • T=100 timesteps, linear β schedule
  • Decode: nearest-neighbour lookup in token embedding table
      │
      ▼
Novel sequences
      │
      ▼
BindingAffinityPredictor        — MLP on mean-pooled embeddings
                                   trained on synthetic affinity data
```

### Components

| Class | File | Role |
|---|---|---|
| `ProteinSequenceEncoder` | `encoder.py` | Amino acid → continuous embedding |
| `DiffusionProteinDesigner` | `diffusion.py` | DDPM in sequence embedding space |
| `BindingAffinityPredictor` | `predictor.py` | Scalar binding score from embedding |

---

## Quick Start

```bash
# Requires PyTorch
pip install torch

# Run the demo
python demo.py
```

Example output:

```
============================================================
  Generated Protein Sequences
============================================================
  #     Sequence                             Length  Affinity
  --------------------------------------------------------
  1     DVTFNRQAVFEIQDIVGWHLDLAVVPHKE            29    0.4742
  2     MFIMFDVRIMEHKVKVFVYWFFLRHG               26    0.4786
  3     QNHGAGEQVCCEVPNHKMNQWMC                  23    0.5110
  4     GQHICIYVDSKDFMALTQVWHNQLNA               26    0.4547
  5     WAHKNVAAEVMEWVTYPQTIHASCQQ               26    0.4896

  Best candidate:
    Sequence:  QNHGAGEQVCCEVPNHKMNQWMC
    Affinity:  0.5110
```

---

## Usage

```python
from src.protein_diffusion import (
    ProteinSequenceEncoder,
    DiffusionProteinDesigner,
    BindingAffinityPredictor,
)

# Build models
encoder   = ProteinSequenceEncoder(embed_dim=64, max_len=128)
designer  = DiffusionProteinDesigner(encoder=encoder, seq_len=30, T=100)
predictor = BindingAffinityPredictor(encoder=encoder)

# Train diffusion model on your sequences
loss = designer.compute_loss(["ACDEFGHIKLM", "MNPQRSTVWY..."])

# Train predictor on synthetic data (or your own labels)
predictor.train_on_synthetic(n_samples=2000, epochs=50)

# Generate 10 novel sequences
sequences, embeddings = designer.sample(n=10)

# Score them
affinities = predictor.predict(sequences)
```

---

## Tests

```bash
python -m pytest tests/ -v
```

23 tests covering encoder tokenisation, diffusion forward/reverse process,
noise schedule properties, and predictor training/inference.

---

## Design Notes

**Embedding-space diffusion** — operating on continuous embeddings rather than
discrete tokens sidesteps the need for learned discretisation and makes the
DDPM math straightforward.  The trade-off is that decoding back to amino acids
is approximate (nearest-neighbour in the embedding table).

**Synthetic affinity** — the `BindingAffinityPredictor` is trained on a
deterministic synthetic function of amino acid composition (cysteine, tryptophan,
and charged residue content).  Swap in your own experimental labels for
real-world use.

**Scale** — this is a proof-of-concept.  For production protein design consider
[RFdiffusion](https://github.com/RosettaCommons/RFdiffusion),
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN), or
[ESM](https://github.com/facebookresearch/esm) as starting points.

---

## License

MIT
