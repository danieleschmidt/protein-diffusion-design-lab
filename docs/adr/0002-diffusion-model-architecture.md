# ADR-0002: Diffusion Model Architecture Choice

## Status
Accepted

## Context
The protein diffusion design lab requires a foundational deep learning architecture for protein scaffold generation. Key considerations include:

- Model expressiveness for complex protein geometries
- Computational efficiency for real-time generation
- Training stability with limited protein structure data
- Integration with existing molecular representation standards

## Decision
We will implement a 1B parameter transformer-based diffusion model with the following specifications:

### Architecture Components
1. **Backbone**: Transformer with rotary positional embeddings (RoPE)
2. **Diffusion Process**: DDPM (Denoising Diffusion Probabilistic Models)
3. **Conditioning**: Classifier-free guidance for motif conditioning
4. **Tokenization**: SELFIES-based molecular representation

### Technical Specifications
- **Model Size**: 1B parameters (24 layers, 1024 hidden dimension)
- **Attention**: Multi-head attention with 16 heads
- **Sequence Length**: Maximum 512 amino acid positions
- **Training**: Mixed precision (FP16) with gradient checkpointing

## Consequences

### Positive
- **Expressiveness**: Large parameter count enables complex protein fold modeling
- **Efficiency**: Transformer architecture allows parallel processing
- **Stability**: DDPM provides stable training dynamics
- **Extensibility**: Architecture supports future conditioning mechanisms

### Negative
- **Memory Requirements**: 16GB+ VRAM needed for inference
- **Training Cost**: Significant computational resources required
- **Model Size**: Large checkpoint files (4GB+) impact deployment

### Neutral
- **Dependency**: Requires PyTorch 2.0+ for optimal performance
- **Hardware**: CUDA-capable GPUs required for practical usage

## Alternatives Considered

### 1. VAE-Based Generation
- **Pros**: Smaller model size, faster inference
- **Cons**: Lower generation quality, mode collapse issues
- **Verdict**: Rejected due to quality limitations

### 2. GAN-Based Architecture  
- **Pros**: Fast generation, high-quality samples
- **Cons**: Training instability, limited conditioning control
- **Verdict**: Rejected due to training difficulties

### 3. Flow-Based Models
- **Pros**: Exact likelihood computation, bidirectional generation
- **Cons**: Complex architecture, slower sampling
- **Verdict**: Rejected due to implementation complexity

### 4. Smaller Transformer Models (100M-500M parameters)
- **Pros**: Lower resource requirements, faster training
- **Cons**: Reduced model capacity for complex protein structures
- **Verdict**: Rejected - quality vs. efficiency tradeoff favors larger model

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM framework
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE implementation
- [SELFIES: A Robust Representation of Semantically Constrained Graphs](https://arxiv.org/abs/1905.13741) - Molecular tokenization
- [ProtDiff: Diffusion Model for Protein Backbone Generation](https://arxiv.org/abs/2206.04119) - Protein-specific diffusion approach