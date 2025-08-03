"""
Protein Diffusion Design Lab

A plug-and-play diffusion pipeline for protein scaffolds that rivals commercial suites.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "your.email@example.com"

# Core API imports
from .diffuser import ProteinDiffuser, ProteinDiffuserConfig
from .ranker import AffinityRanker, AffinityRankerConfig
from .models import DiffusionTransformer, DDPM, DiffusionTransformerConfig, DDPMConfig
from .tokenization import SELFIESTokenizer, ProteinEmbeddings, TokenizerConfig, EmbeddingConfig
from .folding import StructurePredictor, StructurePredictorConfig

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Main interfaces
    "ProteinDiffuser",
    "AffinityRanker",
    # Core models
    "DiffusionTransformer",
    "DDPM",
    # Tokenization
    "SELFIESTokenizer", 
    "ProteinEmbeddings",
    # Structure prediction
    "StructurePredictor",
    # Configurations
    "ProteinDiffuserConfig",
    "AffinityRankerConfig",
    "DiffusionTransformerConfig",
    "DDPMConfig",
    "TokenizerConfig",
    "EmbeddingConfig",
    "StructurePredictorConfig",
]