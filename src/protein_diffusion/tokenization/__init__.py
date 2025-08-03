"""
Tokenization and molecular representation for protein sequences.

This module provides SELFIES-based tokenization and embedding systems
for robust protein sequence representation in diffusion models.
"""

from .selfies_tokenizer import SELFIESTokenizer
from .protein_embeddings import ProteinEmbeddings

__all__ = ['SELFIESTokenizer', 'ProteinEmbeddings']