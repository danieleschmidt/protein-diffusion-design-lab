"""
Protein structure prediction and folding evaluation.

This module provides interfaces for structure prediction using ESMFold
and ColabFold, along with quality assessment and validation tools.
"""

from .structure_predictor import StructurePredictor, StructurePredictorConfig

__all__ = ['StructurePredictor', 'StructurePredictorConfig']