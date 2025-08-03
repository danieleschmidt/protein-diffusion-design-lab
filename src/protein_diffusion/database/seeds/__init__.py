"""
Database seed data for protein diffusion design lab.

This module provides seed data for development and testing,
including sample proteins, structures, and experiments.
"""

from .protein_seeds import seed_proteins
from .experiment_seeds import seed_experiments

__all__ = ['seed_proteins', 'seed_experiments']