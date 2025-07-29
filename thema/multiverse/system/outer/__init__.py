"""
Outer System Module.

This module contains components for data dimension reduction and projection:
- Comet: For dimension reduction algorithms (PCA, t-SNE, UMAP)
- Oort: For coordinating multiple Comet instances for dimension reduction
"""

# Import main classes for easier access from parent modules
from .comet import Comet
from .oort import Oort

__all__ = ["Comet", "Oort"]
