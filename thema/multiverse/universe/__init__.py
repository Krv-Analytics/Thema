"""
Universe Module.

This module contains components for model creation and selection.
"""

# Empty init file to avoid circular imports
# Users should import specific classes directly from their modules

from .galaxy import Galaxy
from .star import Star
from .geodesics import stellar_curvature_distance


__all__ = [
    "Galaxy",
    "Star",
    "stellar_curvature_distance",
]
