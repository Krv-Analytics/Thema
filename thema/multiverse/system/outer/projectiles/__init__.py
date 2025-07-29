"""
Projectiles Module.

This module contains different projection algorithms for data dimensionality reduction.
"""

from .pcaProj import pcaProj
from .tsneProj import tsneProj
from .umapProj import umapProj


__all__ = ["pcaProj", "tsneProj", "umapProj"]
