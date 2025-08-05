"""
Thema: Topological Hyperparameter Evaluation and Mapping Algorithm!

Thema is a package for topological data analysis and hyperparameter optimization.
It provides a unified pipeline for data cleaning, dimension reduction, and
topological analysis.

Main Components:
---------------
- Thema: The main entry point for the package
- Core: Base class for data management
- Moon: For data cleaning and transformation
- Planet: For coordinating multiple Moon instances
- Comet: For dimension reduction
- Oort: For coordinating multiple Comet instances
- Star: Base class for topological data analysis
- Galaxy: For coordinating multiple Star instances
- starGraph: For storing and manipulating topological graphs
"""

# Import key components for easier access
from .core import Core
from .multiverse.system.inner import Moon, Planet
from .multiverse.system.outer import Comet, Oort
from .multiverse.universe.galaxy import Galaxy
from .multiverse.universe.star import Star
from .multiverse.universe.utils.starGraph import starGraph
from .thema import Thema

# Package metadata
__version__ = "0.1.3"
__author__ = "Krv-Analytics"

__all__ = [
    "Thema",
    "Core",
    "Moon",
    "Planet",
    "Comet",
    "Oort",
    "Star",
    "Galaxy",
    "starGraph",
]
