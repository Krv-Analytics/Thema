"""
Multiverse Module.

The core functionality of Thema, containing system and universe components:

Inner System:
- Moon: For data cleaning and transformation
- Planet: For coordinating multiple Moon instances to handle data imputation and preprocessing

Outer System:
- Comet: For dimension reduction algorithms
- Oort: For coordinating multiple Comet instances for dimension reduction

Universe:
- Star: Base class for topological data analysis algorithms
- Galaxy: For coordinating multiple Star instances for topology analysis
- starGraph: For storing and manipulating topological graphs
"""

# Import primary components for easier access
from .system.inner.moon import Moon
from .system.inner.planet import Planet
from .system.outer.comet import Comet
from .system.outer.oort import Oort
from .universe.galaxy import Galaxy
from .universe.star import Star
from .universe.utils.starGraph import starGraph

__all__ = ["Moon", "Planet", "Comet", "Oort", "Galaxy", "Star", "starGraph"]
