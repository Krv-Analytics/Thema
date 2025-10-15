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

import logging
import warnings

# Suppress sklearn deprecation warnings about force_all_finite -> ensure_all_finite
# This affects dependencies like kmapper and hdbscan that haven't updated yet
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
    module="sklearn.*"
)


def enable_logging(level="INFO"):
    """
    Enable thema logging for interactive use (e.g., notebooks).

    Parameters
    ----------
    level : str, optional
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        Default is 'INFO' for moderate verbosity.
        Use 'DEBUG' for detailed operational info.
        Use 'WARNING' for warnings and errors only.
        Use 'ERROR' for errors only.

    Examples
    --------
    >>> import thema
    >>> thema.enable_logging('DEBUG')  # Detailed logging
    >>> thema.enable_logging('INFO')   # Moderate logging
    >>> thema.enable_logging('WARNING')  # Warnings/errors only
    """
    thema_logger = logging.getLogger("thema")

    for handler in thema_logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            thema_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    thema_logger.addHandler(handler)

    # Set level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    thema_logger.setLevel(log_level)

    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("thema."):
            child_logger = logging.getLogger(name)
            # Reset level to NOTSET so parent logger controls the level
            child_logger.setLevel(logging.NOTSET)

    thema_logger.propagate = False

    print(f"Thema logging enabled at {level.upper()} level")


def disable_logging():
    """
    Disable thema logging (return to quiet mode).

    Examples
    --------
    >>> import thema
    >>> thema.disable_logging()
    """
    thema_logger = logging.getLogger("thema")

    for handler in thema_logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            thema_logger.removeHandler(handler)

    thema_logger.setLevel(logging.ERROR)

    print("Thema logging disabled (errors only)")


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
    "enable_logging",
    "disable_logging",
]
