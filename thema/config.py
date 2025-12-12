# File: multiverse/config.py
# Last Updated: 05/15/24
# Updated By: JW


from dataclasses import dataclass, field
from typing import List, Union


# ╭────────────────────────────────╮
# │    Config Data Classes         |
# ╰────────────────────────────────╯
# umapProjConfig has been removed in favor of tsneProjConfig


@dataclass
class tsneProjConfig:
    """
    Configuration class for a t-SNE projectile.

    Activated when exploring the outer system using `Comet` or `Oort`,
    this class is used to configure the parameters to project your data
    into a lower-dimensional space using the t-SNE algorithm.

    Parameters
    ----------
    name : str, optional
        The name of the projection. Default is "tsneProj".
    module : str
        The module where the projection is implemented.
    perplexity : int, optional
        The perplexity parameter for t-SNE. Default is 4.
    dimensions : int, optional
        The number of dimensions in the projection. Default is 2.
    seed : int, optional
    """

    name: str = "tsneProj"
    module: str = "thema.multiverse.system.outer.projectiles.tsneProj"
    perplexity: int = 4
    dimensions: int = 2
    seed: int = 42


@dataclass
class pcaProjConfig:
    """
    Configuration class for a PCA projection.

    Activated when exploring the outer system using `Comet` or `Oort`,
    this class is used to configure the parameters to project your data
    into a lower-dimensional space using principal component analysis.

    Parameters
    ----------
    name : str, optional
        The name of the projection. Default is "pcaProj".
    module : str
        The module where the projection is implemented.
    dimensions : int, optional
        The number of dimensions in the projection. Default is 2.
    seed : int, optional
        The seed for the random number generator. Default is 42.
    """

    name: str = "pcaProj"
    module: str = "thema.multiverse.system.outer.projectiles.pcaProj"
    dimensions: int = 2
    seed: int = 42


@dataclass
class jmapStarConfig:
    """
    Configuration class for a jMAP Star.

    Activated when exploring a universe using `Galaxy`,
    this class used to configure the parameters to project your data
    into a lower-dimensional space using principal component analysis.

    Parameters
    ----------
    name : str, optional
        The name of the star. Default is "jmapStar".
    module : str
        The module where the star is implemented.
    nCubes : int, optional
        The number of cubes. Default is 5.
    percOverlap : float, optional
        The percentage of overlap. Default is 0.3.
    minIntersection : int, optional
        The minimum intersection. Default is -1.
    clusterer : Union[List[str], List[dict]], optional
        The clusterer configuration. Default is HDBSCAN with specific parameters.
    """

    name: str = "jmapStar"
    module: str = "thema.multiverse.universe.stars.jmapStar"
    nCubes: int = 5
    percOverlap: float = 0.3
    minIntersection: int = -1
    clusterer: Union[List[str], List[dict]] = field(
        default_factory=lambda: [
            "HDBSCAN",
            {
                "min_cluster_size": 5,
                "cluster_selection_epsilon": 0.0,
                "max_cluster_size": None,
            },
        ]
    )


@dataclass
class pyballStarConfig:
    """
    Configuration class for a PyBall Mapper star.

    Parameters
    ----------
    name : str, optional
        The name of the star. Default is "pyballStar".
    module : str
        The module where the star is implemented.
    EPS : float, optional
        Epsilon parameter passed to BallMapper. Default is 0.1.
    """

    name: str = "pyballStar"
    module: str = "thema.multiverse.universe.stars.pyballStar"
    EPS: float = 0.1


@dataclass
class gudhiStarConfig:
    """
    Configuration class for a Gudhi Mapper star.

    Parameters
    ----------
    name : str, optional
        The name of the star. Default is "gudhiStar".
    module : str
        The module where the star is implemented.
    clusterer : Union[List[str], List[dict]], optional
        The clusterer configuration.
    N : int, optional
        Subsampling iterations for estimating cover parameters. Default is 100.
    beta : float, optional
        Exponent parameter for estimating cover parameters. Default is 0.0.
    C : float, optional
        Constant parameter for estimating cover parameters. Default is 10.0.
    """

    name: str = "gudhiStar"
    module: str = "thema.multiverse.universe.stars.gudhiStar"
    clusterer: Union[List[str], List[dict]] = field(
        default_factory=lambda: [
            "HDBSCAN",
            {
                "min_cluster_size": 5,
                "cluster_selection_epsilon": 0.0,
                "max_cluster_size": None,
            },
        ]
    )
    N: int = 100
    beta: float = 0.0
    C: float = 10.0


# ╭────────────────────────────────╮
# │    Class Maps                  |
# ╰────────────────────────────────╯

# Map from YAML tags to projector configuration classes
projector_tag_to_config = {
    "tsne": "tsneProjConfig",
    "pca": "pcaProjConfig",
}

# Backwards compatibility alias (scheduled for removal)
tag_to_class = projector_tag_to_config

# Map from star class names to configuration dataclasses
star_name_to_config = {
    "jmapStar": "jmapStarConfig",
    "pyballStar": "pyballStarConfig",
    "gudhiStar": "gudhiStarConfig",
}
# Map from star to observatory class
star_to_observatory = {
    "jmapStar": "jmapObservatoryConfig",
}

# Map from filter YAML tags to filter functions and their parameter names
filter_configs = {
    "component_count": {
        "function": "component_count_filter",
        "params": {"target_components": 1},
    },
    "component_count_range": {
        "function": "component_count_range_filter",
        "params": {"min_components": 1, "max_components": 10},
    },
    "minimum_nodes": {
        "function": "minimum_nodes_filter",
        "params": {"min_nodes": 3},
    },
    "minimum_edges": {
        "function": "minimum_edges_filter",
        "params": {"min_edges": 2},
    },
    "minimum_unique_items": {
        "function": "minimum_unique_items_filter",
        "params": {"min_unique_items": 10},
    },
}
