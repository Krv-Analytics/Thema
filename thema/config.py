# File: multiverse/config.py
# Last Updated: 05/15/24
# Updated By: JW


from dataclasses import dataclass, field
from typing import List, Union


# ╭────────────────────────────────╮
# │    Config Data Classes         |
# ╰────────────────────────────────╯
@dataclass
class umapProjConfig:
    """
    Configuration class for a UMAP projectile.

    Activated when exploring the outer system using `Comet` or `Oort`,
    this class is used to configure the parameters to project your data
    into a lower-dimensional space using the UMAP algorithm.

    Parameters
    ----------
    name : str, optional
        The name of the projection. Default is "umapProj".
    module : str
        The module where the projection is implemented.
    nn : int, optional
        The number of nearest neighbors to consider. Default is 4.
    minDist : float, optional
        The minimum distance between points in the projection. Default is 0.
    dimensions : int, optional
        The number of dimensions in the projection. Default is 2.
    seed : int, optional
        The seed for the random number generator. Default is 42.
    """

    name: str = "umapProj"
    module: str = "thema.multiverse.system.outer.projectiles.umapProj"
    nn: int = 4
    minDist: float = 0
    dimensions: int = 2
    seed: int = 42


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
class jmapObservatoryConfig:
    """
    Configuration class for a jMAP Observatory.

    Activated when probing a universe using `Telescope`,
    this class used to configure data querying and visualization
    for jMAP stars.

    Parameters
    ----------
    name : str, optional
        The name of the observatory. Default is "jmapObservatory".
    module : str
        The module where the observatory is implemented.
    """

    name: str = "jmapObservatory"
    module: str = "thema.probe.observatories.jmapObservatory"


# ╭────────────────────────────────╮
# │    Class Maps                  |
# ╰────────────────────────────────╯

# Map from YAML tags to class names
tag_to_class = {
    "umap": "umapProjConfig",
    "tsne": "tsneProjConfig",
    "pca": "pcaProjConfig",
    "jmap": "jmapStarConfig",
}
# Map from star to observatory class
star_to_observatory = {
    "jmapStar": "jmapObservatoryConfig",
}
