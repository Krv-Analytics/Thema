# File: multiverse/universe/stars/jmapStar.py
# Last Update: 05/15/24
# Updated by: JW


import logging


import networkx as nx
from kmapper import Cover, KeplerMapper


from ..star import Star
from ..utils.starGraph import starGraph
from ..utils.starHelpers import (
    convert_keys_to_alphabet,
    mapper_pseudo_laplacian,
    mapper_unclustered_items,
    get_clusterer,
    Nerve,
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def initialize():
    """
    Returns jmapStar class from module.This is a general method that allows
    us to initialize arbitrary star objects.

    Returns
    -------
    jmapStar : object
        The jMAP projectile object.
    """
    return jmapStar


class jmapStar(Star):
    """
    JMAP Star Class

    Our custom implementation of a Kepler Mapper (K-Mapper) into a Star object.
    Here we allow users to explore the topological structure of their data
    using the Mapper algorithm, which is a powerful tool for visualizing
    high-dimensional data.


    ----------
    - inherts from Star

    Generates a graph representation of projection using Kepler Mapper.

    Members
    ------
    data: pd.DataFrame
        a pandas dataframe of raw data
    clean: pd.DataFrame
        a pandas dataframe of complete, scaled, and encoded data
    projection: np.narray
        a numpy array containing projection coordinates
    nCubes: int
        kmapper paramter relating to covering of space
    percOverlap: float
       kmapper paramter relating to covering of space
    minIntersection: int
        number of shared items required to define an edge. Set to -1
        to create a weighted graph.
    clusterer: function
        Clustering function passed to kmapper (e.g. HDBSCAN).
    mapper: kmapper.mapper
        A kmapper mapper object.
    complex: dict
        A dictionary specifying node membership
    starGraph: thema.multiverse.universe.utils.starGraph class
        An expanded framework for analyzing networkx graphs

    Functions
    --------
    get_data_path() -> str
        returns path to raw data
    get_clean_path() -> str
        returns path to Moon object containing clean data
    get_projection_path()-> str
        returns path to Comet object contatining projection data
    fit() -> None
        Computes a complex and corresponding starGraph
    get_unclustered_items() -> list
        returns list of unclustered items from HDBSCAN
    save() -> None
        Saves object as a .pkl file.

    """

    def __init__(
        self,
        data_path: str,
        clean_path: str,
        projection_path: str,
        nCubes: int,
        percOverlap: float,
        minIntersection: int,
        clusterer: list,
    ):
        """
        Constructs an instance of jmapStar

        Parameters
        ---------
        data_path : str
            A path to the raw data file.
        clean_path : str
            A path to a cofigured Moon object file.
        projection_path : str
            A path to a configured Comet object file.
        nCubes: int
            Number of cubes used in kmapper cover.
        percOverlap: float
            Percent of cube overlap in kmapper cover.
        minIntersection: int
            Number of shared items across nodes to define an edge. Note: set
            to -1 for a weighted graph.
        clusterer: list
            A length 2 list containing in position 0 the name of the clusterer, and
            in position 1 the parameters to configure it.
            *Example*
            clusterer = ["HDBSCAN", {"minDist":0.1}]
        """
        super().__init__(
            data_path=data_path,
            clean_path=clean_path,
            projection_path=projection_path,
        )
        self.nCubes = nCubes
        self.percOverlap = percOverlap
        self.minIntersection = minIntersection
        self.clusterer = get_clusterer(clusterer)
        self.mapper = KeplerMapper()
        self.complex = None

        # Store parameters for potential debugging
        self._params = {
            "nCubes": nCubes,
            "percOverlap": percOverlap,
            "minIntersection": minIntersection,
            "clusterer": clusterer,
        }

    def fit(self):
        """Computes a kmapper complex based on the configuration parameters and
        constructs a resulting graph.

        Returns
        ------
        None
            Initializes complex and starGraph members

        Warning
        ------
        Particular combinations of parameters can result in empty graphs or
        empty complexes.

        """
        self.complex = self.mapper.map(
            lens=self.projection,
            X=self.projection,
            cover=Cover(self.nCubes, self.percOverlap),
            clusterer=self.clusterer,
        )

        if not self.complex or "nodes" not in self.complex:
            logger.debug(
                f"KeplerMapper produced empty complex - params: {self._params}, "
                f"projection shape: {self.projection.shape}"
            )
            self.complex = None
            self.starGraph = None
            return

        self.nodes = convert_keys_to_alphabet(self.complex["nodes"])

        graph = nx.Graph()
        nerve = Nerve(minIntersection=self.minIntersection)

        # Fit Nerve to generate edges
        self.edges = nerve.compute(self.nodes)

        if len(self.edges) == 0:
            # Log when we get empty graphs - this is important for debugging
            logger.debug(
                f"No edges found in graph - params: {self._params}, "
                f"nodes: {len(self.nodes)}, projection shape: {self.projection.shape}"
            )
            self.starGraph = starGraph(
                graph
            )  # Create empty graph instead of None
        else:
            graph.add_nodes_from(self.nodes)
            nx.set_node_attributes(graph, self.nodes, "membership")

            if self.minIntersection == -1:
                graph.add_weighted_edges_from(self.edges)
            else:
                graph.add_edges_from(self.edges)

            self.starGraph = starGraph(graph)

    def get_pseudoLaplacian(self, neighborhood="node"):
        """Calculates and returns a pseudo laplacian n by n matrix representing neighborhoods in the graph. Here, n corresponds to
        the number of items (ie rows in the clean data - keep in mind some raw data rows may have been dropped in cleaning). Here,
        the diagonal element A_ii represents the number of neighborhoods item i appears in. The element A_ij represent the number of
        neighborhoods both item i and j belong to.

        Parameters
        ----------
        neighborhood: str
            Specifies the type of neighborhood. For jmapStar, neighborhood options are 'node' or 'cc'
        """
        if self.complex is None:
            self.fit()

        return mapper_pseudo_laplacian(
            complex=self.complex,
            n=len(self.clean),
            components=self.starGraph.components,
            neighborhood=neighborhood,
        )

    def get_unclustered_items(self):
        """
        Returns the list of items that were not clustered in the
        mapper fitting.

        Returns
        -------
        self._unclustered_item : list
           A list of unclustered item ids
        """
        return mapper_unclustered_items(len(self.clean), self.nodes)
