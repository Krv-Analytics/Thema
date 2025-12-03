# File: multiverse/universe/stars/gudhiStar.py
# Last Update: 11-19-25
# Updated by: JW


import networkx as nx
from gudhi.cover_complex import MapperComplex


from ..star import Star
from ..utils.starHelpers import (
    convert_keys_to_alphabet,
    get_clusterer,
    mapper_unclustered_items,
    mapper_pseudo_laplacian,
)
from ..utils.starGraph import starGraph


def initialize():
    """Returns gudhiStar class from module."""
    return gudhiStar


class gudhiStar(Star):
    """
    GUDHI Star Class
    ----------
    - inherits from Star

    Generates a graph representation of projection using gudhi.

    See: https://gudhi.inria.fr/python/latest/cover_complex_sklearn_isk_ref.html

    Members
    ------
    data: pd.DataFrame
        a pandas dataframe of raw data
    clean: pd.DataFrame
        a pandas dataframe of complete, scaled, and encoded data
    projection: np.narray
        a numpy array containing projection coordinates
    clusterer: list
        A list of length 2 containing clusterer name in pos 0, and kwargs in pos 1.
    mapper: gudhi.cover_complex.MapperComplex
        a mapper object
    starGraph: thema.multiverse.universe.starGraph class
        An expanded framework for analyzing networkx graphs

    Functions
    --------
    get_data_path() -> str
        returns path to raw data
    get_clean_path() -> str
        returns path to Moon object containing clean data
    get_projection_path()-> str
        returns path to Comet object containing projection data
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
        clusterer: list,
        N: int = 100,
        beta: float = 0.0,
        C: float = 10.0,
    ):
        """
        Constructs an instance of gudhiStar

        Parameters
        ---------
        data_path : str
            A path to the raw data file.
        clean_path : str
            A path to a configured Moon object file.
        projection_path : str
            A path to a configured Comet object file.
        N: int
             subsampling iterations (default 100) for estimating scale and resolutions.
        beta: float
            exponent parameter (default 0.) for estimating scale and resolutions.
        C: float
            (float) – constant parameter (default 10.) for estimating scale and resolutions.
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
        self.N = N
        self.C = C
        self.beta = beta
        self.clusterer = get_clusterer(clusterer)

        self.mapper = MapperComplex(
            input_type="point cloud",
            clustering=self.clusterer,
        )
        self.starGraph = None
        self.complex = None
        self.nodes = None

    def fit(self, labels=None):
        """Constructs a cosmic Graph using gudhi's MapperComplex.

        Returns
        ------
        None
            Initializes starGraph member

        Warning
        ------
        Particular combinations of parameters can result in empty graphs or
        empty complexes.

        """
        try:
            self.mapper.fit(X=self.projection, filters=self.projection, colors=labels)
            graph = self.mapper.get_networkx(set_attributes_from_colors=bool(labels))
            for u, v in graph.edges():
                graph[u][v]["weight"] = 1
            self.complex = {"nodes": nx.get_node_attributes(graph, "membership")}
            self.nodes = convert_keys_to_alphabet(self.complex["nodes"])
            relabel_map = {
                old: new
                for old, new in zip(self.complex["nodes"].keys(), self.nodes.keys())
            }
            graph = nx.relabel_nodes(graph, relabel_map)
            nx.set_node_attributes(graph, self.nodes, "membership")
            # Update complex to use the new alphabetic keys (use copy to avoid reference issues)
            self.complex["nodes"] = self.nodes.copy()
            if len(graph) == 0:
                raise ValueError("Empty graph")

            else:
                self.starGraph = starGraph(graph)

        except Exception as e:
            print(f"Failed to fit gudhiStar: {e}")
            self.starGraph = None
            self.complex = None
            self.nodes = None

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
        if self.starGraph is None:
            self.fit()

        return mapper_pseudo_laplacian(
            complex=self.complex,
            n=len(self.clean),
            components=self.starGraph.components,
            neighborhood=neighborhood,
        )

    def get_unclustered_items(self):
        """Returns the list of items that were not clustered in the mapper fitting.

        Returns
        -------
        self._unclustered_item : list
           A list of unclustered item ids
        """
        return mapper_unclustered_items(len(self.clean), self.nodes)
