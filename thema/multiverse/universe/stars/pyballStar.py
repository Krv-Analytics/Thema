# File: multiverse/universe/stars/pyballStar.py
# Last Updated: 11-19-25
# Updated By: JW

import networkx as nx
from pyballmapper import BallMapper

from ..star import Star
from ..utils.starGraph import starGraph
from ..utils.starHelpers import (
    convert_keys_to_alphabet,
    mapper_pseudo_laplacian,
    mapper_unclustered_items,
)


def initialize():
    """Returns pyballStar class from module."""
    return pyballStar


class pyballStar(Star):
    """
    PyBall Mapper Star Class

    Generates a graph representation of projection using PyBall Mapper.

    See: https://github.com/dioscuri-tda/pyBallMapper

    Members
    ------
    data: pd.DataFrame
        a pandas dataframe of raw data
    clean: pd.DataFrame
        a pandas dataframe of complete, scaled, and encoded data
    projection: np.narray
        a numpy array containing projection coordinates
    EPS: float
        epsilon parameter for BallMapper
    mapper: pyballmapper.BallMapper
        a BallMapper object
    starGraph: thema.multiverse.universe.starGraph class
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
        returns list of unclustered items
    save() -> None
        Saves object as a .pkl file.

    """

    def __init__(self, data_path, clean_path, projection_path, EPS=0.1):

        super().__init__(
            data_path=data_path,
            clean_path=clean_path,
            projection_path=projection_path,
        )
        self.EPS = EPS

    def fit(self):
        self.mapper = BallMapper(X=self.projection, eps=self.EPS, verbose=False)
        graph = self.mapper.Graph
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
        self.starGraph = starGraph(graph)

    def get_pseudoLaplacian(self, neighborhood="node"):
        """Calculates and returns a pseudo laplacian n by n matrix representing neighborhoods in the graph. Here, n corresponds to
        the number of items (ie rows in the clean data - keep in mind some raw data rows may have been dropped in cleaning). Here,
        the diagonal element A_ii represents the number of neighborhoods item i appears in. The element A_ij represent the number of
        neighborhoods both item i and j belong to.

        Parameters
        ----------
        neighborhood: str
            Specifies the type of neighborhood. For pyballStar, neighborhood options are 'node' or 'cc'
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
