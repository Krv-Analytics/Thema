# File: multiverse/universe/stars/jmapStar.py
# Last Update: 05/15/24
# Updated by: JW


import itertools
from collections import defaultdict

import networkx as nx
from hdbscan import HDBSCAN
from kmapper import Cover, KeplerMapper
from sklearn.cluster import DBSCAN

from ..star import Star
from ..utils.starGraph import starGraph


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
        try:
            self.complex = self.mapper.map(
                lens=self.projection,
                X=self.projection,
                cover=Cover(self.nCubes, self.percOverlap),
                clusterer=self.clusterer,
            )
            self.nodes = convert_keys_to_alphabet(self.complex["nodes"])

            graph = nx.Graph()
            nerve = Nerve(minIntersection=self.minIntersection)

            # Fit Nerve to generate edges
            edges = nerve.compute(self.nodes)

            if len(edges) == 0:
                self.starGraph = None
            else:
                graph.add_nodes_from(self.nodes)
                nx.set_node_attributes(graph, self.nodes, "membership")

                if self.minIntersection == -1:
                    graph.add_weighted_edges_from(edges)
                else:
                    graph.add_edges_from(edges)

            self.starGraph = starGraph(graph)

        except:
            self.complex = None
            self.starGraph = None

    def get_unclustered_items(self):
        """
        Returns the list of items that were not clustered in the
        mapper fitting.

        Returns
        -------
        self._unclustered_item : list
           A list of unclustered item ids
        """
        N = len(self.clean)
        labels = dict()
        unclustered_items = []
        for idx in range(N):
            place_holder = []
            for node_id in self.nodes.keys():
                if idx in self.nodes[node_id]:
                    place_holder.append(node_id)

            if len(place_holder) == 0:
                place_holder = -1
                unclustered_items.append(idx)
            labels[idx] = place_holder

        return unclustered_items


########################################################################################

# Nerve Class

########################################################################################


class Nerve:
    """
    A class to handle generating weighted graphs from Keppler Mapper Simplicial Complexes.

    Parameters
    ----------
    weighted : bool, optional
        True if you want to generate a weighted graph.
        If False, please specify a `minIntersection`.
    minIntersection : int, optional
        Minimum intersection considered when computing the nerve.
        An edge will be created only when the intersection between
        two nodes is greater than or equal to `minIntersection`.
        Not specifying this parameter will result in an unweighted graph.
    """

    def __init__(self, minIntersection: int = -1):
        self.minIntersection = minIntersection

    def __repr__(self):
        return f"Nerve(minIntersection={self.minIntersection})"

    def compute(self, nodes):
        """
        Compte the nerve of a simplicial complex.

        Parameters
        ----------
        nodes : dict
            A dictionary with entries `{node id}:{list of ids in node}`.

        Returns
        -------
        edges : list
            A 1-skeleton of the nerve (intersecting nodes).

        Examples
        --------
        >>> nodes = {'node1': [1, 2, 3], 'node2': [2, 3, 4]}
        >>> compute(nodes)
        [['node1', 'node2']]
        """
        if self.minIntersection == -1:
            return self.compute_weighted_edges(nodes)
        else:
            return self.compute_unweighted_edges(nodes)

    def compute_unweighted_edges(self, nodes):
        """
        Helper function to find edges of the overlapping clusters.

        Parameters
        ----------
        nodes : dict
            A dictionary with entries `{node id}:{list of ids in node}`.

        Returns
        -------
        edges : list
            A 1-skeleton of the nerve (intersecting nodes).

        simplicies : list
            Complete list of simplices.

        Examples
        --------
        >>> nodes = {'node1': [1, 2, 3], 'node2': [2, 3, 4]}
        >>> compute_unweighted_edges(nodes)
        [['node1', 'node2']]

        """

        result = defaultdict(list)

        # Create links when clusters from different hypercubes have members with the same sample id.
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            # if there are non-unique members in the union
            if (
                len(set(nodes[candidate[0]]).intersection(nodes[candidate[1]]))
                >= self.minIntersection
            ):
                result[candidate[0]].append(candidate[1])

        edges = [[x, end] for x in result for end in result[x]]
        return edges

    def compute_weighted_edges(self, nodes):
        """
        Helper function to find edges of the overlapping clusters.

        Parameters
        ----------
        nodes : dict
            A dictionary with entries `{node id}:{list of ids in node}`.

        Returns
        -------
        edges : list
            A 1-skeleton of the nerve (intersecting nodes).

        simplicies : list
            Complete list of simplices.

        Examples
        --------
        >>> nodes = {'node1': [1, 2, 3], 'node2': [2, 3, 4]}
        >>> compute_weighted_edges(nodes)
        [('node1', 'node2', 0.333)]

        """

        result = []
        # Create links when clusters from different hypercubes have members with the same sample id.
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            # if there are non-unique members in the union
            overlap = len(
                set(nodes[candidate[0]]).intersection(nodes[candidate[1]])
            )
            if overlap > 0:
                result.append(
                    (candidate[0], candidate[1], round(1 / overlap, 3))
                )
        return result


########################################################################################

# Kepler Mapper clustering utility functions

########################################################################################


def get_clusterer(clusterer: list):
    """
    Converts a list configuration to an initialized clusterer.

    Parameters
    ----------
    clusterer: list
        A length 2 list containing in position 0 the name of the clusterer, and
        in position 1 the parameters to configure it.
        *Example*
        clusterer = ["HDBSCAN", {"minDist":0.1}]

    Returns
    -------
    An initialized clustering object
    """
    if clusterer[0] == "HDBSCAN":
        return HDBSCAN(**clusterer[1])

    elif clusterer[0] == "DBSCAN":
        return DBSCAN(**clusterer[1])

    else:
        raise ValueError("Only HDBSCAN and DBSCAN supported at this time.")


def convert_keys_to_alphabet(dictionary):
    """Simple Helper function to make kmapper node labels more readable."""
    base = 26  # Number of letters in the alphabet
    new_dict = {}

    keys = list(dictionary.keys())
    for i, key in enumerate(keys):
        # Calculate the position of each letter in the new key
        position = i
        new_key = ""
        while position >= 0:
            new_key = chr(ord("a") + (position % base)) + new_key
            position = (position // base) - 1

        new_dict[new_key] = dictionary[key]

    return new_dict
