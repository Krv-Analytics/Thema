"""Object file for JMapper."""
import kmapper as km
import networkx as nx
import numpy as np
from hdbscan import HDBSCAN
from kmapper import KeplerMapper

from modeling.nammu.curvature import ollivier_ricci_curvature
from modeling.nammu.topology import PersistenceDiagram, calculate_persistence_diagrams
from modeling.nammu.utils import make_node_filtration 
from modeling.tupper import Tupper


class JMapper:
    """A new spin on scikit-tda's `KMapper` using Graph Curvature
    and Persistent Homology.

    This class allows you to generate graph models of high dimensional data
    based on Singh et al.'s Mapper algorithm. More importantly, the class
    can compute descriptions of these models on novel graph metrics that
    combine discrete curvature and persistent homology.
    """

    def __init__(
        self,
        tupper: Tupper,
        verbose: int = 0,
    ):
        """Constructor for JMapper class.
        Parameters
        -----------
        Tupper: <tupper.Tupper>
            A data container that holds raw, cleaned, and projected
            versions of user data.

        verbose: int, default is 0
            Logging level passed through to `kmapper`. Levels (0,1,2)
            are supported.
        """
        # User Inputs
        self._tupper = tupper

        # Initialize Mapper
        self._mapper = KeplerMapper(verbose=verbose)

        # Inputs for `fit`
        self._clusterer = dict()
        self._cover = None

        # Analysis Objects
        self._complex = dict()
        self._graph = nx.Graph()
        self._min_intersection = None
        self._components = dict()
        self._curvature = np.array([])
        self._diagram = PersistenceDiagram()

        # Number of Policy Group
        self._num_policy_groups = None

    @property
    def tupper(self):
        """Return the data container Tupper used to initialize JMapper."""
        return self._tupper

    @property
    def mapper(self):
        """Return the scikit-tda object generated when executing
        the Mapper algorithm."""
        return self._mapper

    @property
    def cover(self):
        """Return the cover used to fit JMapper."""
        return self._cover

    @property
    def clusterer(self):
        """Return the clusterer used to fit JMapper."""
        return self._clusterer

    @property
    def complex(self):
        """Return the clusterer used to fit JMapper."""
        if len(self._complex["nodes"]) == 0:
            try:
                self.fit(clusterer=self.clusterer)
            except self._complex == dict():
                print("Your simplicial complex is empty!")
                print(
                    "Note: some parameters may produce a trivial\
                    mapper representation. \n"
                )
        return self._complex

    @property
    def graph(self):
        if len(self._graph.nodes()) == 0:
            try:
                self.to_networkx(self.min_intersection)
            except:
                self._complex == dict()
                print("Your simplicial complex is empty!")
                print(
                    "Note: some parameters may produce a trivial mapper representation. \n"
                )
        return self._graph

    @property
    def min_intersection(self):
        if self._min_intersection is None:
            print(
                "Please choose a minimum intersection \
                to generate a networkX graph!"
            )
        return self._min_intersection

    @property
    def components(self):
        if len(self._components) == 0:
            try:
                self.connected_components()
            except self._complex == dict():
                print(
                    "Connected components could not be obtained \
                    from this simplicial complex!"
                )
                print(
                    "Note: some parameters may produce a trivial\
                    mapper representation. \n"
                )
        return self._components

    @property
    def num_policy_groups(self):
        if self._num_policy_groups is None:
            try:
                self.connected_components()
            except self.complex == dict():
                print(
                    "Number of policy groups could not be \
                        obtained from this simplicial complex!"
                )
                print(
                    "Note: some parameters may produce a trivial\
                    mapper representation. \n"
                )

        return self._num_policy_groups

    @property
    def curvature(self):
        """Return the curvature values for the graph of a JMapper object."""
        assert (
            len(self._curvature) > 0
        ), "You don't have any edge curvatures! Try running `to_networkx`"
        return self._curvature

    @curvature.setter
    def curvature(self, curvature_fn=ollivier_ricci_curvature):
        """Setter function for curvature.

        Parameters
        -----------
        curvature_fn: func
            The method for calculating discrete curvature of the graph.
            The default is set to Ollivier-Ricci Curvature.

        """
        assert (
            len(self._graph.nodes()) > 0
        ), "First run `to_networkx` to generate a non-empty networkx graph."

        try:
            curvature = curvature_fn(self.graph)
            assert len(curvature) == len(self._graph.edges())
            self._curvature = curvature
        except len(curvature) != len(self._graph.edges()):
            print("Invalid Curvature function")

    @property
    def diagram(self):
        """Return the persistence diagram based on curvature filtrations
        associated with JMapper graph."""
        if self._diagram is None:
            try:
                self.calculate_homology()
            except self.complex == dict():
                print(
                    "Persistence Diagrams could not be obtained\
                    from this simplicial complex!"
                )
        return self._diagram

    def fit(
        self,
        n_cubes: int = 6,
        perc_overlap: float = 0.4,
        clusterer=HDBSCAN(min_cluster_size=6),
    ):
        """
        Apply scikit-tda's implementation of the Mapper algorithm.
        Returns a dictionary that summarizes the fitted simplicial complex.

        Parameters
        -----------
        n_cubes: int, defualt 6
            Number of cubes used to cover of the latent space.
            Used to construct a kmapper.Cover object.

        perc_overlap: float, default 0.4
            Percentage of intersection between the cubes covering
            the latent space. Used to construct a kmapper.Cover object.

        clusterer: default is HDBSCAN
            Scikit-learn API compatible clustering algorithm.
            Must provide `fit` and `predict`.

        Returns
        -----------
        complex : dict
            A dictionary with "nodes", "links" and "meta"
            information of a simplicial complex.

        """
        # Log cover and clusterer from most recent fit
        self.n_cubes = n_cubes
        self.perc_overlap = perc_overlap
        self._cover = km.Cover(n_cubes, perc_overlap)
        self._clusterer = clusterer

        projection = self.tupper.projection
        # Compute Simplicial Complex
        self._complex = self._mapper.map(
            lens=projection,
            X=projection,
            cover=self.cover,
            clusterer=self.clusterer,
        )

        return self._complex

    def to_networkx(self, min_intersection: int = 1):
        """
        Converts a complex into a networkx graph. Generates the `graph`
        attribute for JMapper. A simplicial complex must already be
        computed to use this function.

        Parameters
        -----------
        min_intersection: int, default is 1
            Minimum intersection considered when computing the graph.
            An edge will be created only when the intersection between
            two nodes is greater than or equal to `min_intersection`.

        Returns
        -----------
        nx.Graph
            A networkx graph based on a Kepler Mapper simplicial complex.
            Nodes determined by clusters and edges based on `min_intersection`.

        """
        # Initialize Nerve
        self._min_intersection = min_intersection
        nerve = km.GraphNerve(min_intersection)
        assert (
            len(self._complex["nodes"]) > 0
        ), "You must first generate a non-empty Simplicial Complex \
        with `fit()` before you can convert to Networkx "

        nodes = self._complex["nodes"]
        _, simplices = nerve.compute(nodes)
        edges = [edge for edge in simplices if len(edge) == 2]

        # Setting self._graph
        self._graph = nx.Graph()
        self._graph.add_nodes_from(nodes)
        nx.set_node_attributes(
            self._graph,
            dict(self.complex["nodes"]),
            "membership",
        )
        self._graph.add_edges_from(edges)

        return self._graph

    def connected_components(self):
        """
        Compute the connected components of `self._graph`

        Returns
        -----------
        components: dict
            A dictionary labeling the connected components of `self._graph`.
            Keys are networkX Graphs and items are integer labels.

        """
        assert (
            len(self.graph.nodes()) > 0
        ), "First run `to_networkx` to generate a non-empty networkx graph."

        self._components = dict(
            [
                (self.graph.subgraph(c).copy(), i)
                for i, c in enumerate(nx.connected_components(self.graph))
            ]
        )
        self._num_policy_groups = len(self._components)
        return self.components

    def calculate_homology(
        self,
        filter_fn=ollivier_ricci_curvature,
        use_min=True,
    ):
        """Compute Persistent Diagrams based on a curvature
        filtration of `self._graph`.

        Parameters
        -----------
        filter_fn: func
            The method for calculating discrete curvature of the graph.
            The default is set to Ollivier-Ricci.

        use_min: bool
            Sequence of edge values. Depending on the `use_min` parameter,
            either the minimum of all edge values or the maximum of all edge
            values is assigned to a vertex.


        Returns
        -----------
        persistence_diagram: src.modeling.nammu.topology.PersistenceDiagram
            An array of tuples (b,d) that represent the birth and death of
            homological features in your graph according to the provided
            filtration function.


        """
        assert (
            len(self.graph.nodes()) > 0
        ), "First run `to_networkx` to generate a non-empty networkx graph."

        if len(self._curvature) > 0:
            self.curvature = filter_fn  # Set curvatures

        G = make_node_filtration(
            self.graph,
            self.curvature,
            attribute_name="curvature",
            use_min=use_min,
        )
        pd = calculate_persistence_diagrams(
            G,
            "curvature",
            "curvature",
        )
        self._diagram = pd
        return self.diagram

    def item_lookup(
        self,
        index: int,
    ):
        """
        For an item in your dataset, find the subgraph and clusters
        that contain the item.

        Parameters
        -----------
        index: int
            identifier for an item in `self.data`.

        Returns
        -----------
        clusters: dict
            A dict of clusters that contain `item`. Keys are cluster labels,
            and values are cluster items.
        subgraph: list
            A subgraph made up of the connected componnets
            generated by clusters. In most cases this is
            the connected component that contains the item.

        """
        assert (
            len(self.complex) > 0
        ), "You must first generate a Simplicial Complex with `fit()` \
            before you perform `item_lookup`."

        clusters = {}

        all_clusters = self.graph.nodes()

        subgraph_nodes = set()

        for cluster in all_clusters:
            elements = self.complex["nodes"][cluster]
            if index in elements:
                clusters[cluster] = elements

                # Note: for min_intersection >1 it is possible that item may
                # lie in clusters spread across different components.

                subgraph_nodes = set.union(
                    *[
                        nx.node_connected_component(self.graph, node)
                        for node in clusters.keys()
                    ]
                )

        subgraph = self.graph.subgraph(subgraph_nodes)

        return clusters, subgraph
