import numpy as np
import networkx as nx
import kmapper as km
import pandas as pd

from kmapper import KeplerMapper


class CoalMapper(KeplerMapper):
    # TODO: Doc String
    def __init__(
        self,
        X: np.array,
        verbose: int = 0,
    ):
        """Constructor for CoalMapper class.
        Parameters
        ===========
        X: np.array
            Dataset features you wish to analyze using the mapper algorithm.

        verbose: int, default is 0
            Logging level. Currently 3 levels (0,1,2) are supported. For no logging, set `verbose=0`. For some logging, set `verbose=1`. For complete logging, set `verbose=2`.
        """

        self.data = X
        self.lens = None
        self.clusterer = None
        self.cover = None
        self.nerve = None
        self.graph = None
        self.components = None

        super(CoalMapper, self).__init__(verbose)

    def compute_mapper(
        self,
        n_cubes: int = 4,
        perc_overlap: float = 0.2,
        projection="l2norm",
        clusterer=None,
    ):
        """
        A wrapper function for kmapper that generates a simplicial complex based on a given lens, cover, and clustering algorithm.

        Parameters
        -----------
        n_cubes: int, defualt 4
            Number of cubes used to cover of the latent space. Used to construct a kmapper.Cover object.
            Only impactful if a cover attribute has not been manually specified.

        perc_overlap: float, default 0.2
            Percentage of intersection between the cubes covering the latent space.
            Used to construct a kmapper.Cover object. Only impactful if a cover attribute has not been manually specified.

        projection: str, default is `l2norm'
            A choice of projection to be used to generate a lens using `fit_transform` inherited method.
            Only impactful if a cover attribute has not been manually specified.

        clusterer: default is DBSCAN
            Scikit-learn API compatible clustering algorithm. Must provide `fit` and `predict`.

        Returns
        -----------
        simplicial_complex : dict
            A dictionary with "nodes", "links" and "meta" information.

        """

        # Create Lens
        if self.lens is None:
            lens = self.fit_transform(self.data, projection)
            self.lens = lens

        # Create Cover
        if self.cover is None:
            cover = km.Cover(n_cubes, perc_overlap)
            self.cover = cover

        # Initialize Clustering Algorithm. Defualt is DBSCAN(eps=0.5, min_samples=3)
        if clusterer:
            self.clusterer = clusterer

        # Compute Simplicial Complex
        self.mapper = self.map(
            lens=self.lens, X=self.data, cover=self.cover, clusterer=self.clusterer
        )

    def mapper_to_networkx(self, min_intersection: int = 1):
        """
        Converts a kmapper simplicial complex into a networkx graph. Generates the `graph` attribute for CoalMapper.
        A simplicial complex must already be computed to use this function.

        Parameters
        -----------
        min_intersection: int, default is 1
            Minimum intersection considered when computing the graph.
            An edge will be created only when the intersection between two nodes is greater than or equal to `min_intersection`

        Returns
        -----------
        nx.Graph()
            A networkx graph based on a Kepler Mapper simplicial complex. Nodes determined by clusters and edges based on `min_intersection`.

        """
        # Initialize Nerve
        if self.nerve is None:
            nerve = km.GraphNerve(min_intersection)
            self.nerve = nerve

        assert (
            self.mapper is not None
        ), "You must first generate a Simplicial Complex with compute_mapper() before you can convert to Networkx "

        nodes = self.mapper["nodes"]
        _, simplices = self.nerve.compute(nodes)
        edges = [edge for edge in simplices if len(edge) == 2]

        G = nx.Graph(nodes=nodes)
        G.add_edges_from(edges)

        # Save Graph attritbute
        self.graph = G
        return self.graph

    def connected_components(self, min_intersection: int = 1):
        """

        Parameters
        -----------
        min_intersection: int, default is 1
            Minimum intersection considered when computing the graph.
            An edge will be created only when the intersection between two nodes is greater than or equal to `min_intersection`.

        Returns
        -----------
        components: list
            A list of graphs comprised of the connected components of the simplicial complex as dictated by `min_intersection`.

        """
        if self.components is None:
            if self.graph:
                components = [
                    self.graph.subgraph(c).copy()
                    for c in nx.connected_components(self.graph)
                ]
                self.components = components
            else:
                # Intialize Graph representation
                self.mapper_to_networkx(min_intersection)
                components = [
                    self.graph.subgraph(c).copy()
                    for c in nx.connected_components(self.graph)
                ]

        return self.components

    def item_lookup(
        self,
        item: str,
    ):
        """
        Parameters
        -----------
        item: str
            identifier for `item` (one element within a given cluster)
            #TODO Need more info on how the coal data will work here

        min_intersection: int, default is 1
            Minimum intersection considered when computing the graph.
            An edge will be created only when the intersection between two nodes is greater than or equal to `min_intersection`

        Returns
        -----------
        clusters: dict
            A dict of clusters that contain `item`. Keys are cluster labels, and values are cluster items.
        subgraph: list
            A subgraph made up of the connected componnets generated by clusters

        """
        assert (
            self.mapper is not None
        ), "First run `compute_mapper` to generate a simplicial complex."

        assert (
            self.graph is not None
        ), "First run `mapper_to_networkx` to generate a networkx graph."

        clusters = {}

        all_clusters = self.graph.nodes()

        for cluster in all_clusters:
            elements = self.mapper["nodes"][cluster]
            if item in elements:
                clusters[cluster] = elements

        # Note: for min_intersection >1 it is possible that item may lie in clusters spread across different components.
        subgraph_nodes = set.union(
            *[nx.node_connected_component(self.graph, node) for node in clusters.keys()]
        )

        subgraph = self.graph.subgraph(subgraph_nodes)

        return clusters, subgraph
