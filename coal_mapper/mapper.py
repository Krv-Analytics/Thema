"""Object file for Coal Mapper computations and analysis"""

import numpy as np
import networkx as nx
import kmapper as km
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import datetime
import seaborn as sns


from persim import plot_diagrams
from nammu.topology import calculate_persistence_diagrams
from nammu.curvature import ollivier_ricci_curvature, forman_curvature
from nammu.utils import make_node_filtration

from kmapper import KeplerMapper


class CoalMapper(KeplerMapper):
    # TODO: Doc String
    def __init__(self, X: np.array, verbose: int = 0):
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
        n_cubes: int = 6,
        perc_overlap: float = 0.4,
        projection=TSNE(
            random_state=None,
        ),
        clusterer=KMeans(8, random_state=None, n_init="auto"),
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
            print("Setting Lens")
            lens = self.fit_transform(self.data, projection)
            self.lens = lens

        # Create Cover
        if self.cover is None:
            print("Setting Cover")
            cover = km.Cover(n_cubes, perc_overlap)
            self.cover = cover

        # Initialize Clustering Algorithm. Defualt is DBSCAN(eps=0.5, min_samples=3)
        if self.clusterer is None:
            self.clusterer = clusterer

        # Compute Simplicial Complex
        self.mapper = self.map(
            lens=self.lens, X=self.data, cover=self.cover, clusterer=self.clusterer
        )

    def to_networkx(self, min_intersection: int = 1):
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
                print("Initializing Full Graph")
                self.to_networkx(min_intersection)
                print(self.graph)
                components = [
                    self.graph.subgraph(c).copy()
                    for c in nx.connected_components(self.graph)
                ]
                self.components = components

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
        ), "First run `to_networkx` to generate a networkx graph."

        clusters = {}

        all_clusters = self.graph.nodes()

        for cluster in all_clusters:
            elements = self.mapper["nodes"][cluster]
            if item in elements:
                clusters[cluster] = elements

                # Note: for min_intersection >1 it is possible that item may lie in clusters spread across different components.

                subgraph_nodes = set.union(
                    *[
                        nx.node_connected_component(self.graph, node)
                        for node in clusters.keys()
                    ]
                )

        subgraph = self.graph.subgraph(subgraph_nodes)

        return clusters, subgraph

    def plot(self, output_dir: str = "../outputs/htmls/"):
        """"""
        assert (
            self.mapper is not None
        ), "First run `set_graph` to generate a simplicial complex."
        time = int(datetime.datetime.now().timestamp())
        path_html = output_dir + f"coal_mapper_{time}.html"
        print(path_html)
        dic = self.mapper
        assert type(dic) == dict, "Not a dictionary"
        _ = self.visualize(
            dic,
            path_html=path_html,
            # include_searchbar=True,
            # include_min_intersection_selector=False
            # title="Coal Mapper",
        )
        print(f"Go to {path_html} for a visualization of your CoalMapper!")
        return path_html


##############################################################################################################################################
##############################################################################################################################################


class MapperTopology:
    """
    Analyzing different mapper graphs using discrete curvature and persistenet homology.

    """

    def __init__(self, X: np.ndarray):

        self.data = X
        self._mapper = None
        self._curvature = None
        self._graph = None
        self._diagram = None

    @property
    def mapper(self):
        return self._mapper

    @property
    def graph(self):
        return self._graph

    @property
    def curvature(self):
        if self._curvature is None:
            print(
                "Curvature has not been computed yet. \
                First generate a networkx Graph from the dataset via Mapper."
            )
        return self._curvature

    @property
    def diagram(self):
        if self._diagram is None:
            print(
                "Persistent Homology has not been computed yet. \
                First generate a networkx Graph from the dataset via Mapper."
            )
        return self._diagram

    @curvature.setter
    def curvature(self, curvature_fn):

        if self._graph is None:
            print(
                "You must first define a graph representation for \
                `X` via using `kmapper` before you can compute edge curvatures"
            )
        else:
            try:
                curvature = curvature_fn(self.graph)
                assert len(curvature) == len(self._graph.edges())
                self._curvature = curvature
            except:
                print("Invalid Curvature function")

    def set_graph(
        self,
        cover,
        clusterer=KMeans(5, n_init="auto", random_state=2023),
        min_intersection: int = 1,
    ):
        """Generate a new networkX graph from Data via mapper. Recompute"""
        # Check that a reasonable Cover is provided
        if (len(cover) == 2) and type(cover) is tuple:
            n_cubes, perc_overlap = cover
            if type(min_intersection) is not int or min_intersection < 1:
                print(
                    "Invalid Minimum Intersection Parameter for Mapper. \
                Defualt value (min_intersection=1) has been applied."
                )
                min_intersection = 1
            else:
                print("Computing Mapper Algorithm...")
                # Save for looping over min_intersection
                if self._mapper is not None:
                    self._graph = self._mapper.to_networkx(min_intersection)
                else:
                    self._mapper = CoalMapper(X=self.data)
                    self._mapper.clusterer = clusterer
                    self._mapper.compute_mapper(n_cubes, perc_overlap)
                print("Generating networkx Graph...")
                self._graph = self._mapper.to_networkx(min_intersection)

                if len(self.graph.nodes()) > 0:
                    # Automatically Compute OR Curvature and corresponding Diagrams when changing a graph
                    print(
                        "Using Ollivier Ricci filtration to compute edge curvature values and persistence diagrams. "
                    )
                    self.curvature = ollivier_ricci_curvature
                    self.calculate_homology(filter_fn=ollivier_ricci_curvature)

        else:
            print(
                "Please enter a valid Cover of the form \
                (n_cubes,perc_overlap)"
            )

    def calculate_homology(self, filter_fn, use_min=True):
        if self._graph is None:
            print(
                "You must first define a graph representation for \
                `X` via using `kmapper` before you can compute persistent diagrams"
            )
        elif len(self.graph.nodes()) > 0:
            # Default to OR Curvature
            if self._curvature is None:
                "Computing edge curvature values"
                self._curvature = filter_fn(self._graph)  # Set curvatures

            G = make_node_filtration(
                self._graph,
                self._curvature,
                attribute_name="curvature",
                use_min=use_min,
            )
            pd = calculate_persistence_diagrams(
                G,
                "curvature",
                "curvature",
            )
            self._diagram = pd
        else:
            print("ERROR: This mapper computation produced a graph with 0 nodes")

    # TODO: Implement Visualization Methods. See starter code in ./nammu/
    def plot_curvature(self):
        # Look at filtration visualization script
        return sns.histplot(self.curvature)

    def plot_diagrams(self):
        persim_diagrams = [
            np.asarray(self.diagram[0]._pairs),
            np.asarray(self.diagram[1]._pairs),
        ]
        return plot_diagrams(persim_diagrams, show=True)
