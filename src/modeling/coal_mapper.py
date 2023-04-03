"""Object file for Coal Mapper computations and analysis"""

import numpy as np
import networkx as nx
import kmapper as km
import pandas as pd
import os
import seaborn as sns
import itertools
import sys

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler

from hdbscan import HDBSCAN

from persim import plot_diagrams

from nammu.topology import calculate_persistence_diagrams, PersistenceDiagram
from nammu.curvature import ollivier_ricci_curvature
from nammu.utils import make_node_filtration

from kmapper import KeplerMapper


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(SRC)
from visualizing.visualization_helper import (
    config_plot_data,
    custom_color_scale,
    mapper_plot_outfile,
)


class CoalMapper:
    def __init__(
        self,
        data: pd.DataFrame,
        projection: np.ndarray,
        verbose: int = 0,
    ):
        """Constructor for CoalMapper class.
        Parameters
        ===========
        X: np.array
            Dataset features you wish to analyze using the mapper algorithm.
        projection: np.array
            Projected data. The low dimensional representation
            (e.g. post UMAP or TSNE) of X.
            Can combine representations, will become `lens` in `kmapper`.
            #TODO: Implement wrapper class here for handing projected data


        verbose: int, default is 0
            Logging level for `kmapper`. Levels (0,1,2) are supported.
        """
        # User Inputs
        self.data = data
        self.projection = projection

        # Initialize Mapper
        self._mapper = KeplerMapper(verbose=verbose)

        # Inputs for `fit`
        self._clusterer = None
        self._cover = None

        # Analysis Objects
        self._complex = dict()
        self._graph = nx.Graph()
        self._components = dict()
        self._curvature = np.array([])
        self._diagram = None  # PersistenceDiagram()

    @property
    def mapper(self):
        return self._mapper

    @property
    def cover(self):
        return self._cover

    @property
    def clusterer(self):
        return self._clusterer

    @property
    def complex(self):
        if len(self._complex["nodes"]) == 0:
            print(
                "Your simplicial complex is empty! \n\
                Run `fit()` to generate a simplicial complex. \n \
                Note: some parameters may produce a trivial mapper representation. \n"
            )
        return self._complex

    @property
    def graph(self):
        if len(self._graph.nodes()) == 0:
            print(
                "Your graph is empty! \n \
                Run `to_networkx()` to generate a graph. \n \
                Note: some parameters may produce a trivial mapper representation.\n"
            )
        return self._graph

    @property
    def components(self):
        if len(self._components) == 0:
            print(
                "You don't have any connected components! \n \
                Run `connected_components()` to generate a graph. \n\
                Note: some parameters may produce a trivial mapper representation.\n"
            )
        return self._components

    @property
    def curvature(self):
        if len(self._curvature) == 0:
            print(
                "You don't have any edge curvatures! \
                First generate a nonempty networkx Graph with `to_networkx()`."
            )
        return self._curvature

    @curvature.setter
    def curvature(self, curvature_fn):
        assert (
            len(self._graph.nodes()) > 0
        ), "First run `to_networkx` to generate a non-empty networkx graph."

        try:
            curvature = curvature_fn(self.graph)
            assert len(curvature) == len(self._graph.edges())
            self._curvature = curvature
        except:
            print("Invalid Curvature function")

    @property
    def diagram(self):
        if self._diagram is None:
            print(
                "Your persistence diagrams are empty! \
                First generate a networkx Graph with `to_networkx()`."
            )
        return self._diagram

    #############################################################################################################################################
    #############################################################################################################################################

    def fit(
        self,
        n_cubes: int = 6,
        perc_overlap: float = 0.4,
        clusterer=HDBSCAN(min_cluster_size=10),
    ):
        """
        A wrapper function for kmapper that generates a simplicial complex based on a given lens, cover, and clustering algorithm.

        Parameters
        -----------
        n_cubes: int, defualt 4
            Number of cubes used to cover of the latent space. Used to construct a kmapper.Cover object.

        perc_overlap: float, default 0.2
            Percentage of intersection between the cubes covering the latent space.Used to construct a kmapper.Cover object.

        clusterer: default is HDBSCAN
            Scikit-learn API compatible clustering algorithm. Must provide `fit` and `predict`.

        Returns
        -----------
        simplicial_complex : dict
            A dictionary with "nodes", "links" and "meta" information.

        """
        # Log cover and clusterer from most recent fit
        self._cover = km.Cover(n_cubes, perc_overlap)
        self._clusterer = clusterer

        # Compute Simplicial Complex
        self._complex = self._mapper.map(
            lens=self.projection,
            X=self.projection,
            cover=self.cover,
            clusterer=self.clusterer,
        )

        return self._complex

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
        nerve = km.GraphNerve(min_intersection)
        assert (
            len(self._complex["nodes"]) > 0
        ), "You must first generate a non-empty Simplicial Complex with `fit()` before you can convert to Networkx "

        nodes = self._complex["nodes"]
        _, simplices = nerve.compute(nodes)
        edges = [edge for edge in simplices if len(edge) == 2]

        # Setting self._graph
        self._graph = nx.Graph()
        self._graph.add_nodes_from(nodes)
        nx.set_node_attributes(self._graph, dict(self.complex["nodes"]), "membership")
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

        return self.components

    def calculate_homology(self, filter_fn=ollivier_ricci_curvature, use_min=True):
        """Compute Persistent Diagrams based on a curvature filtration of `self._graph`."""
        assert (
            len(self.graph.nodes()) > 0
        ), "First run `to_networkx` to generate a non-empty networkx graph."

        if len(self._curvature) > 0:
            "Computing edge curvature values"
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
        For an item in your dataset, find the subgraph and clusters that contain the item.

        Parameters
        -----------
        index: int
            identifier for an item in `self.data`.

        Returns
        -----------
        clusters: dict
            A dict of clusters that contain `item`. Keys are cluster labels, and values are cluster items.
        subgraph: list
            A subgraph made up of the connected componnets generated by clusters.
            In most cases this is the connected component that contains the item.

        """
        assert (
            len(self.complex) > 0
        ), "You must first generate a Simplicial Complex with `fit()` before you perform `item_lookup`."

        clusters = {}

        all_clusters = self.graph.nodes()

        subgraph_nodes = set()

        for cluster in all_clusters:
            elements = self.complex["nodes"][cluster]
            if index in elements:
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

    def mapper_clustering(self):
        """
        Execute a mapper-based clutering based on connected components.
        Append a column to `self.data` labeling each item.

        Returns
        -----------
        data: pd.Dataframe
            An updated dataframe with a column titled `cluster_labels`

        """
        assert (
            len(self.complex) > 0
        ), "You must first generate a Simplicial Complex with `fit()` before you perform clustering."

        # Initialize Labels as -1 (`unclustered`)
        labels = -np.ones(len(self.data))
        count = 0
        for component in self.components.keys():
            cluster_label = self.components[component]
            clusters = component.nodes()

            elements = []
            for cluster in clusters:
                elements.append(self.complex["nodes"][cluster])

            indices = set(itertools.chain(*elements))
            count += len(indices)
            labels[list(indices)] = cluster_label
        self.data["cluster_labels"] = labels
        return labels

    #############################################################################################################################################
    #############################################################################################################################################

    def plot(
        self,
    ):
        """"""
        assert (
            len(self.complex) > 0
        ), "First run `fit()` to generate a nonempty simplicial complex."

        path_html = mapper_plot_outfile(self.cover)

        numeric_data, labels = config_plot_data(self.data)
        colorscale = custom_color_scale()
        _ = self.mapper.visualize(
            self.complex,
            node_color_function=["mean", "median", "std", "min", "max"],
            color_values=numeric_data,
            color_function_name=labels,
            colorscale=colorscale,
            path_html=path_html,
        )
        print(f"Go to {path_html} for a visualization of your CoalMapper!")
        return path_html

    def plot_curvature(self, bins="auto", kde=False):
        """Visualize Curvature of a mapper graph as a histogram."""

        ax = sns.histplot(
            self.curvature,
            discrete=True,
            stat="probability",
            kde=kde,
            bins=bins,
        )
        ax.set(xlabel="Ollivier Ricci Edge Curvatures")

        return ax

    def plot_diagrams(self):
        """Visualize persistence diagrams of a mapper graph."""
        persim_diagrams = [
            np.asarray(self.diagram[0]._pairs),
            np.asarray(self.diagram[1]._pairs),
        ]
        return plot_diagrams(persim_diagrams, show=True)

    def connected_component_heatmaps(self):
        viz = self.data

        targetCols = [
            "NAMEPCAP",
            "GENNTAN",
            "weighted_coal_CAPFAC",
            "weighted_coal_AGE",
            "Retrofit Costs",
            "forwardCosts",
            "PLSO2AN",
        ]

        # mean data
        viz2 = viz.groupby("cluster_labels", as_index=False).mean()[targetCols]
        c_labels = viz.groupby("cluster_labels").mean().reset_index()["cluster_labels"]
        scaler = MinMaxScaler()
        data = scaler.fit_transform(viz2)
        df = pd.DataFrame(data, columns=list(viz2.columns)).set_index(c_labels)

        #% of max data
        # TODO: fix the below to append 'cluster_labels' to targetCols so targetCols can be a user input
        df2 = viz[
            [
                "NAMEPCAP",
                "GENNTAN",
                "weighted_coal_CAPFAC",
                "weighted_coal_AGE",
                "Retrofit Costs",
                "forwardCosts",
                "PLSO2AN",
                "cluster_labels",
            ]
        ]
        df2 = df2.groupby(["cluster_labels"]).sum() / df2[targetCols].sum()

        fig = make_subplots(
            rows=1,
            cols=2,
            vertical_spacing=0.1,
            subplot_titles=(
                "Connected Component Averages",
                "Connected Component % of Total",
            ),
            specs=[[{"type": "Heatmap"}, {"type": "Heatmap"}]],
            y_title="Cluster Label",
        )

        col = 1
        for df in [df, df2]:
            if col == 1:
                text = viz2.round(2).values.tolist()
                texttemplate = "%{text}"
            else:
                text = df.round(2).values.tolist()
                texttemplate = ("%{text}") + "%"

            fig.add_trace(
                go.Heatmap(
                    x=df.columns.to_list(),
                    y=df.index.to_list(),
                    z=df.values.tolist(),
                    text=text,
                    colorscale="rdylgn_r",
                    xgap=2,
                    ygap=2,
                    texttemplate=texttemplate,
                ),
                row=1,
                col=col,
            )
            col = col + 1

        fig.update_xaxes(tickangle=45)
        fig.update_layout(font=dict(size=10))
        fig.update_traces(showscale=False)
        print(
            "Number of Plants in each Connected Component (-1 indicates not displayed on mapper):\n"
        )
        print(
            viz.groupby("cluster_labels")
            .count()
            .rename(columns={"NAMEPCAP": "#CoalPlants"})["#CoalPlants"]
        )
        pio.show(fig)