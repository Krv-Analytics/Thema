import itertools
import pickle
from os.path import isfile

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import seaborn as sns
from jmapper import JMapper
from model_helper import (
    config_plot_data,
    custom_color_scale,
    get_minimal_std,
    mapper_plot_outfile,
)
from persim import plot_diagrams


class Model:
    """A clustering model for point cloud data
    based on the Mapper Algorithm.

    This class interprets the graph representation of a JMapper
    as a clustering:
    Connected Components of the graph are the clusters which we call
    "Policy Groups" for our project. We provide functionality
    for analyzing nodes (or subgroups),
    as well as the policy groups themselves.

    Most notably, we provide functionality for understanding the structural
    equivalency classes of an arbitrary number of different models. Different
    meaning the models are generated from different hyperparameters, but it
    may be that the graphs they generate have very similar structure. We
    analyze these structural equivalence classes using the functioanlity
    in JMapper for computing curvature filtrations on graphs.

    We also provide functionality for visualizing various attributes
    of our graph models.
    """

    def __init__(self, mapper: str):
        """Constructor for Model class.
        Parameters
        -----------
        mapper: <jmapper.JMapper>
            A data container that holds raw, cleaned, and projected
            versions of user data.
        """

        self._mapper = None
        self._tupper = None
        self._complex = None
        self._hyper_parameters = None

        # Check is valid mapper path
        if isfile(mapper):
            self._mapper = mapper

        # Mapper Node Attributes
        self._node_ids = None
        self._node_description = None

        # Mapper Cluster Attributes
        self._cluster_ids = None
        self._cluster_descriptions = None
        self._cluster_sizes = None
        self._unclustered_items = None

    @property
    def mapper(self):
        assert self._mapper, "Please Specify a valid path to a mapper object"
        with open(self._mapper, "rb") as mapper_file:
            reference = pickle.load(mapper_file)
            mapper = reference["mapper"]
        return mapper

    @property
    def tupper(self):
        """Return the Tupper assocaited with `self.mapper`."""
        return self.mapper.tupper

    @property
    def hyper_parameters(self):
        """Return the hyperparameters used to fit this model."""
        if self._hyper_parameters:
            return self._hyper_parameters
        assert self._mapper, "Please Specify a valid path to a mapper object"
        with open(self._mapper, "rb") as mapper_file:
            reference = pickle.load(mapper_file)
            self._hyper_parameters = reference["hyperparameters"]
        return self._hyper_parameters

    @property
    def complex(self):
        """Return the complex from a fitted JMapper."""
        return self.mapper.complex

    @property
    def node_ids(self):
        """Return the node labels according to the graph clustering."""
        if self._node_ids is None:
            self.label_item_by_node()
        return self._node_ids

    @property
    def node_description(self):
        """Return the node descriptions according to the graph clustering."""
        if self._node_description is None:
            self.compute_node_descriptions()
        return self._node_description

    @property
    def cluster_ids(self):
        """Return the policy group labels according to the graph clustering."""
        if self._cluster_ids is None:
            self.label_item_by_cluster()
        return self._cluster_ids

    @property
    def cluster_descriptions(self):
        """Return the policy group descriptions according
        to the graph clustering."""
        if self._cluster_descriptions is None:
            self.compute_cluster_descriptions()
        return self._cluster_descriptions

    @property
    def cluster_sizes(self):
        """Return the policy group sizes according to the graph clustering."""
        if self._cluster_sizes is None:
            self.label_item_by_cluster()
        return self._cluster_sizes

    @property
    def unclustered_items(self):
        """Return the items left out of the graph clustering."""
        if self._unclustered_items is None:
            self.label_item_by_node()
        return self._unclustered_items

    def label_item_by_node(self):
        """Label each item in the data set according to its corresponding
        node in the graph.

        This function loops through each of the connected components in
        the graph and labels each item with the component ID
        of the corresponding node.

        Returns
        -------
        self._cluster_ids : np.ndarray, shape (N,)
            Array with policy group (cluster) ID for each item in the data.
        """
        N = len(self.tupper.clean)
        labels = dict()
        nodes = self.complex["nodes"]
        unclustered_items = []
        for idx in range(N):
            place_holder = []
            for node_id in nodes.keys():
                if idx in nodes[node_id]:
                    place_holder.append(node_id)

            if len(place_holder) == 0:
                place_holder = -1
                unclustered_items.append(idx)
            labels[idx] = place_holder

        self._node_ids = labels
        self._unclustered_items = unclustered_items
        return self._node_ids

    def label_item_by_cluster(self):
        """Label each item in the data set according to its corresponding
        policy group, i.e. connected component in the graph.

        Returns
        -------
        self._node_description : dict
            Dictionary with node ID as key and a string with the node
            description as value.
        """
        assert (
            len(self.complex) > 0
        ), "You must first generate a Simplicial Complex with `fit()` before you perform clustering."

        self._cluster_sizes = {}
        labels = -np.ones(len(self.tupper.clean))
        components = self.mapper.components
        for component in components.keys():
            cluster_id = components[component]
            nodes = component.nodes()

            elements = []
            for node in nodes:
                elements.append(self.complex["nodes"][node])

            indices = set(itertools.chain(*elements))
            size = len(indices)
            labels[list(indices)] = cluster_id
            self._cluster_sizes[cluster_id] = size

        self._cluster_ids = labels
        self._cluster_sizes[-1] = len(self.unclustered_items)

    def compute_node_descriptions(self):
        """
        Compute a simple description of each node in the graph.

        This function labels each node based on the column in
        the dataframe that admits the smallest (normalized)
        standard deviation amongst items in the node.

        Returns
        -------
        self._node_description : dict
            Dictionary with node ID as key and a string with the node
            description as value.
        """
        nodes = self.complex["nodes"]
        self._node_description = {}
        for node in nodes.keys():
            mask = nodes[node]
            label = get_minimal_std(
                df=self.tupper.clean,
                mask=mask,
            )
            size = len(mask)
            self._node_description[node] = {"label": label, "size": size}

    def compute_cluster_descriptions(self):
        """
        Compute a simple description of each cluster (policy group).

        This function creates a density description based on each of
        the node lables in the policy group.

        Returns
        -------
        self._cluster_descriptions : dict
            Dictionary with cluster ID as key and a dictionary with the
            labeled density descriptions as values.
        """
        self._cluster_descriptions = {}
        components = self.mapper.components
        # View cluster as networkX graph
        for G in components.keys():
            cluster_id = components[G]
            nodes = G.nodes()
            holder = {}
            N = 0
            for node in nodes:
                label = self.node_description[node]["label"]
                size = self.node_description[node]["size"]

                N += size
                # If multiple nodes have same identifying column
                if label in holder.keys():
                    size += holder[label]
                holder[label] = size
            density = {label: np.round(size / N, 2) for label, size in holder.items()}
            self._cluster_descriptions[cluster_id] = {
                "density": density,
                "size": self.cluster_sizes[cluster_id],
            }

        unclustered_label = get_minimal_std(
            df=self.tupper.clean, mask=self.unclustered_items
        )
        self._cluster_descriptions[-1] = {
            "density": {unclustered_label: 1.0},
            "size": self.cluster_sizes[-1],
        }

    def get_cluster_dfs(self):
        """
        Generate a DataFrame for each policy group.
        Note: unclustered items are given their own DataFrame.

        Returns
        -------
        subframes: dict
            A dictionary where each key is a policy group (cluster) ID
            and the corresponding value is a DataFrame containing
            the items in that policy group.
        """
        # Load Model
        clean = self.tupper.clean
        # Assign cluster labels
        clean.insert(
            loc=0,
            column="cluster_labels",
            value=self.cluster_ids,
        )
        subframes = {}
        for label in np.unique(self.cluster_ids):
            sub_frame = clean[clean["cluster_labels"] == label]
            # Merge with Raw
            raw_subframe = pd.merge(
                sub_frame["cluster_labels"],
                self.tupper.raw,
                right_index=True,
                left_index=True,
            )
            df_label = f"policy_group_{int(label)}"
            if label == -1:
                df_label = "unclustered"
            subframes[df_label] = raw_subframe
        return subframes

    def visualize_model(self):
        """
        Visualize the clustering as a network. This function plots
        the JMapper's graph in a matplotlib figure, coloring the nodes
        by their respective policy group.
        """
        # Config Pyplot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()
        color_scale = np.array(custom_color_scale()).T[1]
        # Get Node Coords
        pos = nx.spring_layout(self.mapper.graph)

        # Plot and color components
        components, labels = zip(*self.mapper.components.items())
        for i, g in enumerate(components):
            nx.draw_networkx(
                g,
                pos=pos,
                node_color=color_scale[i],
                node_size=100,
                font_size=6,
                with_labels=False,
                ax=ax,
                label=f"Policy Group {labels[i]}",
                edgelist=[],
            )
            nx.draw_networkx_edges(
                g,
                pos=pos,
                width=2,
                ax=ax,
                label=None,
            )
        ax.legend(loc="best", prop={"size": 8})
        plt.axis("off")

    # TODO: 1) Fix Color Scale and match with visualize_model
    def visualize_projection(self):
        """
        Visualize the clustering on the projection point cloud.
        This function plots the projection used to fit JMapper
        and colors points according to their cluster.
        """
        color_scale = np.array(custom_color_scale()).T[1]

        projection, parameters = (
            self.tupper.projection,
            self.tupper.get_projection_parameters(),
        )
        _, ax = plt.subplots(figsize=(6, 6))
        for g in np.unique(self.cluster_ids):
            label = f"Policy Group {int(g)}"
            if g == -1:
                label = "Unclustered Items"
            mask = np.where(self.cluster_ids == g, True, False)
            cluster = projection[mask]
            ax.scatter(
                cluster.T[0],
                cluster.T[1],
                label=label,
                color = color_scale[int(g)],
                s=80,
            )
            ax.legend()
        plt.title(f"UMAP: {parameters}")
        plt.tight_layout()
        plt.show()

    def visualize_curvature(self, bins="auto", kde=False):
        """Visualize th curvature of a graph graph as a histogram.

        Parameters
        ----------
        bins: str, defualt is "auto"
            Method for seaborn to assign bins in the histogram.
        kde: bool
            If true then draw a density plot.
        """

        ax = sns.histplot(
            self.mapper.curvature,
            discrete=True,
            stat="probability",
            kde=kde,
            bins=bins,
        )
        ax.set(xlabel="Ollivier Ricci Edge Curvatures")
        plt.show()

    def visualize_persistence_diagram(self):
        """Visualize persistence diagrams of a mapper graph
        using functionality from Persim."""
        assert len(self.mapper.diagram) > 0, "Persistence Diagram is empty."
        persim_diagrams = [
            np.asarray(self.mapper.diagram[0]._pairs),
            np.asarray(self.mapper.diagram[1]._pairs),
        ]
        return plot_diagrams(persim_diagrams, show=True)

    @DeprecationWarning
    def visualize_mapper(self):
        """
        Plot using the Keppler Mapper html functionality.

        NOTE: These visualizations are no longer maintained by KepplerMapper
        and we do not reccomend using them.
        """
        assert len(self.complex) > 0, "Model needs a `fitted` mapper."
        kepler = self.mapper.mapper
        path_html = mapper_plot_outfile(self.hyper_parameters)
        numeric_data, labels = config_plot_data(self.tupper)

        colorscale = custom_color_scale()
        # Use Kmapper Visualization
        kepler.visualize(
            self.mapper.complex,
            node_color_function=["mean", "median", "std", "min", "max"],
            color_values=numeric_data,
            color_function_name=labels,
            colorscale=colorscale,
            path_html=path_html,
        )

        print(f"Go to {path_html} for a visualization of your JMapper!")
