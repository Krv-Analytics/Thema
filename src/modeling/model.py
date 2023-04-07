from os.path import isfile
import pickle
import numpy as np
import itertools

from persim import plot_diagrams
import seaborn as sns

from coal_mapper import CoalMapper
from model_helper import (
    config_plot_data,
    custom_color_scale,
    mapper_plot_outfile,
    get_minimal_std,
)


class Model:
    def __init__(self, mapper: str, min_intersection=1):

        self._mapper = None
        self._tupper = None
        self._complex = None
        self._hyper_parameters = None

        self.min_intersection = min_intersection

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
            mapper = reference[self.min_intersection]
        return mapper

    @property
    def tupper(self):
        return self.mapper.tupper

    @property
    def hyper_parameters(self):
        if self._hyper_parameters:
            return self._hyper_parameters
        assert self._mapper, "Please Specify a valid path to a mapper object"
        with open(self._mapper, "rb") as mapper_file:
            reference = pickle.load(mapper_file)
            self._hyper_parameters = reference["hyperparameters"]
        return self._hyper_parameters

    @property
    def complex(self):
        return self.mapper.complex

    @property
    def node_ids(self):
        if self._node_ids is None:
            self.label_item_by_node()
        return self._node_ids

    @property
    def node_description(self):
        if self._node_description is None:
            self.compute_node_descriptions()
        return self._node_description

    @property
    def cluster_ids(self):
        if self._cluster_ids is None:
            self.label_item_by_cluster()
        return self._cluster_ids

    @property
    def cluster_descriptions(self):
        if self._cluster_descriptions is None:
            self.compute_cluster_descriptions()
        return self._cluster_descriptions

    @property
    def cluster_sizes(self):
        if self._cluster_sizes is None:
            self.label_item_by_cluster()
        return self._cluster_sizes

    @property
    def unclustered_items(self):
        if self._unclustered_items is None:
            self.label_item_by_node()
        return self._unclustered_items

    def label_item_by_node(self):
        """Function to set the node_id"""

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
        """Choose the column in the `tupper.clean` with the lowest standard deviation."""
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
        """Compute a density based description of the cluster based on its nodes."""
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

    def visualize_mapper(self):
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

        print(f"Go to {path_html} for a visualization of your CoalMapper!")

    def visualize_curvature(self, bins="auto", kde=False):
        """Visualize Curvature of a mapper graph as a histogram."""

        ax = sns.histplot(
            self.mapper.curvature,
            discrete=True,
            stat="probability",
            kde=kde,
            bins=bins,
        )
        ax.set(xlabel="Ollivier Ricci Edge Curvatures")

        return ax

    def visualize_persistence_diagram(self):
        """Visualize persistence diagrams of a mapper graph."""
        persim_diagrams = [
            np.asarray(self.mapper.diagram[0]._pairs),
            np.asarray(self.mapper.diagram[1]._pairs),
        ]
        return plot_diagrams(persim_diagrams, show=True)
