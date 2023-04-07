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
        self._cluster_description = None
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
    def cluster_description(self):
        if self._cluster_description is None:
            self.compute_cluster_descriptions()
        return self._cluster_description

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

        labels = -np.ones(len(self.tupper.clean))
        count = 0
        components = self.mapper.components
        for component in components.keys():
            cluster_id = components[component]
            nodes = component.nodes()

            elements = []
            for node in nodes:
                elements.append(self.complex["nodes"][node])

            indices = set(itertools.chain(*elements))
            count += len(indices)
            labels[list(indices)] = cluster_id

        self._cluster_ids = labels

    def compute_node_descriptions(self):
        nodes = self.complex["nodes"]
        self._node_description = {}
        for node in nodes.keys():
            mask = nodes[node]
            label = get_minimal_std(
                df=self.tupper.clean,
                mask=mask,
            )
            self._node_description[node] = label

    def compute_cluster_descriptions(self):
        """Choose the column in the `tupper.clean` with the lowest standard deviation."""
        clusters = np.unique(self.cluster_ids)
        self._cluster_description = {}
        for cluster in clusters:
            mask = np.array(np.where(self.cluster_ids == cluster, 1, 0), dtype=bool)
            label = get_minimal_std(df=self.tupper.clean, mask=mask)
            self._cluster_description[cluster] = label

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
