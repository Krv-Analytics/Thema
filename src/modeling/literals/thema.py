# model.py

import math
import os
import pickle
import sys
from os.path import isfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
from dotenv import load_dotenv
from persim import plot_diagrams
from plotly.subplots import make_subplots

# TODO: Move this to a defaulted argument for the viz functions
pio.renderers.default = "browser"

########################################################################################
#
#   Handling Local Imports
#
########################################################################################


from visual_utils import custom_color_scale, get_subplot_specs, reorder_colors

load_dotenv()
root = os.getenv("root")
sys.path.append(root + "jmapping/fitting/")

from jbottle import JBottle
from jmapper import JMapper

########################################################################################
#
#   THEMA class Implementation
#
########################################################################################


class THEMA(JBottle):
    """
    A class designed for easy to use and interpret visualizations of JGraphs and JMapper
    objects.


    Members
    -------


    Member Functions
    ----------------

    """

    def __init__(self, jmapper: str):
        """Constructor for Model class.

        Parameters
        -----------
        jmapper:str
          Path to a Jmapper/hyerparameter file

        """

        assert isfile(jmapper)
        with open(jmapper, "rb") as f:
            reference = pickle.load(f)
        self._jmapper = reference["jmapper"]
        self._hyper_parameters = reference["hyperparameters"]

        # Initialize Inherited JBottle
        super().__init__(reference["jmapper"])

        # As a rule, if it has to do with pure data analysis
        # then it belongs in JBottle
        # data analysis utility functions on pd.DataFrames will be written into
        # data_utils.py

        # Plotting member for visualization
        self._cluster_positions = None

    @property
    def jmapper(self):
        """Return the hyperparameters used to fit this model."""
        return self._jmapper

    @property
    def hyper_parameters(self):
        """Return the hyperparameters used to fit this model."""
        return self._hyper_parameters

    def visualize_model(self, k=None, seed=6):
        """
        Visualize the clustering as a network. This function plots
        the JMapper's graph in a matplotlib figure, coloring the nodes
        by their respective policy group.

        Parameters
        -------
        k : float, default is None
            Optimal distance between nodes. If None the distance is set to
            1/sqrt(n) where n is the number of nodes. Increase this value to
            move nodes farther apart.

        """
        # Config Pyplot
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot()
        color_scale = np.array(custom_color_scale()).T[1]
        # Get Node Coords
        pos = nx.spring_layout(self.jmapper.jgraph.graph, k=k, seed=seed)

        # Plot and color components
        labels, components = zip(*self.jmapper.jgraph.components.items())
        for i, g in enumerate(components):
            nx.draw_networkx(
                g,
                pos=pos,
                node_color=color_scale[i],
                node_size=100,
                font_size=6,
                with_labels=False,
                ax=ax,
                label=f"Group {labels[i]}",
                edgelist=[],
            )
            nx.draw_networkx_edges(
                g,
                pos=pos,
                width=2,
                ax=ax,
                label=None,
                alpha=0.6,
            )
        ax.legend(loc="best", prop={"size": 8})
        plt.axis("off")
        self._cluster_positions = pos
        return ax

    def visualize_component(self, component, cluster_labels=True):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()

        color_scale = np.array(custom_color_scale()).T[1]
        labels, components = zip(*self.jmapper.jgraph.components.items())
        for i, g in enumerate(components):
            if i == component:
                nx.draw_networkx(
                    g,
                    pos=self._cluster_positions,
                    node_color=color_scale[i],
                    node_size=100,
                    font_size=10,
                    with_labels=cluster_labels,
                    ax=ax,
                    label=f"Group {labels[i]}",
                    # font_color = color_scale[i],
                    edgelist=[],
                )
                nx.draw_networkx_edges(
                    g,
                    pos=self._cluster_positions,
                    width=2,
                    ax=ax,
                    label=None,
                    alpha=0.6,
                )
        ax.legend(loc="best", prop={"size": 8})
        plt.axis("off")
        return fig

    def visualize_projection(self, show_color=True, show_axis=False):
        """
        Visualize the clustering on the projection point cloud.
        This function plots the projection used to fit JMapper
        and colors points according to their cluster.
        """
        color_scale = np.array(custom_color_scale()).T[1]

        projection, parameters = (
            self.projection,
            self.jmapper.tupper.get_projection_parameters(),
        )
        print(projection)
        if show_color:
            fig = go.Figure()

            for g in self._group_directory.keys():
                idxs = self.get_groups_members(g)
                label = f"Policy Group {int(g)}"
                if g == -1:
                    label = "Unclustered Items"
                # mask = np.where(, True, False)
                cluster = projection[idxs]
                print(f"Cluster: {cluster}")
                fig.add_trace(
                    go.Scatter(
                        x=cluster.T[0],
                        y=cluster.T[1],
                        mode="markers",
                        marker=dict(color=color_scale[int(g)]),
                        name=label,
                    )
                )
        else:
            fig = go.Figure()
            for g in np.unique(self._group_directory.keys()):
                label = f"Policy Group {int(g)}"
                if g == -1:
                    label = "Unclustered Items"
                mask = np.where(self._group_directory.keys() == g, True, False)
                cluster = projection[mask]
                fig.add_trace(
                    go.Scatter(
                        x=cluster.T[0],
                        y=cluster.T[1],
                        mode="markers",
                        marker=dict(color="grey"),
                        showlegend=False,
                    )
                )
        if show_axis:
            fig.update_layout(
                title=f"UMAP: {parameters}",
                legend=dict(
                    title="",
                    bordercolor="black",
                    borderwidth=1,
                ),
                width=800,
                height=600,
            )
        else:
            fig.update_layout(
                # title=f"UMAP: {parameters}",
                width=800,
                height=600,
                showlegend=False,
                xaxis=dict(
                    tickcolor="white",
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                ),
                yaxis=dict(
                    tickcolor="white",
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                ),
            )

        fig.update_layout(template="simple_white")
        return fig

    def visualize_piecharts(self):
        # Define the color map based on the dictionary values
        colors = []

        for i in range(len(custom_color_scale()[:-3])):
            inst = custom_color_scale()[:-3]
            rgb_color = "rgb" + str(
                tuple(int(inst[i][1][j : j + 2], 16) for j in (2, 3, 5))
            )
            colors.append(rgb_color)

        colors = reorder_colors(colors)
        color_map = {
            key: colors[i % len(colors)]
            for i, key in enumerate(
                set.union(
                    *[set(v.keys()) for v in self.get_group_descriptions().values()]
                )
            )
        }
        group_descriptions = self.get_group_descriptions()

        num_rows = math.ceil(max(len(group_descriptions) / 3, 1))
        specs = get_subplot_specs(len(group_descriptions))

        groups = {i: f"Group {i}" for i in range(len(group_descriptions))}
        groups = {-1: "Outliers", **groups}
        fig = make_subplots(
            rows=num_rows,
            cols=3,
            specs=specs,
            subplot_titles=[
                f"<b>{groups[group]}</b>: {len(self.get_groups_members(group))} Members"
                for group in group_descriptions.keys()
            ],
            horizontal_spacing=0.1,
        )

        for group in group_descriptions.keys():

            labels = list(group_descriptions[group].keys())
            sizes = list(group_descriptions[group].values())

            row = group // 3 + 1
            col = group % 3 + 1
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    textinfo="percent",
                    values=sizes,
                    textposition="outside",
                    marker_colors=[color_map[l] for l in labels],
                    scalegroup=group,
                    hole=0.5,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            # height=600,
            height=num_rows * 300,
            width=800,
        )

        fig.update_annotations(yshift=10)

        fig.update_traces(marker=dict(line=dict(color="white", width=3)))

        # TODO: Base this yshift on the number of descriptors
        y_shift = -2

        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=y_shift, xanchor="left", x=0
            )
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "custom_image",
                "scale": 5,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        # Show the subplot
        fig.show(config=config)

    def visualize_boxplots(self, cols=[], target=pd.DataFrame()):

        show_targets = False
        if len(target) == 1:
            numeric_cols = (
                target.select_dtypes(include=["number"]).dropna(axis=1).columns
            )
            target = target[numeric_cols]
            show_targets = True

        df = self.raw
        df["cluster_IDs"] = [
            self.get_items_groupID(item)[0]
            if type(self.get_items_groupID(item)) == list
            else self.get_items_groupID(item)
            for item in df.index
            if item != -1
        ]

        if len(cols) > 0:
            cols.append("cluster_IDs")
            df = df.loc[:, df.columns.isin(cols)]

        fig = make_subplots(
            rows=math.ceil(len(df.columns.drop("cluster_IDs")) / 3),
            cols=3,
            horizontal_spacing=0.1,
            subplot_titles=df.columns.drop("cluster_IDs"),
        )

        row = 1
        col = 1

        groups = list(self._group_directory.keys())

        for column in df.columns.drop("cluster_IDs"):
            for group in list(groups):
                fig.add_trace(
                    go.Box(
                        y=self.get_groups_raw_df(group)[column],
                        name=groups[group],
                        jitter=0.3,
                        showlegend=False,
                        whiskerwidth=0.6,
                        marker_size=3,
                        line_width=1,
                        boxmean=True,
                        # hovertext=df["companyName"],
                        marker=dict(color=custom_color_scale()[int(group)][1]),
                    ),
                    row=row,
                    col=col,
                )

                if show_targets:
                    if column in target.columns:
                        fig.add_hline(
                            y=target[column].mean(),
                            line_width=1,
                            line_dash="solid",
                            line_color="red",
                            col=col,
                            row=row,
                            annotation_text="target",
                            annotation_font_color="red",
                            annotation_position="top left",
                        )

                try:
                    pd.to_numeric(df[column])
                    fig.add_hline(
                        y=df[column].mean(),
                        line_width=1,
                        line_dash="dash",
                        line_color="black",
                        col=col,
                        row=row,
                        annotation_text="mean",
                        annotation_font_color="gray",
                        annotation_position="top right",
                    )
                except ValueError:
                    """"""

            col += 1
            if col == 4:
                col = 1
                row += 1

        fig.update_layout(
            template="simple_white",
            height=(math.ceil(len(df.columns) / 3)) * 300,
            title_font_size=20,
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "custom_image",
                "height": 1200,
                "width": 1000,
                "scale": 5,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        fig.show(config=config)
        return fig

    def visualize_curvature(self, bins="auto", kde=False):
        """Visualize th curvature of a graph graph as a histogram.

        Parameters
        ----------
        bins: str, default is "auto"
            Method for seaborn to assign bins in the histogram.
        kde: bool
            If true then draw a density plot.
        """

        ax = sns.histplot(
            self.jmapper.jgraph.curvature,
            discrete=True,
            stat="probability",
            kde=kde,
            bins=bins,
        )
        ax.set(xlabel="Ollivier Ricci Edge Curvatures")
        plt.show()
        return ax

    def visualize_persistence_diagram(self):
        """Visualize persistence diagrams of a mapper graph
        using functionality from Persim."""
        diagram = self.jmapper.jgraph.diagram
        # TODO: put in catch for weird diagrams...
        assert len(diagram) > 0, "Persistence Diagram is empty."

        persim_diagrams = [
            np.asarray(diagram[0]._pairs),
            np.asarray(diagram[1]._pairs),
        ]
        try:
            fig = plot_diagrams(persim_diagrams, show=True)
        except ValueError:
            print("Weird Diagram")
            fig = 0
        return fig
