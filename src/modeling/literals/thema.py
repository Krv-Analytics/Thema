# model.py

import os
import sys
import math
import pickle
from os.path import isfile
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from persim import plot_diagrams
import plotly.io as pio

# TODO: Move this to a defaulted argument for the viz functions
pio.renderers.default = "browser"

########################################################################################
#
#   Handling Local Imports
#
########################################################################################


from visual_utils import custom_color_scale, reorder_colors, get_subplot_specs

load_dotenv()
root = os.getenv("root")
sys.path.append(root + "jmapping/fitting/")

from jmapper import JMapper
from jbottle import JBottle


########################################################################################
#
#   Model class Implementation
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
    def hyper_parameters(self):
        """Return the hyperparameters used to fit this model."""
        return self._hyper_parameters

    def target_matching(
        self,
        target: pd.DataFrame,
        remove_unclustered: bool = True,
        col_filter: list = None,
    ):

        self.index_dict = self.get_cluster_dfs()
        # Remove Unclustered? -> at least for demo
        if remove_unclustered and len(self.unclustered_items) > 0:
            self.index_dict.pop("group_-1")

        target_cols = target.select_dtypes(include=np.number).dropna(axis=1).columns
        if col_filter:
            raw_cols = col_filter
        else:
            raw_cols = self.tupper.raw.select_dtypes(include=np.number).columns

        def error(x, mu):
            return abs((x - mu) / mu)

        scores = {}
        for group in self.index_dict:
            group_data = self.index_dict[group]
            score = 0
            for col in target_cols:
                if col in raw_cols:
                    x = target[col][0]
                    mu = group_data[col].mean()
                    score += error(x, mu)
            scores[group] = score

        min_index = min(scores, key=scores.get)
        return scores, min_index

    # Wut
    def fib(self, i, g, colorscale):
        return colorscale[i]

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
        pos = nx.spring_layout(self.mapper.graph, k=k, seed=seed)

        # Plot and color components
        components, labels = zip(*self.mapper.components.items())
        for i, g in enumerate(components):
            nx.draw_networkx(
                g,
                pos=pos,
                # node_color=color_scale[i],
                node_color=self.fib(i, g, color_scale),
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
        return plt

    def visualize_component(self, component, cluster_labels=True):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()

        color_scale = np.array(custom_color_scale()).T[1]
        components, labels = zip(*self.mapper.components.items())
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
            self.tupper.projection,
            self.tupper.get_projection_parameters(),
        )
        if show_color:
            fig = go.Figure()
            for g in np.unique(self.cluster_ids):
                label = f"Policy Group {int(g)}"
                if g == -1:
                    label = "Unclustered Items"
                mask = np.where(self.cluster_ids == g, True, False)
                cluster = projection[mask]
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
            for g in np.unique(self.cluster_ids):
                label = f"Policy Group {int(g)}"
                if g == -1:
                    label = "Unclustered Items"
                mask = np.where(self.cluster_ids == g, True, False)
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
                    *[
                        set(v["density"].keys())
                        for v in self.cluster_descriptions.values()
                    ]
                )
            )
        }
        cluster_descriptions = self.cluster_descriptions
        if cluster_descriptions[-1]["size"] == 0:
            cluster_descriptions.pop(-1)

        num_rows = math.ceil(max(len(cluster_descriptions) / 3, 1))
        print(len(cluster_descriptions))
        specs = get_subplot_specs(len(cluster_descriptions))

        dict_2 = {i: f"Group {i}" for i in range(len(cluster_descriptions))}
        dict_2 = {-1: "Outliers", **dict_2}
        fig = make_subplots(
            rows=num_rows,
            cols=3,
            specs=specs,
            subplot_titles=[
                f"<b>{dict_2[key]}</b>: {cluster_descriptions[key]['size']} Members"
                for key in cluster_descriptions
            ],
            horizontal_spacing=0.1,
        )

        for i, key in enumerate(cluster_descriptions):
            density = cluster_descriptions[key]["density"]

            labels = list(density.keys())
            sizes = list(density.values())

            # labels_list = [labels.get(item, item) for item in labels]

            row = i // 3 + 1
            col = i % 3 + 1
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    textinfo="percent",
                    values=sizes,
                    textposition="outside",
                    marker_colors=[color_map[l] for l in labels],
                    scalegroup=key,
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

        df = self.tupper.raw
        df["cluster_IDs"] = self.cluster_ids

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

        if -1.0 not in list(df.cluster_IDs.unique()):
            dict_2 = {i: str(i) for i in range(len(list(df.cluster_IDs.unique())))}
        else:
            dict_2 = {i: str(i) for i in range(len(list(df.cluster_IDs.unique())) - 1)}
        dict_2 = {-1: "Outliers", **dict_2}

        for column in df.columns.drop("cluster_IDs"):
            for pg in list(dict_2.keys()):
                fig.add_trace(
                    go.Box(
                        y=self.get_cluster_dfs()[f"group_{pg}"][column],
                        name=dict_2[pg],
                        jitter=0.3,
                        showlegend=False,
                        whiskerwidth=0.6,
                        marker_size=3,
                        line_width=1,
                        boxmean=True,
                        # hovertext=df["companyName"],
                        marker=dict(color=custom_color_scale()[int(pg)][1]),
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
            self.mapper.curvature,
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
        assert len(self.mapper.diagram) > 0, "Persistence Diagram is empty."
        persim_diagrams = [
            np.asarray(self.mapper.diagram[0]._pairs),
            np.asarray(self.mapper.diagram[1]._pairs),
        ]
        return plot_diagrams(persim_diagrams, show=True)


##########################################################################################
#
#   Unsupported Member functions
#
##########################################################################################


#    # Maybe its time we let the old ways die?


#     @DeprecationWarning
#     def visualize_mapper(self):
#         """
#         Plot using the Keppler Mapper html functionality.

#         NOTE: These visualizations are no longer maintained by KepplerMapper
#         and we do not reccomend using them.
#         """
#         assert len(self.complex) > 0, "Model needs a `fitted` mapper."
#         kepler = self.mapper.mapper
#         path_html = mapper_plot_outfile(self.hyper_parameters)
#         numeric_data, labels = config_plot_data(self.tupper)

#         colorscale = custom_color_scale()
#         # Use Kmapper Visualization
#         kepler.visualize(
#             self.mapper.complex,
#             node_color_function=["mean", "median", "std", "min", "max"],
#             color_values=numeric_data,
#             color_function_name=labels,
#             colorscale=colorscale,
#             path_html=path_html,
#         )

#         print(f"Go to {path_html} for a visualization of your JMapper!")
