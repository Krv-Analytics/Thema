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

from sklearn.preprocessing import LabelEncoder

# TODO: Move this to a defaulted argument for the viz functions
pio.renderers.default = "browser"

########################################################################################
#
#   Handling Local Imports
#
########################################################################################


from visual_utils import custom_color_scale, get_subplot_specs, reorder_colors, interactive_visualization

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
    jmapper: <jmapper.JMapper>
        A JMapper Object

    hyperparaemters: np.array
        An np.array of the hyperparameters for generating the associated JMapper

    Member Functions
    ----------------

    visualize_model:
        Displays networkx graph representation of JMapper model

    visualize_component:
        Displays networkx graph representation of specified CC




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
    
    
    def _define_nx_labels_colors(self, col=None, external_col=None, statistic='average'):
        """
        Helper Function for visualize_model()
        Colors a nx graph by col.
        - defaults to coloring by group
        - ability to color by any col in the raw data or statistics (average, median, min, max, std deviation)

        Parameters
        -------
        all determined by visualize_model()
        - col: str or None, the column name in the raw data to use for coloring (default: None)
        - external_col: None (not fully supported yet)
        - statistic: str, the statistic to use for coloring ('average', 'median', 'min', 'max', 'std' for standard deviation)
        """
        g = self.jmapper.jgraph.graph
        color_dict = {}
        labels_dict = {}

        # Define a helper function to calculate statistics
        def calculate_statistic(node, a, col, statistic):
            if statistic == 'average':
                return self.raw.iloc[a].mean()[col]
            elif statistic == 'median':
                return self.raw.iloc[a].median()[col]
            elif statistic == 'sum':
                return self.raw.iloc[a].sum()[col]
            elif statistic == 'min':
                return self.raw.iloc[a].min()[col]
            elif statistic == 'max':
                return self.raw.iloc[a].max()[col]
            elif statistic == 'std':
                return self.raw.iloc[a].std()[col]
            else:
                raise ValueError("Invalid statistic parameter")

        # to color by connected component group number
        if col is None:
            for node in g.nodes:
                ave = self.get_nodes_groupID(node)
                color_dict[node] = ave
                labels_dict[node] = node

        # to color by a column in the raw df
        elif col in self.raw.columns:
            # color by categorical variables
            if self.raw[col].dtype == 'object':
                encoder = LabelEncoder()
                encoded_col = encoder.fit_transform(self.raw[col])

                for node, encoded_value in zip(g.nodes, encoded_col):
                    color_dict[node] = encoded_value
                    labels_dict[node] = node
            # color by non-categorical variables
            else:
                for node in g.nodes:
                    a = self.get_nodes_raw_df(node).index
                    value = calculate_statistic(node, a, col, statistic)
                    color_dict[node] = value
                    labels_dict[node] = node

        # to color by an external column
        elif external_col == external_col:
            print('coloring by external data not supported at this time')
        else:
            print('error coloring')

        return color_dict, labels_dict

    def calculate_node_sizes(self, col_name=None, node_size_multiplier=10, target_range=(10, 300), aggregation_method='sum'):
        g = self.jmapper.jgraph.graph
        
        if col_name is None:
            # Default behavior: size nodes by the number of items per node
            node_sizes = [len(self.get_nodes_members(node)) * node_size_multiplier for node in g.nodes]
        else:
            # Check the aggregation method and calculate node sizes accordingly
            if aggregation_method == 'mean':
                column_values = [self.get_nodes_raw_df(node)[col_name].mean() * node_size_multiplier for node in g.nodes]
            elif aggregation_method == 'sum':
                column_values = [self.get_nodes_raw_df(node)[col_name].sum() * node_size_multiplier for node in g.nodes]
            elif aggregation_method == 'median':
                column_values = [self.get_nodes_raw_df(node)[col_name].median() * node_size_multiplier for node in g.nodes]
            else:
                raise ValueError("Invalid aggregation method. Supported methods: 'mean', 'sum'")
            
            min_value = min(column_values)
            max_value = max(column_values)
            
            if min_value == max_value:
                # Avoid division by zero if all values are the same
                scaling_factor = 1.0
            else:
                scaling_factor = (target_range[1] - target_range[0]) / (max_value - min_value)
            
            # Apply the scaling factor to the node sizes
            node_sizes = [(value - min_value) * scaling_factor + target_range[0] for value in column_values]
        
        return node_sizes


    def _get_connected_component_label_positions(self):
        """
        Helper Function for visualize_model()
        Adds Group/Cluster Labels to the graph for easier tracking of groups across different color schemes.
        - can be toggled on and off via parameters in visualize_model()
        - TODO center and offset group labels, instead of putting them over first node
            - the commented out code is starter code (not working yet) to change the position of group labels

        Parameters
        -------
        all determined by visualize_model()

        """
        #label_pos = {}
        label_labels = {}
        for group in self._group_directory.keys():
            temp=self.get_groups_member_nodes(group)[0]
            #label_pos[temp] = np.random.rand(1,2)
            label_labels[temp]=f"Group {group}"
        return label_labels #,label_pos
    

    def visualize_model(
                self, col=None, node_size_col = None, node_size_multiplier=10, node_size_aggregation_method='sum', #general params
                node_color_method='average', node_edge_width=0.5, node_edge_color='black', #node viz params
                legend_bar=False, group_labels=False, node_labels=False, show_edge_weights=False, #graph labeling & legend params
                spring_layout_seed=8, k=None, #spring layout params
                matplotlib_cmap = 'coolwarm', figsize=(8, 6), dpi=400, #matplotlib params
            ):
        """
        Visualize the clustering as a network. This function plots
        the JMapper's graph in a matplotlib figure, coloring the nodes
        by their respective policy group or col parameter

        Parameters
        -------
        col : df column, default is none
            Parameter to determine graph coloring
            If none, will default to color by group (connected component)

        node_size_multiplier : int, default 10
            Node sizes are determined by the number of items per node
            For datasets with very large (or small) groups, tuning this parameter ensures
            the entire graph can be visualized in a reasonable manner
                TODO: implement a smarter way to size nodes, based on diff between number of items
                in largest node and smallest node

        node_edge_width : int, default 0.5
            Determines the weight of the node's outline
            Parameter might need to be adjusted when visualizing dense nodes, to ensure edge weights do not cover nodes

        legend_bar : boolean, default false
            to toggle on/off the matplotlib legend bar

        group_labels : boolean, default false
            to toggle on and off group label text being placed over each group to identify group numbers

        node_labels : boolean, default false 
            to toggle on and off node label text being placed over each node to identify node's letter identifier                         
        
        show_edge_weights : boolean, default false
            to toggle on and off edge weight visualizations
            - text on edges to indicate weights
            - dashed vs solid edges at different widths to contrast different edge weights

        spring_layout_seed : int, default 8
                             random seed to change the spring layout

        k : float, default is None
            Optimal distance between nodes. If None the distance is set to
            1/sqrt(n) where n is the number of nodes. Increase this value to
            move nodes farther apart.

        matplotlib_cmap : string, default coolwarm
            colorscale from matplotlib to color nodes by, supports all matplotlib colorscales

        figsize : touple, default (8, 6)
            parameter to size graph by

        dpi : int, default 500
            resulution of the plot: dots per inch.
                TODO: implement a smarter way to assign a dpi, as this will have to be reduced
                for very large datasets to ensure the visulization can be made in a resonable amount of time

                NOTE: if you are getting weird errors, reduce the DPI. Matplotlib has trouble when dpi is too high

        """
        g = self.jmapper.jgraph.graph

        colors_dict, labels_dict = self._define_nx_labels_colors(col=col, statistic=node_color_method)
        pos = nx.spring_layout(g, seed=spring_layout_seed, k=k)
        self._cluster_positions = pos

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_frame_on(False)  # Remove the black border
        ax.set_xticks([])
        ax.set_yticks([])

        node_sizes = self.calculate_node_sizes(col_name=node_size_col, node_size_multiplier=node_size_multiplier, aggregation_method=node_size_aggregation_method)

        nc = nx.draw_networkx_nodes(
            g, 
            pos,
            node_color=list(colors_dict.values()), 
            node_size=node_sizes, 
            ax=ax, 
            cmap=matplotlib_cmap,
            linewidths=node_edge_width,
            edgecolors=node_edge_color,
            )

        # for weighted graphs, to visualize differently weighted edges differently
        if show_edge_weights:
            #split edges into two groups based on weights, for different visualizations
            #TODO catch an error and print a message if no edge weights associated with graph
            elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d["weight"] > 0.5]
            esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d["weight"] <= 0.5]
            nx.draw_networkx_edges(g,
                pos,
                edgelist=elarge, 
                width=1,
                ax=ax
            )
            nx.draw_networkx_edges(g,
                pos,
                edgelist=esmall,
                width=1,
                alpha=0.5,
                edge_color="black",
                style="dashed",
                ax=ax
            )
        else:
            # draw all edges the same, works for both weighted and non-weighted graphs
            nx.draw_networkx_edges(g,
                pos,
                edgelist=g.edges,
                width=1,
                edge_color="grey",
                ax=ax
            )

        if node_labels:
            nx.draw_networkx_labels(g, pos, labels=labels_dict)
        if group_labels:
            nx.draw_networkx_labels(g, pos, labels=self._get_connected_component_label_positions())
        if legend_bar:
            plt.colorbar(nc)

        #for weighted graphs, to add edge weight labels
        if show_edge_weights:
            edge_weight_labels = nx.get_edge_attributes(g,'weight')
            edge_label_options = {
                'edge_labels': edge_weight_labels,
                'pos': pos,
                'font_size': 7,
                'font_color': 'black',  
                'alpha': 1.0,  
                'bbox': dict(facecolor='none', edgecolor='none') 
            }
            nx.draw_networkx_edge_labels(g, **edge_label_options, ax=ax)

        # adjusting so group labels stay in-bounds
        plt.subplots_adjust(left=-0.3)

    def interactive_model(self):
        return interactive_visualization(self)

    def visualize_component(self, component, cluster_labels=True):
        """
        Plots only the specified connected component.

        Parameters
        ----------
        component: int
            The group identifier correpsonding to the desired connected component.

        cluster_labels: bool
            Shows node labels in nx plot.

        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()

        color_scale = np.array(custom_color_scale()).T[1]
        labels, components = zip(*self.jmapper.jgraph.components.items())
        for i, g in enumerate(components):
            if i == component:
                nx.draw_networkx(
                    g,
                    pos=self._cluster_positions,
                    node_color=color_scale[1],
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
        """
        A PieChart Visualization of group descriptions.
        """
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
        y_shift = -.35

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
        """
        Generates an ensemble of box plots of groups over the specified data columns.

        Parameters
        ----------
        cols: list
            A list of the desired data columns to be visualized.

        target: pd.DataFrame
            A data frame with one row to plot target line in boxplot.
        """

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
