# File: probe/telescope.py
# Last Update: 05/15/24
# Updated by: JW

import importlib
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

pio.renderers.default = "browser"

from .. import config
from .data_utils import get_nearestTarget, sunset_dict
from .visual_utils import (
    _column_color_mapping,
    _group_color_mapping,
    _match_column_order,
    _normalize_df,
    _reduce_colorOpacity,
)


class Telescope:
    """
    Telescope Class - to view star objects
        a suite to meet all your visualization needs

    Members
    ------
    pos: n-dimensional array
        node positioning data for graphs and components

    Functions
    ---------
    makeGraph()
        Visualize a graph!
    makeHeatmap()
        Visualize a breakdown of your connected components as a heatmap!
    makeSankey()
        Create a sankey diagram from a custom score_function
    makePathGraph()
        Creates a shortest-path graph for a single component, based based on a custom definition of target nodes

    Example
    --------
    >>> star_fp = '<PATH TO FILE>/jmap_clustererHDBSCANmin_cluster_size10_minIntersection-1_nCubes10_percOverlap0.6_id3_3.pkl'
    >>> telscope_instance = Telescope(star_fp)
    """

    def __init__(self, star_file):
        """
        Constructs a Telescope Instance

        Parameters
        ----------
        star_file : str
            filepath to a pickled star object

        """
        with open(star_file, "rb") as f:
            star_name = type(pickle.load(f)).__name__
        obv_configName = config.star_to_observatory[star_name]
        cfg = getattr(config, obv_configName)
        module = importlib.import_module(cfg.module)
        Observatory = module.initialize()
        self.observatory = Observatory(star_file)

        self._pos = nx.spring_layout(
            self.observatory.star.starGraph.graph, k=0.12, seed=6
        )

    @property
    def pos(self):
        """
        Get the position of the telescope.

        Returns
        -------
        numpy.ndarray
            The position of the telescope.

        Notes
        -----
        This member variable ensures that graph layouts are held constant when viewing
        graphs, groups/components, and path graphs. It is updated when updating seed and k
        in the `makeGraph()` and `makePathGraph()` functions.
        """
        return self._pos

    @pos.setter
    def pos(self, positions):
        """
        Set the positions to support resetting and storing positions across
        make<VISUAL>() functions.

        Parameters
        ----------
        positions : list or array-like
            The positions to be set.

        Returns
        -------
        None
        """
        self._pos = positions

    def makeGraph(
        self,
        group_number: int = None,
        k: float = None,
        seed: int = None,
        col: str = None,
        aggregation_func=None,
        hideLegend: bool = False,
        node_size_multiple: int = 10,
    ):
        """
        Visualize a graph!

        Parameters
        --------
        group_number : int
            graph connected component number to subset the visualization to
            For example, just show component 1 and not the entire graph

        k : float, default None
            value from 0-1, determines optimal distance between nodes
            setting nx.spring_layout positions

        seed : int, default None
            Random state for deterministic node layouts, defaulted so graph representations are reproducable
            setting nx.spring_layout positions

        col : str
            Column to color nodes by - from the raw data

        aggregation_func : np.<function>, defaults to calculating mean
            method by which to aggregate values within a node for coloring
                - color node by the median value or by the sum of values for example
            supports all numpy aggregation functions such as np.mean, np.median, np.sum, etc

        hideLegend : bool, default False
            toggle the graph/component's legend on or off

        node_size_multiple : int, 10
            change the node sizing

        ╭────────────────────────────────╮
        │   NODE SIZING OPTIONS -- WIP   |
        ╰────────────────────────────────╯

        Example
        ------
        Visualize connected component #3 with nodes colored by the sum of total pollution of coal plants in the node
        (example using a dataset on coal plant impacts)
        >>> tel = Telescope(star_filePath)
        >>> tel.makeGraph(group_number=3, col="Total Pollution", aggregation_func=np.sum)

        """
        assert aggregation_func is None or callable(
            aggregation_func
        ), "aggregation_func must be a function or None"

        if col is None and aggregation_func is not None:
            raise KeyError(
                "Cannot use a node-color aggregation function if coloring nodes by GROUP (CONNECTED COMPONENT)"
            )

        if group_number is None:
            G = self.observatory.star.starGraph.graph
        elif group_number in self.observatory.get_group_numbers():
            G = self.observatory.star.starGraph.components[group_number]
        else:
            raise ValueError("Group number not found in graph components list")

        fig = plt.figure(figsize=(14, 8), dpi=500)
        ax = fig.add_subplot()

        if col is not None:
            if aggregation_func is None:
                aggregation_func = lambda x: x.mean()
                aggregation_func.__name__ = "Mean"
            func_name = aggregation_func.__name__
            color_dict, colors, norm = _column_color_mapping(
                obs=self.observatory, col=col, aggregation_func=aggregation_func, G=G
            )
        else:
            color_dict, colors, norm = _group_color_mapping(obs=self.observatory, G=G)

        # --> size by number of items per node (note the *10 multiplier to increase node size)
        node_sizes = [
            len(attrs.get("membership", []) * node_size_multiple)
            for _, attrs in G.nodes(data=True)
        ]

        if k is not None or seed is not None:
            self.pos = nx.spring_layout(
                G, k=k if k is not None else 0.12, seed=seed if seed is not None else 6
            )

        # Gen Graph Viz
        nx.draw(
            G,
            pos=self.pos,
            node_color=colors,
            node_size=node_sizes,
            font_size=8,
            font_color="white",
            width=0.5,
            edgecolors="white",
        )

        ## --> create custom legend
        if not hideLegend:
            if col is None:
                unique_groups = list(set(color_dict.values()))
                legend_labels = [f"Group {group}" for group in unique_groups]
                legend_colors = plt.cm.coolwarm(norm(unique_groups))
                legend_handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=10,
                        label=label,
                    )
                    for color, label in zip(legend_colors, legend_labels)
                ]
                ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc="best",
                    fontsize="small",
                )
            else:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.65)
                ## --> label legend to indicate node-aggregation method
                cbar.set_label(
                    f"{col} -- Node {func_name}", fontsize="small", labelpad=10
                )

        plt.show()

    def makeHeatmap(
        self,
        nodeDescriptorCols: bool = True,
        ncols: int | list[str] = None,
        aggregation_func=None,
        topZscoreCols: bool = False,
    ):
        """
        Visualize a breakdown of your connected components!

        Parameters
        ---------
        ncols : int | List[Any], default 15
            int: the number of columns to visualize, selected from the front of your data
            list[str]: a list of specific columns from your data to create a heatmap of

        aggregation_func : np.<function>, defaults to calculating mean
            method by which to aggregate values within a node for coloring
                - color node by the median value or by the sum of values for example
            supports all numpy aggregation functions such as np.mean, np.median, np.sum, etc

        topZscoreCols : bool = False,
            visualize the ncols with the highest zscores with a group -- in other words, the columns in which one
            or more groups is the MOST different than the dataset norm

            Overrides a ncols int specification

        nodeDescriptorCols : bool = True,
            Smart select columns to view in your heatmap, based on a density representing the
            composition of a group by its nodes' descriptions.

            Overrides a ncols int specification

        Returns
        --------
        n/a : displays an inline matplotlib.plt

        ╭──────────────────────────────────────────────────────────────────────────────────────────────────────╮
        │ TODO - Dynamically select `ncols` based on cols w/ highest variance between groups for default viz   |
        ╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯

        Example
        --------
        >>> tel = Telescope(star_filePath)
        >>> tel.makeHeatmap(ncols=['Pollution', 'Health Impact'], aggregation_func=np.mean)

        """
        assert isinstance(
            ncols, (int, list, type(None))
        ), "`ncols` must be an integer, a list, or None!"

        assert aggregation_func is None or callable(
            aggregation_func
        ), "aggregation_func must be a function or None"

        data = self.observatory.get_aggregatedGroupDf(aggregation_func).drop(
            columns=["Group"]
        )
        if isinstance(ncols, list):
            if all(col in self.observatory.clean.columns for col in ncols):
                data = data[ncols]
            else:
                raise UserWarning(
                    "Please ensure `ncols` contains columns from the CLEAN dataset"
                )
        ## --> dynamic col selection implemented, based on highest zscores
        elif topZscoreCols:
            zscore_cols = list(self.observatory.dataset_zscores_df(ncols).columns)
            data = data[zscore_cols]
        ## --> dynamic col selection implemented, based on node descriptors
        elif nodeDescriptorCols:
            nodeIdentifiers = self.observatory.get_group_descriptions()
            sub_keys = set()
            for _, sub_dict in nodeIdentifiers.items():
                sub_keys.update(sub_dict.keys())

            sub_keys_list = list(sub_keys)
            data = data[sub_keys_list]
        else:
            data = (
                self.observatory.get_aggregatedGroupDf(aggregation_func)
                .iloc[:, :ncols]
                .drop(columns=["Group"])
            )

        ## --> format annotations so base-data is clean/encoded/scaled and box labeling is based on the raw data
        annotations = self.observatory.get_aggregatedGroupDf(
            aggregation_func, clean=False
        ).drop(columns=["Group"])
        annotations = _match_column_order(data, annotations)
        normalized_df = _normalize_df(data)

        plt.figure(figsize=(12, 3), dpi=1000)
        ax = sns.heatmap(
            normalized_df,
            annot=annotations,
            # annot=normalized_df,
            fmt=".2f",
            # vmin=0,
            # vmax=1,
            cbar=False,
            annot_kws={"size": 7},
            cmap="coolwarm",
            linewidths=0.5,
        )

        ## --> reduce font size + format annotations to ensure text does not run out of heatmap box
        for text in ax.texts:
            value = text.get_text()
            num_digits = len(value.replace(".", ""))
            if num_digits <= 6:
                text.set_fontsize(6)
            elif 7 <= num_digits <= 5:
                text.set_fontsize(5)
            elif num_digits == 9:
                text.set_fontsize(4)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=10)

        # --> to remove xaxis labels
        # plt.xticks([])

        plt.ylabel("Group Number")
        plt.show()

    def makeSankey(
        self, score_function, dropUnclustered: bool = True, title_text: str = None
    ):
        """
        Creates a Sankey Diagram based on the score function.

        Parameters
        ----------
        score_function: function, pd:DataFrame -> List
            score_function must take in a dataframe and return a classification (categorical)
            of elements.

        Example
        ------
        Assuming data has columns "height" and "age" columns, one could define a score
        function as follows:

        ```
        def my_score_function(df):
            scores = 0.5 * df['height'] + 2 * df['age']
            labels = ['high' if score > 20 else 'low' for score in scores]
            return labels
        ```

        """

        sankey_df = pd.DataFrame()
        sankey_df["Group"] = self.observatory.data.index.map(
            self.observatory.get_items_groupID
        )
        sankey_df["Labels"] = score_function(self.observatory.data)

        if dropUnclustered:
            sankey_df = sankey_df[sankey_df["Group"] != -1]

        sankey_df = (
            sankey_df.groupby(["Group", "Labels"]).size().reset_index(name="Value")
        )

        # Create nodes from unique sources and targets
        nodes = list(set(sankey_df["Group"]).union(set(sankey_df["Labels"])))

        # Create edges
        edges = []
        for _, row in sankey_df.iterrows():
            edges.append(
                (nodes.index(row["Group"]), nodes.index(row["Labels"]), row["Value"])
            )

        ## TODO --> color TARGETS differently (red:high, blue:low for example)
        # Also add interpolation between landmark colors for larger Sankeys

        num_nodes = len(nodes)
        colors = px.colors.sample_colorscale(
            px.colors.sequential.RdBu_r, [n / (num_nodes - 1) for n in range(num_nodes)]
        )

        link_colors = _reduce_colorOpacity(colors, opacity=0.3)

        # Create Sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=nodes,
                        color=[colors[i] for i in range(len(nodes))],
                    ),
                    link=dict(
                        source=[edge[0] for edge in edges],
                        target=[edge[1] for edge in edges],
                        value=[edge[2] for edge in edges],
                        color=[link_colors[edge[0]] for edge in edges],
                    ),
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title_text=title_text,
            font_size=10,
            template="simple_white",
        )

        fig.show()

    def makePathGraph(
        self,
        col: str,
        group_number: int,
        aggregation_func=None,
        top: bool = True,
        percentage: float = 0.1,
        path_labels: bool = False,
        node_labels: bool = False,
        k: float = None,
        seed: int = None,
        node_size_multiple: int = 10,
    ):
        """
        Make a shortest-path graph by identifying sink (target) nodes and
        visualizing distance to them

        Parameters
        ----------
        col : str
            Column to color nodes by - from the raw data

        group_number : int
            graph connected component number to subset the visualization to
            For example, just show component 1 and not the entire graph

        aggregation_func : np.<function>, defaults to calculating mean
            method by which to aggregate values within a node for coloring
                - color node by the median value or by the sum of values
                for example supports all numpy aggregation functions such as
                np.mean, np.median, np.sum, etc

        top : bool, default True
            Whether to select the top n percentage or the bottom n percentage
            of nodes as target/sink nodes
            NOTE: corresponds to the `percentage` param

        percentage : float
            The n-th percentage of nodes to select as sinks/targets
            NOTE: corresponds to the `top` param

        labels : bool, False
            Add text labeling target nodes and which sink is closet
            to non-targets

        k : float, default 0.12
            value from 0-1, determines optimal distance between nodes
            setting nx.spring_layout positions

        seed : int, default 12
            Random state for deterministic node layouts, defaulted so graph
            representations are reproducible setting nx.spring_layout positions

        node_size_multiple : int, 10
            change the node sizing

        path_labels : bool, default False
            add labels to the nodes indicating target nodes, and the target
            that each node is closest to.

        node_labels : bool, default False
            add labels to the nodes, showing their node IDs for getting
            node-level data.
        """

        node_values = self.observatory.define_nodeValueDict(
            group_number, col, aggregation_func
        )
        target_nodes = sunset_dict(node_values, percentage=percentage, top=top)
        G = self.observatory.star.starGraph.components[group_number]
        nearest_target, nearest_target_distance = get_nearestTarget(
            G,
            target_nodes,
        )

        fig = plt.figure(figsize=(14, 8), dpi=500)
        ax = fig.add_subplot()
        node_sizes = [
            len(attrs.get("membership", []) * node_size_multiple)
            for _, attrs in G.nodes(data=True)
        ]
        norm = plt.Normalize(
            min(nearest_target_distance.values()),
            max(
                nearest_target_distance.values(),
            ),
        )

        if k is not None or seed is not None:
            self.pos = nx.spring_layout(
                G,
                k=k if k is not None else 0.12,
                seed=seed if seed is not None else 6,
            )

        colors = plt.cm.tab20c(norm(list(nearest_target_distance.values())))

        nx.draw(
            G,
            pos=self.pos,
            with_labels=node_labels,
            node_color=colors,
            edgecolors="white",
            node_size=node_sizes,
            font_size=8,
            font_color="white",
            width=0.5,
        )
        ## --> TODO dynamic node sizing
        target_sizes = 400
        nx.draw_networkx_nodes(
            G,
            pos=self.pos,
            nodelist=target_nodes,
            node_color="#3182BD",
            edgecolors="black",
            linewidths=1.5,
            node_size=target_sizes,
        )
        ## --> add "S" labels inside sink nodes to represent they are targets
        if not node_labels:
            for target in target_nodes:
                plt.text(
                    self.pos[target][0],
                    self.pos[target][1],
                    "S",
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        if path_labels:
            ## --> add labels to the target nodes
            for target in target_nodes:
                plt.text(
                    self.pos[target][0],
                    self.pos[target][1],
                    "TARGET",
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="center",
                )
            ## --> add labels to indicate which target is the closest to each non-target node
            for node, target in nearest_target.items():
                if target is not None:
                    if node != target:
                        plt.text(
                            self.pos[node][0],
                            self.pos[node][1],
                            f"Nearest: {target}",
                            color="black",
                            fontsize=8,
                            ha="center",
                            va="center",
                        )

        sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20c, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label="Distance to Nearest Sink", shrink=0.75, ax=ax)
        plt.show()
