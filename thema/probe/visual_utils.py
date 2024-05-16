# File: probe/visual_utils.py
# Last Update: 05/16/24
# Updated by: JW

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from .observatory import Observatory


def _find_matching_indexes(big_dict, small_dict):
    """
    Helper function for coloring.

    Find the indexes in a larger dictionary where the values in a smaller
    dictionary are found. This function is used to subset color schemes to the
    correct size when visualizing groups. It ensures that the component is
    colored the same way it was colored on the full graph.

    Parameters:
    ----------
    big_dict : dict
        The larger dictionary to search in.

    small_dict : dict
        The smaller dictionary to search for.

    Returns:
    -------
    matching_indexes : list
        A list of indexes in the larger dictionary where the values in the
        smaller dictionary are found.

    See Also:
    -------
    visual_utils._group_color_mapping() for usage

    """
    matching_indexes = [
        i for i, item in enumerate(big_dict.items()) if item in small_dict.items()
    ]
    return matching_indexes


def _normalize_coloring(color_dict: dict, G: nx.Graph):
    """
    Helper function to help create a matplotlib legend

    Parameters
    ----------
    color_dict : dict
        A dictionary mapping each node to its color value.

    G : nx.Graph
        The graph object.

    Returns
    -------
    normalized_colors : array
        A 2D array of RGB colors.

    norm_object : matplotlib.colors.Normalize
        A normalized colorscale object used by matplotlib.

    See Also
    -------
    visual_utils._group_color_mapping() for usage
    visual_utils._column_color_mapping() for usage
    """
    norm_object = Normalize(
        vmin=min(color_dict.values()), vmax=max(color_dict.values())
    )
    normalized_colors = plt.cm.coolwarm(
        norm_object([color_dict[node] for node in G.nodes()])
    )
    return normalized_colors, norm_object


def _group_color_mapping(obs: Observatory, G=None):
    """
    Helper function: color graph nodes by GROUP/CONNECTED COMPONENT

    This function ensures that color scales are held constant across graph-level and component-level
    visualizations, so sub-graph visuals are easy to trace back to the original graph

    Parameters
    ----------
    obs : Observatory
        The Observatory object for using observatory functionality to get parse graph data.

    G : nx.Graph, optional
        The graph to parse nodes and info from.

    Returns
    -------
    color_dict : dict
        A mapping of each node to its group, which is used to parse the normalized_colors.

    normalized_colors : array
        A 2D array of RGB colors.

    norm_object : matplotlib.colors.Normalize
        A normalized colorscale object used by matplotlib.

    See Also
    -------
    telescope.makeGraph() for usage of this function
    """
    if G is not None:
        subset_dict = True
        component = G
        sub_color_dict = {}
        for node in component.nodes:
            ave = obs.get_nodes_groupID(node)
            sub_color_dict[node] = ave

    G = obs.star.starGraph.graph

    color_dict = {}
    for node in G.nodes:
        ave = obs.get_nodes_groupID(node)
        color_dict[node] = ave

    normalized_colors, norm_object = _normalize_coloring(color_dict, G)

    # ensure single-component coloring matches total graph color scheme -> component 3 should not be red when viewing graph and blue when viewing component
    if subset_dict:
        inds = _find_matching_indexes(big_dict=color_dict, small_dict=sub_color_dict)
        normalized_colors = normalized_colors[inds]
        # keep legend representative of all coloring
        _, norm_object = _normalize_coloring(color_dict, component)

    return color_dict, normalized_colors, norm_object


def _column_color_mapping(obs: Observatory, col, aggregation_func, G=None):
    """
    Helper function: color graph nodes by A COLUMN IN THE RAW DATA

    Parameters
    ----------
    obs : Observatory
        The Observatory object for using observatory functionality to get parse graph data.

    col : str
        The column from the raw dataset to color nodes by.

    aggregation_func : np.<function>
        The method by which to aggregate values within a node for coloring.
        Supports all numpy aggregation functions such as np.mean, np.median, np.sum, etc.

    G : nx.Graph, optional
        The graph to parse nodes and info from.

    Returns
    -------
    color_dict : dict
        A mapping of each node to its group, which is used to parse the normalized_colors.

    normalized_colors : array
        A 2D array of RGB colors.

    norm_object : matplotlib.colors.Normalize
        A normalized colorscale object used by matplotlib.

    See Also
    -------
    telescope.makeGraph() for usage of this function
    """
    if G is None:
        G = obs.star.starGraph.graph

    df = obs.data
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the raw data columns.")
    if df[col].dtype == "object":
        raise ValueError(
            "Coloring by non-numeric columns not yet supported, please enter a different value for `col`"
        )

    color_dict = {}
    for node in G.nodes:
        a = obs.get_nodes_raw_df(node).index
        value = aggregation_func(df.iloc[a][col])
        color_dict[node] = value

    normalized_colors, norm_object = _normalize_coloring(color_dict, G)
    return color_dict, normalized_colors, norm_object


def _match_column_order(data, annotations):
    """
    Match the column order of the annotations DataFrame to match the data DataFrame.
    For missing columns in annotations, create them and fill with empty strings.

    Parameters
    ----------
    data : DataFrame
        The DataFrame with the desired column order.

    annotations : DataFrame
        The DataFrame to be adjusted.

    Returns
    -------
    annotations_ordered : DataFrame
        The adjusted annotations DataFrame.
    """
    annotations_ordered = pd.DataFrame()

    for col in data.columns:
        if col in annotations.columns:
            annotations_ordered[col] = annotations[col]
        else:
            annotations_ordered[col] = np.nan

    return annotations_ordered


def _normalize_df(data):
    """
    Normalize a DataFrame for easy heatmap visualization.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to be normalized.

    Returns
    -------
    DataFrame
        The normalized DataFrame.
    """
    return (data - data.min()) / (data.max() - data.min())


def _reduce_colorOpacity(colors, opacity):
    """
    Reduce the opacity of a list of colors to a specified opacity level.

    Parameters
    ----------
    colors : list
        List of colors in RGB format.

    opacity : float
        Desired opacity level, between 0 and 1.

    Returns
    -------
    list
        List of colors with reduced opacity.
    """
    return [f"rgba{color[3:-1]}, {opacity})" for color in colors]
