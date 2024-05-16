# File: probe/data_utils.py
# Last Update: 05/16/24
# Updated by: JW

import warnings

import networkx as nx
import numpy as np
import pandas as pd

# ╭──────────────────────────────────────────────────────────────────╮
# │               node_desription Helper functions                   │
# ╰──────────────────────────────────────────────────────────────────╯


def get_minimal_std(df: pd.DataFrame, mask: np.array, density_cols=None):
    """
    Find the column with the minimal standard deviation
    within a subset of a Dataframe.

    Parameters
    -----------
    df: pd.Dataframe
        A cleaned dataframe.

    mask: np.array
        A boolean array indicating which indices of the dataframe
        should be included in the computation.

    Returns
    -----------
    col_label: int
        The index idenitfier for the column in the dataframe with minimal std.

    """
    if density_cols is None:
        density_cols = df.columns
    sub_df = df.iloc[mask][density_cols]
    col_label = sub_df.columns[sub_df.std(axis=0).argmin()]
    return col_label


# ╭──────────────────────────────────────────────────────────────────╮
# │               group_identity Helper functions                    │
# ╰──────────────────────────────────────────────────────────────────╯


# Filter functions that assign a value to columns => minimum is taken to
# be the most important columns


def std_zscore_threshold_filter(
    col, global_stats: dict, std_threshold=1, zscore_threshold=1
):
    """
    Calculate the filter value based on the standard deviation and z-score of a column.

    Parameters
    ----------
    col : pd.Series
        The column for which to calculate the filter value.

    global_stats : dict
        A dictionary containing global statistics for the dataset.

    std_threshold : float, optional
        The threshold for the standard deviation. Columns with absolute standard deviation
        below this threshold will be filtered out. Default is 1.

    zscore_threshold : float, optional
        The threshold for the z-score. Columns with absolute z-score above this threshold
        will be filtered out. Default is 1.

    Returns
    -------
    int
        The filter value. 0 if the column should be filtered out, 1 otherwise.
    """
    std = np.std(col)
    if std == 0:
        zscore = np.inf
    else:
        zscore = (np.mean(col) - global_stats["clean"]["mean"][col.name]) / std

    if abs(zscore) > zscore_threshold and abs(std) < std_threshold:
        return 0
    else:
        return 1


def get_best_std_filter(col, global_stats: dict):
    """
    Calculate the standard deviation of a column.

    Parameters
    ----------
    col : pd.Series
        The column for which to calculate the standard deviation.

    global_stats : dict
        A dictionary containing global statistics for the dataset.

    Returns
    -------
    float
        The standard deviation of the column.
    """
    std = np.std(col)
    return std


def get_best_zscore_filter(col, global_stats: dict):
    """
    Calculate the z-score of a column.

    Parameters
    ----------
    col : pd.Series
        The column for which to calculate the z-score.

    global_stats : dict
        A dictionary containing global statistics for the dataset.

    Returns
    -------
    float
        The z-score of the column.
    """
    zscore = (np.mean(col) - global_stats["clean"]["mean"][col.name]) / np.std(col)

    return zscore


# ╭──────────────────────────────────────────────────────────────────╮
# │                        Auxillary functions                       │
# ╰──────────────────────────────────────────────────────────────────╯


def error(x, mu):
    """
    Calculate the error between a value and its expected value.

    Parameters
    ----------
    x : float
        The value.

    mu : float
        The expected value.

    Returns
    -------
    float
        The error between the value and its expected value.
    """
    return abs((x - mu) / mu)


# ╭──────────────────────────────────────────────────────────────────────────────────────╮
# │    helpers for manipluating data in telescope/observatory class for visualization    │
# ╰──────────────────────────────────────────────────────────────────────────────────────╯


def sunset_dict(d: dict, percentage: float = 0.1, top: bool = True) -> dict:
    """
    Return the top/bottom n percentage of a dictionary based on values.

    Parameters
    ----------
    d : dict
        The dictionary to subset, with node : value mappings.

    percentage : float, optional
        The percentage of the dictionary to take when subsetting to contain the top n%
        of values. Default is 0.1.

    top : bool, optional
        If True, take the top percentage. If False, take the bottom percentage.
        Default is True.

    Returns
    -------
    dict
        A dictionary containing only the nodes and their values that made the cut based
        on the n percentage.
    """
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=top)
    top_percent = int(len(sorted_items) * percentage)
    if top:
        selected_items = sorted_items[:top_percent]
    else:
        selected_items = sorted_items[-top_percent:]
    if len(selected_items) == 0:
        warnings.warn(
            f"Subsetting to top {percentage} creates an empty dict, selecting the {'top' if top else 'bottom'} node as a sink/target."
        )
        return (
            {sorted_items[0][0]: sorted_items[0][1]}
            if top
            else {sorted_items[-1][0]: sorted_items[-1][1]}
        )
    return dict(selected_items)


def get_nearestTarget(G: nx.Graph, targets: dict):
    """
    Get the nodes and corresponding distances that are closest to each target.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    targets : dictionary
        A dictionary of target nodes and their aggregated values, obtained from the
        sunset_dict() function.

    Returns
    -------
    nearest_target : dict
        A dictionary where keys are nodes in the graph and values are the nearest target node.

    nearest_target_distance : dict
        A dictionary where keys are nodes in the graph and values are the shortest distance
        to the nearest target node.
    """
    # Initialize dictionaries to store distance to nearest target and the nearest target node
    nearest_target_distance = {node: float("inf") for node in G.nodes()}
    nearest_target = {node: None for node in G.nodes()}

    # Loop through all nodes and find the nearest target node
    for node in G.nodes():
        for target in targets:
            path_length = nx.shortest_path_length(
                G, source=node, target=target, weight="weight"
            )
            if path_length < nearest_target_distance[node]:
                nearest_target_distance[node] = path_length
                nearest_target[node] = target

    return nearest_target, nearest_target_distance


def custom_Zscore(global_df, subset_df, column_name):
    """
    Calculate the z-score for a subset of a DataFrame relative to the entire DataFrame.

    Parameters
    ----------
    global_df : pd.DataFrame
        The entire DataFrame containing the global dataset.

    subset_df : pd.DataFrame
        The subset of the DataFrame for which to calculate the z-score.

    column_name : str
        The name of the column in both DataFrames for which to calculate the z-score.

    Returns
    -------
    float
        The z-score of the subset relative to the global dataset.
    """
    subset_mean = subset_df[column_name].mean()
    global_mean = global_df[column_name].mean()
    global_std = global_df[column_name].std()

    z_score = (subset_mean - global_mean) / global_std

    return z_score


def select_highestZscoreCols(zscores, n_cols):
    """
    Select the columns in a DataFrame that have the highest absolute z-scores.

    Parameters
    ----------
    zscores : pd.DataFrame
        A DataFrame containing z-scores.

    n_cols : int
        The number of columns to select with the highest absolute z-scores.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the top n columns with the highest absolute z-scores.
    """
    max_abs_values = zscores.abs().max()
    sorted_columns = max_abs_values.sort_values(ascending=False)
    top_n_columns = sorted_columns.head(n_cols).index.tolist()
    return zscores[top_n_columns]
