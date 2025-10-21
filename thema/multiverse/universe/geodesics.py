# File: multiverse/universe/geodesics.py
# Lasted Updated: 07/29/25
# Updated By: JW

import os
import pickle
from typing import Callable

import numpy as np
import networkx as nx
from scott import Comparator

from .utils.starFilters import nofilterfunction


def stellar_curvature_distance(
    files: str | list,
    filterfunction: Callable | None = None,
    curvature="ollivier_ricci_curvature",
    vectorization="landscape",
):
    """
    Compute a pairwise distance matrix between graphs based on curvature filtrations.

    Parameters
    ----------
    files : str or list
        Either a path to a directory containing starGraphs or a list of paths to starGraph files.
    filterfunction : Callable, optional
        A customizable filter function for pulling a subset of cosmic graphs.
        Default is None.
    curvature : str, optional
        The curvature measure to use. Default is "forman_curvature".
    vectorization : str, optional
        Vectorization method for computing distances. Default is "landscape".

    Returns
    -------
    keys : np.array
        List of keys for the models being compared.
    distance_matrix : np.ndarray
        Pairwise distance matrix between the persistence landscapes of the starGraphs.
    """
    # Detect if files is a list; if not, assume directory
    starGraphs = _load_starGraphs(files, graph_filter=filterfunction)

    keys = list(starGraphs.keys())
    starGraph_list = list(starGraphs.values())

    # Extract the actual NetworkX graphs
    graphs = [sg.graph for sg in starGraph_list]

    # Map string node IDs to integers for GUDHI compatibility
    mapped_graphs, node_mapping = _map_string_nodes_to_integers(graphs)

    # Create a Curvature Comparator
    C = Comparator(measure=curvature, weight="weight")

    n = len(mapped_graphs)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = C.fit_transform(
                [mapped_graphs[i]],
                [mapped_graphs[j]],
                metric=vectorization,
            )
            distance_matrix[i, j] = d_ij
            distance_matrix[j, i] = d_ij

    return np.array(keys), distance_matrix


def _load_starGraphs(
    dir: str | list, graph_filter: Callable | None = None
) -> dict:
    """
    Load starGraphs from a directory or a list of pickle files.
    Only returns starGraphs that satisfy the `graph_filter`.

    Parameters
    ----------
    dir : str or list
        Directory containing .pkl graphs, or a list of .pkl file paths.
    graph_filter : Callable, optional
        Function that returns True for graphs to include. Defaults to nofilterfunction.

    Returns
    -------
    dict
        Mapping of file path to starGraph object.
    """
    if graph_filter is None:
        graph_filter = nofilterfunction

    # Handle list vs directory
    if isinstance(dir, list):
        files = [str(f) for f in dir]  # ensure string paths
    else:
        assert os.path.isdir(dir), "Invalid graph Directory"
        assert len(os.listdir(dir)) > 0, "Graph directory appears to be empty!"
        files = [
            os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".pkl")
        ]

    if not files:
        raise ValueError("No .pkl files found to load.")

    starGraphs = {}
    for graph_file in files:
        with open(graph_file, "rb") as f:
            graph_object = pickle.load(f)

        if graph_filter(graph_object):
            if graph_object.starGraph is not None:
                starGraphs[graph_file] = graph_object.starGraph

    if not starGraphs:
        raise ValueError(
            "No valid starGraphs produced. Your filter function may be too stringent."
        )

    return starGraphs


def _map_string_nodes_to_integers(graphs):
    """
    Map string node IDs to integers for GUDHI compatibility.

    GUDHI's SimplexTree requires integer node IDs, but jmapStar creates
    graphs with string node IDs ('a', 'b', 'c', etc.). This function
    creates a consistent mapping across all graphs.

    Parameters
    ----------
    graphs : list
        List of networkx graphs that may have string node IDs

    Returns
    -------
    tuple
        (mapped_graphs, node_mapping) where mapped_graphs have integer
        node IDs and node_mapping is the string->int mapping dict
    """
    # Collect all unique nodes across all graphs
    all_nodes = set()
    for graph in graphs:
        all_nodes.update(graph.nodes())

    # Create consistent mapping from string nodes to integers
    node_mapping = {node: i for i, node in enumerate(sorted(all_nodes))}

    # Map all graphs to use integer node IDs
    mapped_graphs = []
    for graph in graphs:
        # Only remap if we have non-integer nodes
        if any(not isinstance(node, int) for node in graph.nodes()):
            mapped_graph = nx.relabel_nodes(graph, node_mapping)
            mapped_graphs.append(mapped_graph)
        else:
            # Graph already has integer nodes
            mapped_graphs.append(graph.copy())

    return mapped_graphs, node_mapping
