# File: multiverse/universe/geodesics.py
# Lasted Updated: 07/29/25
# Updated By: JW

import os
import pickle
from typing import Callable

import networkx as nx
import numpy as np
from scott import Comparator

from .utils.starFilters import nofilterfunction


def stellar_curvature_distance(
    files,
    filterfunction: Callable | None = None,
    curvature="forman_curvature",
    vectorization="landscape",
):
    """
    Compute a pairwise distance matrix between graphs based on curvature filtrations.

    Parameters
    ----------
    files : str
        A path pointing to the directory containing starGraphs.
    filterfunction : Callable, optional
        A customizable filter function for pulling a subset of cosmic graphs.
        Default is None.
    kernel : str, optional
        The kernel to be used for computing pairwise distances.
        Default is "shortest_path".

    Returns
    -------
    keys : np.array
        A list of the keys for the models being compared.
    distance_matrix : np.ndarray
        A pairwise distance matrix between the persistence landscapes of the starGraphs.
    """
    starGraphs = _load_starGraphs(files, filterfunction)
    keys = list(starGraphs.keys())
    # Convert starGraphs values to a list for indexed access
    starGraph_list = list(starGraphs.values())

    # Extract the actual networkx graphs
    graphs = [sg.graph for sg in starGraph_list]

    # Map string node IDs to integers for GUDHI compatibility
    mapped_graphs, node_mapping = _map_string_nodes_to_integers(graphs)

    # Create a Curvature Comparator
    C = Comparator(
        measure=curvature,
        weight="weight",
    )
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


def _load_starGraphs(dir: str, graph_filter: Callable | None = None) -> dict:
    """
    Load starGraphs in a given directory. This function only
    returns diagrams for starGraphs that satisfy the constraint
    given by `graph_filter`.

    Parameters
    ----------
    dir : str
        The directory containing the graphs,
        from which diagrams can be extracted.
    graph_filter : Callable, optional
        Default to None (ie no filter). Only select graph object based on filter
        function criteria (returns 1 to include and 0 to exclude)

    Returns
    -------
    dict
        A dictionary mapping the graph object file paths
        to the corresponding persistence diagram object.
    """

    assert os.path.isdir(dir), "Invalid graph Directory"
    assert len(os.listdir(dir)) > 0, "Graph directory appears to be empty!"

    if graph_filter is None:
        graph_filter = nofilterfunction

    starGraphs = {}
    for file in os.listdir(dir):
        if file.endswith(".pkl"):
            graph_file = os.path.join(dir, file)
            with open(graph_file, "rb") as f:
                graph_object = pickle.load(f)

            if graph_filter(graph_object):
                if graph_object.starGraph is not None:
                    starGraphs[graph_file] = graph_object.starGraph
    assert (
        len(starGraphs) > 0
    ), "You haven't produced any valid starGraphs. \
        Your filter function may be too stringent."

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
