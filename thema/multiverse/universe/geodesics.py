# File: multiverse/universe/geodesics.py
# Lasted Updated: 05/15/24
# Updated By: JW

import os
import pickle
from typing import Callable

import networkx as nx
import numpy as np
from grakel import GraphKernel
from grakel.utils import graph_from_networkx


def stellar_kernel_distance(
    files,
    filterfunction: Callable = None,
    kernel="shortest_path",
):
    """
    Compute a pairwise distance matrix between graphs based on `grakel` kernels.

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
    # Assign class 0 label to all nodes
    nx_graphs = []
    for sG in starGraphs.values():
        G = sG.graph
        node_labels = {node: 0 for node in G.nodes()}
        nx.set_node_attributes(G, node_labels, "class")
        nx_graphs.append(G)

    keys = list(starGraphs.keys())
    # Convert to GraKel Graphs
    G = graph_from_networkx(nx_graphs, node_labels_tag="class")
    gk = GraphKernel(kernel=kernel, normalize=True)

    # Distance as 1 - similarity
    distance_matrix = 1 - gk.fit_transform(G)
    return np.array(keys), distance_matrix


def _load_starGraphs(dir: str, graph_filter: Callable = None) -> dict:
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


def nofilterfunction(graphobject):
    return 1
