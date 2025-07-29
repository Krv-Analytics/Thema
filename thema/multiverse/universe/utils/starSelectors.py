# File: multiverse/universe/starSelectors.py
# Lasted Updated: 07/29/25
# Updated By: JW

import random as rand
import numpy as np
import pickle


def random(subgroup):
    return rand.choice(subgroup)


def max_nodes(subgroup):
    """
    Returns the file path of the graph with the highest number of nodes.

    Parameters:
    -----------
    subgroup: list
        A list of file paths to saved graph files.

    Returns:
    --------
    largest_graph_file: str
        The file path of the graph with the highest number of nodes.
    """

    node_counts = []
    for graph_file in subgroup:
        with open(graph_file, "rb") as file:
            graph_data = pickle.load(file)
        node_count = len(graph_data.complex["nodes"])
        node_counts.append(node_count)

    assert len(node_counts) == len(subgroup)
    largest_graph_file = subgroup[np.argmax(node_counts)]
    return largest_graph_file
