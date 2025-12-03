import numpy as np
import itertools
from collections import defaultdict
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN


def mapper_pseudo_laplacian(complex, n, components, neighborhood="node") -> np.ndarray:
    """Calculates and returns a pseudo laplacian n by n matrix representing neighborhoods in the graph. Here, n corresponds to
    the number of items (ie rows in the clean data - keep in mind some raw data rows may have been dropped in cleaning). Here,
    the diagonal element A_ii represents the number of neighborhoods item i appears in. The element A_ij represent the number of
    neighborhoods both item i and j belong to.

    Parameters
    ----------
    neighborhood: str
        Specifies the type of neighborhood. For jmapStar, neighborhood options are 'node' or 'cc'
    """
    if complex is None:
        raise ValueError("Complex cannot be None when calculating pseudoLaplacian.")
    nodes = complex["nodes"]
    pseudoLaplacian = np.zeros((n, n), dtype=int)

    if neighborhood == "node":
        neighborhoods = complex["nodes"]
    elif neighborhood == "cc":
        neighborhoods = {}
        for i in components:
            group_members = []
            for node in components[i].nodes:
                group_members += nodes[node]
            neighborhoods[i] = list(set(group_members))
    else:
        raise ValueError(
            "Only 'cc' and 'nodes' supported as neighorhoods for jmapStar."
        )

    for indices in neighborhoods.values():
        for i in indices:
            for j in indices:
                if i == j:
                    pseudoLaplacian[i, j] += 1
                else:
                    pseudoLaplacian[i, j] -= 1
    return pseudoLaplacian


################################################################################

# Nerve Class

################################################################################


class Nerve:
    """
    A class to handle generating weighted graphs from Keppler Mapper Simplicial Complexes.

    Parameters
    ----------
    weighted : bool, optional
        True if you want to generate a weighted graph.
        If False, please specify a `minIntersection`.
    minIntersection : int, optional
        Minimum intersection considered when computing the nerve.
        An edge will be created only when the intersection between
        two nodes is greater than or equal to `minIntersection`.
        Not specifying this parameter will result in an unweighted graph.
    """

    def __init__(self, minIntersection: int = -1):
        self.minIntersection = minIntersection

    def __repr__(self):
        return f"Nerve(minIntersection={self.minIntersection})"

    def compute(self, nodes):
        """
        Compute the nerve of a simplicial complex.

        Parameters
        ----------
        nodes : dict
            A dictionary with entries `{node id}:{list of ids in node}`.

        Returns
        -------
        edges : list
            A 1-skeleton of the nerve (intersecting nodes).

        Examples
        --------
        >>> nodes = {'node1': [1, 2, 3], 'node2': [2, 3, 4]}
        >>> compute(nodes)
        [['node1', 'node2']]
        """
        if self.minIntersection == -1:
            return self.compute_weighted_edges(nodes)
        else:
            return self.compute_unweighted_edges(nodes)

    def compute_unweighted_edges(self, nodes):
        """
        Helper function to find edges of the overlapping clusters.

        Parameters
        ----------
        nodes : dict
            A dictionary with entries `{node id}:{list of ids in node}`.

        Returns
        -------
        edges : list
            A 1-skeleton of the nerve (intersecting nodes).

        simplicies : list
            Complete list of simplices.

        Examples
        --------
        >>> nodes = {'node1': [1, 2, 3], 'node2': [2, 3, 4]}
        >>> compute_unweighted_edges(nodes)
        [['node1', 'node2']]

        """

        result = defaultdict(list)

        # Create links when clusters from different hypercubes have members with the same sample id.
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            # if there are non-unique members in the union
            if (
                len(set(nodes[candidate[0]]).intersection(nodes[candidate[1]]))
                >= self.minIntersection
            ):
                result[candidate[0]].append(candidate[1])

        edges = [[x, end] for x in result for end in result[x]]
        return edges

    def compute_weighted_edges(self, nodes):
        """
        Helper function to find edges of the overlapping clusters.

        Parameters
        ----------
        nodes : dict
            A dictionary with entries `{node id}:{list of ids in node}`.

        Returns
        -------
        edges : list
            A 1-skeleton of the nerve (intersecting nodes).

        simplicies : list
            Complete list of simplices.

        Examples
        --------
        >>> nodes = {'node1': [1, 2, 3], 'node2': [2, 3, 4]}
        >>> compute_weighted_edges(nodes)
        [('node1', 'node2', 0.333)]

        """

        result = []
        # Create links when clusters from different hypercubes have members with the same sample id.
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            # if there are non-unique members in the union
            overlap = len(set(nodes[candidate[0]]).intersection(nodes[candidate[1]]))
            if overlap > 0:
                result.append((candidate[0], candidate[1], round(1 / overlap, 3)))
        return result


################################################################################

# Kepler Mapper clustering utility functions

################################################################################


def get_clusterer(clusterer: list):
    """
    Converts a list configuration to an initialized clusterer.

    Parameters
    ----------
    clusterer: list
        A length 2 list containing in position 0 the name of the clusterer, and
        in position 1 the parameters to configure it.
        *Example*
        clusterer = ["HDBSCAN", {"minDist":0.1}]

    Returns
    -------
    An initialized clustering object
    """
    if clusterer[0] == "HDBSCAN":
        return HDBSCAN(**clusterer[1])

    elif clusterer[0] == "DBSCAN":
        return DBSCAN(**clusterer[1])

    else:
        raise ValueError("Only HDBSCAN and DBSCAN supported at this time.")


def mapper_unclustered_items(N, nodes):
    """
    Returns the list of items that were not clustered in the
    mapper fitting.

    Returns
    -------
    self._unclustered_item : list
       A list of unclustered item ids
    """
    labels = dict()
    unclustered_items = []
    for idx in range(N):
        place_holder = []
        for node_id in nodes.keys():
            if idx in nodes[node_id]:
                place_holder.append(node_id)

        if len(place_holder) == 0:
            place_holder = -1
            unclustered_items.append(idx)
        labels[idx] = place_holder

    return unclustered_items


def convert_keys_to_alphabet(dictionary):
    """Simple Helper function to make kmapper node labels more readable."""
    base = 26  # Number of letters in the alphabet
    new_dict = {}

    keys = list(dictionary.keys())
    for i, key in enumerate(keys):
        # Calculate the position of each letter in the new key
        position = i
        new_key = ""
        while position >= 0:
            new_key = chr(ord("a") + (position % base)) + new_key
            position = (position // base) - 1

        new_dict[new_key] = dictionary[key]

    return new_dict
