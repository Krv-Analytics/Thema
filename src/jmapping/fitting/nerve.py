import itertools
from collections import defaultdict


class Nerve:
    """
    A class to handle generating weighted graphs from Keppler Mapper Simplicial Complexes.
       Parameters
    -----------
    weighted: bool, default is True
        True if you want to generate a weighted graph.
        If False, please specify a `min_intersection`

    min_intersection: int, default is None
        Minimum intersection considered when computing the nerve.
        An edge will be created only when the intersection between
        two nodes is greater than or equal to `min_intersection`.
        Not
    """

    def __init__(self, min_intersection: int = -1):
        self.min_intersection = min_intersection

    def __repr__(self):
        return f"Nerve(min_intersection={self.min_intersection})"

    def compute(self, nodes):
        # TODO: Refactor this! Need to deal with >= for min intersection, and how to include the weights
        if self.min_intersection == -1:
            return self.compute_weighted_edges(nodes)
        else:
            return self.compute_unweighted_edges(nodes)

    def compute_unweighted_edges(self, nodes):
        """Helper function to find edges of the overlapping clusters.

        Parameters
        ----------
        nodes:
            A dictionary with entires `{node id}:{list of ids in node}`

        Returns
        -------
        edges:
            A 1-skeleton of the nerve (intersecting  nodes)

        simplicies:
            Complete list of simplices

        """

        result = defaultdict(list)

        # Create links when clusters from different hypercubes have members with the same sample id.
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            # if there are non-unique members in the union
            if (
                len(set(nodes[candidate[0]]).intersection(nodes[candidate[1]]))
                >= self.min_intersection
            ):
                result[candidate[0]].append(candidate[1])

        edges = [[x, end] for x in result for end in result[x]]
        return edges

    def compute_weighted_edges(self, nodes):
        """Helper function to find edges of the overlapping clusters.

        Parameters
        ----------
        nodes:
            A dictionary with entires `{node id}:{list of ids in node}`

        Returns
        -------
        edges:
            A 1-skeleton of the nerve (intersecting  nodes)

        simplicies:
            Complete list of simplices

        """

        result = []
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            overlap = len(set(nodes[candidate[0]]).intersection(nodes[candidate[1]]))
            if overlap > 0:
                result.append((candidate[0], candidate[1], round(1 / overlap, 3)))
        return result
