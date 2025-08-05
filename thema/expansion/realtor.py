import numpy as np
from scipy.spatial.distance import cdist


class Realtor:
    """
    Find a the best location for an incoming target node in the cosmic neighborhood.
    """

    def __init__(self, target_vector, graph, node_features, group_features):
        self.target = target_vector
        self.node_features = node_features
        self.group_features = group_features
        self.graph = graph

    def random_walk(self, n_samples=1000, m_steps=1000, metric="euclidean"):
        """
        A MCMC inspired method for obtaining a collection of node locations from a
        distribution considering both graph structure and feature differences.

        """
        G = self.graph
        node_distances = cdist([self.target], self.node_features, metric=metric)
        group_distances = cdist(
            [self.target], self.group_features, metric=metric
        )
        samples = []
        for _ in range(n_samples):
            # select a random starting point
            current_node = np.random.choice(list(G.nodes))
            for __ in range(m_steps):
                # choose a neighbor (including self loops) based on feature distances
                neighbors = list(G.neighbors(current_node)) + [current_node]
                neighbor_distances = node_distances[
                    0, [int(nb) for nb in neighbors]
                ]
                t_probabilities = neighbor_distances / sum(neighbor_distances)
                current_node = np.random.choice(neighbors, p=t_probabilities)

                # if no good local choices, jump somewhere else with prob 1/4
                if (
                    max(neighbor_distances)
                    > np.average(
                        group_distances
                    )  # could take Nth quantile here.
                    and np.random.rand() < 1 / 4
                ):
                    current_node = np.random.choice(list(G.nodes))

            # After 1000 steps, add to sample
            samples.append(current_node)
        return samples

    def node_docking(self, metric="euclidean"):
        distances = cdist([self.target], self.node_features, metric=metric)
        best_node_index = np.argmin(distances)
        return best_node_index
