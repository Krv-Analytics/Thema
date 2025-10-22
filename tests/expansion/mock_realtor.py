import numpy as np
from thema.expansion.realtor import Realtor


class MockRealtor(Realtor):
    """
    A mock version of Realtor that handles string node IDs properly for testing.

    This class overrides the random_walk method to handle graphs with string node IDs
    without requiring complex mock objects.
    """

    def __init__(self, target_vector, graph, node_features, group_features):
        super().__init__(target_vector, graph, node_features, group_features)
        # Create a mapping from string node IDs to integer indices
        self.node_to_index = {node: i for i, node in enumerate(graph.nodes())}
        self.index_to_node = {i: node for node, i in self.node_to_index.items()}

    def random_walk(self, n_samples=1000, m_steps=1000, metric="euclidean"):
        """
        Override random_walk to handle string node IDs properly.
        """
        from scipy.spatial.distance import cdist

        G = self.graph
        node_distances = cdist([self.target], self.node_features, metric=metric)
        group_distances = cdist([self.target], self.group_features, metric=metric)

        samples = []
        for _ in range(n_samples):
            # Select a random starting point
            current_node = np.random.choice(list(G.nodes))

            for __ in range(m_steps):
                # Choose a neighbor (including self loops) based on feature distances
                neighbors = list(G.neighbors(current_node)) + [current_node]

                # Convert string node IDs to indices for distance lookup
                neighbor_indices = [self.node_to_index[nb] for nb in neighbors]
                neighbor_distances = node_distances[0, neighbor_indices]

                # Create transition probabilities (inverse distance weighting)
                # Add small epsilon to avoid division by zero
                eps = 1e-8
                inv_distances = 1.0 / (neighbor_distances + eps)
                t_probabilities = inv_distances / np.sum(inv_distances)

                current_node = np.random.choice(neighbors, p=t_probabilities)

                # If no good local choices, jump somewhere else with prob 1/4
                if (
                    np.max(neighbor_distances) > np.mean(group_distances)
                    and np.random.rand() < 1 / 4
                ):
                    current_node = np.random.choice(list(G.nodes))

            # After m_steps, add to sample
            samples.append(current_node)

        return samples

    def node_docking(self, metric="euclidean"):
        """
        Override node_docking to return the actual node ID instead of just the index.
        """
        from scipy.spatial.distance import cdist

        distances = cdist([self.target], self.node_features, metric=metric)
        best_node_index = np.argmin(distances)

        # Return the actual node ID from the graph
        return self.index_to_node[best_node_index]
