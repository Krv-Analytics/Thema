import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock

from thema.expansion.utils import star_link
from tests.expansion.mock_realtor import MockRealtor
from thema.multiverse.universe.utils.starGraph import starGraph


class TestStarLinkIntegration:
    """Integration tests for the star_link function with real-world usage scenarios."""

    def test_end_to_end_workflow(self):
        """Test the complete workflow from Star to Realtor."""
        # 1. Create a mock Star object with a realistic graph structure
        mock_star = Mock()

        # Create a graph with community structure
        G = nx.barbell_graph(10, 2)  # Two communities connected by a path

        # Convert integer node IDs to strings (as in real jmapStar objects)
        mapping = {i: f"node_{i}" for i in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

        # Add weights to edges (representing node overlaps in mapper graphs)
        for u, v in G.edges():
            G[u][v]["weight"] = round(1.0 / np.random.randint(1, 5), 3)

        mock_star.starGraph = starGraph(G)

        # 2. Create features for nodes and communities
        # Node features - we'll create embedding-like features
        # First community: nodes 0-9 have features closer to [0,0]
        # Second community: nodes 10-19 have features closer to [1,1]
        # Path nodes: 20-21 are in between
        node_features = np.zeros((22, 2))

        # First community features
        for i in range(10):
            node_features[i] = [
                np.random.normal(0, 0.2),
                np.random.normal(0, 0.2),
            ]

        # Second community features
        for i in range(10, 20):
            node_features[i - 10 + 10] = [
                np.random.normal(1, 0.2),
                np.random.normal(1, 0.2),
            ]

        # Path nodes features (in between)
        node_features[20] = [0.4, 0.4]
        node_features[21] = [0.6, 0.6]

        # Group features (centers of the two communities)
        group_features = np.array(
            [
                [0.0, 0.0],  # First community center
                [1.0, 1.0],  # Second community center
            ]
        )

        # 3. Extract the graph using star_link
        star_data = star_link(mock_star, node_features, group_features)

        # 4. Test with target vectors from different scenarios

        # Case 1: Target is close to first community
        target1 = np.array([0.1, 0.1])
        realtor1 = MockRealtor(
            target_vector=target1,
            graph=star_data["graph"],
            node_features=star_data["node_features"],
            group_features=star_data["group_features"],
        )
        best_node1 = realtor1.node_docking()
        # Best node should be in the first community (0-9)
        assert int(best_node1.split("_")[1]) < 10

        # Case 2: Target is close to second community
        target2 = np.array([0.9, 0.9])
        realtor2 = MockRealtor(
            target_vector=target2,
            graph=star_data["graph"],
            node_features=star_data["node_features"],
            group_features=star_data["group_features"],
        )
        best_node2 = realtor2.node_docking()
        # Best node should be in the second community (10-19)
        assert int(best_node2.split("_")[1]) >= 10

        # Case 3: Target is in between communities
        target3 = np.array([0.5, 0.5])
        realtor3 = MockRealtor(
            target_vector=target3,
            graph=star_data["graph"],
            node_features=star_data["node_features"],
            group_features=star_data["group_features"],
        )
        best_node3 = realtor3.node_docking()
        # Best node should be one of the path nodes (20-21) or nearby
        node_num = int(best_node3.split("_")[1])
        assert abs(node_num - 20) <= 2

        # Case 4: Random walk should show community preference
        samples = realtor3.random_walk(n_samples=100, m_steps=200)
        unique_samples, counts = np.unique(samples, return_counts=True)

        # Convert node names to indices for easier analysis
        node_indices = [int(node.split("_")[1]) for node in unique_samples]

        community1_count = sum(
            counts[i] for i, idx in enumerate(node_indices) if idx < 10
        )
        community2_count = sum(
            counts[i] for i, idx in enumerate(node_indices) if 10 <= idx < 20
        )

        # With target in middle, both communities should be sampled
        assert community1_count > 0
        assert community2_count > 0

    def test_with_different_metrics(self):
        """Test star_link and Realtor with different distance metrics."""
        # Create a simple star with a graph
        mock_star = Mock()
        G = nx.Graph()

        # Add nodes in 3D space
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B", weight=0.5)
        G.add_edge("B", "C", weight=0.8)

        mock_star.starGraph = starGraph(G)

        # Create node features in 3D space
        node_features = np.array(
            [
                [1, 0, 0],  # Node A - along x-axis
                [0, 1, 0],  # Node B - along y-axis
                [0, 0, 1],  # Node C - along z-axis
            ]
        )

        group_features = np.array(
            [
                [0.5, 0.5, 0.5],  # Center
            ]
        )

        # Extract graph
        star_data = star_link(mock_star, node_features, group_features)

        # Test with different target vectors and metrics

        # Target at [1, 1, 1] - equidistant in Euclidean space
        target = np.array([1, 1, 1])

        # Test with different metrics
        metrics = ["euclidean", "cosine", "cityblock", "chebyshev"]
        for metric in metrics:
            realtor = MockRealtor(
                target_vector=target,
                graph=star_data["graph"],
                node_features=star_data["node_features"],
                group_features=star_data["group_features"],
            )

            best_node = realtor.node_docking(metric=metric)
            # We're just checking it returns a valid node
            assert best_node in ["A", "B", "C"]

            # Random walk should also work
            samples = realtor.random_walk(n_samples=10, m_steps=20, metric=metric)
            assert len(samples) == 10
            assert all(node in ["A", "B", "C"] for node in samples)

    def test_star_link_with_large_graph(self):
        """Test star_link with a larger graph structure."""
        # Create a larger mock Star object
        mock_star = Mock()

        # Create a larger graph (100 nodes)
        G = nx.watts_strogatz_graph(100, 4, 0.1)

        # Convert integer node IDs to strings
        mapping = {i: f"node_{i}" for i in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

        # Add weights to edges
        for u, v in G.edges():
            G[u][v]["weight"] = round(np.random.random(), 3)

        mock_star.starGraph = starGraph(G)

        # Create features for 100 nodes (5D)
        np.random.seed(42)
        node_features = np.random.rand(100, 5)

        # Group features (assuming 3 groups)
        group_features = np.array(
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.8, 0.8, 0.8, 0.8, 0.8],
            ]
        )

        # Extract graph
        star_data = star_link(mock_star, node_features, group_features)

        # Verify the extracted graph
        assert len(star_data["graph"].nodes) == 100
        assert len(star_data["graph"].edges) > 0
        assert star_data["node_features"].shape == (100, 5)

        # Use a target vector
        target = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Center point

        # Create realtor
        realtor = MockRealtor(
            target_vector=target,
            graph=star_data["graph"],
            node_features=star_data["node_features"],
            group_features=star_data["group_features"],
        )

        # Find best node
        best_node = realtor.node_docking()
        assert best_node in star_data["graph"].nodes

        # Run a shorter random walk for testing
        samples = realtor.random_walk(n_samples=20, m_steps=30)
        assert len(samples) == 20
        assert all(node in star_data["graph"].nodes for node in samples)
