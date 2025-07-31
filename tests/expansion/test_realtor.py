import networkx as nx
import pytest
import numpy as np
from unittest.mock import Mock, patch
from thema.expansion.realtor import Realtor
from thema.expansion.utils import star_link
from thema.multiverse.universe.utils.starGraph import starGraph
from tests.expansion.mock_realtor import MockRealtor


class TestStarLink:
    def test_star_link_with_valid_star(self):
        """Test star_link function with a valid Star object."""
        # Create a mock Star object with starGraph attribute
        mock_star = Mock()
        mock_graph = nx.Graph()
        mock_star.starGraph = starGraph(mock_graph)

        # Define some test features
        node_features = np.array([[1.0, 2.0], [3.0, 4.0]])
        group_features = np.array([[2.0, 3.0]])

        # Call star_link
        result = star_link(mock_star, node_features, group_features)

        # Check the result
        assert result["graph"] == mock_graph
        assert np.array_equal(result["node_features"], node_features)
        assert np.array_equal(result["group_features"], group_features)

    def test_star_link_without_node_features(self):
        """Test star_link function without providing node_features."""
        # Create a mock Star object with starGraph attribute
        mock_star = Mock()
        mock_graph = nx.Graph()
        mock_star.starGraph = starGraph(mock_graph)

        # Call star_link without node_features
        result = star_link(mock_star)

        # Check the result
        assert result["graph"] == mock_graph
        assert result["node_features"] is None
        assert result["group_features"] is None

    def test_star_link_with_missing_starGraph(self):
        """Test star_link function with a Star object that has no starGraph attribute."""
        # Create a mock Star object without starGraph attribute
        mock_star = Mock()
        mock_star.starGraph = None

        # Call star_link and expect an error
        with pytest.raises(
            ValueError,
            match="Star object does not have an initialized starGraph attribute",
        ):
            star_link(mock_star)

    def test_star_link_with_invalid_starGraph(self):
        """Test star_link function with an invalid starGraph attribute."""
        # Create a mock Star object with invalid starGraph attribute
        mock_star = Mock()
        mock_star.starGraph = Mock()
        # The starGraph doesn't have a graph attribute
        del mock_star.starGraph.graph  # Ensure graph attribute doesn't exist

        # Call star_link and expect an error
        with pytest.raises(
            ValueError,
            match="starGraph does not contain a valid graph attribute",
        ):
            star_link(mock_star)


class TestRealtor:
    def test_node_docking(
        self, _test_graphData_0, _test_node_features_0, _test_group_features_0
    ):
        # Define target vector and node features
        target_vector = np.array([1.0, 2.0])
        realtor = Realtor(
            target_vector,
            _test_graphData_0,
            _test_node_features_0,
            _test_group_features_0,
        )

        best_node_index = realtor.node_docking(metric="euclidean")

        assert (target_vector == _test_node_features_0[0]).all()
        assert best_node_index == 0

    def test_node_docking_custom_metric(
        self, _test_graphData_0, _test_node_features_0, _test_group_features_0
    ):
        """Test node_docking with a custom distance metric."""
        target_vector = np.array([1.0, 2.0])
        realtor = Realtor(
            target_vector,
            _test_graphData_0,
            _test_node_features_0,
            _test_group_features_0,
        )

        # Use Manhattan distance (l1)
        best_node_index = realtor.node_docking(metric="cityblock")

        # With Manhattan distance, the first node should still be closest
        assert best_node_index == 0

    def test_random_walk_samples(
        self, _test_graphData_0, _test_node_features_0, _test_group_features_0
    ):
        """Test random_walk generates the correct number of samples."""
        target_vector = np.array([2.0, 3.0])  # Matches node 1
        
        # Use MockRealtor to handle string node IDs
        realtor = MockRealtor(
            target_vector,
            _test_graphData_0,
            _test_node_features_0,
            _test_group_features_0,
        )

        # Run random_walk with small parameters for testing
        n_samples = 5
        samples = realtor.random_walk(n_samples=n_samples, m_steps=10)

        # Check number of samples
        assert len(samples) == n_samples
        # Check all samples are valid nodes
        assert all(node in _test_graphData_0.nodes for node in samples)

    def test_random_walk_distribution(
        self, _test_graphData_0, _test_node_features_0, _test_group_features_0
    ):
        """Test random_walk distribution is affected by target vector."""
        # Using a target vector that matches node 1
        target_vector = np.array([2.0, 3.0])
        
        # Use MockRealtor to handle string node IDs
        realtor = MockRealtor(
            target_vector,
            _test_graphData_0,
            _test_node_features_0,
            _test_group_features_0,
        )

        # Run with more samples for statistical significance
        samples = realtor.random_walk(n_samples=100, m_steps=50)

        # Count occurrences
        unique, counts = np.unique(samples, return_counts=True)
        sample_counts = dict(zip(unique, counts))

        # The samples should include the target node (which is 'B' as it matches [2.0, 3.0])
        assert "B" in sample_counts

    def test_integration_with_star_link(self):
        """Test integration between star_link and Realtor."""
        # Create a mock Star object with a starGraph
        mock_star = Mock()
        graph = nx.Graph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1)
        mock_star.starGraph = starGraph(graph)

        # Create features
        node_features = np.array([[1.0, 2.0], [3.0, 4.0]])
        group_features = np.array([[2.0, 3.0]])

        # Use star_link to extract the graph
        star_data = star_link(mock_star, node_features, group_features)

        # Create a Realtor with the extracted data
        target_vector = np.array([0.9, 1.9])  # Close to node 0
        realtor = Realtor(
            target_vector=target_vector,
            graph=star_data["graph"],
            node_features=star_data["node_features"],
            group_features=star_data["group_features"],
        )

        # Test node_docking works with the extracted graph
        best_node = realtor.node_docking()
        assert best_node == 0  # Should select node 0 as it's closest to target

    def test_realtor_with_mock_star(self, mock_star_object, mock_star_features):
        """Test Realtor using a mock Star object with more complex structure."""
        # Extract star data using star_link
        star_data = star_link(
            mock_star_object,
            mock_star_features["node_features"],
            mock_star_features["group_features"],
        )

        # Create a target vector similar to node_1
        target_vector = mock_star_features["node_features"][1].copy()
        target_vector += np.array([0.05, -0.05])  # Small deviation

        # Create MockRealtor instance to handle string node IDs
        realtor = MockRealtor(
            target_vector=target_vector,
            graph=star_data["graph"],
            node_features=star_data["node_features"],
            group_features=star_data["group_features"],
        )

        # Test node_docking
        best_node = realtor.node_docking()
        assert best_node == "node_1"  # Should be closest to node_1

        # Test random_walk
        n_samples = 50
        samples = realtor.random_walk(n_samples=n_samples, m_steps=100)
        assert len(samples) == n_samples
        # Should contain the closest node frequently
        unique_samples, counts = np.unique(samples, return_counts=True)
        sample_dict = dict(zip(unique_samples, counts))
        assert "node_1" in sample_dict

    def test_realtor_with_complex_star(self, complex_star_graph):
        """Test Realtor with a more complex star graph."""
        # Extract data
        mock_star = complex_star_graph["star"]
        node_features = complex_star_graph["node_features"]
        group_features = complex_star_graph["group_features"]

        # Use star_link to get the graph
        star_data = star_link(mock_star, node_features, group_features)

        # Create a target vector similar to node_10 (middle of the graph)
        target_vector = node_features[10].copy()
        target_vector += np.array([0.02, -0.03, 0.01])  # Small deviation

        # Create MockRealtor to handle string node IDs
        realtor = MockRealtor(
            target_vector=target_vector,
            graph=star_data["graph"],
            node_features=star_data["node_features"],
            group_features=star_data["group_features"],
        )

        # Test node_docking with multiple metrics
        euclidean_best = realtor.node_docking(metric="euclidean")
        assert euclidean_best == "node_10"  # Should be closest to node_10

        cosine_best = realtor.node_docking(metric="cosine")
        # Cosine should find nodes with similar direction, not necessarily same node
        assert cosine_best in star_data["graph"].nodes

        # Test random_walk with different parameters
        samples_short = realtor.random_walk(n_samples=30, m_steps=50)
        assert len(samples_short) == 30

        samples_long = realtor.random_walk(n_samples=30, m_steps=200)
        assert len(samples_long) == 30

        # The distribution should be affected by both steps and distance metrics
        samples_manhattan = realtor.random_walk(
            n_samples=30, m_steps=100, metric="cityblock"
        )
        assert len(samples_manhattan) == 30

    def test_realtor_edge_cases(self, _test_graphData_0):
        """Test Realtor with various edge cases."""
        # Case 1: Target vector is equally distant from all nodes
        node_features = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        group_features = np.array([[3.0, 3.0]])
        target_vector = np.array([3.0, 3.0])  # Equally distant from all nodes

        # Use MockRealtor to handle string node IDs
        realtor = MockRealtor(
            target_vector=target_vector,
            graph=_test_graphData_0,
            node_features=node_features,
            group_features=group_features,
        )

        best_node = realtor.node_docking()
        assert best_node == "B"  # Should pick the middle node (B corresponds to index 1)

        # Case 2: Target vector is in a completely different space
        target_vector = np.array([100.0, 100.0])  # Far from all nodes
        realtor = MockRealtor(
            target_vector=target_vector,
            graph=_test_graphData_0,
            node_features=node_features,
            group_features=group_features,
        )

        best_node = realtor.node_docking()
        # Should still find the closest node (furthest one in this case)
        assert (
            best_node == "C"
        )  # Node C with [5.0, 5.0] is closest to [100.0, 100.0]

        # Case 3: High-dimensional features
        high_dim_features = np.random.rand(3, 10)  # 10-dimensional features
        high_dim_group = np.random.rand(1, 10)
        high_dim_target = np.random.rand(10)

        realtor = MockRealtor(
            target_vector=high_dim_target,
            graph=_test_graphData_0,
            node_features=high_dim_features,
            group_features=high_dim_group,
        )

        # Both methods should work with high-dimensional data
        best_node = realtor.node_docking()
        assert best_node in _test_graphData_0.nodes

        samples = realtor.random_walk(n_samples=5, m_steps=10)
        assert len(samples) == 5
