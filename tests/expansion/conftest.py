import pytest
import networkx as nx
import numpy as np
from unittest.mock import Mock

from thema.multiverse.universe.utils.starGraph import starGraph


@pytest.fixture
def _test_graphData_0():
    # Create a simple graph
    graph = nx.Graph()

    # Add nodes with attributes
    graph.add_node("A", type="numeric")
    graph.add_node("B", type="numeric")
    graph.add_node("C", type="categorical")

    # Add edges
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")

    return graph


@pytest.fixture
def _test_mappergraphData_0():
    # Create mapper style graph
    graph = nx.Graph()

    return graph


@pytest.fixture
def _test_node_features_0():
    # Define node features as a numpy array
    node_features = np.array(
        [[1.0, 2.0], [2.0, 3.0], [5.0, 6.0]]  # Node A  # Node B  # Node C
    )

    return node_features


@pytest.fixture
def _test_group_features_0():
    # Define group features as a numpy array
    group_features = np.array([[1.5, 2.5], [3.0, 4.0]])  # Group 1  # Group 2

    return group_features


@pytest.fixture
def mock_star_object():
    """Create a mock Star object with a starGraph."""
    mock_star = Mock()

    # Create a graph for the starGraph
    graph = nx.Graph()

    # Add nodes with string IDs (similar to jmapStar)
    for i in range(10):
        node_id = f"node_{i}"
        graph.add_node(node_id)

    # Add some edges with weights
    edges = [
        ("node_0", "node_1", 0.5),
        ("node_1", "node_2", 0.7),
        ("node_2", "node_3", 0.3),
        ("node_3", "node_4", 1.0),
        ("node_4", "node_5", 0.8),
        ("node_5", "node_6", 0.6),
        ("node_6", "node_7", 0.9),
        ("node_7", "node_8", 0.4),
        ("node_8", "node_9", 0.5),
        ("node_9", "node_0", 1.2),
    ]

    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)

    # Assign the starGraph to the mock
    mock_star.starGraph = starGraph(graph)

    return mock_star


@pytest.fixture
def mock_star_features():
    """Create mock features for the star graph nodes."""
    # Generate random node features (2D for simplicity)
    np.random.seed(42)  # For reproducibility
    node_features = np.random.rand(10, 2)

    # Generate group features (assuming 2 groups)
    group_features = np.array(
        [
            [0.3, 0.7],  # Group 1
            [0.8, 0.2],  # Group 2
        ]
    )

    return {"node_features": node_features, "group_features": group_features}


@pytest.fixture
def complex_star_graph():
    """Create a more complex star graph for testing."""
    # Create a more complex graph
    graph = nx.Graph()

    # Add 20 nodes
    for i in range(20):
        node_id = f"node_{i}"
        graph.add_node(node_id)

    # Add edges to create a connected structure with weights
    for i in range(19):
        graph.add_edge(
            f"node_{i}", f"node_{i+1}", weight=0.5 + np.random.random() * 0.5
        )

    # Add some cross-edges to make the graph more complex
    cross_edges = [
        ("node_0", "node_5", 0.8),
        ("node_5", "node_10", 1.2),
        ("node_10", "node_15", 0.7),
        ("node_15", "node_0", 0.9),
        ("node_2", "node_7", 0.6),
        ("node_7", "node_12", 1.0),
        ("node_12", "node_17", 0.5),
        ("node_17", "node_2", 0.8),
    ]

    for u, v, w in cross_edges:
        graph.add_edge(u, v, weight=w)

    # Create a mock Star with this graph
    mock_star = Mock()
    mock_star.starGraph = starGraph(graph)

    # Create features for this graph
    np.random.seed(42)  # For reproducibility
    node_features = np.random.rand(20, 3)  # 3D features
    group_features = np.array(
        [
            [0.2, 0.4, 0.6],  # Group 1
            [0.7, 0.5, 0.3],  # Group 2
            [0.1, 0.8, 0.2],  # Group 3
        ]
    )

    return {
        "star": mock_star,
        "node_features": node_features,
        "group_features": group_features,
    }
