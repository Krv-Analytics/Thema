import pytest
import networkx as nx
import numpy as np


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
