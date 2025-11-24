import pytest
import numpy as np
from unittest.mock import Mock
from thema.multiverse.universe.utils.starHelpers import mapper_pseudo_laplacian


def test_mapper_pseudo_laplacian_node_neighborhood():
    # Test with neighborhood="node"
    complex_data = {"nodes": {"a": [0, 1], "b": [1, 2]}}
    n = 3
    components = None  # Not used for "node"
    result = mapper_pseudo_laplacian(complex_data, n, components, neighborhood="node")
    expected = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_mapper_pseudo_laplacian_cc_neighborhood():
    # Test with neighborhood="cc"
    complex_data = {"nodes": {"a": [0, 1], "b": [1, 2]}}
    n = 3
    # Mock components
    mock_component = Mock()
    mock_component.nodes = ["a", "b"]
    components = {0: mock_component}
    result = mapper_pseudo_laplacian(complex_data, n, components, neighborhood="cc")
    # When all nodes are in one component, all items [0,1,2] are in one neighborhood
    # So the matrix should have all items connected
    expected = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_mapper_pseudo_laplacian_complex_none():
    # Test ValueError when complex is None
    with pytest.raises(
        ValueError,
        match="Complex cannot be None when calculating pseudoLaplacian.",
    ):
        mapper_pseudo_laplacian(None, 3, None, neighborhood="node")


def test_mapper_pseudo_laplacian_invalid_neighborhood():
    # Test ValueError for invalid neighborhood
    complex_data = {"nodes": {"a": [0, 1]}}
    with pytest.raises(
        ValueError,
        match="Only 'cc' and 'nodes' supported as neighorhoods for jmapStar.",
    ):
        mapper_pseudo_laplacian(complex_data, 2, None, neighborhood="invalid")


def test_mapper_pseudo_laplacian_empty_nodes():
    # Test with empty nodes
    complex_data = {"nodes": {}}
    n = 3
    result = mapper_pseudo_laplacian(complex_data, n, None, neighborhood="node")
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_mapper_pseudo_laplacian_single_item():
    # Test with single item in node
    complex_data = {"nodes": {"a": [0]}}
    n = 1
    result = mapper_pseudo_laplacian(complex_data, n, None, neighborhood="node")
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)
