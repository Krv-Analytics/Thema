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
        match="Only 'cc' and 'nodes' supported as neighborhoods for our current mapper-based stars.",
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


def test_mapper_pseudo_laplacian_type_validation():
    # Test that function validates input types correctly
    complex_data = {"nodes": {"a": [0, 1], "b": [1, 2]}}
    n = 3

    # Test with valid types
    result = mapper_pseudo_laplacian(complex_data, n, None, neighborhood="node")
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64 or result.dtype == np.int32
    assert result.shape == (n, n)


def test_mapper_pseudo_laplacian_with_jmap_style_complex():
    # Test with a complex structure similar to what jmapStar produces
    # jmapStar uses alphabetic node IDs after calling convert_keys_to_alphabet
    complex_data = {
        "nodes": {
            "a": [0, 1, 2],
            "b": [2, 3, 4],
            "c": [4, 5],
            "d": [6, 7],
        }
    }
    n = 8
    components = None

    result = mapper_pseudo_laplacian(complex_data, n, components, neighborhood="node")

    # Verify result properties
    assert isinstance(result, np.ndarray)
    assert result.shape == (n, n)

    # Verify diagonal elements (count of neighborhoods each item appears in)
    assert result[0, 0] == 1  # item 0 in node "a" only
    assert result[2, 2] == 2  # item 2 in nodes "a" and "b"
    assert result[4, 4] == 2  # item 4 in nodes "b" and "c"

    # Verify off-diagonal elements (negative shared neighborhoods)
    assert result[0, 1] == -1  # items 0 and 1 share node "a"
    assert result[2, 3] == -1  # items 2 and 3 share node "b"
    assert result[6, 7] == -1  # items 6 and 7 share node "d"


def test_mapper_pseudo_laplacian_connected_components_type():
    # Test with cc neighborhood using mock components similar to what
    # different star implementations might produce
    complex_data = {
        "nodes": {
            "a": [0, 1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
    }
    n = 7

    # Mock two separate connected components
    mock_component_0 = Mock()
    mock_component_0.nodes = ["a", "b"]  # Component with nodes a and b

    mock_component_1 = Mock()
    mock_component_1.nodes = ["c"]  # Component with node c

    components = {0: mock_component_0, 1: mock_component_1}

    result = mapper_pseudo_laplacian(complex_data, n, components, neighborhood="cc")

    # Verify result properties
    assert isinstance(result, np.ndarray)
    assert result.shape == (n, n)
    assert result.dtype == np.int64 or result.dtype == np.int32

    # Component 0 contains items [0,1,2,3,4], so they should all be connected
    assert result[0, 0] == 1  # item 0 in 1 component
    assert result[0, 1] == -1  # items 0 and 1 share component 0
    assert result[2, 4] == -1  # items 2 and 4 share component 0

    # Component 1 contains items [5,6], so they should be connected to each other
    assert result[5, 5] == 1  # item 5 in 1 component
    assert result[5, 6] == -1  # items 5 and 6 share component 1

    # Items from different components should not be connected
    assert result[0, 5] == 0  # items 0 and 5 in different components
