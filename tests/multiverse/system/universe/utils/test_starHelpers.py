import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock
from thema.multiverse.universe.utils.starHelpers import (
    mapper_pseudo_laplacian,
    normalize_cosmicGraph,
)


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
        match="Only 'cc' and 'node' supported as neighborhoods for our current mapper-based stars.",
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


def test_mapper_pseudo_laplacian_normaliztion():
    pass


################################################################################

# normalize_cosmicGraph Tests

################################################################################


def test_normalize_cosmicGraph_basic():
    """Test normalize_cosmicGraph with a simple 3x3 pseudo-Laplacian."""
    # Create a simple pseudo-Laplacian
    galactic_pseudoLaplacian = np.array(
        [[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float
    )
    threshold = 0.0

    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold
    )

    # Check return types
    assert isinstance(cosmicGraph, nx.Graph)
    assert isinstance(cosmic_wadj, np.ndarray)
    assert isinstance(cosmic_adj, np.ndarray)

    # Check shapes
    assert cosmic_wadj.shape == (3, 3)
    assert cosmic_adj.shape == (3, 3)

    # Check that diagonal is zero (no self-loops)
    assert np.all(np.diag(cosmic_wadj) == 0)
    assert np.all(np.diag(cosmic_adj) == 0)


def test_normalize_cosmicGraph_weights_normalized():
    """Test that weights are properly normalized between 0 and 1."""
    galactic_pseudoLaplacian = np.array(
        [[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float
    )
    threshold = 0.0

    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold
    )

    # All non-zero weights should be between 0 and 1
    nonzero_weights = cosmic_wadj[cosmic_wadj != 0]
    assert np.all(nonzero_weights >= 0) and np.all(nonzero_weights <= 1)


def test_normalize_cosmicGraph_threshold_effect():
    """Test that threshold properly filters edges."""
    galactic_pseudoLaplacian = np.array(
        [[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float
    )

    # With threshold=0, all edges where weight > 0 should be included
    cosmicGraph_low, cosmic_wadj_low, cosmic_adj_low = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold=0.0
    )

    # With high threshold, fewer edges
    cosmicGraph_high, cosmic_wadj_high, cosmic_adj_high = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold=0.9
    )

    # High threshold should result in fewer edges
    assert np.sum(cosmic_adj_low) >= np.sum(cosmic_adj_high)
    assert cosmicGraph_low.number_of_edges() >= cosmicGraph_high.number_of_edges()


def test_normalize_cosmicGraph_zero_denominator():
    """Test handling of zero denominators (should result in zero weight)."""
    # Create a pseudo-Laplacian where some pairs have zero denominator
    galactic_pseudoLaplacian = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    threshold = 0.0

    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold
    )

    # All weights should be zero (no edges)
    assert np.all(cosmic_wadj == 0)
    assert np.all(cosmic_adj == 0)
    assert cosmicGraph.number_of_edges() == 0


def test_normalize_cosmicGraph_negative_pseudoLaplacian():
    """Test with a pseudo-Laplacian containing negative off-diagonals."""
    # Standard pseudo-Laplacian: positive diagonal, negative off-diagonal
    galactic_pseudoLaplacian = np.array(
        [[3, -1, -1], [-1, 3, -1], [-1, -1, 3]], dtype=float
    )
    threshold = 0.0

    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold
    )

    # Weights should be positive (since -G[ij] is positive when G[ij] is negative)
    assert np.all(cosmic_wadj >= 0)


def test_normalize_cosmicGraph_edge_weights():
    """Test that edge weights are correctly attached to the NetworkX graph."""
    galactic_pseudoLaplacian = np.array(
        [[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float
    )
    threshold = 0.0

    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold
    )

    # For each edge in the graph, verify weight attribute matches cosmic_wadj
    for u, v in cosmicGraph.edges():
        graph_weight = cosmicGraph[u][v]["weight"]
        matrix_weight = cosmic_wadj[u, v]
        assert np.isclose(graph_weight, matrix_weight)


def test_normalize_cosmicGraph_symmetry():
    """Test that the normalized graph is symmetric (undirected)."""
    galactic_pseudoLaplacian = np.array(
        [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float
    )
    threshold = 0.0

    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold
    )

    # cosmic_adj should be symmetric
    assert np.allclose(cosmic_adj, cosmic_adj.T)

    # Graph should be undirected (inherently so from nx.from_numpy_array)
    assert not cosmicGraph.is_directed()


def test_normalize_cosmicGraph_large_matrix():
    """Test with a larger pseudo-Laplacian matrix."""
    n = 10
    # Create a random pseudo-Laplacian-like matrix
    np.random.seed(42)
    pseudo_L = np.zeros((n, n))
    for i in range(n):
        pseudo_L[i, i] = np.random.rand() * 5 + 1  # Positive diagonal
        for j in range(i + 1, n):
            if np.random.rand() > 0.7:  # Sparse
                val = -np.random.rand()
                pseudo_L[i, j] = val
                pseudo_L[j, i] = val

    threshold = 0.1
    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(pseudo_L, threshold)

    assert cosmic_wadj.shape == (n, n)
    assert cosmic_adj.shape == (n, n)
    assert cosmicGraph.number_of_nodes() == n


def test_normalize_cosmicGraph_single_node():
    """Test with a 1x1 pseudo-Laplacian (edge case)."""
    galactic_pseudoLaplacian = np.array([[1.0]])
    threshold = 0.0

    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold
    )

    assert cosmic_wadj.shape == (1, 1)
    assert cosmic_adj.shape == (1, 1)
    assert cosmicGraph.number_of_edges() == 0


def test_normalize_cosmicGraph_two_nodes():
    """Test with a 2x2 pseudo-Laplacian."""
    galactic_pseudoLaplacian = np.array([[2.0, -1.0], [-1.0, 2.0]])
    threshold = 0.0

    cosmicGraph, cosmic_wadj, cosmic_adj = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold
    )

    # Expected: weight between nodes 0 and 1 is -(-1) / (2 + 2 + (-1)) = 1 / 3 ≈ 0.333
    expected_weight = 1.0 / 3.0
    assert np.isclose(cosmic_wadj[0, 1], expected_weight)
    assert np.isclose(cosmic_wadj[1, 0], expected_weight)

    # Both should pass threshold of 0
    assert cosmic_adj[0, 1] == 1
    assert cosmic_adj[1, 0] == 1


def test_normalize_cosmicGraph_threshold_exact_boundary():
    """Test threshold behavior at exact boundary values."""
    galactic_pseudoLaplacian = np.array([[2.0, -1.0], [-1.0, 2.0]])

    # Weight is 1/3 ≈ 0.333
    weight = 1.0 / 3.0

    # Threshold just below weight
    _, _, cosmic_adj_below = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold=weight - 0.001
    )
    assert cosmic_adj_below[0, 1] == 1

    # Threshold at weight (should fail strict > comparison)
    _, _, cosmic_adj_at = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold=weight
    )
    assert cosmic_adj_at[0, 1] == 0

    # Threshold above weight
    _, _, cosmic_adj_above = normalize_cosmicGraph(
        galactic_pseudoLaplacian, threshold=weight + 0.001
    )
    assert cosmic_adj_above[0, 1] == 0
