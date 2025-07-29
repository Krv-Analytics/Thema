# File: tests/multiverse/universe/test_geodesics.py
# Lasted Updated: 07/29/25
# Updated By: JW


import os
import numpy as np
import networkx as nx

from thema.multiverse.universe import geodesics
from thema.multiverse.universe.utils.starGraph import starGraph


class TestGeodesics:
    """Pytest class for Geodesics"""

    def test_load_starGraphs(self, temp_starGraphs):
        files = temp_starGraphs
        graphs = geodesics._load_starGraphs(files)
        assert len(graphs) == len(os.listdir(files))
        for id_, sG in graphs.items():
            assert os.path.exists(id_)
            assert isinstance(sG.graph, nx.Graph)

    def test_stellar_curvature_distance(self, temp_starGraphs):
        files = temp_starGraphs
        keys, M = geodesics.stellar_curvature_distance(
            files,
            filterfunction=None,
        )
        assert len(keys) == M.shape[0]
        assert isinstance(M, np.ndarray)
        for key in keys:
            assert os.path.exists(key)
        assert np.allclose(M, M.T)

    def test_stellar_curvature_distance_with_float_weights(
        self, temp_float_weighted_graphs
    ):
        """Test that the distance calculation works with float edge weights"""
        files = temp_float_weighted_graphs
        keys, M = geodesics.stellar_curvature_distance(
            files,
            filterfunction=None,
        )
        assert len(keys) == M.shape[0]
        assert isinstance(M, np.ndarray)
        assert np.allclose(M, M.T)

        # Verify the values are proper distances
        assert np.all(M >= 0)  # Non-negative
        assert np.all(np.diag(M) == 0)  # Zero self-distance

    def test_stellar_curvature_distance_with_mixed_weights(
        self, temp_mixed_weight_graphs
    ):
        """Test that the distance calculation works with mixed integer and float weights"""
        files = temp_mixed_weight_graphs
        keys, M = geodesics.stellar_curvature_distance(
            files,
            filterfunction=None,
        )
        assert len(keys) == M.shape[0]
        assert isinstance(M, np.ndarray)
        assert np.allclose(M, M.T)

        # Verify the values are proper distances
        assert np.all(M >= 0)  # Non-negative
        assert np.all(np.diag(M) == 0)  # Zero self-distance

    def test_stellar_curvature_distance_with_scale_free_graphs(
        self, temp_scale_free_graphs
    ):
        """Test that the distance calculation works with scale-free graphs"""
        files = temp_scale_free_graphs
        keys, M = geodesics.stellar_curvature_distance(
            files,
            filterfunction=None,
        )
        assert len(keys) == M.shape[0]
        assert isinstance(M, np.ndarray)
        assert np.allclose(M, M.T)

        # Scale-free networks should have different properties reflected in the distance matrix
        # Check that the matrix is not uniform (distances vary significantly)
        off_diag = M[~np.eye(M.shape[0], dtype=bool)]
        assert np.std(off_diag) > 0.01  # There should be variation in distances

    def test_starGraph_with_float_weights(self):
        """Test starGraph class directly with float weights"""
        # Create a small graph with float weights
        G = nx.cycle_graph(5)
        weights = [0.1, 0.25, 1.5, 2.75, 3.2]

        for i, (u, v) in enumerate(G.edges()):
            G[u][v]["weight"] = weights[i]

        sg = starGraph(G)

        # Test MST
        mst = sg.get_MST()
        assert nx.is_tree(mst)
        assert len(mst.edges) == 4  # n-1 edges for a tree of n nodes

        # Test shortest path
        path, length = sg.get_shortest_path(0, 2)
        # For a cycle graph with 5 nodes, there should be two paths from 0 to 2
        # The shortest path depends on the weights
        assert path is not None
        assert length > 0

        # Verify the path length calculation
        manual_length = sum(
            G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
        )
        assert abs(length - manual_length) < 1e-10

    def test_starGraph_with_very_small_weights(self):
        """Test starGraph with very small float weights"""
        G = nx.complete_graph(5)
        # Use very small float values
        for u, v in G.edges():
            G[u][v]["weight"] = 1e-5 + 1e-6 * np.random.random()

        sg = starGraph(G)

        # Test MST with tiny weights
        mst = sg.get_MST()
        assert nx.is_tree(mst)
        assert len(mst.edges) == 4  # n-1 edges

        # Check shortest path with tiny weights
        path, length = sg.get_shortest_path(0, 4)
        assert path is not None
        assert length > 0 and length < 1e-4  # Should be a small value

    def test_starGraph_with_large_weights(self):
        """Test starGraph with very large float weights"""
        G = nx.complete_graph(5)
        # Use large float values
        for u, v in G.edges():
            G[u][v]["weight"] = 1e6 + 1e5 * np.random.random()

        sg = starGraph(G)

        # Test MST with large weights
        mst = sg.get_MST()
        assert nx.is_tree(mst)

        # Test shortest path with large weights
        path, length = sg.get_shortest_path(0, 4)
        assert path is not None
        assert length > 1e6  # Should be a large value

    def test_starGraph_with_decimal_precision(self):
        """Test starGraph with weights requiring decimal precision"""
        G = nx.path_graph(5)
        # Set weights that differ by small decimal values
        weights = [1.001, 1.002, 1.003, 1.004]

        for i, (u, v) in enumerate(G.edges()):
            G[u][v]["weight"] = weights[i]

        sg = starGraph(G)

        # Test shortest path calculation is precise
        path, length = sg.get_shortest_path(0, 4)
        expected_length = sum(weights)
        assert abs(length - expected_length) < 1e-10

    def test_stellar_distance_across_varied_graph_types(
        self, temp_weighted_graphs_collection
    ):
        """Test that distance calculation works across varied graph types"""
        files = temp_weighted_graphs_collection
        keys, M = geodesics.stellar_curvature_distance(
            files,
            filterfunction=None,
        )
        assert len(keys) == 5  # 5 different graphs
        assert M.shape == (5, 5)

        # Check basic distance properties
        assert np.all(M >= 0)  # Non-negative
        assert np.all(np.diag(M) == 0)  # Zero self-distance
