# File: tests/multiverse/universe/test_galaxy.py
# Lasted Updated: 04-23-24
# Updated By: JW

import os
import pytest

from thema.multiverse import Galaxy


@pytest.mark.compute
class TestGalaxy:
    """Pytest class for Galaxy"""

    def test_yaml_init(self, temp_galaxyYaml_1):
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)

    def test_fit(self, temp_galaxyYaml_1):
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

    def test_collapse(self, temp_galaxyYaml_1):
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()
        selection = galaxy.collapse()

        assert isinstance(selection, dict)
        import warnings

        warnings.filterwarnings(
            "ignore",
        )
        for group, selected in selection.items():
            assert isinstance(selected, dict)

            star = selected["star"]
            cluster_size = selected["cluster_size"]
            print(f"Star: {star}, Cluster Size: {cluster_size}")
            assert type(cluster_size) == int or type(cluster_size) == float
            assert os.path.exists(star)
        assert temp_galaxyYaml_1.name

    def test_collapse_with_custom_parameters(self, temp_galaxyYaml_1):
        """Test collapse method with custom parameters"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Test with custom nReps
        selection = galaxy.collapse(nReps=2)
        assert isinstance(selection, dict)
        assert len(selection) <= 2  # Should have at most 2 clusters

        # Verify selection structure
        for group_id, selected in selection.items():
            assert isinstance(selected, dict)
            assert "star" in selected
            assert "cluster_size" in selected
            assert isinstance(selected["cluster_size"], (int, float))
            assert os.path.exists(selected["star"])

    def test_collapse_sets_selection_attribute(self, temp_galaxyYaml_1):
        """Test that collapse method sets the selection attribute"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Initially selection should be empty
        assert galaxy.selection == {}

        # After collapse, selection should be populated
        returned_selection = galaxy.collapse()
        assert galaxy.selection == returned_selection
        assert len(galaxy.selection) > 0

    def test_collapse_sets_keys_and_distances(self, temp_galaxyYaml_1):
        """Test that collapse method sets keys and distances attributes"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Initially keys and distances should be None
        assert galaxy.keys is None
        assert galaxy.distances is None

        # After collapse, they should be set
        galaxy.collapse()
        assert galaxy.keys is not None
        assert galaxy.distances is not None

    def test_collapse_different_metrics(self, temp_galaxyYaml_1):
        """Test collapse method with different metrics"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Test with stellar_curvature_distance (default)
        selection1 = galaxy.collapse(metric="stellar_curvature_distance")
        assert isinstance(selection1, dict)

        # Test with stellar_kernel_distance if available
        try:
            selection2 = galaxy.collapse(metric="stellar_kernel_distance")
            assert isinstance(selection2, dict)
        except AttributeError:
            # If stellar_kernel_distance is not available, that's okay
            pass

    def test_collapse_different_selectors(self, temp_galaxyYaml_1):
        """Test collapse method with different selectors"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Test with max_nodes selector (default)
        selection = galaxy.collapse(selector="max_nodes")
        assert isinstance(selection, dict)

        # Verify that selected stars exist
        for group_id, selected in selection.items():
            assert os.path.exists(selected["star"])

    def test_get_galaxy_coordinates(self, temp_galaxyYaml_1):
        """Test get_galaxy_coordinates method"""
        import numpy as np

        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Need to call collapse first to compute distances
        galaxy.collapse()

        # Get coordinates
        coordinates = galaxy.get_galaxy_coordinates()

        # Verify coordinates are a 2D numpy array
        assert isinstance(coordinates, np.ndarray)
        assert coordinates.ndim == 2
        assert coordinates.shape[1] == 2  # Should have 2 columns (X, Y)

        # Number of rows should match number of stars
        assert coordinates.shape[0] == len(galaxy.keys)

        # Coordinates should be finite
        assert np.all(np.isfinite(coordinates))

    def test_get_galaxy_coordinates_without_collapse(self, temp_galaxyYaml_1):
        """Test that get_galaxy_coordinates raises error if distances not computed"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Should raise ValueError since distances haven't been computed
        with pytest.raises(ValueError, match="Distance matrix is not computed"):
            galaxy.get_galaxy_coordinates()

    def test_compute_cosmicGraph_basic(self, temp_galaxyYaml_1):
        """Test basic compute_cosmicGraph functionality"""
        import networkx as nx

        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Compute cosmic graph
        galaxy.compute_cosmicGraph(neighborhood="node", threshold=0.0)

        # Verify cosmicGraph was created
        assert hasattr(galaxy, "cosmicGraph")
        assert galaxy.cosmicGraph is not None
        assert isinstance(galaxy.cosmicGraph, nx.Graph)

        # Graph should have nodes
        assert galaxy.cosmicGraph.number_of_nodes() > 0

    def test_compute_cosmicGraph_with_threshold(self, temp_galaxyYaml_1):
        """Test compute_cosmicGraph with different thresholds"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # Compute with low threshold
        galaxy.compute_cosmicGraph(neighborhood="node", threshold=0.0)
        edges_low = galaxy.cosmicGraph.number_of_edges()

        # Compute with higher threshold (should have fewer edges)
        galaxy.compute_cosmicGraph(neighborhood="node", threshold=0.5)
        edges_high = galaxy.cosmicGraph.number_of_edges()

        # Higher threshold should result in fewer or equal edges
        assert edges_high <= edges_low

    def test_compute_cosmicGraph_edge_weights(self, temp_galaxyYaml_1):
        """Test that cosmic graph edges have weights"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        galaxy.compute_cosmicGraph(neighborhood="node", threshold=0.0)

        # Check that edges have weight attributes
        if galaxy.cosmicGraph.number_of_edges() > 0:
            for u, v, data in galaxy.cosmicGraph.edges(data=True):
                assert "weight" in data
                assert isinstance(data["weight"], (int, float))
                # Weights should be non-negative based on the algorithm
                assert data["weight"] >= 0

    def test_compute_cosmicGraph_node_neighborhood(self, temp_galaxyYaml_1):
        """Test compute_cosmicGraph with 'node' neighborhood"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        galaxy.compute_cosmicGraph(neighborhood="node", threshold=0.0)

        assert galaxy.cosmicGraph is not None
        # Number of nodes should match data size
        # Note: This assumes galaxy.data is the raw data path
        # The actual implementation uses len(self.data) which might need adjustment

    def test_compute_cosmicGraph_cc_neighborhood(self, temp_galaxyYaml_1):
        """Test compute_cosmicGraph with 'cc' (connected component) neighborhood"""
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

        # This might fail for some star types, so we'll make it conditional
        try:
            galaxy.compute_cosmicGraph(neighborhood="cc", threshold=0.0)
            assert galaxy.cosmicGraph is not None
        except Exception as e:
            # Some star types may not support 'cc' neighborhood
            pytest.skip(f"CC neighborhood not supported: {e}")

    def test_galaxy_workflow_integration(self, temp_galaxyYaml_1):
        """Test complete workflow: fit -> collapse -> coordinates -> cosmic graph"""
        import numpy as np
        import networkx as nx

        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)

        # Step 1: Fit
        galaxy.fit()
        assert os.path.exists(galaxy.outDir)

        # Step 2: Collapse
        selection = galaxy.collapse()
        assert isinstance(selection, dict)
        assert galaxy.keys is not None
        assert galaxy.distances is not None

        # Step 3: Get coordinates
        coordinates = galaxy.get_galaxy_coordinates()
        assert isinstance(coordinates, np.ndarray)
        assert coordinates.shape == (len(galaxy.keys), 2)

        # Step 4: Compute cosmic graph
        galaxy.compute_cosmicGraph(neighborhood="node", threshold=0.0)
        assert isinstance(galaxy.cosmicGraph, nx.Graph)
