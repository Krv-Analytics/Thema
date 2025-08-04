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
