import os
import pytest
from thema.thema import Thema


def test_init_with_valid_yaml(valid_yaml_file):
    t = Thema(valid_yaml_file)
    assert t.YAML_PATH == valid_yaml_file
    assert hasattr(t, "params")
    assert t.params.outDir is not None
    assert t.params.runName is not None


def test_genesis_workflow_success(valid_yaml_file, monkeypatch):
    t = Thema(valid_yaml_file)
    # Patch all sub-methods to just record calls
    called = []
    monkeypatch.setattr(
        t,
        "spaghettify_innerSystem",
        lambda: called.append("spaghettify_innerSystem"),
    )
    monkeypatch.setattr(
        t, "innerSystem_genesis", lambda: called.append("innerSystem_genesis")
    )
    monkeypatch.setattr(
        t,
        "spaghettify_outerSystem",
        lambda: called.append("spaghettify_outerSystem"),
    )
    monkeypatch.setattr(
        t, "outerSystem_genesis", lambda: called.append("outerSystem_genesis")
    )
    monkeypatch.setattr(
        t, "spaghettify_galaxy", lambda: called.append("spaghettify_galaxy")
    )
    monkeypatch.setattr(t, "galaxy_genesis", lambda: called.append("galaxy_genesis"))
    t.genesis()
    assert called == [
        "spaghettify_innerSystem",
        "innerSystem_genesis",
        "spaghettify_outerSystem",
        "outerSystem_genesis",
        "spaghettify_galaxy",
        "galaxy_genesis",
    ]


def test_spaghettify_cleans_directories(valid_yaml_file):
    t = Thema(valid_yaml_file)
    # Setup directory structure
    out_dir = os.path.join(t.params.outDir, t.params.runName)
    clean_dir = os.path.join(out_dir, "clean")
    proj_dir = os.path.join(out_dir, "projections")
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(clean_dir)
    os.makedirs(proj_dir)
    os.makedirs(model_dir)
    # Place dummy files
    for d in [clean_dir, proj_dir, model_dir]:
        with open(os.path.join(d, "dummy.txt"), "w") as f:
            f.write("test")
    t.spaghettify()
    # All directories should be removed
    assert not os.path.exists(clean_dir)
    assert not os.path.exists(proj_dir)
    assert not os.path.exists(model_dir)
    assert not os.path.exists(out_dir)


def test_init_with_invalid_yaml_path(tmp_path):
    # Non-existent file
    invalid_path = str(tmp_path / "doesnotexist.yaml")
    with pytest.raises(ValueError):
        Thema(invalid_path)
    # Not a yaml file
    not_yaml = tmp_path / "params.txt"
    not_yaml.write_text("outDir: foo\nrunName: bar")
    with pytest.raises(ValueError):
        Thema(str(not_yaml))


def test_innerSystem_genesis_nonempty_directory(valid_yaml_file):
    t = Thema(valid_yaml_file)
    clean_outdir = os.path.join(t.params.outDir, t.params.runName + "/clean")
    os.makedirs(clean_outdir)
    # Add a dummy file to make directory non-empty
    with open(os.path.join(clean_outdir, "dummy.txt"), "w") as f:
        f.write("test")
    with pytest.raises(AssertionError):
        t.innerSystem_genesis()


def test_spaghettify_innerSystem_removal_failure(valid_yaml_file, monkeypatch):
    t = Thema(valid_yaml_file)
    clean_outdir = os.path.join(t.params.outDir, t.params.runName + "/clean")
    os.makedirs(clean_outdir)
    file_path = os.path.join(clean_outdir, "dummy.txt")
    with open(file_path, "w") as f:
        f.write("test")

    # Patch os.remove to simulate a failure
    def mock_remove(path):
        if path == file_path:
            raise PermissionError(f"Permission denied: '{path}'")
        else:
            os.remove(path)

    monkeypatch.setattr("os.remove", mock_remove)

    # Patch print to capture output
    output = []
    monkeypatch.setattr("builtins.print", lambda msg: output.append(msg))
    t.spaghettify_innerSystem()

    # Should have printed an error message
    assert any("Error while deleting" in msg for msg in output)


@pytest.mark.compute
def test_galaxy_genesis_creates_cosmic_graph(complete_yaml_file):
    """Test that galaxy_genesis creates and assigns cosmicGraph"""
    import networkx as nx

    t = Thema(complete_yaml_file)

    # Setup prerequisites
    t.spaghettify_innerSystem()
    t.innerSystem_genesis()
    t.spaghettify_outerSystem()
    t.outerSystem_genesis()
    t.spaghettify_galaxy()

    # Run galaxy_genesis
    t.galaxy_genesis()

    # Verify galaxy was created
    assert hasattr(t, "galaxy")
    assert t.galaxy is not None

    # Verify collapse was called (selection should be populated)
    assert hasattr(t.galaxy, "selection")
    assert isinstance(t.galaxy.selection, dict)
    assert len(t.galaxy.selection) > 0

    # Verify selected_model_files was populated
    assert hasattr(t, "selected_model_files")
    assert isinstance(t.selected_model_files, list)
    assert len(t.selected_model_files) > 0

    # Verify cosmicGraph was created and assigned
    assert hasattr(t, "cosmicGraph")
    assert t.cosmicGraph is not None
    assert isinstance(t.cosmicGraph, nx.Graph)

    # Verify cosmicGraph is the same as galaxy.cosmicGraph
    assert t.cosmicGraph is t.galaxy.cosmicGraph


@pytest.mark.compute
def test_galaxy_genesis_selected_files_match_selection(complete_yaml_file):
    """Test that selected_model_files matches galaxy.selection"""
    t = Thema(complete_yaml_file)

    # Setup prerequisites
    t.spaghettify_innerSystem()
    t.innerSystem_genesis()
    t.spaghettify_outerSystem()
    t.outerSystem_genesis()
    t.spaghettify_galaxy()

    # Run galaxy_genesis
    t.galaxy_genesis()

    # Extract star files from selection
    selection_files = [str(info["star"]) for info in t.galaxy.selection.values()]

    # Should match selected_model_files
    assert set(t.selected_model_files) == set(selection_files)


@pytest.mark.compute
def test_full_genesis_workflow_with_cosmic_graph(complete_yaml_file):
    """Test complete genesis workflow including cosmic graph creation"""
    import networkx as nx

    t = Thema(complete_yaml_file)

    # Run full pipeline
    t.genesis()

    # Verify all components were created
    assert hasattr(t, "planet")
    assert hasattr(t, "oort")
    assert hasattr(t, "galaxy")

    # Verify file lists were populated
    assert t.clean_files is not None
    assert len(t.clean_files) > 0
    assert t.projection_files is not None
    assert len(t.projection_files) > 0
    assert t.model_files is not None
    assert len(t.model_files) > 0

    # Verify cosmic graph functionality
    assert t.selected_model_files is not None
    assert len(t.selected_model_files) > 0
    assert hasattr(t, "cosmicGraph")
    assert isinstance(t.cosmicGraph, nx.Graph)
    assert t.cosmicGraph.number_of_nodes() > 0


@pytest.mark.compute
def test_galaxy_genesis_model_files_populated(complete_yaml_file):
    """Test that galaxy_genesis populates model_files correctly"""
    t = Thema(complete_yaml_file)

    # Setup prerequisites
    t.spaghettify_innerSystem()
    t.innerSystem_genesis()
    t.spaghettify_outerSystem()
    t.outerSystem_genesis()
    t.spaghettify_galaxy()

    # Run galaxy_genesis
    t.galaxy_genesis()

    # Verify model_files was populated
    assert hasattr(t, "model_files")
    assert isinstance(t.model_files, list)
    assert len(t.model_files) > 0

    # All model files should exist
    for file_path in t.model_files:
        assert os.path.exists(file_path)


def test_galaxy_genesis_workflow_order(valid_yaml_file, monkeypatch):
    """Test that galaxy_genesis calls methods in correct order"""
    t = Thema(valid_yaml_file)

    # Track method calls
    calls = []

    # Mock Galaxy class
    class MockGalaxy:
        def __init__(self, YAML_PATH):
            calls.append("Galaxy.__init__")
            self.selection = {"0": {"star": "/fake/path.pkl", "cluster_size": 5}}
            self.cosmicGraph = None

        def fit(self):
            calls.append("Galaxy.fit")

        def collapse(self):
            calls.append("Galaxy.collapse")
            return self.selection

        def compute_cosmicGraph(self):
            calls.append("Galaxy.compute_cosmicGraph")
            import networkx as nx

            self.cosmicGraph = nx.Graph()

    # Patch Galaxy
    import thema.thema

    original_galaxy = thema.thema.Galaxy
    monkeypatch.setattr(thema.thema, "Galaxy", MockGalaxy)

    # Create model directory
    model_outdir = os.path.join(t.params.outDir, t.params.runName + "/models/")
    os.makedirs(model_outdir, exist_ok=True)

    try:
        # Run galaxy_genesis
        t.galaxy_genesis()

        # Verify call order
        assert calls == [
            "Galaxy.__init__",
            "Galaxy.fit",
            "Galaxy.collapse",
            "Galaxy.compute_cosmicGraph",
        ]

        # Verify attributes were set
        assert hasattr(t, "galaxy")
        assert hasattr(t, "selected_model_files")
        assert hasattr(t, "cosmicGraph")

    finally:
        # Restore original Galaxy
        monkeypatch.setattr(thema.thema, "Galaxy", original_galaxy)


def test_cosmicGraph_attribute_assignment(valid_yaml_file, monkeypatch):
    """Test that cosmicGraph is correctly assigned from galaxy.cosmicGraph"""
    import networkx as nx

    t = Thema(valid_yaml_file)

    # Mock Galaxy with a specific cosmicGraph
    class MockGalaxy:
        def __init__(self, YAML_PATH):
            self.selection = {"0": {"star": "/fake/path.pkl", "cluster_size": 5}}
            self.cosmicGraph = nx.Graph()
            self.cosmicGraph.add_node(0)
            self.cosmicGraph.add_node(1)
            self.cosmicGraph.add_edge(0, 1, weight=0.5)

        def fit(self):
            pass

        def collapse(self):
            return self.selection

        def compute_cosmicGraph(self):
            pass

    # Patch Galaxy
    import thema.thema

    original_galaxy = thema.thema.Galaxy
    monkeypatch.setattr(thema.thema, "Galaxy", MockGalaxy)

    # Create model directory
    model_outdir = os.path.join(t.params.outDir, t.params.runName + "/models/")
    os.makedirs(model_outdir, exist_ok=True)

    try:
        # Run galaxy_genesis
        t.galaxy_genesis()

        # Verify cosmicGraph was assigned correctly
        assert t.cosmicGraph is t.galaxy.cosmicGraph
        assert t.cosmicGraph.number_of_nodes() == 2
        assert t.cosmicGraph.number_of_edges() == 1
        assert t.cosmicGraph[0][1]["weight"] == 0.5

    finally:
        # Restore original Galaxy
        monkeypatch.setattr(thema.thema, "Galaxy", original_galaxy)
