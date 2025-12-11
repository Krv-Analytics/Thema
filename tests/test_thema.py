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

    # When cosmic graph is enabled, collapse is NOT called
    # So selection and selected_model_files should NOT be populated
    assert hasattr(t, "selected_model_files")
    # selected_model_files should not be set when cosmic graph is enabled
    assert not hasattr(t, "selected_model_files") or t.selected_model_files is None

    # Verify cosmicGraph was created and assigned
    assert hasattr(t, "cosmicGraph")
    assert t.cosmicGraph is not None
    assert isinstance(t.cosmicGraph, nx.Graph)

    # Verify cosmicGraph is the same as galaxy.cosmicGraph
    assert t.cosmicGraph is t.galaxy.cosmicGraph

    # Verify the cosmic graph has nodes (it should have been computed)
    assert t.cosmicGraph.number_of_nodes() > 0


@pytest.mark.compute
def test_galaxy_genesis_selected_files_match_selection(tmp_path):
    """Test that selected_model_files matches galaxy.selection when cosmic graph is disabled"""
    import pandas as pd
    import numpy as np
    import yaml
    from thema.multiverse import Planet, Oort

    # Generate test data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "A": np.random.choice(["cat", "dog", "bird"], 100),
            "B": np.random.randn(100),
            "Num1": np.random.randn(100),
            "Num2": np.random.randn(100),
            "Num3": np.random.randn(100),
            "Num4": np.random.randn(100),
            "Num5": np.random.randn(100),
        }
    )

    # Save data
    data_file = tmp_path / "test_data.pkl"
    data.to_pickle(str(data_file))

    # Setup directories
    out_dir = tmp_path / "out"
    run_name = "test_run"
    clean_dir = out_dir / run_name / "clean"
    proj_dir = out_dir / run_name / "projections"

    clean_dir.mkdir(parents=True, exist_ok=True)
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Run Planet
    planet = Planet(
        data=str(data_file),
        outDir=str(clean_dir) + "/",
        imputeColumns=["B"],
        imputeMethods=["sampleNormal"],
        numSamples=2,
        seeds=[42, 41],
        encoding="one_hot",
        scaler="standard",
    )
    planet.fit()

    # Run Oort
    oort = Oort(
        data=str(data_file),
        cleanDir=str(clean_dir) + "/",
        outDir=str(proj_dir) + "/",
        params={
            "tsne": {"perplexity": [2], "dimensions": [2], "seed": [42]},
            "pca": {"dimensions": [2], "seed": [42]},
        },
    )
    oort.fit()

    # Create YAML configuration WITHOUT cosmic graph (to test collapse behavior)
    yaml_content = {
        "runName": run_name,
        "data": str(data_file),
        "outDir": str(out_dir),
        "Planet": {
            "scaler": "standard",
            "encoding": "one_hot",
            "numSamples": 2,
            "seeds": [42, 41],
            "dropColumns": None,
            "imputeColumns": ["B"],
            "imputeMethods": ["sampleNormal"],
        },
        "Oort": {
            "projectiles": ["tsne", "pca"],
            "tsne": {"perplexity": [2], "dimensions": [2], "seed": [42]},
            "pca": {"dimensions": [2], "seed": [42]},
        },
        "Galaxy": {
            "stars": ["jmap"],
            "metric": "stellar_curvature_distance",
            "nReps": 2,
            "selector": "max_nodes",
            "filter": None,
            # Cosmic graph disabled to test collapse behavior
            "cosmic_graph": {
                "enabled": False,
            },
            "jmap": {
                "nCubes": [4],
                "percOverlap": [0.5],
                "minIntersection": [-1],
                "clusterer": [
                    ["HDBSCAN", {"min_cluster_size": 2}],
                ],
            },
        },
    }

    yaml_path = tmp_path / "params.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    t = Thema(str(yaml_path))

    # Setup prerequisites
    t.spaghettify_innerSystem()
    t.innerSystem_genesis()
    t.spaghettify_outerSystem()
    t.outerSystem_genesis()
    t.spaghettify_galaxy()

    # Run galaxy_genesis
    t.galaxy_genesis()

    # When cosmic graph is disabled, collapse IS called
    # Extract star files from selection
    selection_files = [str(info["star"]) for info in t.galaxy.selection.values()]

    # Should match selected_model_files
    assert t.selected_model_files is not None
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

    # When cosmic graph is enabled, selected_model_files should NOT be set
    # (collapse is not called to avoid expensive computation)
    assert not hasattr(t, "selected_model_files") or t.selected_model_files is None

    # Verify cosmic graph functionality
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

        def compute_cosmicGraph(self, neighborhood="cc", threshold=0.0):
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

        # Verify call order - cosmic_graph is disabled by default
        assert calls == [
            "Galaxy.__init__",
            "Galaxy.fit",
            "Galaxy.collapse",
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
            self.cosmicGraph = None

        def fit(self):
            pass

        def collapse(self):
            return self.selection

        def compute_cosmicGraph(self, neighborhood="cc", threshold=0.0):
            self.cosmicGraph = nx.Graph()
            self.cosmicGraph.add_node(0)
            self.cosmicGraph.add_node(1)
            self.cosmicGraph.add_edge(0, 1, weight=0.5)

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

        # Verify cosmicGraph is None (disabled by default)
        assert t.cosmicGraph is None

    finally:
        # Restore original Galaxy
        monkeypatch.setattr(thema.thema, "Galaxy", original_galaxy)


def test_collapse_called_when_cosmic_graph_disabled(valid_yaml_file, monkeypatch):
    """Test that collapse() is called when cosmic graphs are disabled"""
    t = Thema(valid_yaml_file)

    # Track method calls
    calls = []

    # Mock Galaxy to track calls
    class MockGalaxy:
        def __init__(self, YAML_PATH):
            self.selection = {"0": {"star": "/fake/path.pkl", "cluster_size": 5}}
            self.cosmicGraph = None

        def fit(self):
            calls.append("fit")

        def collapse(self):
            calls.append("collapse")
            return self.selection

        def compute_cosmicGraph(self, neighborhood="cc", threshold=0.0):
            calls.append("compute_cosmicGraph")
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
        # Run galaxy_genesis (cosmic graphs disabled by default)
        t.galaxy_genesis()

        # Verify collapse was called
        assert "collapse" in calls
        # Verify compute_cosmicGraph was NOT called
        assert "compute_cosmicGraph" not in calls
        # Verify cosmicGraph is None
        assert t.cosmicGraph is None
        # Verify selected_model_files was set
        assert t.selected_model_files is not None
        assert isinstance(t.selected_model_files, list)

    finally:
        monkeypatch.setattr(thema.thema, "Galaxy", original_galaxy)


def test_collapse_not_called_when_cosmic_graph_enabled(valid_yaml_file, monkeypatch):
    """Test that collapse() is NOT called when cosmic graphs are enabled"""
    t = Thema(valid_yaml_file)

    # Enable cosmic graphs in the config
    from omegaconf import OmegaConf

    t.params.Galaxy = OmegaConf.create(
        {
            "cosmic_graph": {
                "enabled": True,
                "neighborhood": "cc",
                "threshold": 0.0,
            }
        }
    )

    # Track method calls
    calls = []

    # Mock Galaxy to track calls
    class MockGalaxy:
        def __init__(self, YAML_PATH):
            self.selection = {"0": {"star": "/fake/path.pkl", "cluster_size": 5}}
            self.cosmicGraph = None

        def fit(self):
            calls.append("fit")

        def collapse(self):
            calls.append("collapse")
            return self.selection

        def compute_cosmicGraph(self, neighborhood="cc", threshold=0.0):
            calls.append("compute_cosmicGraph")
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

        # Verify compute_cosmicGraph was called
        assert "compute_cosmicGraph" in calls
        # Verify collapse was NOT called
        assert "collapse" not in calls
        # Verify cosmicGraph is set
        assert t.cosmicGraph is not None
        # Verify selected_model_files was NOT set (only set when collapse is called)
        assert t.selected_model_files is None

    finally:
        monkeypatch.setattr(thema.thema, "Galaxy", original_galaxy)


def test_cosmic_graph_config_parameters_passed_correctly(valid_yaml_file, monkeypatch):
    """Test that cosmic graph configuration parameters are passed correctly"""
    t = Thema(valid_yaml_file)

    # Enable cosmic graphs with specific parameters
    from omegaconf import OmegaConf

    t.params.Galaxy = OmegaConf.create(
        {
            "cosmic_graph": {
                "enabled": True,
                "neighborhood": "knn",
                "threshold": 0.5,
            }
        }
    )

    # Track parameters passed to compute_cosmicGraph
    captured_params = {}

    # Mock Galaxy to capture parameters
    class MockGalaxy:
        def __init__(self, YAML_PATH):
            self.selection = {"0": {"star": "/fake/path.pkl", "cluster_size": 5}}
            self.cosmicGraph = None

        def fit(self):
            pass

        def collapse(self):
            return self.selection

        def compute_cosmicGraph(self, neighborhood="cc", threshold=0.0):
            captured_params["neighborhood"] = neighborhood
            captured_params["threshold"] = threshold
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

        # Verify parameters were passed correctly
        assert captured_params["neighborhood"] == "knn"
        assert captured_params["threshold"] == 0.5

    finally:
        monkeypatch.setattr(thema.thema, "Galaxy", original_galaxy)


def test_default_cosmic_graph_behavior_without_config(valid_yaml_file, monkeypatch):
    """Test default behavior when Galaxy config is not present"""
    t = Thema(valid_yaml_file)

    # Ensure Galaxy config doesn't exist
    if hasattr(t.params, "Galaxy"):
        delattr(t.params, "Galaxy")

    # Track method calls
    calls = []

    # Mock Galaxy to track calls
    class MockGalaxy:
        def __init__(self, YAML_PATH):
            self.selection = {"0": {"star": "/fake/path.pkl", "cluster_size": 5}}
            self.cosmicGraph = None

        def fit(self):
            calls.append("fit")

        def collapse(self):
            calls.append("collapse")
            return self.selection

        def compute_cosmicGraph(self, neighborhood="cc", threshold=0.0):
            calls.append("compute_cosmicGraph")
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

        # Verify collapse was called (default behavior)
        assert "collapse" in calls
        # Verify compute_cosmicGraph was NOT called
        assert "compute_cosmicGraph" not in calls
        # Verify cosmicGraph is None
        assert t.cosmicGraph is None

    finally:
        monkeypatch.setattr(thema.thema, "Galaxy", original_galaxy)


def test_cosmic_graph_enabled_false_explicitly(valid_yaml_file, monkeypatch):
    """Test behavior when cosmic graphs are explicitly disabled"""
    t = Thema(valid_yaml_file)

    # Explicitly disable cosmic graphs
    from omegaconf import OmegaConf

    t.params.Galaxy = OmegaConf.create({"cosmic_graph": {"enabled": False}})

    # Track method calls
    calls = []

    # Mock Galaxy to track calls
    class MockGalaxy:
        def __init__(self, YAML_PATH):
            self.selection = {"0": {"star": "/fake/path.pkl", "cluster_size": 5}}
            self.cosmicGraph = None

        def fit(self):
            calls.append("fit")

        def collapse(self):
            calls.append("collapse")
            return self.selection

        def compute_cosmicGraph(self, neighborhood="cc", threshold=0.0):
            calls.append("compute_cosmicGraph")
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

        # Verify collapse was called
        assert "collapse" in calls
        # Verify compute_cosmicGraph was NOT called
        assert "compute_cosmicGraph" not in calls
        # Verify cosmicGraph is None
        assert t.cosmicGraph is None
        # Verify selected_model_files was set
        assert t.selected_model_files is not None

    finally:
        monkeypatch.setattr(thema.thema, "Galaxy", original_galaxy)
