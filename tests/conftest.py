# File:tests/conftest.py
# Last Updated: 07/29/25
# Updated By: JW

import os
import sys
import pytest
import yaml
import types

from thema.thema import Thema


# Dummy classes to patch imports if needed
class DummyPlanet:
    def __init__(self, YAML_PATH=None):
        self.YAML_PATH = YAML_PATH

    def fit(self):
        pass


class DummyOort:
    def __init__(self, YAML_PATH=None):
        self.YAML_PATH = YAML_PATH

    def fit(self):
        pass


class DummyGalaxy:
    def __init__(self, YAML_PATH=None):
        self.YAML_PATH = YAML_PATH

    def fit(self):
        pass


def pytest_configure(config):
    """
    Called before test collection starts.
    """
    # Add the root directory of the project to the Python path
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_path)

    # Monkeypatch sklearn for compatibility with dependencies using removed force_all_finite
    import sklearn.utils.validation

    _original_check_array = sklearn.utils.validation.check_array

    def _check_array_monkeypatch(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _original_check_array(*args, **kwargs)

    sklearn.utils.validation.check_array = _check_array_monkeypatch


@pytest.fixture(autouse=True)
def patch_multiverse_modules(request, monkeypatch):
    """
    Automatically patch multiverse modules with dummies for unit tests.
    Skip patching for compute tests (marked with @pytest.mark.compute).
    """
    # Check if this is a compute test
    if "compute" in request.keywords:
        # Skip patching for compute tests - they need real classes
        yield
        return

    # Apply patches for unit tests
    dummy_multiverse = types.SimpleNamespace(
        Planet=DummyPlanet, Oort=DummyOort, Galaxy=DummyGalaxy
    )
    monkeypatch.setattr("thema.multiverse.Planet", DummyPlanet)
    monkeypatch.setattr("thema.multiverse.Oort", DummyOort)
    monkeypatch.setattr("thema.multiverse.Galaxy", DummyGalaxy)
    monkeypatch.setitem(sys.modules, "thema.multiverse", dummy_multiverse)
    yield


@pytest.fixture
def valid_yaml_file(tmp_path):
    yaml_content = {"outDir": str(tmp_path / "out"), "runName": "test_run"}
    yaml_path = tmp_path / "params.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    return str(yaml_path)


@pytest.fixture
def thema_instance(valid_yaml_file):
    return Thema(valid_yaml_file)


@pytest.fixture
def complete_yaml_file(tmp_path):
    """
    Create a complete YAML configuration file with all required fields
    for running the full Thema pipeline including Galaxy.
    """
    import tempfile
    import pandas as pd
    import numpy as np
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

    # Run Planet to create clean files
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

    # Run Oort to create projection files
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

    # Create YAML configuration
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
            "cosmic_graph": {
                "enabled": True,
                "neighborhood": "cc",
                "threshold": 0.0,
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

    yaml_path = tmp_path / "complete_params.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    return str(yaml_path)
