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


@pytest.fixture(autouse=True)
def patch_multiverse_modules(monkeypatch):
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
