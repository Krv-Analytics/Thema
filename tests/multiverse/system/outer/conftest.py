# File:tests/multiverse/system/outer/conftest.py
# Last Updated: 04-07-24
# Updated By: SW

import os
import sys
import yaml
import pytest
import shutil
import tempfile
import pandas as pd


from tests import test_utils as ut
from thema.multiverse.system.inner.moon import Moon
from thema.multiverse.system.inner.planet import Planet


@pytest.fixture
def tmp_outDir():
    with tempfile.TemporaryDirectory() as tmp_outDir:
        yield tmp_outDir
    if os.path.isdir(tmp_outDir):
        shutil.rmtree(tmp_outDir)


@pytest.fixture
def tmp_moonAndData():
    """
    Creates a temporary pre-configured Moon object file.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".pkl",
        mode="wb",
        delete=True,
    ) as tmp_dataFile:
        ut._test_data_0.to_pickle(tmp_dataFile.name)
        with tempfile.NamedTemporaryFile(
            suffix=".pkl",
            mode="wb",
            delete=True,
        ) as tmp_moon:
            moon = Moon(
                data=tmp_dataFile.name,
                imputeColumns=["B"],
                imputeMethods=["sampleNormal"],
                encoding="one_hot",
                scaler="standard",
                seed=42,
            )
            moon.fit()
            moon.save(tmp_moon.name)
            yield tmp_dataFile, tmp_moon
        tmp_moon.close()
    tmp_dataFile.close()


@pytest.fixture
def tmp_planetAndData():
    """
    Creates an orbiting moon directory from a temporary preconfigured Planet
    """
    with tempfile.NamedTemporaryFile(
        suffix=".pkl",
        mode="wb",
        delete=True,
    ) as tmp_dataFile:
        ut._test_data_0.to_pickle(tmp_dataFile.name)
        with tempfile.TemporaryDirectory() as tmpdir:
            planet = Planet(
                data=tmp_dataFile.name,
                encoding="one_hot",
                scaler="standard",
                imputeColumns=["B"],
                imputeMethods=["sampleNormal"],
                outDir=tmpdir,
                numSamples=3,
                seeds=[42, 41, 40],
            )
            planet.fit()
            yield tmp_dataFile, tmpdir
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
    tmp_dataFile.close()


@pytest.fixture
def test_params1():
    return {
        "umap": {
            "minDist": [0.1, 0.2],
            "nn": [2],
            "dimensions": [2, 3],
            "seed": [2, 4],
        },
        "tsne": {
            "perplexity": [
                2,
            ],
            "dimensions": [2, 3],
            "seed": [52, 34, 88],
        },
        "pca": {
            "dimensions": [2, 3],
            "seed": [52, 34, 88],
        },
    }


@pytest.fixture
def test_erroneous_params0():
    return {
        "nonexistant_projectile": {
            "minDist": [0.1, 0.2],
            "nn": [2, 3],
            "dimensions": [2, 3],
            "seed": [2, 4],
        }
    }


@pytest.fixture
def test_erroneous_params1():
    return {
        "umap": {
            "minDist": [0.1, 0.2],
            "dimensions": [2, 3],
            "seed": [2, 4],
        }
    }


@pytest.fixture
def temp_projYaml_0():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_dataFile:
        ut._test_data_0.to_pickle(tmp_dataFile.name)
        data = tmp_dataFile.name
        runName = "test"
        with tempfile.TemporaryDirectory() as outDir:
            outDir = outDir
            cleaning = {
                "scaler": "standard",
                "encoding": "one_hot",
                "numSamples": 3,
                "seeds": [42, 41, 40],
                "dropColumns": None,
                "imputeColumns": ["B"],
                "imputeMethods": ["sampleNormal"],
            }

            projecting = {
                "projectiles": ["umap", "tsne", "pca"],
                "umap": {
                    "nn": [2],
                    "minDist": [0.1, 0.2],
                    "dimensions": [2],
                    "seed": [42],
                },
                "tsne": {"perplexity": [2], "dimensions": [2], "seed": [42]},
                "pca": {"dimensions": [2], "seed": [42]},
            }

            parameters = {
                "runName": runName,
                "data": data,
                "outDir": outDir,
                "Planet": cleaning,
                "Oort": projecting,
            }

            planet = Planet(
                data=tmp_dataFile.name,
                outDir=outDir + "/test/clean/",
                imputeColumns=["B"],
                imputeMethods=["sampleNormal"],
                numSamples=3,
                seeds=[42, 41, 40],
            )
            planet.fit()
            with tempfile.NamedTemporaryFile(
                suffix=".yaml",
                mode="w",
            ) as yaml_temp_file:
                yaml.dump(parameters, yaml_temp_file, default_flow_style=False)
                yield yaml_temp_file
            yaml_temp_file.close()
        if os.path.isdir(outDir):
            shutil.rmtree(outDir)
    tmp_dataFile.close()
