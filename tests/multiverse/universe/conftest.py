# File:tests/multiverse/system/outer/conftest.py
# Last Updated: 04-23-24
# Updated By: JW

import os
import sys
import yaml
import pytest
import shutil
import tempfile
import pandas as pd
import numpy as np
import networkx as nx
import pickle

from tests import test_utils as ut

from thema.multiverse import Planet, Oort
from thema.multiverse.system.inner.moon import Moon
from thema.multiverse.system.outer.projectiles.pcaProj import pcaProj
from thema.multiverse.system.outer.projectiles.tsneProj import tsneProj
from thema.multiverse.system.outer.projectiles.umapProj import umapProj
from thema.multiverse.universe.starGraph import starGraph


@pytest.fixture
def tmp_file():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
        yield temp_file
    temp_file.close()


@pytest.fixture
def tmp_outDir():
    with tempfile.TemporaryDirectory() as tmp_outDir:
        yield tmp_outDir
    if os.path.isdir(tmp_outDir):
        shutil.rmtree(tmp_outDir)


@pytest.fixture
def tmp_umapMoonAndData():
    """
    Creates a temporary pre-configured tuple consisting of
    raw data, a moon object, and a umap projectile.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".pkl",
        mode="wb",
        delete=True,
    ) as tmp_dataFile:
        ut.generate_dataframe().to_pickle(tmp_dataFile.name)
        with tempfile.NamedTemporaryFile(
            suffix=".pkl",
            mode="wb",
        ) as tmp_moon:
            moon = Moon(
                data=tmp_dataFile.name,
                imputeColumns=[
                    "Num1",
                    "Num2",
                    "Num3",
                    "Num4",
                    "Num5",
                    "Num6",
                    "Num7",
                    "Num8",
                    "Num9",
                    "Num10",
                ],
                imputeMethods=[
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                ],
                encoding="one_hot",
                scaler="standard",
                seed=42,
            )
            moon.fit()
            moon.save(tmp_moon.name)
            with tempfile.NamedTemporaryFile(
                suffix=".pkl", mode="wb"
            ) as tmp_projectile:
                umapproj = umapProj(
                    data_path=tmp_dataFile.name,
                    clean_path=tmp_moon.name,
                    nn=4,
                    minDist=0.2,
                    dimensions=2,
                    seed=42,
                )
                umapproj.fit()
                umapproj.save(file_path=tmp_projectile.name)
                yield tmp_dataFile, tmp_moon, tmp_projectile
            tmp_projectile.close()
        tmp_moon.close()
    tmp_dataFile.close()


@pytest.fixture
def tmp_tsneMoonAndData():
    """
    Creates a temporary pre-configured tuple consisting of
    raw data, a moon object, and a tsne projectile.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".pkl",
        mode="wb",
        delete=True,
    ) as tmp_dataFile:
        ut.generate_dataframe().to_pickle(tmp_dataFile.name)
        with tempfile.NamedTemporaryFile(
            suffix=".pkl",
            mode="wb",
        ) as tmp_moon:
            moon = Moon(
                data=tmp_dataFile.name,
                imputeColumns=[
                    "Num1",
                    "Num2",
                    "Num3",
                    "Num4",
                    "Num5",
                    "Num6",
                    "Num7",
                    "Num8",
                    "Num9",
                    "Num10",
                ],
                imputeMethods=[
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                ],
                encoding="one_hot",
                scaler="standard",
                seed=42,
            )
            moon.fit()
            moon.save(tmp_moon.name)
            with tempfile.NamedTemporaryFile(
                suffix=".pkl", mode="wb"
            ) as tmp_projectile:
                tsneproj = tsneProj(
                    data_path=tmp_dataFile.name,
                    clean_path=tmp_moon.name,
                    perplexity=2,
                    dimensions=2,
                    seed=42,
                )
                tsneproj.fit()
                tsneproj.save(file_path=tmp_projectile.name)
                yield tmp_dataFile, tmp_moon, tmp_projectile
            tmp_projectile.close()
        tmp_moon.close()
    tmp_dataFile.close()


@pytest.fixture
def tmp_pcaMoonAndData():
    """
    Creates a temporary pre-configured tuple consisting of
    raw data, a moon object, and a pca projectile.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".pkl",
        mode="wb",
        delete=True,
    ) as tmp_dataFile:
        ut.generate_dataframe().to_pickle(tmp_dataFile.name)
        with tempfile.NamedTemporaryFile(
            suffix=".pkl",
            mode="wb",
        ) as tmp_moon:
            moon = Moon(
                data=tmp_dataFile.name,
                imputeColumns=[
                    "Num1",
                    "Num2",
                    "Num3",
                    "Num4",
                    "Num5",
                    "Num6",
                    "Num7",
                    "Num8",
                    "Num9",
                    "Num10",
                ],
                imputeMethods=[
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                    "sampleNormal",
                ],
                encoding="one_hot",
                scaler="standard",
                seed=42,
            )
            moon.fit()
            moon.save(tmp_moon.name)
            with tempfile.NamedTemporaryFile(
                suffix=".pkl", mode="wb"
            ) as tmp_projectile:
                pcaproj = pcaProj(
                    data_path=tmp_dataFile.name,
                    clean_path=tmp_moon.name,
                    dimensions=2,
                    seed=42,
                )
                pcaproj.fit()
                pcaproj.save(file_path=tmp_projectile.name)
                yield tmp_dataFile, tmp_moon, tmp_projectile
            tmp_projectile.close()
        tmp_moon.close()
    tmp_dataFile.close()


@pytest.fixture
def temp_galaxyYaml_1():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_dataFile:
        ut.generate_dataframe().to_pickle(tmp_dataFile.name)
        data = tmp_dataFile.name
        runName = "test1"
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

            modeling = {
                "stars": ["jmap"],
                "metric": "stellar_kernel_distance",
                "nReps": 3,
                "selector": "random",
                "jmap": {
                    "nCubes": [4],
                    "percOverlap": [0.5],
                    "minIntersection": [-1],
                    "clusterer": [
                        ["HDBSCAN", {"min_cluster_size": 2}],
                        ["HDBSCAN", {"min_cluster_size": 3}],
                    ],
                },
            }

            params = {
                "umap": {
                    "nn": [2],
                    "minDist": [0.1],
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
                "Galaxy": modeling,
            }

            planet = Planet(
                data=tmp_dataFile.name,
                outDir=outDir + "/test1/clean/",
                imputeColumns="all",
                imputeMethods="sampleNormal",
                numSamples=3,
                seeds=[42, 41, 40],
            )
            planet.fit()

            oort = Oort(
                data=tmp_dataFile.name,
                cleanDir=outDir + "/test1/clean/",
                outDir=outDir + "/test1/projections/",
                params=params,
            )
            oort.fit()
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


class test_star:
    def __init__(self,graph) -> None:
        self.starGraph = starGraph(graph=graph)



@pytest.fixture
def temp_starGraphs():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate and save 30 Erdos-Renyi random graphs
        for i in range(30):
            G = nx.erdos_renyi_graph(n=np.random.randint(low=10,high=100),p=np.random.random_sample())  
            # Set all edge weights to 1
            nx.set_edge_attributes(G, 1, 'weight')
            graph_object = test_star(graph=G)
            file_path = os.path.join(temp_dir, f"graph_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(graph_object, f)
        
        yield temp_dir
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)







 
    
    