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
from thema.multiverse.universe.utils.starGraph import starGraph


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
                "metric": "stellar_curvature_distance",
                "nReps": 3,
                "selector": "max_nodes",
                "filter": None,
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
    def __init__(self, graph) -> None:
        self.starGraph = starGraph(graph=graph)


@pytest.fixture
def temp_starGraphs():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate and save graphs that mimic real jmapStar characteristics
        for i in range(10):  # Reduced from 30 for faster tests
            # Create base graph structure
            n_nodes = np.random.randint(low=20, high=80)
            G = nx.erdos_renyi_graph(n_nodes, 0.05 + 0.15 * np.random.random())
            
            # Convert to string node IDs like jmapStar's convert_keys_to_alphabet function
            def generate_node_id(index):
                """Generate string node IDs like jmapStar: a, b, c, ..., z, aa, ab, etc."""
                base = 26
                result = ""
                position = index
                while position >= 0:
                    result = chr(ord("a") + (position % base)) + result
                    position = (position // base) - 1
                return result
            
            mapping = {old_id: generate_node_id(old_id) for old_id in G.nodes()}
            G = nx.relabel_nodes(G, mapping)
            
            # Set jmap-style float weights mimicking Nerve.compute_weighted_edges
            # Real jmap weights are round(1/overlap, 3) where overlap is node intersection count
            for u, v in G.edges():
                # Simulate realistic overlap counts (1-10 shared members)
                overlap = np.random.randint(1, 11)
                weight = round(1.0 / overlap, 3)
                G[u][v]["weight"] = weight
            
            graph_object = test_star(graph=G)
            file_path = os.path.join(temp_dir, f"jmap_realistic_graph_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(graph_object, f)

        yield temp_dir
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_real_jmap_starGraphs():
    """Create starGraphs using actual jmapStar objects with realistic parameters"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a small dataset for jmapStar to work with
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_dataFile:
            test_data = ut.generate_dataframe()
            test_data.to_pickle(tmp_dataFile.name)
            
            # Create minimal moon and comet objects
            with tempfile.TemporaryDirectory() as pipeline_dir:
                # Create Moon
                moon = Moon(
                    data=tmp_dataFile.name,
                    imputeColumns=['B'],
                    imputeMethods=['sampleNormal'],
                    encoding='one_hot',
                    scaler='standard',
                    seed=42
                )
                moon.fit()
                moon_file = os.path.join(pipeline_dir, 'moon.pkl')
                moon.save(moon_file)
                
                # Create UMAP projection
                umap_proj = umapProj(
                    data_path=tmp_dataFile.name,
                    clean_path=moon_file,
                    nn=4,
                    minDist=0.1,
                    dimensions=2,
                    seed=42
                )
                umap_proj.fit()
                proj_file = os.path.join(pipeline_dir, 'umap.pkl')
                umap_proj.save(proj_file)
                
                # Create multiple jmapStar objects with different parameters
                jmap_configs = [
                    {'nCubes': 4, 'percOverlap': 0.5, 'minIntersection': -1, 
                     'clusterer': ['HDBSCAN', {'min_cluster_size': 2}]},
                    {'nCubes': 6, 'percOverlap': 0.3, 'minIntersection': -1,
                     'clusterer': ['HDBSCAN', {'min_cluster_size': 3}]},
                    {'nCubes': 5, 'percOverlap': 0.4, 'minIntersection': -1,
                     'clusterer': ['HDBSCAN', {'min_cluster_size': 2}]},
                ]
                
                from thema.multiverse.universe.stars.jmapStar import jmapStar
                
                for i, config in enumerate(jmap_configs):
                    try:
                        star = jmapStar(
                            data_path=tmp_dataFile.name,
                            clean_path=moon_file,
                            projection_path=proj_file,
                            **config
                        )
                        star.fit()
                        
                        # Only save if we got a valid graph
                        if star.starGraph is not None and star.starGraph.graph.number_of_nodes() > 0:
                            file_path = os.path.join(temp_dir, f"real_jmap_star_{i}.pkl")
                            with open(file_path, 'wb') as f:
                                pickle.dump(star, f)
                    except Exception as e:
                        # Skip failed star creation (some parameter combinations might fail)
                        print(f"Skipping jmapStar config {i}: {e}")
                        continue
        
        # Clean up temp data file
        os.unlink(tmp_dataFile.name)
        
        yield temp_dir
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_float_weighted_graphs():
    """Create temporary starGraphs with float edge weights"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate and save 10 graphs with float weights
        for i in range(10):
            G = nx.erdos_renyi_graph(
                n=np.random.randint(low=10, high=50),
                p=0.1 + np.random.random() * 0.4,
            )
            # Set float weights between 0.01 and 5.0
            for u, v in G.edges():
                G[u][v]["weight"] = 0.01 + 4.99 * np.random.random()

            graph_object = test_star(graph=G)
            file_path = os.path.join(temp_dir, f"float_graph_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(graph_object, f)

        yield temp_dir
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_mixed_weight_graphs():
    """Create temporary starGraphs with mixed integer and float weights"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate and save graphs with mixed int/float weights
        for i in range(10):
            G = nx.erdos_renyi_graph(
                n=np.random.randint(low=10, high=50),
                p=0.1 + np.random.random() * 0.4,
            )
            # Set mixed weights (integers and floats)
            for u, v in G.edges():
                # 50% chance of integer, 50% chance of float
                if np.random.random() < 0.5:
                    G[u][v]["weight"] = int(np.random.randint(1, 10))
                else:
                    G[u][v]["weight"] = round(0.1 + 9.9 * np.random.random(), 2)

            graph_object = test_star(graph=G)
            file_path = os.path.join(temp_dir, f"mixed_graph_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(graph_object, f)

        yield temp_dir
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_scale_free_graphs():
    """Create temporary starGraphs with scale-free topology and float weights"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate and save scale-free graphs with float weights
        for i in range(5):
            # Barabási-Albert scale-free graph
            G = nx.barabasi_albert_graph(
                n=np.random.randint(low=20, high=100),
                m=np.random.randint(2, 5),
            )
            # Set float weights
            for u, v in G.edges():
                # Power-law distributed weights
                G[u][v]["weight"] = np.random.pareto(3) + 0.1

            graph_object = test_star(graph=G)
            file_path = os.path.join(temp_dir, f"scale_free_graph_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(graph_object, f)

        yield temp_dir
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_weighted_graphs_collection():
    """Create a collection of different graph types with various weight structures"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Create a graph with uniform weights (all 1.0)
        G1 = nx.erdos_renyi_graph(20, 0.2)
        nx.set_edge_attributes(G1, 1.0, "weight")
        graph_object1 = test_star(graph=G1)

        # 2. Create a graph with varying float weights
        G2 = nx.erdos_renyi_graph(20, 0.2)
        for u, v in G2.edges():
            G2[u][v]["weight"] = 0.1 + 9.9 * np.random.random()
        graph_object2 = test_star(graph=G2)

        # 3. Create a graph with scientific notation range weights
        G3 = nx.erdos_renyi_graph(20, 0.2)
        for u, v in G3.edges():
            G3[u][v]["weight"] = 10 ** (np.random.uniform(-3, 3))
        graph_object3 = test_star(graph=G3)

        # 4. Create a dense weighted graph
        G4 = nx.complete_graph(15)
        for u, v in G4.edges():
            G4[u][v]["weight"] = 0.5 + np.random.random()
        graph_object4 = test_star(graph=G4)

        # 5. Create a sparse weighted graph
        G5 = nx.gnm_random_graph(50, 60)  # Very sparse
        for u, v in G5.edges():
            G5[u][v]["weight"] = 0.5 + np.random.random()
        graph_object5 = test_star(graph=G5)

        # Save all graphs to files
        graphs = [
            graph_object1,
            graph_object2,
            graph_object3,
            graph_object4,
            graph_object5,
        ]
        for i, g in enumerate(graphs):
            file_path = os.path.join(temp_dir, f"varied_graph_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(g, f)

        yield temp_dir
        shutil.rmtree(temp_dir)
