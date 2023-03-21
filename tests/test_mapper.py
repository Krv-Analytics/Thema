import pytest
import os
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import kmapper as km
import pickle


from hdbscan import HDBSCAN
from umap import UMAP
from src.nammu.topology import PersistenceDiagram
from src.nammu.curvature import ollivier_ricci_curvature, forman_curvature
from src.mapper import CoalMapper

cwd = os.path.dirname(__file__)

file = os.path.join(cwd, "../data/coal_mapper_one_hot_scaled_TSNE.pkl")
# Randomly Sampled Data
with open(file, "rb") as f:
    df = pickle.load(f)
    print(f"The dataframe currently has {len(df.columns)}")
    df = df.dropna()

data = df.values
projection = UMAP(
    min_dist=0,
    n_neighbors=10,
    n_components=2,
    init="random",
    random_state=0,
)

print("Loaded data")

N = np.random.randint(1, 15)
clusterer = HDBSCAN(min_cluster_size=N)


class TestCoalMapper:
    def test_init(self):
        test = CoalMapper(data=data, projection=projection)
        assert type(test.data) == pd.DataFrame
        assert type(test.projection) == np.ndarray

        assert test.clusterer == None
        assert test.cover == None

        assert test.complex == dict()
        assert test.graph == nx.Graph()
        assert test.components == dict()
        assert test.curvature == np.array([])
        assert test.diagram == PersistenceDiagram()

    def test_compute_mapper(self):
        test = CoalMapper(data=data, projection=projection)

        test.fit()
        G = test.to_networkx()
        components = test.connected_components()

        # Post Computation
        assert type(test.cover) == km.Cover
        assert type(G) == nx.Graph
        assert len(G.nodes) > 0
        assert type(components) == list
        assert len(components) >= 1 and len(components) <= len(test.graph.nodes())
        assert type(components[0]) == nx.Graph

    def test_topology(self):
        test = CoalMapper(data=data, projection=projection)
        test.to_networkx()
        # Graph is networkx and Undirected
        assert not nx.is_directed(test.graph)

        for curvature_fn in [ollivier_ricci_curvature, forman_curvature]:
            # Setter Method
            test.curvature = curvature_fn
            assert type(test.curvature) == np.ndarray
            assert len(test.curvature) == len(test.graph.edges())

            test.calculate_homology(ollivier_ricci_curvature)

            assert len(test.diagram) == 2
            dgm0 = test.diagram[0]
            dgm1 = test.diagram[1]

            assert isinstance(dgm0.betti, int)
            assert isinstance(dgm1.betti, int)

            assert isinstance(dgm0.p_norm(), float)
            assert isinstance(dgm1.p_norm(), float)
