import pytest
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import kmapper as km


from coal_mapper.mapper import CoalMapper

# Randomly Sampled Data
data = np.random.rand(100, 15)
kmeans = KMeans(n_clusters=5, random_state=1618033, n_init="auto")


class TestCoalMapper:
    def test_init(self):
        test = CoalMapper(X=data)
        assert type(test.data) == np.ndarray
        assert test.lens == None
        assert test.clusterer == None
        assert test.cover == None
        assert test.nerve == None
        assert test.graph == None
        assert test.components == None

    def test_compute_mapper(self):
        n_cubes, perc_overlap = (3, 0.2)
        test = CoalMapper(X=data)
        test.clusterer = kmeans
        test.compute_mapper(n_cubes, perc_overlap)
        G = test.to_networkx()

        # Post Computation
        assert type(test.cover) == km.Cover
        assert type(G) == nx.Graph
        assert len(G.nodes) > 0

    def test_connected_components(self):
        n_cubes, perc_overlap = (np.random.randint(2, 10), 0.2)
        test = CoalMapper(X=data)
        test.clusterer = kmeans
        test.compute_mapper(n_cubes, perc_overlap)
        components = test.connected_components()

        assert type(components) == list
        assert len(components) >= 1 and len(components) <= len(test.graph.nodes())
        assert type(components[0]) == nx.Graph
