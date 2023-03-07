import pytest
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import kmapper as km
import pickle


from src.mapper import CoalMapper

file = "/Users/jeremy.wayland/Desktop/Dev/coal_mapper/data_processing/local_data/coal_mapper.pkl"
# Randomly Sampled Data
with open(file, "rb") as f:
    df = pickle.load(f)
    print(f"The dataframe currently has {len(df.columns)}")
    df = pd.get_dummies(df, prefix="One_hot", prefix_sep="_")
    df = df.sample(n=5, axis="columns")


print("Loaded data")
data = df.select_dtypes(include=np.number).values[:100]

kmeans = KMeans(n_clusters=8, random_state=1618033, n_init="auto")


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
        n_cubes, perc_overlap = (6, 0.4)
        test = CoalMapper(X=data)
        test.clusterer = kmeans
        test.compute_mapper(n_cubes, perc_overlap)
        G = test.to_networkx()
        components = test.connected_components()

        # Post Computation
        assert type(test.cover) == km.Cover
        assert type(G) == nx.Graph
        assert len(G.nodes) > 0
        assert type(components) == list
        assert len(components) >= 1 and len(components) <= len(test.graph.nodes())
        assert type(components[0]) == nx.Graph
