import pytest
import networkx as nx
import pandas as pd
import numpy as np

from coal_mapper.mapper import CoalMapper

# Randomly Sampled Numeric Data
data = (
    pd.read_csv("../data_processing/data.csv")
    .select_dtypes(include=np.number)
    .sample(200)
).values


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
        test.compute_mapper(n_cubes, perc_overlap)
        G = test.to_networkx()
        assert type(G) == nx.Graph
        assert len(G.nodes) > 0
