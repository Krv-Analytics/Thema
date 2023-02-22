import pytest
import networkx as nx
import pandas as pd
import numpy as np

from coal_mapper.analysis import MapperAnalysis

# Randomly Sampled Numeric Data
data = (
    pd.read_csv("../data_processing/data.csv")
    .select_dtypes(include=np.number)
    .sample(200)
).values


class TestAttributes:
    def test_init(self):
        test = MapperAnalysis(X=data)
        assert type(test.data) == np.ndarray
        assert test.graph == None
        assert test.curvature == None
        assert test.diagrams == None

    def test_graph_set(self):
        cover = (3, 0.2)
        test = MapperAnalysis(X=data)
        test.graph = cover
        assert not nx.is_directed(test.graph)


class TestCurvature:
    def check():

        assert 1 == 1


class TestPersistentHomology:
    def check():
        assert True == True
