import pytest
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from coal_mapper.nammu.curvature import ollivier_ricci_curvature, forman_curvature
from coal_mapper.nammu.topology import PersistenceDiagram

from coal_mapper.utils import MapperTopology as MTop

# Randomly Sampled Data
data = np.random.rand(100, 15)
cover = (3, 0.2)  # (n_cubes,perc_overlap)


class TestMapperTopology:
    def test_init(self):
        test = MTop(X=data)
        assert type(test.data) == np.ndarray
        assert test._graph == None
        assert test.curvature == None
        assert test.diagram == None

    def test_attributes(self):
        test = MTop(X=data)
        test.set_graph(cover)
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

            assert isinstance(dgm0.total_persistence(), float)
            assert isinstance(dgm1.total_persistence(), float)
