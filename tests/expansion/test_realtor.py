import networkx as nx
import pytest
import numpy as np
from thema.expansion.realtor import Realtor


class TestRealtor:
    def test_node_docking(
        self, _test_graphData_0, _test_node_features_0, _test_group_features_0
    ):
        # Define target vector and node features
        target_vector = np.array([1.0, 2.0])
        realtor = Realtor(
            target_vector,
            _test_graphData_0,
            _test_node_features_0,
            _test_group_features_0,
        )

        best_node_index = realtor.node_docking(metric="euclidean")

        assert (target_vector == _test_node_features_0[0]).all()
        assert best_node_index == 0
