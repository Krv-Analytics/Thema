# File: tests/multiverse/universe/test_galaxy.py
# Lasted Updated: 04-06-24
# Updated By: SW


import os
import numpy as np
import networkx as nx

from thema.multiverse.universe import geodesics,starGraph



class Test_Geodesics:
    """Pytest class for Geodesics"""

    def test_load_starGraphs(self, temp_starGraphs):
        files = temp_starGraphs
        graphs = geodesics._load_starGraphs(files)
        assert len(graphs) == len(os.listdir(files))
        for id_,sG in graphs.items():
            assert os.path.exists(id_)
            assert isinstance(sG.graph,nx.Graph)
        


    def test_stellar_kernel_distance(self, temp_starGraphs):
        files = temp_starGraphs
        keys,M = geodesics.stellar_kernel_distance(files,filterfunction=None)
        assert len(keys) == M.shape[0]
        assert isinstance(M, np.ndarray)
        for key in keys:
            assert os.path.exists(key)    
        assert np.allclose(M, M.T)
	
