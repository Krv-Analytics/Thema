# File: tests/test_iSpace.py 
# Lasted Updated: 03-13-24 
# Updated By: SW 

import os
import pytest 
import tempfile
import pickle
from pandas.testing import assert_frame_equal
import numpy as np
from thema.projecting import pGen
from tests import test_utils as ut 


class Test_pGen: 
    """
    Testing class for pGen 
    """

    def test_init_empty(self): 
        with pytest.raises(TypeError): 
            pGen()
    
    def test_init_defaults(self): 
        x = pGen(ut.test_cleanData_0)

        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == "UMAP"
        assert x.minDist == .1
        assert x.data_path == -1 
        assert x.nn == 4
        assert x.dimensions == 2 
        assert x.seed == 42 

    
    def test_init_UMAP_a(self): 
        projector = "UMAP"
        minDist = .1
        nn = 4
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, projector=projector, minDist=minDist,nn=nn, dimensions=dimensions, seed=seed)

        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.minDist == minDist
        assert x.data_path == -1 
        assert x.nn == nn
        assert x.dimensions == dimensions 
        assert x.seed == seed


    def test_init_UMAP_b(self): 
        temp_file = ut.create_temp_data_file(ut.test_cleanData_0, 'pkl')
        projector = "UMAP"
        minDist = .1
        nn = 4
        dimensions = 2 
        seed = 42 
        x = pGen(data=temp_file, projector=projector, minDist=minDist,nn=nn, dimensions=dimensions, seed=seed)

        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.minDist == minDist
        assert x.data_path == temp_file 
        assert x.nn == nn
        assert x.dimensions == dimensions 
        assert x.seed == seed   


    def test_init_TSNE_a(self): 
        projector = "TSNE"
        perplexity = 2
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, projector=projector, perplexity=perplexity, dimensions=dimensions, seed=seed)

        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.data_path == -1 
        assert x.perplexity == perplexity
        assert x.dimensions == dimensions 
        assert x.seed == seed


    def test_init_TSNE_b(self): 
        temp_file = ut.create_temp_data_file(ut.test_cleanData_0, 'pkl')
        projector = "TSNE"
        perplexity = 2
        dimensions = 2 
        seed = 42 
        x = pGen(data=temp_file, projector=projector, perplexity=perplexity, dimensions=dimensions, seed=seed)

        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.data_path == temp_file
        assert x.perplexity == perplexity
        assert x.dimensions == dimensions 
        assert x.seed == seed

    def test_init_PCA_a(self): 
        projector = "PCA"
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, projector=projector, dimensions=dimensions, seed=seed)

        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.data_path == -1 
        assert x.dimensions == dimensions 
        assert x.seed == seed


    def test_init_PCA_b(self): 
        temp_file = ut.create_temp_data_file(ut.test_cleanData_0, 'pkl')
        projector = "PCA"
        dimensions = 2 
        seed = 42 
        x = pGen(data=temp_file, projector=projector, dimensions=dimensions, seed=seed)

        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.data_path == temp_file
        assert x.dimensions == dimensions 
        assert x.seed == seed

    def test_fit_UMAP(self): 
        projector = "UMAP"
        minDist = .1
        nn = 2
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, projector=projector, minDist=minDist,nn=nn, dimensions=dimensions, seed=seed)
        x.fit()
        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.minDist == minDist
        assert x.data_path == -1 
        assert x.nn == nn
        assert x.dimensions == dimensions 
        assert x.seed == seed
        assert x.projection.shape[1] == dimensions 
        assert x.projection.shape[0] == x.data.shape[0]

    
    def test_fit_TSNE(self): 
        projector = "TSNE"
        perplexity = 2
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, projector=projector, perplexity=perplexity, dimensions=dimensions, seed=seed)
        x.fit() 
        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.perplexity == perplexity
        assert x.data_path == -1 
        assert x.dimensions == dimensions 
        assert x.seed == seed
        assert x.projection.shape[1] == dimensions 
        assert x.projection.shape[0] == x.data.shape[0]
