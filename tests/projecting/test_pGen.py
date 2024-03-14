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
    def test_init_dict(self):
        projector = "UMAP"
        minDist = .1
        nn = 4
        dimensions = 2 
        seed = 42
        temp_file = ut.create_temp_data_file(ut.test_dict_1, 'pkl') 
        x = pGen(data = temp_file, projector=projector, minDist=minDist,nn=nn, dimensions=dimensions, seed=seed)
    
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

    
    def test_fit_PCA(self): 
        projector = "PCA"
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, projector=projector, dimensions=dimensions, seed=seed)
        x.fit() 
        assert_frame_equal(x.data,ut.test_cleanData_0)
        assert x.projector == projector
        assert x.data_path == -1 
        assert x.dimensions == dimensions 
        assert x.seed == seed
        assert x.projection.shape[1] == dimensions 
        assert x.projection.shape[0] == x.data.shape[0]



    def test_save(self): 
        projector = "UMAP"
        minDist = .1
        nn = 2
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, projector=projector, minDist=minDist,nn=nn, dimensions=dimensions, seed=seed)
        x.fit()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            x.save(temp_file.name)
        
        assert(os.path.exists(temp_file.name))  

        
        with open(temp_file.name, "rb") as f:
            y = pickle.load(f)
            
        assert_frame_equal(x.data, y.data)
        assert np.array_equal(x.projection, y.projection)
        assert x.data_path == y.data_path 
        assert x.projector == y.projector
        assert x.nn == y.nn
        assert x.minDist == y.minDist

    
    def test_dump_UMAP_a(self): 
        projector = "UMAP"
        minDist = .1
        nn = 2
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, id=949, projector=projector, minDist=minDist,nn=nn, dimensions=dimensions, seed=seed)
        x.fit()
        with tempfile.TemporaryDirectory() as temp_dir:
            test_id = 123
            x.dump(temp_dir, id=test_id)
            assert len(os.listdir(temp_dir)) > 0

            with open(os.path.join(temp_dir, "UMAP_2D_2nn_0.1minDist_42rs__949123.pkl"), "rb") as f:
                y = pickle.load(f)

            assert np.array_equal(y["projection"], x.projection) 
            assert y["description"]["projector"] == x.projector 
            assert y["description"]["nn"] == x.nn 
            assert y["description"]["minDist"] == x.minDist 
            assert y["description"]["dimensions"] == x.dimensions 
            assert  y["description"]["seed"] == x.seed 
    

    def test_dump_UMAP_b(self): 
        projector = "UMAP"
        minDist = .1
        nn = 2
        dimensions = 2 
        seed = 42 
        temp_file = ut.create_temp_data_file(ut.test_cleanData_0, 'pkl')
        x = pGen(data=temp_file, id=949, projector=projector, minDist=minDist,nn=nn, dimensions=dimensions, seed=seed)
        x.fit()
        with tempfile.TemporaryDirectory() as temp_dir:
            test_id = 123
            x.dump(temp_dir, id=test_id)
            assert len(os.listdir(temp_dir)) > 0

            with open(os.path.join(temp_dir, "UMAP_2D_2nn_0.1minDist_42rs__949123.pkl"), "rb") as f:
                y = pickle.load(f)

            assert np.array_equal(y["projection"], x.projection) 
            assert y["description"]["projector"] == x.projector 
            assert y["description"]["nn"] == x.nn 
            assert y["description"]["minDist"] == x.minDist 
            assert y["description"]["dimensions"] == x.dimensions 
            assert  y["description"]["seed"] == x.seed 
            assert y["description"]["clean"] == temp_file 
    

    def test_dump_TSNE_a(self): 
        projector = "TSNE"
        perplexity = 2
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, id=949, projector=projector, perplexity=perplexity, dimensions=dimensions, seed=seed)
        x.fit()
        with tempfile.TemporaryDirectory() as temp_dir:
            test_id = 123
            x.dump(temp_dir, id=test_id)
            assert len(os.listdir(temp_dir)) > 0

            with open(os.path.join(temp_dir, "TSNE_2D_2perp_42rs__949123.pkl"), "rb") as f:
                y = pickle.load(f)

            assert np.array_equal(y["projection"], x.projection) 
            assert y["description"]["projector"] == x.projector 
            assert y["description"]["perplexity"] == x.perplexity
            assert y["description"]["dimensions"] == x.dimensions 
            assert  y["description"]["seed"] == x.seed 
    
    def test_dump_TSNE_b(self): 
        projector = "TSNE"
        perplexity = 2
        dimensions = 2 
        seed = 42 
        temp_file = ut.create_temp_data_file(ut.test_cleanData_0, 'pkl')
        x = pGen(data=temp_file, id=949, projector=projector, perplexity=perplexity, dimensions=dimensions, seed=seed)
        x.fit()
        with tempfile.TemporaryDirectory() as temp_dir:
            test_id = 123
            x.dump(temp_dir, id=test_id)
            assert len(os.listdir(temp_dir)) > 0

            with open(os.path.join(temp_dir, "TSNE_2D_2perp_42rs__949123.pkl"), "rb") as f:
                y = pickle.load(f)

            assert np.array_equal(y["projection"], x.projection) 
            assert y["description"]["projector"] == x.projector 
            assert y["description"]["perplexity"] == x.perplexity
            assert y["description"]["dimensions"] == x.dimensions 
            assert  y["description"]["seed"] == x.seed 
            assert y["description"]["clean"] == temp_file 

    def test_dump_PCA_a(self): 
        projector = "PCA"
        dimensions = 2 
        seed = 42 
        x = pGen(data=ut.test_cleanData_0, id=949, projector=projector, dimensions=dimensions, seed=seed)
        x.fit()
        with tempfile.TemporaryDirectory() as temp_dir:
            test_id = 123
            x.dump(temp_dir, id=test_id)
            assert len(os.listdir(temp_dir)) > 0

            with open(os.path.join(temp_dir, "PCA_2D_42rs_949123.pkl"), "rb") as f:
                y = pickle.load(f)

            assert np.array_equal(y["projection"], x.projection) 
            assert y["description"]["projector"] == x.projector 
            assert y["description"]["dimensions"] == x.dimensions 
            assert  y["description"]["seed"] == x.seed 

    def test_dump_PCA_b(self): 
        projector = "PCA"
        dimensions = 2 
        seed = 42 
        temp_file = ut.create_temp_data_file(ut.test_cleanData_0, 'pkl')
        x = pGen(data=temp_file, id=949, projector=projector, dimensions=dimensions, seed=seed)
        x.fit()
        with tempfile.TemporaryDirectory() as temp_dir:
            test_id = 123
            x.dump(temp_dir, id=test_id)
            assert len(os.listdir(temp_dir)) > 0

            with open(os.path.join(temp_dir, "PCA_2D_42rs_949123.pkl"), "rb") as f:
                y = pickle.load(f)

            assert np.array_equal(y["projection"], x.projection) 
            assert y["description"]["projector"] == x.projector 
            assert y["description"]["dimensions"] == x.dimensions 
            assert  y["description"]["seed"] == x.seed 
            assert y["description"]["clean"] == temp_file 
        

    def test_determinism(self): 
        pass 

    