# File: tests/test_iSpace.py 
# Lasted Updated: 03-12-24 
# Updated By: SW 

import os
import pytest 
import tempfile
import pickle
from pandas.testing import assert_frame_equal
import numpy as np
from thema import cleaning as c 
from tests import test_utils as ut 


class Test_iSpace:
    """
    A PyTest class for iSpace
    """

    def test_init_empty(self):
        with pytest.raises(ValueError):
            c.iSpace()

    def test_init_missingYamlFile(self):
        with pytest.raises(AssertionError):
            c.iSpace(YAML_PATH="../a/very/junk/file")

    def test_init_incorrectYamlFile(self):
            temp_file_path = ut.create_temp_data_file(ut.test_data_0, 'pkl') 
            with pytest.raises(ValueError):
                c.iSpace(YAML_PATH=temp_file_path)
    
    
    def test_init_defaults(self):
        my_iSpace = c.iSpace(data=ut.test_data_0)
        
        assert_frame_equal(my_iSpace.data, ut.test_data_0)
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.dropColumns == [] 
        assert my_iSpace.data_path is None 
        assert my_iSpace.imputeColumns == [] 
        assert my_iSpace.imputeMethods == [] 
        assert my_iSpace.imputed_data is None 

    
    def test_init_dataPath_pkl(self): 
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "pkl")
        my_iSpace = c.iSpace(data=temp_file_path)
        
        assert_frame_equal(my_iSpace.data, ut.test_data_0)
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.dropColumns == [] 
        assert my_iSpace.data_path == temp_file_path
        assert my_iSpace.imputeColumns == [] 
        assert my_iSpace.imputeMethods == [] 
        assert my_iSpace.imputed_data is None 

    def test_init_yaml_dataPath_pkl(self): 
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "pkl")
        print(temp_file_path)
        temp_yaml_path = ut.create_temp_yaml(data=temp_file_path)

        my_iSpace = c.iSpace(YAML_PATH=temp_yaml_path)

        assert_frame_equal(my_iSpace.data, ut.test_data_0)
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.dropColumns == [] 
        assert my_iSpace.data_path == temp_file_path
        assert my_iSpace.imputeColumns == [] 
        assert my_iSpace.imputeMethods == [] 
        assert my_iSpace.imputed_data is None 


    def test_init_dataPath_csv(self): 
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "csv")
        my_iSpace = c.iSpace(data=temp_file_path)

        assert_frame_equal(my_iSpace.data, ut.test_data_0)
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.dropColumns == [] 
        assert my_iSpace.data_path == temp_file_path
        assert my_iSpace.imputeColumns == [] 
        assert my_iSpace.imputeMethods == [] 
        assert my_iSpace.imputed_data is None 


    def test_init_yaml_dataPath_csv(self): 
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "csv")
        temp_yaml_path = ut.create_temp_yaml(data=temp_file_path)

        my_iSpace = c.iSpace(YAML_PATH=temp_yaml_path)


        assert_frame_equal(my_iSpace.data, ut.test_data_0)
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.dropColumns == [] 
        assert my_iSpace.data_path == temp_file_path
        assert my_iSpace.imputeColumns == [] 
        assert my_iSpace.imputeMethods == [] 
        assert my_iSpace.imputed_data is None 

    def test_init_dataPath_xlsx(self): 
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "xlsx")
        my_iSpace = c.iSpace(data=temp_file_path)

        assert_frame_equal(my_iSpace.data, ut.test_data_0)
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.dropColumns == [] 
        assert my_iSpace.data_path == temp_file_path 
        assert my_iSpace.imputeColumns == [] 
        assert my_iSpace.imputeMethods == [] 
        assert my_iSpace.imputed_data is None


    def test_init_yaml_dataPath_xlsx(self): 
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "xlsx")
        temp_yaml_path = ut.create_temp_yaml(data=temp_file_path)

        my_iSpace = c.iSpace(YAML_PATH=temp_yaml_path)

        assert_frame_equal(my_iSpace.data, ut.test_data_0)
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.dropColumns == [] 
        assert my_iSpace.data_path == temp_file_path 
        assert my_iSpace.imputeColumns == [] 
        assert my_iSpace.imputeMethods == [] 
        assert my_iSpace.imputed_data is None


    def test_init_scaler(self):
        with pytest.raises(AssertionError):
            c.iSpace(data=ut.test_data_0, scaler="a good one") 

    def test_init_encoding(self):
        with pytest.raises(AssertionError):
            c.iSpace(data=ut.test_data_0, encoding="a better one") 

        x = c.iSpace(data=ut.test_data_0, encoding="one_hot")
        y = c.iSpace(data=ut.test_data_0, encoding="integer")
        z = c.iSpace(data=ut.test_data_0, encoding="hash")

        assert x.encoding == "one_hot"
        assert y.encoding == "integer"
        assert z.encoding == "hash"


    def test_init_dropColumns(self):
        with pytest.raises(AssertionError):
            c.iSpace(data=ut.test_data_0, dropColumns="all") 
    
        my_iSpace2= c.iSpace(data=ut.test_data_0, dropColumns=['A'])
        assert my_iSpace2.dropColumns == ['A']

    def test_init_imputeColumns(self): 
        w = c.iSpace(data=ut.test_data_0, imputeColumns="ABCs")
        x = c.iSpace(data=ut.test_data_0, imputeColumns="all")
        y = c.iSpace(data=ut.test_data_0, imputeColumns="None")
        z = c.iSpace(data=ut.test_data_0, imputeColumns=["A"])

        assert w.imputeColumns == []
        assert x.imputeColumns == ['B']
        assert y.imputeColumns == [] 
        assert z.imputeColumns == ['A']
        

    def test_init_imputeMethods(self):
        with pytest.raises(AssertionError):
            c.iSpace(data=ut.test_data_0, imputeColumns=["A", "B"], imputeMethods=["mean", "median", "drop"])

        w = c.iSpace(data=ut.test_data_0, imputeColumns="all", imputeMethods="wrong")
        x = c.iSpace(data=ut.test_data_0, imputeColumns="all", imputeMethods="random_sampling")
        y = c.iSpace(data=ut.test_data_0, imputeColumns=["A", "B"], imputeMethods=None)
        z = c.iSpace(data=ut.test_data_0, imputeColumns=["A"], imputeMethods=["mean"])

        assert w.imputeMethods == ["drop"]
        assert x.imputeMethods == ["random_sampling"]
        assert y.imputeMethods == ["drop", "drop"]
        assert z.imputeMethods == ["mean"]
    

    def test_get_column_summary(self): 
        x = c.iSpace(data=ut.test_data_0) 
        y = c.iSpace(data=ut.test_data_1) 
        z = c.iSpace(data=ut.test_data_2) 
        
        assert x.get_missingData_summary() == ut.test_data_0_missingData_summary
        assert y.get_missingData_summary() == ut.test_data_1_missingData_summary
        assert z.get_missingData_summary() == ut.test_data_2_missingData_summary


    def test_get_na_as_list(self): 
        x = c.iSpace(data=ut.test_data_0) 
        y = c.iSpace(data=ut.test_data_1) 
        z = c.iSpace(data=ut.test_data_2)

        assert x.get_na_as_list() == ['B']
        assert y.get_na_as_list() == ['A', 'B', 'C', 'D', 'E', 'F']
        assert z.get_na_as_list() == ['Y', 'Z', 'V']

    def test_get_reccomended_sampling_method(self): 
        x = c.iSpace(data=ut.test_data_0) 
        y = c.iSpace(data=ut.test_data_1) 
        z = c.iSpace(data=ut.test_data_2)

        assert x.get_reccomended_sampling_method() == ['random_sampling']
        assert y.get_reccomended_sampling_method() == ['random_sampling', 'mode', 'mode', 'random_sampling', 'mode', 'mode']
        assert z.get_reccomended_sampling_method() == ['mode', 'random_sampling', 'mode']

    
    def test_fit(self): 
        x = c.iSpace(data=ut.test_data_0) 
        x.fit()
        assert len(x.imputed_data.select_dtypes(include='object').columns) == 0
        assert x.imputed_data.isna().sum().sum() == 0

        y = c.iSpace(data=ut.test_data_1, imputeColumns="all") 
        y.imputeMethods = y.get_reccomended_sampling_method()
        y.fit()
        assert len(y.imputed_data.select_dtypes(include='object').columns) == 0
        assert y.imputed_data.isna().sum().sum() == 0

        z = c.iSpace(data=ut.test_data_2, dropColumns=['Y', 'Z', 'V'])
        z.fit() 
        assert len(z.imputed_data.select_dtypes(include='object').columns) == 0
        assert z.imputed_data.isna().sum().sum() == 0


    
    def test_save(self): 

        x = c.iSpace(data=ut.test_data_0)
        x.fit()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            x.save(temp_file.name)
        
        assert(os.path.exists(temp_file.name))  

        
        with open(temp_file.name, "rb") as f:
            y = pickle.load(f)
            
        assert_frame_equal(x.data, y.data)
        assert_frame_equal(x.imputed_data, y.imputed_data)
        assert x.data_path == y.data_path 
        assert x.encoding == y.encoding
        assert x.dropColumns == y.dropColumns 
        assert x.scaler == y.scaler 
        assert x.imputeColumns == y.imputeColumns 
        assert x.imputeMethods == y.imputeMethods 
    

    def test_dump(self): 
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "pkl")
        x = c.iSpace(data=temp_file_path)
        x.fit() 
        with tempfile.TemporaryDirectory() as temp_dir:
            test_id = 123
            x.dump(temp_dir, id=test_id)
            assert len(os.listdir(temp_dir)) > 0

            filename_without_extension, extension = os.path.splitext(x.data_path)
            data_name = filename_without_extension.split('/')[-1] + f"_{x.scaler}_{x.encoding}_imputed_123.pkl"

            assert (os.path.exists(os.path.join(temp_dir, data_name))) 
            
            with open(os.path.join(temp_dir, data_name), "rb") as f:
                y = pickle.load(f)
            
            assert_frame_equal(x.imputed_data, y['data'])
            assert x.data_path == y["description"]["raw"]
            assert x.scaler == y["description"]["scaler"]
            assert x.encoding == y["description"]["encoding"]
            assert x.dropColumns == y["description"]["dropColumns"]
            assert x.imputeColumns == y["description"]["imputeColumns"]
            assert x.imputeMethods == y["description"]["imputeMethods"]




    def test_fit_space(self): 
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "pkl")
        x = c.iSpace(data=temp_file_path)

        numSamples = np.random.randint(1, 51)
        with tempfile.TemporaryDirectory() as temp_dir:
            x.fit_space(temp_dir, numSamples=numSamples)
            assert len(os.listdir(temp_dir)) == numSamples



