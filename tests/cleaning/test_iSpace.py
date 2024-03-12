# File: tests/test_iSpace.py 
# Lasted Updated: 03-12-24 
# Updated By: SW 

import os 
import pytest
import yaml
import pickle 
import tempfile
import pandas as pd 
from ...src import cleaning as c 


# TODO: Move these objects to a tests_utils file 
# Testing Data Frames 

test_data_0 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, None],
        'C': ['a', 'b', 'c']
    })

def create_temp_yaml(Run_Name, data, cleaning):
    """
    Creates a temporary YAML file containing the provided input parameters.

    Parameters:
        Run_Name (str): The name of the run.
        data (str): The path to the data file.
        cleaning (dict): Dictionary containing cleaning configuration.

    Returns:
        str: The path to the temporary YAML file.
    """
    # Create dictionary to hold the parameters
    parameters = {
        "Run_Name": Run_Name,
        "data": data,
        "cleaning": cleaning
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        # Write parameters to the temporary YAML file
        yaml.dump(parameters, temp_file, default_flow_style=False)
        # Get the file path
        temp_file_path = temp_file.name
    return temp_file_path


def create_temp_data_file(data, file_format):
    """
    Creates a temporary file containing the given DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame to be saved.
        file_format (str): The format in which to save the DataFrame. Supported formats: 'pickle', 'csv', 'xlsx'.

    Returns:
        str: The path to the temporary file.
    """
    if file_format not in ['pkl', 'csv', 'xlsx']:
        raise ValueError("Unsupported file format. Supported formats: 'pickle', 'csv', 'xlsx'")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.' + file_format, delete=False) as temp_file:
        # Save the DataFrame into the temporary file based on the specified format
        if file_format == 'pkl':
            data.to_pickle(temp_file.name)
        elif file_format == 'csv':
            data.to_csv(temp_file.name, index=False)
        elif file_format == 'xlsx':
            data.to_excel(temp_file.name, index=False)
        # Get the file path
        temp_file_path = temp_file.name
    return temp_file_path


class test_iSpace:
    """
    Test class for iSpace
    """

    def test_init_0(self, data=test_data_0):
        my_iSpace = c.iSpace(data=data)
        
        assert my_iSpace.data == test_data_0
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.drop_columns == [] 
        assert my_iSpace.data_path == None 
        assert my_iSpace.impute_columns == [] 
        assert my_iSpace.impute_methods == [] 
        assert my_iSpace.imputed_data == None 

    
    def test_init_1a(self): 
        temp_file_path = create_temp_data_file(test_data_0, "pkl")
        my_iSpace = c.iSpace(data=temp_file_path)
        
        assert my_iSpace.data == test_data_0
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.drop_columns == [] 
        assert my_iSpace.data_path == None 
        assert my_iSpace.impute_columns == [] 
        assert my_iSpace.impute_methods == [] 
        assert my_iSpace.imputed_data == None 

    def test_init_1b(self): 
        temp_file_path = create_temp_data_file(test_data_0, "pkl")
        create_temp_yaml(Run_Name="test_init_1b", data=temp_file_path, cleaning={   "scaler": "standard_scaler",
                                                                                    "encoding": "one_hot",
                                                                                    "num_samples": 1,
                                                                                    "drop_columns": None,
                                                                                    "impute_columns": ["Something"],
                                                                                    "impute_methods": ["Something Else"]
                                                                                })

        my_iSpace = c.iSpace(data=temp_file_path)

        assert my_iSpace.data == test_data_0
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.drop_columns == [] 
        assert my_iSpace.data_path == None 
        assert my_iSpace.impute_columns == [] 
        assert my_iSpace.impute_methods == [] 
        assert my_iSpace.imputed_data == None 


    def test_init_2(self): 
        temp_file_path = create_temp_data_file(test_data_0, "csv")
        my_iSpace = c.iSpace(data=temp_file_path)

        assert my_iSpace.data == test_data_0
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.drop_columns == [] 
        assert my_iSpace.data_path == None 
        assert my_iSpace.impute_columns == [] 
        assert my_iSpace.impute_methods == [] 
        assert my_iSpace.imputed_data == None 

    def test_init_3(self): 
        temp_file_path = create_temp_data_file(test_data_0, "xlsx")
        my_iSpace = c.iSpace(data=temp_file_path)

        assert my_iSpace.data == test_data_0
        assert my_iSpace.scaler == "standard"
        assert my_iSpace.encoding == "one_hot"
        assert my_iSpace.drop_columns == [] 
        assert my_iSpace.data_path == None 
        assert my_iSpace.impute_columns == [] 
        assert my_iSpace.impute_methods == [] 
        assert my_iSpace.imputed_data == None

