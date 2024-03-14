# File:tests/test_utils.py
# Last Updated: 03-12-24 
# Updated By: SW 

import yaml 
import tempfile
import pickle
import pandas as pd 


def create_temp_yaml(data:str,
                     out_dir:str,
                     runName="test", 
                     cleaning={ 
                        "scaler": "standard",
                        "encoding": "one_hot",
                        "num_samples": 1,
                        "dropColumns": None,
                        "imputeColumns": [],
                        "impute_methods": []},
                        projecting={
                            "projector": "UMAP",
                            "umap":{"nn": [2],
                                    "minDists": [0.1, 0.2],
                                    "seed": [30, 31]
                                    }
                        }
                        ):
    """
    Creates a temporary YAML file containing the provided input parameters.

    Parameters:
        Run_Name (str): The name of the run.
        data (str): The path to the data file.
        cleaning (dict): Dictionary containing cleaning configuration.
        projecting (dict): Dictionary containing projecting configuration.

    Returns:
        str: The path to the temporary YAML file.
    """
    # Create dictionary to hold the parameters
    parameters = {
        "runName": runName,
        "data": data,
        "cleaning": cleaning,
        "projecting": projecting
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.yaml',mode = "w", delete=False) as temp_file:
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
            if isinstance(data, pd.DataFrame):
                data.to_pickle(temp_file.name)
            else: 
                with open(temp_file.name, 'wb') as f:
                    pickle.dump(data, f)
        elif file_format == 'csv':
            data.to_csv(temp_file.name, index=False)
        elif file_format == 'xlsx':
            data.to_excel(temp_file.name, index=False)
        # Get the file path
        temp_file_path = temp_file.name
    return temp_file_path


test_data_0 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, None],
        'C': ['a', 'b', 'c']
    })


test_data_1 = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': ['a', 'b', None, 'd', 'd'],
    'C': ['x', 'y', 'z', None, 'w'],
    'D': [None, 10, 20, 30, 40],
    'E': ['p', 'q', 'r', 'r', None],
    'F': ['u', 'v', None, 'x', 'y']
})

test_data_2 = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': ['a', None, 'c', 'd', 'e'],
    'Z': [None, 10, 20, 30, 40],
    'W': ['p', 'q', 'r', 's', 't'],
    'V': ['u', 'v', 'w', 'x', None],
    'U': ['i', 'j', 'k', 'l', 'm']
})

test_data_0_missingData_summary = {
            'numericMissing': ['B'],
            'numericComplete': ['A'],
            'categoricalMissing': [],
            'categoricalComplete': ['C']
}

test_data_1_missingData_summary = {
            'numericMissing': ['A', 'D'],
            'numericComplete': [],
            'categoricalMissing': ['B', 'C', 'E', 'F'],
            'categoricalComplete': []
}

test_data_2_missingData_summary = {
            'numericMissing': ['Z'],
            'numericComplete': ['X'],
            'categoricalMissing': ['Y','V'],
            'categoricalComplete': ['W', 'U']
}


test_cleanData_0 = pd.DataFrame({
    'A': [-1.224745, 0.000000, 1.224745],
    'B': [-0.656570, 1.413037, -0.756467],
    'impute_B': [-0.707107, -0.707107, 1.414214],
    'impute_C': [-0.707107, -0.707107, 1.414214],
    'OH_C_a': [0.707107, -1.414214, 0.707107],
    'OH_C_c': [-0.707107, 1.414214, -0.707107]
})



test_dict_1 = {"data": test_cleanData_0,
               "description": "Nonsense"}


