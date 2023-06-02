# test_tupper.py 
# 
# Description: 
#   Testing functionality of src/modeling/tupper.py  



import pytest
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import random
import pickle

from dotenv import load_dotenv

###############################################################################################################################
#
# Loading file paths to import modeling/jmapper functionality 
#


load_dotenv()
path_to_src = os.getenv("src")
root = os.getenv("root")
sys.path.append(path_to_src)


import modeling as md

################################################################################################################################
# 
#   Outline of Unit Tests    
#
#       1) Empty initialization (invalid paths ) 
#       2) Unsupported file types (ie non-pickle files) 
#       3) Randomly generated raw/clean/projected (correctness)
#
################################################################################################################################

#
# Setting Temporary Testing Files 
# 

@pytest.fixture(scope="class")
def tmp_files(request, tmp_path_factory):
    
    # Setting temporary files handled by pyTest 
    #path_to_root = os.path.normpath(root)
    temp_dir = Path(root + "temp_testing_dir")
    os.mkdir(temp_dir)

    raw_file = temp_dir / "raw.pkl"
    clean_file = temp_dir / "clean.pkl"
    projection_file = temp_dir / "projection.pkl"
    invalid_file = temp_dir / "invalid.txt"

    # Generating random raw data frame 
    num_rows = 30 
    num_columns = 25

    columns = ['Column' + str(i) for i in range(num_columns)]
    data = {}
    for column in columns:
        data[column] = np.random.randint(1, 10, size=num_rows)

    raw_df = pd.DataFrame(data)  

    # Number of data columns to drop (at random) to create clean data frame  
    num_dropped = 5

    columns_to_drop = random.sample(list(raw_df.columns), k=num_dropped)
    undropped_df = raw_df.drop(columns_to_drop, axis=1)
    clean_df = {"clean_data": undropped_df, "dropped_columns": columns_to_drop}

    # Creating a fake sample projection
    # 
    projection_data = np.random.randn(num_rows, 2)
    
    projection = pd.DataFrame(projection_data)
    results = {"projection": projection, "hyperparameters": [10, 10, 2]}


    # Populate temporary files 
    
    with open(raw_file, 'wb') as f1:
        pickle.dump(raw_df, f1)

    with open(clean_file, 'wb') as f2:
        pickle.dump(clean_df, f2)
    
    with open(projection_file, 'wb') as f3: 
        pickle.dump(results, f3)

    with open(invalid_file, 'w') as f4: 
        f4.write("This is an invalid file for the tupper container!")

    # Provide the paths to the test class
    request.cls.raw_file = "temp_testing_dir/raw.pkl"
    request.cls.clean_file = "temp_testing_dir/clean.pkl"
    request.cls.projection_file = "temp_testing_dir/projection.pkl"
    request.cls.invalid_file = "temp_testing_dir/invalid.txt"

    yield

    # Clean up temporary files
    if os.path.exists(raw_file):
        os.remove(raw_file)

    if os.path.exists(clean_file):
        os.remove(clean_file)
    
    if os.path.exists(projection_file):
        os.remove(projection_file)
    
    if os.path.exists(invalid_file):
        os.remove(invalid_file)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


################################################################################################################################
#
# Testing class 
#

@pytest.mark.usefixtures("tmp_files")
class TestTupper:
    def test_init(self):
        # Invalid paths 
        phantom_file = "path/to/a/nonexisting/location"

        with pytest.raises(AssertionError) as exc_info1: 
            test1a = md.Tupper(raw = phantom_file, clean=self.clean_file, projection=self.projection_file)

        with pytest.raises(AssertionError) as exc_info2: 
            test1b = md.Tupper(raw = self.raw_file, clean=phantom_file, projection=self.projection_file)
        
        with pytest.raises(AssertionError) as exc_info3: 
            test1c = md.Tupper(raw = self.raw_file, clean=self.clean_file, projection=phantom_file)
        
        assert f"Invalid raw file path: {phantom_file}" == str(exc_info1.value)
        assert f"Invalid clean file path: {phantom_file}" == str(exc_info2.value)
        assert f"Invalid projection file path: {phantom_file}" == str(exc_info3.value)

    @pytest.mark.usefixtures("tmp_files")
    def test_raw(self): 
        # Testing Incorrect data types

        test2 = md.Tupper(raw = self.invalid_file, clean=self.clean_file, projection=self.projection_file)
        with pytest.raises(Exception):
            test2.raw() 
        
        test3 = md.Tupper(raw = self.raw_file, clean=self.clean_file, projection=self.projection_file)
        with open(root + self.raw_file, 'rb') as f1:
            expected_raw_data = pickle.load(f1)
        
        assert isinstance(test3.raw, pd.DataFrame)
        assert expected_raw_data.equals(test3.raw)
        

    @pytest.mark.usefixtures("tmp_files")
    def test_clean(self): 
        # Testing Incorrect data type
        
        test2 = md.Tupper(raw = self.raw_file, clean= self.invalid_file, projection= self.projection_file)

        with pytest.raises(Exception):
            test2.clean()
        
        test3 = md.Tupper(raw = self.raw_file, clean=self.clean_file, projection=self.projection_file)

        with open(root + self.clean_file, 'rb') as f2:
            reference = pickle.load(f2)
        
        expected_clean_data = reference["clean_data"]
        expected_dropped_columns = reference["dropped_columns"]

        actual_dropped_columns = test3.get_dropped_columns()

        assert isinstance(test3.clean, pd.DataFrame)
        assert expected_clean_data.equals(test3.clean)
        assert isinstance(actual_dropped_columns, list)
        assert actual_dropped_columns == expected_dropped_columns
        

    @pytest.mark.usefixtures("tmp_files")
    def test_projection(self): 
        # Testing Incorrect data type
        test2 = md.Tupper(raw = self.raw_file, clean=self.clean_file, projection=self.invalid_file)
        with pytest.raises(Exception):
            test2.projection()
        
        test3 = md.Tupper(raw = self.raw_file, clean=self.clean_file, projection=self.projection_file)
        with open(root + self.projection_file, 'rb') as f3:
            reference = pickle.load(f3)
        expected_projection_data = reference["projection"]
        expected_hyperparameters = reference["hyperparameters"]

        actual_hyperparameters = test3.get_projection_parameters()
        
        assert isinstance(test3.projection, pd.DataFrame)
        assert expected_projection_data.equals(test3.projection)
        assert isinstance(actual_hyperparameters, list)
        assert actual_hyperparameters == expected_hyperparameters
