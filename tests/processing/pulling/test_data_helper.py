# test_data_helper.py

# Description: 
#   Testing functionality of src/processing/data_helper.py and the script
#   src/processing/data_generator.py   

import pytest
import tempfile
import os
import sys
import pandas as pd
import pymongo
 
from dotenv import load_dotenv

# ##############################################################################################################################
#
# Loading file paths to import modeling functionality 
#

load_dotenv()
path_to_src = os.getenv("src")
sys.path.append(path_to_src)
import processing as pc 

################################################################################################################################
#
#
#  Outline of Necessary Unit Tests    
#      1) Edge Case handling for get_raw_data
#       2) Test file paths and edge case handling for generator script 
#
################################################################################################################################

class TestData: 
    def test_get_raw_data(self):
        # STUB! 
        assert 1 == 1
    def test_data_generator_script(self):
        # STUB! 
        assert 1 == 1
