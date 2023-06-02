# test_cleaner.py

# Description: 
#   Testing functionality of src/processing/cleaning/cleaner_helper.py and the script
#   src/processing/cleaning/cleaner.py   

import pytest
import tempfile
import os
import sys
import pandas as pd
import pymongo
 
from dotenv import load_dotenv

# ##############################################################################################################################
#
# Loading file paths to import processing functionality 
#

load_dotenv()
path_to_src = os.getenv("src")
sys.path.append(path_to_src)
import processing as pc 

################################################################################################################################
#
#
#  Outline of Necessary Unit Tests    
#       1) Edge Case handling for data_cleaner 
#       2) Correctnes of integer encoder 
#       3) edge case handling for clean_data_filename 
#       4) Testing file paths for cleaner script 
#
################################################################################################################################


class TestCleaner: 
    def test_data_cleaner(self):
        # STUB! 
        assert 1 == 1 
    def test_integer_encoder(self):
        # STUB! 
        assert 1 == 1 
    def test_clean_data_filename(self):
        # STUB! 
        assert 1 == 1 
    def test_data_cleaner_script(self):
        # STUB! 
        assert 1 == 1 