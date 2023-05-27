# test_projector.py

# Description: 
#   Testing functionality of src/processing/projecting/projector_helper.py and the script
#   src/processing/projecting/projector.py   

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
#       1) Edge Case handling for projection_driver
#       2) Edge case handling for projection_file_name
#       3) File paths for projector script 
#
################################################################################################################################


class TestProjector: 
    def test_projection_driver(self):
        # STUB! 
        assert 1 == 1 
    def test_projection_file_name(self):
        # STUB! 
        assert 1 == 1 
    def test_projector_script(self):
        # STUB! 
        assert 1 == 1 