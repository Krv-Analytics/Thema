# test_mongo.py

# Description: 
#   Testing functionality of src/processing/mongo.py 

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
#      1) Edge Case handling for push and pull
#
################################################################################################################################


class TestMongo: 
    def test_init(self):
        #STUB!
        assert 1 == 1
    def test_pull(self): 
        #STUB! 
        assert 1==1 
    def test_push(self): 
        # STUB!
        assert 1==1 
