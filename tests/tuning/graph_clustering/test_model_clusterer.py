# test_model_clusterer.py

# Description: 
#   Testing functionality of src/tuning/graph_clustering/model_clusterer.py and the script
#   src/tuning/graph_clustering/model_clusterer_helper.py   

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
import tuning as tn 

################################################################################################################################
#
#
#  Outline of Necessary Unit Tests    
#       1) Edge case handling for cluster_models 
#       2) Edge Case handling for read_distance_metrics
#       3) Paths/parameter arguments for model clusterer script
#
################################################################################################################################

class TestModel_Clusterer:
    
    def test_cluster_models(self):
        # STUB! 
        assert 1 == 1
    
    def test_read_distance_matrices(self):
        # STUB! 
        assert 1==1
    
    def test_model_clusterer_script(self):
        # STUB! 
        assert 1==1 