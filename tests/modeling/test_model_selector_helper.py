# test_model_helper.py

# Description: 
#   Testing functionality of src/modeling/model_selector_helper.py 

import pytest
import tempfile
import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
 
from dotenv import load_dotenv

# ##############################################################################################################################
#
# Loading file paths to import modeling functionality 
#

load_dotenv()
path_to_src = os.getenv("src")
sys.path.append(path_to_src)
sys.path.append(path_to_src + "/modeling/")

import modeling as md
from modeling.model_selector_helper import (
    read_graph_clustering,
    select_models,
    plot_mapper_histogram,
)

################################################################################################################################
#
#
#  Outline of Necessary Unit Tests    
#       1) Edge case handling for select_models, get_clustering_subgroups 
#       2) Edge case handling for read_graph_clustering
#       3) Edge case handling of unpack_policy_group_dir
#       4) Correctness for get_viable_models
#       5) Correctness for plot_mapper_histogram 
#       6) Correctness for get_best_covered
#           
#       7) Testing script 
#
################################################################################################################################


class TestModel_Selector: 

    def test_select_models(self):
        # STUB! 
        assert 1 == 1 
    def test_read_graph_clustering(self): 
        # STUB! 
        assert 1 == 1 
    def test_unpack_policy_group_dir(self): 
        # STUB! 
        assert 1 == 1 
    def test_get_viable_models(self): 
        # STUB! 
        assert 1 == 1 
    def test_plot_mapper_histogram(self): 
        # STUB! 
        assert 1 == 1 
    def get_best_covered(self): 
        # STUB! 
        assert 1 == 1 
    
    def test_model_selector_script(self): 
        # STUB! 
        assert 1 == 1 
    
