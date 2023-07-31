# test_jmap.py
# 
# Description: 
#   Testing functionality of src/jmapping/model.py  


import pytest
import tempfile
import os
import sys

import itertools
import pickle
from os.path import isfile

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
 
from dotenv import load_dotenv

# ##############################################################################################################################
#
# Loading file paths to import modeling functionality 
#

load_dotenv()
path_to_src = os.getenv("src")
sys.path.append(path_to_src)
sys.path.append(path_to_src + "/modeling/")

import jmapping as md
from jmapping import nammu

################################################################################################################################
#
# TODO: Summarize neccessary unit tests 
#
#  Outline of Necessary Unit Tests    
#       1) Empty initialization (Edge cases ie empty models, invalid tuppers, etc) 
#
################################################################################################################################

#
# Setting Temporary Testing Files 
#

# TODO 
# Create the necessary files to initialize models to serve as unit tests 


################################################################################################################################

class TestModel: 
    def test_init(self):
        # STUB!
        assert 1==1
    def test_mapper(self):
        # STUB!
        assert 1==1
    def test_tupper(self):
        # STUB!
        assert 1==1
    def test_hyper_parameters(self):
        # STUB!
        assert 1==1
    def test_complex(self): 
        # STUB! 
        assert 1==1 
    def test_node_ids(self): 
        # STUB!
        assert 1==1
    def test_node_description(self):
        # STUB!
        assert 1==1
    def test_cluster_ids(self):
        # STUB!
        assert 1==1
    def test_cluster_descriptions(self):
        # STUB!
        assert 1==1
    def test_cluster_sizes(self):
        # STUB!
        assert 1==1
    def test_unclustered_items(self):
        # STUB!
        assert 1==1
    def test_label_item_by_node(self):
        # STUB!
        assert 1==1
    def test_label_item_by_cluster(self):
        # STUB!
        assert 1==1
    def test_compute_node_descriptions(self):
        # STUB!
        assert 1==1
    def test_compute_cluster_descriptions(self): 
        # STUB!
        assert 1==1
    def test_get_node_dfs(self):
        # STUB!
        assert 1==1
    def test_get_cluster_dfs(self):
        # STUB!
        assert 1==1
    def test_visualize_model(self): 
        # STUB!
        assert 1==1
    def test_visualize_component(self):
        # STUB!
        assert 1==1
    def test_visualize_projection(self):
        # STUB!
        assert 1==1
    def test_visualize_curvature(self): 
        # STUB!
        assert 1==1
    def test_visualize_persistence_diagram(self):
        # STUB!
        assert 1==1
    def test_visualize_mapper(self):
        # STUB!
        assert 1==1