# test_model.py
# 
# Description: 
#   Testing functionality of src/modeling/model.py  


import pytest
import tempfile
import os
import sys
import pandas as pd
import numpy as np
import random
import pickle

import kmapper as km
import networkx as nx
import numpy as np
from hdbscan import HDBSCAN
from kmapper import KeplerMapper
 
from dotenv import load_dotenv

# ##############################################################################################################################
#
# Loading file paths to import modeling/jmapper functionality 
#

load_dotenv()
path_to_src = os.getenv("src")
sys.path.append(path_to_src)
sys.path.append(path_to_src + "/modeling/")

import modeling as md
from nammu.curvature import ollivier_ricci_curvature
from nammu.topology import PersistenceDiagram, calculate_persistence_diagrams
from nammu.utils import make_node_filtration

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