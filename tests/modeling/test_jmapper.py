# test_jmapper.py 
# 
# Description: 
#   Testing functionality of src/modeling/jmapper.py  

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
#       1) Empty initialization (Edge cases ie empty mappers, invalid tuppers, etc) 
#
################################################################################################################################

#
# Setting Temporary Testing Files 
#

# TODO 
# Create the necessary files to initialize mappers to serve as unit tests 


################################################################################################################################


# TODO: Complete these Stubs! 

class TestJMapper:
    def test_init(self):
        # Stub
        assert 1==1
    def test_tupper(self):
        # Stub 
        assert 1==1
    def test_mapper(self):
        # Stub 
        assert 1==1
    def test_cover(self):
        # Stub 
        assert 1==1
    def test_clusterer(self):
        # Stub 
        assert 1==1
    def test_complex(self):
        # Stub 
        assert 1==1
    def test_graph(self):
         # Stub 
        assert 1==1
    def test_min_intersection(self):
         # Stub 
        assert 1==1
    def test_components(self):
         # Stub 
        assert 1==1
    def test_num_policy_groups(self):
         # Stub 
        assert 1==1
    def test_curvature(self):
         # Stub 
        assert 1==1
    def test_diagram(self):
         # Stub 
        assert 1==1
    def test_to_networkx(self):
         # Stub 
        assert 1==1
    def test_connected_components(self):
         # Stub 
        assert 1==1
    def test_calculate_homology(self):
         # Stub 
        assert 1==1
    def test_item_lookup(self):
        # Stub 
        assert 1==1 
