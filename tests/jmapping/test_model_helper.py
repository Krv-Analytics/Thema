# test_jmap_helper.py

# Description: 
#   Testing functionality of src/jmapping/jmap_helper.py 

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
# Loading file paths to import jmapping functionality 
#

load_dotenv()
path_to_src = os.getenv("src")
sys.path.append(path_to_src)
sys.path.append(path_to_src + "/jmapping/")

import jmapping as md
from jmapping.jmap_helper import (
    generate_jmap_filename,
    jmap_generator,
    env,
    script_paths,
)

################################################################################################################################
#
#
#  Outline of Necessary Unit Tests    
#       1) Edge case handling for HDBSCAN and min_intersection
#       2) Testing Correctness of file name generation 
#       3) Testing correctness of min standard deviation 
#       4) Test jmap_generator.py script 
#
################################################################################################################################

#
# Setting Temporary Testing Files 
#

# TODO:  
# create neccessary data frames for intilization of JMapper objects for testing min intersection and testing of HDBSCAN 
# Expected filename generation 
# dummy dataset and edge case handling for min standard deviation 

################################################################################################################################


class Testjmap_helper: 

    def test_jmap_generator(self):
        # STUB!
        assert 1 == 1 
    def test_generate_jmap_filename(self): 
        # STUB! 
        assert 1 == 1
    def test_mapper_plot_outfile(self):
        # STUB! 
        assert 1 == 1 
    def test_get_minimal_std(self): 
        # STUB! 
        assert 1 == 1
    def test_jmap_generator_script(self):
        # STUB! 
        assert 1==1 

