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

from dotenv import load_dotenv

# ##############################################################################################################################
#
# Loading file paths to import modeling/jmapper functionality 
#
load_dotenv()
path_to_src = os.getenv("src")
sys.path.append(path_to_src)

import modeling as md

################################################################################################################################
#
# Initilizing Data for Unit Tests 
#
#  Unit Tests    
#       1) Empty initialization (invalid paths ) 
#       2) Unsupported file types (ie non-pickle files) 
#       3) Randomly generated raw/clean/projected (correctness)
#
################################################################################################################################

#
# Setting Temporary Testing Files 
#