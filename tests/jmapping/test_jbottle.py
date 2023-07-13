"Testing file for the JBottle Class"

import pytest
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import random
import pickle

from dotenv import load_dotenv

###############################################################################################################################
#
# Loading file paths to import modeling/jmapper functionality 
#


load_dotenv()
path_to_src = os.getenv("src")
root = os.getenv("root")
sys.path.append(path_to_src)


import jmapping as md


###############################################################################################################################


# Initialize a tupper 
# Create raw/cleaned/projected data frames 

# create a Kepler Mapper Simiplicial Complex 
# 



