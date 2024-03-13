# File: src/projecting/__init__.py 
# Last Update: 03-04-24
# Updated by: SW 


import os

# Silencing UMAP Warnings
import warnings
from numba import NumbaDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
os.environ["KMP_WARNINGS"] = "off"

# Relative Imports
from .pSpace import pSpace 
from .pGen import pGen 
