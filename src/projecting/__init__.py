import os

######################################################################
# Silencing UMAP Warnings
import warnings

from numba import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

os.environ["KMP_WARNINGS"] = "off"
######################################################################

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
