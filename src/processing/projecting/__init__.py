import os
import sys

######################################################################
# Silencing UMAP Warnings
import warnings

from numba import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

os.environ["KMP_WARNINGS"] = "off"
######################################################################

from dotenv import load_dotenv

load_dotenv()
root = os.getenv("root")
sys.path.append(root)

from scripts.python.utils import env
