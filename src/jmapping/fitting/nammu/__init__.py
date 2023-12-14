__version__ = "0.1.0"
import os
import sys

from dotenv import load_dotenv

load_dotenv()
src = os.getenv("src")
sys.path.append(src + "/jmapping/fitting/nammu")

from curvature import ollivier_ricci_curvature
from nammu_utils import UnionFind, make_node_filtration
from topology import PersistenceDiagram, calculate_persistence_diagrams
