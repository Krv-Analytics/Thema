
# Sys path update, then call from modeling import *.
import os
import sys

from dotenv import load_dotenv

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(root)
sys.path.append(src)
sys.path.append(src + "jmapping/fitting")
from scripts.python.utils import env

