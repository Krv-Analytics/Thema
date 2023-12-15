# Sys path update, then call from modeling import *.
import os
import sys

from dotenv import load_dotenv

load_dotenv()
src = os.getenv("src")
root = os.getenv("root")
sys.path.append(root)
sys.path.append(src)
sys.path.append(src + "jmapping/selecting")

from jmap_selector_helper import *

from scripts.python.utils import env
