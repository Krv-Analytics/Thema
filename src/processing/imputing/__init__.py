import os
import sys

from dotenv import load_dotenv

load_dotenv()
root = os.getenv("root")
sys.path.append(root)
sys.path.append(root + "src/")

import processing
from scripts.python.utils import env
