import os
import sys

from dotenv import load_dotenv

load_dotenv()
root = os.getenv("root")
sys.path.append(root)

from scripts.python.utils import env
