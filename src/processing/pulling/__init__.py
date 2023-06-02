import os
import sys 
from dotenv import load_dotenv
load_dotenv()
src = os.getenv("src")
sys.path.append(src + "/processing/pulling")

from mongo import Mongo 
from data_helper import * 