# Sys path update, then call from modeling import *. 
import os 
import sys 
from dotenv import load_dotenv
load_dotenv() 
src = os.getenv("src")
sys.path.append(src)
sys.path.append(src + "processing/")

from processing.pulling import *
from processing.projecting import *
from processing.cleaning import * 
