# Sys path update, then call from modeling import *. 
import os 
import sys 
from dotenv import load_dotenv
load_dotenv() 
root = os.getenv("root")
sys.path.append(root + "/run_logging/")

from run_logging.run_log import Run_Log