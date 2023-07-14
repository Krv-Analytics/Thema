# Sys path update, then call from modeling import *. 
import os 
import sys 
from dotenv import load_dotenv
load_dotenv() 
src = os.getenv("src")
sys.path.append(src)
sys.path.append(src + "jmapping/fitting")

from tupper import Tupper
from jgraph import JGraph 
from jmapper import JMapper

sys.path.append(src + "modeling/literals")
from jbottle import JBottle
from model import Model



