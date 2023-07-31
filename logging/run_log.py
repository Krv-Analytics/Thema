import json 
import os 
import sys

from dotenv import load_dotenv
from datetime import datetime
from os.path import isfile 



#####################################################################################
#
#  Functions to interact with parmater json file 
#
#####################################################################################

def get_default_jmap_log():
    load_dotenv()
    JSON_PATH = os.getenv("params")
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)
    return os.getenv("root") + params_json["jmap_runlog"]

def get_default_projection_log():
    load_dotenv()
    JSON_PATH = os.getenv("params")
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)
    return os.getenv("root")+ params_json["projection_runlog"]

def get_default_runlog_dir():
    load_dotenv()
    JSON_PATH = os.getenv("params")
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)
    return os.getenv("root") + params_json["runlog_dir"]


class Run_Log():
    """
    TODO: Update Doc String

    A Class to faciliate the tracking,logging, and display of events for a gridsearch run.


    Members:
    --------

   projection_runlog
        Path to projector runlog
    
    jmap_runlog: 
        Path to the jmap generator's runlog 
    
    

    Member Functions: 
    -----------------

    log_jmap_startTime(): 
        Logs the start time  of the jmap grid search

    log_jmap_finishTime():
        Logs the finishing time of the jmap grid search

    log_projector_startTime():
        Logs the start time of the projection grid search 

    log_projector_finishTime():
        Logs the finish time of the projectino grid search 

    log_OverPopulatedMapper_EVENT(): 
        Logs a case where there are more connected componenets generated than elements 

    log_EmptyComplex_EVENT(): 
        Logs a case where an empty simplicial complex is generated
    
    """
    
    def __init__(self, jmap_runlog : str = get_default_jmap_log(), projection_runlog: str = get_default_projection_log(), runlog_dir: str = get_default_runlog_dir()):
        
        self._jmap_runlog = jmap_runlog
        self._projection_runlog = projection_runlog
        self._runlog_dir = runlog_dir


    
###########################################################################################
#
#   Member Helper Functions 
#
###########################################################################################

    def _open_jmap_runlog(self):
        with open(self._jmap_runlog, 'r') as f:
            return json.load(f)
    
    def _open_projection_runlog(self):
        with open(self._projection_runlog, 'r') as f:
            return json.load(f)
        
    def _write_jmap_runlog(self, updated_log:dict):
        with open(self._jmap_runlog, 'w') as f:
            json.dump(updated_log, f)
    
    def _write_projection_runlog(self, updated_log:dict):
        with open(self._projection_runlog, 'w') as f:
            json.dump(updated_log, f)

###########################################################################################
#
#   jmap Logging
#
###########################################################################################

    def start_jmap_log(self): 
             
            jmap_runlog_contents = {
                "name" : "jmap Grid Run Log",
                "startTime" : -1,
                "finishTime" : -1,
                "total_runTime": -1,
                "gridSize" : -1,

                # All jmap logging information should be set here
                "num_overPopulated_mappers": 0,
                "num_emptyGraphs": 0,
                "num_emptyComplex" : 0,
                "unknown_errors" : 0
            }

            if os.path.isdir(self._runlog_dir):
                with open(self._jmap_runlog, "w") as f1:
                    json.dump(jmap_runlog_contents, f1)
            else:
                os.makedirs(self._runlog_dir, exist_ok=True)
                with open(self._jmap_runlog, "w") as runlog:
                    json.dump(jmap_runlog_contents, runlog)
    
    
    def set_jmap_gridSize(self, size:int): 
        log = self._open_jmap_runlog() 
        log["gridSize"] = size 
        self._write_jmap_runlog(log)
    
    
    def log_jmap_startTime(self): 
        log = self._open_jmap_runlog() 
        log['startTime'] = str(datetime.now())
        self._write_jmap_runlog(log)
    
    def log_jmap_finishTime(self): 
        log = self._open_jmap_runlog() 
        start_time = datetime.strptime(log['startTime'], "%Y-%m-%d %H:%M:%S.%f")
        finish_time = datetime.now()
        log['finishTime'] = str(finish_time)
        log['total_runTime'] = str(finish_time - start_time)

        self._write_jmap_runlog(log)
    

    def log_overPopulatedMapper_EVENT(self):
        log = self._open_jmap_runlog() 
        log['num_overPopulated_mappers'] = log['num_overPopulated_mappers'] + 1 
        self._write_jmap_runlog(log)
    
    def log_emptyComplex_EVENT(self): 
        log = self._open_jmap_runlog() 
        log['num_emptyComplex'] = log['num_emptyComplex'] + 1 
        self._write_jmap_runlog(log)
    
    def log_emptyGraph_EVENT(self): 
        log = self._open_jmap_runlog() 
        log['num_emptyGraphs'] = log['num_emptyGraphs'] + 1 
        self._write_jmap_runlog(log)

    
    def log_unkownError_EVENT(self): 
        log = self._open_jmap_runlog() 
        log['unknown_errors'] = log['unknown_errors'] + 1 
        self._write_jmap_runlog(log)


###########################################################################################
#
#   Projector Logging
#
###########################################################################################

    def start_projector_log(self):
        if not isfile(self._projection_runlog):
            projection_runlog_contents = {
                "name": "Projector Run Log",
                "startTime" : -1,
                "finishTime" : -1,
                "total_runTime": -1,
                "gridSize" : -1,

                # All desired projector logging information should be set here
        
                "unknown_errors": 0 # TODO: Update tracked information for a projection search
            }
            
            if os.path.isdir(self._runlog_dir):
                with open(self._projection_runlog, "w") as f1:
                    json.dump(projection_runlog_contents, f1)
            else:
                os.makedirs(self._runlog_dir, exist_ok=True)
                with open(self._projection_runlog, "w") as runlog:
                    json.dump(projection_runlog_contents, runlog)
    
    
    def set_projector_gridSize(self, size:int): 
        log = self._open_projection_runlog() 
        log["gridSize"] = size 
        self._write_projection_runlog(log)
    
    
    def log_projector_startTime(self): 
        log = self._open_projector_runlog() 
        log['startTime'] = datetime.now()
        self._write_projector_runlog(log)

    def log_projector_finishTime(self): 
        log = self._open_projector_runlog() 
        log['finishTime'] = datetime.now()
        log['total_runTime'] = log['finishTime'] = log['startTime']
        self._write_projector_runlog(log)