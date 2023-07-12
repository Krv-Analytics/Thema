import json 
import os 
import sys

from dotenv import load_dotenv
from datetime import datetime
from os.path import isfile 


def get_default_model_log():
    load_dotenv()
    JSON_PATH = os.getenv("params")
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)
    return params_json["model_runlog"]

def get_default_projection_log():
    load_dotenv()
    JSON_PATH = os.getenv("params")
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)
    return params_json["projection_runlog"]

def get_default_runlog_dir():
    load_dotenv()
    JSON_PATH = os.getenv("params")
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)
    return params_json["runlog_dir"]


class Run_Log():
    """
    A Class to faciliate the tracking,logging, and display of events for a gridsearch run.


    Members:
    --------

   projection_runlog
        Path to projector runlog
    
    model_runlog: 
        Path to the model generator's runlog 
    
    

    Member Functions: 
    -----------------

    log_model_startTime(): 
        Logs the start time  of the model grid search

    log_model_finishTime():
        Logs the finishing time of the model grid search

    log_projector_startTime():
        Logs the start time of the projection grid search 

    log_projector_finishTime():
        Logs the finish time of the projectino grid search 

    log_OverPopulatedMapper_EVENT(): 
        Logs a case where there are more connected componenets generated than elements 

    log_EmptyComplex_EVENT(): 
        Logs a case where an empty simplicial complex is generated
    
    """
    
    def __init__(self, model_runlog : str = get_default_model_log(), projection_runlog: str = get_default_projection_log(), runlog_dir: str = get_default_runlog_dir()):
        
        self._model_runlog = model_runlog
        self._projection_runlog = projection_runlog
        self._runlog_dir = runlog_dir

        if not isfile(model_runlog): 
            model_runlog_contents = {
                "name" : "Model Grid Run Log",
                "start_time" : -1,
                "finish_time" : -1,
                "total_runTime": -1,
                "size_of_grid" : -1,

                # All model logging information should be set here
                "num_overPopulated_mappers": 0,
                "num_empty_mappers": 0,
                "unknown_errors" : 0
            }

            if os.path.isdir(self._runlog_dir):
                with open(model_runlog, "w") as f1:
                    json.dump(model_runlog_contents, f1)
            else:
                os.makedirs(runlog_dir, exist_ok=True)
                with open(model_runlog, "w") as runlog:
                    json.dump(model_runlog_contents, runlog)


        if not isfile(projection_runlog):
            projection_runlog_contents = {
                "name": "Projector Run Log",
                "start_time" : -1,
                "finish_time" : -1,
                "total_runTime": -1,
                "size_of_grid" : -1,

                # All desired projector logging information should be set here
        
                "unknown_errors": 0 # TODO: Improve error info 
            }
            
            if os.path.isdir(self._runlog_dir):
                with open(model_runlog, "w") as f1:
                    json.dump(projection_runlog_contents, f1)
            else:
                os.makedirs(runlog_dir, exist_ok=True)
                with open(model_runlog, "w") as runlog:
                    json.dump(projection_runlog_contents, runlog)

    

    def _open_model_runlog(self):
        with open(self._model_runlog) as f:
            return json.load(f)
    
    def _open_projection_runlog(self):
        with open(self._projection_runlog) as f:
            return json.load(f)
        
    def _write_model_runlog(self, updated_log:dict):
        with open(self._model_runlog) as f:
            json.dump(updated_log, f)
    
    def _write_projection_runlog(self, updated_log:dict):
        with open(self._projection_runlog) as f:
            json.dump(updated_log, f)

    def log_model_startTime(self): 
        model_log = self._open_model_runlog() 
        model_log['startTime'] = datetime.now()
        self._write_model_runlog(model_log)
    
    def log_model_finishTime(self): 
        model_log = self._open_model_runlog() 
        model_log['finishTime'] = datetime.now()
        model_log['total_runTime'] = model_log['finishTime'] = model_log['startTime']
        self._write_model_runlog(model_log)

    def log_projector_startTime(self): 
        projector_log = self._open_projector_runlog() 
        projector_log['startTime'] = datetime.now()
        self._write_projector_runlog(projector_log)

    def log_projector_finishTime(self): 
        projector_log = self._open_projector_runlog() 
        projector_log['finishTime'] = datetime.now()
        projector_log['total_runTime'] = projector_log['finishTime'] = projector_log['startTime']
        self._write_projector_runlog(projector_log)
    

    def log_overPopulatedMapper_EVENT(self):
        model_log = self._open_model_runlog() 
        model_log['num_overPopulated_mappers'] = model_log['num_overPopulated_mappers'] + 1 
        self._write_model_runlog(model_log)
    
    def log_emptyComplex_EVENT(self): 
        model_log = self._open_model_runlog() 
        model_log['num_empty_mappers'] = model_log['num_empty_mappers'] + 1 
        self._write_model_runlog(model_log)
    
    def log_unkownError_EVENT(self): 
        model_log = self._open_model_runlog() 
        model_log['unknown_errors'] = model_log['unknown_errors'] + 1 
        self._write_model_runlog(model_log)