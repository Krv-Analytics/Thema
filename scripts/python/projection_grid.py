import os
import sys
import json
import logging

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter


################################################################################################
#  Handling Local Imports  
################################################################################################

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(root + "logging/")

from gridTracking_helper import (
    subprocess_scheduler, 
    log_error
)

################################################################################################
#   Loading and writing JSON Data  
################################################################################################


if __name__ == "__main__":

    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")

    # DATA 
    raw = params_json["raw_data"]
    clean = params_json["clean_data"]

    # UMAP Parameters
    N_neighbors = params_json["projector_Nneighbors"]
    min_Dists = params_json["projector_minDists"]
    projector = params_json["projector"]

    # Projections Script
    projector_script = os.path.join(src, "processing/projecting/projector.py")

    # Updating Projections Path in params.json
    params_json["projected_data"] = "data/" + params_json["Run_Name"] + "/projections/" + params_json["projector"] + "/"
    try:
        with open(JSON_PATH, "w") as f:
            json.dump(params_json, f, indent=4)
    except:
        log_error("Unable to write to you parameter file. Please make sure you have set proper file permissions.")


################################################################################################
#   Checking for necessary files 
################################################################################################
    
    # Check that raw data exists 
    if not os.path.isfile(os.path.join(root, raw)): 
        log_error("No raw data found. Please make sure you have specified the correct path in your params file.") 


    # Check that clean data exits 
    if not os.path.isfile(os.path.join(root, clean)):
        log_error("No clean data found. Please make sure you generated clean data using `make process-data`.") 
   
   
################################################################################################
#   Logging 
################################################################################################

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = IndentedLoggerAdapter(logging.getLogger(__name__))

    log.info("Computing Projection Grid Search!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Choice of Projector: {projector}")
    log.info(f"Choices for n_neighbors: {N_neighbors}")
    log.info(f"Choices for min_dist: {min_Dists}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()

################################################################################################
#   Scheduling Subprocesses 
################################################################################################

    # Number of loops 
    num_loops = len(N_neighbors)*len(min_Dists)

    # Creating list of Subprocesses 
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for n in N_neighbors:
        for d in min_Dists:
            cmd = ["python", f"{projector_script}", f"-n {n}", f"-d {d}", f"--projector={projector}"]
            subprocesses.append(cmd)

    # Handles Process scheduling 
    subprocess_scheduler(subprocesses, num_loops, "SUCCESS: Completed projections grid.")
    