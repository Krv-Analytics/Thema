import os
import sys
import json
import logging
import warnings

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter

warnings.simplefilter("ignore")

################################################################################################
#  Handling Local Imports  
################################################################################################

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src + "jmapping/selecting/")
sys.path.append(root + "logging/")

from jmap_selector_helper import unpack_policy_group_dir
from gridTracking_helper import (
    subprocess_scheduler,
    log_error
)

################################################################################################
#   Loading JSON Data  
################################################################################################

if __name__ == "__main__":

    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")

    dir = "data/" + params_json["Run_Name"] + f"/jmaps/"
    dir = os.path.join(root, dir)
    
    # DATA 
    raw = params_json["raw_data"]
    clean = params_json["clean_data"]
    projections = params_json["projected_data"]
    jmap_dir = os.path.join(root, "data/" + params_json["Run_Name"] + f"/jmaps/")
    curvature_distances = os.path.join(root, "data/" + params_json["Run_Name"] + f"/jmap-analysis/distance_matrices/" + str(params_json["coverage_filter"]) + "_coverage/")

    # Metric Generator Configuratiosn
    jmap_clusterer = os.path.join(src, "tuning/graph_clustering/jmap_clusterer.py")
    coverage = params_json["coverage_filter"]


################################################################################################
#   Checking for necessary files 
################################################################################################
     
     # Check that raw data exists 
    if not os.path.isfile(os.path.join(root, raw)): 
        log_error("No raw data found. Please make sure you have specified the correct path in your params file.") 


    # Check that clean data exits 
    if not os.path.isfile(os.path.join(root, clean)):
        log_error("No clean data found. Please make sure you generated clean data using `make process-data`.") 
   

    # Check that Projections Exist
    if not os.path.isdir(os.path.join(root, projections)) or not os.listdir(os.path.join(root, projections)):
        log_error("No projections found. Please make sure you have generated projections using `make projections`.")
    
    # Check that JMAPS Exist
    if not os.path.isdir(jmap_dir) or not os.listdir(jmap_dir): 
        log_error("No JMAPS found. Please make sure you have generated jmaps using `make jmaps`. Otherwise, the hyperparameters in your params folder may not be generating any jmaps.")
    
    # Check that Curvature distances Exist
    if not os.path.isdir(curvature_distances) or not os.listdir(curvature_distances): 
        log_error("No Curvature distances found. Please make sure you have generated enough jmaps to warrant curvature analysis. ")
    


################################################################################################
#   Logging 
################################################################################################
    
    group_ranks = []
    for folder in os.listdir(dir):
        i = unpack_policy_group_dir(folder)
        group_ranks.append(i)
    
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = IndentedLoggerAdapter(logging.getLogger(__name__))
    # LOGGING
    log.info("Clustering Graph jmaps!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Policy Groups in consideration: {group_ranks}")
    log.info(f"jmap Coverage Filter: {coverage}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()

################################################################################################
#   Scheduling Subprocesses 
################################################################################################
  
    # Number of loops
    num_loops = len(group_ranks)
    # Running Grid in Parallel
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for i in group_ranks:
        subprocesses.append(
            [
                "python",
                f"{jmap_clusterer}",
                f"-n{i}",
                "-s",
            ]
        )

    # Handling Parallelism
    subprocess_scheduler(subprocesses, num_loops, "SUCESS: Completed JMAP Clustering.", resilient=True)

