import logging
import os
import warnings

from omegaconf import OmegaConf
from python_log_indenter import IndentedLoggerAdapter
from utils import env, get_imputed_files

warnings.simplefilter("ignore")

################################################################################################
#  Handling Local Imports
################################################################################################

root, src = env()  # Load .env

from gridTracking_helper import log_error, subprocess_scheduler

################################################################################################
#   Loading and writing Config Data
################################################################################################


if __name__ == "__main__":
    YAML_PATH = os.getenv("params")
    if os.path.isfile(YAML_PATH):
        with open(YAML_PATH, "r") as f:
            params = OmegaConf.load(f)
    else:
        print("params.yaml file note found!")

    # DATA
    raw = params["raw_data"]
    clean = params["clean_data"]

    # UMAP Parameters
    N_neighbors = params["projector_Nneighbors"]
    min_Dists = params["projector_minDists"]
    projector = params["projector"]

    # Projections Script
    projector_script = os.path.join(src, "processing/projecting/projector.py")

    # Updating Projections Path in params.yaml
    params["projected_data"] = (
        "data/" + params["Run_Name"] + "/projections/" + params["projector"] + "/"
    )
    try:
        with open(YAML_PATH, "w") as f:
            OmegaConf.save(params, f)
    except:
        log_error(
            "Unable to write to you parameter file. Please make sure you have set proper file permissions."
        )

    ################################################################################################
    #   Checking for necessary files
    ################################################################################################

    # Check that raw data exists
    if not os.path.isfile(os.path.join(root, raw)):
        log_error(
            "No raw data found. Please make sure you have specified the correct path in your params file."
        )

    # Check that clean data exits
    if not os.path.isfile(os.path.join(root, clean)):
        log_error(
            "No clean data found. Please make sure you generated clean data using `make process-data`."
        )

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

    file_path = os.path.join(root, os.path.dirname(params.clean_data))

    # TODO: add logging information pertaining to data handling when NAs are present
    imputation_files = get_imputed_files(file_path, key=params.data_imputation.method)

    # Number of loops
    num_loops = len(N_neighbors) * len(min_Dists) * len(imputation_files)

    # Creating list of Subprocesses
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for n in N_neighbors:
        for d in min_Dists:
            for f in imputation_files:
                cmd = [
                    "python",
                    f"{projector_script}",
                    f"--clean_data=" + f,
                    f"-n {n}",
                    f"-d {d}",
                    f"--projector={projector}",
                ]
                subprocesses.append(cmd)

    # Handles Process scheduling
    subprocess_scheduler(
        subprocesses, num_loops, "SUCCESS: Completed projections grid."
    )
