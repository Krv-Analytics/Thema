import logging
import os
import warnings

from omegaconf import OmegaConf
from python_log_indenter import IndentedLoggerAdapter
from modeling.generation.utils import env

warnings.simplefilter("ignore")

################################################################################################
#  Handling Local Imports
################################################################################################

root, src = env()  # Load .env

from gridTracking_helper import log_error, subprocess_scheduler
from jmap_selector_helper import unpack_policy_group_dir

################################################################################################
#   Loading Config Data
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
    projections = params["projected_data"]
    jmap_dir = os.path.join(root, "data/" + params["Run_Name"] + f"/jmaps/")
    curvature_distances = os.path.join(
        root,
        "data/"
        + params["Run_Name"]
        + f"/jmap_analysis/distance_matrices/"
        + str(params["coverage_filter"])
        + "_coverage",
    )

    # Metric Generator Configuratiosn
    jmap_selector = os.path.join(src, "jmapping/selecting/jmap_selector.py")
    coverage = params["coverage_filter"]

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

    # Check that Projections Exist
    if not os.path.isdir(os.path.join(root, projections)) or not os.listdir(
        os.path.join(root, projections)
    ):
        log_error(
            "No projections found. Please make sure you have generated projections using `make projections`."
        )

    # Check that JMAPS Exist
    if not os.path.isdir(jmap_dir) or not os.listdir(jmap_dir):
        log_error(
            "No JMAPS found. Please make sure you have generated jmaps using `make jmaps`. Otherwise, the hyperparameters in your params folder may not be generating any jmaps."
        )

    # Check that Curvature distances Exist
    if not os.path.isdir(curvature_distances) or not os.listdir(curvature_distances):
        log_error(
            "No Curvature distances found. Please make sure you have generated enough jmaps to warrant curvature analysis. "
        )

    ################################################################################################
    #   Logging
    ################################################################################################

    group_ranks = []
    for folder in os.listdir(jmap_dir):
        i = unpack_policy_group_dir(folder)
        group_ranks.append(i)

    group_ranks.sort()
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = IndentedLoggerAdapter(logging.getLogger(__name__))
    # LOGGING
    log.info("Selecting jmaps!")
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
                f"{jmap_selector}",
                f"-n{i}",
                f"-f{coverage}",
            ]
        )

    subprocess_scheduler(
        subprocesses=subprocesses,
        num_processes=num_loops,
        resilient=False,
        success_message="SUCCESS: Completed JMAP selection process",
    )
