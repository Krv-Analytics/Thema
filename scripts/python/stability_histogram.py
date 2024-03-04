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

from gridTracking_helper import log_error
from jmap_selector_helper import get_viable_jmaps, unpack_policy_group_dir
from meta_utils import plot_stability_histogram

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
    run_dir = os.path.join(root, "data/" + params["Run_Name"])
    jmap_dir = os.path.join(root, "data/" + params["Run_Name"] + f"/jmaps/")

    # Counting Length of updated file
    coverage = params["coverage_filter"]
    distance_matrices = (
        "data/"
        + params["Run_Name"]
        + f"/jmap_analysis/distance_matrices/{coverage}_coverage/"
    )
    distance_matrices = os.path.join(root, distance_matrices)

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
    if not os.path.isdir(distance_matrices) or not os.listdir(distance_matrices):
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

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = IndentedLoggerAdapter(logging.getLogger(__name__))

    # LOGGING
    log.info("Computing Two Layer Histogram!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Policy Groups in consideration: {group_ranks}")
    log.info(f"jmap Coverage Filter: {coverage}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()
    # Counting Length of updated file

    histogram = plot_stability_histogram(dir=run_dir, coverage=coverage)
