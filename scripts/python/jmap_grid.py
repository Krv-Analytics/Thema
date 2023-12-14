import logging
import os
import warnings

from omegaconf import OmegaConf
from python_log_indenter import IndentedLoggerAdapter
from utils import env

warnings.simplefilter("ignore")

################################################################################################
#  Handling Local Imports
################################################################################################

root, src = env()  # Load .env

from gridTracking_helper import log_error, subprocess_scheduler

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

    # HDBSCAN
    min_cluster_size = params["jmap_min_cluster_size"]
    max_cluster_size = params["jmap_max_cluster_size"]

    # MAPPER
    n_cubes = params["jmap_nCubes"]
    perc_overlap = params["jmap_percOverlap"]
    min_intersection = params["jmap_minIntersection"]
    random_seed = params["jmap_random_seed"]

    # DATA
    raw = params["raw_data"]
    clean = params["clean_data"]
    projections = params["projected_data"]

    jmap_generator = os.path.join(src, "jmapping/fitting/jmap_generator.py")

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

    ################################################################################################
    #   Logging
    ################################################################################################

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = IndentedLoggerAdapter(logging.getLogger(__name__))

    log.info("Computing jmap Grid Search!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Choices for min_cluster_size: {min_cluster_size}")
    log.info(f"Choices for n_cubes: {n_cubes}")
    log.info(f"Choices for perc_overlap: {perc_overlap}")
    log.info(f"Choices for min_intersection: {min_intersection}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()

    ################################################################################################
    #   Scheduling Subprocesses
    ################################################################################################

    # Number of loops
    num_loops = (
        len(n_cubes)
        * len(perc_overlap)
        * len(min_intersection)
        * len(min_cluster_size)
        * len(os.listdir(os.path.join(root, projections)))
    )

    # Creating list of Subprocesses
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for N in n_cubes:
        for P in perc_overlap:
            for I in min_intersection:
                for C in min_cluster_size:
                    for file in os.listdir(os.path.join(root, projections)):
                        if file.endswith(".pkl"):
                            D = os.path.join(projections, file)
                            subprocesses.append(
                                [
                                    "python",
                                    f"{jmap_generator}",
                                    f"-n{N}",
                                    f"-r{raw}",
                                    f"-c{clean}",
                                    f"-D{D}",
                                    f"-m{C}",
                                    f"-p {P}",
                                    f"-I {I}",
                                ]
                            )

    # Handles Process scheduling
    subprocess_scheduler(
        subprocesses,
        num_loops,
        "SUCCESS: Completed JMAP generation grid.",
        resilient=True,
    )
