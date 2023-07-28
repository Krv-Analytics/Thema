import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter


load_dotenv()
root = os.getenv("root")
sys.path.append(root + "logging/")
from gridTracking_helper import subprocess_scheduler


if __name__ == "__main__":

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = IndentedLoggerAdapter(logging.getLogger(__name__))
    load_dotenv()
    root = os.getenv("root")
    src = os.getenv("src")
    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")


    # HDBSCAN
    min_cluster_size = params_json["jmap_min_cluster_size"]
    max_cluster_size = params_json["jmap_max_cluster_size"]

    # MAPPER
    n_cubes = params_json["jmap_nCubes"]
    perc_overlap = params_json["jmap_percOverlap"]
    min_intersection = params_json["jmap_minIntersection"]
    random_seed = params_json["jmap_random_seed"]

    # DATA
    raw = params_json["raw_data"]
    clean = params_json["clean_data"]
    projections = params_json["projected_data"]

    jmap_generator = os.path.join(src, "jmapping/fitting/jmap_generator.py")
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

    # Number of loops 
    num_loops = len(n_cubes)*len(perc_overlap)*len(min_intersection)*len(min_cluster_size) *len(os.listdir(os.path.join(root, projections)))
    
    # Running Grid in Parallel 
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
                                    f"-I {I}"
                                ]
                            )

    
    # Handles Process scheduling 
    subprocess_scheduler(subprocesses, num_loops, "SUCCESS: Completed JMAP generation grid.", resilient=True)