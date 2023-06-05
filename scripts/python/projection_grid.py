import json
import logging
import os
import subprocess

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter

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

    N_neighbors = params_json["projector_Nneighbors"]
    min_Dists = params_json["projector_minDists"]

    projector = os.path.join(src, "processing/projecting/projector.py")
    log.info("Computing UMAP Projection Grid Search!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Choices for n_neighbors: {N_neighbors}")
    log.info(f"Choices for min_dist: {min_Dists}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()
    ## GRID SEARCH PROJECTIONS
    for n in N_neighbors:
        for d in min_Dists:
            subprocess.run(["python", f"{projector}", f"-n {n}", f"-d {d}"])
