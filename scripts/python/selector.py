import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter
from tqdm import tqdm

load_dotenv()
src = os.getenv("src")
sys.path.append(src)
sys.path.append(src + "modeling/")

from modeling.model_selector_helper import unpack_policy_group_dir

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

    dir = "data/" + params_json["Run_Name"] + f"/models/"
    dir = os.path.join(root, dir)
    group_ranks = []
    for folder in os.listdir(dir):
        i = unpack_policy_group_dir(folder)
        group_ranks.append(i)

    # Metric Generator Configuratiosn
    model_selector = os.path.join(src, "modeling/model_selector.py")
    model_clusterer = os.path.join(src, "tuning/graph_clustering/model_clusterer.py")
    coverage = params_json["coverage_filter"]

    # LOGGING
    log.info("Selecting Models!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Policy Groups in consideration: {group_ranks}")
    log.info(f"Model Coverage Filter: {coverage}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()

    # Number of loops
    num_loops = len(group_ranks)
    # Running Grid in Parallel
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for i in group_ranks:
        subprocesses.append(
            [
                "python",
                f"{model_selector}",
                f"-n{i}",
                f"-f{coverage}",
            ]
        )

    # Running processes in Parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(subprocess.run, cmd) for cmd in subprocesses]
        # Setting Progress bar to track number of completed subprocesses
        progress_bar = tqdm(total=num_loops, desc="Progress", unit="subprocess")
        for future in as_completed(futures):
            # Update the progress bar for each completed subprocess
            progress_bar.update(1)
        progress_bar.close()
