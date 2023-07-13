import json
import logging
import os
import pickle
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
sys.path.append(src + "jmapping/")

from jmapping.jmap_selector_helper import unpack_policy_group_dir


def plot_curvature_histogram(dir):
    num_curvature_profiles = {}
    for folder in os.listdir(dir):
        i = unpack_policy_group_dir(folder)
        folder = os.path.join(dir, folder)
        if os.path.isdir(folder):
            holder = []
            for file in os.listdir(folder):
                if file.endswith(".pkl"):
                    file = os.path.join(folder, file)
                    with open(file, "rb") as f:
                        matrix = pickle.load(f)["distances"]
                holder.append(int(len(matrix)))
            # For now take the maximum number of unique curvature profiles over different metrics
            num_curvature_profiles[i] = max(holder)

    fig = plt.figure(figsize=(15, 10))
    ax = sns.barplot(
        x=list(num_curvature_profiles.keys()),
        y=list(num_curvature_profiles.values()),
    )
    ax.set(xlabel="Number of Policy Groups", ylabel="Number of Curvature Profiles")
    ax.set_title(f"{coverage*100} % Coverage Filter")
    plt.show()
    return fig


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
    dir = "data/" + params_json["Run_Name"] + f"/jmaps/"
    dir = os.path.join(root, dir)
    group_ranks = []
    for folder in os.listdir(dir):
        i = unpack_policy_group_dir(folder)
        group_ranks.append(i)

    # Metric Generator Configuratiosn
    metric_generator = os.path.join(src, "tuning/metrics/metric_generator.py")
    coverage = params_json["coverage_filter"]

    # LOGGING
    log.info("Computing Curvature Histogram!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Policy Groups in consideration: {group_ranks}")
    log.info(f"jmap Coverage Filter: {coverage}")
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
                f"{metric_generator}",
                f"-n{i}",
                f"-f{coverage}",
            ]
        )

    # Running processes in Parallel
    # TODO: optimize based on max_workers
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(subprocess.run, cmd) for cmd in subprocesses]
        # Setting Progress bar to track number of completed subprocesses
        progress_bar = tqdm(total=num_loops, desc="Progress", unit="subprocess")
        for future in as_completed(futures):
            # Update the progress bar for each completed subprocess
            progress_bar.update(1)
        progress_bar.close()
