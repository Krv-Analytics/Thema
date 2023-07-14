import json
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter

load_dotenv()
src = os.getenv("src")
sys.path.append(src)
sys.path.append(src + "jmapping/fitting")

from fitting.jmap_selector_helper import unpack_policy_group_dir, get_viable_jmaps
from curvature_histogram import plot_curvature_histogram


def plot_histogram(root, coverage_filter):

    dir = "data/" + params_json["Run_Name"] + "/"
    dir = os.path.join(root, dir)
    jmap_dir = os.path.join(dir, "jmaps/")
    metric_files = os.path.join(
        root,
        "data/"
        + params_json["Run_Name"]
        + f"/jmap_analysis/distance_matrices/{coverage}_coverage/",
    )

    num_jmaps = {}
    for folder in os.listdir(jmap_dir):
        i = unpack_policy_group_dir(folder)
        folder = os.path.join(jmap_dir, folder)
        jmaps = get_viable_jmaps(folder, i, coverage_filter=coverage_filter)
        num_jmaps[i] = len(jmaps)

    num_curvature_profiles = {}
    assert os.path.isdir(metric_files), "Not a valid directory"
    for folder in os.listdir(metric_files):
        i = unpack_policy_group_dir(folder)
        folder = os.path.join(metric_files, folder)
        print(folder)
        if os.path.isdir(folder):
            holder = []
            for file in os.listdir(folder):
                if file.endswith(".pkl"):
                    file = os.path.join(folder, file)
                    with open(file, "rb") as f:
                        matrix = pickle.load(f)["distances"]
                    print(len(matrix))
                holder.append(int(len(matrix)))
            # For now take the maximum number of unique curvature profiles over different metrics
            num_curvature_profiles[i] = max(holder)
    stability_ratio = {}
    for key in num_curvature_profiles.keys():
        stability_ratio[key] = num_jmaps[key] / num_curvature_profiles[key]
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(15, 20))
    fig.suptitle(f"{coverage*100}% Coverage Filter")
    sns.barplot(
        x=list(stability_ratio.keys()),
        y=list(stability_ratio.values()),
        ax=ax,
    )
    ax.set(xlabel="Number of Policy Groups", ylabel="Stability Ratio")

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

    coverage = params_json["coverage_filter"]
    histogram = plot_histogram(root=root, coverage_filter=coverage)
