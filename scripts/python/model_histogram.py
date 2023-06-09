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
sys.path.append(src + "modeling/")

from modeling.model_selector_helper import unpack_policy_group_dir, get_viable_models


def plot_mapper_histogram(dir, coverage_filter=0.8):
    """
    Plots a histogram of the number of viable models for each rank
    of policy groupings. This function will count the models
    (per `num_policy_groups`) that have been generated
    according to a hyperparameter grid search.

    Parameters:
    -----------
    coverage_filter: float
        The minimum coverage percentage required for a model
        to be considered viable.

    Returns:
    --------
    fig: matplotlib.figure.Figure
        The plotted figure object.
    """
    mappers = os.path.join(root, dir)
    # Get list of folder names in the directory
    policy_groups = os.listdir(mappers)
    # Initialize counting dictionary
    counts = {}
    for folder in policy_groups:
        n = unpack_policy_group_dir(folder)
        path_to_models = dir + folder
        models = get_viable_models(path_to_models, n, coverage_filter)
        counts[n] = len(models)
    keys = list(counts.keys())
    keys.sort()
    sorted_counts = {i: counts[i] for i in keys}
    # plot the histogram
    fig = plt.figure(figsize=(15, 10))
    ax = sns.barplot(
        x=list(sorted_counts.keys()),
        y=list(sorted_counts.values()),
    )
    ax.set(xlabel="Number of Policy Groups", ylabel="Number of Viable Models")
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

    dir = "data/" + params_json["Run_Name"] + f"/models/"
    dir = os.path.join(root, dir)
    group_ranks = []
    for folder in os.listdir(dir):
        i = unpack_policy_group_dir(folder)
        group_ranks.append(i)

    # Metric Generator Configuratiosn
    coverage = params_json["coverage_filter"]

    # LOGGING
    log.info("Computing Model Histogram!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Policy Groups in consideration: {group_ranks}")
    log.info(f"Model Coverage Filter: {coverage}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()

    fig = plot_mapper_histogram(dir=dir, coverage_filter=coverage)
    plt.show()
