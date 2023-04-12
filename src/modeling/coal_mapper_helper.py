import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from hdbscan import HDBSCAN

from coal_mapper import CoalMapper
from model import Model
from nammu.curvature import ollivier_ricci_curvature

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src)


def coal_mapper_generator(
    tupper,
    n_cubes,
    perc_overlap,
    hdbscan_params,
    min_intersection_vals,
    verbose=False,
):
    """ """

    # HDBSCAN
    min_cluster_size, max_cluster_size = hdbscan_params
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )
    # Configure CoalMapper
    coal_mapper = CoalMapper(tupper)
    coal_mapper.fit(n_cubes, perc_overlap, clusterer)

    results = {}
    if len(coal_mapper.complex["links"]) > 0:
        for val in min_intersection_vals:
            # Generate Graph
            try:
                coal_mapper.to_networkx(min_intersection=val)
                coal_mapper.connected_components()
                # Compute Curvature and Persistence Diagram
                coal_mapper.curvature = ollivier_ricci_curvature
                coal_mapper.calculate_homology()
                results[val] = coal_mapper
            except:
                if verbose:
                    print("Empty Mapper!")
        return results
    else:
        if verbose:
            print(
                "-------------------------------------------------------------------------------- \n\n"
            )
            print(f"Empty Simplicial Complex. No file written")

            print(
                "\n\n -------------------------------------------------------------------------------- "
            )
        return results


def generate_mapper_filename(args, n_neighbors, min_dist, min_intersection):
    """Generate output filename string from CLI arguments when running  script."""

    min_cluster_size, p, n = (
        args.min_cluster_size,
        args.perc_overlap,
        args.n_cubes,
    )
    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{min_cluster_size}_UMAP_{n_neighbors}Nbors_minD{min_dist}_min_int{min_intersection}.pkl"

    return output_file


def unpack_policy_group_dir(folder):
    n = int(folder[: folder.index("_")])
    return n


def get_viable_models(n: int, coverage_filter: float):
    dir = f"data/mappers/{n}_policy_groups/"
    dir = os.path.join(root, dir)
    files = os.listdir(dir)
    models = []
    for file in files:
        file = os.path.join(dir, file)
        model = Model(file)
        N = len(model.tupper.clean)
        num_unclustered_items = len(model.unclustered_items)
        if num_unclustered_items / N <= 1 - coverage_filter:
            models.append(file)
    print(
        f"{n}_policy_groups: {np.round(len(models)/len(files)*100,1)}% of models had at least {coverage_filter*100}% coverage."
    )
    return models


def plot_mapper_histogram(coverage_filter=0.8):
    mappers = os.path.join(root, "data/mappers/")
    policy_groups = os.listdir(mappers)  # get a list of folder names in the directory
    counts = {}  # initialize an empty list to store the number of files in each folder
    for folder in policy_groups:
        n = unpack_policy_group_dir(folder)
        models = get_viable_models(n, coverage_filter)
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
    return fig
