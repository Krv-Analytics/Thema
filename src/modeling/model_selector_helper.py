import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jmapper import JMapper
from model_helper import env

# Configure paths
root = env()


def select_models(dir, keys, clustering, n):
    """
    Choose the best covered model from a set of equivalency classes.
    These equivalency classes are based on an Agglomerative clustering
    of graphs (models) using a curvature filtration metric.

    This function will return a single model for each equivalency class.

    Parameters:
    -----------
    keys: np.ndarray
        An array of keys used to fit the clustering model.

    clustering: sklearn.cluster.DBSCAN
        A clustering model object.

    n: int
        The number of policy groups.

    Returns:
    --------
    selection: list
        A list of saved model file locations.
    """
    subgroups = get_clustering_subgroups(dir, keys, clustering, n)
    selection = {}
    for key in subgroups.keys():
        subgroup = subgroups[key]
        best_model = get_best_covered_model(subgroup)
        selection[key] = {"model": best_model, "cluster_size": len(subgroup)}
    return selection


def read_graph_clustering(dir, metric, n):
    """
    Reads in a pre-generated agglomerative clustering model of graphs
    and returns the relevant data.

    Parameters:
    -----------
    metric: str
        The metric used in the clustering model.

    n: int
        The number of policy groups in the graph clustering model.

    Returns:
    --------
    keys: np.ndarray
        An array of identifiers for the graphs.

    clustering: sklearn.cluster.AgglomerativeClustering
        Hierarchical clustering model fitted to graphs.

    distance_threshold: float
        The distance threshold used in the clustering model to
        determine labels (i.e. equivalency classes).
    """
    # dir = os.path.join(root, f"data/model_analysis/graph_clustering/{n}_policy_groups/")
    assert os.path.isdir(
        dir
    ), f"No model clustering model yet for {n} policy groups! Please run `model_clusterer.py -n {n}` first."
    file = os.path.join(dir, f"curvature_{metric}_clustering_model.pkl")

    assert os.path.isfile(file)
    with open(file, "rb") as f:
        reference = pickle.load(f)

    return (
        np.array(reference["keys"]),
        reference["model"],
        reference["distance_threshold"],
    )


def get_model_file(dir, key, n):
    """
    Returns the file location of a saved model with the given keys.

    Parameters:
    -----------
    key: tuple
        A tuple of keys that identify a specific model.

    n: int
        The number of policy groups.

    Returns:
    --------
    file: str
        The file location of the saved model.
    """
    # Unpack key
    n_cubes, p, n_neighbors, min_dist, hdbscan_params, min_intersection = key
    # dir = os.path.join(root, f"data/models/{n}_policy_groups/")
    file = f"mapper_ncubes{n_cubes}_{int(p*100)}perc_hdbscan{hdbscan_params[0]}_UMAP_{n_neighbors}Nbors_minDist{min_dist}_min_int{min_intersection}.pkl"
    file = os.path.join(dir, file)
    assert os.path.isfile(file), f"No saved model with these keys: {key}"
    return file


def get_clustering_subgroups(dir, keys, clustering, n):
    """
    Returns a dictionary of subgroups for a graph clustering along
    with the corresponding model identifiers (keys).

    Parameters:
    -----------
    keys: np.ndarray
        An array of keys used to fit the clustering model.

    clustering: sklearn.cluster.AgglomerativeClustering
        Hierarchical clustering model fitted to graphs.

    n: int
        The number of policy groups.

    Returns:
    --------
    subgroups: dict
        A dictionary of subgroups: keys are subgroup labels
        and values are a list of models in each subgroup.
    """
    labels = clustering.labels_
    subgroups = {}
    for label in labels:
        mask = np.where(labels == label, True, False)
        subkeys = keys[mask]
        files = []
        for key in subkeys:
            files.append(get_model_file(dir, key, n))
        subgroups[label] = files

    return subgroups


def get_best_covered_model(models):
    """
    Returns the saved model with the highest coverage percentage.

    Parameters:
    -----------
    models: list
        A list of saved model file locations.

    Returns:
    --------
    best_model: str
        The file location of the saved model with the highest coverage percentage.
    """

    coverages = []
    for file in models:
        with open(file, 'rb') as f:
            reference = pickle.load(f)
        jmapper = reference['jmapper']
        coverages.append(len(jmapper.get_unclustered_items()))
    assert len(coverages) == len(models)
    best_model = models[np.argmin(coverages)]
    return best_model


def get_viable_models(dir, n: int, coverage_filter: float):
    """
    Returns a list of saved models that have at least the specified coverage percentage.

    Parameters:
    -----------
    n: int
        The number of policy groups.

    coverage_filter: float
        The minimum coverage percentage required for a model to be considered viable.

    Returns:
    --------
    models: list
        A list of saved model file locations that meet the specified coverage percentage.
    """

    dir = os.path.join(root, dir)
    files = os.listdir(dir)
    models = []
    for file in files:
        file = os.path.join(dir, file)
        with open(file, 'rb') as f:
            reference = pickle.load(f)
        jmapper = reference['jmapper']
        N = len(jmapper.tupper.clean)
        num_unclustered_items = len(jmapper.get_unclustered_items())
        if num_unclustered_items / N <= 1 - coverage_filter:
            models.append(file)
    print(
        f"{n}_policy_groups: {np.round(len(models)/len(files)*100,1)}% of models had at least {coverage_filter*100}% coverage."
    )
    return models


def unpack_policy_group_dir(folder):
    """
    Extracts the number of policy groups from the folder name.

    Parameters:
    -----------
    folder: str
        The name of a folder containing models for a certain number of policy groups.

    Returns:
    --------
    n: int
        The number of policy groups corresponding to the given folder name.
    """

    n = int(folder[: folder.index("_")])
    return n
