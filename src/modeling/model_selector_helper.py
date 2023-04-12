import os
import pickle
import sys
import numpy as np

from dotenv import load_dotenv
from model import Model

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src)


def read_graph_clustering(metric, n):
    dir = os.path.join(root, f"data/model_analysis/graph_clustering/{n}_policy_groups/")
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


def get_model_file(key, n):
    n_cubes, p, n_neighbors, min_dist, hdbscan_params, min_intersection = key
    dir = os.path.join(root, f"data/mappers/{n}_policy_groups/")
    file = f"mapper_ncubes{n_cubes}_{int(p*100)}perc_hdbscan{hdbscan_params[0]}_UMAP_{n_neighbors}Nbors_minD{min_dist}_min_int{min_intersection}.pkl"
    file = os.path.join(dir, file)
    assert os.path.isfile(file), f"No saved model with these keys: {key}"
    return file


def get_clustering_subgroups(keys, clustering, n):
    labels = clustering.labels_
    subgroups = {}
    for label in labels:
        mask = np.where(labels == label, True, False)
        subkeys = keys[mask]
        files = []
        for key in subkeys:
            files.append(get_model_file(key, n))
        subgroups[label] = files

    return subgroups


def get_best_covered_model(models):
    coverages = []
    for file in models:
        model = Model(file)
        coverages.append(len(model.unclustered_items))
    assert len(coverages) == len(models)
    return models[np.argmin(coverages)]


def select_models(keys, clustering, n):
    subgroups = get_clustering_subgroups(keys, clustering, n)
    selection = []
    for subgroup in subgroups.values():
        best_model = get_best_covered_model(subgroup)
        print(f"Selected {best_model}")
        selection.append(best_model)
    return selection
