import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


from sklearn.cluster import AgglomerativeClustering

cwd = os.path.dirname(__file__)


def cluster_hyperparams(keys, distances, metric, p_frac=0.1):
    model = AgglomerativeClustering(
        affinity="precomputed", linkage="average", compute_distances=True
    )
    model.fit(distances)

    keys = [key[:2] for key in keys]

    # What percentage of Labels do you want to visualize?
    p = int(p_frac * len(keys))
    plot_dendrogram(model, keys, metric, truncate_mode="level", p=p)

    return model


def read_distance_matrices(metric):
    dir = os.path.join(cwd, "./../../../data/parameter_modeling/distance_matrices/")
    file = os.path.join(dir, f"curvature_{metric}_pairwise_distances.pkl")
    with open(file, "rb") as f:
        d = pickle.load(f)

    keys = d["keys"]
    distances = d["distances"]
    return keys, distances
