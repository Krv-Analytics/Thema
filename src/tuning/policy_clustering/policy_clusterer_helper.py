import os
import pickle
import sys

from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering

load_dotenv()
src = os.getenv("src")
root = os.getenv("root")
sys.path.append(src)

from visualizing.visualization_helper import plot_dendrogram


def cluster_policy_reccomendations(keys, distances, metric, num_policy_groups, p=3):

    model = AgglomerativeClustering(
        affinity="precomputed", linkage="average", compute_distances=True
    )
    model.fit(distances)

    # keys = [key[:2] for key in keys]
    id_labels = [i for i in range(len(keys))]
    # What percentage of Labels do you want to visualize?
    plot_dendrogram(
        model=model,
        labels=None,
        distance=metric,
        truncate_mode="level",
        p=p,
        n=num_policy_groups,
    )

    return model


def read_distance_matrices(metric, n):
    dir = os.path.join(
        root, f"data/model_analysis/distance_matrices/{n}_policy_groups/"
    )
    assert os.path.isdir(
        dir
    ), f"No pairwise distances yet for {n} policy groups! Please run `metric_generator.py -n {n}` first."
    file = os.path.join(dir, f"curvature_{metric}_pairwise_distances.pkl")

    assert os.path.isfile(file)
    with open(file, "rb") as f:
        d = pickle.load(f)

    keys = d["keys"]
    distances = d["distances"]
    return keys, distances
