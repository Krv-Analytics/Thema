import os
import pickle
import sys

from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering

load_dotenv()
src = os.getenv("src")
root = os.getenv("root")
sys.path.append(src)

from summarizing.visualization_helper import plot_dendrogram


def cluster_models(
    distances,
    metric,
    num_policy_groups,
    p=3,
    distance_threshold=0.5,
    plot=True,
):

    """This function performs agglomerative clustering on the given
    pairwise distances using the average linkage method.
    These pairwise distances are between the persistence diagrams,
    gererated using curvature filtrations of the model graphs.

    Parameters:
    -----------
    distances: numpy array
        The pairwise distance matrix of the dataset to be clustered.

    metric: str
        The metric used to calculate pairwise distances.

    num_policy_groups: int
        The number of clusters (policy groups) to form.

    p: int, default=3
        The truncation level for visualization.

    distance_threshold: float, default=0.5
        The threshold below which, clusters will not be merged.

    plot: bool, default=True
        If True, the dendrogram of the clustering will be plotted.

    Returns:
    --------
    model: AgglomerativeClustering
        The agglomerative clustering model.
    """

    model = AgglomerativeClustering(
        affinity="precomputed",
        linkage="average",
        compute_distances=True,
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    model.fit(distances)

    # What percentage of Labels do you want to visualize?
    if plot:
        plot_dendrogram(
            model=model,
            labels=None,
            distance=metric,
            truncate_mode="level",
            p=p,
            n=num_policy_groups,
            distance_threshold=distance_threshold,
        )

    return model


def read_distance_matrices(dir, metric, n):
    """
    This function reads the pairwise distance matrices stored in the directory,
    given the metric and number of policy groups.

    Parameters:
    -----------
    metric: str
        The metric used to calculate pairwise distances.

    n: int
        The number of policy groups.

    Returns:
    --------
    keys: list
        The list of keys (model IDs) used to generate the pairwise distances.

    distances: numpy array
        The pairwise distance matrix of the dataset.
    """


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
