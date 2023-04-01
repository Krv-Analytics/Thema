import pickle
import os
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, labels, distance, **kwargs):
    """Create linkage matrix and then plot the dendrogram for Hierarchical clustering."""

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    d = dendrogram(linkage_matrix, labels=labels, **kwargs)
    plt.title("Hyperparameter Dendrogram")
    plt.xlabel("Coordinates: (n_cubes,perc_overlap,min_intersection).")
    plt.ylabel(f"{distance} distance between persistence diagrams")
    plt.show()
    return d


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
