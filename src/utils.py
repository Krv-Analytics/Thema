"Utility Functions"
import os
import pickle
import numpy as np

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


from mapper import MapperTopology
from nammu.curvature import ollivier_ricci_curvature


def curvature_iterator(
    X,
    n_cubes,
    perc_overlap,
    K,
    min_intersection_vals,
    random_state=None,
):

    # Configure CoalMapper
    mapper = MapperTopology(X=X)
    clusterer = KMeans(n_clusters=K, random_state=random_state, n_init=1)
    cover = (n_cubes, perc_overlap)
    # Generate Graphs
    results = {}

    print("Computing Curvature Values and Persistence Diagrams")
    for val in min_intersection_vals:
        mapper.set_graph(cover=cover, clusterer=clusterer, min_intersection=val)
        mapper.calculate_homology(filter_fn=ollivier_ricci_curvature, use_min=True)
        results[val] = mapper
    return results


def convert_to_gtda(diagrams):
    """Pad a set of persistence diagrams so they are compatible with Giotto-TDA."""
    diagrams = [
        [
            np.asarray(diagram[0]._pairs),
            np.asarray(diagram[1]._pairs),
        ]
        for diagram in diagrams
    ]
    homology_dimensions = (0, 1)

    slices = {
        dim: slice(None) if (dim) else slice(None, -1) for dim in homology_dimensions
    }
    Xt = [
        {dim: diagram[dim][slices[dim]] for dim in homology_dimensions}
        for diagram in diagrams
    ]
    start_idx_per_dim = np.cumsum(
        [0]
        + [
            np.max([len(diagram[dim]) for diagram in Xt] + [1])
            for dim in homology_dimensions
        ]
    )
    min_values = [
        min(
            [
                np.min(diagram[dim][:, 0]) if diagram[dim].size else np.inf
                for diagram in Xt
            ]
        )
        for dim in homology_dimensions
    ]
    min_values = [min_value if min_value != np.inf else 0 for min_value in min_values]
    n_features = start_idx_per_dim[-1]
    Xt_padded = np.empty((len(Xt), n_features, 3), dtype=float)

    for i, dim in enumerate(homology_dimensions):
        start_idx, end_idx = start_idx_per_dim[i : i + 2]
        padding_value = min_values[i]
        # Add dimension as the third elements of each (b, d) tuple globally
        Xt_padded[:, start_idx:end_idx, 2] = dim
        for j, diagram in enumerate(Xt):
            subdiagram = diagram[dim]
            end_idx_nontrivial = start_idx + len(subdiagram)
            # Populate nontrivial part of the subdiagram
            if len(subdiagram) > 0:
                Xt_padded[j, start_idx:end_idx_nontrivial, :2] = subdiagram
            # Insert padding triples
            Xt_padded[j, end_idx_nontrivial:end_idx, :2] = [padding_value] * 2

    return Xt_padded


def generate_results_filename(args, suffix=".pkl"):
    """Generate output filename string from CLI arguments when running compute_curvature script."""

    K, p, n = args.KMeans, args.perc_overlap, args.n_cubes

    output_file = f"results_ncubes{n}_{int(p*100)}perc_K{K}.pkl"

    return output_file


def read_curvature_results():
    src_dir = os.path.dirname(__file__)
    curvature_dir = src_dir + "/../outputs/curvature/"
    assert os.path.isdir(
        curvature_dir
    ), "Please first compute curvature results using `compute_curvature.py`"

    data = {}

    for file in os.listdir(curvature_dir):
        if file.endswith(".pkl"):
            with open(curvature_dir + file, "rb") as f:
                result_dictionary = pickle.load(f)
                hyper_params = result_dictionary["hyperparameters"]
                result_dictionary.pop("hyperparameters")
                data[hyper_params] = result_dictionary

    return data


def get_diagrams():
    data = read_curvature_results()
    diagrams = {}
    for hyper_params in data.keys():
        mappers = data[hyper_params]
        for min_intersection in mappers.keys():
            diagram = mappers[min_intersection].diagram
            new_key = (*hyper_params, min_intersection)
            diagrams[new_key] = diagram
    keys = list(diagrams.keys())
    keys.sort()
    sorted_diagrams = {i: diagrams[i] for i in keys}

    return keys, sorted_diagrams


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
    dendrogram(linkage_matrix, labels=labels, **kwargs)
    plt.title("Hyperparameter Dendrogram")
    plt.xlabel("Coordinates: (n_cubes,perc_overlap,K,min_intersection).")
    plt.ylabel(f"{distance} distance between persistence diagrams")
    plt.show()
