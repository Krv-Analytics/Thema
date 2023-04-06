import os
import pickle
import numpy as np
import sys
from dotenv import load_dotenv

from gtda.diagrams import PairwiseDistance


load_dotenv()
src = os.getenv("src")
root = os.getenv("root")
sys.path.append(src)

modeling = os.path.join(src, "modeling/")
sys.path.append(modeling)


from modeling.coal_mapper import CoalMapper


def topology_metric(
    files,
    metric="bottleneck",
):
    keys, diagrams = get_diagrams(files)
    curvature_dgms = convert_to_gtda(diagrams.values())
    distance_metric = PairwiseDistance(metric=metric)
    distance_metric.fit(curvature_dgms)
    distances = distance_metric.transform(curvature_dgms)

    return keys, distances


def mapper_reader(dir):
    assert os.path.isdir(
        dir
    ), "Please first compute mapper objects using `coal_mapper_generator.py`"

    # TODO: add a filter here for `unlcustered` plants
    data = {}
    cwd = os.path.dirname(__file__)
    dir = os.path.join(cwd, dir)
    for file in os.listdir(dir):
        if file.endswith(".pkl"):
            mapper_file = os.path.join(dir, file)
            with open(mapper_file, "rb") as f:
                result_dictionary = pickle.load(f)
            hyper_params = result_dictionary["hyperparameters"]
            result_dictionary.pop("hyperparameters")
            data[hyper_params] = result_dictionary

    return data


def get_diagrams(dir):
    data = mapper_reader(dir)
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
