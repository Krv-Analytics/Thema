import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gtda.diagrams import PairwiseDistance

load_dotenv()
src = os.getenv("src")
root = os.getenv("root")
sys.path.append(src)

modeling = os.path.join(src, "modeling/")
sys.path.append(modeling)


from modeling.model import Model


def topology_metric(
    files,
    metric="bottleneck",
    coverage=0.6,
):
    """
    Compute the pairwise distance matrix between topological persistence diagrams using Giotto-TDA.

    Parameters:
    -----------
    files : str
        The directory containing the graph models to use.
    metric : str, optional
        The distance metric to use for computing the pairwise distances
        between diagrams. Default is "bottleneck".
    coverage : float, optional
        The minimum proportion of items in a mapper cluster required for its
        corresponding diagram to be included in the analysis. Default is 0.6.

    Returns:
    --------
    tuple
        A tuple containing two arrays: the trimmed diagram keys
        (i.e., the hyperparameters used to generate each diagram)
        and the corresponding trimmed pairwise distance matrix.
    """
    diagrams = get_diagrams(files, coverage)
    assert len(diagrams) > 0, "Coverage parameter is too strict"
    curvature_dgms = convert_to_gtda(diagrams.values())
    distance_metric = PairwiseDistance(metric=metric)
    distance_metric.fit(curvature_dgms)
    all_distances = distance_metric.transform(curvature_dgms)

    keys = np.array(list(diagrams.keys()), dtype=object)
    trimmed_keys, trimmed_distances = collapse_equivalent_models(keys, all_distances)

    return trimmed_keys, trimmed_distances


def collapse_equivalent_models(keys, distance_matrix):
    """
    Collapse diagrams with identical curvature profiles
    into a single representative diagram.

    Parameters:
    -----------
    keys : array-like
        A list of the keys for each diagram in the distance matrix.
    distance_matrix : array-like
        The pairwise distance matrix between the diagrams.

    Returns:
    --------
    tuple
        A tuple containing two arrays: the trimmed diagram keys
        (i.e., the hyperparameters used to generate each diagram)
        and the corresponding trimmed pairwise distance matrix.
    """

    equivalent_items = np.argwhere(distance_matrix == 0)
    drops = set()
    for pair in equivalent_items:
        if pair[0] != pair[1]:  # non-diagonal elements
            drops.add(max(pair))
    df = pd.DataFrame(distance_matrix)
    df.drop(labels=list(drops), axis=0, inplace=True)
    df.drop(labels=list(drops), axis=1, inplace=True)

    trimmed_keys = keys[list(df.index)]

    trimmed_matrix = df.values
    return trimmed_keys, trimmed_matrix


def get_diagrams(dir, coverage):
    """
    Load the persistence diagrams from fitted models
    and return them as a dictionary. This function only
    returns diagrams for models that satisfy the coverage constraint.

    Parameters:
    -----------
    dir : str
        The directory containing the models,
        from which diagrams can be extracted.
    coverage : float
        The minimum proportion of items in a mapper cluster required
        for its corresponding diagram to be included in the analysis.

    Returns:
    --------
    dict
        A dictionary mapping the hyperparameters used to generate each diagram
        to the corresponding persistence diagram object.
    """

    assert os.path.isdir(
        dir
    ), "Please first compute mapper objects using `coal_mapper_generator.py`"

    # TODO: add a filter here for `unlcustered` plants
    diagrams = {}
    cwd = os.path.dirname(__file__)
    dir = os.path.join(cwd, dir)
    for file in os.listdir(dir):
        if file.endswith(".pkl"):
            mapper_file = os.path.join(dir, file)
            model = Model(mapper_file)
            if len(model.unclustered_items) / len(model.tupper.clean) < 1 - coverage:
                mapper = model.mapper
                hyper_params = model.hyper_parameters
                diagrams[hyper_params] = mapper.diagram

    keys = list(diagrams.keys())
    keys.sort()
    sorted_diagrams = {i: diagrams[i] for i in keys}

    return sorted_diagrams


def convert_to_gtda(diagrams):
    """
    Converts a list of persistence diagrams to the format expected
    by Giotto-TDA's `gtda.homology.VietorisRipsPersistence` transformer.
    This function pads persistence diagrams.

    Parameters:
    -----------
        diagrams : list
        A list of persistence diagrams, where each diagram is a list of tuples
        representing the birth and death values of each persistence point.

    Returns:
    -----------
        Xt_padded: numpy.ndarray
        A 3D numpy array of shape (n_samples, n_features, 3).
            * `n_samples` is the number of persistence diagrams,
            * `n_features` is the total number of persistence points in all diagrams padded with zeros
            * (birth, death, homology_dimension) of points in the diagrams.
    """
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
