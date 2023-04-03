""" Reducing Coal Mapper Dataset to low dimensions using UMAP"""

import os
import pickle

from umap import UMAP

cwd = os.path.dirname(__file__)


def projection_file_name(projector, dimensions=2):
    output_file = f"{projector}_{dimensions}D.pkl"
    return output_file


def projection_driver(df, projection_params, dimensions=2, projector="UMAP"):

    assert projector == "UMAP", "No other projections supported at this time."
    assert (
        len(projection_params) == 2
    ), "Must pass min_dist and num_neighbors parameters for UMAP."
    dists, neighbors = projection_params
    data = df.dropna()
    assert type(dists) == list, "Not list"
    assert type(neighbors) == list, "Not list"

    print(f"Computing UMAP Grid Search! ")
    print(
        "--------------------------------------------------------------------------------"
    )
    print(f"Choices for n_neighbors: {neighbors}")
    print(f"Choices for m_dist: {dists}")
    print(
        "-------------------------------------------------------------------------------- \n"
    )

    results = {}
    for d in dists:
        for n in neighbors:
            umap_2d = UMAP(
                min_dist=d,
                n_neighbors=n,
                n_components=dimensions,
                init="random",
                random_state=0,
            )

            projection = umap_2d.fit_transform(data)

            results[(n, d)] = projection

    keys = list(results.keys())
    keys.sort()
    sorted_results = {i: results[i] for i in keys}
    return sorted_results
