""" Reducing Coal Mapper Dataset to low dimensions using UMAP"""

import os
import pickle
import pandas as pd

from umap import UMAP

cwd = os.path.dirname(__file__)


def projection_file_name(projector, n, d, dimensions=2):
    output_file = f"{projector}_N{n}_minDist_{d}_{dimensions}D.pkl"
    return output_file


def projection_driver(
    df: pd.DataFrame,
    n: int,
    d: float,
    dimensions: int = 2,
    projector: str = "UMAP",
):

    assert projector == "UMAP", "No other projections supported at this time."
    data = df.dropna()

    umap_2d = UMAP(
        min_dist=d,
        n_neighbors=n,
        n_components=dimensions,
        init="random",
        random_state=0,
    )

    projection = umap_2d.fit_transform(data)

    results = {"projection": projection, "hyperparameters": [n, d, dimensions]}

    return results
