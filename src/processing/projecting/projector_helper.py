""" Reducing Coal Mapper Dataset to low dimensions using UMAP"""

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from umap import UMAP


def projection_file_name(projector, n, d, dimensions=2):
    output_file = f"{projector}_Nbors{n}_minDist_{d}_{dimensions}D.pkl"
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


def env():
    """Load .env file and add necessary folders to your `sys` path."""
    load_dotenv()
    root = os.getenv("root")
    src = os.getenv("src")
    sys.path.append(root)
    sys.path.append(src)
    sys.path.append(src + "modeling/nammu/")
    return root
