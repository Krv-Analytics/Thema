""" Reducing Coal Mapper Dataset to low dimensions using UMAP"""

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from umap import UMAP


def projection_driver(
    df: pd.DataFrame,
    n: int,
    d: float,
    dimensions: int = 2,
    projector: str = "UMAP",
):
    """
    This function performs a projection of a DataFrame.
    At the moment we only support the UMAP method.

    Parameters:
    -----------
    df : pd.DataFrame
        The data to be projected.
    n : int
        The number of neighbors to be used in the projection.
    d : float
        The minimum distance to be used in the projection.
    dimensions : int, default 2
        The number of dimensions in the projected dataset.
    projector : str, default "UMAP"
        The projection method to be used.

    Returns:
    -----------
    dict
        A dictionary containing the projection and hyperparameters.
    """

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


def projection_file_name(projector, n, d, dimensions=2):
    """
    This function generates a filename for a projected dataset.

    Parameters:
    -----------
    projector : str
        The projection method used.
    n : int
        The number of neighbors used in the projection.
    d : float
        The minimum distance used in the projection.
    dimensions : int, default 2
        The number of dimensions in the projected dataset.

    Returns:
    -----------
    str
        The filename for the projected dataset.
    """
    output_file = f"{projector}_Nbors{n}_minDist_{d}_{dimensions}D.pkl"
    return output_file


def env():
    """
    This function loads the .env file and adds necessary folders to the system path.

    Returns:
    -----------
    str
        The root directory of the project.
    """

    load_dotenv()
    root = os.getenv("root")
    src = os.getenv("src")
    sys.path.append(root)
    sys.path.append(src)
    sys.path.append(src + "modeling/nammu/")
    return root
