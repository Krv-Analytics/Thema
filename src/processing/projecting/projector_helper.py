""" Reducing Coal Mapper Dataset to low dimensions using UMAP"""

import os
import sys

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from termcolor import colored


######################################################################
# Silencing UMAP Warnings 
import warnings 
from numba import NumbaDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
######################################################################


def projection_driver(
    df: pd.DataFrame,
    n: int,
    d: float,
    dimensions: int = 2,
    projector: str = "UMAP",
    seed : int = 42,
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

    
    data = df.dropna()

    if projector == "UMAP":
        umap_2d = UMAP(
            min_dist=d,
            n_neighbors=n,
            n_components=dimensions,
            init="random",
            random_state= seed,
        )

        projection = umap_2d.fit_transform(data)
    
    if projector == "TSNE": 
        num_samples = df.shape[0]
        perplexity = min(30, num_samples - 1)
        tsne = TSNE(n_components=dimensions, random_state=seed, perplexity=perplexity)
        projection = tsne.fit_transform(data)
    

    if projector == "PCA": 
        pca = PCA(n_components=dimensions, random_state=42)
        projection = pca.fit_transform(data)

    results = {"projection": projection, "hyperparameters": [n, d, dimensions]}

    return results


def projection_file_name(projector, n, d, dimensions=2, seed=42):
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
    output_file = f"{projector}_Nbors{n}_minDist_{d}_{dimensions}D_rs_{seed}.pkl"
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
