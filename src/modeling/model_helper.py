"Helper functions for Mapper Policy Model"

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from jmapper import JMapper
from model import Model
from nammu.curvature import ollivier_ricci_curvature

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src)


def model_generator(
    tupper,
    n_cubes,
    perc_overlap,
    hdbscan_params,
    min_intersection_vals,
    verbose=False,
):
    """ """

    # HDBSCAN
    min_cluster_size, max_cluster_size = hdbscan_params
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )
    # Configure JMapper
    coal_mapper = JMapper(tupper)
    coal_mapper.fit(n_cubes, perc_overlap, clusterer)

    results = {}
    if len(coal_mapper.complex["links"]) > 0:
        for val in min_intersection_vals:
            # Generate Graph
            try:
                coal_mapper.to_networkx(min_intersection=val)
                coal_mapper.connected_components()
                # Compute Curvature and Persistence Diagram
                coal_mapper.curvature = ollivier_ricci_curvature
                coal_mapper.calculate_homology()
                results[val] = coal_mapper
            except:
                if verbose:
                    print("Empty Mapper!")
        return results
    else:
        if verbose:
            print(
                "-------------------------------------------------------------------------------- \n\n"
            )
            print(f"Empty Simplicial Complex. No file written")

            print(
                "\n\n -------------------------------------------------------------------------------- "
            )
        return results


def generate_mapper_filename(args, n_neighbors, min_dist, min_intersection):
    """Generate output filename string from CLI arguments when running  script."""

    min_cluster_size, p, n = (
        args.min_cluster_size,
        args.perc_overlap,
        args.n_cubes,
    )
    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{min_cluster_size}_UMAP_{n_neighbors}Nbors_minD{min_dist}_min_int{min_intersection}.pkl"

    return output_file


def get_minimal_std(df: pd.DataFrame, mask: np.array):
    subset = df.iloc[mask]
    col_label = subset.columns[subset.std(axis=0).argmin()]
    return col_label


def mapper_plot_outfile(
    hyper_parameters,
):
    n, p, nbors, d, hdbscan_params, min_intersection = hyper_parameters
    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{hdbscan_params[0]}_UMAP_{nbors}Nbors_minD{d}_min_Intersection{min_intersection}.html"
    output_dir = os.path.join(root, "data/visualizations/mapper_htmls/")

    if os.path.isdir(output_dir):
        output_file = os.path.join(output_dir, output_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

    return output_file


def config_plot_data(tupper):
    temp_data = tupper.clean
    string_cols = temp_data.select_dtypes(exclude="number").columns
    numeric_data = temp_data.drop(string_cols, axis=1).dropna()
    labels = list(numeric_data.columns)
    return numeric_data, labels


def custom_color_scale():
    colorscale = [
        [0.0, "#001219"],
        [0.1, "#005f73"],
        [0.2, "#0a9396"],
        [0.3, "#94d2bd"],
        [0.4, "#e9d8a6"],
        [0.5, "#ee9b00"],
        [0.6, "#ca6702"],
        [0.7, "#bb3e03"],
        [0.8, "#ae2012"],
        [0.9, "#9b2226"],
        [1.0, "#a50026"],
    ]
    return colorscale
