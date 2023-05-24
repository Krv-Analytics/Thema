"Helper functions for Mapper Policy Models"

import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hdbscan import HDBSCAN

from jmapper import JMapper
from tupper import Tupper
from nammu.curvature import ollivier_ricci_curvature


def model_generator(
    tupper,
    n_cubes,
    perc_overlap,
    hdbscan_params,
    min_intersection_vals: list,
    verbose=False,
):
    """
    Generate a Graph Clustering Model by fitting a JMapper.

    Parameters
    -----------
    tupper: <tupper.Tupper>
            A data container that holds raw, cleaned, and projected
            versions of user data.

    n_cubes: int
            Number of cubes used to cover of the latent space.
            Used to construct a kmapper.Cover object.

    perc_overlap: float
        Percentage of intersection between the cubes covering
        the latent space. Used to construct a kmapper.Cover object.

    hdbscan_params: tuple
        (m:int,M:int) where `m` is min_cluster_size
        and `M` is max_cluster_size.

    min_interesection_vals: list
        List of min_intersection values over which to calculate various
        graph representations of a given JMapper fit.

    verbose: bool, default is False
            If True, use kmapper logging levels.

    Returns
    -----------
    results : dict
        A dictionary with various models.
        Keys are `min_intersection` parameters
        Vals are JMappers.

    """

    # HDBSCAN
    min_cluster_size, max_cluster_size = hdbscan_params
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )
    # Configure JMapper
    j_mapper = JMapper(tupper)
    j_mapper.fit(n_cubes, perc_overlap, clusterer)

    results = {}
    if len(j_mapper.complex["links"]) > 0:
        for val in min_intersection_vals:
            print(f"Here is my min_intersection value: {val}")
            # Generate Graph
            try:
                j_mapper.to_networkx(min_intersection=val)
                j_mapper.connected_components()
                # Compute Curvature and Persistence Diagram
                j_mapper.curvature = ollivier_ricci_curvature
                j_mapper.calculate_homology()
                results[val] = j_mapper
            except:
                j_mapper.complex == dict()
                if verbose:
                    print("Empty Mapper!")
        return results
    else:
        if verbose:
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )
            print("Empty Simplicial Complex. No file written")
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )
        return results


def generate_model_filename(args, n_neighbors, min_dist, min_intersection):
    """Generate output filename string from CLI arguments.
    A helper function for the `model_generator` script.

    Parameters
    -----------
    args: argparse.ArgumentParser
        CLI arguments from the `model_generator` script.

    n_neighbors: int
        Number of neighbors used in the UMAP projection used
        to fit a JMapper.

    min_dist: float
        Minimal distance between points in the UMAP projection used
        to fit a JMapper.

    min_intersection: int
        Minimum intersection considered when computing the graph.
        An edge will be created only when the intersection between
        two nodes is greater than or equal to `min_intersection`.

    Returns
    -----------
    output_file: str
        A unique filename to identify JMappers.

    """

    min_cluster_size, p, n = (
        args.min_cluster_size,
        args.perc_overlap,
        args.n_cubes,
    )
    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{min_cluster_size}_UMAP_{n_neighbors}Nbors_minDist{min_dist}_min_int{min_intersection}.pkl"

    return output_file


def get_minimal_std(df: pd.DataFrame, mask: np.array, density_cols=None):
    """Find the column with the minimal standard deviation
    within a subset of a Dataframe.

    Parameters
    -----------
    df: pd.Dataframe
        A cleaned dataframe.

    mask: np.array
        A boolean array indicating which indices of the dataframe
        should be included in the computation.

    Returns
    -----------
    col_label: int
        The index idenitfier for the column in the dataframe with minimal std.

    """
    if density_cols is None:
        density_cols = df.columns
    sub_df = df.iloc[mask][density_cols]
    col_label = sub_df.columns[sub_df.std(axis=0).argmin()]
    return col_label


def mapper_plot_outfile(
    hyper_parameters,
):
    """Generate output filename for Kepler Mapper HTML Visualizations.
    NOTE: These visualizations are no longer maintained by KepplerMapper
    and we do not reccomend using them.

    Parameters
    -----------
    hyper_parameters: list
        A list of hyperparameters used to generate a particular Mapper.

    Returns
    -----------
    output_file: str
        A unique filename to identify JMapper Visualization.
    """
    root = env()
    n, p, nbors, d, hdbscan_params, min_intersection = hyper_parameters
    (
        min_cluster_size,
        max_cluster_size,
    ) = hdbscan_params  # max_cluster size always set to zero, i.e. no upper bound
    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{min_cluster_size}_{max_cluster_size}UMAP_{nbors}Nbors_minD{d}_min_Intersection{min_intersection}.html"
    output_dir = os.path.join(root, "data/visualizations/mapper_htmls/")

    if os.path.isdir(output_dir):
        output_file = os.path.join(output_dir, output_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

    return output_file


# Unecessary?
def config_plot_data(tupper: Tupper):
    """ "Configure the data in a tupper to agree with KepplerMapper
    visualizations.
    NOTE: These visualizations are no longer maintained by KepplerMapper
    and we do not reccomend using them.

    Parameters
    -----------
    tupper: <tupper.Tupper>
        A data container that holds raw, cleaned, and projected
        versions of user data.

    Returns
    -----------
    numeric_data: pd.DataFrame
        Only numeric columns in the tupper.
    labels : list
        List of columns in `numeric_data`
    """
    temp_data = tupper.clean
    string_cols = temp_data.select_dtypes(exclude="number").columns
    numeric_data = temp_data.drop(string_cols, axis=1).dropna()
    labels = list(numeric_data.columns)
    return numeric_data, labels


def custom_color_scale():
    "Our own colorscale, feel free to use!"
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

    extended_colorscale = []
    for i in range(100):
        t = i / 99.0
        for j in range(len(colorscale) - 1):
            if t >= colorscale[j][0] and t <= colorscale[j + 1][0]:
                r1, g1, b1 = (
                    colorscale[j][1][1:3],
                    colorscale[j][1][3:5],
                    colorscale[j][1][5:],
                )
                r2, g2, b2 = (
                    colorscale[j + 1][1][1:3],
                    colorscale[j + 1][1][3:5],
                    colorscale[j + 1][1][5:],
                )
                r = int(r1, 16) + int(
                    (t - colorscale[j][0])
                    / (colorscale[j + 1][0] - colorscale[j][0])
                    * (int(r2, 16) - int(r1, 16))
                )
                g = int(g1, 16) + int(
                    (t - colorscale[j][0])
                    / (colorscale[j + 1][0] - colorscale[j][0])
                    * (int(g2, 16) - int(g1, 16))
                )
                b = int(b1, 16) + int(
                    (t - colorscale[j][0])
                    / (colorscale[j + 1][0] - colorscale[j][0])
                    * (int(b2, 16) - int(b1, 16))
                )
                hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                extended_colorscale.append([t, hex_color])
                break

    return colorscale


def env():
    """Load .env file and add necessary folders to your `sys` path."""
    load_dotenv()
    root = os.getenv("root")
    src = os.getenv("src")
    sys.path.append(root)
    sys.path.append(src)
    sys.path.append(src + "modeling/")
    return root


def script_paths(paths):
    root = env()
    scripts_dir = os.path.join(root, "scripts/")
    new_paths = []
    for rel_path in paths:
        abs_path = os.path.join(scripts_dir, rel_path)
        new_paths.append(abs_path)
    return new_paths
