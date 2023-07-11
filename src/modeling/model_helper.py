"Helper functions for Mapper Policy Models"

import os
import sys
import math

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hdbscan import HDBSCAN

from jmapper import JMapper
from tupper import Tupper
from jgraph import JGraph
from nammu.curvature import ollivier_ricci_curvature

def model_generator(
    tupper,
    n_cubes,
    perc_overlap,
    hdbscan_params,
    min_intersection,
    verbose=False,
    # random_seed,
):
    """
    Fit a graph model using JMapper.

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

    min_interesection: int
        Min_intersection value for generating a networkx graph
        from a simplicial complex.

    verbose: bool, default is False
            If True, use kmapper logging levels.

    Returns
    -----------
    j_mapper : <jmapper.JMapper>
        A jmapper object with precomputed curvature values,
        connected components, and .

    """

    # HDBSCAN
    min_cluster_size, max_cluster_size = hdbscan_params
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )
    # Configure JMapper
    j_mapper = JMapper(tupper, n_cubes, perc_overlap, clusterer)  

    results = {}
    if len(j_mapper.complex["links"]) > 0:
        try:
            j_mapper.min_intersection = min_intersection
            j_mapper.jgraph = JGraph(tupper, j_mapper.nodes, min_intersection)
            # Compute Curvature and Persistence Diagram
            j_mapper.jgraph.curvature = ollivier_ricci_curvature
            j_mapper.jgraph.calculate_homology()
        except:
            if verbose:
                print("Empty Mapper!")
        return j_mapper
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

def _define_zscore_df(model):
    df_builder = pd.DataFrame()
    dfs = model.tupper.clean

    column_to_drop = [col for col in dfs.columns if dfs[col].nunique() == 1]
    dfs = dfs.drop(column_to_drop, axis=1)

    dfs['cluster_IDs']=(list(model.cluster_ids))

    #loop through all policy group dataframes
    for group in list(dfs['cluster_IDs'].unique()):
        zscore0 = pd.DataFrame()
        group0 = dfs[dfs['cluster_IDs']==group].drop(columns={'cluster_IDs'})
        #loop through all columns in a policy group dataframe
        for col in group0.columns:
            if col != "cluster_IDs":
                mean = dfs[col].mean()
                std = dfs[col].std()
                zscore0[col] = group0[col].map(lambda x: (x-mean)/std)
        zscore0_temp = zscore0.copy()
        zscore0_temp['cluster_IDs'] = group
        df_builder = pd.concat([df_builder,zscore0_temp])
    return df_builder

def custom_color_scale():
    "Our own colorscale, feel free to use!"

    # colorscale = [
    # [0.0, "#001219"],
    # [0.04, "#004165"],
    # [0.08, "#0070b3"],
    # [0.12, "#00a1d6"],
    # [0.16, "#00c6eb"],
    # [0.20, "#00e0ff"],
    # [0.24, "#2cefff"],
    # [0.28, "#64ffff"],
    # [0.32, "#9bfff3"],
    # [0.36, "#ceffea"],
    # [0.40, "#e8ffdb"],
    # [0.44, "#f8ffb4"],
    # [0.48, "#ffff7d"],
    # [0.52, "#ffd543"],
    # [0.56, "#ffae00"],
    # [0.60, "#ff9000"],
    # [0.64, "#ff7300"],
    # [0.68, "#ff5500"],
    # [0.72, "#ff3500"],
    # [0.76, "#ff1600"],
    # [0.80, "#ff0026"],
    # [0.84, "#d70038"],
    # [0.88, "#b2004a"],
    # [0.92, "#8e0060"],
    # [0.96, "#690075"],
    # [1.0, "#a50026"]]

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

def reorder_colors(colors):
    n = len(colors)
    ordered = []
    for i in range(n):
        if i % 2 == 0:
            ordered.append(colors[i // 2])
        else:
            ordered.append(colors[n - (i // 2) - 1])
    return ordered

def get_subplot_specs(n):
    """
    Returns subplot specs based on the number of subplots.
    
    Parameters:
        n (int): number of subplots
        
    Returns:
        specs (list): 2D list of subplot specs
    """
    num_cols = 3
    num_rows = max(math.ceil(n / num_cols), 1)
    specs = [[{"type": "pie"} for c in range(num_cols)] for r in range(num_rows)]
    return specs


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
