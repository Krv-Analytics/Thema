"""Compute ollivier-Ricci curvature for Coal-Plant Mapper Graphs."""

import argparse
import os
import sys
import numpy as np
import pickle
import pandas as pd

from utils import curvature_iterator, generate_results_filename
from umap import UMAP


cwd = os.path.dirname(__file__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=os.path.join(cwd, "./../data/coal_mapper_one_hot_scaled_TSNE.pkl"),
        help="Select location of local data set, as pulled from Mongo.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, overwrites existing output files.",
    )

    parser.add_argument(
        "--min_cluster_size",
        default=10,
        type=int,
        help="Sets number of clusters for the KMeans algorithm used in KMapper.",
    )
    parser.add_argument(
        "--max_cluster_size",
        default=0,
        type=int,
        help="Sets number of clusters for the KMeans algorithm used in KMapper.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Set random seed to ensure reproducibility.",
    )

    parser.add_argument(
        "-n",
        "--n_cubes",
        default=6,
        type=int,
        help="Number of cubes used to cover your dataset.",
    )

    parser.add_argument(
        "-p",
        "--perc_overlap",
        default=0.4,
        type=float,
        help="Percentage overlap of cubes in the cover.",
    )
    parser.add_argument(
        "--min_intersection",
        nargs="+",
        default=[1],
        type=int,
        help="Minimum intersection reuired between cluster elements to form an edge in the graph representation.",
    )
    parser.add_argument(
        "-v",
        "--Verbose",
        default=False,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    assert os.path.isfile(args.data), "Invalid Input Data"
    # Load Dataframe
    with open(args.data, "rb") as f:
        print("Reading pickle file")
        df = pickle.load(f)

    data = df.dropna()

    # TODO: Read in a particular projection
    # For now we generate one
    print("Generating UMAP Projection")
    proj_2D = UMAP(
        min_dist=0,
        n_neighbors=10,
        n_components=2,
        init="random",
        random_state=0,
    ).fit_transform(data)

    output_file = generate_results_filename(args)

    output_dir = os.path.join(cwd, "../outputs/curvature/")

    # Check if output directory already exists
    if os.path.isdir(output_dir):
        output_file = os.path.join(output_dir, output_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

    n, p = args.n_cubes, args.perc_overlap
    min_intersections = args.min_intersection
    hdbscan_params = args.min_cluster_size, args.max_cluster_size

    results = curvature_iterator(
        data=data,
        projection=proj_2D,
        n_cubes=n,
        perc_overlap=p,
        hdbscan_params=hdbscan_params,
        min_intersection_vals=min_intersections,
        random_state=args.seed,
    )

    results["hyperparameters"] = (n, p, proj_2D, hdbscan_params)
    out_dir_message = output_file
    out_dir_message = "/".join(out_dir_message.split("/")[-2:])

    with open(output_file, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if args.Verbose:
        print("\n")
        print(
            "-------------------------------------------------------------------------------- \n\n"
        )
        print(f"Successfully written curvature analysis output to {out_dir_message}")

        print(
            "\n\n -------------------------------------------------------------------------------- "
        )
