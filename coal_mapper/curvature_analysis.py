"""Ollivier-Ricci based analysis of Coal-Plant Mapper Graphs."""

import argparse
import os
import sys
import numpy as np
import pickle
import pandas as pd


from utils import curvature_analysis


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="../data_processing/local_data/coal_mapper.pkl",
        help="Select location of local data set, as pulled from Mongo.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, overwrites existing output files.",
    )

    parser.add_argument(
        "-K",
        "--KMeans",
        default=8,
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
        default=[1, 2],
        type=int,
        help="Minimum intersection reuired between cluster elements to form an edge in the graph representation.",
    )
    parser.add_argument(
        "--column_sample",
        default=0,
        type=int,
        help="Number of columns to sample from full datafile.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    assert os.path.isfile(args.data), "Invalid Input Data"
    # Load Dataframe and One hot encode categorical variables
    with open(args.data, "rb") as f:
        print("Reading pickle file")
        df = pickle.load(f)
        df = pd.get_dummies(df, prefix="One_hot", prefix_sep="_")
        if args.column_sample:
            df = df.sample(n=args.column_sample, axis="columns")

    data = df.select_dtypes(include=np.number).values

    K, p, n = args.KMeans, args.perc_overlap, args.n_cubes
    min_intersection_vals = args.min_intersection

    # Manually Set Output Directory
    # TODO: Add functionality to generate output directory if needed
    output_file = f"../outputs/curvature/results_ncubes{n}_{p*10}perc_K{K}.pkl"

    results = curvature_analysis(
        X=data,
        n_cubes=n,
        perc_overlap=p,
        K=K,
        min_intersection_vals=min_intersection_vals,
        random_state=args.seed,
    )
    with open(output_file, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n")
