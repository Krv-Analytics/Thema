"Project cleaned data using UMAP."

import argparse
import os
import pickle
import sys
import time
import json


######################################################################
# Silencing UMAP Warnings 
import warnings 
from numba import NumbaDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


######################################################################

from projector_helper import env, projection_driver, projection_file_name

if __name__ == "__main__":

    root = env()
    parser = argparse.ArgumentParser()
    
    JSON_PATH = os.getenv("JSON_PATH")
    try: 
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    except: 
        print("params.json file note found!")

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.join(
            root, params_json["clean_data"]),
        help="Select location of local data set, as pulled from Mongo.",
    )

    parser.add_argument(
        "--umap",
        type=bool,
        default=True,
        help="Use UMAP algorithm to compute projection. ",
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=params_json["projector_dimension"],
        help="Set dimension of UMAP projection. ",
    )

    parser.add_argument(
        "-n",
        "--n_neighbors",
        type=int,
        default=5,
        help="Set UMAP parameter for `n_neighbors`",
    )

    parser.add_argument(
        "-d",
        "--min_dist",
        type=float,
        default=0,
        help="Set UMAP parameter for `min_dist`",
    )

    parser.add_argument(
        "--random",
        default=False,
        action='store_true',
        help="If set, will generate projections with a random seed (if applicable)",
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

    assert os.path.isfile(args.path), "Invalid Input Data"
    # Load Dataframe
    with open(args.path, "rb") as f:
        reference = pickle.load(f)
        df = reference["clean_data"]
    output_dir = os.path.join(root, "data/projections/UMAP/")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    random_seed = 42
    if args.random:
        random_seed = int(time.time())

    if args.umap:
        # Generate Projection
        results = projection_driver(
            df, n=args.n_neighbors, d=args.min_dist, dimensions=args.dim, seed=random_seed,
        )

        output_file = projection_file_name(
            projector="UMAP", n=args.n_neighbors, d=args.min_dist, dimensions=2, seed = random_seed
        )
        output_file = os.path.join(output_dir, output_file)

        # Output Message
        rel_outdir = "/".join(output_file.split("/")[-3:])

        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        if args.Verbose:
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )

            print(f"Finished projecting! Written to {rel_outdir}")
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )
    else:
        print("UMAP is only dimensionality reduction algorithm supported at this time.")
        print("Please set `--umap` to True.")
