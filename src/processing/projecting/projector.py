"Project cleaned data using UMAP."

import argparse
import os
import pickle
import sys
import time

######################################################################
# Silencing UMAP Warnings
import warnings

from dotenv import load_dotenv
from numba import NumbaDeprecationWarning
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

os.environ["KMP_WARNINGS"] = "off"
######################################################################

from __init__ import env
from projector_helper import projection_driver, projection_file_name

root, src = env()  # Load .env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    YAML_PATH = os.getenv("params")
    if os.path.isfile(YAML_PATH):
        with open(YAML_PATH, "r") as f:
            params = OmegaConf.load(f)
    else:
        print("params.yaml file note found!")

    parser.add_argument(
        "-c",
        "--clean_data",
        type=str,
        default=os.path.join(root, params["clean_data"]),
        help="Location of Cleaned data set",
    )

    parser.add_argument(
        "--projector",
        type=str,
        default="UMAP",
        help="Set to the name of projector for dimensionality reduction. ",
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=params["projector_dimension"],
        help="Set dimension of projection. ",
    )

    parser.add_argument(
        "-n",
        "--n_neighbors",
        type=int,
        default=5,
        help=" (UMAP ONLY:Set UMAP parameter for `n_neighbors`",
    )

    parser.add_argument(
        "-d",
        "--min_dist",
        type=float,
        default=0,
        help="UMAP ONLY: Set UMAP parameter for `min_dist`",
    )

    parser.add_argument(
        "--random_seed",
        default=params["projector_random_seed"],
        type=int,
        help="UMAP ONLY: Projections generated with random seed set in parameters. (set to -1 for random)",
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

    assert os.path.isfile(args.clean_data), "\n Invalid path to Clean Data"
    # Load Dataframe
    with open(args.clean_data, "rb") as clean:
        reference = pickle.load(clean)
        df = reference["clean_data"]
    rel_outdir = (
        "data/" + params["Run_Name"] + "/projections/" + params["projector"] + "/"
    )
    output_dir = os.path.join(root, rel_outdir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.random_seed == -1:
        args.random_seed = int(time.time())

    # Check Projections is valid
    assert args.projector in ["UMAP", "TSNE", "PCA"], "\n UMAP is the only supported dimensionality reduction algorithm supported at this time. Please check that you have correctly set your params.json."

    results = projection_driver(
        df,
        n=args.n_neighbors,
        d=args.min_dist,
        dimensions=args.dim,
        projector=args.projector,
        seed=args.random_seed,
    )

    output_file = projection_file_name(
        projector=params["projector"],
        n=args.n_neighbors,
        d=args.min_dist,
        dimensions=2,
        seed=args.random_seed,
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

        print(f"SUCCESS: Completed Projection!", "green"), "Written to {rel_outdir}"
        print("\n")
        print(
            "-------------------------------------------------------------------------------------- \n\n"
        )
