import os
import argparse
import sys
import pickle

from projections_helper import projection_driver


cwd = os.path.dirname(__file__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.join(cwd, "./../../data/coal_mapper_one_hot_scaled.pkl"),
        help="Select location of local data set, as pulled from Mongo.",
    )

    parser.add_argument(
        "-n",
        "--neighbors_list",
        type=int,
        nargs="+",
        default=[3, 5, 10, 20, 40],
        help="Insert a list of n_neighbors to grid search",
    )

    parser.add_argument(
        "-d",
        "--min_dists",
        type=float,
        nargs="+",
        default=[0, 0.01, 0.05, 0.1, 0.5, 1],
        help="Insert a list of min_dists to grid search",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    assert os.path.isfile(args.path), "Invalid Input Data"
    # Load Dataframe
    with open(args.path, "rb") as f:
        df = pickle.load(f)

    output_dir = os.path.join(cwd, "../outputs/projections/")

    # Check if output directory already exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    projection_params = (args.min_dists, args.neighbors_list)
    projection_driver(df, projection_params)

    print(
        "\n################################################################################## \n\n"
    )
    print(f"Successfully Generated UMAP Projection")

    print(
        "\n\n##################################################################################\n"
    )
