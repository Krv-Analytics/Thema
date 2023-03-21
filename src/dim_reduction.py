""" Reducing Coal Mapper Dataset to low dimensions using UMAP"""

import os
import sys
import argparse
import pickle

from umap import UMAP

cwd = os.path.dirname(__file__)


def uMAP_file_name(n_neighbors=10, min_dist=0.2, dimensions=2):
    output_dir = os.path.join(cwd, "../outputs/projections/")
    output_file = os.path.join(
        output_dir, f"umap_{dimensions}D_nbors{n_neighbors}_minD{min_dist}.pkl"
    )
    return output_file


def uMAP_grid(df, dists, neighbors, dimensions=2):
    data = df.dropna()
    assert type(dists) == list, "Not list"
    assert type(neighbors) == list, "Not list"
    print(f"Computing UMAP Grid Search! ")
    print(
        "--------------------------------------------------------------------------------"
    )
    print(f"Choices for n_neighbors: {neighbors}")
    print(f"Choices for m_dist: {dists}")
    print(
        "-------------------------------------------------------------------------------- \n"
    )

    # generate figure
    for d in dists:
        for n in neighbors:
            umap_2d = UMAP(
                min_dist=d,
                n_neighbors=n,
                n_components=dimensions,
                init="random",
                random_state=0,
            )
            projection = umap_2d.fit_transform(data)
            output_file = uMAP_file_name(n, d, dimensions=dimensions)
            with open(output_file, "wb") as f:
                pickle.dump(projection, f)


######################################################################################################
##############################################################################
###################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.join(cwd, "./../data/coal_mapper_one_hot_scaled.pkl"),
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

    uMAP_grid(df, dists=args.min_dists, neighbors=args.neighbors_list)

    print(
        "\n################################################################################## \n\n"
    )
    print(f"Successfully Generated UMAP Projection")

    print(
        "\n\n##################################################################################\n"
    )
