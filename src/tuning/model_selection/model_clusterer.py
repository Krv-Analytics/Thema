"Group policy Models based on their graph structure using Ollivier Ricci Curvature and TDA"

import argparse
import sys
import os
import pickle
from dotenv import load_dotenv

from model_clusterer_helper import (
    cluster_models,
    read_distance_matrices,
)

load_dotenv()
src = os.getenv("src")
root = os.getenv("root")
sys.path.append(src)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="landscape",
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )

    parser.add_argument(
        "-n",
        "--num_policy_groups",
        type=int,
        default=2,
        help="Select folder of mapper objects to compare,identified by the number of policy groups.",
    )

    parser.add_argument(
        "-p",
        "--dendrogram_levels",
        type=int,
        default=3,
        help="Numnber of levels to see in dendrogram plot.",
    )

    parser.add_argument(
        "-s",
        "--save",
        default=True,
        help="If True, save the clustering model and distances as pickle files.",
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

    # Read in Keys and distances from pickle file
    n = args.num_policy_groups
    keys, distances = read_distance_matrices(metric=args.metric, n=n)
    model = cluster_models(
        keys,
        distances,
        p=args.dendrogram_levels,
        metric=args.metric,
        num_policy_groups=n,
    )

    results = {"keys": keys, "model": model}
    if args.save:
        model_file = f"curvature_{args.metric}_clustering_model.pkl"

        out_dir_message = f"{model_file} successfully written."

        output_dir = os.path.join(
            root, f"data/model_analysis/models/{n}_policy_groups/"
        )

        # Check if output directory already exists
        if os.path.isdir(output_dir):
            model_file = os.path.join(output_dir, model_file)

        else:
            os.makedirs(output_dir, exist_ok=True)
            model_file = os.path.join(output_dir, model_file)
        with open(model_file, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if args.Verbose:
            print("\n")
            print(
                "-------------------------------------------------------------------------------- \n\n"
            )
            print(f"{out_dir_message}")

            print(
                "\n\n -------------------------------------------------------------------------------- "
            )
