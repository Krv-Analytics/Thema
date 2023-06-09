"Group Models based on their graph structure via curvature filtrations."

import argparse
import sys
import os
import pickle
import json
from dotenv import load_dotenv

from model_clusterer_helper import (
    cluster_models,
    read_distance_matrices,
)

load_dotenv()
src = os.getenv("src")
sys.path.append(src)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    root = os.getenv("root")
    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")

    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default=params_json["dendrogram_metric"],
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
        "-d",
        "--distance_threshold",
        type=float,
        default=params_json["dendrogram_cut"],
        help="Select distance threshold for agglomerative clustering model.",
    )

    parser.add_argument(
        "-p",
        "--dendrogram_levels",
        type=int,
        default=params_json["dendrogram_levels"],
        help="Number of levels to see in dendrogram plot.",
    )

    parser.add_argument(
        "-s",
        "--save",
        default=False,
        action="store_true",
        help="If tagged, save the clustering model as a pickle files.",
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
    rel_distance_dir = "data/" + params_json["Run_Name"] + f"/model_analysis/distance_matrices/{n}_policy_groups/"
    distance_dir = os.path.join(
        root, rel_distance_dir
    )
    keys, distances = read_distance_matrices(distance_dir, metric=args.metric, n=n)

    # Fit Hierarchical Clustering
    model = cluster_models(
        distances,
        p=args.dendrogram_levels,
        metric=args.metric,
        num_policy_groups=n,
        distance_threshold=args.distance_threshold,
        plot=not args.save,
    )

    results = {
        "keys": keys,
        "model": model,
        "distance_threshold": args.distance_threshold,
    }
    if args.save:
        model_file = f"curvature_{args.metric}_clustering_model.pkl"

        out_dir_message = f"{model_file} successfully written."
        
        rel_outdir = "data/" + params_json["Run_Name"] + f"/model_analysis/graph_clustering/{n}_policy_groups/"
        output_dir = os.path.join(
            root, rel_outdir
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
