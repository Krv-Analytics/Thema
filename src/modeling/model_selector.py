"Select models for analysis from structural equivalency classes of Graph Models based on best coverage."


import argparse
import sys
import os
import pickle
from dotenv import load_dotenv
import json
import numpy as np

from model_selector_helper import (
    read_graph_clustering,
    select_models,
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
        help="Select metric that defines the precomputed agglomerative clustering model.",
    )

    parser.add_argument(
        "-n",
        "--num_groups",
        type=int,
        default=2,
        help="Select folder of mapper objects to compare,identified by the number of policy groups.",
    )

    parser.add_argument(
        "-f",
        "--coverage_filter",
        type=float,
        default=params_json["coverage_filter"],
        help="A minimum model coverage for visualizing a histogram. Only set when using '-H' tag as well.",
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
    n = args.num_groups
    coverage = params_json["coverage_filter"]
    models_dir = "data/" + params_json["Run_Name"] + f"/models/"

    # Choose ~best~ models from curvature equivalency classes.
    # Current implementation chooses Model with the best coverage.

    rel_cluster_dir = (
        "data/"
        + params_json["Run_Name"]
        + f"/model_analysis/graph_clustering/{coverage}_coverage/{n}_policy_groups/"
    )
    cluster_dir = os.path.join(root, rel_cluster_dir)
    try:
        keys, clustering, distance_threshold = read_graph_clustering(
            cluster_dir, metric=args.metric, n=n
        )

        rel_model_dir = (
            "data/" + params_json["Run_Name"] + f"/models/{n}_policy_groups/"
        )
        model_dir = os.path.join(root, rel_model_dir)
        selection = select_models(model_dir, keys, clustering, n)

        model_file = (
            f"equivalence_class_candidates_{args.metric}_{distance_threshold}DT.pkl"
        )

        out_dir_message1 = f"{model_file} successfully written."

        output_dir1 = (
            "data/"
            + params_json["Run_Name"]
            + f"/model_analysis/token_models/{coverage}_coverage/{n}_policy_groups/"
        )
        output_dir1 = os.path.join(root, output_dir1)
        # Check if output directory already exists
        if os.path.isdir(output_dir1):
            model_file = os.path.join(output_dir1, model_file)

        else:
            os.makedirs(output_dir1, exist_ok=True)
            model_file = os.path.join(output_dir1, model_file)
        # Writing Selected Models
        with open(model_file, "wb") as handle:
            pickle.dump(selection, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Stability Selection
        cluster_sizes = []
        for key in selection.keys():
            cluster_sizes.append(selection[key]["cluster_size"])

        stable_cluster = np.argmax(cluster_sizes)
        stable_model = selection[stable_cluster]["model"]

        output_dir2 = (
            "data/" + params_json["Run_Name"] + f"/final_models/{coverage}_coverage/"
        )
        output_dir2 = os.path.join(root, output_dir2)
        stable_model_file = f"{n}_policy_group_model.pkl"
        out_dir_message2 = f"{stable_model_file} successfully written."

        # Check if output directory already exists
        if os.path.isdir(output_dir2):
            stable_model_file = os.path.join(output_dir2, stable_model_file)

        else:
            os.makedirs(output_dir2, exist_ok=True)
            stable_model_file = os.path.join(output_dir2, stable_model_file)
        # Writing Selected Models
        with open(stable_model_file, "wb") as handle:
            with open(stable_model, "rb") as f:
                model = pickle.load(f)
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if args.Verbose:
            print("\n")
            print(
                "-------------------------------------------------------------------------------- \n\n"
            )
            print(f"Model Selection based on Stability and Coverage complete!")
            print()
            print(f"Token Models written to: \n {out_dir_message1}")
            print(f"Final Model written to: \n {out_dir_message2}")

            print(
                "\n\n -------------------------------------------------------------------------------- "
            )
    except AssertionError:
        assert 1 == 1
