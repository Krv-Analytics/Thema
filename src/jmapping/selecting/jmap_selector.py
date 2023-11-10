"Select jmaps for analysis from structural equivalency classes of Graph jmaps based on best coverage."
import argparse
import os
import pickle
import sys

import numpy as np
from jmap_selector_helper import (
    get_best_covered_jmap,
    get_most_nodes_jmap,
    read_graph_clustering,
    select_jmaps,
)
from omegaconf import OmegaConf

from . import env

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
        "-m",
        "--metric",
        type=str,
        default=params["dendrogram_metric"],
        help="Select metric that defines the precomputed agglomerative clustering jmap.",
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
        default=params["coverage_filter"],
        help="A minimum jmap coverage for visualizing a histogram. Only set when using '-H' tag as well.",
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
    coverage = params["coverage_filter"]
    jmaps_dir = "data/" + params["Run_Name"] + f"/jmaps/"

    # Choose ~best~ jmaps from curvature equivalency classes.
    # Current implementation chooses jmap with the best coverage.

    rel_cluster_dir = (
        "data/"
        + params["Run_Name"]
        + f"/jmap_analysis/graph_clustering/{coverage}_coverage/{n}_policy_groups/"
    )
    cluster_dir = os.path.join(root, rel_cluster_dir)
    try:
        keys, clustering, distance_threshold = read_graph_clustering(
            cluster_dir, metric=args.metric, n=n
        )

        rel_jmap_dir = "data/" + params["Run_Name"] + f"/jmaps/{n}_policy_groups/"
        jmap_dir = os.path.join(root, rel_jmap_dir)
        selection = select_jmaps(
            jmap_dir, keys, clustering, n, selection_fn=get_most_nodes_jmap
        )

        jmap_file = (
            f"equivalence_class_candidates_{args.metric}_{distance_threshold}DT.pkl"
        )

        out_dir_message1 = f"{jmap_file} successfully written."

        output_dir1 = (
            "data/"
            + params["Run_Name"]
            + f"/jmap_analysis/token_jmaps/{coverage}_coverage/{n}_policy_groups/"
        )
        output_dir1 = os.path.join(root, output_dir1)
        # Check if output directory already exists
        if os.path.isdir(output_dir1):
            jmap_file = os.path.join(output_dir1, jmap_file)

        else:
            os.makedirs(output_dir1, exist_ok=True)
            jmap_file = os.path.join(output_dir1, jmap_file)
        # Writing Selected jmaps
        with open(jmap_file, "wb") as handle:
            pickle.dump(selection, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Stability Selection
        cluster_sizes = []
        for key in selection.keys():
            cluster_sizes.append(selection[key]["cluster_size"])

        stable_cluster = np.argmax(cluster_sizes)
        stable_jmap = selection[stable_cluster]["jmap"]

        output_dir2 = (
            "data/" + params["Run_Name"] + f"/final_jmaps/{coverage}_coverage/"
        )
        output_dir2 = os.path.join(root, output_dir2)
        stable_jmap_file = f"{n}_policy_group_jmap.pkl"
        out_dir_message2 = f"{stable_jmap_file} successfully written."

        # Check if output directory already exists
        if os.path.isdir(output_dir2):
            stable_jmap_file = os.path.join(output_dir2, stable_jmap_file)

        else:
            os.makedirs(output_dir2, exist_ok=True)
            stable_jmap_file = os.path.join(output_dir2, stable_jmap_file)
        # Writing Selected jmaps
        with open(stable_jmap_file, "wb") as handle:
            with open(stable_jmap, "rb") as f:
                jmap = pickle.load(f)
            pickle.dump(jmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if args.Verbose:
            print("\n")
            print(
                "-------------------------------------------------------------------------------- \n\n"
            )
            print(f"jmap Selection based on Stability and Coverage complete!")
            print()
            print(f"Token jmaps written to: \n {out_dir_message1}")
            print(f"Final jmap written to: \n {out_dir_message2}")

            print(
                "\n\n -------------------------------------------------------------------------------- "
            )
    except AssertionError:
        assert 1 == 1
