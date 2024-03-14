import argparse
import os
import pickle
import sys

from metric_helper import topology_metric
from omegaconf import OmegaConf

from __init__ import env

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
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )
    parser.add_argument(
        "-f",
        "--coverage_filter",
        type=float,
        default=params["coverage_filter"],
        help="Select the percentage of unqiue samples that need to be covered by Mapper's fit.",
    )

    parser.add_argument(
        "-n",
        "--num_policy_groups",
        type=int,
        default=6,
        help="Select folder of mapper objects to compare,identified by the number of policy groups.",
    )

    parser.add_argument(
        "-s",
        "--save",
        default=True,
        help="If True, save the clustering jmap and distances as pickle files.",
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

    n = args.num_policy_groups
    coverage = args.coverage_filter
    rel_path_to_mappers = "data/" + params["Run_Name"] + f"/jmaps/{n}_policy_groups/"
    path_to_mappers = os.path.join(root, rel_path_to_mappers)

    try:
        keys, distances = topology_metric(
            files=path_to_mappers, metric=args.metric, coverage=coverage
        )

        results = {"keys": keys, "distances": distances}

        distance_file = f"curvature_{args.metric}_pairwise_distances.pkl"

        out_dir_message = f"{distance_file} successfully written."

        rel_ouput_dir = (
            "data/"
            + params["Run_Name"]
            + f"/jmap_analysis/distance_matrices/{coverage}_coverage/{n}_policy_groups/"
        )
        output_dir = os.path.join(
            root,
            rel_ouput_dir,
        )

        # Check if output directory already exists
        if os.path.isdir(output_dir):
            distance_file = os.path.join(output_dir, distance_file)

        else:
            os.makedirs(output_dir, exist_ok=True)
            distance_file = os.path.join(output_dir, distance_file)

        with open(distance_file, "wb") as handle:
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

    except AssertionError:
        # TODO: Log something here?
        assert 1 == 1