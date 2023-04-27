import argparse
import os
import pickle
import sys

from dotenv import load_dotenv
from metric_helper import topology_metric

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
        "-f",
        "--coverage_filter",
        type=float,
        default=0.8,
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

    n = args.num_policy_groups
    path_to_mappers = os.path.join(root, f"data/models/{n}_policy_groups")

    keys, distances = topology_metric(
        files=path_to_mappers, metric=args.metric, coverage=args.coverage_filter
    )
    results = {"keys": keys, "distances": distances}

    if args.save:
        distance_file = f"curvature_{args.metric}_pairwise_distances.pkl"

        out_dir_message = f"{distance_file} successfully written."

        output_dir = os.path.join(
            root,
            f"data/model_analysis/distance_matrices/{n}_policy_groups/",
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
