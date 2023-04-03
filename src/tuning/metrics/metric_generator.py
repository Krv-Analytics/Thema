import argparse
import sys
import os
import pickle

from metric_helper import topology_metric


cwd = os.path.dirname(__file__)

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

    keys, distances = topology_metric(args.metric)
    results = {"keys": keys, "distances": distances}

    if args.save:
        distance_file = f"curvature_{args.metric}_pairwise_distances.pkl"

        out_dir_message = f"{distance_file} successfully written."

        output_dir = os.path.join(
            cwd, "./../../../data/parameter_modeling/distance_matrices/"
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
