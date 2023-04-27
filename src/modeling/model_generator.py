"""Generate Graph Models using JMapper."""

import argparse
import os
import pickle
import sys

from model_helper import (
    generate_model_filename,
    model_generator,
    env,
    script_paths,
)
from tupper import Tupper


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    root = env()

    parser.add_argument(
        "--raw",
        type=str,
        help="Select location of raw data set, as pulled from Mongo.",
    )
    parser.add_argument(
        "--clean",
        type=str,
        help="Select location of clean data.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        help="Select location of projection.",
    )

    parser.add_argument(
        "--min_cluster_size",
        default=4,
        type=int,
        help="Sets `min_cluster_size`, a parameter for HDBSCAN.",
    )
    parser.add_argument(
        "--max_cluster_size",
        default=0,
        type=int,
        help="Sets `max_cluster_size`, a parameter for HDBSCAN.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Set random seed to ensure reproducibility.",
    )

    parser.add_argument(
        "-n",
        "--n_cubes",
        default=10,
        type=int,
        help="Number of cubes used to cover your dataset.",
    )

    parser.add_argument(
        "-p",
        "--perc_overlap",
        default=0.6,
        type=float,
        help="Percentage overlap of cubes in the cover.",
    )
    parser.add_argument(
        "--min_intersection",
        nargs="+",
        default=[1],
        type=int,
        help="Minimum intersection reuired between cluster elements to \
            form an edge in the graph representation.",
    )
    parser.add_argument(
        "--script",
        default=False,
        action="store_true",
        help="If set, we update paths to match the `scripts` dir.",
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

    # Adjust Paths when running model_grid_search.sh
    if args.script:
        raw, clean, projection = script_paths([args.raw, args.clean, args.projection])
    else:
        raw, clean, projection = args.raw, args.clean, args.projection

    # Initialize a `Tupper`
    tupper = Tupper(raw, clean, projection)

    nbors, d, dimension = tupper.get_projection_parameters()

    n, p = args.n_cubes, args.perc_overlap
    min_intersections = args.min_intersection
    hdbscan_params = args.min_cluster_size, args.max_cluster_size

    # GENERATE MODELS
    # Given our hyperparameters, we generate graphs, curvature,
    # and diagrams for all min_intersection values from a single JMapper fit.
    # This is done for efficiency purposes.
    results = model_generator(
        tupper,
        n_cubes=n,
        perc_overlap=p,
        hdbscan_params=hdbscan_params,
        min_intersection_vals=min_intersections,
        verbose=args.Verbose,
    )

    # Unpack each graph (based on min_intersection) into it's own output file.
    for val in min_intersections:
        try:
            mapper = results[val]
            output = {"mapper": results[val]}
            num_policy_groups = mapper.num_policy_groups
            if num_policy_groups > len(mapper.tupper.clean):
                print("More components than elements!!")
                sys.exit(1)
            output_dir = os.path.join(
                root, f"data/models/{num_policy_groups}_policy_groups/"
            )
            output_file = generate_model_filename(
                args,
                nbors,
                d,
                min_intersection=val,
            )
            # Check if output directory already exists
            if os.path.isdir(output_dir):
                output_file = os.path.join(output_dir, output_file)
            else:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, output_file)

            output["hyperparameters"] = (
                n,
                p,
                nbors,
                d,
                hdbscan_params,
                val,
            )

            out_dir_message = output_file
            out_dir_message = "/".join(out_dir_message.split("/")[-2:])

            if len(mapper.complex) > 0:
                with open(output_file, "wb") as handle:
                    pickle.dump(
                        output,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                if args.Verbose:
                    print("\n")
                    print(
                        "-------------------------------------------------------------------------------------- \n\n"
                    )
                    print("Successfully generated `JMapper`.")
                    print("Written to:")
                    print(out_dir_message)

                    print(
                        "\n\n -------------------------------------------------------------------------------------- "
                    )
        except val not in results.keys():
            if args.Verbose:
                print(
                    "These parameters resulted in an \
                    empty Mapper representation."
                )
