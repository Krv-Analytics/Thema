"""Compute ollivier-Ricci curvature for Coal-Plant Mapper Graphs."""

import argparse
import os
import sys
import pickle
from dotenv import load_dotenv
from coal_mapper_helper import coal_mapper_generator, generate_mapper_filename


load_dotenv()
src = os.getenv("src")
sys.path.append(src)
sys.path.append(src + "modeling/nammu/")


from processing.cleaning.tupper import Tupper


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    root = os.getenv("root")

    parser.add_argument(
        "--raw",
        type=str,
        default=os.path.join(
            root,
            "data/raw/coal_plant_data_raw.pkl",
        ),
        help="Select location of raw data set, as pulled from Mongo.",
    )
    parser.add_argument(
        "--clean",
        type=str,
        default=os.path.join(
            root,
            "data/clean/clean_data_standard_scaled_integer-encdoding_filtered.pkl",
        ),
        help="Select location of clean data.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        help="Select location of projection.",
    )

    parser.add_argument(
        "--min_cluster_size",
        default=2,
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
        help="Minimum intersection reuired between cluster elements to form an edge in the graph representation.",
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

    # Initialize a `Tupper`
    tupper = Tupper(raw=args.raw, clean=args.clean, projection=args.projection)

    nbors, d = tupper.get_projection_parameters()

    n, p = args.n_cubes, args.perc_overlap
    min_intersections = args.min_intersection
    hdbscan_params = args.min_cluster_size, args.max_cluster_size
    results = coal_mapper_generator(
        tupper,
        n_cubes=n,
        perc_overlap=p,
        hdbscan_params=hdbscan_params,
        min_intersection_vals=min_intersections,
        verbose=args.Verbose,
    )

    # Generate File for each min intersection value
    for val in min_intersections:
        try:
            assert val in results.keys(), "Empty Mapper!"
            mapper = results[val]
            output = {"mapper": results[val]}
            num_policy_groups = mapper.num_policy_groups
            output_dir = os.path.join(
                root, f"data/mappers/{num_policy_groups}_policy_groups/"
            )
            output_file = generate_mapper_filename(args, nbors, d, min_intersection=val)
            # Check if output directory already exists
            if os.path.isdir(output_dir):
                output_file = os.path.join(output_dir, output_file)
            else:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, output_file)

            # TODO: configure hyperparameters as a dictionary
            output["hyperparameters"] = (
                n,
                p,
                nbors,
                d,
                hdbscan_params,
            )

            out_dir_message = output_file
            out_dir_message = "/".join(out_dir_message.split("/")[-2:])

            if len(mapper.complex) > 0:
                with open(output_file, "wb") as handle:
                    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if args.Verbose:
                    print("\n")
                    print(
                        "-------------------------------------------------------------------------------- \n\n"
                    )
                    print(
                        f"Successfully generated `CoalMapper object`. Written to {out_dir_message}"
                    )

                    print(
                        "\n\n -------------------------------------------------------------------------------- "
                    )
        except:
            if args.Verbose:
                print("These parameters resulted in an empty Mapper representation")
