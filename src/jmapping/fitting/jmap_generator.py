"""Generate Graph jmaps using JMapper."""

import os
import sys
import json
import argparse
import pickle

from dotenv import load_dotenv

################################################################################################
#  Handling Local Imports
################################################################################################

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src + "/jmapping/")
sys.path.append(root + "logging/")

from run_log import Run_Log
from tupper import Tupper
from jmap_helper import generate_jmap_filename, jmap_generator

########################################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    root = os.getenv("root")

    # Initalize a runlog Manager
    runlog = Run_Log()

    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")
    parser.add_argument(
        "-r",
        "--raw",
        default=params_json["raw_data"],
        help="Select location of raw data relative from `root`",
    )
    parser.add_argument(
        "-c",
        "--clean",
        type=str,
        default=params_json["clean_data"],
        help="Select location of clean data relative from  `root`.",
    )
    parser.add_argument(
        "-D",
        "--projection",
        type=str,
        help="Select location of projection relative from `root`.",
    )

    parser.add_argument(
        "-m",
        "--min_cluster_size",
        default=4,
        type=int,
        help="Sets `min_cluster_size`, a parameter for HDBSCAN.",
    )
    parser.add_argument(
        "-M",
        "--max_cluster_size",
        default=0,
        type=int,
        help="Sets `max_cluster_size`, a parameter for HDBSCAN.",
    )

    parser.add_argument(
        "-s",
        "--random_seed",
        type=int,
        default=params_json["jmap_random_seed"],
        help="Set random seed to ensure Mapper/Graph reproducibility.",
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
        "-I",
        "--min_intersection",
        default=-1,
        type=int,
        help="Minimum intersection reuired between cluster elements to \
            form an edge in the graph representation.",
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
    tupper = Tupper(args.raw, args.clean, args.projection)
    nbors, d, dimension = tupper.get_projection_parameters()

    n, p = args.n_cubes, args.perc_overlap
    min_intersections = args.min_intersection
    hdbscan_params = args.min_cluster_size, args.max_cluster_size

    # GENERATE jmaps
    # Given our hyperparameters, we generate graphs, curvature,
    # and diagrams for all min_intersection values from a single JMapper fit.
    # This is done for efficiency purposes.

    jmapper = jmap_generator(
        tupper,
        n_cubes=n,
        perc_overlap=p,
        hdbscan_params=hdbscan_params,
        min_intersection=args.min_intersection,
    )
    # Unpack each graph (based on min_intersection) into it's own output file.
    output = {"jmapper": jmapper}
    try:
        num_policy_groups = len(jmapper.jgraph.components)
        if num_policy_groups > len(jmapper.tupper.clean):
            # Runlog Event Tracking
            # TODO: Improve Runlog tracking
            runlog.log_overPopulatedMapper_EVENT()
            sys.exit(1)
    except:
        # Runlog Event Tracking
        # TODO: Improve Runlog tracking
        runlog.log_unkownError_EVENT()
        sys.exit(1)

    rel_outdir = (
        "data/" + params_json["Run_Name"] + f"/jmaps/{num_policy_groups}_policy_groups/"
    )
    output_dir = os.path.join(root, rel_outdir)
    output_file = generate_jmap_filename(
        args,
        nbors,
        d,
        min_intersection=args.min_intersection,
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
        args.min_intersection,
    )

    out_dir_message = output_file
    out_dir_message = "/".join(out_dir_message.split("/")[-2:])

    assert not jmapper == -1, "ERROR 1"

    assert not jmapper == -2, "ERROR 2 "

    # Check for error codes from jmap_generator
    if jmapper == -1:
        print("EMPTY!")
        runlog.log_emptyGraph_EVENT()
        # TODO: Write out the hyperparameter culprits

    elif jmapper == -2:
        runlog.log_emptyComplex_EVENT()
        # TODO: Write out the hyperparameter culprits

    else:
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
