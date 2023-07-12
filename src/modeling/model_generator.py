"""Generate Graph Models using JMapper."""

import argparse
import os
import pickle
import sys
import json


########################################################################################################################

from dotenv import load_dotenv

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src + "/modeling/")

# imports from modeling module
from model_helper import (
    generate_model_filename,
    model_generator,
    script_paths,
)
from tupper import Tupper

sys.path.append(root + "logging/")

#from run_log import Run_Log

########################################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    root = os.getenv("root")

    # Initalize a runlog Manager 
    #runlog = Run_Log()



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
        default=params_json["model_random_seed"],
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
        default=1,
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

    # GENERATE MODELS
    # Given our hyperparameters, we generate graphs, curvature,
    # and diagrams for all min_intersection values from a single JMapper fit.
    # This is done for efficiency purposes.

    jmapper = model_generator(
        tupper,
        n_cubes=n,
        perc_overlap=p,
        hdbscan_params=hdbscan_params,
        min_intersection=args.min_intersection
    )
    # Unpack each graph (based on min_intersection) into it's own output file.
    output = {"jmapper": jmapper}
    try:
        num_policy_groups = len(jmapper.jgraph.components)
        if num_policy_groups > len(jmapper.tupper.clean):
            # Runlog Event Tracking
            #runlog.log_overPopulatedMapper_EVENT() 
            sys.exit(1)
    except:
         # Runlog Event Tracking
        #runlog.log_unkownError_EVENT()
        sys.exit(1)

    rel_outdir = "data/" + params_json["Run_Name"] + f"/models/{num_policy_groups}_policy_groups/"
    output_dir = os.path.join(root, rel_outdir)
    output_file = generate_model_filename(
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

    # Check for error codes from model_generator 
    if jmapper == -1: 
        print('hurt')
        #runlog.log_emptyGraph_EVENT() 
        # TODO: Write out to a different log the hyperparameter culprits
    
    elif jmapper == -2: 
        print('hurt')
        #runlog.log_emptyComplex_EVENT() 
        # TODO: Write out to a different log the hyperparameter culprits

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
