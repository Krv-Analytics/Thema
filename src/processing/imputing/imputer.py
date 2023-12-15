"Esure data is full before projecting. Options to impute or drop missing values."

import argparse
import os
import pickle
import sys

import imputer_helper
from __init__ import env
from imputer_helper import (
    add_imputed_flags,
    clear_current_imputations,
    impute_data,
    imputed_filename,
    sampling_methods,
)
from omegaconf import OmegaConf
from termcolor import colored

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
        "-c",
        "--scaled_data",
        type=str,
        default=os.path.join(root, params["clean_data"]),
        help="Location of Cleaned data set",
    )
    parser.add_argument(
        "-n",
        "--num_imputations",
        type=int,
        default=params.data_imputation.num_samplings,
        help="Number of (distinct) imputations to generate",
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

    assert os.path.isfile(args.scaled_data), "\n Invalid path to Clean Data"
    # Load Clean Dataframe
    with open(args.scaled_data, "rb") as clean:
        reference = pickle.load(clean)
        df = reference["clean_data"]

    # Flag and track missing values
    df = add_imputed_flags(df)

    try:
        fill_method = params.data_imputation.method
        filler = getattr(imputer_helper, fill_method)
        # Configure outputs
        rel_outdir = "data/" + params["Run_Name"] + "/clean/"
        output_dir = os.path.join(root, rel_outdir)

        # Clear Previous imputes
        clear_current_imputations(output_dir, key=fill_method)

        # Non-sampling methods require only a single imputation
        if fill_method not in sampling_methods:
            if args.num_imputations != 1:
                print("\n")
                print(
                    "-------------------------------------------------------------------------------------- \n\n"
                )

                print(
                    colored("WARNING:", "yellow"),
                    f"`{fill_method}` does not require sampling and will only produce a single, deterministic imputation.",
                )
                print("\n")
                print(
                    "-------------------------------------------------------------------------------------- \n\n"
                )
            args.num_imputations = 1

        # Impute Scaled Data (multiple versions of)
        for i in range(1, args.num_imputations + 1):
            imputed_data = impute_data(df, fillna_method=filler)
            file_name = imputed_filename(
                params.Run_Name, fill_method=fill_method, number=i
            )
            output_filepath = output_dir + file_name

            # TODO: log random sample seed
            output = {"clean_data": imputed_data, "random_seed": "TODO"}
            with open(output_filepath, "wb") as f:
                pickle.dump(output, f)

        # Save the last imputation as a defualt reference
        params["clean_data"] = os.path.join(rel_outdir, file_name)

        try:
            with open(YAML_PATH, "w") as f:
                OmegaConf.save(params, f)
        except:
            print(
                colored(
                    "ERROR: Unable to write to params.yaml file. \n Make sure it exists and you have set appropriate file permissions.  ",
                    "red",
                )
            )

        if args.Verbose:
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )

            print(
                colored(f"SUCCESS: Completed Imputation!", "green"),
                f"{args.num_imputations} imputation(s) filled using {fill_method}.",
            )
            print(f"Written to `{rel_outdir}`.")

            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )

    except (AttributeError, TypeError) as e:
        print(colored("ERROR:", "red"), f"{e}")
        print(
            "Make sure your imputation method is supported (and spelled correctly) in `src/processing/imputing/imputer_helper.py`"
        )
        sys.exit(-1)
