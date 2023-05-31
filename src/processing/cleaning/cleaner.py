"Preprocess your data before applying JMapper."
import argparse
import os
import pickle
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import json 
import ast

from cleaner_helper import data_cleaner, clean_data_filename



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    load_dotenv()
    root = os.getenv("root")

    JSON_PATH = os.getenv("JSON_PATH")
    try: 
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    except: 
        print("params.json file note found!")

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=params_json['raw_data'],
        help="Select raw data to clean.",
    )
    parser.add_argument(
        "-s",
        "--scaler",
        default=params_json['cleaning_scalar'],
        help="Select `sklearn` compatible method to scale data.",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        type=str,
        default=params_json['cleaning_encoding'],
        help="Method for encoding categorical fields.",
    )

    parser.add_argument(
        "-r",
        "--remove_columns",
        type=list,
        default=params_json['cleaning_remove_columns'],
        help="Specify list of columns to drop.",
    )
    parser.add_argument(
        "-v",
        "--Verbose",
        default=False,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()

    raw_data_path = os.path.join(root + args.data)
    # Read in Raw Data
    assert os.path.isfile(raw_data_path), "Invalid path to Raw Data"

    with open(raw_data_path, "rb") as raw_data:
        df = pickle.load(raw_data)
    assert (
        args.scaler == "standard_scaler"
    ), "Currently we only support `StandardScaler`"
    scaler = StandardScaler()

    # Clean
    clean_data = data_cleaner(
        data=df,
        scaler=scaler,
        column_filter=args.remove_columns,
        encoding=args.encoding,
    )

    output_dir = os.path.join(root, "data/clean/")

    column_filter = False
    if len(args.remove_columns) > 0:
        column_filter = True

    output_file = clean_data_filename(
        scaler=scaler, encoding=args.encoding, filter=column_filter
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, output_file)

    # TODO: WRITE THIS filename to the json file
    rel_outfile = "/".join(output_file.split("/")[-3:])

    output = {"clean_data": clean_data, "dropped_columns": args.remove_columns}
    # Write to pickle
    with open(output_file, "wb") as f:
        pickle.dump(output, f)

    
    # Populating parameter file with location of clean data
    
    params_json["clean_data"] = rel_outfile 

    try: 
        with open(JSON_PATH, "w") as f: 
            json.dump(params_json, f, indent=4)
    except:
        print("There was a problem writing to your parameter file!")
    
    
    if args.Verbose:
        print(
            "\n\n-------------------------------------------------------------------------------- \n\n"
        )
        print(f"Finished cleaning data! Written to {rel_outfile}")

        print(
            "\n\n-------------------------------------------------------------------------------- "
        )
