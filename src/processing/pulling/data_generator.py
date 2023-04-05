"""Pull Latest Data from MongoDB"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv

from mongo import Mongo
from data_helper import get_data


if __name__ == "__main__":

    load_dotenv()
    client = os.getenv("mongo_client")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default="cleaned",
        help="Select database to pull from Mongo.",
    )
    parser.add_argument(
        "-c",
        "--col",
        type=str,
        default="coal_mapper",
        help="Select collection to pull from within `--database`.",
    )
    parser.add_argument(
        "-v",
        "--Verbose",
        default=False,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()

    cwd = os.path.dirname(__file__)
    output_dir = os.path.join(cwd, "../../../data/raw/")
    # Create Data dir and Processed subdir on first run
    if not os.path.isdir(output_dir):
        print("Creating local data directory...")
        os.makedirs(output_dir, exist_ok=True)

    mongo_object = Mongo(client_id=client)
    file = get_data(
        mongo_object=mongo_object,
        out_dir=output_dir,
        data_base=args.database,
        col=args.col,
    )

    output_path = os.path.join(output_dir, file)
    assert os.path.isfile(f"{output_path}.pkl"), "Failed to write data locally"

    # Setting message to print(readable) location of output message
    out_dir_message = output_path
    out_dir_message = "/".join(out_dir_message.split("/")[-2:])
    if args.Verbose:
        print(
            "-------------------------------------------------------------------------------- \n\n"
        )
        print(
            f"Successfully pulled from the `{args.col}` collection in the `{args.database}` Mongo database!"
        )
        print(f"Written to {out_dir_message}.pkl")

        print(
            "\n\n -------------------------------------------------------------------------------- "
        )
