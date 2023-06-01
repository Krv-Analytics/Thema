"""Pull Latest Data from MongoDB"""

import argparse
import os

from data_helper import get_raw_data
from dotenv import load_dotenv
from mongo import Mongo

if __name__ == "__main__":

    # Load Mongo Configuration from .env
    load_dotenv()
    client = os.getenv("mongo_client")
    root = os.getenv("root")
    database = os.getenv("mongo_database")
    collection = os.getenv("mongo_collection")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-L",
        "--local",
        default=False,
        action="store_true",
        help="If set, this script only builds an empty data directory.",
    )
    parser.add_argument(
        "-v",
        "--Verbose",
        default=False,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()

    output_dir = os.path.join(root, "data/raw/")
    # Create Data dir and Processed subdir on first run
    if not os.path.isdir(output_dir):
        print("Creating local data directory...")
        os.makedirs(output_dir, exist_ok=True)

    if args.local:
        exit(1)
    try:
        mongo_object = Mongo(client_id=client)

    # TODO: make this error out gracefully
    except ValueError:
        print(
            "Failed Connect to Mongo Data Base. Please make sure you have filled in the mongo_client field in .env "
        )
        exit(1)

    file = get_raw_data(
        mongo_object=mongo_object,
        out_dir=output_dir,
        data_base=database,
        col=collection,
    )

    output_path = os.path.join(output_dir, file)
    assert os.path.isfile(f"{output_path}.pkl"), "Failed to write data locally"

    # Setting message to print(readable) location of output message
    out_dir_message = output_path
    out_dir_message = "/".join(out_dir_message.split("/")[-2:])
    if args.Verbose:
        print(
            "\n\n-------------------------------------------------------------------------------- \n\n"
        )
        print(
            f"Successfully pulled from the `{collection}` collection in the `{database}` Mongo database!"
        )
        print(f"Written to {out_dir_message}.pkl")

        print(
            "\n\n-------------------------------------------------------------------------------- "
        )
