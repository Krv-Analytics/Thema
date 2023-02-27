"""Pull Latest Data from MongoDB"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv

from accessMongo import mongo_pull


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
        "-t",
        "--type",
        type=str,
        default="pkl",
        help="Select file type for local data.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./local_data/",
        help="Output Dir for results.",
    )

    args = parser.parse_args()
    mongo_pull(
        client,
        database=args.database,
        type=args.type,
        col=args.col,
        filepath=args.output,
    )

    assert os.path.isfile(
        f"{args.output+args.col}.{args.type}"
    ), "Failed to write data locally"
    print(
        f"Data successfully pulled from the `{args.col}` collection in the `{args.database}` Mongo database."
    )
